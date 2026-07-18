"""Segment independently detected Form regions into stable OCR/VLM targets.

The segmenter uses ruled geometry first, then only large OCR vertical gaps as a
fallback.  It stores rows and leaf Cells as side metadata and never rewrites
MinerU layout blocks, Table HTML, or source OCR spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import cv2
import numpy as np

from projects.custom_hybrid.form_detection import (
    FORM_DETECTOR_VERSION,
    FormDetectionSettings,
    collect_existing_table_bboxes,
    detect_form_regions,
    render_document_pages,
)


FORM_SEGMENTER_VERSION = 4


@dataclass(frozen=True)
class FormSegmentationSettings:
    """Conservative geometry thresholds for ruled and sparse forms."""

    horizontal_kernel_ratio: float = 0.06
    vertical_kernel_ratio: float = 0.02
    min_separator_coverage: float = 0.68
    max_separator_gap_ratio: float = 0.18
    min_row_height_points: float = 6.0
    min_content_gap_points: float = 12.0
    min_content_margin_points: float = 6.0
    min_ocr_gap_band_height_points: float = 96.0
    min_column_segment_width_ratio: float = 0.10
    min_column_coverage: float = 0.35
    max_column_gap_ratio: float = 0.35
    min_header_height_points: float = 9.0
    min_grid_vertical_coverage: float = 0.72
    min_local_vertical_coverage: float = 0.55
    boundary_column_distance_ratio: float = 0.18
    cell_padding_points: float = 2.0
    recognition_overflow_enabled: bool = True
    recognition_grouping_horizontal_points: float = 3.0
    recognition_grouping_vertical_points: float = 1.5
    recognition_min_anchor_ratio: float = 0.18
    recognition_min_overflow_points: float = 6.0
    recognition_max_overflow_points: float = 24.0
    recognition_padding_points: float = 1.5


@dataclass(frozen=True)
class TextSpanGeometry:
    bbox: tuple[float, float, float, float]
    text: str
    sequence: int


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _rect_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    return _rect_area(
        (
            max(left[0], right[0]),
            max(left[1], right[1]),
            min(left[2], right[2]),
            min(left[3], right[3]),
        )
    )


def collect_page_text_spans(page: Mapping[str, Any]) -> list[TextSpanGeometry]:
    """Collect deduplicated OCR text spans without changing source structures."""

    spans: list[TextSpanGeometry] = []
    seen: set[tuple[tuple[float, float, float, float], str]] = set()
    sequence = 0

    def visit(block: Any) -> None:
        nonlocal sequence
        if not isinstance(block, Mapping):
            return
        for line in block.get("lines", []):
            if not isinstance(line, Mapping):
                continue
            for span in line.get("spans", []):
                if not isinstance(span, Mapping) or span.get("type") not in {
                    "text",
                    "hyperlink",
                }:
                    continue
                bbox = _valid_bbox(span.get("bbox"))
                if bbox is None:
                    continue
                text = str(span.get("content", span.get("text", "")) or "")
                key = bbox, text
                if key in seen:
                    continue
                seen.add(key)
                spans.append(TextSpanGeometry(bbox, text, sequence))
                sequence += 1
        for child in block.get("blocks", []):
            visit(child)

    block_key = "preproc_blocks" if page.get("preproc_blocks") else "para_blocks"
    for block in page.get(block_key, []):
        visit(block)
    return spans


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
    max_gap: float,
) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _line_groups(mask: np.ndarray, tolerance: float) -> list[dict[str, Any]]:
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    groups: list[dict[str, Any]] = []
    for bbox in sorted(
        (cv2.boundingRect(contour) for contour in contours),
        key=lambda item: item[1] + item[3] / 2,
    ):
        center = bbox[1] + bbox[3] / 2
        if not groups or center - groups[-1]["center"] > tolerance:
            groups.append({"center": center, "bboxes": [bbox]})
        else:
            groups[-1]["bboxes"].append(bbox)
            groups[-1]["center"] = sum(item[1] + item[3] / 2 for item in groups[-1]["bboxes"]) / len(groups[-1]["bboxes"])
    return groups


def _group_intervals(
    group: Mapping[str, Any],
    region_width: int,
    min_width_ratio: float,
    merge_gap: float,
) -> list[list[float]]:
    return _merge_intervals(
        (
            (max(0, bbox[0]), min(region_width, bbox[0] + bbox[2]))
            for bbox in group["bboxes"]
            if bbox[2] >= region_width * min_width_ratio
        ),
        merge_gap,
    )


def _cluster_positions(values: Iterable[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _merge_close_boundaries(
    boundaries: list[dict[str, Any]],
    minimum_gap: float,
    region_height: int,
) -> list[dict[str, Any]]:
    ordered = sorted(boundaries, key=lambda item: item["y"])
    merged: list[dict[str, Any]] = []
    for boundary in ordered:
        boundary = dict(boundary)
        boundary["y"] = min(float(region_height), max(0.0, boundary["y"]))
        if not merged or boundary["y"] - merged[-1]["y"] >= minimum_gap:
            merged.append(boundary)
            continue
        previous = merged[-1]
        if boundary.get("strength", 0.0) > previous.get("strength", 0.0):
            merged[-1] = boundary
    if not merged or merged[0]["y"] > 0:
        merged.insert(0, {"y": 0.0, "source": "outer", "strength": 1.0})
    else:
        merged[0] = {"y": 0.0, "source": "outer", "strength": 1.0}
    if merged[-1]["y"] < region_height:
        merged.append({"y": float(region_height), "source": "outer", "strength": 1.0})
    else:
        merged[-1] = {
            "y": float(region_height),
            "source": "outer",
            "strength": 1.0,
        }
    return merged


def _pixel_spans_for_region(
    spans: Sequence[TextSpanGeometry],
    region_bbox: Sequence[float],
    scale_x: float,
    scale_y: float,
    region_width: int,
    region_height: int,
) -> list[dict[str, Any]]:
    converted = []
    for span in spans:
        x0 = (span.bbox[0] - region_bbox[0]) * scale_x
        y0 = (span.bbox[1] - region_bbox[1]) * scale_y
        x1 = (span.bbox[2] - region_bbox[0]) * scale_x
        y1 = (span.bbox[3] - region_bbox[1]) * scale_y
        if x1 <= 0 or y1 <= 0 or x0 >= region_width or y0 >= region_height:
            continue
        converted.append(
            {
                "bbox": [x0, y0, x1, y1],
                "source": span,
            }
        )
    return converted


def _add_ocr_gap_boundaries(
    boundaries: list[dict[str, Any]],
    pixel_spans: Sequence[Mapping[str, Any]],
    scale_y: float,
    settings: FormSegmentationSettings,
) -> list[dict[str, Any]]:
    gap_threshold = settings.min_content_gap_points * scale_y
    margin_threshold = settings.min_content_margin_points * scale_y
    join_gap = max(4.0, 4.0 * scale_y)
    extras = []
    ordered = sorted(boundaries, key=lambda item: item["y"])
    for upper, lower in zip(ordered, ordered[1:]):
        row_top, row_bottom = upper["y"], lower["y"]
        if row_bottom - row_top < settings.min_ocr_gap_band_height_points * scale_y:
            continue
        intervals = _merge_intervals(
            (
                (
                    max(row_top, span["bbox"][1]),
                    min(row_bottom, span["bbox"][3]),
                )
                for span in pixel_spans
                if min(row_bottom, span["bbox"][3]) > max(row_top, span["bbox"][1])
            ),
            join_gap,
        )
        for first, second in zip(intervals, intervals[1:]):
            gap = second[0] - first[1]
            if gap >= gap_threshold and first[1] - row_top >= margin_threshold and row_bottom - second[0] >= margin_threshold:
                extras.append(
                    {
                        "y": (first[1] + second[0]) / 2,
                        "source": "ocr_gap",
                        "strength": 0.82,
                    }
                )
    return ordered + extras


def _grid_separators_for_row(
    vertical_bboxes: Sequence[tuple[int, int, int, int]],
    row_top: float,
    row_bottom: float,
    region_width: int,
    settings: FormSegmentationSettings,
) -> list[float]:
    row_height = row_bottom - row_top
    positions = []
    for x, y, width, height in vertical_bboxes:
        overlap = max(0.0, min(row_bottom, y + height) - max(row_top, y))
        if row_height <= 0 or overlap / row_height < settings.min_grid_vertical_coverage:
            continue
        if y > row_top + row_height * 0.2 or y + height < row_bottom - row_height * 0.2:
            continue
        center = x + width / 2
        if region_width * 0.04 < center < region_width * 0.96:
            positions.append(center)
    return _cluster_positions(positions, max(4.0, region_width * 0.004))


def _column_group_for_row(
    horizontal_groups: Sequence[Mapping[str, Any]],
    row_top: float,
    row_bottom: float,
    region_width: int,
    settings: FormSegmentationSettings,
) -> dict[str, Any] | None:
    choices = []
    row_height = row_bottom - row_top
    for group in horizontal_groups:
        if not row_top + row_height * 0.45 <= group["center"] <= row_bottom + 4:
            continue
        intervals = _group_intervals(
            group,
            region_width,
            settings.min_column_segment_width_ratio,
            max(5.0, region_width * 0.006),
        )
        if not 2 <= len(intervals) <= 6:
            continue
        coverage = sum(end - start for start, end in intervals) / region_width
        gaps = [intervals[index + 1][0] - intervals[index][1] for index in range(len(intervals) - 1)]
        if coverage < settings.min_column_coverage or max(gaps, default=0.0) > region_width * settings.max_column_gap_ratio:
            continue
        choices.append(
            (
                coverage + 0.03 * len(intervals),
                {"center": float(group["center"]), "intervals": intervals},
            )
        )
    return max(choices, default=(0.0, None), key=lambda item: item[0])[1]


def _merge_boundary_intervals_without_vertical_support(
    intervals: Sequence[Sequence[float]],
    vertical_bboxes: Sequence[tuple[int, int, int, int]],
    field_top: float,
    row_bottom: float,
    region_width: int,
    settings: FormSegmentationSettings,
) -> list[list[float]]:
    """Merge apparent columns when a boundary line was merely interrupted.

    A horizontal cell border can mix true vertical dividers with interruptions
    caused by handwriting or scan noise.  When at least one gap has a real
    vertical divider, unsupported gaps are treated as colspans.  If none of the
    gaps has vertical support, the intervals are preserved as ordinary
    underline fields instead of being collapsed.
    """

    if not intervals:
        return []
    field_height = row_bottom - field_top
    if field_height <= 0:
        return [list(interval) for interval in intervals]
    margin = max(4.0, region_width * 0.008)
    support_by_gap = []
    for left, right in zip(intervals, intervals[1:]):
        gap_start = left[1]
        gap_end = right[0]
        # A very narrow interruption is normally the stroke width of a real
        # vertical divider.  Wider interruptions still require local vertical
        # line evidence because stamps and handwriting can erase long pieces
        # of a horizontal border.
        supported = gap_end - gap_start <= max(4.0, region_width * 0.012)
        for x, y, width, height in vertical_bboxes:
            center = x + width / 2
            if not gap_start - margin <= center <= gap_end + margin:
                continue
            if height < max(8.0, width * 4.0):
                continue
            overlap = max(
                0.0,
                min(row_bottom, y + height) - max(field_top, y),
            )
            if overlap / field_height >= settings.min_local_vertical_coverage:
                supported = True
                break
        support_by_gap.append(supported)
    if not any(support_by_gap):
        return [list(interval) for interval in intervals]

    merged = [list(intervals[0])]
    for interval, supported in zip(intervals[1:], support_by_gap):
        if supported:
            merged.append(list(interval))
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def _tiled_column_intervals(
    intervals: Sequence[Sequence[float]],
    region_width: float,
) -> list[list[float]]:
    """Convert detected column evidence into a gap-free row partition.

    Horizontal rules and underlines are evidence for column count and divider
    positions, not reliable crop edges: stamps, handwriting, and scan noise can
    shorten them.  Leaf Cells therefore tile the complete parent row from the
    left Form boundary to the right Form boundary.
    """

    if not intervals:
        return []
    separators = [
        min(
            region_width,
            max(0.0, (left[1] + right[0]) / 2),
        )
        for left, right in zip(intervals, intervals[1:])
    ]
    boundaries = [0.0, *separators, float(region_width)]
    return [
        [left, right]
        for left, right in zip(boundaries, boundaries[1:])
        if right > left
    ]


def _column_content_start(
    pixel_spans: Sequence[Mapping[str, Any]],
    intervals: Sequence[Sequence[float]],
    row_top: float,
    row_bottom: float,
    scale_y: float,
) -> float | None:
    row_spans = [span for span in pixel_spans if min(row_bottom, span["bbox"][3]) > max(row_top, span["bbox"][1])]
    if not row_spans:
        return None
    heights = [span["bbox"][3] - span["bbox"][1] for span in row_spans]
    tolerance = max(8.0, median(heights) * 0.65)
    line_groups: list[dict[str, Any]] = []
    for span in sorted(
        row_spans,
        key=lambda item: (item["bbox"][1] + item["bbox"][3]) / 2,
    ):
        center = (span["bbox"][1] + span["bbox"][3]) / 2
        if not line_groups or center - line_groups[-1]["center"] > tolerance:
            line_groups.append(
                {
                    "center": center,
                    "top": span["bbox"][1],
                    "spans": [span],
                }
            )
        else:
            line_groups[-1]["spans"].append(span)
            line_groups[-1]["top"] = min(
                line_groups[-1]["top"],
                span["bbox"][1],
            )
            line_groups[-1]["center"] = sum((item["bbox"][1] + item["bbox"][3]) / 2 for item in line_groups[-1]["spans"]) / len(
                line_groups[-1]["spans"]
            )

    for group in line_groups:
        occupied = set()
        for span in group["spans"]:
            span_bbox = span["bbox"]
            span_width = max(1.0, span_bbox[2] - span_bbox[0])
            for column_index, (start, end) in enumerate(intervals):
                overlap = max(
                    0.0,
                    min(end, span_bbox[2]) - max(start, span_bbox[0]),
                )
                if overlap / span_width >= 0.6 and span_width <= (end - start) * 1.2:
                    occupied.add(column_index)
        if len(occupied) >= 2:
            return max(row_top, group["top"] - 3.0 * scale_y)
    return None


def _cell_content(
    cell_bbox: Sequence[float],
    spans: Sequence[TextSpanGeometry],
) -> tuple[list[float] | None, int, str]:
    selected = []
    for span in spans:
        span_area = _rect_area(span.bbox)
        intersection = _intersection_area(cell_bbox, span.bbox)
        center_x = (span.bbox[0] + span.bbox[2]) / 2
        center_y = (span.bbox[1] + span.bbox[3]) / 2
        if intersection > 0 and (
            intersection / span_area >= 0.5
            or (cell_bbox[0] <= center_x <= cell_bbox[2] and cell_bbox[1] <= center_y <= cell_bbox[3])
        ):
            selected.append(span)
    if not selected:
        return None, 0, ""
    selected.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.sequence))
    content_bbox = [
        max(cell_bbox[0], min(item.bbox[0] for item in selected)),
        max(cell_bbox[1], min(item.bbox[1] for item in selected)),
        min(cell_bbox[2], max(item.bbox[2] for item in selected)),
        min(cell_bbox[3], max(item.bbox[3] for item in selected)),
    ]
    text = " ".join(item.text.strip() for item in selected if item.text.strip())
    return [round(value, 3) for value in content_bbox], len(selected), text


def _ink_ratio(
    ink_without_rules: np.ndarray,
    pixel_bbox: Sequence[float],
) -> float:
    height, width = ink_without_rules.shape
    x0 = max(0, min(width, int(np.floor(pixel_bbox[0]))))
    y0 = max(0, min(height, int(np.floor(pixel_bbox[1]))))
    x1 = max(0, min(width, int(np.ceil(pixel_bbox[2]))))
    y1 = max(0, min(height, int(np.ceil(pixel_bbox[3]))))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = ink_without_rules[y0:y1, x0:x1]
    return float(np.count_nonzero(crop) / crop.size)


def _recognition_components(
    ink_without_rules: np.ndarray,
    scale_x: float,
    scale_y: float,
    settings: FormSegmentationSettings,
) -> list[tuple[float, float, float, float]]:
    """Group nearby ink strokes into candidate handwriting/text components."""

    horizontal = max(
        1,
        round(settings.recognition_grouping_horizontal_points * scale_x),
    )
    vertical = max(
        1,
        round(settings.recognition_grouping_vertical_points * scale_y),
    )
    grouped = cv2.morphologyEx(
        ink_without_rules,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal, vertical)),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (grouped > 0).astype(np.uint8),
        connectivity=8,
    )
    minimum_area = max(4.0, scale_x * scale_y * 0.75)
    components = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < minimum_area or width < 2 or height < 2:
            continue
        components.append(
            (float(x), float(y), float(x + width), float(y + height))
        )
    return components


def _recognition_pixel_bbox(
    cell_bbox: Sequence[float],
    components: Sequence[Sequence[float]],
    region_width: int,
    region_height: int,
    scale_x: float,
    scale_y: float,
    settings: FormSegmentationSettings,
) -> tuple[list[float], bool]:
    """Expand a structural Cell only around ink that crosses its boundary."""

    left, top, right, bottom = map(float, cell_bbox)
    expanded = [left, top, right, bottom]
    maximum_x = settings.recognition_max_overflow_points * scale_x
    maximum_y = settings.recognition_max_overflow_points * scale_y
    minimum_x = settings.recognition_min_overflow_points * scale_x
    minimum_y = settings.recognition_min_overflow_points * scale_y
    padding_x = settings.recognition_padding_points * scale_x
    padding_y = settings.recognition_padding_points * scale_y
    changed = False
    for component in components:
        comp_left, comp_top, comp_right, comp_bottom = component
        intersection_width = max(
            0.0,
            min(right, comp_right) - max(left, comp_left),
        )
        intersection_height = max(
            0.0,
            min(bottom, comp_bottom) - max(top, comp_top),
        )
        component_area = max(1.0, comp_right - comp_left) * max(
            1.0,
            comp_bottom - comp_top,
        )
        anchor_ratio = intersection_width * intersection_height / component_area
        center_x = (comp_left + comp_right) / 2
        center_y = (comp_top + comp_bottom) / 2
        center_inside = left <= center_x <= right and top <= center_y <= bottom
        if (
            not center_inside
            and anchor_ratio < settings.recognition_min_anchor_ratio
        ):
            continue
        crosses_left = left - comp_left >= minimum_x
        crosses_top = top - comp_top >= minimum_y
        crosses_right = comp_right - right >= minimum_x
        crosses_bottom = comp_bottom - bottom >= minimum_y
        if not any((crosses_left, crosses_top, crosses_right, crosses_bottom)):
            continue
        if crosses_left:
            expanded[0] = min(
                expanded[0],
                max(left - maximum_x, comp_left - padding_x),
            )
        if crosses_top:
            expanded[1] = min(
                expanded[1],
                max(top - maximum_y, comp_top - padding_y),
            )
        if crosses_right:
            expanded[2] = max(
                expanded[2],
                min(right + maximum_x, comp_right + padding_x),
            )
        if crosses_bottom:
            expanded[3] = max(
                expanded[3],
                min(bottom + maximum_y, comp_bottom + padding_y),
            )
        changed = True
    expanded = [
        max(0.0, min(float(region_width), expanded[0])),
        max(0.0, min(float(region_height), expanded[1])),
        max(0.0, min(float(region_width), expanded[2])),
        max(0.0, min(float(region_height), expanded[3])),
    ]
    return expanded, changed


def segment_form_region(
    image: np.ndarray,
    page_size: Sequence[float],
    form_region: Mapping[str, Any],
    text_spans: Sequence[TextSpanGeometry],
    form_region_index: int = 0,
    settings: FormSegmentationSettings | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return parent rows and leaf Cells for one detected Form region."""

    settings = settings or FormSegmentationSettings()
    region_bbox = _valid_bbox(form_region.get("bbox"))
    if region_bbox is None:
        return [], []
    grayscale = np.asarray(image)
    if grayscale.ndim == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_RGB2GRAY)
    if grayscale.dtype != np.uint8:
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
    image_height, image_width = grayscale.shape
    page_width, page_height = float(page_size[0]), float(page_size[1])
    page_to_pixel_x = image_width / page_width
    page_to_pixel_y = image_height / page_height
    x0 = max(0, min(image_width, round(region_bbox[0] * page_to_pixel_x)))
    y0 = max(0, min(image_height, round(region_bbox[1] * page_to_pixel_y)))
    x1 = max(0, min(image_width, round(region_bbox[2] * page_to_pixel_x)))
    y1 = max(0, min(image_height, round(region_bbox[3] * page_to_pixel_y)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return [], []
    crop = grayscale[y0:y1, x0:x1]
    region_height, region_width = crop.shape
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    _threshold, ink = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    horizontal_mask = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(30, round(region_width * settings.horizontal_kernel_ratio)), 1),
        ),
    )
    vertical_mask = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(16, round(region_height * settings.vertical_kernel_ratio))),
        ),
    )
    ink_without_rules = cv2.bitwise_and(
        ink,
        cv2.bitwise_not(cv2.bitwise_or(horizontal_mask, vertical_mask)),
    )
    recognition_components = (
        _recognition_components(
            ink_without_rules,
            page_to_pixel_x,
            page_to_pixel_y,
            settings,
        )
        if settings.recognition_overflow_enabled
        else []
    )
    horizontal_groups = _line_groups(
        horizontal_mask,
        max(4.0, region_height * 0.003),
    )
    boundaries = [
        {"y": 0.0, "source": "outer", "strength": 1.0},
        {"y": float(region_height), "source": "outer", "strength": 1.0},
    ]
    for group in horizontal_groups:
        intervals = _group_intervals(
            group,
            region_width,
            0.08,
            max(6.0, region_width * 0.015),
        )
        coverage = sum(end - start for start, end in intervals) / region_width
        gaps = [intervals[index + 1][0] - intervals[index][1] for index in range(len(intervals) - 1)]
        maximum_gap_ratio = max(gaps, default=0.0) / region_width
        if (
            coverage >= settings.min_separator_coverage
            and maximum_gap_ratio <= settings.max_separator_gap_ratio
            and 8 < group["center"] < region_height - 8
        ):
            boundaries.append(
                {
                    "y": group["center"],
                    "source": "rule",
                    "strength": min(1.0, coverage),
                }
            )
    minimum_row_height = settings.min_row_height_points * page_to_pixel_y
    boundaries = _merge_close_boundaries(
        boundaries,
        minimum_row_height,
        region_height,
    )
    pixel_spans = _pixel_spans_for_region(
        text_spans,
        region_bbox,
        page_to_pixel_x,
        page_to_pixel_y,
        region_width,
        region_height,
    )
    boundaries = _merge_close_boundaries(
        _add_ocr_gap_boundaries(
            boundaries,
            pixel_spans,
            page_to_pixel_y,
            settings,
        ),
        minimum_row_height,
        region_height,
    )

    contours, _hierarchy = cv2.findContours(
        vertical_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    vertical_bboxes = [
        bbox for bbox in (cv2.boundingRect(contour) for contour in contours) if bbox[2] <= max(12.0, region_width * 0.03)
    ]
    rows = []
    cells = []
    pixel_to_page_x = page_width / image_width
    pixel_to_page_y = page_height / image_height
    for row_index, (upper, lower) in enumerate(zip(boundaries, boundaries[1:])):
        row_top, row_bottom = upper["y"], lower["y"]
        if row_bottom - row_top < minimum_row_height:
            continue
        row_page_bbox = [
            region_bbox[0],
            region_bbox[1] + row_top * pixel_to_page_y,
            region_bbox[2],
            region_bbox[1] + row_bottom * pixel_to_page_y,
        ]
        row_confidence = min(
            1.0,
            0.70 + 0.15 * float(upper.get("strength", 0.8)) + 0.15 * float(lower.get("strength", 0.8)),
        )
        rows.append(
            {
                "bbox": [round(value, 3) for value in row_page_bbox],
                "form_region_index": form_region_index,
                "row_index": row_index,
                "confidence": round(row_confidence, 4),
                "top_boundary": upper["source"],
                "bottom_boundary": lower["source"],
            }
        )

        cell_specs: list[tuple[float, float, float, float, str, float]] = []
        grid_separators = _grid_separators_for_row(
            vertical_bboxes,
            row_top,
            row_bottom,
            region_width,
            settings,
        )
        if grid_separators:
            x_boundaries = [0.0, *grid_separators, float(region_width)]
            for left, right in zip(x_boundaries, x_boundaries[1:]):
                if right - left >= region_width * 0.04:
                    cell_specs.append((left, row_top, right, row_bottom, "grid_cell", 0.98))
        else:
            column_group = _column_group_for_row(
                horizontal_groups,
                row_top,
                row_bottom,
                region_width,
                settings,
            )
            intervals = column_group["intervals"] if column_group is not None else None
            column_start = (
                _column_content_start(
                    pixel_spans,
                    intervals,
                    row_top,
                    row_bottom,
                    page_to_pixel_y,
                )
                if intervals is not None
                else None
            )
            if intervals is not None and column_start is not None:
                boundary_distance = row_bottom - float(column_group["center"])
                boundary_distance_limit = max(
                    6.0,
                    (row_bottom - row_top) * settings.boundary_column_distance_ratio,
                )
                if boundary_distance <= boundary_distance_limit:
                    intervals = _merge_boundary_intervals_without_vertical_support(
                        intervals,
                        vertical_bboxes,
                        column_start,
                        row_bottom,
                        region_width,
                        settings,
                    )
                if len(intervals) < 2:
                    intervals = None
            if intervals is not None and column_start is not None:
                minimum_header_height = settings.min_header_height_points * page_to_pixel_y
                header_has_text = any(
                    min(column_start, span["bbox"][3])
                    > max(row_top, span["bbox"][1])
                    for span in pixel_spans
                )
                if (
                    column_start - row_top >= minimum_header_height
                    and header_has_text
                ):
                    cell_specs.append(
                        (
                            0.0,
                            row_top,
                            float(region_width),
                            column_start,
                            "row_header",
                            0.90,
                        )
                    )
                else:
                    column_start = row_top
                for left, right in _tiled_column_intervals(
                    intervals,
                    float(region_width),
                ):
                    cell_specs.append(
                        (
                            left,
                            column_start,
                            right,
                            row_bottom,
                            "field_cell",
                            0.94,
                        )
                    )
            else:
                cell_specs.append(
                    (
                        0.0,
                        row_top,
                        float(region_width),
                        row_bottom,
                        "semantic_row",
                        row_confidence,
                    )
                )

        for column_index, spec in enumerate(cell_specs):
            left, top, right, bottom, kind, confidence = spec
            page_bbox = [
                region_bbox[0] + left * pixel_to_page_x,
                region_bbox[1] + top * pixel_to_page_y,
                region_bbox[0] + right * pixel_to_page_x,
                region_bbox[1] + bottom * pixel_to_page_y,
            ]
            recognition_pixel_bbox, recognition_overflow = (
                _recognition_pixel_bbox(
                    (left, top, right, bottom),
                    recognition_components,
                    region_width,
                    region_height,
                    page_to_pixel_x,
                    page_to_pixel_y,
                    settings,
                )
                if settings.recognition_overflow_enabled
                and kind in {"field_cell", "grid_cell"}
                else ([left, top, right, bottom], False)
            )
            recognition_bbox = [
                region_bbox[0] + recognition_pixel_bbox[0] * pixel_to_page_x,
                region_bbox[1] + recognition_pixel_bbox[1] * pixel_to_page_y,
                region_bbox[0] + recognition_pixel_bbox[2] * pixel_to_page_x,
                region_bbox[1] + recognition_pixel_bbox[3] * pixel_to_page_y,
            ]
            content_bbox, span_count, ocr_text = _cell_content(
                page_bbox,
                text_spans,
            )
            cells.append(
                {
                    "bbox": [round(value, 3) for value in page_bbox],
                    "form_region_index": form_region_index,
                    "row_index": row_index,
                    "column_index": column_index,
                    "kind": kind,
                    "confidence": round(float(confidence), 4),
                    "recognition_bbox": [
                        round(value, 3) for value in recognition_bbox
                    ],
                    "recognition_overflow": recognition_overflow,
                    "content_bbox": content_bbox,
                    "ocr_span_count": span_count,
                    "ocr_text": ocr_text,
                    "ink_ratio": round(
                        _ink_ratio(ink_without_rules, (left, top, right, bottom)),
                        5,
                    ),
                }
            )
    return rows, cells


def segment_form_page(
    image: np.ndarray,
    page: Mapping[str, Any],
    settings: FormSegmentationSettings | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = settings or FormSegmentationSettings()
    page_size = page.get("page_size")
    if not isinstance(page_size, (list, tuple)) or len(page_size) != 2:
        page_size = [image.shape[1], image.shape[0]]
    text_spans = collect_page_text_spans(page)
    rows = []
    cells = []
    for region_index, region in enumerate(page.get("form_regions", [])):
        if not isinstance(region, Mapping):
            continue
        region_rows, region_cells = segment_form_region(
            image,
            page_size,
            region,
            text_spans,
            region_index,
            settings,
        )
        rows.extend(region_rows)
        cells.extend(region_cells)
    return rows, cells


def annotate_form_structure(
    middle_json: MutableMapping[str, Any],
    source_document: str | Path,
    detection_settings: FormDetectionSettings | None = None,
    segmentation_settings: FormSegmentationSettings | None = None,
) -> dict[str, Any]:
    """Detect Form regions and segment them in a single page-render pass."""

    detection_settings = detection_settings or FormDetectionSettings()
    segmentation_settings = segmentation_settings or FormSegmentationSettings()
    pages = middle_json.get("pdf_info")
    if not isinstance(pages, list):
        raise ValueError("middle JSON must contain a pdf_info array")
    source_path = Path(source_document).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Form detection source does not exist: {source_path}")

    detection_changed_pages = []
    segmentation_changed_pages = []
    detection_pages = []
    detection_page_results = []
    segmentation_page_results = []
    rendered_pages = 0
    for page_index, image in enumerate(render_document_pages(source_path, detection_settings.render_scale)):
        if page_index >= len(pages):
            raise ValueError("Rendered document has more pages than middle JSON")
        page = pages[page_index]
        if not isinstance(page, MutableMapping):
            raise ValueError(f"pdf_info[{page_index}] must be an object")
        page_size = page.get("page_size")
        if not isinstance(page_size, (list, tuple)) or len(page_size) != 2:
            page_size = [
                image.shape[1] / detection_settings.render_scale,
                image.shape[0] / detection_settings.render_scale,
            ]
        regions = detect_form_regions(
            image,
            page_size,
            collect_existing_table_bboxes(page),
            detection_settings,
        )
        if page.get("form_regions") != regions:
            page["form_regions"] = regions
            detection_changed_pages.append(page_index)
        rows, cells = segment_form_page(image, page, segmentation_settings)
        if page.get("form_rows") != rows or page.get("form_cells") != cells:
            page["form_rows"] = rows
            page["form_cells"] = cells
            segmentation_changed_pages.append(page_index)
        if regions:
            detection_pages.append(page_index)
        detection_page_results.append({"page": page_index, "regions": len(regions), "form_regions": regions})
        segmentation_page_results.append({"page": page_index, "rows": len(rows), "cells": len(cells)})
        rendered_pages += 1
    if rendered_pages != len(pages):
        raise ValueError(
            f"Rendered document page count does not match middle JSON: rendered={rendered_pages}, middle={len(pages)}"
        )

    detection_marker = {
        "version": FORM_DETECTOR_VERSION,
        "render_scale": detection_settings.render_scale,
    }
    segmentation_marker = {"version": FORM_SEGMENTER_VERSION}
    detection_marker_changed = middle_json.get("_form_detection") != detection_marker
    segmentation_marker_changed = middle_json.get("_form_segmentation") != segmentation_marker
    middle_json["_form_detection"] = detection_marker
    middle_json["_form_segmentation"] = segmentation_marker
    detection_report = {
        "version": FORM_DETECTOR_VERSION,
        "pages": len(pages),
        "regions": sum(item["regions"] for item in detection_page_results),
        "detected_pages": detection_pages,
        "changed_pages": detection_changed_pages,
        "changed": bool(detection_changed_pages or detection_marker_changed),
        "page_results": detection_page_results,
    }
    segmentation_report = {
        "version": FORM_SEGMENTER_VERSION,
        "pages": len(pages),
        "rows": sum(item["rows"] for item in segmentation_page_results),
        "cells": sum(item["cells"] for item in segmentation_page_results),
        "changed_pages": segmentation_changed_pages,
        "changed": bool(segmentation_changed_pages or segmentation_marker_changed),
        "page_results": segmentation_page_results,
    }
    return {
        "form_detection": detection_report,
        "form_segmentation": segmentation_report,
        "changed": detection_report["changed"] or segmentation_report["changed"],
    }

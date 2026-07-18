"""Detect ruled Form/Table regions without trusting MinerU block types.

This module deliberately stops at routing.  It records outer regions that look
like ruled forms, but it does not rewrite layout blocks, Table HTML, OCR spans,
or Cell geometry.  Later stages can therefore opt into Cell segmentation and
VLM recognition without corrupting MinerU's original structural output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import cv2
import numpy as np


FORM_DETECTOR_VERSION = 1


@dataclass(frozen=True)
class FormDetectionSettings:
    """Conservative defaults for large ruled Form/Table regions."""

    render_scale: float = 2.0
    horizontal_kernel_ratio: float = 0.06
    vertical_kernel_ratio: float = 0.08
    min_region_width_ratio: float = 0.35
    min_region_height_ratio: float = 0.12
    min_region_area_ratio: float = 0.08
    min_rule_length_ratio: float = 0.18
    min_line_span_ratio: float = 0.15
    min_horizontal_rules: int = 3
    min_vertical_borders: int = 2
    min_enclosure_ratio: float = 0.35
    min_confidence: float = 0.55
    existing_table_coverage_threshold: float = 0.75

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FormDetectionSettings":
        """Load the two operational settings exposed in workflow JSON."""

        config = value if isinstance(value, Mapping) else {}
        return cls(
            render_scale=float(config.get("render_scale", cls.render_scale)),
            existing_table_coverage_threshold=float(
                config.get(
                    "existing_table_coverage_threshold",
                    cls.existing_table_coverage_threshold,
                )
            ),
        )


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _rect_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersect_bbox(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float] | None:
    intersection = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return intersection if _rect_area(intersection) > 0 else None


def _rectangle_union_area(rectangles: Iterable[Sequence[float]]) -> float:
    """Return exact union area for a small set of axis-aligned rectangles."""

    rects = [tuple(rectangle) for rectangle in rectangles if _rect_area(rectangle) > 0]
    if not rects:
        return 0.0
    x_edges = sorted({edge for rectangle in rects for edge in (rectangle[0], rectangle[2])})
    area = 0.0
    for x0, x1 in zip(x_edges, x_edges[1:]):
        if x1 <= x0:
            continue
        intervals = sorted((rectangle[1], rectangle[3]) for rectangle in rects if rectangle[0] < x1 and rectangle[2] > x0)
        merged: list[list[float]] = []
        for y0, y1 in intervals:
            if not merged or y0 > merged[-1][1]:
                merged.append([y0, y1])
            else:
                merged[-1][1] = max(merged[-1][1], y1)
        area += (x1 - x0) * sum(y1 - y0 for y0, y1 in merged)
    return area


def table_coverage_ratio(
    candidate_bbox: Sequence[float],
    table_bboxes: Iterable[Sequence[float]],
) -> float:
    """Measure how much of a candidate is already routed as a MinerU Table."""

    candidate_area = _rect_area(candidate_bbox)
    if candidate_area <= 0:
        return 0.0
    intersections = []
    for table_bbox in table_bboxes:
        intersection = _intersect_bbox(candidate_bbox, table_bbox)
        if intersection is not None:
            intersections.append(intersection)
    return min(1.0, _rectangle_union_area(intersections) / candidate_area)


def collect_existing_table_bboxes(page: Mapping[str, Any]) -> list[list[float]]:
    """Collect valid Table layout regions without inspecting or changing HTML."""

    collected: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()

    def visit(block: Any) -> None:
        if not isinstance(block, Mapping):
            return
        if block.get("type") == "table":
            bbox = _valid_bbox(block.get("bbox"))
            if bbox is not None and bbox not in seen:
                seen.add(bbox)
                collected.append(list(bbox))
        for child in block.get("blocks", []):
            visit(child)

    for block_key in ("preproc_blocks", "para_blocks"):
        for block in page.get(block_key, []):
            visit(block)
    return collected


def _cluster_positions(values: Iterable[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _contour_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [cv2.boundingRect(contour) for contour in contours]


def _edge_coverage(
    line_bboxes: Iterable[tuple[int, int, int, int]],
    target: float,
    tolerance: float,
    span_index: int,
    candidate_span: float,
    position_index: int,
) -> float:
    matches = (
        bbox[span_index] / candidate_span
        for bbox in line_bboxes
        if abs(bbox[position_index] + bbox[position_index + 2] / 2 - target) <= tolerance
    )
    return min(1.0, max(matches, default=0.0))


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 4:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Form detection expects a grayscale, RGB, or RGBA image")
    if grayscale.dtype != np.uint8:
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
    return grayscale


def detect_form_regions(
    image: np.ndarray,
    page_size: Sequence[float],
    existing_table_bboxes: Iterable[Sequence[float]] = (),
    settings: FormDetectionSettings | None = None,
) -> list[dict[str, Any]]:
    """Detect large ruled regions in one rendered page.

    Coordinates in the returned metadata use MinerU's top-left page coordinate
    system, matching ``page_size`` and layout block bboxes.
    """

    settings = settings or FormDetectionSettings()
    grayscale = _to_grayscale(np.asarray(image))
    image_height, image_width = grayscale.shape
    if image_width < 2 or image_height < 2:
        return []
    if len(page_size) != 2 or float(page_size[0]) <= 0 or float(page_size[1]) <= 0:
        raise ValueError("page_size must contain positive width and height")

    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    _threshold, ink = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    horizontal_kernel_width = max(
        30,
        round(image_width * settings.horizontal_kernel_ratio),
    )
    vertical_kernel_height = max(
        30,
        round(image_height * settings.vertical_kernel_ratio),
    )
    horizontal_mask = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_kernel_width, 1),
        ),
    )
    vertical_mask = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, vertical_kernel_height),
        ),
    )
    line_network = cv2.bitwise_or(horizontal_mask, vertical_mask)
    line_network = cv2.morphologyEx(
        line_network,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    component_count, _labels, component_stats, _centroids = cv2.connectedComponentsWithStats(
        (line_network > 0).astype(np.uint8),
        8,
    )
    horizontal_lines = _contour_bboxes(horizontal_mask)
    vertical_lines = _contour_bboxes(vertical_mask)
    scale_x = float(page_size[0]) / image_width
    scale_y = float(page_size[1]) / image_height
    page_area = image_width * image_height
    existing_tables = [bbox for value in existing_table_bboxes if (bbox := _valid_bbox(value)) is not None]
    regions: list[dict[str, Any]] = []

    for component_index in range(1, component_count):
        x, y, width, height, _pixel_area = (int(value) for value in component_stats[component_index])
        region_area_ratio = width * height / page_area
        if (
            width / image_width < settings.min_region_width_ratio
            or height / image_height < settings.min_region_height_ratio
            or region_area_ratio < settings.min_region_area_ratio
        ):
            continue

        horizontal_candidates = [
            bbox
            for bbox in horizontal_lines
            if bbox[2] >= width * settings.min_rule_length_ratio
            and y - 3 <= bbox[1] + bbox[3] / 2 <= y + height + 3
            and min(x + width, bbox[0] + bbox[2]) - max(x, bbox[0]) >= width * settings.min_line_span_ratio
        ]
        vertical_candidates = [
            bbox
            for bbox in vertical_lines
            if bbox[3] >= height * settings.min_rule_length_ratio
            and x - 3 <= bbox[0] + bbox[2] / 2 <= x + width + 3
            and min(y + height, bbox[1] + bbox[3]) - max(y, bbox[1]) >= height * settings.min_line_span_ratio
        ]
        horizontal_rules = _cluster_positions(
            (bbox[1] + bbox[3] / 2 for bbox in horizontal_candidates),
            max(3.0, image_height * 0.003),
        )
        vertical_borders = _cluster_positions(
            (bbox[0] + bbox[2] / 2 for bbox in vertical_candidates),
            max(3.0, image_width * 0.003),
        )
        if len(horizontal_rules) < settings.min_horizontal_rules or len(vertical_borders) < settings.min_vertical_borders:
            continue

        edge_x_tolerance = max(6.0, width * 0.03)
        edge_y_tolerance = max(6.0, height * 0.03)
        left_coverage = _edge_coverage(
            vertical_candidates,
            x,
            edge_x_tolerance,
            3,
            height,
            0,
        )
        right_coverage = _edge_coverage(
            vertical_candidates,
            x + width,
            edge_x_tolerance,
            3,
            height,
            0,
        )
        top_coverage = _edge_coverage(
            horizontal_candidates,
            y,
            edge_y_tolerance,
            2,
            width,
            1,
        )
        bottom_coverage = _edge_coverage(
            horizontal_candidates,
            y + height,
            edge_y_tolerance,
            2,
            width,
            1,
        )
        enclosure_ratio = (left_coverage + right_coverage + top_coverage + bottom_coverage) / 4
        if enclosure_ratio < settings.min_enclosure_ratio:
            continue

        confidence = (
            0.30 * min(1.0, len(horizontal_rules) / 8)
            + 0.20 * min(1.0, len(vertical_borders) / 2)
            + 0.35 * enclosure_ratio
            + 0.15 * min(1.0, region_area_ratio / 0.4)
        )
        if confidence < settings.min_confidence:
            continue

        page_bbox = [
            x * scale_x,
            y * scale_y,
            min(float(page_size[0]), (x + width) * scale_x),
            min(float(page_size[1]), (y + height) * scale_y),
        ]
        existing_table_coverage = table_coverage_ratio(page_bbox, existing_tables)
        if existing_table_coverage >= settings.existing_table_coverage_threshold:
            continue
        regions.append(
            {
                "bbox": [round(value, 3) for value in page_bbox],
                "confidence": round(confidence, 4),
                "evidence": {
                    "horizontal_rules": len(horizontal_rules),
                    "vertical_borders": len(vertical_borders),
                    "enclosure_ratio": round(enclosure_ratio, 4),
                    "existing_table_coverage": round(existing_table_coverage, 4),
                },
            }
        )

    return sorted(regions, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def render_document_pages(
    source_document: Path,
    render_scale: float,
) -> Iterable[np.ndarray]:
    if source_document.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(source_document))
        try:
            for page_index in range(len(document)):
                page = document[page_index]
                bitmap = page.render(scale=render_scale)
                try:
                    yield np.asarray(bitmap.to_pil().convert("L"))
                finally:
                    bitmap.close()
                    page.close()
        finally:
            document.close()
        return

    from PIL import Image, ImageSequence

    with Image.open(source_document) as image:
        for frame in ImageSequence.Iterator(image):
            grayscale = frame.convert("L")
            if render_scale != 1.0:
                grayscale = grayscale.resize(
                    (
                        max(1, round(grayscale.width * render_scale)),
                        max(1, round(grayscale.height * render_scale)),
                    )
                )
            yield np.asarray(grayscale)


def annotate_form_regions(
    middle_json: MutableMapping[str, Any],
    source_document: str | Path,
    settings: FormDetectionSettings | None = None,
) -> dict[str, Any]:
    """Attach isolated ``form_regions`` metadata and return an audit report."""

    settings = settings or FormDetectionSettings()
    pages = middle_json.get("pdf_info")
    if not isinstance(pages, list):
        raise ValueError("middle JSON must contain a pdf_info array")
    source_path = Path(source_document).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Form detection source does not exist: {source_path}")

    changed_pages = []
    detected_pages = []
    page_results = []
    rendered_pages = 0
    for page_index, image in enumerate(render_document_pages(source_path, settings.render_scale)):
        if page_index >= len(pages):
            raise ValueError("Rendered document has more pages than middle JSON")
        page = pages[page_index]
        if not isinstance(page, MutableMapping):
            raise ValueError(f"pdf_info[{page_index}] must be an object")
        page_size = page.get("page_size")
        if not isinstance(page_size, (list, tuple)) or len(page_size) != 2:
            page_size = [
                image.shape[1] / settings.render_scale,
                image.shape[0] / settings.render_scale,
            ]
        regions = detect_form_regions(
            image,
            page_size,
            collect_existing_table_bboxes(page),
            settings,
        )
        if page.get("form_regions") != regions:
            page["form_regions"] = regions
            changed_pages.append(page_index)
        if regions:
            detected_pages.append(page_index)
        page_results.append(
            {
                "page": page_index,
                "regions": len(regions),
                "form_regions": regions,
            }
        )
        rendered_pages += 1
    if rendered_pages != len(pages):
        raise ValueError(
            f"Rendered document page count does not match middle JSON: rendered={rendered_pages}, middle={len(pages)}"
        )

    marker = {
        "version": FORM_DETECTOR_VERSION,
        "render_scale": settings.render_scale,
    }
    marker_changed = middle_json.get("_form_detection") != marker
    if marker_changed:
        middle_json["_form_detection"] = marker
    return {
        "version": FORM_DETECTOR_VERSION,
        "pages": len(pages),
        "regions": sum(item["regions"] for item in page_results),
        "detected_pages": detected_pages,
        "changed_pages": changed_pages,
        "changed": bool(changed_pages or marker_changed),
        "page_results": page_results,
    }

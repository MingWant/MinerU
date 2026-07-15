"""Shared quality assessment for table Cell geometry and OCR content boxes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MAX_CELLS = 200
DEFAULT_MIN_CONTENT_CONTAINMENT = 0.7
DEFAULT_MAX_SEVERE_OVERLAPS_PER_CELL = 0.5
SEVERE_OVERLAP_RATIO = 0.35


@dataclass(frozen=True)
class TableCellGeometryQuality:
    reliable: bool
    reasons: tuple[str, ...]
    cell_count: int
    content_box_count: int
    content_containment_ratio: float | None
    severe_overlap_count: int
    severe_overlaps_per_cell: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, Real) and math.isfinite(float(item)) for item in value):
        return None
    bbox = tuple(float(item) for item in value)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def deduplicate_bboxes(values: Iterable[Any]) -> list[list[float]]:
    result = []
    seen = set()
    for value in values:
        bbox = normalize_bbox(value)
        if bbox is None or bbox in seen:
            continue
        seen.add(bbox)
        result.append(list(bbox))
    return result


def cell_content_bboxes(cell: Mapping[str, Any]) -> list[list[float]]:
    raw_spans = cell.get("content_spans", [])
    span_boxes = (
        [span.get("bbox") for span in raw_spans if isinstance(span, Mapping)]
        if isinstance(raw_spans, list)
        else []
    )
    content_spans = deduplicate_bboxes(span_boxes)
    if content_spans:
        return content_spans
    return deduplicate_bboxes([cell.get("content_bbox")])


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _contains(
    outer: Sequence[float],
    inner: Sequence[float],
    tolerance: float = 2.0,
) -> bool:
    return (
        outer[0] - tolerance <= inner[0]
        and outer[1] - tolerance <= inner[1]
        and outer[2] + tolerance >= inner[2]
        and outer[3] + tolerance >= inner[3]
    )


def assess_table_cell_geometry(
    span: Mapping[str, Any],
    *,
    max_cells: int = DEFAULT_MAX_CELLS,
    min_content_containment: float = DEFAULT_MIN_CONTENT_CONTAINMENT,
    max_severe_overlaps_per_cell: float = DEFAULT_MAX_SEVERE_OVERLAPS_PER_CELL,
) -> TableCellGeometryQuality:
    raw_cells = span.get("table_cells", [])
    cells = [
        cell
        for cell in raw_cells
        if isinstance(cell, Mapping) and normalize_bbox(cell.get("bbox")) is not None
    ] if isinstance(raw_cells, list) else []
    cell_boxes = deduplicate_bboxes(cell.get("bbox") for cell in cells)
    content_total = 0
    content_inside = 0
    for cell in cells:
        cell_bbox = normalize_bbox(cell.get("bbox"))
        if cell_bbox is None:
            continue
        for content_bbox in cell_content_bboxes(cell):
            content_total += 1
            content_inside += int(_contains(cell_bbox, content_bbox))
    containment = content_inside / content_total if content_total else None

    severe_overlaps = 0
    for index, left in enumerate(cell_boxes):
        left_area = _bbox_area(left)
        for right in cell_boxes[index + 1 :]:
            smaller_area = min(left_area, _bbox_area(right))
            if smaller_area <= 0:
                continue
            if _intersection_area(left, right) / smaller_area > SEVERE_OVERLAP_RATIO:
                severe_overlaps += 1
    overlaps_per_cell = severe_overlaps / len(cell_boxes) if cell_boxes else 0.0

    reasons = []
    if not cell_boxes:
        reasons.append("missing_cell_bboxes")
    if len(cell_boxes) > max_cells:
        reasons.append("excessive_cell_count")
    if containment is not None and containment < min_content_containment:
        reasons.append("low_content_containment")
    if overlaps_per_cell > max_severe_overlaps_per_cell:
        reasons.append("excessive_cell_overlap")
    return TableCellGeometryQuality(
        reliable=not reasons,
        reasons=tuple(reasons),
        cell_count=len(cell_boxes),
        content_box_count=content_total,
        content_containment_ratio=containment,
        severe_overlap_count=severe_overlaps,
        severe_overlaps_per_cell=overlaps_per_cell,
    )

# Copyright (c) Opendatalab. All rights reserved.
import math
from typing import Any

import numpy as np
from bs4 import BeautifulSoup

from mineru.utils.bbox_utils import normalize_to_int_bbox


def get_table_crop_bbox(
    table_bbox: Any,
    image_size: tuple[int, int],
    scale: float = 1.0,
) -> list[int] | None:
    """Reproduce the table crop rounding and return its exact page-image bbox."""
    if table_bbox is None or not scale:
        return None
    try:
        scaled_bbox = [float(value) / float(scale) for value in table_bbox]
    except (TypeError, ValueError):
        return None
    coarse_bbox = normalize_to_int_bbox(scaled_bbox)
    if coarse_bbox is None:
        return None
    return normalize_to_int_bbox(
        [float(value) * float(scale) for value in coarse_bbox],
        image_size=image_size,
    )


def _bbox_points(bbox: Any) -> list[tuple[float, float]]:
    """Normalize a rectangular or quadrilateral bbox to corner points."""
    try:
        values = np.asarray(bbox, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return []

    if values.size == 4:
        x0, y0, x1, y1 = values.tolist()
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if values.size == 8:
        return list(zip(values[0::2].tolist(), values[1::2].tolist()))
    return []


def _inverse_rotate_point(
    x: float,
    y: float,
    rotation_label: str,
    original_width: float,
    original_height: float,
) -> tuple[float, float]:
    """Map a point from the orientation-corrected crop back to the source crop."""
    if rotation_label == "270":
        # The source crop was rotated 90 degrees clockwise before recognition.
        return y, original_height - x
    if rotation_label == "90":
        # The source crop was rotated 90 degrees counter-clockwise before recognition.
        return original_width - y, x
    return x, y


def table_cell_bbox_to_page(
    cell_bbox: Any,
    crop_bbox: list[int] | tuple[int, int, int, int],
    rotation_label: str | int = "0",
) -> list[int] | None:
    """Convert a table-model cell bbox from crop coordinates to page-image coordinates."""
    if not crop_bbox or len(crop_bbox) != 4:
        return None

    crop_x0, crop_y0, crop_x1, crop_y1 = [float(value) for value in crop_bbox]
    crop_width = crop_x1 - crop_x0
    crop_height = crop_y1 - crop_y0
    if crop_width <= 0 or crop_height <= 0:
        return None

    points = _bbox_points(cell_bbox)
    if not points:
        return None

    source_points = [
        _inverse_rotate_point(
            x,
            y,
            str(rotation_label or "0"),
            crop_width,
            crop_height,
        )
        for x, y in points
    ]
    xs = [min(max(x, 0.0), crop_width) for x, _ in source_points]
    ys = [min(max(y, 0.0), crop_height) for _, y in source_points]

    page_bbox = [
        math.floor(crop_x0 + min(xs)),
        math.floor(crop_y0 + min(ys)),
        math.ceil(crop_x0 + max(xs)),
        math.ceil(crop_y0 + max(ys)),
    ]
    if page_bbox[2] <= page_bbox[0] or page_bbox[3] <= page_bbox[1]:
        return None
    return page_bbox


def _normalize_logic_point(logic_point: Any) -> list[int] | None:
    try:
        values = np.asarray(logic_point, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 4:
        return None
    return [int(value) for value in values[:4]]


def build_table_cells(
    cell_bboxes: Any,
    logic_points: Any,
    html_code: str,
    crop_bbox: list[int] | tuple[int, int, int, int],
    rotation_label: str | int = "0",
) -> list[dict]:
    """Build JSON-serializable table cells with page-image bboxes and structure."""
    if cell_bboxes is None:
        return []
    try:
        boxes = list(cell_bboxes)
    except TypeError:
        return []

    try:
        logic_items = list(logic_points) if logic_points is not None else []
    except TypeError:
        logic_items = []

    soup = BeautifulSoup(html_code or "", "html.parser")
    html_cells = soup.find_all(["td", "th"])
    normalized_logic_points = [
        _normalize_logic_point(item) for item in logic_items
    ]
    html_cell_by_box_index = {}
    sorted_logic_indices = sorted(
        (
            index
            for index, logic_point in enumerate(normalized_logic_points)
            if logic_point is not None and index < len(boxes)
        ),
        key=lambda index: (
            normalized_logic_points[index][0],
            normalized_logic_points[index][2],
            normalized_logic_points[index][1],
            normalized_logic_points[index][3],
        ),
    )
    for html_index, box_index in enumerate(sorted_logic_indices):
        if html_index >= len(html_cells):
            break
        html_cell_by_box_index[box_index] = html_cells[html_index]
    table_cells = []

    for index, cell_bbox in enumerate(boxes):
        page_bbox = table_cell_bbox_to_page(
            cell_bbox,
            crop_bbox,
            rotation_label=rotation_label,
        )
        if page_bbox is None:
            continue

        cell = {
            "bbox": page_bbox,
            "text": "",
        }
        html_cell = html_cell_by_box_index.get(index)
        if html_cell is None and not sorted_logic_indices and index < len(html_cells):
            html_cell = html_cells[index]
        if html_cell is not None:
            cell["text"] = html_cell.get_text(" ", strip=True)
            cell["is_header"] = html_cell.name == "th"

        if index < len(normalized_logic_points):
            logic_point = normalized_logic_points[index]
            if logic_point is not None:
                cell.update(
                    {
                        "row_start": logic_point[0],
                        "row_end": logic_point[1],
                        "col_start": logic_point[2],
                        "col_end": logic_point[3],
                    }
                )

        table_cells.append(cell)

    return table_cells

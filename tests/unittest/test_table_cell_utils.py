import json

import numpy as np

from mineru.utils.table_cell_utils import (
    build_table_cells,
    get_table_crop_bbox,
    table_cell_bbox_to_page,
)


def test_table_cell_bbox_to_page_without_rotation():
    assert table_cell_bbox_to_page(
        [10.2, 5.1, 30.3, 20.4],
        [100, 200, 200, 250],
    ) == [110, 205, 131, 221]


def test_wired_crop_bbox_preserves_the_actual_rounded_crop_origin():
    assert get_table_crop_bbox(
        [101, 202, 299, 401],
        image_size=(1000, 1000),
        scale=10 / 3,
    ) == [100, 200, 300, 404]


def test_table_cell_bbox_to_page_after_clockwise_rotation():
    # Original local bbox [10, 5, 30, 20] becomes [30, 10, 45, 30]
    # after rotating a 100x50 crop clockwise.
    assert table_cell_bbox_to_page(
        [30, 10, 45, 30],
        [100, 200, 200, 250],
        rotation_label="270",
    ) == [110, 205, 130, 220]


def test_table_cell_bbox_to_page_after_counter_clockwise_rotation():
    # Original local bbox [10, 5, 30, 20] becomes [5, 70, 20, 90]
    # after rotating a 100x50 crop counter-clockwise.
    assert table_cell_bbox_to_page(
        [5, 70, 20, 90],
        [100, 200, 200, 250],
        rotation_label="90",
    ) == [110, 205, 130, 220]


def test_build_table_cells_uses_logic_order_and_serializable_values():
    cell_bboxes = np.array(
        [
            [50, 0, 100, 20],
            [0, 0, 50, 20],
        ],
        dtype=np.float32,
    )
    logic_points = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    cells = build_table_cells(
        cell_bboxes,
        logic_points,
        "<table><tr><th>Key</th><td>Value</td></tr></table>",
        [100, 200, 200, 220],
    )

    assert cells == [
        {
            "bbox": [150, 200, 200, 220],
            "text": "Value",
            "is_header": False,
            "row_start": 0,
            "row_end": 0,
            "col_start": 1,
            "col_end": 1,
        },
        {
            "bbox": [100, 200, 150, 220],
            "text": "Key",
            "is_header": True,
            "row_start": 0,
            "row_end": 0,
            "col_start": 0,
            "col_end": 0,
        },
    ]
    json.dumps(cells)

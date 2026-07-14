from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    make_blocks_to_content_list,
    make_blocks_to_content_list_v2,
)
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.enum_class import BlockType, ContentType


def _table_para_block():
    return {
        "type": BlockType.TABLE,
        "bbox": [100, 200, 300, 400],
        "blocks": [
            {
                "type": BlockType.TABLE_BODY,
                "lines": [
                    {
                        "spans": [
                            {
                                "type": ContentType.TABLE,
                                "html": "<table><tr><td>Value</td></tr></table>",
                                "image_path": "table.jpg",
                                "table_cells": [
                                    {
                                        "bbox": [120, 220, 220, 260],
                                        "text": "Value",
                                        "row_start": 0,
                                        "row_end": 0,
                                        "col_start": 0,
                                        "col_end": 0,
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }
        ],
    }


def test_content_lists_include_normalized_table_cell_bboxes():
    para_block = _table_para_block()

    legacy = make_blocks_to_content_list(
        para_block,
        "images",
        page_idx=2,
        page_size=[400, 800],
    )
    v2 = make_blocks_to_content_list_v2(
        para_block,
        "images",
        page_size=[400, 800],
    )

    expected_cell = {
        "bbox": [300, 275, 550, 325],
        "text": "Value",
        "row_start": 0,
        "row_end": 0,
        "col_start": 0,
        "col_end": 0,
    }
    assert legacy["table_cells"] == [expected_cell]
    assert v2["content"]["table_cells"] == [expected_cell]


def test_magic_model_scales_table_cells_with_the_table_block():
    magic_model = MagicModel.__new__(MagicModel)
    magic_model._MagicModel__scale = 2
    magic_model._MagicModel__page_model_info = {
        "layout_dets": [
            {
                "bbox": [100, 200, 300, 400],
                "table_cells": [
                    {"bbox": [120, 220, 220, 260], "text": "Value"}
                ],
            }
        ]
    }

    magic_model._MagicModel__fix_axis()

    layout_det = magic_model._MagicModel__page_model_info["layout_dets"][0]
    assert layout_det["bbox"] == [50, 100, 150, 200]
    assert layout_det["table_cells"][0]["bbox"] == [60, 110, 110, 130]

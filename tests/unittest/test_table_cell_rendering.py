from io import BytesIO

from reportlab.pdfgen import canvas

from mineru.utils import draw_bbox as draw_bbox_module
from mineru.utils.enum_class import BlockType, ContentType


def _blank_pdf_bytes(width=200, height=200):
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=(width, height))
    pdf_canvas.showPage()
    pdf_canvas.save()
    return buffer.getvalue()


def test_span_bbox_renderer_uses_page_level_table_cell_bboxes(monkeypatch, tmp_path):
    rendered_table_cells = []
    rendered_content_spans = []

    def record_bbox(i, bbox_list, page, pdf_canvas, rgb_config, fill_config):
        if rgb_config == [255, 128, 0]:
            rendered_table_cells.extend(bbox_list[i])
        elif rgb_config == [0, 180, 255]:
            rendered_content_spans.extend(bbox_list[i])
        return pdf_canvas

    monkeypatch.setattr(draw_bbox_module, "draw_bbox_without_number", record_bbox)

    pdf_info = [
        {
            "discarded_blocks": [],
            "preproc_blocks": [
                {
                    "type": BlockType.TABLE,
                    "blocks": [
                        {
                            "type": BlockType.TABLE_BODY,
                            "lines": [
                                {
                                    "spans": [
                                        {
                                            "type": ContentType.TABLE,
                                            "bbox": [40, 50, 160, 150],
                                            "table_cells": [
                                                {
                                                    "bbox": [50, 60, 100, 100],
                                                    "content_bbox": [55, 68, 88, 84],
                                                    "text": "Key",
                                                },
                                                {
                                                    "bbox": [100, 60, 150, 100],
                                                    "content_spans": [
                                                        {"bbox": [108, 68, 142, 84]}
                                                    ],
                                                    "text": "Value",
                                                },
                                            ],
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    draw_bbox_module.draw_span_bbox(
        pdf_info,
        _blank_pdf_bytes(),
        str(tmp_path),
        "table-cells.pdf",
    )

    assert rendered_table_cells == [
        [50, 60, 100, 100],
        [100, 60, 150, 100],
    ]
    assert rendered_content_spans == [
        [55, 68, 88, 84],
        [108, 68, 142, 84],
    ]


def test_span_bbox_renderer_finds_cells_in_finalized_para_blocks(monkeypatch, tmp_path):
    rendered_table_cells = []
    rendered_content_spans = []

    def record_bbox(i, bbox_list, page, pdf_canvas, rgb_config, fill_config):
        if rgb_config == [255, 128, 0]:
            rendered_table_cells.extend(bbox_list[i])
        elif rgb_config == [0, 180, 255]:
            rendered_content_spans.extend(bbox_list[i])
        return pdf_canvas

    monkeypatch.setattr(draw_bbox_module, "draw_bbox_without_number", record_bbox)
    table_span = {
        "type": ContentType.TABLE,
        "bbox": [40, 50, 160, 150],
        "table_cells": [
            {
                "bbox": [50, 60, 100, 100],
                "content_bbox": [55, 68, 88, 84],
            }
        ],
    }
    table_block = {
        "type": BlockType.TABLE,
        "blocks": [
            {
                "type": BlockType.TABLE_BODY,
                "lines": [{"spans": [table_span]}],
            }
        ],
    }
    pdf_info = [
        {
            "discarded_blocks": [],
            "preproc_blocks": [],
            "para_blocks": [table_block],
        }
    ]

    draw_bbox_module.draw_span_bbox(
        pdf_info,
        _blank_pdf_bytes(),
        str(tmp_path),
        "finalized-table-cells.pdf",
    )

    assert rendered_table_cells == [[50, 60, 100, 100]]
    assert rendered_content_spans == [[55, 68, 88, 84]]


def test_unreliable_overlapping_cell_geometry_is_hidden_but_content_remains():
    span = {
        "table_cells": [
            {
                "bbox": [10 + index, 10, 100 + index, 100],
                "content_bbox": [20 + index, 20, 50 + index, 40],
            }
            for index in range(4)
        ]
    }

    cell_boxes, content_boxes = draw_bbox_module._table_cell_render_bboxes(span)

    assert cell_boxes == []
    assert len(content_boxes) == 4


def test_reliable_non_overlapping_cell_geometry_is_retained():
    span = {
        "table_cells": [
            {"bbox": [10, 10, 100, 50], "content_bbox": [20, 20, 80, 40]},
            {"bbox": [100, 10, 190, 50], "content_bbox": [110, 20, 180, 40]},
        ]
    }

    cell_boxes, content_boxes = draw_bbox_module._table_cell_render_bboxes(span)

    assert cell_boxes == [[10, 10, 100, 50], [100, 10, 190, 50]]
    assert content_boxes == [[20, 20, 80, 40], [110, 20, 180, 40]]

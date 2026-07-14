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

    def record_bbox(i, bbox_list, page, pdf_canvas, rgb_config, fill_config):
        if rgb_config == [255, 128, 0]:
            rendered_table_cells.extend(bbox_list[i])
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
                                                    "text": "Key",
                                                },
                                                {
                                                    "bbox": [100, 60, 150, 100],
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

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


def test_span_bbox_renderer_draws_content_without_cell_geometry(monkeypatch, tmp_path):
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

    assert rendered_table_cells == []
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

    assert rendered_table_cells == []
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


def test_reliable_cell_geometry_is_not_returned_for_rendering():
    span = {
        "table_cells": [
            {"bbox": [10, 10, 100, 50], "content_bbox": [20, 20, 80, 40]},
            {"bbox": [100, 10, 190, 50], "content_bbox": [110, 20, 180, 40]},
        ]
    }

    cell_boxes, content_boxes = draw_bbox_module._table_cell_render_bboxes(span)

    assert cell_boxes == []
    assert content_boxes == [[20, 20, 80, 40], [110, 20, 180, 40]]


def test_content_bbox_renders_without_cell_geometry():
    cell_boxes, content_boxes = draw_bbox_module._table_cell_render_bboxes(
        {"table_cells": [{"content_bbox": [20, 20, 80, 40]}]}
    )

    assert cell_boxes == []
    assert content_boxes == [[20, 20, 80, 40]]


def test_span_bbox_renderer_draws_key_and_value_boxes(monkeypatch, tmp_path):
    rendered_keys = []
    rendered_values = []

    def record_bbox(i, bbox_list, page, pdf_canvas, rgb_config, fill_config):
        if rgb_config == [0, 160, 90]:
            rendered_keys.extend(bbox_list[i])
        elif rgb_config == [30, 90, 255]:
            rendered_values.extend(bbox_list[i])
        return pdf_canvas

    monkeypatch.setattr(draw_bbox_module, "draw_bbox_without_number", record_bbox)
    pdf_info = [
        {
            "discarded_blocks": [],
            "preproc_blocks": [],
            "form_fields": [
                {
                    "key": "Policy No.",
                    "value": "A123",
                    "key_bbox": [20, 20, 80, 35],
                    "value_bbox": [100, 20, 150, 35],
                }
            ],
        }
    ]

    draw_bbox_module.draw_span_bbox(
        pdf_info,
        _blank_pdf_bytes(),
        str(tmp_path),
        "form-fields.pdf",
    )

    assert rendered_keys == [[20, 20, 80, 35]]
    assert rendered_values == [[100, 20, 150, 35]]


def test_form_region_renderer_draws_only_detected_outer_regions(monkeypatch, tmp_path):
    rendered_regions = []

    def record_bbox(i, bbox_list, page, pdf_canvas, rgb_config, fill_config):
        if rgb_config == [128, 0, 255]:
            rendered_regions.extend(bbox_list[i])
        return pdf_canvas

    monkeypatch.setattr(draw_bbox_module, "draw_bbox_without_number", record_bbox)
    pdf_info = [
        {
            "form_regions": [
                {
                    "bbox": [20, 25, 180, 175],
                    "confidence": 0.99,
                    "evidence": {
                        "horizontal_rules": 8,
                        "vertical_borders": 2,
                        "enclosure_ratio": 1.0,
                    },
                }
            ],
            "preproc_blocks": [
                {"type": BlockType.TEXT, "bbox": [40, 40, 100, 60]}
            ],
        }
    ]

    draw_bbox_module.draw_form_region_bbox(
        pdf_info,
        _blank_pdf_bytes(),
        str(tmp_path),
        "form-regions.pdf",
    )

    assert rendered_regions == [[20, 25, 180, 175]]


def test_form_cell_renderer_draws_outer_region_and_cyan_cells(monkeypatch, tmp_path):
    rendered_regions = []
    rendered_cells = []
    rendered_recognition_bboxes = []

    def record_bbox(i, bbox_list, page, pdf_canvas, rgb_config, fill_config):
        if rgb_config == [128, 0, 255]:
            rendered_regions.extend(bbox_list[i])
        elif rgb_config == [0, 180, 255]:
            rendered_cells.extend(bbox_list[i])
        elif rgb_config == [30, 90, 255]:
            rendered_recognition_bboxes.extend(bbox_list[i])
        return pdf_canvas

    monkeypatch.setattr(draw_bbox_module, "draw_bbox_without_number", record_bbox)
    pdf_info = [
        {
            "form_regions": [{"bbox": [20, 20, 180, 180]}],
            "form_cells": [
                {"bbox": [20, 20, 180, 80], "kind": "semantic_row"},
                {
                    "bbox": [20, 80, 100, 180],
                    "recognition_bbox": [20, 75, 108, 180],
                    "recognition_overflow": True,
                    "kind": "field_cell",
                },
                {"bbox": [100, 80, 180, 180], "kind": "field_cell"},
            ],
        }
    ]

    draw_bbox_module.draw_form_cell_bbox(
        pdf_info,
        _blank_pdf_bytes(),
        str(tmp_path),
        "form-cells.pdf",
    )

    assert rendered_regions == [[20, 20, 180, 180]]
    assert rendered_cells == [
        [20, 20, 180, 80],
        [20, 80, 100, 180],
        [100, 80, 180, 180],
    ]
    assert rendered_recognition_bboxes == [[20, 75, 108, 180]]

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.semantic_markdown import (
    generate_semantic_markdown,
    write_semantic_markdown,
)


def content_span(text, bbox, **metadata):
    return {"type": "text", "text": text, "bbox": list(bbox), **metadata}


def table_block(cells, bbox=(10, 10, 590, 780)):
    table = {
        "type": "table",
        "bbox": list(bbox),
        "table_cells": cells,
    }
    return {
        "type": "table",
        "bbox": list(bbox),
        "lines": [{"bbox": list(bbox), "spans": [table]}],
    }


def text_block(text, bbox, *, block_type="text"):
    span = content_span(text, bbox)
    return {
        "type": block_type,
        "bbox": list(bbox),
        "lines": [{"bbox": list(bbox), "spans": [span]}],
    }


def middle(*blocks, pages=None):
    if pages is None:
        pages = [{"page_size": [600, 800], "preproc_blocks": list(blocks)}]
    return {"_backend": "hybrid", "pdf_info": pages}


class SemanticMarkdownTests(unittest.TestCase):
    def test_form_section_and_field_value_are_rendered_semantically(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 560, 45],
                "content_spans": [
                    content_span("PART 1 INSURED'S INFORMATION", [25, 25, 400, 40])
                ],
            },
            {
                "row_start": 1,
                "col_start": 0,
                "bbox": [20, 50, 560, 85],
                "content_spans": [
                    content_span("Policy Number", [25, 58, 150, 75]),
                    content_span("P-12345", [230, 58, 340, 75]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("### PART 1 INSURED'S INFORMATION", markdown)
        self.assertIn("- **Policy Number**: P-12345", markdown)
        self.assertNotIn("<table", markdown)

    def test_grouped_checkbox_becomes_task_and_hidden_marker_is_not_repeated(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 560, 65],
                "content_spans": [
                    content_span(
                        "☐",
                        [25, 30, 37, 42],
                        fusion_visualization_hidden=True,
                    ),
                    content_span(
                        "Out-patient treatment",
                        [25, 27, 260, 46],
                        fusion_checkbox_grouped=True,
                        fusion_checkbox_state="checked",
                    ),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- [x] Out-patient treatment", markdown)
        self.assertEqual(markdown.count("Out-patient treatment"), 1)
        self.assertNotIn("☐", markdown)

    def test_hidden_list_marker_does_not_duplicate_joined_list_item(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 560, 65],
                "content_spans": [
                    content_span(
                        "1.",
                        [25, 30, 38, 42],
                        fusion_visualization_hidden=True,
                        fusion_grouped_list_marker=True,
                    ),
                    content_span(
                        "1. ContentABCDEFG",
                        [25, 27, 260, 46],
                        fusion_grouped_list_item=True,
                    ),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertEqual(markdown.count("1. ContentABCDEFG"), 1)

    def test_isolated_unpunctuated_list_marker_joins_nearby_content(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 560, 65],
                "content_spans": [
                    content_span("1", [25, 30, 32, 42]),
                    content_span("Please sign here", [38, 28, 180, 44]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("1. Please sign here", markdown)
        self.assertNotIn("\n\n1\n\n", markdown)

    def test_embedded_data_image_is_not_emitted_as_markdown_text(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 560, 65],
                "content_spans": [
                    content_span("Histology Sent", [40, 28, 180, 44]),
                    content_span(
                        '<img src="data:image/jpeg;base64,/9j/example"/>',
                        [200, 28, 210, 44],
                    ),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("Histology Sent", markdown)
        self.assertNotIn("base64", markdown)
        self.assertNotIn("<img", markdown)

    def test_ledger_columns_are_rebuilt_from_header_geometry(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [10, 10, 590, 120],
                "content_spans": [
                    content_span("DATE", [20, 20, 70, 35]),
                    content_span("CODE報", [120, 20, 170, 35]),
                    content_span("PARTICULARS", [220, 20, 330, 35]),
                    content_span("AMOUNT", [400, 20, 470, 35]),
                    content_span("BALANCE", [510, 20, 580, 35]),
                    content_span("01/07/2026", [20, 55, 90, 70]),
                    content_span("LAB", [120, 55, 160, 70]),
                    content_span("Blood test", [220, 55, 320, 70]),
                    content_span("50.00", [410, 55, 460, 70]),
                    content_span("74,791.00", [510, 55, 580, 70]),
                    content_span("02/07/2026", [20, 75, 90, 90]),
                    content_span("MED", [120, 75, 160, 90]),
                    content_span("Medicine", [220, 75, 320, 90]),
                    content_span("1.4.00", [410, 75, 460, 90]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn(
            "| Date / 日期 | Code / 代號 | Particulars / 項目 | Amount / 金額 | Balance / 結餘 |",
            markdown,
        )
        self.assertIn(
            "| 01/07/2026 | LAB | Blood test | 50.00 | 74,791.00 |",
            markdown,
        )
        self.assertIn("| 02/07/2026 | MED | Medicine | 1.4.00 |  |", markdown)

    def test_ledger_tail_preserves_admission_and_discharge_fields(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [10, 10, 590, 180],
                "content_spans": [
                    content_span("DATE", [20, 20, 70, 35]),
                    content_span("CODE", [120, 20, 170, 35]),
                    content_span("PARTICULARS", [220, 20, 330, 35]),
                    content_span("AMOUNT", [400, 20, 470, 35]),
                    content_span("BALANCE", [510, 20, 580, 35]),
                    content_span("01/07/2026", [20, 55, 90, 70]),
                    content_span("100", [120, 55, 160, 70]),
                    content_span("Room", [220, 55, 320, 70]),
                    content_span("50.00", [410, 55, 460, 70]),
                    content_span("附註", [20, 100, 70, 115]),
                    content_span("入院日期\nDate Admitted", [300, 100, 390, 130]),
                    content_span("01/07/2026", [420, 100, 500, 115]),
                    content_span("10:30", [510, 100, 560, 115]),
                    content_span("出院日期\nDate Discharged", [300, 130, 400, 160]),
                    content_span("02/07/2026:11:45", [420, 135, 560, 150]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Date Admitted / 入院日期**: 01/07/2026 10:30", markdown)
        self.assertIn("- **Date Discharged / 出院日期**: 02/07/2026 11:45", markdown)
        self.assertNotIn("| 附註", markdown)

    def test_repeated_margin_page_number_and_stamp_noise_are_removed(self):
        pages = []
        for page_number in (1, 2):
            pages.append(
                {
                    "page_size": [600, 800],
                    "preproc_blocks": [
                        text_block("Hospital Claims Department", [20, 15, 300, 30]),
                        text_block(f"Unique body {page_number}", [20, 200, 300, 220]),
                        text_block("PAID", [250, 400, 310, 430]),
                        text_block(f"Page {page_number} of 2", [250, 770, 350, 790]),
                    ],
                }
            )

        markdown = generate_semantic_markdown(middle(pages=pages))

        self.assertEqual(markdown.count("Hospital Claims Department"), 1)
        self.assertIn("Unique body 1", markdown)
        self.assertIn("Unique body 2", markdown)
        self.assertNotIn("PAID", markdown)
        self.assertNotIn("Page 1 of 2", markdown)
        self.assertNotIn("Page 2 of 2", markdown)

    def test_person_name_in_title_block_is_not_promoted_to_heading(self):
        markdown = generate_semantic_markdown(
            middle(
                text_block("Blaine Bai", [20, 20, 150, 40], block_type="title"),
                text_block(
                    "Hospital Bill",
                    [20, 60, 180, 80],
                    block_type="title",
                ),
            )
        )

        self.assertIn("\n\nBlaine Bai\n\n", markdown)
        self.assertNotIn("## Blaine Bai", markdown)
        self.assertIn("## Hospital Bill", markdown)

    def test_write_semantic_markdown_uses_default_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            middle_path = Path(temp_dir) / "sample_middle.json"
            middle_path.write_text(
                json.dumps(middle(text_block("Hello", [20, 20, 100, 40]))),
                encoding="utf-8",
            )

            output = write_semantic_markdown(middle_path)

            self.assertEqual(output.name, "sample_semantic.md")
            self.assertIn("Hello", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

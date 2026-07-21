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

    def test_ledger_tail_omits_empty_date_field_and_maps_single_value(self):
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
                    content_span("入院日期\nDate Admitted", [300, 100, 390, 130]),
                    content_span("出院日期\nDate Discharged", [300, 130, 400, 160]),
                    content_span("02/07/2026 11:45", [420, 135, 560, 150]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertNotIn("Date Admitted / 入院日期", markdown)
        self.assertIn("- **Date Discharged / 出院日期**: 02/07/2026 11:45", markdown)

    def test_repeated_ledger_tail_metadata_is_emitted_once_per_document(self):
        def ledger_page():
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
                        content_span("附註", [20, 100, 70, 115]),
                        content_span("Payment is due on receipt.", [20, 120, 220, 135]),
                        content_span("入院日期\nDate Admitted", [300, 100, 390, 130]),
                        content_span("01/07/2026", [420, 100, 500, 115]),
                    ],
                }
            ]
            return {
                "page_size": [600, 800],
                "preproc_blocks": [table_block(cells)],
            }

        markdown = generate_semantic_markdown(
            middle(pages=[ledger_page(), ledger_page()])
        )

        self.assertEqual(markdown.count("Payment is due on receipt."), 1)
        self.assertEqual(markdown.count("Date Admitted / 入院日期"), 1)

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

    def test_adjacent_label_row_pairs_with_aligned_value_row(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 50],
                "content_spans": [
                    content_span("Policy No. 保單號碼", [25, 25, 120, 35]),
                    content_span("Name 姓名", [160, 25, 230, 35]),
                    content_span("Age 年齡 / Sex 性別", [300, 25, 390, 35]),
                    content_span("ID / Passport No.", [440, 25, 555, 35]),
                ],
            },
            {
                "row_start": 1,
                "col_start": 0,
                "bbox": [20, 45, 580, 85],
                "content_spans": [
                    content_span("P-12345", [25, 48, 120, 72]),
                    content_span("陳嘉欣", [160, 48, 230, 72]),
                    content_span("42 M", [300, 48, 390, 72]),
                    content_span("K456789", [440, 48, 555, 72]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Policy No. 保單號碼**: P-12345", markdown)
        self.assertIn("- **Name 姓名**: 陳嘉欣", markdown)
        self.assertIn("- **Age 年齡 / Sex 性別**: 42 M", markdown)
        self.assertIn("- **ID / Passport No.**: K456789", markdown)

    def test_adjacent_value_cell_groups_multiline_address_and_drops_fallback_label(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 1,
                "bbox": [110, 20, 260, 60],
                "text": "Address 地址",
            },
            {
                "row_start": 0,
                "col_start": 2,
                "bbox": [270, 20, 335, 60],
                "content_spans": [
                    content_span("Address 地址", [275, 32, 330, 42]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 3,
                "bbox": [340, 20, 590, 70],
                "content_spans": [
                    content_span("ASIA CLINIC", [345, 22, 430, 31]),
                    content_span("27/F, 26 Nathan Road, H.K.", [345, 34, 520, 45]),
                    content_span("Tel: 2317 1717", [345, 48, 430, 59]),
                    content_span("Fax: 2736 8877", [445, 48, 535, 59]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertEqual(markdown.count("**Address 地址**"), 1)
        self.assertIn(
            "- **Address 地址**: ASIA CLINIC / 27/F, 26 Nathan Road, H.K. / "
            "Tel: 2317 1717 / Fax: 2736 8877",
            markdown,
        )

    def test_field_with_own_cell_value_does_not_absorb_right_neighbor(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 250, 70],
                "content_spans": [
                    content_span("Policy No.", [25, 25, 100, 36]),
                    content_span("P-12345", [25, 45, 100, 62]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 1,
                "bbox": [255, 20, 580, 70],
                "content_spans": [
                    content_span("Name 姓名", [260, 25, 340, 36]),
                    content_span("陳嘉欣", [260, 45, 340, 62]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Policy No.**: P-12345", markdown)
        self.assertIn("- **Name 姓名**: 陳嘉欣", markdown)
        self.assertNotIn("P-12345 / 陳嘉欣", markdown)

    def test_cross_cell_aggregate_uses_each_cells_authoritative_value(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [15, 591, 156, 639],
                "text": "Policy No. 保單號碼SLHk4455667",
                "content_spans": [
                    content_span("Policy No. 保單號碼", [16, 588, 90, 601]),
                    content_span(
                        "SLHk4455667 Ho Tin Yan",
                        [16, 606, 270, 637],
                    ),
                ],
            },
            {
                "row_start": 0,
                "col_start": 1,
                "bbox": [158, 591, 299, 639],
                "text": "Name of Insured 受保人姓名Ho Tin Yan",
                "content_spans": [
                    content_span(
                        "Name of Insured 受保人姓名",
                        [160, 590, 261, 600],
                    ),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Policy No. 保單號碼**: SLHk4455667", markdown)
        self.assertIn("- **Name of Insured 受保人姓名**: Ho Tin Yan", markdown)
        self.assertNotIn("SLHk4455667 Ho Tin Yan", markdown)

    def test_value_mounted_on_later_cell_returns_to_matching_field_cell(self):
        cells = [
            {
                "row_start": 3,
                "row_end": 4,
                "col_start": 0,
                "bbox": [17, 453, 149, 486],
                "text": (
                    "CONSULTANT'S INFORMATION 顧問資料 "
                    "Name 姓名 Mary So"
                ),
                "content_spans": [
                    content_span("Name 姓名", [20, 469, 60, 479]),
                ],
            },
            {
                "row_start": 5,
                "col_start": 0,
                "bbox": [18, 480, 345, 509],
                "text": "Claimed Benefit(s) 索償保障類別:",
                "content_spans": [
                    content_span("Mary So", [23, 469, 127, 505]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Name 姓名**: Mary So", markdown)
        self.assertEqual(markdown.count("Mary So"), 1)

    def test_cross_cell_aggregate_label_does_not_steal_individual_fields(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 150, 75],
                "text": "Name 姓名梁志往",
                "content_spans": [
                    content_span("Name 姓名", [25, 25, 75, 36]),
                    content_span("梁志", [30, 48, 75, 68]),
                    content_span("往", [70, 48, 92, 68]),
                    content_span(
                        "District/Branch 區域/分行 Code 編號 "
                        "Contact Phone No. 聯絡電話",
                        [145, 24, 570, 40],
                    ),
                ],
            },
            {
                "row_start": 0,
                "col_start": 1,
                "bbox": [155, 20, 290, 75],
                "text": "District/Branch 區域/分行Hong Kong Island",
                "content_spans": [
                    content_span(
                        "District/Branch 區域/分行",
                        [160, 25, 265, 36],
                    ),
                    content_span("Hong Kong Island", [165, 48, 275, 68]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 2,
                "bbox": [295, 20, 430, 75],
                "text": "Code 編號1112233",
                "content_spans": [
                    content_span("Code 編號", [300, 25, 350, 36]),
                    content_span("1112233", [305, 48, 365, 68]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 3,
                "bbox": [435, 20, 580, 75],
                "text": "Contact Phone No. 聯絡電話9876 5432",
                "content_spans": [
                    content_span(
                        "Contact Phone No. 聯絡電話",
                        [440, 25, 555, 36],
                    ),
                    content_span("9876 5432", [445, 48, 525, 68]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Name 姓名**: 梁志往", markdown)
        self.assertIn(
            "- **District/Branch 區域/分行**: Hong Kong Island",
            markdown,
        )
        self.assertIn("- **Code 編號**: 1112233", markdown)
        self.assertIn(
            "- **Contact Phone No. 聯絡電話**: 9876 5432",
            markdown,
        )
        self.assertNotIn(
            "District/Branch 區域/分行 Code 編號 Contact Phone No.",
            markdown,
        )

    def test_signature_zone_uses_fixed_name_id_and_date_columns(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 130],
                "text": (
                    "Signature of Policy Owner 保單主權人簽署 X\n"
                    "Name (in block letters) 姓名(大寫) CHEUNG SIU LING\n"
                    "ID / Passport No. V123456(7)\n"
                    "Date (DD/MM/YY) 15/03/26\n"
                    "Signature of Insured 受保人簽署 X\n"
                    "Name (in block letters) 姓名(大寫) CHEUNG SIU LING\n"
                    "ID / Passport No. V123456(7)\n"
                    "Date (DD/MM/YY) 15/03/26"
                ),
                "content_spans": [
                    content_span(
                        "Signature of Policy Owner 保單主權人簽署 X",
                        [25, 30, 170, 40],
                    ),
                    content_span(
                        "Name (in block letters) 姓名(大寫)",
                        [25, 42, 150, 52],
                    ),
                    content_span("CHEUNG SIU LING", [160, 42, 280, 60]),
                    content_span("ID / Passport No.", [330, 30, 405, 40]),
                    content_span("V/23456()", [395, 42, 460, 60]),
                    content_span("Date (DD/MM/YY)", [470, 30, 560, 40]),
                    content_span("15/03/26", [505, 42, 570, 60]),
                    content_span(
                        "Signature of Insured 受保人簽署 X",
                        [25, 75, 165, 85],
                    ),
                    content_span(
                        "Name (in block letters) 姓名(大寫)",
                        [25, 87, 150, 97],
                    ),
                    content_span("CHEUNG SIU LING", [160, 87, 280, 105]),
                    content_span("ID / Passport No.", [330, 75, 405, 85]),
                    content_span("V123456(7)", [395, 87, 460, 105]),
                    content_span("Date (DD/MM/YY)", [470, 75, 560, 85]),
                    content_span("15/03/26", [505, 87, 570, 105]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertEqual(markdown.count("CHEUNG SIU LING"), 2)
        self.assertEqual(markdown.count("V123456(7)"), 2)
        self.assertNotIn("V/23456()", markdown)
        self.assertEqual(markdown.count("15/03/26"), 2)
        self.assertNotIn("CHEUNG SIU LING / Signature", markdown)

    def test_medical_grid_preserves_columns_and_stops_at_next_question(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 180],
                "content_spans": [
                    content_span(
                        "7. Type of diagnostic procedures, medication, treatment or operation required.",
                        [25, 25, 555, 38],
                    ),
                    content_span("Date 日期", [30, 50, 75, 62]),
                    content_span(
                        "Investigation/ Result 檢查/結果",
                        [140, 50, 275, 62],
                    ),
                    content_span(
                        "Medication/ Treatment/ Operation 藥物/治療/手術",
                        [350, 50, 555, 62],
                    ),
                    content_span("10/03/26", [30, 70, 90, 90]),
                    content_span("X-ray of right ankle", [140, 70, 280, 88]),
                    content_span("Partial ligament tear", [140, 90, 280, 108]),
                    content_span("NSAIDS and RICE", [350, 70, 500, 90]),
                    content_span(
                        "8. Was the patient admitted into hospital?",
                        [25, 125, 330, 140],
                    ),
                    content_span("No", [40, 145, 70, 160]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn(
            "| Date 日期 | Investigation/ Result 檢查/結果 | "
            "Medication/ Treatment/ Operation 藥物/治療/手術 |",
            markdown,
        )
        self.assertIn(
            "| 10/03/26 | X-ray of right ankle<br>Partial ligament tear | "
            "NSAIDS and RICE |",
            markdown,
        )
        self.assertIn("8. Was the patient admitted into hospital?", markdown)
        self.assertNotIn("NSAIDS and RICE<br>8. Was", markdown)

    def test_physician_footer_uses_fixed_label_and_value_columns(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 155],
                "content_spans": [
                    content_span("Certificate body", [25, 25, 300, 38]),
                    content_span("Signed 簽名:", [25, 55, 90, 68]),
                    content_span("scribble", [115, 48, 240, 75]),
                    content_span("Name of physician", [275, 52, 365, 64]),
                    content_span("(with stamp)", [275, 64, 340, 75]),
                    content_span("Dr. Chan Chi Wai", [400, 52, 540, 75]),
                    content_span("Qualifications 資歷", [25, 88, 105, 100]),
                    content_span("MBBS", [115, 86, 180, 103]),
                    content_span("FHKAM", [115, 101, 190, 118]),
                    content_span("Address 地址", [275, 88, 345, 100]),
                    content_span("10 Nathan Road", [400, 86, 540, 103]),
                    content_span("Date 日期:", [25, 125, 85, 137]),
                    content_span("31/03/26", [115, 122, 185, 145]),
                    content_span("Telephone Number 電話號碼", [275, 125, 390, 137]),
                    content_span("27894321", [410, 122, 500, 145]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Signed 簽名**: [Signature]", markdown)
        self.assertIn(
            "- **Name of physician (with stamp) / 醫生的姓名(蓋印)**: "
            "Dr. Chan Chi Wai",
            markdown,
        )
        self.assertIn("- **Qualifications / 資歷**: MBBS / FHKAM", markdown)
        self.assertIn("- **Address 地址**: 10 Nathan Road", markdown)
        self.assertIn("- **Date / 日期**: 31/03/26", markdown)
        self.assertIn("- **Telephone Number / 電話號碼**: 27894321", markdown)

    def test_semantic_replay_suppresses_impossible_thin_recovery_sentence(self):
        hallucination = (
            "1. 2016年，公司与上海华谊（集团）股份有限公司"
            "（以下简称“公司”）签署的《股份转让协议》。"
        )
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [430, 312, 565, 340],
                "content_spans": [
                    content_span(
                        hallucination,
                        [432.696, 318.7, 563.051, 326.233],
                        fusion_recovery_source="local_uncovered_pixel_ink",
                        fusion_recognition_original_ocr="",
                        fusion_recognition_source="vlm",
                    ),
                    content_span(
                        "ID / Passport No. 身份證/護照號碼",
                        [432, 328, 560, 338],
                    ),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertNotIn("股份转让协议", markdown)
        self.assertIn("ID / Passport No.", markdown)

    def test_semantic_replay_suppresses_repeated_empty_recovery_phrase(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 300, 70],
                "content_spans": [
                    content_span(
                        "1. 证明：证明：证明：证明：",
                        [25, 30, 280, 55],
                        fusion_recovery_source="local_uncovered_pixel_ink",
                        fusion_recognition_original_ocr="",
                        fusion_recognition_source="vlm",
                    )
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertNotIn("证明", markdown)

    def test_question_and_partial_date_are_not_absorbed_as_name_or_age(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 280, 90],
                "content_spans": [
                    content_span("Name of Patient", [25, 25, 130, 36]),
                    content_span("陳嘉欣", [25, 43, 100, 60]),
                    content_span("Are you the patient's usual doctor?", [25, 68, 250, 80]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 1,
                "bbox": [285, 20, 580, 90],
                "content_spans": [
                    content_span("Sex / Age", [290, 25, 370, 36]),
                    content_span("M 42", [290, 43, 340, 60]),
                    content_span("11/22", [290, 68, 340, 82]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Name of Patient**: 陳嘉欣", markdown)
        self.assertIn("- **Sex / Age**: M 42", markdown)
        self.assertNotIn("陳嘉欣 / Are you", markdown)
        self.assertNotIn("M 42 / 11/22", markdown)

    def test_stacked_bilingual_field_labels_share_adjacent_value_cell(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 2,
                "bbox": [260, 20, 390, 80],
                "content_spans": [
                    content_span("Name of physician", [265, 25, 370, 36]),
                    content_span("(with stamp)", [265, 37, 340, 48]),
                    content_span("醫生的姓名", [265, 50, 340, 63]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 3,
                "bbox": [395, 20, 580, 80],
                "content_spans": [
                    content_span("林浩然醫生", [400, 32, 520, 60]),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn(
            "- **Name of physician / 醫生的姓名**: 林浩然醫生",
            markdown,
        )
        self.assertNotIn("(with stamp)", markdown)

    def test_formula_noise_is_not_assigned_to_address_field(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 180, 80],
                "content_spans": [
                    content_span("Address 地址", [25, 25, 140, 38]),
                ],
            },
            {
                "row_start": 0,
                "col_start": 1,
                "bbox": [185, 20, 580, 80],
                "content_spans": [
                    content_span(
                        "Room 1502, Mong Kok Plaza, Kowloon",
                        [190, 25, 430, 42],
                    ),
                    content_span(
                        r"<eq>N a = H _ { 2 } \cdot R _ { 2 } d</eq>",
                        [190, 48, 430, 68],
                    ),
                ],
            },
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn(
            "- **Address 地址**: Room 1502, Mong Kok Plaza, Kowloon",
            markdown,
        )
        self.assertNotIn(r"\cdot", markdown)
        self.assertNotIn("H _ { 2 }", markdown)

    def test_trailing_date_punctuation_does_not_move_date_into_name_field(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 120],
                "content_spans": [
                    content_span("Date of Operation", [25, 25, 150, 36]),
                    content_span("7/1/25.", [200, 38, 270, 62]),
                    content_span("Name of Surgeon", [25, 70, 150, 82]),
                    content_span("Dr Lee Tai Yam", [200, 82, 340, 108]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Date of Operation**: 7/1/25", markdown)
        self.assertIn("- **Name of Surgeon**: Dr Lee Tai Yam", markdown)
        self.assertNotIn("Name of Surgeon**: 7/1/25", markdown)

    def test_id_field_rejects_plain_name_but_keeps_alphanumeric_identifier(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 300, 90],
                "content_spans": [
                    content_span("ID / Passport No.", [25, 25, 150, 36]),
                    content_span("ELVIN", [25, 43, 100, 58]),
                    content_span("X510(b)", [25, 63, 110, 82]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **ID / Passport No.**: X510(b)", markdown)
        self.assertNotIn("ID / Passport No.**: ELVIN", markdown)

    def test_external_member_form_uses_generic_identifiers_contacts_and_iso_values(self):
        fields = [
            ("Member ID", "AB-12345"),
            ("Date of Birth", "1984-03-14"),
            ("Email Address", "alex@example.com"),
            ("Mailing Address", "10 Downing Street, London"),
            ("Postal Code", "SW1A 1AA"),
            ("Claim Amount", "USD 1,250.00"),
        ]
        cells = []
        for row, (label, value) in enumerate(fields):
            cells.extend(
                [
                    {
                        "row_start": row,
                        "col_start": 0,
                        "bbox": [20, 20 + row * 35, 250, 50 + row * 35],
                        "content_spans": [
                            content_span(label, [25, 25 + row * 35, 180, 37 + row * 35])
                        ],
                    },
                    {
                        "row_start": row,
                        "col_start": 1,
                        "bbox": [255, 20 + row * 35, 580, 50 + row * 35],
                        "content_spans": [
                            content_span(value, [265, 28 + row * 35, 500, 45 + row * 35])
                        ],
                    },
                ]
            )

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        for label, value in fields:
            self.assertIn(f"- **{label}**: {value}", markdown)

    def test_generic_instruction_with_member_word_is_not_promoted_to_field(self):
        paragraph = (
            "The member may update this application after approval and should keep "
            "the reference number for future enquiries."
        )
        markdown = generate_semantic_markdown(
            middle(table_block([
                {
                    "row_start": 0,
                    "col_start": 0,
                    "bbox": [20, 20, 580, 70],
                    "content_spans": [content_span(paragraph, [25, 25, 560, 45])],
                }
            ]))
        )

        self.assertIn(paragraph, markdown)
        self.assertNotIn(f"- **{paragraph}", markdown)

    def test_nearby_ocr_label_fragment_does_not_block_field_value_pair(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 280, 90],
                "content_spans": [
                    content_span("Name 姓名", [25, 25, 90, 36]),
                    content_span("ne 姓名", [50, 33, 120, 47]),
                    content_span("陳嘉欣", [50, 47, 120, 64]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Name 姓名**: 陳嘉欣", markdown)
        self.assertNotIn("ne 姓名", markdown)

    def test_long_legal_text_with_field_word_is_not_rendered_as_field(self):
        paragraph = (
            "The Policy Owner agrees to bear all bank charges and amount differences "
            "subject to the exchange rate determined at the time of payment."
        )
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 80],
                "content_spans": [content_span(paragraph, [25, 25, 560, 45])],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn(paragraph, markdown)
        self.assertNotIn(f"- **{paragraph}", markdown)

    def test_compact_date_and_uppercase_handwriting_are_rendered_conservatively(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 300, 90],
                "content_spans": [
                    content_span("Date 日期", [25, 25, 90, 36]),
                    content_span("7/1125", [25, 42, 100, 66]),
                    content_span("CH KELVIN", [130, 42, 250, 66]),
                    content_span("21|11|25", [25, 68, 100, 84]),
                    content_span("74.791.00", [130, 68, 220, 84]),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("7/11/25", markdown)
        self.assertIn("21/11/25", markdown)
        self.assertIn("74,791.00", markdown)
        self.assertIn("CH KELVIN", markdown)
        self.assertNotIn("### CH KELVIN", markdown)

    def test_identifier_prefix_conflict_uses_preserved_ocr_value(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 300, 80],
                "text": "Code 編號\n11112233",
                "content_spans": [
                    content_span("Code 編號", [25, 25, 90, 36]),
                    content_span(
                        "11112233",
                        [25, 42, 100, 66],
                        fusion_recognition_source="vlm",
                        fusion_recognition_original_ocr="H112233",
                        fusion_recognition_vlm="11112233",
                    ),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn("- **Code 編號**: H112233", markdown)
        self.assertNotIn("11112233", markdown)

    def test_crowded_multiline_empty_ocr_recovery_is_suppressed(self):
        cells = [
            {
                "row_start": 0,
                "col_start": 0,
                "bbox": [20, 20, 580, 90],
                "text": "District/Branch 區域/分行\nHong Kong Island",
                "content_spans": [
                    content_span(
                        "District/Branch 區域/分行",
                        [25, 25, 180, 36],
                    ),
                    content_span("Hong Kong Island", [25, 48, 180, 68]),
                    content_span(
                        "ON 顧問資料\nDistrict Research 區域分行\nQ.1. 錢號\nQ.1.4. Place N.",
                        [190, 44, 570, 60],
                        fusion_recovery_source="local_split_pixel_ink_merge",
                        fusion_recovery_spanning_cells=True,
                        fusion_recognition_original_ocr="",
                        fusion_recognition_reason="empty_ocr_vlm_recovery",
                    ),
                ],
            }
        ]

        markdown = generate_semantic_markdown(middle(table_block(cells)))

        self.assertIn(
            "- **District/Branch 區域/分行**: Hong Kong Island",
            markdown,
        )
        self.assertNotIn("District Research", markdown)
        self.assertNotIn("Q.1.4. Place", markdown)

    def test_fragment_only_ocr_noise_page_is_suppressed(self):
        noise_blocks = [
            text_block(text, [20, 20 + index * 12, 100, 30 + index * 12])
            for index, text in enumerate(
                (
                    "1",
                    "11",
                    "1r",
                    "BTSLY",
                    "611111",
                    ".1",
                    "T3190",
                    "j",
                    "1 1",
                    "31h",
                    "F1",
                    "1118",
                )
            )
        ]
        pages = [
            {
                "page_size": [600, 800],
                "preproc_blocks": noise_blocks,
            },
            {
                "page_size": [600, 800],
                "preproc_blocks": [text_block("Useful body text", [20, 100, 200, 120])],
            },
        ]

        markdown = generate_semantic_markdown(middle(pages=pages))

        self.assertNotIn("BTSLY", markdown)
        self.assertNotIn("<!-- Page 1 -->", markdown)
        self.assertIn("Useful body text", markdown)

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

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from projects.custom_hybrid.document_output import (
    DocumentOutput,
    build_document_output,
    write_document_output,
)


def _table_block() -> dict[str, Any]:
    return {
        "type": "table",
        "bbox": [10, 100, 190, 200],
        "blocks": [
            {
                "type": "table_body",
                "lines": [
                    {
                        "spans": [
                            {
                                "type": "table",
                                "table_cells": [
                                    {
                                        "bbox": [10, 100, 100, 130],
                                        "text": "Item",
                                        "row_start": 0,
                                        "row_end": 0,
                                        "col_start": 0,
                                        "col_end": 0,
                                        "is_header": True,
                                        "confidence": 0.98,
                                    },
                                    {
                                        "bbox": [100, 100, 190, 130],
                                        "text": "Amount",
                                        "row_start": 0,
                                        "row_end": 0,
                                        "col_start": 1,
                                        "col_end": 1,
                                        "is_header": True,
                                        "confidence": 0.97,
                                    },
                                ],
                            }
                        ]
                    }
                ],
            }
        ],
    }


def _artifacts() -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    middle_json = {
        "_backend": "hybrid",
        "_version_name": "test",
        "pdf_info": [
            {"page_idx": 0, "page_size": [200, 400], "para_blocks": []},
            {
                "page_idx": 1,
                "page_size": [200, 400],
                "para_blocks": [_table_block()],
                "discarded_blocks": [],
            },
            {"page_idx": 2, "page_size": [200, 400], "para_blocks": []},
        ],
    }
    content_list_v2 = [
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Hospital receipt"}
                    ]
                },
                "bbox": [100, 50, 600, 90],
            }
        ],
        [
            {
                "type": "table",
                "content": {
                    "html": "<table><tr><th>Item</th><th>Amount</th></tr></table>",
                    "table_type": "simple_table",
                    "table_nest_level": 1,
                    "table_caption": [],
                    "table_footnote": [],
                    "image_source": {"path": "images/table.png"},
                },
                "bbox": [50, 250, 950, 500],
            }
        ],
        [
            {
                "type": "image",
                "content": {
                    "content": "Round stamp",
                    "image_caption": [],
                    "image_footnote": [],
                    "image_source": {"path": "images/stamp.png"},
                },
                "bbox": [750, 800, 900, 950],
            }
        ],
    ]
    sorting_report = {
        "grouping_status": "complete",
        "ordering_status": "preserved_packet_order",
        "can_auto_group": True,
        "can_auto_sort": True,
        "grouping_strategy": "packet_document_segmentation",
        "groups": [
            {
                "group_id": "doc-001",
                "document_kind": "receipt",
                "grouping_status": "complete",
                "ordering_status": "preserved_packet_order",
                "member_page_ids": ["p0000", "p0001"],
                "resolved_order": ["p0000", "p0001"],
            },
            {
                "group_id": "doc-002",
                "document_kind": "receipt",
                "grouping_status": "complete",
                "ordering_status": "validated_internal_pagination",
                "member_page_ids": ["p0002"],
                "resolved_order": ["p0002"],
            },
        ],
    }
    return middle_json, content_list_v2, sorting_report


class DocumentOutputTests(unittest.TestCase):
    def test_reference_style_payload_remains_valid(self) -> None:
        output = DocumentOutput.model_validate(
            {
                "documents": [
                    {
                        "document_index": 0,
                        "page_range": "1",
                        "pages": [
                            {
                                "page_number": 1,
                                "blocks": [
                                    {
                                        "block_id": "p1-b1",
                                        "page_number": 1,
                                        "text": "Receipt",
                                        "bbox": [0.1, 0.05, 0.5, 0.04],
                                        "type": "text",
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "raw_metadata": {"fixture": "reference-style"},
            }
        )

        self.assertEqual(output.schema_version, "1.0")
        self.assertEqual(output.coordinate_system.bbox_format, "xywh")

    def test_merges_grouping_text_visuals_and_table_cells(self) -> None:
        middle_json, content_list_v2, sorting_report = _artifacts()

        output = build_document_output(
            middle_json,
            content_list_v2,
            sorting_report,
        )

        self.assertEqual([item.page_range for item in output.documents], ["1-2", "3"])
        self.assertEqual(output.documents[0].group_id, "doc-001")
        self.assertEqual(
            output.raw_metadata["document_grouping"]["mode"],
            "sorting_report",
        )
        first_block = output.documents[0].pages[0].blocks[0]
        self.assertEqual(first_block.text, "Hospital receipt")
        self.assertEqual(first_block.bbox, (0.1, 0.05, 0.5, 0.04))
        table_page = output.documents[0].pages[1]
        self.assertEqual(
            [block.block_id for block in table_page.blocks],
            ["p2-t0-r0-c0", "p2-t0-r0-c1"],
        )
        self.assertEqual(table_page.blocks[0].bbox, (0.05, 0.25, 0.45, 0.075))
        self.assertTrue(table_page.blocks[0].table_cell.is_header)
        self.assertEqual(
            table_page.tables[0].cell_block_ids,
            ["p2-t0-r0-c0", "p2-t0-r0-c1"],
        )
        figure = output.documents[1].pages[0].blocks[0]
        self.assertEqual((figure.type, figure.raw_type), ("figure", "image"))
        self.assertEqual(figure.asset_path, "images/stamp.png")

    def test_invalid_partition_falls_back_without_dropping_pages(self) -> None:
        middle_json, content_list_v2, sorting_report = _artifacts()
        sorting_report["groups"][1]["member_page_ids"] = ["p0001", "p0002"]

        output = build_document_output(
            middle_json,
            content_list_v2,
            sorting_report,
        )

        self.assertEqual(len(output.documents), 1)
        self.assertEqual(output.documents[0].page_range, "1-3")
        self.assertTrue(output.documents[0].needs_review)
        self.assertEqual(
            output.raw_metadata["document_grouping"]["reason"],
            "incomplete_or_overlapping_partition",
        )

    def test_write_document_output_uses_adjacent_artifacts(self) -> None:
        middle_json, content_list_v2, sorting_report = _artifacts()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            middle_path = root / "sample_middle.json"
            middle_path.write_text(json.dumps(middle_json), encoding="utf-8")
            (root / "sample_content_list_v2.json").write_text(
                json.dumps(content_list_v2),
                encoding="utf-8",
            )
            (root / "sample_sorting_report.json").write_text(
                json.dumps(sorting_report),
                encoding="utf-8",
            )

            output_path = write_document_output(middle_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(output_path.name, "sample_document.json")
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(len(payload["documents"]), 2)


if __name__ == "__main__":
    unittest.main()

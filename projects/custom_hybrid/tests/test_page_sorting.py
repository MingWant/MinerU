import copy
import json
import tempfile
import unittest
from pathlib import Path

from projects.custom_hybrid.page_sorting import (
    REPORT_ONLY_MODE,
    analyze_middle_json,
    build_evidence,
    write_page_sorting_reports,
)


def text_block(
    text: str,
    block_type: str = "text",
    bbox: list[float] | None = None,
) -> dict:
    resolved_bbox = bbox or [20, 120, 560, 145]
    return {
        "type": block_type,
        "bbox": resolved_bbox,
        "lines": [
            {
                "bbox": resolved_bbox,
                "spans": [
                    {
                        "type": "text",
                        "bbox": resolved_bbox,
                        "content": text,
                    }
                ],
            }
        ],
    }


def numbered_page(
    current: str | int,
    total: str | int,
    family: str = "Example Claim Form",
    identifier: str | None = None,
    identifier_label: str = "Case ID",
    page_idx: int | None = None,
    marker: str | None = None,
) -> dict:
    footer = marker or f"{family} P.{current}/{total}"
    page = {
        "page_idx": page_idx,
        "page_size": [600, 800],
        "discarded_blocks": [
            text_block(footer, "footer", [100, 770, 500, 790])
        ],
        "preproc_blocks": [],
    }
    if identifier is not None:
        page["preproc_blocks"].append(
            text_block(
                f"{identifier_label}: {identifier}",
                bbox=[350, 80, 570, 105],
            )
        )
    return page


def payload(pages: list[dict]) -> dict:
    result = copy.deepcopy(pages)
    for index, page in enumerate(result):
        if page.get("page_idx") is None:
            page["page_idx"] = index
    return {"pdf_info": result}


class PageSortingTests(unittest.TestCase):
    def test_groups_and_sorts_interleaved_documents_with_different_totals(self) -> None:
        alpha = [
            numbered_page(index, 3, "Alpha Claim Form")
            for index in range(1, 4)
        ]
        beta = [
            numbered_page(index, 2, "Beta Invoice")
            for index in range(1, 3)
        ]
        mixed = [alpha[2], beta[1], alpha[0], beta[0], alpha[1]]

        _manifest, report = analyze_middle_json(payload(mixed))

        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["can_auto_sort"])
        self.assertEqual(report["unresolved"], [])
        groups = {group["expected_total"]: group for group in report["groups"]}
        self.assertEqual(groups[3]["resolved_order"], ["p0002", "p0004", "p0000"])
        self.assertEqual(groups[2]["resolved_order"], ["p0003", "p0001"])

    def test_same_template_uses_identifier_present_on_every_page(self) -> None:
        left = [
            numbered_page(index, 3, identifier="CASE-A111")
            for index in range(1, 4)
        ]
        right = [
            numbered_page(index, 3, identifier="CASE-B222")
            for index in range(1, 4)
        ]
        mixed = [page for pair in zip(left, right) for page in pair]

        _manifest, report = analyze_middle_json(payload(mixed))

        self.assertEqual(report["status"], "complete")
        groups = {
            group["identifiers"]["case"][0]: group
            for group in report["groups"]
        }
        self.assertEqual(
            groups["casea111"]["resolved_order"],
            ["p0000", "p0002", "p0004"],
        )
        self.assertEqual(
            groups["caseb222"]["resolved_order"],
            ["p0001", "p0003", "p0005"],
        )

    def test_same_total_uses_distinct_header_footer_fingerprints(self) -> None:
        left = [
            numbered_page(index, 2, family="Alpha Medical Report")
            for index in range(1, 3)
        ]
        right = [
            numbered_page(index, 2, family="Beta Invoice")
            for index in range(1, 3)
        ]
        mixed = [left[1], right[1], left[0], right[0]]

        _manifest, report = analyze_middle_json(payload(mixed))

        self.assertEqual(report["status"], "complete")
        groups = {group["seed_page_id"]: group for group in report["groups"]}
        self.assertEqual(groups["p0002"]["resolved_order"], ["p0002", "p0000"])
        self.assertEqual(groups["p0003"]["resolved_order"], ["p0003", "p0001"])

    def test_same_template_with_identifier_only_on_page_one_stays_unresolved(self) -> None:
        left = [
            numbered_page(
                index,
                3,
                identifier="CASE-A111" if index == 1 else None,
            )
            for index in range(1, 4)
        ]
        right = [
            numbered_page(
                index,
                3,
                identifier="CASE-B222" if index == 1 else None,
            )
            for index in range(1, 4)
        ]
        mixed = [page for pair in zip(left, right) for page in pair]

        _manifest, report = analyze_middle_json(payload(mixed))

        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["can_auto_sort"])
        self.assertEqual(report["unresolved_count"], 4)
        self.assertTrue(
            all(item["reason"] == "ambiguous_group" for item in report["unresolved"])
        )

    def test_ocr_near_identifier_completes_unique_missing_slot(self) -> None:
        pages = [
            numbered_page(
                3,
                3,
                "Accident Benefit Claim Form",
                "SLHK5544332",
                identifier_label="Policy No.",
            ),
            numbered_page(1, 3, "Accident Benefit Claim Form"),
            numbered_page(
                2,
                3,
                "Accident Benefit Claim Form",
                "SLH1<554-4332",
                identifier_label="Policy No.",
            ),
        ]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["can_auto_sort"])
        self.assertEqual(
            report["groups"][0]["resolved_order"],
            ["p0001", "p0002", "p0000"],
        )
        decisions = {
            decision["page_id"]: decision
            for decision in report["groups"][0]["decisions"]
        }
        self.assertIn("policy_ocr_near_match", decisions["p0002"]["reasons"])

    def test_non_confusable_identifier_difference_remains_hard_conflict(self) -> None:
        pages = [
            numbered_page(
                1,
                3,
                "Accident Benefit Claim Form",
                "SLHK5544332",
                identifier_label="Policy No.",
            ),
            numbered_page(
                2,
                3,
                "Accident Benefit Claim Form",
                "SLHM5544332",
                identifier_label="Policy No.",
            ),
            numbered_page(
                3,
                3,
                "Accident Benefit Claim Form",
                "SLHK5544332",
                identifier_label="Policy No.",
            ),
        ]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["can_auto_sort"])
        unresolved = {item["page_id"]: item for item in report["unresolved"]}
        self.assertEqual(unresolved["p0001"]["reason"], "no_compatible_group")

    def test_ocr_near_identifier_requires_unique_missing_slot(self) -> None:
        pages = [
            numbered_page(
                1,
                4,
                "Accident Benefit Claim Form",
                "SLHK5544332",
                identifier_label="Policy No.",
            ),
            numbered_page(
                2,
                4,
                "Accident Benefit Claim Form",
                "SLH1<554-4332",
                identifier_label="Policy No.",
            ),
        ]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "needs_review")
        unresolved = {item["page_id"]: item for item in report["unresolved"]}
        self.assertEqual(unresolved["p0001"]["reason"], "no_compatible_group")

    def test_ocr_near_identifier_requires_matching_family(self) -> None:
        pages = [
            numbered_page(
                1,
                2,
                "Accident Benefit Claim Form",
                "SLHK5544332",
                identifier_label="Policy No.",
            ),
            numbered_page(
                2,
                2,
                "Warehouse Invoice",
                "SLH1<554-4332",
                identifier_label="Policy No.",
            ),
        ]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "needs_review")
        unresolved = {item["page_id"]: item for item in report["unresolved"]}
        self.assertEqual(unresolved["p0001"]["reason"], "no_compatible_group")

    def test_missing_page_keeps_group_incomplete(self) -> None:
        pages = [numbered_page(1, 3), numbered_page(3, 3)]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["can_auto_sort"])
        self.assertEqual(report["groups"][0]["status"], "incomplete")
        self.assertEqual(report["groups"][0]["missing_numbers"], [2])

    def test_duplicate_or_misread_logical_page_is_not_auto_sorted(self) -> None:
        pages = [
            numbered_page(1, 3),
            numbered_page(3, 3),
            numbered_page(3, 3),
        ]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["can_auto_sort"])
        self.assertEqual(report["groups"][0]["status"], "conflict")
        self.assertEqual(
            report["groups"][0]["duplicates"],
            {"3": ["p0001", "p0002"]},
        )

    def test_conflicting_explicit_markers_are_reported(self) -> None:
        conflict = numbered_page(2, 3)
        conflict["discarded_blocks"].append(
            text_block("Page 3 of 3", "header", [20, 5, 130, 25])
        )
        pages = [numbered_page(1, 3), conflict, numbered_page(3, 3)]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertFalse(report["can_auto_sort"])
        self.assertEqual(report["groups"][0]["status"], "conflict")
        self.assertEqual(
            report["groups"][0]["ambiguous_pages"]["p0001"],
            "conflicting_candidates",
        )

    def test_group_total_does_not_hide_conflicting_explicit_total(self) -> None:
        conflict = numbered_page(2, 3)
        conflict["discarded_blocks"].append(
            text_block("Page 2 of 4", "header", [20, 5, 130, 25])
        )
        pages = [numbered_page(1, 3), conflict, numbered_page(3, 3)]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertFalse(report["can_auto_sort"])
        self.assertEqual(report["groups"][0]["status"], "conflict")
        self.assertEqual(
            report["groups"][0]["ambiguous_pages"]["p0001"],
            "conflicting_candidates",
        )

    def test_bare_number_does_not_override_conflicting_explicit_total(self) -> None:
        conflict = numbered_page(2, 4, marker="Page 2 of 4")
        conflict["discarded_blocks"].append(
            text_block("2", "page_number", [560, 10, 575, 30])
        )
        pages = [numbered_page(1, 3), conflict, numbered_page(3, 3)]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertFalse(report["can_auto_sort"])
        unresolved = {item["page_id"]: item for item in report["unresolved"]}
        self.assertEqual(unresolved["p0001"]["reason"], "no_compatible_group")

    def test_body_number_is_not_page_number_evidence(self) -> None:
        page = {
            "page_idx": 0,
            "page_size": [600, 800],
            "preproc_blocks": [text_block("2")],
        }

        evidence = build_evidence(payload([page]))

        self.assertEqual(evidence[0].record.candidates, [])

    def test_roman_ocr_page_one_anchor_reports_missing_page(self) -> None:
        page = numbered_page("I", 2, marker="Page I of 2")

        manifest, report = analyze_middle_json(payload([page]))

        self.assertEqual(report["anchor_count"], 1)
        self.assertEqual(report["groups"][0]["missing_numbers"], [2])
        candidate = manifest["pages"][0]["candidates"][0]
        self.assertEqual(candidate["current"], 1)
        self.assertEqual(candidate["total"], 2)

    def test_chinese_page_marker_is_explicit_evidence(self) -> None:
        pages = [
            numbered_page(1, 2, marker="\u7b2c 1 \u9801\uff0c\u5171 2 \u9801"),
            numbered_page(2, 2, marker="\u5171 2 \u9801\uff0c\u7b2c 2 \u9801"),
        ]

        manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["can_auto_sort"])
        self.assertTrue(
            all(page["candidates"][0]["explicit"] for page in manifest["pages"])
        )

    def test_no_page_one_anchor_is_explicitly_unresolved(self) -> None:
        pages = [
            {
                "page_idx": 0,
                "page_size": [600, 800],
                "preproc_blocks": [text_block("Narrative text")],
            }
        ]

        _manifest, report = analyze_middle_json(payload(pages))

        self.assertEqual(report["status"], "no_page_anchors")
        self.assertEqual(report["unresolved_count"], 1)
        self.assertEqual(report["unresolved"][0]["reason"], "no_page_one_anchor")

    def test_write_reports_is_report_only_and_includes_semantic_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            middle_path = root / "sample_middle.json"
            semantic_path = root / "sample_semantic_report.json"
            original = json.dumps(payload([numbered_page(1, 1)]), indent=2)
            middle_path.write_text(original, encoding="utf-8")
            semantic_path.write_text(
                json.dumps(
                    {
                        "semantic_markdown_version": 5,
                        "pages": 1,
                        "fragment_heavy_pages": [1],
                        "unstructured_table_fallback_pages": [],
                        "text_bearing_pages_not_emitted": [],
                        "trace_accounting_ratio": 0.95,
                    }
                ),
                encoding="utf-8",
            )

            manifest_path, report_path = write_page_sorting_reports(
                middle_path,
                config={
                    "mode": REPORT_ONLY_MODE,
                    "include_semantic_diagnostics": True,
                },
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(middle_path.read_text(encoding="utf-8"), original)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["report_only"])
            self.assertEqual(report["semantic_quality"]["semantic_markdown_version"], 5)
            self.assertEqual(manifest["pages"][0]["semantic_flags"], ["fragment_heavy"])


if __name__ == "__main__":
    unittest.main()

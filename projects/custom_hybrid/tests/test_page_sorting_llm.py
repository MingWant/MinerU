import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from projects.custom_hybrid.page_sorting import analyze_middle_json, write_page_sorting_reports
from projects.custom_hybrid.page_sorting_llm import (
    PROMPT_VERSION,
    build_llm_input,
    parse_prediction,
    prediction_coverage_error,
    run_llm_assist,
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


def page(
    index: int,
    text: str,
    *,
    title: str | None = None,
    marker: str | None = None,
) -> dict:
    preproc = [text_block(title, "title", [50, 50, 550, 90])] if title else []
    discarded = [text_block(marker, "header", [10, 2, 100, 14])] if marker else []
    preproc.append(text_block(text))
    return {
        "page_idx": index,
        "page_size": [600, 800],
        "preproc_blocks": preproc,
        "discarded_blocks": discarded,
    }


def payload(pages: list[dict]) -> dict:
    return {"pdf_info": pages}


class FakeResponse:
    def __init__(self, body: dict):
        self.body = body
        self.headers = {"x-request-id": body.get("id", "fake-request")}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.body


class FakeClient:
    responses: list[dict] = []
    calls: list[dict] = []

    def __init__(self, *args: object, **kwargs: object):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        del args
        return None

    def post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.responses[len(self.calls) - 1])


class PageSortingLlmTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeClient.responses = []
        FakeClient.calls = []

    def test_parser_and_coverage_require_a_complete_partition(self) -> None:
        predicted, needs_review, error = parse_prediction(
            '```json\n{"documents":[{"pages":[2,1]},{"pages":[3]}],"needs_review":false}\n```'
        )

        self.assertIsNone(error)
        self.assertFalse(needs_review)
        self.assertEqual(predicted, [[2, 1], [3]])
        self.assertIsNone(prediction_coverage_error(predicted, 3))
        self.assertIn(
            "duplicate_pages=[2]",
            prediction_coverage_error([[1, 2], [2, 3]], 3) or "",
        )
        self.assertIn(
            "missing_pages=[3]",
            prediction_coverage_error([[1], [2]], 3) or "",
        )
        _predicted, _review, unassigned_error = parse_prediction(
            '{"documents":[{"pages":[1]}],"unassigned_pages":[2],"needs_review":true}'
        )
        self.assertEqual(unassigned_error, "unassigned_pages_must_be_empty")
        _predicted, _review, non_integer_error = parse_prediction(
            '{"documents":[{"pages":[1.5]}],"needs_review":true}'
        )
        self.assertEqual(non_integer_error, "source_pages_must_be_integers")

    def test_llm_input_marks_confirmed_packet_wrapper_as_non_merge_evidence(self) -> None:
        middle = payload(
            [
                page(0, "Invoice details", title="INVOICE", marker="Page 1 of 4"),
                page(1, "Invoice total", title="INVOICE", marker="Page 2 of 4"),
                page(2, "Guarantee terms", title="Letter of Guarantee", marker="Page 3 of 4"),
                page(3, "Signature", title="Letter of Guarantee", marker="Page 4 of 4"),
            ]
        )
        manifest, report = analyze_middle_json(middle)

        llm_input, stats = build_llm_input(middle, manifest, report, {})

        self.assertEqual(stats["page_count"], 4)
        self.assertNotIn("deterministic_baseline", llm_input)
        self.assertFalse(llm_input["packet_context"]["packet_wrapper"]["merge_evidence"])
        self.assertTrue(
            all(
                candidate["evidence_role"] == "packet_wrapper"
                for item in llm_input["pages"]
                for candidate in item["pagination_evidence"]
            )
        )
        _limited_input, limited_stats = build_llm_input(
            middle,
            manifest,
            report,
            {
                "max_total_input_chars": 40,
                "max_page_input_chars": 100,
                "min_page_input_chars": 50,
            },
        )
        self.assertEqual(limited_stats["page_text_limit"], 10)
        baseline_input, _baseline_stats = build_llm_input(
            middle,
            manifest,
            report,
            {"include_deterministic_baseline": True},
        )
        self.assertIn("deterministic_baseline", baseline_input)

    def test_invalid_first_response_gets_one_corrective_retry(self) -> None:
        middle = payload([page(0, "Invoice"), page(1, "Letter")])
        manifest, report = analyze_middle_json(middle)
        FakeClient.responses = [
            {
                "id": "attempt-1",
                "choices": [{"message": {"content": '{"documents":[],"needs_review":true}'}}],
                "usage": {"total_tokens": 10},
            },
            {
                "id": "attempt-2",
                "choices": [
                    {
                        "message": {
                            "content": '{"documents":[{"pages":[1]},{"pages":[2]}],"needs_review":false}'
                        }
                    }
                ],
                "usage": {"total_tokens": 12},
            },
        ]

        with mock.patch("httpx.Client", FakeClient):
            result = run_llm_assist(
                middle,
                manifest,
                report,
                {
                    "enabled": True,
                    "base_url": "https://example.test/v1",
                    "model": "qwen3-instruct",
                    "trigger": "always",
                },
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 22)
        self.assertEqual(
            result["proposal"]["documents"][1]["page_ids"],
            ["p0001"],
        )
        self.assertIn(
            "previous JSON failed validation".casefold(),
            FakeClient.calls[1]["json"]["messages"][-1]["content"].casefold(),
        )

    def test_write_reports_adds_llm_proposal_without_mutating_middle_json(self) -> None:
        middle = payload([page(0, "Invoice"), page(1, "Letter")])
        FakeClient.responses = [
            {
                "id": "valid",
                "choices": [
                    {
                        "message": {
                            "content": '{"documents":[{"pages":[2,1]}],"needs_review":true}'
                        }
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            middle_path = Path(temp_dir) / "sample_middle.json"
            original = json.dumps(middle, ensure_ascii=False, indent=2)
            middle_path.write_text(original, encoding="utf-8")

            with mock.patch("httpx.Client", FakeClient):
                manifest_path, report_path = write_page_sorting_reports(
                    middle_path,
                    config={
                        "mode": "report_only",
                        "llm": {
                            "enabled": True,
                            "base_url": "https://example.test",
                            "model": "qwen3-instruct",
                            "trigger": "always",
                        },
                    },
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(middle_path.read_text(encoding="utf-8"), original)
            self.assertEqual(report["llm_assist"]["status"], "complete")
            self.assertTrue(report["llm_assist"]["deterministic_result_retained"])
            self.assertFalse(report["llm_assist"]["applied"])
            self.assertEqual(manifest["pages"][1]["llm_document_group_id"], "llm-doc-001")
            self.assertEqual(manifest["pages"][1]["llm_sequence_position"], 1)
            self.assertEqual(manifest["pages"][0]["llm_sequence_position"], 2)

    def test_endpoint_failure_retains_deterministic_result(self) -> None:
        class FailingClient(FakeClient):
            def post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
                del url, headers, json
                raise RuntimeError("endpoint unavailable")

        middle = payload([page(0, "Invoice")])
        manifest, report = analyze_middle_json(middle)

        with mock.patch("httpx.Client", FailingClient):
            result = run_llm_assist(
                middle,
                manifest,
                report,
                {
                    "enabled": True,
                    "base_url": "https://example.test",
                    "model": "qwen3-instruct",
                    "trigger": "always",
                },
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("endpoint unavailable", result["error"])
        self.assertTrue(result["deterministic_result_retained"])

    def test_llm_cannot_silently_merge_across_confirmed_document_boundaries(self) -> None:
        middle = payload(
            [
                page(0, "Invoice details", title="INVOICE", marker="Page 1 of 4"),
                page(1, "Invoice total", title="INVOICE", marker="Page 2 of 4"),
                page(2, "Guarantee terms", title="Letter of Guarantee", marker="Page 3 of 4"),
                page(3, "Signature", title="Letter of Guarantee", marker="Page 4 of 4"),
            ]
        )
        manifest, report = analyze_middle_json(middle)
        FakeClient.responses = [
            {
                "id": "merge-all",
                "choices": [
                    {
                        "message": {
                            "content": '{"documents":[{"pages":[1,2,3,4]}],"needs_review":false}'
                        }
                    }
                ],
            }
        ]

        with mock.patch("httpx.Client", FakeClient):
            result = run_llm_assist(
                middle,
                manifest,
                report,
                {
                    "enabled": True,
                    "base_url": "https://example.test",
                    "model": "qwen3-instruct",
                    "trigger": "always",
                },
            )

        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["safe_for_automatic_use"])
        self.assertIn(
            "deterministic_grouping_conflict",
            {item["type"] for item in result["agreement"]["hard_conflicts"]},
        )
        self.assertTrue(result["deterministic_result_retained"])

    def test_unverified_trigger_skips_fully_validated_document_pagination(self) -> None:
        middle = payload(
            [
                page(0, "Report body", marker="Report Page 1 of 2"),
                page(1, "Report ending", marker="Report Page 2 of 2"),
            ]
        )
        manifest, report = analyze_middle_json(middle)
        self.assertEqual(report["ordering_status"], "validated_internal_pagination")

        result = run_llm_assist(
            middle,
            manifest,
            report,
            {
                "enabled": True,
                "base_url": "https://example.test",
                "model": "qwen3-instruct",
                "trigger": "unverified",
            },
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["prompt_version"], PROMPT_VERSION)
        self.assertEqual(result["trigger_reason"], "deterministic_internal_pagination_validated")


if __name__ == "__main__":
    unittest.main()

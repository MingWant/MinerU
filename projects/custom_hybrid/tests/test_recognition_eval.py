import unittest

from projects.custom_hybrid.recognition_eval import (
    evaluate_bbox_recognition,
    reselect_bbox_recognition_report,
)


class RecognitionEvaluationTests(unittest.TestCase):
    def test_reports_cer_improvement_and_bbox_reference_coverage(self):
        report = {
            "counts": {
                "candidates": 2,
                "requests": 1,
                "responses": 2,
                "invalid_outputs": 0,
                "errors": 0,
                "protocol_echoes": 0,
                "batch_quality_fallbacks": 0,
            },
            "recognition_batches": [{"status": "ok", "ids": ["a", "b"]}],
            "recognition_invariants": {
                "enabled": True,
                "bbox_unchanged": True,
                "table_structure_unchanged": True,
            },
            "decisions": [
                {
                    "kind": "bbox_recognition",
                    "page": 0,
                    "bbox": [10, 20, 80, 35],
                    "ocr_text": "25 DEQ 2Q25",
                    "vlm_text": "25 DEC 2023",
                    "selected_text": "25 DEC 2023",
                    "selected_source": "vlm",
                    "reason": "validated_date_repair",
                },
                {
                    "kind": "bbox_recognition",
                    "page": 0,
                    "bbox": [100, 20, 150, 35],
                    "ocr_text": "30156",
                    "vlm_text": "301S6",
                    "selected_text": "30156",
                    "selected_source": "ocr",
                    "reason": "high_risk_identifier_conflict",
                },
            ]
        }
        reference = {
            "items": [
                {"page": 0, "bbox": [10, 20, 80, 35], "text": "25 DEC 2023"},
                {"page": 0, "bbox": [100, 20, 150, 35], "text": "30156"},
            ]
        }

        result = evaluate_bbox_recognition(report, reference)

        self.assertGreater(
            result["sources"]["ocr_text"]["cer"],
            result["sources"]["selected_text"]["cer"],
        )
        self.assertEqual(result["sources"]["selected_text"]["exact_accuracy"], 1.0)
        self.assertEqual(len(result["improved_items"]), 1)
        self.assertEqual(result["regressed_items"], [])
        self.assertTrue(result["bbox_keys_unchanged"])
        self.assertTrue(result["table_structure_unchanged"])
        self.assertTrue(result["operational_health"]["healthy"])
        self.assertTrue(result["recommended_default_enabled"])

    def test_rejects_default_enable_when_geometry_invariant_fails(self):
        report = {
            "recognition_invariants": {
                "enabled": True,
                "bbox_unchanged": False,
                "table_structure_unchanged": True,
            },
            "decisions": [
                {
                    "kind": "bbox_recognition",
                    "page": 0,
                    "bbox": [10, 20, 80, 35],
                    "ocr_text": "B1aine",
                    "vlm_text": "Blaine",
                    "selected_text": "Blaine",
                    "selected_source": "vlm",
                    "reason": "quality_repair",
                }
            ],
        }
        reference = {
            "items": [
                {"page": 0, "bbox": [10, 20, 80, 35], "text": "Blaine"}
            ]
        }

        result = evaluate_bbox_recognition(report, reference)

        self.assertFalse(result["bbox_keys_unchanged"])
        self.assertFalse(result["recommended_default_enabled"])

    def test_rejects_default_enable_when_operational_evidence_is_unhealthy(self):
        report = {
            "counts": {
                "candidates": 1,
                "requests": 1,
                "responses": 1,
                "invalid_outputs": 0,
                "errors": 1,
            },
            "recognition_batches": [
                {"status": "error", "ids": ["p0-bbox-0"]}
            ],
            "recognition_invariants": {
                "enabled": True,
                "bbox_unchanged": True,
                "table_structure_unchanged": True,
            },
            "decisions": [
                {
                    "kind": "bbox_recognition",
                    "page": 0,
                    "bbox": [10, 20, 80, 35],
                    "ocr_text": "B1aine",
                    "vlm_text": "Blaine",
                    "selected_text": "Blaine",
                    "selected_source": "vlm",
                    "reason": "bbox_conditioned_vlm",
                }
            ],
        }
        reference = {
            "items": [
                {"page": 0, "bbox": [10, 20, 80, 35], "text": "Blaine"}
            ]
        }

        result = evaluate_bbox_recognition(report, reference)

        self.assertFalse(result["operational_health"]["healthy"])
        self.assertEqual(result["operational_health"]["counts"]["errors"], 1)
        self.assertFalse(result["recommended_default_enabled"])

    def test_replays_current_policy_over_captured_protocol_echo(self):
        report = {
            "counts": {
                "bbox_recognition_vlm_selected": 1,
                "bbox_recognition_ocr_kept": 0,
                "bbox_recognition_high_risk_fallbacks": 0,
            },
            "decisions": [
                {
                    "kind": "bbox_recognition",
                    "page": 6,
                    "bbox": [73, 171, 132, 185],
                    "block_type": "text",
                    "ocr_text": "Blaine Bai",
                    "ocr_confidence": None,
                    "vlm_text": "p6-0",
                    "selected_source": "vlm",
                    "selected_text": "p6-0",
                    "reason": "validated_identifier_repair",
                }
            ],
        }

        replayed = reselect_bbox_recognition_report(
            report,
            {"recognizer": {"enabled": True}},
        )

        decision = replayed["decisions"][0]
        self.assertEqual(decision["selected_source"], "ocr")
        self.assertEqual(decision["selected_text"], "Blaine Bai")
        self.assertEqual(decision["reason"], "bbox_protocol_id_echo")
        self.assertEqual(
            replayed["counts"]["bbox_recognition_vlm_selected"],
            0,
        )
        self.assertEqual(
            replayed["counts"]["bbox_recognition_protocol_echoes"],
            1,
        )
        self.assertTrue(replayed["selection_policy_replayed"])

    def test_replay_applies_batch_quality_guard(self):
        decisions = []
        for index, (ocr_text, vlm_text) in enumerate(
            (
                ("B1aine Bai", "Blaine Bai"),
                ("Room No.", "p0-0"),
                ("Date Discharged", "}{"),
            )
        ):
            decisions.append(
                {
                    "kind": "bbox_recognition",
                    "id": f"p0-bbox-{index}",
                    "page": 0,
                    "bbox": [10, 10 + index * 20, 100, 25 + index * 20],
                    "block_type": "text",
                    "ocr_text": ocr_text,
                    "ocr_confidence": None,
                    "vlm_text": vlm_text,
                }
            )
        report = {
            "counts": {},
            "decisions": decisions,
            "recognition_batches": [
                {
                    "ids": [decision["id"] for decision in decisions],
                    "status": "ok",
                }
            ],
        }

        replayed = reselect_bbox_recognition_report(
            report,
            {"recognizer": {"enabled": True}},
        )

        self.assertEqual(
            replayed["counts"]["bbox_recognition_batch_quality_fallbacks"],
            3,
        )
        self.assertTrue(
            replayed["recognition_batches"][0]["quality_guard_triggered"]
        )
        self.assertTrue(
            all(
                decision["reason"] == "recognizer_batch_quality_guard"
                for decision in replayed["decisions"]
            )
        )
        self.assertEqual(replayed["decisions"][0]["selected_text"], "B1aine Bai")


if __name__ == "__main__":
    unittest.main()

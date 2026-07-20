import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.fusion import (
    FusionSettings,
    OpenAIVisionVerifier,
    _parse_verifier_text,
    _parse_reconciliation_ids,
    apply_bbox_recognition,
    apply_bbox_recovery_proposals,
    build_bbox_recognition_manifest,
    build_bbox_recovery_manifest,
    assign_ocr_lines,
    collect_table_geometry_quality,
    collect_table_ocr_lines,
    collect_text_lines,
    collect_unreliable_table_ocr_lines,
    fuse_middle_json,
    recover_table_cell_geometry,
    select_bbox_recognition_candidate,
    synchronize_recognized_table_html,
)
from projects.custom_hybrid.table_fusion import (
    TableCellContext,
    align_table_cells,
    parse_table_html,
    rebuild_table_html,
)


def middle(text, score=None, *, span_type="text", bbox=(10, 10, 190, 30)):
    span = {"type": span_type, "content": text, "bbox": list(bbox)}
    if score is not None:
        span["score"] = score
    return {
        "_backend": "hybrid",
        "pdf_info": [
            {
                "page_size": [200, 300],
                "preproc_blocks": [
                    {
                        "type": "text",
                        "bbox": list(bbox),
                        "lines": [{"bbox": list(bbox), "spans": [span]}],
                    }
                ],
            }
        ],
    }


def structured_middle(
    span_type,
    *,
    html=None,
    content=None,
    score=None,
    table_cells=None,
):
    span = {
        "type": span_type,
        "bbox": [10, 10, 190, 80],
    }
    if html is not None:
        span["html"] = html
    if content is not None:
        span["content"] = content
    if score is not None:
        span["score"] = score
    if table_cells is not None:
        span["table_cells"] = table_cells
    block_type = "table_body" if span_type == "table" else "interline_equation"
    return {
        "_backend": "hybrid",
        "pdf_info": [
            {
                "page_size": [200, 300],
                "preproc_blocks": [
                    {
                        "type": block_type,
                        "lines": [
                            {"bbox": [10, 10, 190, 80], "spans": [span]}
                        ],
                    }
                ],
            }
        ],
    }


class FusionTests(unittest.TestCase):
    def test_bbox_vlm_settings_enable_repair_and_table_recognizer(self):
        settings = FusionSettings.from_mapping(
            {
                "mode": "bbox_vlm",
                "recognizer": {"enabled": False, "normal_ocr_enabled": True},
                "recovery": {"enabled": False},
            }
        )

        self.assertTrue(settings.bbox_recovery_enabled)
        self.assertTrue(settings.bbox_recognition_enabled)
        self.assertFalse(settings.bbox_recognition_normal_ocr_enabled)
        self.assertTrue(settings.bbox_recognition_table_ocr_enabled)
        self.assertEqual(settings.bbox_recognition_selection_policy, "vlm_primary")
        legacy = FusionSettings.from_mapping({"mode": "bbox_vlm_recovery"})
        self.assertEqual(legacy.mode, "bbox_vlm")
        self.assertTrue(legacy.bbox_recovery_enabled)

    def test_bbox_recovery_manifest_and_safety_gate_preserve_table_grid(self):
        cells = [
            {
                "bbox": [10, 10, 100, 40],
                "text": "Missing",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [100, 10, 190, 40],
                "content_spans": [
                    {"bbox": [110, 18, 175, 32], "text": "Existing"}
                ],
                "text": "Existing",
                "row_start": 0,
                "row_end": 0,
                "col_start": 1,
                "col_end": 1,
            },
        ]
        page = structured_middle(
            "table",
            html="<table><tr><td>Missing</td><td>Existing</td></tr></table>",
            table_cells=cells,
        )["pdf_info"][0]
        manifest = build_bbox_recovery_manifest(page, 0)

        self.assertEqual(manifest[0]["cells"][0]["reasons"], [
            "missing_content_bbox",
            "metadata_text_without_bbox",
        ])
        settings = FusionSettings.from_mapping({"mode": "bbox_vlm"})
        stats, decisions, _batches, unchanged = apply_bbox_recovery_proposals(
            page,
            0,
            {
                "items": [
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 18, 85, 32],
                        "confidence": 0.95,
                        "recovery_source": "local_pixel_ink",
                    },
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c1",
                        "target_id": "",
                        "bbox": [110, 18, 175, 32],
                        "confidence": 0.99,
                    },
                    {
                        "action": "adjust",
                        "cell_id": "p0-t0-c1",
                        "target_id": "p0-t0-c1-b0",
                        "bbox": [105, 16, 180, 34],
                        "confidence": 0.95,
                    },
                ]
            },
            settings,
            remaining_document_budget=10,
        )

        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["adjusted"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(decisions[1]["reason"], "duplicate_bbox")
        self.assertTrue(unchanged)
        recovered = page["preproc_blocks"][0]["lines"][0]["spans"][0][
            "table_cells"
        ][0]["content_spans"][0]
        self.assertEqual(recovered["bbox"], [20.0, 18.0, 85.0, 32.0])
        self.assertEqual(recovered["fusion_recovery_confidence"], 0.95)
        self.assertEqual(
            recovered["fusion_recovery_source"],
            "local_pixel_ink",
        )
        adjusted = page["preproc_blocks"][0]["lines"][0]["spans"][0][
            "table_cells"
        ][1]["content_spans"][0]
        self.assertEqual(adjusted["bbox"], [105.0, 16.0, 180.0, 34.0])
        self.assertEqual(
            adjusted["fusion_recovery_original_bbox"],
            [110, 18, 175, 32],
        )

    def test_bbox_recovery_rejects_cross_cell_duplicate_content_boxes(self):
        cells = [
            {
                "bbox": [10, 10, 110, 45],
                "text": "First",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [12, 12, 112, 47],
                "text": "Duplicate",
                "row_start": 1,
                "row_end": 1,
                "col_start": 0,
                "col_end": 0,
            },
        ]
        page = structured_middle(
            "table",
            html="<table><tr><td>First</td></tr><tr><td>Duplicate</td></tr></table>",
            table_cells=cells,
        )["pdf_info"][0]

        stats, decisions, _batches, unchanged = apply_bbox_recovery_proposals(
            page,
            0,
            {
                "items": [
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 18, 90, 35],
                        "confidence": 0.95,
                    },
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c1",
                        "target_id": "",
                        "bbox": [21, 18, 91, 35],
                        "confidence": 0.95,
                    },
                ]
            },
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            remaining_document_budget=10,
        )

        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(decisions[1]["reason"], "cross_cell_duplicate_bbox")
        self.assertTrue(unchanged)

    def test_bbox_vlm_repairs_box_then_transcribes_isolated_empty_crop(self):
        pipeline = structured_middle(
            "table",
            html="<table><tr><td></td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 40],
                    "text": "",
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )

        def review(_page, _size, manifest):
            self.assertEqual(manifest[0]["cells"][0]["existing"], [])
            return {
                "tables_reviewed": 1,
                "requests": 1,
                "local_proposals": 1,
                "local_only_tables": 1,
                "pixel_cells_analyzed": 1,
                "pixel_cells_skipped": 4,
                "diagonal_rules_removed": 2,
                "pixel_analysis_ms": 2.5,
                "items": [
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 18, 170, 32],
                        "confidence": 0.96,
                    }
                ],
                "batches": [{"status": "ok", "table_id": "p0-table-0"}],
            }

        def recognize(_page, _size, candidates):
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["bbox"], [20.0, 18.0, 170.0, 32.0])
            self.assertTrue(candidates[0]["recovered"])
            return {
                "items": [{"id": candidates[0]["id"], "text": "Recovered Value"}],
                "native_requests": 0,
                "native_cache_hits": 1,
                "native_cache_misses": 0,
                "native_cache_writes": 0,
                "native_budget_skipped": 0,
                "native_deduplicated_candidates": 0,
            }

        fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            bbox_recovery_reviewer=review,
            bbox_recognizer=recognize,
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertIn("<td>Recovered Value</td>", span["html"])
        self.assertEqual(report["counts"]["bbox_recovery_added"], 1)
        self.assertEqual(report["counts"]["bbox_recovery_local_proposals"], 1)
        self.assertEqual(report["counts"]["bbox_recovery_pixel_cells_skipped"], 4)
        self.assertEqual(
            report["counts"]["bbox_recovery_diagonal_rules_removed"],
            2,
        )
        self.assertEqual(report["counts"]["bbox_recovery_pixel_analysis_ms"], 2.5)
        self.assertEqual(report["counts"]["bbox_recognition_vlm_selected"], 1)
        self.assertEqual(
            report["counts"]["bbox_recognition_recovered_candidates"],
            1,
        )
        self.assertEqual(
            report["counts"]["bbox_recognition_recovered_responses"],
            1,
        )
        self.assertEqual(report["counts"]["bbox_recognition_native_cache_hits"], 1)
        self.assertTrue(
            report["recovery_invariants"]["table_and_cell_geometry_unchanged"]
        )

    def test_bbox_vlm_candidate_budget_preserves_recovered_crop(self):
        pipeline = structured_middle(
            "table",
            html="<table><tr><td>Existing</td><td></td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 95, 40],
                    "text": "Existing",
                    "content_spans": [
                        {"bbox": [20, 18, 80, 30], "text": "Existing"}
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                },
                {
                    "bbox": [95, 10, 190, 40],
                    "text": "",
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 1,
                    "col_end": 1,
                },
            ],
        )

        def review(_page, _size, _manifest):
            return {
                "tables_reviewed": 1,
                "local_proposals": 1,
                "items": [
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c1",
                        "target_id": "",
                        "bbox": [105, 18, 175, 32],
                        "confidence": 0.95,
                    }
                ],
            }

        def recognize(_page, _size, candidates):
            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0]["recovered"])
            return {"items": [{"id": candidates[0]["id"], "text": "Filled"}]}

        _fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping(
                {
                    "mode": "bbox_vlm",
                    "recognizer": {"max_candidates_per_document": 1},
                }
            ),
            bbox_recovery_reviewer=review,
            bbox_recognizer=recognize,
        )

        self.assertEqual(report["counts"]["bbox_recognition_candidate_limit"], 1)
        self.assertEqual(
            report["counts"]["bbox_recognition_recovered_candidates"],
            1,
        )
        self.assertEqual(
            report["counts"]["bbox_recognition_recovered_responses"],
            1,
        )

    def test_bbox_vlm_repairs_and_transcribes_table_orphan(self):
        pipeline = structured_middle(
            "table",
            html="<table><tr><td>Header</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 40],
                    "text": "Header",
                    "content_spans": [
                        {"bbox": [20, 18, 80, 30], "text": "Header"}
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )

        def review(_page, _size, _manifest):
            return {
                "tables_reviewed": 1,
                "local_proposals": 1,
                "orphan_tables_analyzed": 1,
                "orphan_proposals": 1,
                "items": [
                    {
                        "action": "add_orphan",
                        "table_id": "p0-table-0",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 60, 180, 75],
                        "confidence": 0.92,
                        "recovery_source": "local_table_orphan_ink",
                    }
                ],
            }

        def recognize(_page, _size, candidates):
            orphan = next(item for item in candidates if not item["ocr_text"])
            return {
                "items": [
                    {
                        "id": orphan["id"],
                        "text": "Delivery Option 退回方式",
                    }
                ]
            }

        fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            bbox_recovery_reviewer=review,
            bbox_recognizer=recognize,
        )

        table_span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0][
            "spans"
        ][0]
        cell = table_span["table_cells"][0]
        orphan = next(
            item
            for item in cell["content_spans"]
            if item.get("fusion_recovery_orphan")
        )
        self.assertEqual(orphan["bbox"], [20.0, 60.0, 180.0, 75.0])
        self.assertEqual(orphan["text"], "Delivery Option 退回方式")
        self.assertIn("Delivery Option 退回方式", table_span["html"])
        self.assertEqual(report["counts"]["bbox_recovery_orphan_added"], 1)
        self.assertEqual(
            report["counts"]["bbox_recovery_orphan_tables_analyzed"],
            1,
        )
        self.assertTrue(
            report["recovery_invariants"]["table_and_cell_geometry_unchanged"]
        )

    def test_bbox_vlm_repairs_and_transcribes_table_bottom_fringe(self):
        pipeline = structured_middle(
            "table",
            html="<table><tr><td>Header</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 40],
                    "text": "Header",
                    "content_spans": [
                        {"bbox": [20, 18, 80, 30], "text": "Header"}
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )

        def review(_page, _size, manifest):
            self.assertIn("page_existing", manifest[0])
            return {
                "tables_reviewed": 1,
                "local_proposals": 1,
                "fringe_proposals": 1,
                "items": [
                    {
                        "action": "add_fringe",
                        "table_id": "p0-table-0",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [155, 90, 178, 98],
                        "confidence": 0.92,
                        "recovery_source": "local_table_fringe_ink",
                    }
                ],
            }

        def recognize(_page, _size, candidates):
            fringe = next(item for item in candidates if not item["ocr_text"])
            self.assertGreater(fringe["bbox"][1], 80)
            return {"items": [{"id": fringe["id"], "text": "50.00"}]}

        fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            bbox_recovery_reviewer=review,
            bbox_recognizer=recognize,
        )

        table_span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0][
            "spans"
        ][0]
        cell = table_span["table_cells"][0]
        fringe = next(
            item
            for item in cell["content_spans"]
            if item.get("fusion_recovery_fringe")
        )
        self.assertEqual(fringe["bbox"], [155.0, 90.0, 178.0, 98.0])
        self.assertEqual(fringe["text"], "50.00")
        self.assertEqual(report["counts"]["bbox_recovery_fringe_added"], 1)
        self.assertEqual(report["counts"]["bbox_recovery_fringe_proposals"], 1)

    def test_rejected_empty_recovery_bbox_is_hidden_from_span_renderer(self):
        from mineru.utils.draw_bbox import _table_cell_render_bboxes

        pipeline = structured_middle(
            "table",
            html="<table><tr><td>Header</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "text": "Header",
                    "content_spans": [
                        {"bbox": [20, 18, 80, 30], "text": "Header"}
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )

        def review(_page, _size, _manifest):
            return {
                "tables_reviewed": 1,
                "local_proposals": 1,
                "items": [
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 50, 80, 62],
                        "confidence": 0.95,
                        "recovery_source": "local_uncovered_pixel_ink",
                    }
                ],
            }

        def recognize(_page, _size, candidates):
            recovered = next(item for item in candidates if not item["ocr_text"])
            return {
                "items": [
                    {"id": recovered["id"], "text": "[Non-Text]"}
                ]
            }

        fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            bbox_recovery_reviewer=review,
            bbox_recognizer=recognize,
        )

        table_span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0][
            "spans"
        ][0]
        recovered = next(
            item
            for item in table_span["table_cells"][0]["content_spans"]
            if item.get("fusion_recovery_source")
        )
        self.assertTrue(recovered["fusion_visualization_hidden"])
        _cells, visible = _table_cell_render_bboxes(table_span)
        self.assertNotIn([20.0, 50.0, 80.0, 62.0], visible)
        self.assertEqual(
            report["counts"]["bbox_recognition_recovered_hidden"],
            1,
        )

    def test_recovered_amount_with_repeated_separator_is_normalized(self):
        page = structured_middle(
            "table",
            html="<table><tr><td></td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "text": "",
                    "content_spans": [
                        {
                            "bbox": [120, 50, 180, 62],
                            "text": "",
                            "fusion_recovery_action": "add_orphan",
                            "fusion_recovery_confidence": 0.95,
                        }
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )["pdf_info"][0]
        lines = collect_table_ocr_lines(page, 0)

        def recognize(_page, _size, candidates):
            return {
                "items": [
                    {"id": candidates[0]["id"], "text": "74.791.00"}
                ]
            }

        stats, decisions, _batches = apply_bbox_recognition(
            0,
            [200, 300],
            lines,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            recognize,
        )

        self.assertEqual(stats["recovered_vlm_selected"], 1)
        self.assertEqual(decisions[0]["vlm_text"], "74.791.00")
        self.assertEqual(decisions[0]["normalized_vlm_text"], "74,791.00")
        self.assertEqual(decisions[0]["selected_text"], "74,791.00")
        self.assertEqual(lines[0].text, "74,791.00")

    def test_bbox_recovery_rejects_orphan_inside_existing_cell(self):
        page = structured_middle(
            "table",
            html="<table><tr><td>Header</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "text": "Header",
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )["pdf_info"][0]

        stats, decisions, _batches, unchanged = apply_bbox_recovery_proposals(
            page,
            0,
            {
                "items": [
                    {
                        "action": "add_orphan",
                        "table_id": "p0-table-0",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 20, 170, 40],
                        "confidence": 0.95,
                    }
                ]
            },
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            remaining_document_budget=10,
        )

        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(decisions[0]["reason"], "orphan_cell_overlap_guard")
        self.assertTrue(unchanged)

    def test_bbox_recovery_adds_local_checkbox_without_vlm_request(self):
        pipeline = structured_middle(
            "table",
            html="<table><tr><td>Option</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "text": "Option",
                    "content_spans": [
                        {"bbox": [40, 20, 100, 32], "text": "Option"}
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )

        def review(_page, _size, _manifest):
            return {
                "tables_reviewed": 1,
                "checkbox_tables_analyzed": 1,
                "checkbox_candidates": 1,
                "checkbox_proposals": 1,
                "checkbox_unchecked": 1,
                "items": [
                    {
                        "action": "add_checkbox",
                        "table_id": "p0-table-0",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 20, 30, 30],
                        "confidence": 0.95,
                        "text": "☐",
                        "checkbox_state": "unchecked",
                        "checkbox_interior_density": 0.0,
                        "recovery_source": "local_checkbox_detector",
                    }
                ],
            }

        def recognize(_page, _size, candidates):
            checkbox = next(item for item in candidates if item["ocr_text"] == "☐")
            self.assertLess(checkbox["bbox"][3] - checkbox["bbox"][1], 20)
            return {"items": [], "native_requests": 0}

        fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            bbox_recovery_reviewer=review,
            bbox_recognizer=recognize,
        )

        cell = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0][
            "table_cells"
        ][0]
        checkbox = next(
            item
            for item in cell["content_spans"]
            if item.get("fusion_recovery_checkbox")
        )
        self.assertEqual(checkbox["bbox"], [20.0, 20.0, 30.0, 30.0])
        self.assertEqual(checkbox["text"], "☐")
        self.assertEqual(checkbox["fusion_checkbox_state"], "unchecked")
        self.assertEqual(report["counts"]["bbox_recovery_checkbox_added"], 1)
        self.assertEqual(report["counts"]["bbox_recovery_checkbox_unchecked"], 1)
        self.assertTrue(
            report["recovery_invariants"]["table_and_cell_geometry_unchanged"]
        )

    def test_bbox_recovery_merges_checkbox_with_right_label_bbox(self):
        page = structured_middle(
            "table",
            html="<table><tr><td>Option</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "text": "Option",
                    "content_spans": [
                        {"bbox": [40, 20, 100, 32], "text": "Option"}
                    ],
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )["pdf_info"][0]

        stats, decisions, _batches, unchanged = apply_bbox_recovery_proposals(
            page,
            0,
            {
                "items": [
                    {
                        "action": "merge_checkbox",
                        "table_id": "p0-table-0",
                        "cell_id": "p0-t0-c0",
                        "target_id": "p0-t0-c0-b0",
                        "bbox": [20, 20, 100, 32],
                        "confidence": 0.95,
                        "checkbox_state": "checked",
                        "checkbox_interior_density": 0.2,
                        "recovery_source": "local_checkbox_detector",
                    }
                ]
            },
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            remaining_document_budget=10,
        )

        span = page["preproc_blocks"][0]["lines"][0]["spans"][0][
            "table_cells"
        ][0]["content_spans"][0]
        self.assertEqual(stats["checkbox_merged"], 1)
        self.assertEqual(span["bbox"], [20.0, 20.0, 100.0, 32.0])
        self.assertTrue(span["fusion_checkbox_grouped"])
        self.assertEqual(span["fusion_checkbox_state"], "checked")
        self.assertEqual(decisions[0]["result"], "accepted")
        self.assertTrue(unchanged)

    def test_bbox_recovery_enforces_confidence_area_and_document_budgets(self):
        page = structured_middle(
            "table",
            html="<table><tr><td></td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "text": "",
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )["pdf_info"][0]
        settings = FusionSettings.from_mapping({"mode": "bbox_vlm"})
        stats, decisions, _batches, unchanged = apply_bbox_recovery_proposals(
            page,
            0,
            {
                "items": [
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 20, 80, 35],
                        "confidence": 0.5,
                    },
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [10, 10, 190, 80],
                        "confidence": 0.99,
                    },
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [20, 20, 80, 35],
                        "confidence": 0.99,
                    },
                    {
                        "action": "add",
                        "cell_id": "p0-t0-c0",
                        "target_id": "",
                        "bbox": [100, 20, 160, 35],
                        "confidence": 0.99,
                    },
                ]
            },
            settings,
            remaining_document_budget=1,
        )

        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["rejected"], 3)
        self.assertEqual(
            [decision.get("reason") for decision in decisions],
            ["low_confidence", "area_ratio_guard", None, "document_budget"],
        )
        self.assertTrue(unchanged)

    def test_bbox_vlm_settings_force_table_only_vlm_primary_recognition(self):
        settings = FusionSettings.from_mapping(
            {
                "mode": "bbox_vlm",
                "recognizer": {
                    "enabled": False,
                    "normal_ocr_enabled": True,
                    "table_ocr_enabled": False,
                    "selection_policy": "conservative",
                },
            }
        )

        self.assertEqual(settings.mode, "bbox_vlm")
        self.assertTrue(settings.bbox_recognition_enabled)
        self.assertFalse(settings.bbox_recognition_normal_ocr_enabled)
        self.assertTrue(settings.bbox_recognition_table_ocr_enabled)
        self.assertEqual(settings.bbox_recognition_selection_policy, "vlm_primary")

    def test_vlm_primary_uses_valid_text_and_rejects_invalid_structured_values(self):
        settings = FusionSettings.from_mapping({"mode": "bbox_vlm"})
        general = collect_text_lines(middle("B1aine Bai")["pdf_info"][0], 0)[0]
        amount = collect_text_lines(middle("9.000.09")["pdf_info"][0], 0)[0]
        date = collect_text_lines(middle("25 DEQ 2Q25")["pdf_info"][0], 0)[0]
        identifier = collect_text_lines(middle("AB123")["pdf_info"][0], 0)[0]

        self.assertEqual(
            select_bbox_recognition_candidate(general, "Blaine Bai", settings)[:2],
            ("vlm", "bbox_vlm_primary"),
        )
        self.assertEqual(
            select_bbox_recognition_candidate(amount, "nine thousand", settings)[:2],
            ("ocr", "invalid_vlm_amount"),
        )
        self.assertEqual(
            select_bbox_recognition_candidate(date, "tomorrow", settings)[:2],
            ("ocr", "invalid_vlm_date"),
        )
        self.assertEqual(
            select_bbox_recognition_candidate(identifier, "wrong value", settings)[:2],
            ("ocr", "invalid_vlm_identifier"),
        )

    def test_bbox_vlm_mode_replaces_only_table_text_and_preserves_pipeline_geometry(self):
        original_html = "<table><tr><td>Name</td><td>B1aine Bai</td></tr></table>"
        cells = [
            {
                "bbox": [10, 10, 100, 40],
                "content_spans": [{"bbox": [20, 18, 70, 32], "text": "Name"}],
                "text": "Name",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [100, 10, 190, 40],
                "content_spans": [
                    {"bbox": [110, 18, 180, 32], "text": "B1aine Bai"}
                ],
                "text": "B1aine Bai",
                "row_start": 0,
                "row_end": 0,
                "col_start": 1,
                "col_end": 1,
            },
        ]
        pipeline = structured_middle(
            "table",
            html=original_html,
            table_cells=cells,
        )
        seen = []

        def recognize(_page, _size, candidates):
            seen.extend(candidates)
            return {
                "items": [
                    {
                        "id": candidate["id"],
                        "text": "Blaine Bai"
                        if candidate["ocr_text"] == "B1aine Bai"
                        else candidate["ocr_text"],
                    }
                    for candidate in candidates
                ]
            }

        fused, report = fuse_middle_json(
            pipeline,
            pipeline,
            FusionSettings.from_mapping({"mode": "bbox_vlm"}),
            bbox_recognizer=recognize,
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertTrue(seen)
        self.assertTrue(all(candidate["type"] == "table_ocr" for candidate in seen))
        self.assertIn("<td>Blaine Bai</td>", span["html"])
        self.assertEqual(
            parse_table_html(span["html"]).structure_signature,
            parse_table_html(original_html).structure_signature,
        )
        self.assertEqual(
            [cell["bbox"] for cell in span["table_cells"]],
            [[10, 10, 100, 40], [100, 10, 190, 40]],
        )
        self.assertEqual(
            [cell["content_spans"][0]["bbox"] for cell in span["table_cells"]],
            [[20, 18, 70, 32], [110, 18, 180, 32]],
        )
        self.assertEqual(
            [
                tuple(cell[key] for key in ("row_start", "row_end", "col_start", "col_end"))
                for cell in span["table_cells"]
            ],
            [(0, 0, 0, 0), (0, 0, 1, 1)],
        )
        self.assertEqual(report["mode"], "bbox_vlm")
        self.assertEqual(report["counts"]["bbox_recognition_vlm_selected"], 1)
        self.assertTrue(report["recognition_invariants"]["bbox_unchanged"])
        self.assertTrue(report["recognition_invariants"]["table_structure_unchanged"])

    def test_unkeyed_pipeline_cells_are_retained_for_bbox_rendering(self):
        html = "<table><tr><td>Key</td><td>Value</td></tr></table>"
        table_cells = [
            {
                "bbox": [10, 10, 90, 40],
                "content_bbox": [18, 18, 72, 32],
                "text": "Key",
            },
            {
                "bbox": [90, 10, 190, 40],
                "content_spans": [
                    {"bbox": [105, 18, 172, 32], "text": "Value"}
                ],
                "text": "Value",
            },
        ]

        fused, _report = fuse_middle_json(
            structured_middle("table", html=html),
            structured_middle("table", html=html, table_cells=table_cells),
            FusionSettings(),
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["table_cells"], table_cells)

    def test_missing_fused_cell_geometry_is_recovered_from_ocr_middle(self):
        html = "<table><tr><td>Key</td><td>Value</td></tr></table>"
        cells = [
            {"bbox": [10, 10, 90, 40], "content_bbox": [18, 18, 72, 32]},
            {
                "bbox": [90, 10, 190, 40],
                "content_spans": [{"bbox": [105, 18, 172, 32]}],
            },
        ]
        fused = structured_middle("table", html=html)
        ocr = structured_middle("table", html=html, table_cells=cells)

        attached, changed = recover_table_cell_geometry(fused, ocr)

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(attached, 2)
        self.assertTrue(changed)
        self.assertEqual(span["table_cells"], cells)

    def test_invalid_hybrid_table_falls_back_to_valid_pipeline_table(self):
        hybrid = structured_middle("table", html="<table><tr><td>broken")
        valid_html = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        ocr = structured_middle("table", html=valid_html)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], valid_html)
        self.assertEqual(span["fusion_source"], "pipeline_table_fallback")
        self.assertEqual(report["counts"]["table_fallback_replacements"], 1)

    def test_unsafe_pipeline_table_is_not_used_as_whole_table_fallback(self):
        hybrid_html = "<table><tr><td>broken"
        unsafe_html = "<table><tr><td><script>alert(1)</script>value</td></tr></table>"

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=unsafe_html),
            FusionSettings(),
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], hybrid_html)
        self.assertEqual(report["counts"]["table_fallback_replacements"], 0)

    def test_valid_hybrid_table_is_never_overwritten_by_fallback(self):
        hybrid_html = "<table><tr><td>Hybrid</td></tr></table>"
        pipeline_html = "<table><tr><td>Pipeline</td></tr></table>"
        hybrid = structured_middle("table", html=hybrid_html)
        ocr = structured_middle("table", html=pipeline_html)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], hybrid_html)
        self.assertEqual(report["counts"]["table_fallback_replacements"], 0)

    def test_visual_table_choice_can_select_pipeline_without_generating_html(self):
        hybrid_html = "<table><tr><td>1008</td></tr></table>"
        pipeline_html = "<table><tr><td>100B</td><td>USD</td></tr></table>"

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html),
            FusionSettings(),
            candidate_chooser=lambda kind, *_args: "pipeline" if kind == "table" else None,
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], pipeline_html)
        self.assertEqual(span["fusion_source"], "visual_pipeline_table")
        self.assertEqual(report["counts"]["table_conflicts"], 1)
        self.assertEqual(report["counts"]["table_visual_pipeline_replacements"], 1)

    def test_invalid_structured_choice_keeps_hybrid(self):
        hybrid_html = "<table><tr><td>Hybrid</td></tr></table>"
        pipeline_html = "<table><tr><td>Pipeline</td></tr></table>"

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html),
            FusionSettings(table_cell_fusion_enabled=False),
            candidate_chooser=lambda *_args: "merged",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], hybrid_html)
        self.assertEqual(report["counts"]["structured_verifier_rejections"], 1)

    def test_empty_formula_falls_back_but_existing_formula_is_preserved(self):
        ocr = structured_middle("interline_equation", content="x^2+y^2", score=0.97)

        fused_empty, report_empty = fuse_middle_json(
            structured_middle("interline_equation", content=""),
            ocr,
            FusionSettings(),
        )
        fused_existing, report_existing = fuse_middle_json(
            structured_middle("interline_equation", content="a+b"),
            ocr,
            FusionSettings(),
        )

        empty_span = fused_empty["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        existing_span = fused_existing["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(empty_span["content"], "x^2+y^2")
        self.assertEqual(empty_span["fusion_source"], "pipeline_formula_fallback")
        self.assertEqual(report_empty["counts"]["formula_fallback_replacements"], 1)
        self.assertEqual(existing_span["content"], "a+b")
        self.assertEqual(report_existing["counts"]["formula_fallback_replacements"], 0)

    def test_visual_formula_choice_can_keep_hybrid(self):
        fused, report = fuse_middle_json(
            structured_middle("interline_equation", content="x^2+y^2"),
            structured_middle("interline_equation", content="x^2-y^2"),
            FusionSettings(),
            candidate_chooser=lambda kind, *_args: "hybrid" if kind == "formula" else None,
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "x^2+y^2")
        self.assertEqual(report["counts"]["formula_conflicts"], 1)
        self.assertEqual(report["counts"]["formula_visual_hybrid_selections"], 1)

    def test_structured_verification_limit_keeps_hybrid(self):
        hybrid_html = "<table><tr><td>Hybrid</td></tr></table>"
        pipeline_html = "<table><tr><td>Pipeline</td></tr></table>"

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html),
            FusionSettings(
                max_structured_verifications_per_document=0,
                table_cell_fusion_enabled=False,
            ),
            candidate_chooser=lambda *_args: "pipeline",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], hybrid_html)
        self.assertEqual(report["counts"]["structured_verification_limit"], 1)

    def test_ocr_line_is_assigned_once_to_most_specific_overlapping_target(self):
        page = {
            "preproc_blocks": [
                {
                    "type": "text",
                    "lines": [
                        {
                            "bbox": [0, 0, 200, 100],
                            "spans": [
                                {
                                    "type": "text",
                                    "content": "large",
                                    "bbox": [0, 0, 200, 100],
                                }
                            ],
                        },
                        {
                            "bbox": [10, 10, 100, 30],
                            "spans": [
                                {
                                    "type": "text",
                                    "content": "specific",
                                    "bbox": [10, 10, 100, 30],
                                }
                            ],
                        },
                    ],
                }
            ]
        }
        ocr_page = middle("specific", 0.99, bbox=(10, 10, 100, 30))["pdf_info"][0]
        targets = collect_text_lines(page, 0)
        ocr_lines = collect_text_lines(ocr_page, 0)

        assignments = assign_ocr_lines(targets, ocr_lines, 0.5)

        self.assertEqual(assignments[id(targets[0])], [])
        self.assertEqual(assignments[id(targets[1])], ocr_lines)

    def test_table_grid_parser_and_rebuilder_preserve_rowspan_colspan(self):
        hybrid_html = (
            '<table class="data"><tr><th rowspan="2">Item</th><th>Value</th></tr>'
            "<tr><td><b>1008</b></td></tr></table>"
        )
        pipeline_html = (
            '<table><tr><th rowspan="2">Item</th><th>Value</th></tr>'
            "<tr><td><b>100B</b></td></tr></table>"
        )
        hybrid = parse_table_html(hybrid_html)
        pipeline = parse_table_html(pipeline_html)

        pairs = align_table_cells(hybrid, pipeline)
        replacement_key = (1, 1, 1, 1)
        rebuilt, error = rebuild_table_html(
            hybrid,
            {replacement_key: pipeline.cell_map[replacement_key].inner_html},
        )

        self.assertTrue(hybrid.valid)
        self.assertTrue(hybrid.coverage_complete)
        self.assertIsNotNone(pairs)
        self.assertIsNone(error)
        self.assertIn('rowspan="2"', rebuilt)
        self.assertIn("<b>100B</b>", rebuilt)
        self.assertEqual(
            parse_table_html(rebuilt).structure_signature,
            hybrid.structure_signature,
        )

        ragged = parse_table_html(
            "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
        )
        self.assertFalse(ragged.coverage_complete)
        self.assertIsNone(align_table_cells(ragged, ragged))

    def test_empty_hybrid_cell_uses_pipeline_without_replacing_table_structure(self):
        hybrid_html = (
            '<table><tr><th rowspan="2">Item</th><th>Value</th></tr>'
            "<tr><td></td></tr></table>"
        )
        pipeline_html = (
            '<table><tr><th rowspan="2">Item</th><th>Value</th></tr>'
            "<tr><td>42</td></tr></table>"
        )

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html),
            FusionSettings(),
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertIn("<td>42</td>", span["html"])
        self.assertIn('rowspan="2"', span["html"])
        self.assertEqual(span["fusion_source"], "cell_fused_table")
        self.assertEqual(report["counts"]["table_cell_empty_replacements"], 1)
        self.assertEqual(report["counts"]["table_structure_matches"], 1)

    def test_table_rebuilder_rejects_unsafe_pipeline_cell_html(self):
        hybrid = parse_table_html("<table><tr><td>safe</td></tr></table>")

        rebuilt, error = rebuild_table_html(
            hybrid,
            {(0, 0, 0, 0): '<img src="data:image/png;base64,secret">'},
        )

        self.assertIsNone(rebuilt)
        self.assertEqual(error, "unsafe_cell_content")

    def test_visual_table_cell_choice_replaces_only_target_cell(self):
        hybrid_html = "<table><tr><th>Code</th><th>Amount</th></tr><tr><td>1008</td><td>50</td></tr></table>"
        pipeline_html = "<table><tr><th>Code</th><th>Amount</th></tr><tr><td>100B</td><td>50</td></tr></table>"
        table_cells = [
            {
                "bbox": [10, 10, 90, 30],
                "text": "Code",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [90, 10, 190, 30],
                "text": "Amount",
                "row_start": 0,
                "row_end": 0,
                "col_start": 1,
                "col_end": 1,
            },
            {
                "bbox": [10, 30, 90, 60],
                "text": "100B",
                "content_bbox": [18, 38, 72, 52],
                "confidence": 0.99,
                "content_spans": [
                    {
                        "bbox": [18, 38, 72, 52],
                        "polygon": [18, 38, 72, 38, 72, 52, 18, 52],
                        "text": "100B",
                        "score": 0.99,
                    }
                ],
                "row_start": 1,
                "row_end": 1,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [90, 30, 190, 60],
                "text": "50",
                "row_start": 1,
                "row_end": 1,
                "col_start": 1,
                "col_end": 1,
            },
        ]
        seen = []

        def choose_cell(_page, _page_size, _table_bbox, context, hybrid, pipeline):
            seen.append((context, hybrid, pipeline))
            return "pipeline"

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html, table_cells=table_cells),
            FusionSettings(),
            table_cell_candidate_chooser=choose_cell,
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertIn("<td>100B</td><td>50</td>", span["html"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0].cell_bbox, (10.0, 30.0, 90.0, 60.0))
        self.assertEqual(len(seen[0][0].row_cells), 2)
        self.assertEqual(len(seen[0][0].column_cells), 2)
        fused_cells = span["table_cells"]
        self.assertEqual(fused_cells[2]["text"], "100B")
        self.assertEqual(fused_cells[2]["content_bbox"], [18, 38, 72, 52])
        self.assertEqual(fused_cells[2]["content_spans"][0]["score"], 0.99)
        self.assertEqual(report["counts"]["table_cell_visual_pipeline_replacements"], 1)
        self.assertEqual(report["counts"]["table_cell_consensus"], 3)

    def test_table_cell_conflict_without_reliable_bbox_keeps_hybrid(self):
        hybrid_html = "<table><tr><td>1008</td></tr></table>"
        pipeline_html = "<table><tr><td>100B</td></tr></table>"

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html),
            FusionSettings(),
            table_cell_candidate_chooser=lambda *_args: "pipeline",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], hybrid_html)
        self.assertEqual(report["counts"]["table_cell_missing_bbox"], 1)

    def test_table_cell_verification_limit_keeps_hybrid(self):
        hybrid_html = "<table><tr><td>1008</td></tr></table>"
        pipeline_html = "<table><tr><td>100B</td></tr></table>"
        table_cells = [
            {
                "bbox": [10, 10, 190, 80],
                "text": "100B",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            }
        ]

        fused, report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html, table_cells=table_cells),
            FusionSettings(max_table_cell_verifications_per_document=0),
            table_cell_candidate_chooser=lambda *_args: "pipeline",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["html"], hybrid_html)
        self.assertEqual(report["counts"]["table_cell_verification_limit"], 1)

    def test_suspicious_table_cell_requires_scored_ocr_by_default(self):
        hybrid_html = "<table><tr><td>abcdabcdabcd</td></tr></table>"
        pipeline_html = "<table><tr><td>abcd</td></tr></table>"
        unscored = [
            {
                "bbox": [10, 10, 190, 80],
                "text": "abcd",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            }
        ]
        scored = [{**unscored[0], "score": 0.99}]

        kept, kept_report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html, table_cells=unscored),
            FusionSettings(),
        )
        replaced, replaced_report = fuse_middle_json(
            structured_middle("table", html=hybrid_html),
            structured_middle("table", html=pipeline_html, table_cells=scored),
            FusionSettings(),
        )

        kept_span = kept["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        replaced_span = replaced["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(kept_span["html"], hybrid_html)
        self.assertIn("<td>abcd</td>", replaced_span["html"])
        self.assertEqual(kept_report["counts"]["table_cell_replacements"], 0)
        self.assertEqual(
            replaced_report["counts"]["table_cell_suspicious_replacements"],
            1,
        )

    def test_visual_verifier_crops_image_calls_endpoint_and_parses_json(self):
        from PIL import Image

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class FakeHttpx:
            def __init__(self):
                self.received = []

            def get(self, *_args, **_kwargs):
                return Response({"data": [{"id": "vision-model"}]})

            def post(self, *_args, json=None, **_kwargs):
                payload = json
                self.received.append(payload)
                prompt = payload["messages"][0]["content"][0]["text"]
                if "insert_candidate_ids" in prompt:
                    result = '{"insert_candidate_ids":["p0-ocr-0","unknown"]}'
                elif '"hybrid" or "ocr"' in prompt:
                    result = '{"source":"ocr"}'
                else:
                    result = '{"source":"pipeline"}'
                return Response({"choices": [{"message": {"content": result}}]})

        fake_httpx = FakeHttpx()
        verifier = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "page.png"
                Image.new("RGB", (200, 100), "white").save(image_path)
                verifier = OpenAIVisionVerifier(
                    "http://vision.test",
                    image_path,
                    {
                        "model": None,
                        "render_scale": 1.0,
                        "padding_ratio": 0.0,
                        "timeout_seconds": 5,
                    },
                )
                verifier.httpx = fake_httpx

                result = verifier(
                    0,
                    [200, 100],
                    [20, 10, 180, 40],
                    "invoice 1008",
                    "invoice 100B",
                    0.98,
                )
                selected = verifier.choose_candidate(
                    "table",
                    0,
                    [200, 100],
                    [20, 10, 180, 40],
                    '<table><tr><td><img src="data:image/png;base64,SECRET_TABLE_DATA">A</td></tr></table>',
                    "<table><tr><td>B</td></tr></table>",
                )
                selected_cell = verifier.choose_table_cell(
                    0,
                    [200, 100],
                    [10, 5, 190, 80],
                    TableCellContext(
                        key=(1, 1, 0, 0),
                        cell_bbox=(20, 30, 90, 50),
                        row_bbox=(20, 30, 180, 50),
                        column_bbox=(20, 10, 90, 70),
                        row_cells=(
                            {
                                "row_start": 1,
                                "row_end": 1,
                                "col_start": 0,
                                "col_end": 0,
                                "hybrid": "1008",
                                "pipeline": "100B",
                            },
                        ),
                        column_cells=(),
                    ),
                    "1008",
                    "100B",
                )
                reconciled = verifier.reconcile_page(
                    0,
                    [200, 100],
                    "existing text",
                    [
                        {
                            "id": "p0-ocr-0",
                            "bbox": [20, 50, 180, 70],
                            "text": "missing text",
                            "confidence": 0.8,
                            "type": "text",
                        }
                    ],
                    [],
                )

            self.assertEqual(result, "ocr")
            self.assertEqual(selected, "pipeline")
            self.assertEqual(selected_cell, "pipeline")
            self.assertEqual(reconciled, ["p0-ocr-0"])
            self.assertEqual(fake_httpx.received[0]["model"], "vision-model")
            self.assertEqual(
                fake_httpx.received[0]["response_format"],
                {"type": "json_object"},
            )
            text_content = fake_httpx.received[0]["messages"][0]["content"]
            self.assertIn("untrusted", text_content[0]["text"])
            self.assertTrue(
                text_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            )
            table_prompt = fake_httpx.received[1]["messages"][0]["content"][0]["text"]
            self.assertIn("embedded-data-removed", table_prompt)
            self.assertNotIn("SECRET_TABLE_DATA", table_prompt)
            cell_content = fake_httpx.received[2]["messages"][0]["content"]
            self.assertIn("whole_table, target_row, target_column, target_cell", cell_content[0]["text"])
            self.assertEqual(len(cell_content), 5)
            reconciliation_prompt = fake_httpx.received[3]["messages"][0]["content"][0]["text"]
            self.assertIn("Every returned ID must occur exactly", reconciliation_prompt)
        finally:
            if verifier is not None:
                verifier.close()

    def test_collects_editable_text_lines(self):
        lines = collect_text_lines(middle("hello")["pdf_info"][0], 0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "hello")

    def test_consensus_keeps_hybrid_text(self):
        hybrid = middle("Hello, world!")
        ocr = middle("Hello world", 0.99)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "Hello, world!")
        self.assertEqual(report["counts"]["consensus"], 1)

    def test_repeated_vlm_hallucination_uses_high_confidence_ocr(self):
        hybrid = middle("abcdabcdabcd")
        ocr = middle("abcd", 0.99)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "abcd")
        self.assertEqual(span["fusion_source"], "ocr_vlm")
        self.assertEqual(report["counts"]["ocr_replacements"], 1)

    def test_visual_verifier_can_select_ocr_candidate(self):
        hybrid = middle("invoice 1008")
        ocr = middle("invoice 100B", 0.98)

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(consensus_similarity=0.99),
            verifier=lambda *_args: "invoice 100B",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "invoice 100B")
        self.assertEqual(report["counts"]["verified_replacements"], 1)

    def test_source_only_visual_verifier_uses_exact_candidate(self):
        hybrid = middle("invoice 1008")
        ocr = middle("invoice 100B", 0.98)

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(consensus_similarity=0.99),
            verifier=lambda *_args: "ocr",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "invoice 100B")
        self.assertEqual(report["decisions"][0]["reason"], "visual_verifier_ocr")

    def test_high_confidence_unmatched_ocr_is_recovered_in_anchor_order(self):
        hybrid = middle("First", bbox=(10, 10, 190, 30))
        hybrid_block = hybrid["pdf_info"][0]["preproc_blocks"][0]
        hybrid_block["index"] = 10
        second = middle("Third", bbox=(10, 70, 190, 90))["pdf_info"][0][
            "preproc_blocks"
        ][0]
        second["index"] = 30
        hybrid["pdf_info"][0]["preproc_blocks"].append(second)

        ocr = middle("First", 0.99, bbox=(10, 10, 190, 30))
        missing = middle("Second", 0.97, bbox=(10, 40, 190, 60))["pdf_info"][0][
            "preproc_blocks"
        ][0]
        third = middle("Third", 0.99, bbox=(10, 70, 190, 90))["pdf_info"][0][
            "preproc_blocks"
        ][0]
        ocr["pdf_info"][0]["preproc_blocks"].extend([missing, third])

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        blocks = fused["pdf_info"][0]["preproc_blocks"]
        texts = [block["lines"][0]["spans"][0]["content"] for block in blocks]
        self.assertEqual(texts, ["First", "Second", "Third"])
        self.assertEqual(blocks[1]["fusion_source"], "ocr_recovered")
        self.assertEqual(blocks[1]["index"], 20)
        self.assertEqual(report["counts"]["missing_ocr_blocks_recovered"], 1)

    def test_missing_ocr_recovery_rejects_low_confidence_and_honors_budget(self):
        hybrid = middle("Anchor", bbox=(10, 10, 190, 30))
        ocr = middle("Anchor", 0.99, bbox=(10, 10, 190, 30))
        for text, score, bbox in (
            ("Low", 0.89, (10, 40, 190, 55)),
            ("Keep", 0.99, (10, 60, 190, 75)),
            ("Over budget", 0.99, (10, 80, 190, 95)),
        ):
            block = middle(text, score, bbox=bbox)["pdf_info"][0]["preproc_blocks"][0]
            ocr["pdf_info"][0]["preproc_blocks"].append(block)

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(max_missing_ocr_blocks_per_document=1),
        )

        contents = [
            block["lines"][0]["spans"][0]["content"]
            for block in fused["pdf_info"][0]["preproc_blocks"]
        ]
        self.assertEqual(contents, ["Anchor", "Keep"])
        self.assertEqual(report["counts"]["missing_ocr_candidates"], 2)
        self.assertEqual(report["counts"]["missing_ocr_blocks_recovered"], 1)

    def test_page_reconciliation_can_only_insert_manifest_candidate(self):
        hybrid = middle("Anchor", bbox=(10, 10, 190, 30))
        ocr = middle("Anchor", 0.99, bbox=(10, 10, 190, 30))
        low = middle("Visibly missing", 0.2, bbox=(10, 50, 190, 70))["pdf_info"][0][
            "preproc_blocks"
        ][0]
        ocr["pdf_info"][0]["preproc_blocks"].append(low)
        seen = []

        def reconcile(_page, _size, _hybrid, candidates, _recovered):
            seen.extend(candidates)
            return [candidates[0]["id"], "invented-id"]

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(page_reconciliation_enabled=True),
            page_reconciler=reconcile,
        )

        contents = [
            block["lines"][0]["spans"][0]["content"]
            for block in fused["pdf_info"][0]["preproc_blocks"]
        ]
        self.assertEqual(contents, ["Anchor", "Visibly missing"])
        self.assertEqual(seen[0]["text"], "Visibly missing")
        self.assertEqual(report["counts"]["page_reconciliation_requests"], 1)
        self.assertEqual(report["counts"]["page_reconciliation_insertions"], 1)

    def test_reconciliation_parser_rejects_free_text_and_deduplicates_ids(self):
        self.assertEqual(
            _parse_reconciliation_ids(
                '{"insert_candidate_ids":["p0-ocr-1","p0-ocr-1",7]}'
            ),
            ["p0-ocr-1"],
        )
        self.assertEqual(_parse_reconciliation_ids('{"text":"invented"}'), [])

    def test_bbox_recognition_preserves_geometry_and_recovers_vlm_text(self):
        hybrid = middle("Anchor", 0.99, bbox=(10, 10, 190, 30))
        ocr = middle("Anchor", 0.99, bbox=(10, 10, 190, 30))
        missing = middle("B1aine Bai", None, bbox=(20, 50, 100, 70))["pdf_info"][0][
            "preproc_blocks"
        ][0]
        ocr["pdf_info"][0]["preproc_blocks"].append(missing)
        seen = []

        def recognize(_page, _size, candidates):
            seen.extend(candidates)
            return {
                "items": [
                    {
                        "id": item["id"],
                        "text": "Blaine Bai"
                        if item["ocr_text"] == "B1aine Bai"
                        else item["ocr_text"],
                    }
                    for item in candidates
                ],
                "batches": [{"status": "ok", "latency_ms": 12.5}],
            }

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(bbox_recognition_enabled=True),
            bbox_recognizer=recognize,
        )

        recovered = [
            block
            for block in fused["pdf_info"][0]["preproc_blocks"]
            if block.get("fusion_source") == "ocr_recovered"
        ]
        self.assertEqual(recovered[0]["bbox"], [20.0, 50.0, 100.0, 70.0])
        self.assertEqual(
            recovered[0]["lines"][0]["spans"][0]["content"],
            "Blaine Bai",
        )
        candidate = next(item for item in seen if item["ocr_text"] == "B1aine Bai")
        self.assertEqual(candidate["bbox"], [20.0, 50.0, 100.0, 70.0])
        self.assertEqual(report["counts"]["bbox_recognition_vlm_selected"], 1)
        self.assertEqual(report["recognition_batches"][0]["latency_ms"], 12.5)
        self.assertEqual(
            report["recognition_invariants"],
            {
                "enabled": True,
                "bbox_unchanged": True,
                "table_structure_unchanged": True,
            },
        )

    def test_bbox_recognition_aggregates_request_and_error_audit_counts(self):
        def recognize(_page, _size, candidates):
            return {
                "items": [],
                "batches": [
                    {"status": "invalid_schema", "invalid_outputs": 2},
                    {"status": "error", "error": "TimeoutException"},
                ],
                "requests": 2,
                "invalid_outputs": 2,
                "errors": 1,
                "rebatches": 1,
            }

        _fused, report = fuse_middle_json(
            middle("OCR", 0.99),
            middle("OCR", 0.99),
            FusionSettings(bbox_recognition_enabled=True),
            bbox_recognizer=recognize,
        )

        counts = report["counts"]
        self.assertEqual(counts["bbox_recognition_requests"], 2)
        self.assertEqual(counts["bbox_recognition_invalid_outputs"], 2)
        self.assertEqual(counts["bbox_recognition_errors"], 1)
        self.assertEqual(counts["bbox_recognition_rebatches"], 1)
        self.assertEqual(counts["bbox_recognition_ocr_kept"], 1)
        self.assertEqual(len(report["recognition_batches"]), 2)

    def test_bbox_recognition_feature_flag_keeps_existing_output_unchanged(self):
        called = False

        def recognize(*_args):
            nonlocal called
            called = True
            return {"items": []}

        hybrid = middle("Hybrid")
        ocr = middle("OCR", 0.99)
        baseline, _baseline_report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(),
        )
        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(bbox_recognition_enabled=False),
            bbox_recognizer=recognize,
        )

        self.assertFalse(called)
        self.assertEqual(fused, baseline)
        self.assertEqual(report["counts"]["bbox_recognition_candidates"], 0)

    def test_bbox_recognition_high_risk_validators_are_conservative(self):
        date_line = collect_text_lines(middle("25 DEQ 2Q25")["pdf_info"][0], 0)[0]
        amount_line = collect_text_lines(middle("9.000.09")["pdf_info"][0], 0)[0]
        identifier_line = collect_text_lines(middle("3O156")["pdf_info"][0], 0)[0]
        settings = FusionSettings(bbox_recognition_enabled=True)

        date = select_bbox_recognition_candidate(date_line, "25 DEC 2023", settings)
        amount = select_bbox_recognition_candidate(amount_line, "9,000.00", settings)
        identifier = select_bbox_recognition_candidate(
            identifier_line,
            "30156",
            settings,
        )

        self.assertEqual(date[:3], ("vlm", "validated_date_repair", "date"))
        self.assertEqual(amount[:3], ("vlm", "validated_amount_repair", "amount"))
        self.assertEqual(
            identifier[:3],
            ("ocr", "high_risk_identifier_conflict", "identifier"),
        )

        protocol_echo = select_bbox_recognition_candidate(
            collect_text_lines(middle("FOQD/DRINKS")["pdf_info"][0], 0)[0],
            "p12-bbox-0",
            settings,
        )
        self.assertEqual(protocol_echo[0:2], ("ocr", "bbox_protocol_id_echo"))

        slot_echo = select_bbox_recognition_candidate(
            collect_text_lines(middle("Blaine Bai")["pdf_info"][0], 0)[0],
            "p6-0",
            settings,
        )
        self.assertEqual(slot_echo[0:2], ("ocr", "bbox_protocol_id_echo"))

        identifier_repair = select_bbox_recognition_candidate(
            collect_text_lines(middle("3O156")["pdf_info"][0], 0)[0],
            "30156",
            settings,
        )
        self.assertEqual(
            identifier_repair[0:2],
            ("ocr", "high_risk_identifier_conflict"),
        )

        word_with_digit_confusion = select_bbox_recognition_candidate(
            collect_text_lines(middle("B1aine")["pdf_info"][0], 0)[0],
            "Blaine",
            settings,
        )
        self.assertEqual(
            word_with_digit_confusion[0:3],
            ("vlm", "bbox_conditioned_vlm", "general"),
        )

        unrelated_valid_date = select_bbox_recognition_candidate(
            collect_text_lines(middle("Patient name")["pdf_info"][0], 0)[0],
            "25 DEC 2023",
            settings,
        )
        self.assertNotEqual(
            unrelated_valid_date[0:2],
            ("vlm", "validated_date_repair"),
        )

    def test_bbox_recognition_rejects_captured_protocol_echo_regressions(self):
        captured = (
            (
                "(9/12/2023 PRE-OP + LAPAROSCOPIC SIGMOIDECTOMY)",
                "p2-0",
            ),
            ("TOTAL FEES DUÉ TO DOCTOR/PRIVATÉ NURSE", "p2-0"),
            ("25 DEQ 2Q25", "p6-0"),
            ("Blaine Bai", "p6-0"),
            ("Room No.", "p6-0"),
            (
                "(9/12/2O23 PRE-OP + LAPAROSCOP1Q SIGMQIDECTOMYD",
                "p6-0",
            ),
            ("FOQD/DRINKS", "p12-bbox-0"),
        )
        settings = FusionSettings(bbox_recognition_enabled=True)

        for ocr_text, vlm_text in captured:
            with self.subTest(ocr_text=ocr_text, vlm_text=vlm_text):
                line = collect_text_lines(
                    middle(ocr_text)["pdf_info"][0],
                    0,
                )[0]
                selected = select_bbox_recognition_candidate(
                    line,
                    vlm_text,
                    settings,
                )
                self.assertEqual(selected[0], "ocr")
                self.assertEqual(selected[1], "bbox_protocol_id_echo")

        def recognize(_page, _size, candidates):
            return {
                "items": [
                    {"id": candidate["id"], "text": "p6-0"}
                    for candidate in candidates
                ]
            }

        _fused, report = fuse_middle_json(
            middle("Blaine Bai"),
            middle("Blaine Bai"),
            settings,
            bbox_recognizer=recognize,
        )
        self.assertEqual(
            report["counts"]["bbox_recognition_protocol_echoes"],
            1,
        )
        self.assertEqual(report["counts"]["bbox_recognition_vlm_selected"], 0)

    def test_bbox_recognition_batch_guard_rejects_isolated_plausible_output(self):
        lines = [
            collect_text_lines(middle(text)["pdf_info"][0], 0)[0]
            for text in ("B1aine Bai", "Room No.", "Date Discharged")
        ]

        def recognize(_page, _size, candidates):
            texts = ("Blaine Bai", "p0-0", "}{")
            return {
                "items": [
                    {"id": candidate["id"], "text": text}
                    for candidate, text in zip(candidates, texts)
                ],
                "batches": [
                    {
                        "status": "ok",
                        "ids": [candidate["id"] for candidate in candidates],
                    }
                ],
            }

        stats, decisions, batches = apply_bbox_recognition(
            0,
            [200, 300],
            lines,
            FusionSettings(bbox_recognition_enabled=True),
            recognize,
        )

        self.assertEqual(stats["vlm_selected"], 0)
        self.assertEqual(stats["batch_quality_fallbacks"], 3)
        self.assertTrue(batches[0]["quality_guard_triggered"])
        self.assertEqual(batches[0]["quality_acceptable_responses"], 1)
        self.assertTrue(
            all(
                decision["reason"] == "recognizer_batch_quality_guard"
                for decision in decisions
            )
        )
        self.assertEqual(decisions[0]["candidate_reason"], "bbox_conditioned_vlm")

    def test_empty_ocr_bbox_requires_healthy_batch_context(self):
        settings = FusionSettings(bbox_recognition_enabled=True)
        empty_line = collect_text_lines(middle("")["pdf_info"][0], 0)[0]

        def recognize_isolated(_page, _size, candidates):
            return {
                "items": [
                    {"id": candidates[0]["id"], "text": "Missing Value"}
                ],
                "batches": [{"status": "ok", "ids": [candidates[0]["id"]]}],
            }

        isolated_stats, isolated_decisions, _batches = apply_bbox_recognition(
            0,
            [200, 300],
            [empty_line],
            settings,
            recognize_isolated,
        )

        self.assertEqual(isolated_stats["vlm_selected"], 0)
        self.assertEqual(isolated_stats["empty_ocr_context_fallbacks"], 1)
        self.assertEqual(
            isolated_decisions[0]["reason"],
            "empty_ocr_context_guard",
        )

        lines = [
            collect_text_lines(middle(text)["pdf_info"][0], 0)[0]
            for text in ("", "B1aine", "R0om")
        ]

        def recognize_batch(_page, _size, candidates):
            replacements = {
                "": "Missing Value",
                "B1aine": "Blaine",
                "R0om": "Room",
            }
            return {
                "items": [
                    {
                        "id": candidate["id"],
                        "text": replacements[candidate["ocr_text"]],
                    }
                    for candidate in candidates
                ],
                "batches": [
                    {
                        "status": "ok",
                        "ids": [candidate["id"] for candidate in candidates],
                    }
                ],
            }

        stats, decisions, batches = apply_bbox_recognition(
            0,
            [200, 300],
            lines,
            settings,
            recognize_batch,
        )

        recovered = next(decision for decision in decisions if not decision["ocr_text"])
        self.assertEqual(stats["vlm_selected"], 3)
        self.assertEqual(stats["empty_ocr_recoveries"], 1)
        self.assertEqual(stats["empty_ocr_context_fallbacks"], 0)
        self.assertEqual(recovered["selected_text"], "Missing Value")
        self.assertEqual(recovered["reason"], "empty_ocr_vlm_recovery")
        self.assertTrue(batches[0]["quality_guard_evaluated"])
        self.assertNotIn("quality_guard_triggered", batches[0])

    def test_empty_ocr_candidate_rejects_rules_formula_echo_and_overflow(self):
        settings = FusionSettings(bbox_recognition_enabled=True)
        line = collect_text_lines(
            middle("", bbox=(10, 10, 110, 20))["pdf_info"][0],
            0,
        )[0]
        line.block_type = "table_ocr"

        separator = select_bbox_recognition_candidate(
            line,
            "---=---=---",
            settings,
        )
        formula = select_bbox_recognition_candidate(
            line,
            r"(1) \because {AD} = \frac{1}{2}{AB}",
            settings,
        )
        overflow = select_bbox_recognition_candidate(
            line,
            "A" * 100,
            settings,
        )
        valid = select_bbox_recognition_candidate(
            line,
            "Missing Value",
            settings,
        )
        placeholder = select_bbox_recognition_candidate(
            line,
            "[Non-Text]",
            settings,
        )

        self.assertEqual(separator[1], "empty_ocr_non_text_candidate")
        self.assertEqual(formula[1], "empty_ocr_formula_guard")
        self.assertEqual(overflow[1], "empty_ocr_geometry_capacity_guard")
        self.assertEqual(valid[:2], ("vlm", "empty_ocr_vlm_recovery"))
        self.assertEqual(placeholder[1], "empty_ocr_placeholder_guard")

    def test_table_collector_keeps_empty_content_bbox_for_recognition(self):
        page = structured_middle(
            "table",
            html="<table><tr><td></td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 40],
                    "content_spans": [
                        {"bbox": [20, 18, 170, 32], "text": ""}
                    ],
                    "text": "",
                    "row_start": 0,
                    "row_end": 0,
                    "col_start": 0,
                    "col_end": 0,
                }
            ],
        )["pdf_info"][0]

        lines = collect_table_ocr_lines(page, 0)
        manifest, _by_id = build_bbox_recognition_manifest(0, lines)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "")
        self.assertEqual(lines[0].bbox, (20.0, 18.0, 170.0, 32.0))
        self.assertEqual(manifest[0]["ocr_text"], "")
        self.assertEqual(manifest[0]["contexts"]["target"], [20.0, 18.0, 170.0, 32.0])

    def test_healthy_batch_recovers_empty_table_cell_without_geometry_change(self):
        original_html = (
            "<table><tr><td></td><td>B1aine</td><td>R0om</td></tr></table>"
        )
        cells = [
            {
                "bbox": [10 + index * 60, 10, 70 + index * 60, 40],
                "content_spans": [
                    {
                        "bbox": [15 + index * 60, 18, 65 + index * 60, 32],
                        "text": text,
                    }
                ],
                "text": text,
                "row_start": 0,
                "row_end": 0,
                "col_start": index,
                "col_end": index,
            }
            for index, text in enumerate(("", "B1aine", "R0om"))
        ]
        page = structured_middle(
            "table",
            html=original_html,
            table_cells=cells,
        )["pdf_info"][0]
        original_bboxes = [
            (list(cell["bbox"]), list(cell["content_spans"][0]["bbox"]))
            for cell in cells
        ]
        lines = collect_table_ocr_lines(page, 0)

        def recognize(_page, _size, candidates):
            replacements = {
                "": "Missing Value",
                "B1aine": "Blaine",
                "R0om": "Room",
            }
            return {
                "items": [
                    {
                        "id": candidate["id"],
                        "text": replacements[candidate["ocr_text"]],
                    }
                    for candidate in candidates
                ],
                "batches": [
                    {
                        "status": "ok",
                        "ids": [candidate["id"] for candidate in candidates],
                    }
                ],
            }

        stats, _decisions, _batches = apply_bbox_recognition(
            0,
            [200, 300],
            lines,
            FusionSettings(bbox_recognition_enabled=True),
            recognize,
        )
        rebuilt, rejected = synchronize_recognized_table_html(page, lines, 0)

        span = page["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(stats["empty_ocr_recoveries"], 1)
        self.assertEqual(rebuilt, 1)
        self.assertEqual(rejected, 0)
        self.assertIn("<td>Missing Value</td>", span["html"])
        self.assertIn("<td>Blaine</td>", span["html"])
        self.assertIn("<td>Room</td>", span["html"])
        self.assertEqual(
            parse_table_html(span["html"]).structure_signature,
            parse_table_html(original_html).structure_signature,
        )
        self.assertEqual(
            [
                (list(cell["bbox"]), list(cell["content_spans"][0]["bbox"]))
                for cell in span["table_cells"]
            ],
            original_bboxes,
        )

    def test_table_recognition_manifest_contains_multiscale_context(self):
        cells = [
            {
                "bbox": [10, 10, 100, 40],
                "content_spans": [{"bbox": [20, 18, 70, 32], "text": "Date"}],
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [100, 10, 190, 40],
                "content_spans": [
                    {"bbox": [110, 18, 170, 32], "text": "25 DEQ 2Q25"}
                ],
                "row_start": 0,
                "row_end": 0,
                "col_start": 1,
                "col_end": 1,
            },
        ]
        page = structured_middle(
            "table",
            html="<table><tr><td>Date</td><td>25 DEQ 2Q25</td></tr></table>",
            table_cells=cells,
        )["pdf_info"][0]
        lines = collect_table_ocr_lines(page, 0)

        manifest, _by_id = build_bbox_recognition_manifest(0, lines)

        value = next(item for item in manifest if item["ocr_text"] == "25 DEQ 2Q25")
        self.assertEqual(value["bbox"], [110.0, 18.0, 170.0, 32.0])
        self.assertEqual(value["contexts"]["cell"], [100.0, 10.0, 190.0, 40.0])
        self.assertEqual(value["contexts"]["row"], [20.0, 18.0, 170.0, 32.0])
        self.assertEqual(value["contexts"]["table"], [10.0, 10.0, 190.0, 80.0])

    def test_recognized_cell_updates_pipeline_html_without_changing_bbox_or_grid(self):
        cells = [
            {
                "bbox": [10, 10, 100, 40],
                "content_spans": [{"bbox": [20, 18, 70, 32], "text": "Date"}],
                "text": "Date",
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 0,
            },
            {
                "bbox": [100, 10, 190, 40],
                "content_spans": [
                    {"bbox": [110, 18, 170, 32], "text": "25 DEQ 2Q25"}
                ],
                "text": "25 DEQ 2Q25",
                "row_start": 0,
                "row_end": 0,
                "col_start": 1,
                "col_end": 1,
            },
        ]
        original_html = "<table><tr><td>Date</td><td>25 DEQ 2Q25</td></tr></table>"
        page = structured_middle(
            "table",
            html=original_html,
            table_cells=cells,
        )["pdf_info"][0]
        lines = collect_table_ocr_lines(page, 0)

        def recognize(_page, _size, candidates):
            return {
                "items": [
                    {
                        "id": item["id"],
                        "text": "25 DEC 2023"
                        if item["ocr_text"] == "25 DEQ 2Q25"
                        else item["ocr_text"],
                    }
                    for item in candidates
                ]
            }

        stats, _decisions, _batches = apply_bbox_recognition(
            0,
            [200, 300],
            lines,
            FusionSettings(bbox_recognition_enabled=True),
            recognize,
        )
        rebuilt, rejected = synchronize_recognized_table_html(page, lines, 0)

        span = page["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(stats["vlm_selected"], 1)
        self.assertEqual(rebuilt, 1)
        self.assertEqual(rejected, 0)
        self.assertIn("<td>25 DEC 2023</td>", span["html"])
        self.assertEqual(
            parse_table_html(span["html"]).structure_signature,
            parse_table_html(original_html).structure_signature,
        )
        self.assertEqual(span["table_cells"][1]["bbox"], [100, 10, 190, 40])
        self.assertEqual(
            span["table_cells"][1]["content_spans"][0]["bbox"],
            [110, 18, 170, 32],
        )
        self.assertEqual(
            span["table_cells"][1]["content_spans"][0]["text"],
            "25 DEC 2023",
        )

    def test_missing_ocr_inside_table_is_not_recovered_as_paragraph(self):
        hybrid = structured_middle(
            "table",
            html="<table><tr><td>Value</td></tr></table>",
            table_cells=[
                {
                    "bbox": [10, 10, 190, 80],
                    "content_spans": [
                        {"bbox": [20, 20, 180, 60], "text": "Value"}
                    ],
                    "text": "Value",
                }
            ],
        )
        ocr = middle("Value", 0.99, bbox=(20, 20, 180, 60))

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        self.assertEqual(len(fused["pdf_info"][0]["preproc_blocks"]), 1)
        self.assertEqual(report["counts"]["missing_ocr_candidates"], 0)

    def test_unreliable_table_content_spans_are_recovered_as_form_fields(self):
        html = "<table><tr><td>Policy No.</td><td>A123</td></tr></table>"
        cells = [
            {
                "bbox": [10, 10, 150, 70],
                "content_spans": [
                    {"bbox": [20, 20, 80, 35], "text": "Policy No."}
                ],
                "text": "Policy No.",
            },
            {
                "bbox": [50, 10, 190, 70],
                "content_spans": [
                    {"bbox": [110, 20, 160, 35], "text": "A123"}
                ],
                "text": "A123",
            },
            {
                "bbox": [30, 10, 170, 70],
                "content_spans": [
                    {"bbox": [85, 45, 105, 60], "text": "Extra"}
                ],
                "text": "Extra",
            },
        ]
        hybrid = structured_middle(
            "table",
            html=(
                "<table><tr><td>Existing A</td><td>Existing B</td>"
                "<td>Existing C</td></tr></table>"
            ),
        )
        ocr = structured_middle("table", html=html, table_cells=cells)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        blocks = fused["pdf_info"][0]["preproc_blocks"]
        recovered = [
            block
            for block in blocks
            if block.get("fusion_recovery_type") in {"form_field", "table_ocr"}
        ]
        self.assertEqual(
            [block["lines"][0]["spans"][0]["content"] for block in recovered],
            ["Policy No.", "A123", "Extra"],
        )
        self.assertTrue(all(block["type"] == "text" for block in recovered))
        self.assertEqual(report["counts"]["unreliable_table_targets"], 1)
        self.assertEqual(report["counts"]["unreliable_table_ocr_lines"], 3)
        self.assertEqual(report["counts"]["unreliable_table_lines_recovered"], 3)
        self.assertEqual(report["counts"]["key_value_pairs"], 1)
        self.assertEqual(report["key_value_pairs"][0]["key"], "Policy No.")
        self.assertEqual(report["key_value_pairs"][0]["value"], "A123")
        self.assertEqual(
            fused["pdf_info"][0]["form_fields"][0]["value_bbox"],
            [110.0, 20.0, 160.0, 35.0],
        )
        self.assertEqual(report["counts"]["ocr_spatial_coverage"], 1.0)
        self.assertEqual(report["coverage"][0]["uncovered"], 0)

    def test_unscored_unreliable_table_recovery_can_be_disabled(self):
        html = "<table><tr><td>Value</td></tr></table>"
        cells = [
            {
                "bbox": [10, 10, 100, 70],
                "content_spans": [{"bbox": [20, 20, 70, 35], "text": "Value"}],
            },
            {
                "bbox": [40, 10, 190, 70],
                "content_spans": [{"bbox": [120, 20, 170, 35], "text": "Other"}],
            },
            {
                "bbox": [30, 10, 170, 70],
                "content_spans": [{"bbox": [85, 45, 105, 60], "text": "Extra"}],
            },
        ]
        hybrid = structured_middle("table", html=html)
        ocr = structured_middle("table", html=html, table_cells=cells)

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(unreliable_table_allow_unscored_ocr=False),
        )

        self.assertEqual(len(fused["pdf_info"][0]["preproc_blocks"]), 1)
        self.assertEqual(report["counts"]["unreliable_table_ocr_lines"], 3)
        self.assertEqual(report["counts"]["missing_ocr_candidates"], 0)

    def test_unreliable_table_ocr_already_in_html_is_not_duplicated(self):
        html = "<table><tr><td>Policy No.</td><td>A123</td><td>Extra</td></tr></table>"
        cells = [
            {
                "bbox": [10, 10, 150, 70],
                "content_spans": [
                    {"bbox": [20, 20, 80, 35], "text": "Policy No."}
                ],
            },
            {
                "bbox": [50, 10, 190, 70],
                "content_spans": [{"bbox": [110, 20, 160, 35], "text": "A123"}],
            },
            {
                "bbox": [30, 10, 170, 70],
                "content_spans": [{"bbox": [85, 45, 105, 60], "text": "Extra"}],
            },
        ]

        fused, report = fuse_middle_json(
            structured_middle("table", html=html),
            structured_middle("table", html=html, table_cells=cells),
            FusionSettings(),
        )

        recovered = [
            block
            for block in fused["pdf_info"][0]["preproc_blocks"]
            if block.get("fusion_source") == "ocr_recovered"
        ]
        self.assertEqual(recovered, [])
        self.assertEqual(
            report["counts"]["unreliable_table_lines_structurally_covered"],
            3,
        )
        self.assertEqual(report["coverage"][0]["structured"], 3)
        self.assertEqual(report["counts"]["ocr_spatial_coverage"], 1.0)

    def test_unreliable_table_recovery_switch_restores_table_exclusion(self):
        html = "<table><tr><td>Value</td></tr></table>"
        cells = [
            {
                "bbox": [10, 10, 100, 70],
                "content_spans": [
                    {"bbox": [20, 20, 70, 35], "text": "Value", "score": 0.99}
                ],
            },
            {
                "bbox": [40, 10, 190, 70],
                "content_spans": [
                    {"bbox": [120, 20, 170, 35], "text": "Other", "score": 0.99}
                ],
            },
            {
                "bbox": [30, 10, 170, 70],
                "content_spans": [
                    {"bbox": [85, 45, 105, 60], "text": "Extra", "score": 0.99}
                ],
            },
        ]
        hybrid = structured_middle("table", html=html)
        ocr = structured_middle("table", html=html, table_cells=cells)

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(unreliable_table_recovery_enabled=False),
        )

        self.assertEqual(len(fused["pdf_info"][0]["preproc_blocks"]), 1)
        self.assertEqual(report["counts"]["unreliable_table_ocr_lines"], 0)
        self.assertEqual(report["counts"]["missing_ocr_candidates"], 0)

    def test_existing_hybrid_text_covers_unreliable_table_ocr_without_duplicate(self):
        html = "<table><tr><td>A123</td></tr></table>"
        cells = [
            {
                "bbox": [10, 10, 100, 70],
                "content_spans": [{"bbox": [20, 20, 70, 35], "text": "A123"}],
            },
            {
                "bbox": [40, 10, 190, 70],
                "content_spans": [{"bbox": [120, 20, 170, 35], "text": "Other"}],
            },
            {
                "bbox": [30, 10, 170, 70],
                "content_spans": [{"bbox": [85, 45, 105, 60], "text": "Extra"}],
            },
        ]
        hybrid = middle("A123", bbox=(20, 20, 70, 35))
        hybrid["pdf_info"][0]["preproc_blocks"].append(
            structured_middle("table", html=html)["pdf_info"][0]["preproc_blocks"][0]
        )
        ocr = structured_middle("table", html=html, table_cells=cells)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        contents = [
            block["lines"][0]["spans"][0].get("content")
            for block in fused["pdf_info"][0]["preproc_blocks"]
            if block.get("type") == "text"
        ]
        self.assertEqual(contents.count("A123"), 1)
        self.assertEqual(report["coverage"][0]["matched"], 1)
        self.assertEqual(report["coverage"][0]["recovered"], 2)

    def test_table_quality_and_recall_collectors_report_unreliable_geometry(self):
        cells = [
            {
                "bbox": [10, 10, 100, 70],
                "content_spans": [{"bbox": [20, 20, 70, 35], "text": "One"}],
            },
            {
                "bbox": [40, 10, 190, 70],
                "content_spans": [{"bbox": [120, 20, 170, 35], "text": "Two"}],
            },
            {
                "bbox": [30, 10, 170, 70],
                "content_spans": [{"bbox": [85, 45, 105, 60], "text": "Three"}],
            },
        ]
        page = structured_middle(
            "table",
            html="<table><tr><td>One</td><td>Two</td></tr></table>",
            table_cells=cells,
        )["pdf_info"][0]

        quality = collect_table_geometry_quality(page, 0)
        lines = collect_unreliable_table_ocr_lines(page, 0)

        self.assertFalse(quality[0].quality.reliable)
        self.assertIn("excessive_cell_overlap", quality[0].quality.reasons)
        self.assertEqual([line.text for line in lines], ["One", "Two", "Three"])

    def test_unrelated_verifier_output_is_rejected(self):
        hybrid = middle("invoice 1008")
        ocr = middle("invoice 100B", 0.98)

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(consensus_similarity=0.99, candidate_guard_similarity=0.8),
            verifier=lambda *_args: "unrelated hallucination",
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "invoice 1008")
        self.assertEqual(report["counts"]["verifier_rejections"], 1)

    def test_verifier_error_keeps_hybrid_and_continues(self):
        hybrid = middle("invoice 1008")
        ocr = middle("invoice 100B", 0.98)

        def failing_verifier(*_args):
            raise TimeoutError("unavailable")

        fused, report = fuse_middle_json(
            hybrid,
            ocr,
            FusionSettings(consensus_similarity=0.99),
            verifier=failing_verifier,
        )

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "invoice 1008")
        self.assertEqual(report["counts"]["verifier_errors"], 1)
        self.assertEqual(report["decisions"][0]["reason"], "verifier_error")

    def test_inline_equation_line_is_not_fused(self):
        hybrid = middle("x+y", span_type="inline_equation")
        ocr = middle("xy", 0.99)

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        self.assertEqual(report["counts"]["targets"], 0)
        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "x+y")

    def test_parses_json_and_fenced_verifier_responses(self):
        self.assertEqual(_parse_verifier_text('{"text":"abc"}'), "abc")
        self.assertEqual(_parse_verifier_text('```json\n{"text":"xyz"}\n```'), "xyz")
        self.assertIsNone(_parse_verifier_text("not json"))


if __name__ == "__main__":
    unittest.main()

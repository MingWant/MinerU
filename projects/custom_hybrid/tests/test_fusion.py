import sys
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.fusion import (
    FusionSettings,
    OpenAIVisionVerifier,
    _parse_verifier_text,
    assign_ocr_lines,
    collect_text_lines,
    fuse_middle_json,
    recover_table_cell_geometry,
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

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                body = json.dumps({"data": [{"id": "vision-model"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.server.received.append(payload)
                prompt = payload["messages"][0]["content"][0]["text"]
                if '"hybrid" or "ocr"' in prompt:
                    result = '{"source":"ocr"}'
                else:
                    result = '{"source":"pipeline"}'
                body = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": result}}
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.received = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        verifier = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "page.png"
                Image.new("RGB", (200, 100), "white").save(image_path)
                verifier = OpenAIVisionVerifier(
                    f"http://127.0.0.1:{server.server_address[1]}",
                    image_path,
                    {
                        "model": None,
                        "render_scale": 1.0,
                        "padding_ratio": 0.0,
                        "timeout_seconds": 5,
                    },
                )

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

            self.assertEqual(result, "ocr")
            self.assertEqual(selected, "pipeline")
            self.assertEqual(selected_cell, "pipeline")
            self.assertEqual(server.received[0]["model"], "vision-model")
            text_content = server.received[0]["messages"][0]["content"]
            self.assertIn("untrusted", text_content[0]["text"])
            self.assertTrue(
                text_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            )
            table_prompt = server.received[1]["messages"][0]["content"][0]["text"]
            self.assertIn("embedded-data-removed", table_prompt)
            self.assertNotIn("SECRET_TABLE_DATA", table_prompt)
            cell_content = server.received[2]["messages"][0]["content"]
            self.assertIn("whole_table, target_row, target_column, target_cell", cell_content[0]["text"])
            self.assertEqual(len(cell_content), 5)
        finally:
            if verifier is not None:
                verifier.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

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

    def test_missing_ocr_inside_table_is_not_recovered_as_paragraph(self):
        hybrid = structured_middle(
            "table",
            html="<table><tr><td>Value</td></tr></table>",
        )
        ocr = middle("Value", 0.99, bbox=(20, 20, 180, 60))

        fused, report = fuse_middle_json(hybrid, ocr, FusionSettings())

        self.assertEqual(len(fused["pdf_info"][0]["preproc_blocks"]), 1)
        self.assertEqual(report["counts"]["missing_ocr_candidates"], 0)

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

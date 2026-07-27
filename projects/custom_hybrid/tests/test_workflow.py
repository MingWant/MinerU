import importlib.util
import json
import random
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import httpx

PDF_RENDERING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("pypdf", "reportlab")
)

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.workflow import (
    WorkflowConfigError,
    apply_generation_policy,
    build_mineru_command,
    build_pipeline_command,
    build_sweep_variants,
    build_vllm_server_command,
    evaluate_paths,
    evaluate_benchmark_runs,
    evaluate_table_texts,
    evaluate_texts,
    extract_request_text,
    fuse_output_trees,
    load_config,
    levenshtein_distance,
    prepare_parameter_proxy_config,
    regenerate_fused_visualizations,
    resolve_upstream_max_model_len,
    run_extract,
    run_doctor,
    _optional_bearer_headers,
    _generate_page_sorting_outputs,
    _generate_semantic_markdown_outputs,
    _generate_fused_visualizations,
    _index_input_documents,
    _resolve_visualization_pdf,
    _start_parameter_proxy,
    _stop_parameter_proxy,
    _visualization_renderer_is_current,
)
from mineru.utils.draw_bbox import BBOX_RENDERER_VERSION, _form_table_overlay_bboxes


class WorkflowTests(unittest.TestCase):
    def test_span_overlay_treats_detected_forms_and_demoted_regions_as_tables(self):
        regions, cells = _form_table_overlay_bboxes(
            {
                "form_regions": [{"bbox": [10, 10, 190, 200]}],
                "form_cells": [{"bbox": [10, 10, 190, 60]}],
                "demoted_narrative_tables": [
                    {
                        "bbox": [10, 210, 190, 290],
                        "cells": [{"bbox": [10, 210, 190, 250]}],
                    }
                ],
            }
        )

        self.assertEqual(
            regions,
            [[10.0, 10.0, 190.0, 200.0], [10.0, 210.0, 190.0, 290.0]],
        )
        self.assertEqual(
            cells,
            [[10.0, 10.0, 190.0, 60.0], [10.0, 210.0, 190.0, 250.0]],
        )

    def test_span_overlay_hides_overlapping_demoted_pseudo_table_cells(self):
        regions, cells = _form_table_overlay_bboxes(
            {
                "demoted_narrative_tables": [
                    {
                        "bbox": [10, 10, 190, 290],
                        "cells": [
                            {"bbox": [15, 20, 170, 100]},
                            {"bbox": [16, 50, 171, 130]},
                            {"bbox": [17, 80, 172, 160]},
                            {"bbox": [18, 110, 173, 190]},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(regions, [[10.0, 10.0, 190.0, 290.0]])
        self.assertEqual(cells, [])

    def test_visualization_renderer_version_invalidates_cached_preview(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            marker_path = parse_dir / "sample_visualization.json"
            marker = {
                "bbox_renderer_version": 10,
                "form_detector_version": 1,
                "form_segmenter_version": 4,
            }
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            self.assertFalse(
                _visualization_renderer_is_current(parse_dir, "sample")
            )
            marker["bbox_renderer_version"] = BBOX_RENDERER_VERSION
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertTrue(
                _visualization_renderer_is_current(parse_dir, "sample")
            )

    def test_semantic_markdown_replaces_primary_and_preserves_native_ab(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            middle_path = parse_dir / "sample_middle.json"
            primary_path = parse_dir / "sample.md"
            middle_path.write_text(
                json.dumps(
                    {
                        "pdf_info": [
                            {
                                "page_size": [200, 300],
                                "preproc_blocks": [
                                    {
                                        "type": "text",
                                        "bbox": [10, 10, 100, 30],
                                        "lines": [
                                            {
                                                "bbox": [10, 10, 100, 30],
                                                "spans": [
                                                    {
                                                        "type": "text",
                                                        "bbox": [10, 10, 100, 30],
                                                        "content": "Hello",
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            primary_path.write_text("native markdown", encoding="utf-8")

            generated = _generate_semantic_markdown_outputs(
                parse_dir,
                "sample",
                {
                    "enabled": True,
                    "replace_primary": True,
                    "preserve_native": True,
                },
            )

            native_path = parse_dir / "sample_native.md"
            report_path = parse_dir / "sample_semantic_report.json"
            self.assertEqual(generated, (native_path, report_path))
            self.assertEqual(
                native_path.read_text(encoding="utf-8"),
                "native markdown",
            )
            semantic = primary_path.read_text(encoding="utf-8")
            self.assertIn("semantic-markdown-v6", semantic)
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["semantic_markdown_version"], 6)
            self.assertEqual(report["pages_emitted"], 1)
            self.assertIn("Hello", semantic)

    def test_page_sorting_outputs_are_additive_report_only_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            middle_path = parse_dir / "sample_middle.json"
            middle_payload = {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "page_size": [200, 300],
                        "discarded_blocks": [
                            {
                                "type": "footer",
                                "bbox": [40, 275, 160, 290],
                                "lines": [
                                    {
                                        "bbox": [40, 275, 160, 290],
                                        "spans": [
                                            {
                                                "type": "text",
                                                "content": "Sample Form P.1/1",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
            original = json.dumps(middle_payload)
            middle_path.write_text(original, encoding="utf-8")

            generated = _generate_page_sorting_outputs(
                parse_dir,
                "sample",
                {
                    "enabled": True,
                    "mode": "report_only",
                    "include_semantic_diagnostics": True,
                },
            )

            self.assertEqual(
                generated,
                (
                    parse_dir / "sample_sorting_manifest.json",
                    parse_dir / "sample_sorting_report.json",
                ),
            )
            self.assertEqual(middle_path.read_text(encoding="utf-8"), original)
            report = json.loads(generated[1].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["report_only"])

    def test_page_sorting_error_does_not_replace_or_mutate_fused_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            middle_path = parse_dir / "sample_middle.json"
            original = json.dumps({"pdf_info": "invalid"})
            middle_path.write_text(original, encoding="utf-8")

            generated = _generate_page_sorting_outputs(
                parse_dir,
                "sample",
                {"enabled": True, "mode": "report_only"},
            )

            self.assertEqual(generated, (parse_dir / "sample_sorting_report.json",))
            self.assertEqual(middle_path.read_text(encoding="utf-8"), original)
            report = json.loads(generated[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["can_auto_sort"])

    def test_bearer_header_requires_explicit_environment_setting(self):
        with mock.patch.dict(
            "os.environ",
            {"VLLM_API_KEY": "local-secret"},
            clear=False,
        ):
            self.assertEqual(_optional_bearer_headers({}), {})
            self.assertEqual(
                _optional_bearer_headers({"api_key_env": None}),
                {},
            )
            self.assertEqual(
                _optional_bearer_headers({"api_key_env": "VLLM_API_KEY"}),
                {"authorization": "Bearer local-secret"},
            )

    def test_visualization_source_falls_back_to_fused_origin_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            origin = parse_dir / "renamed_origin.pdf"
            origin.write_bytes(b"pdf")

            resolved = _resolve_visualization_pdf(
                parse_dir,
                "renamed",
                source_document=None,
            )

            self.assertEqual(resolved, origin)

    def test_visualization_and_bbox_source_prefers_pipeline_origin_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            source = parse_dir / "uploaded.pdf"
            source.write_bytes(b"source")
            origin = parse_dir / "renamed_origin.pdf"
            origin.write_bytes(b"normalized")

            resolved = _resolve_visualization_pdf(
                parse_dir,
                "renamed",
                source,
            )

            self.assertEqual(resolved, origin)

    def test_visualization_refresh_recovers_cells_from_ocr_middle(self):
        html = "<table><tr><td>Key</td></tr></table>"

        def table_middle(table_cells=None):
            span = {
                "type": "table",
                "bbox": [10, 10, 190, 80],
                "html": html,
            }
            if table_cells is not None:
                span["table_cells"] = table_cells
            return {
                "pdf_info": [
                    {
                        "page_size": [200, 300],
                        "preproc_blocks": [
                            {
                                "type": "table_body",
                                "lines": [
                                    {
                                        "bbox": [10, 10, 190, 80],
                                        "spans": [span],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

        fused_middle = table_middle()
        ocr_middle = table_middle(
            [
                {
                    "bbox": [10, 10, 190, 80],
                    "content_bbox": [20, 20, 100, 40],
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fused_dir = root / "output" / "fused" / "sample"
            ocr_dir = root / "output" / "ocr" / "sample"
            input_dir = root / "input"
            fused_dir.mkdir(parents=True)
            ocr_dir.mkdir(parents=True)
            input_dir.mkdir()
            (input_dir / "sample.pdf").write_bytes(b"pdf")
            middle_path = fused_dir / "sample_middle.json"
            middle_path.write_text(json.dumps(fused_middle), encoding="utf-8")
            (ocr_dir / "sample_middle.json").write_text(
                json.dumps(ocr_middle),
                encoding="utf-8",
            )
            (fused_dir / "sample_span.pdf").write_bytes(b"old")

            def render(parse_dir, stem, source_document):
                refreshed = json.loads(middle_path.read_text(encoding="utf-8"))
                span = refreshed["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
                self.assertEqual(len(span["table_cells"]), 1)
                output = parse_dir / f"{stem}_span.pdf"
                output.write_bytes(b"new")
                return (output,)

            with mock.patch(
                "projects.custom_hybrid.workflow._generate_fused_visualizations",
                side_effect=render,
            ) as regenerate:
                generated = regenerate_fused_visualizations(
                    root / "output" / "fused",
                    input_dir,
                )

            self.assertEqual((fused_dir / "sample_span.pdf").read_bytes(), b"new")
            self.assertIn((fused_dir / "sample_span.pdf").resolve(), generated)
            regenerate.assert_called_once()

    @unittest.skipUnless(PDF_RENDERING_AVAILABLE, "PDF rendering dependencies missing")
    def test_fused_visualization_uses_origin_pdf_fallback_and_creates_span(self):
        from pypdf import PdfReader, PdfWriter

        with tempfile.TemporaryDirectory() as temp_dir:
            parse_dir = Path(temp_dir)
            middle = {
                "pdf_info": [
                    {
                        "discarded_blocks": [],
                        "preproc_blocks": [],
                    }
                ]
            }
            (parse_dir / "renamed_middle.json").write_text(
                json.dumps(middle),
                encoding="utf-8",
            )
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=300)
            with (parse_dir / "renamed_origin.pdf").open("wb") as stream:
                writer.write(stream)

            generated = _generate_fused_visualizations(
                parse_dir,
                "renamed",
                source_document=None,
            )

            span_path = parse_dir / "renamed_span.pdf"
            self.assertIn(span_path, generated)
            self.assertTrue(span_path.is_file())
            self.assertEqual(len(PdfReader(str(span_path)).pages), 1)
            marker = json.loads(
                (parse_dir / "renamed_visualization.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(marker["bbox_renderer_version"], BBOX_RENDERER_VERSION)
            self.assertEqual(marker["form_detector_version"], 1)
            self.assertEqual(marker["form_segmenter_version"], 4)
            self.assertTrue((parse_dir / "renamed_forms.pdf").is_file())
            self.assertTrue((parse_dir / "renamed_form_cells.pdf").is_file())
            self.assertFalse(list(parse_dir.glob(".*-span.pdf")))

    def test_real_proxy_rewrites_openai_request_and_writes_safe_audit(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.server.received.append(payload)
                body = json.dumps({"received": payload}).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        upstream.daemon_threads = True
        upstream.received = []
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        proxy_server = proxy_thread = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                config = load_config(
                    Path(__file__).parents[1] / "workflow.example.json"
                )
                config["vllm"]["upstream_url"] = (
                    f"http://127.0.0.1:{upstream.server_address[1]}"
                )
                config["vllm"]["proxy"]["port"] = 0
                config["vllm"]["audit_log"] = str(root / "audit.jsonl")
                config["vllm"]["audit_prompt_preview_chars"] = 200
                proxy_server, proxy_thread, proxy_url = _start_parameter_proxy(config)
                request_payload = {
                    "model": "mineru",
                    "temperature": 0.7,
                    "top_k": 1,
                    "presence_penalty": 0.4,
                    "frequency_penalty": 0.2,
                    "vllm_xargs": {"bad_words": ["stale"]},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract text exactly"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,SECRET_IMAGE"
                                    },
                                },
                            ],
                        }
                    ],
                }

                response = httpx.post(
                    proxy_url + "/v1/chat/completions",
                    json=request_payload,
                    timeout=5,
                )
                audit = json.loads(
                    (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0]
                )

            self.assertEqual(response.status_code, 200)
            forwarded = upstream.received[0]
            self.assertEqual(forwarded["temperature"], 0.0)
            self.assertEqual(forwarded["top_p"], 1.0)
            self.assertEqual(forwarded["seed"], 42)
            self.assertEqual(forwarded["max_tokens"], 2048)
            self.assertNotIn("top_k", forwarded)
            self.assertNotIn("presence_penalty", forwarded)
            self.assertNotIn("frequency_penalty", forwarded)
            self.assertNotIn("vllm_xargs", forwarded)
            self.assertEqual(response.json()["received"], forwarded)
            self.assertIn("deterministic-ocr", audit["matched_rules"])
            self.assertEqual(audit["changed_parameters"]["temperature"], 0.0)
            self.assertEqual(
                audit["effective_generation_parameters"],
                {
                    "max_tokens": 2048,
                    "seed": 42,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            )
            self.assertIn("Extract text exactly", audit["prompt_preview"])
            self.assertNotIn("SECRET_IMAGE", audit["prompt_preview"])
        finally:
            if proxy_server is not None and proxy_thread is not None:
                _stop_parameter_proxy(proxy_server, proxy_thread)
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)

    def test_bit_parallel_levenshtein_matches_dynamic_programming(self):
        def dynamic(left, right):
            previous = list(range(len(right) + 1))
            for left_index, left_item in enumerate(left, 1):
                current = [left_index]
                for right_index, right_item in enumerate(right, 1):
                    current.append(
                        min(
                            current[-1] + 1,
                            previous[right_index] + 1,
                            previous[right_index - 1] + (left_item != right_item),
                        )
                    )
                previous = current
            return previous[-1]

        generator = random.Random(42)
        alphabet = "abc中文"
        for _ in range(200):
            left = "".join(generator.choice(alphabet) for _ in range(generator.randrange(20)))
            right = "".join(generator.choice(alphabet) for _ in range(generator.randrange(20)))
            self.assertEqual(levenshtein_distance(left, right), dynamic(left, right))

    def test_generation_policy_applies_defaults_overrides_rules_and_removals(self):
        policy = {
            "defaults": {"temperature": 0.0, "top_p": 1.0},
            "overrides": {"seed": 42},
            "task_overrides": {"temperature": 0.35, "seed": 123},
            "remove": ["min_p"],
            "rules": [
                {
                    "name": "ocr",
                    "match": {"text_regex": "extract text"},
                    "overrides": {"max_tokens": 12000, "temperature": 0.0},
                }
            ],
        }
        original = {
            "model": "mineru",
            "messages": [{"role": "user", "content": "Extract text from this page"}],
            "temperature": 0.2,
            "min_p": 0.1,
        }

        applied = apply_generation_policy("/v1/chat/completions", original, policy)

        self.assertEqual(applied.body["temperature"], 0.35)
        self.assertEqual(applied.body["top_p"], 1.0)
        self.assertEqual(applied.body["seed"], 123)
        self.assertEqual(applied.body["max_tokens"], 12000)
        self.assertNotIn("min_p", applied.body)
        self.assertEqual(applied.matched_rules, ("ocr",))
        self.assertEqual(original["min_p"], 0.1)

    def test_generation_policy_reserves_context_for_prompt_tokens(self):
        policy = {
            "task_overrides": {"max_tokens": 8192},
            "_resolved_max_context_tokens": 8192,
            "context_reserve_tokens": 4096,
        }

        applied = apply_generation_policy(
            "/v1/chat/completions",
            {"model": "mineru", "max_tokens": 8192},
            policy,
        )

        self.assertEqual(applied.body["max_tokens"], 4096)
        self.assertEqual(applied.changed_parameters["max_tokens"], 4096)

    def test_generation_policy_honors_internal_stage_token_cap(self):
        policy = {
            "task_overrides": {
                "max_tokens": 8192,
                "max_completion_tokens": 8192,
            }
        }

        applied = apply_generation_policy(
            "/v1/chat/completions",
            {"model": "mineru", "max_tokens": 2048},
            policy,
            request_max_tokens_cap=512,
            request_protocol="mineru_native",
        )

        self.assertEqual(applied.body["max_tokens"], 512)
        self.assertEqual(applied.body["max_completion_tokens"], 512)
        self.assertEqual(applied.body["temperature"], 0.0)
        self.assertEqual(applied.body["top_p"], 0.01)

    def test_proxy_config_discovers_remote_model_context_length(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        response = mock.Mock()
        response.json.return_value = {
            "data": [
                {"id": "larger", "max_model_len": 32768},
                {"id": "mineru", "max_model_len": 8192},
            ]
        }

        with mock.patch("httpx.get", return_value=response) as get_models:
            resolved = resolve_upstream_max_model_len(config)
            prepared = prepare_parameter_proxy_config(config)

        self.assertEqual(resolved, 8192)
        self.assertEqual(
            prepared["vllm"]["generation"]["_resolved_max_context_tokens"],
            8192,
        )
        self.assertEqual(get_models.call_count, 2)

    def test_prompt_extraction_excludes_data_uri_images(self):
        body = {
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "Read the page"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ]
                }
            ]
        }
        text = extract_request_text(body)
        self.assertIn("Read the page", text)
        self.assertNotIn("base64", text)

    def test_evaluate_identical_markdown_is_perfect(self):
        result = evaluate_texts("# Title\n\n| A | B |", "# Title\n\n| A | B |", {})
        self.assertEqual(result["quality_score"], 1.0)
        self.assertEqual(result["char_error_rate"], 0.0)
        self.assertEqual(result["structure_f1"], 1.0)

    def test_table_metrics_separate_structure_and_cell_text_accuracy(self):
        reference = (
            '<table><tr><th rowspan="2">Item</th><th>Value</th></tr>'
            "<tr><td>100B</td></tr></table>"
        )
        wrong_text = (
            '<table><tr><th rowspan="2">Item</th><th>Value</th></tr>'
            "<tr><td>1008</td></tr></table>"
        )
        wrong_structure = (
            "<table><tr><th>Item</th><th>Value</th></tr>"
            "<tr><td></td><td>100B</td></tr></table>"
        )

        text_scores = evaluate_table_texts(reference, wrong_text)
        structure_scores = evaluate_table_texts(reference, wrong_structure)

        self.assertEqual(text_scores["table_count_f1"], 1.0)
        self.assertEqual(text_scores["table_structure_f1"], 1.0)
        self.assertLess(text_scores["table_cell_text_similarity"], 1.0)
        self.assertLess(text_scores["table_cell_exact_match"], 1.0)
        self.assertLess(structure_scores["table_structure_f1"], 1.0)
        self.assertLess(structure_scores["table_quality_score"], text_scores["table_quality_score"])

    def test_table_metrics_recognize_markdown_pipe_tables(self):
        reference = "| Code | Amount |\n| --- | ---: |\n| A\\|B | 50 |"
        candidate = "| Code | Amount |\n| --- | ---: |\n| A\\|B | 50 |"

        scores = evaluate_table_texts(reference, candidate)

        self.assertEqual(scores["reference_table_count"], 1)
        self.assertEqual(scores["table_structure_f1"], 1.0)
        self.assertEqual(scores["table_cell_text_similarity"], 1.0)

    def test_directory_evaluation_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            (reference / "shared.md").write_text("correct", encoding="utf-8")
            (candidate / "shared.md").write_text("correct", encoding="utf-8")
            (reference / "missing.md").write_text("missing", encoding="utf-8")

            report = evaluate_paths(reference, candidate, {})

        self.assertEqual(report["document_count"], 1)
        self.assertEqual(report["aggregate"]["quality_score"], 1.0)
        self.assertEqual(report["missing_candidate"], ["missing.md"])

    def test_table_aggregate_excludes_documents_without_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            (reference / "plain.md").write_text("plain text", encoding="utf-8")
            (candidate / "plain.md").write_text("plain text", encoding="utf-8")
            reference_table = "<table><tr><td>100B</td></tr></table>"
            candidate_table = "<table><tr><td>1008</td></tr></table>"
            (reference / "table.md").write_text(reference_table, encoding="utf-8")
            (candidate / "table.md").write_text(candidate_table, encoding="utf-8")

            report = evaluate_paths(reference, candidate, {})
            table_only = evaluate_table_texts(reference_table, candidate_table)

        self.assertEqual(report["table_document_count"], 1)
        self.assertEqual(
            report["aggregate"]["table_quality_score"],
            table_only["table_quality_score"],
        )

    def test_config_validation_rejects_invalid_rule_regex(self):
        config = json.loads(
            (Path(__file__).parents[1] / "workflow.example.json").read_text(encoding="utf-8")
        )
        config["vllm"]["generation"]["rules"][0]["match"]["text_regex"] = "["
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "Invalid text_regex"):
                load_config(config_path)

    def test_config_rejects_invalid_form_detection_coverage(self):
        config = json.loads(
            (Path(__file__).parents[1] / "workflow.example.json").read_text(
                encoding="utf-8"
            )
        )
        config["fusion"]["form_detection"][
            "existing_table_coverage_threshold"
        ] = 1.1
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                WorkflowConfigError,
                "existing_table_coverage_threshold",
            ):
                load_config(config_path)

    def test_config_validates_report_only_page_sorting(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        for key, value, message in (
            ("enabled", "yes", "page_sorting.enabled"),
            ("mode", "apply", "page_sorting.mode"),
            (
                "include_semantic_diagnostics",
                "yes",
                "include_semantic_diagnostics",
            ),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                config = json.loads(source.read_text(encoding="utf-8"))
                config["fusion"]["page_sorting"][key] = value
                config_path = Path(temp_dir) / "invalid.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(WorkflowConfigError, message):
                    load_config(config_path)

    def test_config_validates_optional_page_sorting_llm(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        cases = (
            ({"enabled": "yes"}, "llm.enabled"),
            ({"base_url": "not-a-url"}, "llm.base_url"),
            ({"model": ""}, "llm.model"),
            ({"trigger": "sometimes"}, "llm.trigger"),
            ({"top_p": 0}, "llm.top_p"),
            ({"max_tokens": 0}, "llm.max_tokens"),
            (
                {
                    "enabled": True,
                    "base_url": None,
                    "model": None,
                },
                "requires base_url and model",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temp_dir:
                config = json.loads(source.read_text(encoding="utf-8"))
                config["fusion"]["page_sorting"]["llm"].update(overrides)
                config_path = Path(temp_dir) / "invalid.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(WorkflowConfigError, message):
                    load_config(config_path)

    def test_config_requires_hybrid_http_client(self):
        config = json.loads(
            (Path(__file__).parents[1] / "workflow.example.json").read_text(encoding="utf-8")
        )
        config["mineru"]["backend"] = "hybrid-engine"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "hybrid-http-client"):
                load_config(config_path)

    def test_config_rejects_implicit_public_proxy_bind(self):
        config = json.loads(
            (Path(__file__).parents[1] / "workflow.example.json").read_text(encoding="utf-8")
        )
        config["vllm"]["proxy"]["host"] = "0.0.0.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "allow_public_bind"):
                load_config(config_path)

    def test_config_validates_table_cell_thresholds_and_render_scale(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["table_cell_metadata_text_similarity"] = 1.1
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-threshold.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                WorkflowConfigError,
                "table_cell_metadata_text_similarity",
            ):
                load_config(path)

        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["verifier"]["enabled"] = True
        config["fusion"]["verifier"]["table_cell_render_scale"] = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-scale.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "table_cell_render_scale"):
                load_config(path)

    def test_config_validates_page_reconciliation_limits(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["reconciliation"]["max_candidates_per_page"] = -1
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-reconciliation.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                WorkflowConfigError,
                "max_candidates_per_page",
            ):
                load_config(path)

    def test_config_validates_bbox_recognizer_limits(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["recognizer"]["min_length_ratio"] = 3.0
        config["fusion"]["recognizer"]["max_length_ratio"] = 2.0
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-recognizer.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "length ratios"):
                load_config(path)

        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["recognizer"]["max_context_tokens"] = 1024
        config["fusion"]["recognizer"]["context_reserve_tokens"] = 2048
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-recognizer-context.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "context token limit"):
                load_config(path)

        for key, value, message in (
            ("temperature", 2.1, "temperature"),
            ("top_p", 0.0, "top_p"),
            ("seed", 1.5, "seed"),
            ("max_batch_size", 1.5, "max_batch_size"),
            ("context_reserve_tokens", 0, "context_reserve_tokens"),
            ("min_batch_acceptable_ratio", 1.1, "min_batch_acceptable_ratio"),
            ("batch_guard_min_candidates", 0, "batch_guard_min_candidates"),
            (
                "empty_thin_line_max_chars_per_em",
                0,
                "empty_thin_line_max_chars_per_em",
            ),
            ("max_image_limit_retries", -1, "max_image_limit_retries"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                config = json.loads(source.read_text(encoding="utf-8"))
                config["fusion"]["recognizer"][key] = value
                path = Path(temp_dir) / f"invalid-{key}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(WorkflowConfigError, message):
                    load_config(path)

        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["recognizer"]["structured_output_mode"] = "best_effort"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-structured-output-mode.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                WorkflowConfigError,
                "structured_output_mode",
            ):
                load_config(path)

        for key, value in (
            ("base_url", "10.100.0.30:8205"),
            ("model", ""),
            ("api_key_env", 42),
            ("empty_ocr_enabled", "yes"),
            ("script_guard_enabled", "yes"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                config = json.loads(source.read_text(encoding="utf-8"))
                config["fusion"]["recognizer"][key] = value
                path = Path(temp_dir) / f"invalid-{key}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(WorkflowConfigError, key):
                    load_config(path)

    def test_config_validates_bbox_vlm_mode_and_selection_policy(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        cases = (
            (("fusion", "mode"), "unknown", "fusion.mode"),
            (
                ("fusion", "recognizer", "selection_policy"),
                "always_vlm",
                "selection_policy",
            ),
            (
                ("fusion", "recognizer", "vlm_primary_min_quality"),
                1.1,
                "vlm_primary_min_quality",
            ),
            (
                ("fusion", "recovery", "min_confidence"),
                1.1,
                "fusion.recovery.min_confidence",
            ),
            (
                ("fusion", "recovery", "max_tables_per_document"),
                -1,
                "max_tables_per_document",
            ),
            (
                ("fusion", "recovery", "table_orphan_recovery_enabled"),
                "yes",
                "table_orphan_recovery_enabled",
            ),
            (
                ("fusion", "recovery", "table_orphan_min_ink_density"),
                0,
                "table_orphan_min_ink_density",
            ),
            (
                ("fusion", "recovery", "table_orphan_axis_rule_enabled"),
                "yes",
                "table_orphan_axis_rule_enabled",
            ),
            (
                ("fusion", "recovery", "table_orphan_axis_rule_min_length"),
                0,
                "table_orphan_axis_rule_min_length",
            ),
            (
                ("fusion", "recovery", "table_orphan_axis_rule_max_angle"),
                46,
                "table_orphan_axis_rule_max_angle",
            ),
            (
                ("fusion", "recovery", "table_fringe_horizontal_extension"),
                -1,
                "table_fringe_horizontal_extension",
            ),
            (
                ("fusion", "recovery", "table_fringe_separator_enabled"),
                "yes",
                "table_fringe_separator_enabled",
            ),
            (
                (
                    "fusion",
                    "recovery",
                    "page_recovery_rule_min_component_width_ratio",
                ),
                1.1,
                "page_recovery_rule_min_component_width_ratio",
            ),
            (
                ("fusion", "recovery", "checkbox_tick_min_square_coverage"),
                1.1,
                "checkbox_tick_min_square_coverage",
            ),
            (
                (
                    "fusion",
                    "recovery",
                    "form_full_cell_recovery_max_width_ratio",
                ),
                1.1,
                "form_full_cell_recovery_max_width_ratio",
            ),
        )
        for path_parts, value, message in cases:
            with self.subTest(path_parts=path_parts), tempfile.TemporaryDirectory() as temp_dir:
                config = json.loads(source.read_text(encoding="utf-8"))
                target = config
                for key in path_parts[:-1]:
                    target = target[key]
                target[path_parts[-1]] = value
                path = Path(temp_dir) / "invalid.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(WorkflowConfigError, message):
                    load_config(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = json.loads(source.read_text(encoding="utf-8"))
            config["fusion"]["mode"] = "bbox_vlm"
            config["fusion"]["enabled"] = False
            path = Path(temp_dir) / "disabled.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "enabled"):
                load_config(path)

    def test_config_rejects_protruding_checkbox_size_above_regular_minimum(self):
        source = Path(__file__).parents[1] / "workflow.example.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["fusion"]["recovery"][
            "checkbox_protruding_tick_min_size"
        ] = 6.0
        config["fusion"]["recovery"]["checkbox_min_size"] = 5.5
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-checkbox-tick-size.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                WorkflowConfigError,
                "checkbox_protruding_tick_min_size must not exceed",
            ):
                load_config(path)

    def test_mineru_command_uses_proxy_and_hybrid_settings(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = build_mineru_command(
                config,
                root / "input.pdf",
                root / "out",
                "http://127.0.0.1:32100",
            )

        self.assertIn("hybrid-http-client", command)
        self.assertIn("medium", command)
        self.assertIn("http://127.0.0.1:32100", command)

    def test_pipeline_command_forces_ocr_backend(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = build_pipeline_command(config, root / "input.pdf", root / "out")

        self.assertIn("pipeline", command)
        self.assertIn("ocr", command)
        self.assertNotIn("http://127.0.0.1:30000", command)

    def test_bbox_vlm_extract_runs_pipeline_only_and_uses_it_as_fusion_baseline(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["fusion"]["mode"] = "bbox_vlm"
        config["fusion"]["recognizer"]["enabled"] = False
        server = object()
        thread = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.pdf"
            input_path.write_bytes(b"pdf")
            output_path = root / "output"
            with (
                mock.patch(
                    "projects.custom_hybrid.workflow._start_parameter_proxy",
                    return_value=(server, thread, "http://127.0.0.1:32100"),
                ),
                mock.patch(
                    "projects.custom_hybrid.workflow._stop_parameter_proxy"
                ) as stop_proxy,
                mock.patch(
                    "projects.custom_hybrid.workflow.build_pipeline_command",
                    return_value=["pipeline-command"],
                ) as build_pipeline,
                mock.patch(
                    "projects.custom_hybrid.workflow.build_mineru_command"
                ) as build_hybrid,
                mock.patch(
                    "projects.custom_hybrid.workflow._run_mineru_command"
                ) as run_command,
                mock.patch(
                    "projects.custom_hybrid.workflow.fuse_output_trees",
                    return_value={"documents": {}, "failed": {}},
                ) as fuse_trees,
            ):
                result = run_extract(config, input_path, output_path)

        self.assertEqual(result, 0)
        build_pipeline.assert_called_once_with(
            config,
            input_path,
            output_path.resolve() / "ocr",
        )
        build_hybrid.assert_not_called()
        run_command.assert_called_once_with(["pipeline-command"])
        fuse_trees.assert_called_once_with(
            config,
            input_path,
            output_path.resolve() / "ocr",
            output_path.resolve() / "ocr",
            output_path.resolve() / "fused",
            "http://127.0.0.1:32100",
        )
        stop_proxy.assert_called_once_with(server, thread)

    def test_hybrid_fusion_extract_keeps_dual_parse_workflow(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        self.assertEqual(config["fusion"]["mode"], "hybrid_fusion")
        server = object()
        thread = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.pdf"
            input_path.write_bytes(b"pdf")
            output_path = root / "output"
            with (
                mock.patch(
                    "projects.custom_hybrid.workflow._start_parameter_proxy",
                    return_value=(server, thread, "http://127.0.0.1:32100"),
                ),
                mock.patch(
                    "projects.custom_hybrid.workflow._stop_parameter_proxy"
                ) as stop_proxy,
                mock.patch(
                    "projects.custom_hybrid.workflow.build_mineru_command",
                    return_value=["hybrid-command"],
                ) as build_hybrid,
                mock.patch(
                    "projects.custom_hybrid.workflow.build_pipeline_command",
                    return_value=["pipeline-command"],
                ) as build_pipeline,
                mock.patch(
                    "projects.custom_hybrid.workflow._run_mineru_command"
                ) as run_command,
                mock.patch(
                    "projects.custom_hybrid.workflow.fuse_output_trees",
                    return_value={"documents": {}, "failed": {}},
                ) as fuse_trees,
            ):
                result = run_extract(config, input_path, output_path)

        self.assertEqual(result, 0)
        build_hybrid.assert_called_once_with(
            config,
            input_path,
            output_path.resolve() / "hybrid",
            "http://127.0.0.1:32100",
        )
        build_pipeline.assert_called_once_with(
            config,
            input_path,
            output_path.resolve() / "ocr",
        )
        self.assertEqual(
            run_command.call_args_list,
            [mock.call(["hybrid-command"]), mock.call(["pipeline-command"])],
        )
        fuse_trees.assert_called_once_with(
            config,
            input_path,
            output_path.resolve() / "hybrid",
            output_path.resolve() / "ocr",
            output_path.resolve() / "fused",
            "http://127.0.0.1:32100",
        )
        stop_proxy.assert_called_once_with(server, thread)

    def test_bbox_vlm_repair_extract_uses_pipeline_only(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["fusion"]["mode"] = "bbox_vlm"
        config["fusion"]["recovery"]["enabled"] = False
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.pdf"
            input_path.write_bytes(b"pdf")
            output_path = root / "output"
            with (
                mock.patch(
                    "projects.custom_hybrid.workflow._start_parameter_proxy",
                    return_value=(object(), object(), "http://127.0.0.1:32100"),
                ),
                mock.patch("projects.custom_hybrid.workflow._stop_parameter_proxy"),
                mock.patch(
                    "projects.custom_hybrid.workflow.build_pipeline_command",
                    return_value=["pipeline-command"],
                ) as build_pipeline,
                mock.patch(
                    "projects.custom_hybrid.workflow.build_mineru_command"
                ) as build_hybrid,
                mock.patch("projects.custom_hybrid.workflow._run_mineru_command"),
                mock.patch(
                    "projects.custom_hybrid.workflow.fuse_output_trees",
                    return_value={"documents": {}, "failed": {}},
                ) as fuse_trees,
            ):
                result = run_extract(config, input_path, output_path)

        self.assertEqual(result, 0)
        build_hybrid.assert_not_called()
        build_pipeline.assert_called_once()
        self.assertEqual(
            fuse_trees.call_args.args[2:4],
            (output_path.resolve() / "ocr", output_path.resolve() / "ocr"),
        )

    def test_vllm_server_command_encodes_object_arguments_as_json(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["vllm"]["server_args"]["compilation_config"] = {"level": 2}

        command = build_vllm_server_command(config)

        value = command[command.index("--compilation-config") + 1]
        self.assertEqual(value, '{"level":2}')

    def test_doctor_returns_structured_runtime_report(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["vllm"]["upstream_url"] = "http://127.0.0.1:1"

        report = run_doctor(config)

        self.assertIn("ready", report)
        self.assertTrue(report["local_mineru_source"])
        self.assertIn("pypdf", report["dependencies"])
        self.assertIn("reportlab", report["dependencies"])
        self.assertIn("six", report["dependencies"])
        self.assertEqual(report["upstream"]["url"], "http://127.0.0.1:1")
        self.assertFalse(report["upstream"]["reachable"])
        self.assertFalse(report["recognizer"]["enabled"])
        self.assertTrue(report["recognizer"]["ready"])
        self.assertFalse(report["recovery"]["enabled"])
        self.assertTrue(report["recovery"]["ready"])

    def test_doctor_checks_bbox_vlm_recognizer_model_and_structured_output(self):
        class Response:
            def __init__(self, payload):
                self.status_code = 200
                self.is_success = True
                self.payload = payload

            def json(self):
                return self.payload

        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["vllm"]["upstream_url"] = "http://vision.test"
        config["fusion"]["mode"] = "bbox_vlm"
        recognizer = config["fusion"]["recognizer"]
        recognizer["enabled"] = False
        recognizer["base_url"] = "http://vision.test"
        recognizer["model"] = "document-vision"
        recognizer["structured_output_mode"] = "json_schema"
        models = Response(
            {
                "data": [
                    {
                        "id": "document-vision",
                        "max_model_len": 8192,
                    }
                ]
            }
        )
        openapi = Response(
            {
                "components": {
                    "schemas": {
                        "ChatCompletionRequest": {
                            "properties": {"response_format": {}}
                        },
                        "ResponseFormat": {
                            "properties": {
                                "type": {
                                    "enum": ["text", "json_object", "json_schema"]
                                }
                            }
                        },
                    }
                }
            }
        )

        with mock.patch("httpx.get", side_effect=[models, models, openapi]):
            report = run_doctor(config)

        self.assertTrue(report["recognizer"]["ready"])
        self.assertTrue(report["recognizer"]["enabled"])
        self.assertTrue(report["recognizer"]["model_available"])
        self.assertTrue(report["recognizer"]["structured_output_supported"])
        self.assertEqual(report["recognizer"]["max_model_len"], 8192)
        self.assertTrue(report["recognizer"]["capability_trial_required"])

    def test_doctor_uses_native_protocol_for_mineru_model_without_openapi_probe(self):
        class Response:
            def __init__(self, payload):
                self.status_code = 200
                self.is_success = True
                self.payload = payload

            def json(self):
                return self.payload

        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["vllm"]["upstream_url"] = "http://vision.test"
        config["fusion"]["mode"] = "bbox_vlm"
        models = Response(
            {
                "data": [
                    {"id": "mineru-claim-forms", "max_model_len": 8192}
                ]
            }
        )

        with mock.patch("httpx.get", side_effect=[models, models, models]) as get:
            report = run_doctor(config)

        self.assertEqual(get.call_count, 3)
        self.assertTrue(report["recognizer"]["ready"])
        self.assertEqual(report["recognizer"]["protocol"], "mineru_native")
        self.assertEqual(
            report["recognizer"]["structured_output_mode"],
            "mineru_native",
        )
        self.assertTrue(report["recovery"]["ready"])
        self.assertEqual(report["recovery"]["protocol"], "local_pixel")

    def test_doctor_checks_recovery_endpoint_model_and_json_schema(self):
        class Response:
            def __init__(self, payload):
                self.status_code = 200
                self.is_success = True
                self.payload = payload

            def json(self):
                return self.payload

        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["vllm"]["upstream_url"] = "http://vision.test"
        config["fusion"]["mode"] = "bbox_vlm"
        config["fusion"]["recognizer"]["model"] = "document-vision"
        recovery = config["fusion"]["recovery"]
        recovery["base_url"] = "http://recovery.test"
        recovery["model"] = "geometry-reviewer"
        models = Response(
            {
                "data": [
                    {"id": "document-vision", "max_model_len": 8192},
                ]
            }
        )
        recovery_models = Response(
            {
                "data": [
                    {"id": "geometry-reviewer", "max_model_len": 4096},
                ]
            }
        )
        openapi = Response(
            {
                "components": {
                    "schemas": {
                        "ChatCompletionRequest": {
                            "properties": {"response_format": {}}
                        },
                        "ResponseFormat": {
                            "properties": {
                                "type": {
                                    "enum": ["text", "json_object", "json_schema"]
                                }
                            }
                        },
                    }
                }
            }
        )

        with mock.patch(
            "httpx.get",
            side_effect=[models, models, openapi, recovery_models, openapi],
        ):
            report = run_doctor(config)

        self.assertTrue(report["recovery"]["enabled"])
        self.assertTrue(report["recovery"]["ready"])
        self.assertEqual(report["recovery"]["url"], "http://recovery.test")
        self.assertTrue(report["recovery"]["model_available"])
        self.assertTrue(report["recovery"]["structured_output_supported"])
        self.assertEqual(report["recovery"]["max_model_len"], 4096)

    def test_benchmark_ranks_runs_and_groups_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_dir = root / "reference"
            good_dir = root / "good" / "nested"
            bad_dir = root / "bad"
            reference_dir.mkdir()
            good_dir.mkdir(parents=True)
            bad_dir.mkdir()
            (reference_dir / "scan.md").write_text("# Invoice\nTotal 100", encoding="utf-8")
            (good_dir / "scan.md").write_text("# Invoice\nTotal 100", encoding="utf-8")
            (bad_dir / "scan.md").write_text("Invoice Total", encoding="utf-8")
            manifest = {
                "version": 1,
                "documents": [
                    {
                        "id": "scan",
                        "reference": "reference/scan.md",
                        "tags": ["scan", "table"],
                        "weight": 2,
                    }
                ],
            }
            manifest_path = root / "benchmark.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = evaluate_benchmark_runs(
                manifest_path,
                [f"good={good_dir.parent}", f"bad={bad_dir}"],
                {},
            )

        self.assertEqual(report["leaderboard"][0]["label"], "good")
        self.assertEqual(report["runs"]["good"]["categories"]["scan"]["document_count"], 1)
        self.assertGreater(
            report["runs"]["good"]["aggregate"]["quality_score"],
            report["runs"]["bad"]["aggregate"]["quality_score"],
        )
        self.assertEqual(report["baseline"], "good")
        self.assertEqual(report["comparisons"]["bad"]["regressions"][0]["id"], "scan")

    def test_benchmark_penalizes_missing_candidate_by_weighted_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            for name in ("easy", "hard"):
                (reference / f"{name}.md").write_text(name, encoding="utf-8")
            (candidate / "easy.md").write_text("easy", encoding="utf-8")
            manifest = {
                "version": 1,
                "documents": [
                    {
                        "id": "easy",
                        "reference": "reference/easy.md",
                        "tags": ["native"],
                        "weight": 1,
                    },
                    {
                        "id": "hard",
                        "reference": "reference/hard.md",
                        "tags": ["scan"],
                        "weight": 3,
                    },
                ],
            }
            manifest_path = root / "benchmark.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = evaluate_benchmark_runs(
                manifest_path, [f"partial={candidate}"], {}
            )

        run = report["runs"]["partial"]
        self.assertEqual(run["coverage"], 0.25)
        self.assertEqual(run["coverage_adjusted_quality_score"], 0.25)
        self.assertEqual(run["categories"]["scan"]["coverage"], 0.0)

    def test_sweep_variants_apply_cartesian_generation_overrides(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["sweep"] = {
            "generation": {"temperature": [0.0, 0.1], "top_p": [0.9, 1.0]},
            "max_runs": 4,
        }

        variants = build_sweep_variants(config)

        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[0][0], "run-001")
        self.assertEqual(
            variants[-1][2]["vllm"]["generation"]["overrides"]["temperature"],
            0.1,
        )
        self.assertEqual(
            variants[-1][2]["vllm"]["generation"]["overrides"]["top_p"],
            1.0,
        )

    def test_sweep_rejects_grid_larger_than_limit(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["sweep"] = {
            "generation": {"temperature": [0.0, 0.1], "top_p": [0.9, 1.0]},
            "max_runs": 3,
        }

        with self.assertRaisesRegex(WorkflowConfigError, "exceeding"):
            build_sweep_variants(config)

    def test_input_index_matches_mineru_duplicate_stem_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "sample.pdf"
            second = root / "sample.png"
            nested = root / "nested"
            first.write_bytes(b"pdf")
            second.write_bytes(b"png")
            nested.mkdir()
            (nested / "ignored.pdf").write_bytes(b"pdf")

            indexed = _index_input_documents(root)

        self.assertEqual(list(indexed), ["sample", "sample_2"])

    def test_fuse_output_trees_copies_fuses_and_regenerates(self):
        def build_middle(text, score=None, backend="hybrid"):
            span = {
                "type": "text",
                "content": text,
                "bbox": [10, 10, 100, 30],
            }
            if score is not None:
                span["score"] = score
            return {
                "_backend": backend,
                "pdf_info": [
                    {
                        "page_size": [200, 300],
                        "preproc_blocks": [
                            {
                                "type": "text",
                                "lines": [
                                    {"bbox": [10, 10, 100, 30], "spans": [span]}
                                ],
                            }
                        ],
                    }
                ],
            }

        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["fusion"]["verifier"]["enabled"] = False
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_pdf = root / "sample.pdf"
            input_pdf.write_bytes(b"pdf")
            hybrid_root = root / "hybrid" / "sample" / "hybrid_ocr"
            ocr_root = root / "ocr" / "sample" / "ocr"
            fused_root = root / "fused"
            hybrid_root.mkdir(parents=True)
            ocr_root.mkdir(parents=True)
            (hybrid_root / "sample_middle.json").write_text(
                json.dumps(build_middle("abcdabcdabcd")), encoding="utf-8"
            )
            (ocr_root / "sample_middle.json").write_text(
                json.dumps(build_middle("abcd", 0.99, backend="pipeline")),
                encoding="utf-8",
            )

            with mock.patch(
                "projects.custom_hybrid.workflow._regenerate_fused_outputs"
            ) as regenerate:
                summary = fuse_output_trees(
                    config,
                    input_pdf,
                    root / "hybrid",
                    root / "ocr",
                    fused_root,
                    "http://127.0.0.1:30001",
                )

            fused_path = next(fused_root.rglob("sample_middle.json"))
            fused = json.loads(fused_path.read_text(encoding="utf-8"))

        span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["content"], "abcd")
        self.assertFalse(summary["failed"])
        self.assertEqual(summary["documents"]["sample"]["counts"]["ocr_replacements"], 1)
        regenerate.assert_called_once_with(
            fused_path.parent,
            "sample",
            input_pdf.resolve(),
            {
                "enabled": True,
                "replace_primary": True,
                "preserve_native": True,
            },
            config["fusion"]["page_sorting"],
        )

    def test_fuse_output_trees_wires_and_closes_enabled_bbox_recognizer(self):
        def build_middle(text, score=None):
            span = {
                "type": "text",
                "content": text,
                "bbox": [10, 10, 100, 30],
            }
            if score is not None:
                span["score"] = score
            return {
                "pdf_info": [
                    {
                        "page_size": [200, 300],
                        "preproc_blocks": [
                            {
                                "type": "text",
                                "bbox": [10, 10, 100, 30],
                                "lines": [
                                    {
                                        "bbox": [10, 10, 100, 30],
                                        "spans": [span],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["fusion"]["verifier"]["enabled"] = False
        config["fusion"]["recognizer"]["enabled"] = True
        config["fusion"]["recognizer"]["base_url"] = "http://vision.test"
        recognizer = mock.Mock()
        recognizer.return_value = {
            "items": [],
            "batches": [{"status": "ok", "responses": 0}],
            "requests": 1,
            "invalid_outputs": 0,
            "errors": 0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_pdf = root / "sample.pdf"
            input_pdf.write_bytes(b"pdf")
            hybrid_root = root / "hybrid" / "sample" / "hybrid_ocr"
            ocr_root = root / "ocr" / "sample" / "ocr"
            fused_root = root / "fused"
            hybrid_root.mkdir(parents=True)
            ocr_root.mkdir(parents=True)
            (hybrid_root / "sample_middle.json").write_text(
                json.dumps(build_middle("Hybrid")),
                encoding="utf-8",
            )
            (hybrid_root / "sample_origin.pdf").write_bytes(b"normalized")
            (ocr_root / "sample_middle.json").write_text(
                json.dumps(build_middle("OCR", 0.99)),
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "projects.custom_hybrid.workflow.OpenAIBBoxRecognizer",
                    return_value=recognizer,
                ) as recognizer_class,
                mock.patch(
                    "projects.custom_hybrid.workflow._regenerate_fused_outputs"
                ),
            ):
                summary = fuse_output_trees(
                    config,
                    input_pdf,
                    root / "hybrid",
                    root / "ocr",
                    fused_root,
                    "http://127.0.0.1:30001",
                )

        recognizer_class.assert_called_once_with(
            "http://vision.test",
            fused_root / "sample" / "hybrid_ocr" / "sample_origin.pdf",
            config["fusion"]["recognizer"],
        )
        recognizer.assert_called_once()
        recognizer.close.assert_called_once_with()
        self.assertFalse(summary["failed"])
        self.assertEqual(
            summary["documents"]["sample"]["counts"][
                "bbox_recognition_requests"
            ],
            1,
        )

    def test_fuse_output_trees_wires_bbox_recovery_reviewer(self):
        def build_table_middle():
            return {
                "pdf_info": [
                    {
                        "page_size": [200, 100],
                        "preproc_blocks": [
                            {
                                "type": "table_body",
                                "lines": [
                                    {
                                        "bbox": [10, 10, 190, 80],
                                        "spans": [
                                            {
                                                "type": "table",
                                                "bbox": [10, 10, 190, 80],
                                                "html": "<table><tr><td>Value</td></tr></table>",
                                                "table_cells": [
                                                    {
                                                        "bbox": [10, 10, 190, 80],
                                                        "content_spans": [
                                                            {
                                                                "bbox": [20, 20, 170, 40],
                                                                "text": "Value",
                                                            }
                                                        ],
                                                        "text": "Value",
                                                        "row_start": 0,
                                                        "row_end": 0,
                                                        "col_start": 0,
                                                        "col_end": 0,
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        config["fusion"]["mode"] = "bbox_vlm"
        config["fusion"]["recognizer"]["base_url"] = "http://vision.test"
        config["fusion"]["recovery"].update(
            {
                "enabled": True,
                "base_url": "http://recovery.test",
                "model": "geometry-reviewer",
            }
        )
        recognizer = mock.Mock(
            return_value={"items": [], "requests": 0, "errors": 0}
        )
        recognizer.target_crop_provider = mock.sentinel.shared_page_provider
        reviewer = mock.Mock(
            return_value={
                "items": [],
                "tables_reviewed": 1,
                "requests": 1,
                "errors": 0,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_pdf = root / "sample.pdf"
            input_pdf.write_bytes(b"pdf")
            ocr_dir = root / "ocr" / "sample" / "ocr"
            ocr_dir.mkdir(parents=True)
            (ocr_dir / "sample_middle.json").write_text(
                json.dumps(build_table_middle()),
                encoding="utf-8",
            )
            (ocr_dir / "sample_origin.pdf").write_bytes(b"normalized")
            with (
                mock.patch(
                    "projects.custom_hybrid.workflow.OpenAIBBoxRecognizer",
                    return_value=recognizer,
                ),
                mock.patch(
                    "projects.custom_hybrid.workflow.OpenAIBBoxRecoveryReviewer",
                    return_value=reviewer,
                ) as reviewer_class,
                mock.patch(
                    "projects.custom_hybrid.workflow._regenerate_fused_outputs"
                ),
            ):
                summary = fuse_output_trees(
                    config,
                    input_pdf,
                    root / "ocr",
                    root / "ocr",
                    root / "fused",
                    "http://127.0.0.1:30001",
                )

        effective = dict(config["fusion"]["recognizer"])
        effective.update(config["fusion"]["recovery"])
        reviewer_class.assert_called_once_with(
            "http://recovery.test",
            root / "fused" / "sample" / "ocr" / "sample_origin.pdf",
            effective,
            page_provider=mock.sentinel.shared_page_provider,
        )
        reviewer.assert_called_once()
        reviewer.close.assert_called_once_with()
        self.assertFalse(summary["failed"])
        self.assertEqual(
            summary["documents"]["sample"]["counts"][
                "bbox_recovery_tables_reviewed"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()

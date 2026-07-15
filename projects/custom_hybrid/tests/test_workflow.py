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
    run_doctor,
    _index_input_documents,
    _start_parameter_proxy,
    _stop_parameter_proxy,
)


class WorkflowTests(unittest.TestCase):
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
            self.assertEqual(forwarded["max_tokens"], 4096)
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
                    "max_tokens": 4096,
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
            "remove": ["min_p"],
            "rules": [
                {
                    "name": "ocr",
                    "match": {"text_regex": "extract text"},
                    "overrides": {"max_tokens": 12000},
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

        self.assertEqual(applied.body["temperature"], 0.2)
        self.assertEqual(applied.body["top_p"], 1.0)
        self.assertEqual(applied.body["seed"], 42)
        self.assertEqual(applied.body["max_tokens"], 12000)
        self.assertNotIn("min_p", applied.body)
        self.assertEqual(applied.matched_rules, ("ocr",))
        self.assertEqual(original["min_p"], 0.1)

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
        config["fusion"]["verifier"]["table_cell_render_scale"] = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-scale.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowConfigError, "table_cell_render_scale"):
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
        self.assertIn("high", command)
        self.assertIn("http://127.0.0.1:32100", command)

    def test_pipeline_command_forces_ocr_backend(self):
        config = load_config(Path(__file__).parents[1] / "workflow.example.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = build_pipeline_command(config, root / "input.pdf", root / "out")

        self.assertIn("pipeline", command)
        self.assertIn("ocr", command)
        self.assertNotIn("http://127.0.0.1:30000", command)

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
        self.assertIn("six", report["dependencies"])
        self.assertEqual(report["upstream"]["url"], "http://127.0.0.1:1")
        self.assertFalse(report["upstream"]["reachable"])

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
        regenerate.assert_called_once()


if __name__ == "__main__":
    unittest.main()

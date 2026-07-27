import copy
import importlib.util
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

PDF_RENDERING_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("pypdf", "reportlab")
)

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.api import _normalize_task_parameters, create_app
from projects.custom_hybrid.api_client import build_parser, collect_inputs


class CustomHybridApiTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        source = Path(__file__).parents[1] / "workflow.example.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["vllm"]["proxy"]["port"] = 0
        path = root / "workflow.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    @staticmethod
    def _successful_runner(_config, input_path, output_path):
        input_names = sorted(item.name for item in Path(input_path).iterdir())
        output_root = Path(output_path)
        fused = output_root / "fused" / "document"
        fused.mkdir(parents=True)
        (fused / "document.md").write_text(
            "# Fused result\n\n![](images/page.png)",
            encoding="utf-8",
        )
        (fused / "document_content_list.json").write_text(
            json.dumps([{"type": "text", "text": "Fused result"}]),
            encoding="utf-8",
        )
        (fused / "document_document.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "coordinate_system": {
                        "bbox_format": "xywh",
                        "unit": "normalized",
                        "origin": "top_left",
                    },
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
                                            "text": "Fused result",
                                            "bbox": [0.1, 0.1, 0.8, 0.1],
                                            "type": "text",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    "raw_metadata": {"source": "test"},
                }
            ),
            encoding="utf-8",
        )
        (fused / "images").mkdir()
        (fused / "images" / "page.png").write_bytes(b"image-data")
        (fused / "document_span.pdf").write_bytes(b"bbox-pdf")
        (fused / "document_form_cells.pdf").write_bytes(b"form-cell-pdf")
        (fused / "document_sorting_report.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "mode": "report_only",
                    "status": "complete",
                    "page_count": 4,
                    "anchor_count": 2,
                    "group_count": 2,
                    "can_auto_sort": True,
                    "physical_order": ["p0002", "p0003", "p0000", "p0001"],
                    "proposed_document_order": [],
                    "unresolved": [],
                    "groups": [
                        {
                            "group_id": "doc-001",
                            "expected_total": 2,
                            "member_page_ids": ["p0002", "p0000"],
                            "identifiers": {"case": ["casea111"]},
                            "status": "complete",
                            "resolved_order": ["p0000", "p0002"],
                            "duplicates": {},
                            "missing_numbers": [],
                            "unexpected_numbers": [],
                            "ambiguous_pages": {},
                            "decisions": [],
                        },
                        {
                            "group_id": "doc-002",
                            "expected_total": 2,
                            "member_page_ids": ["p0003", "p0001"],
                            "identifiers": {"case": ["caseb222"]},
                            "status": "complete",
                            "resolved_order": ["p0001", "p0003"],
                            "duplicates": {},
                            "missing_numbers": [],
                            "unexpected_numbers": [],
                            "ambiguous_pages": {},
                            "decisions": [],
                        },
                    ],
                    "assignment_count": 2,
                    "unresolved_count": 0,
                    "report_only": True,
                }
            ),
            encoding="utf-8",
        )
        (output_root / "fusion_summary.json").write_text(
            json.dumps({"documents": input_names, "failed": {}}),
            encoding="utf-8",
        )
        return 0

    @staticmethod
    def _successful_runner_without_preview(_config, input_path, output_path):
        from pypdf import PdfWriter

        input_names = sorted(item.name for item in Path(input_path).iterdir())
        output_root = Path(output_path)
        fused = output_root / "fused" / "renamed"
        fused.mkdir(parents=True)
        (fused / "renamed_middle.json").write_text(
            json.dumps(
                {
                    "pdf_info": [
                        {
                            "discarded_blocks": [],
                            "preproc_blocks": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=300)
        with (fused / "renamed_origin.pdf").open("wb") as stream:
            writer.write(stream)
        (output_root / "fusion_summary.json").write_text(
            json.dumps({"documents": input_names, "failed": {}}),
            encoding="utf-8",
        )
        return 0

    def test_unauthenticated_async_task_returns_fused_zip_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                api_key="secret",
                runner=self._successful_runner,
            )
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertFalse(health.json()["authentication_required"])
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                )
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertEqual(status["status"], "completed")
                result = client.get(f"/tasks/{task_id}/result")
                self.assertEqual(result.status_code, 200)
                archive_path = root / "result.zip"
                archive_path.write_bytes(result.content)
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertIn("fused/document/document.md", archive.namelist())
                    self.assertIn(
                        "fused/document/document_sorting_report.json",
                        archive.namelist(),
                    )
                    self.assertIn(
                        "fused/document/document_document.json",
                        archive.namelist(),
                    )
                    self.assertIn("fusion_summary.json", archive.namelist())
                    self.assertIn("task_parameters.json", archive.namelist())
                    task_parameters = json.loads(
                        archive.read("task_parameters.json").decode("utf-8")
                    )
                    self.assertEqual(task_parameters["cost_profile"], "balanced")
                    self.assertEqual(task_parameters["extraction_mode"], "bbox_vlm")
                    self.assertEqual(task_parameters["mineru"]["effort"], "medium")
                report = client.get(f"/tasks/{task_id}/report")
                self.assertEqual(report.json()["documents"], ["invoice.pdf"])
                sorting = client.get(f"/tasks/{task_id}/sorting")
                self.assertEqual(sorting.status_code, 200)
                sorting_documents = sorting.json()["documents"]
                self.assertEqual(len(sorting_documents), 1)
                self.assertEqual(
                    sorting_documents[0]["id"],
                    "document/document_sorting_report.json",
                )
                self.assertEqual(sorting_documents[0]["name"], "document")
                self.assertTrue(sorting_documents[0]["report"]["can_auto_sort"])
                self.assertEqual(
                    sorting_documents[0]["report"]["groups"][0]["resolved_order"],
                    ["p0000", "p0002"],
                )
                preview = client.get(f"/tasks/{task_id}/preview")
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(preview.content, b"bbox-pdf")
                self.assertEqual(preview.headers["content-type"], "application/pdf")
                form_cells = client.get(
                    f"/tasks/{task_id}/preview",
                    params={"kind": "form_cells"},
                )
                self.assertEqual(form_cells.status_code, 200)
                self.assertEqual(form_cells.content, b"form-cell-pdf")
                markdown = client.get(
                    f"/tasks/{task_id}/markdown",
                )
                self.assertEqual(markdown.status_code, 200)
                self.assertEqual(markdown.json()["selected"], "document/document.md")
                self.assertIn("# Fused result", markdown.json()["markdown"])
                self.assertIn("Fused result", markdown.json()["content_list"])
                structured = client.get(f"/tasks/{task_id}/structured")
                self.assertEqual(structured.status_code, 200)
                self.assertEqual(
                    structured.json()["selected"],
                    "document/document_document.json",
                )
                self.assertEqual(
                    structured.json()["output"]["documents"][0]["pages"][0][
                        "blocks"
                    ][0]["text"],
                    "Fused result",
                )
                schema = client.get("/schemas/document-output")
                self.assertEqual(schema.status_code, 200)
                self.assertIn("documents", schema.json()["properties"])
                asset = client.get(
                    f"/tasks/{task_id}/asset",
                    params={
                        "document": "document/document.md",
                        "path": "images/page.png",
                    },
                )
                self.assertEqual(asset.status_code, 200)
                self.assertEqual(asset.content, b"image-data")
                traversal = client.get(
                    f"/tasks/{task_id}/asset",
                    params={
                        "document": "document/document.md",
                        "path": "../../task_parameters.json",
                    },
                )
                self.assertEqual(traversal.status_code, 404)
                deleted = client.delete(f"/tasks/{task_id}")
                self.assertEqual(deleted.json(), {"task_id": task_id, "deleted": True})

    def test_synchronous_parse_returns_zip_and_rejects_unsupported_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=self._successful_runner,
            )
            with TestClient(app) as client:
                rejected = client.post(
                    "/file_parse",
                    files={"files": ("payload.exe", b"bad")},
                )
                self.assertEqual(rejected.status_code, 400)
                response = client.post(
                    "/file_parse",
                    files={"files": ("scan.png", b"image")},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "application/zip")

    def test_openapi_renders_upload_arrays_as_file_pickers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=self._successful_runner,
            )
            with TestClient(app) as client:
                openapi = client.get("/openapi.json").json()
                for path in ("/tasks", "/file_parse"):
                    request_schema = openapi["paths"][path]["post"]["requestBody"][
                        "content"
                    ]["multipart/form-data"]["schema"]
                    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
                    files_schema = openapi["components"]["schemas"][component_name][
                        "properties"
                    ]["files"]

                    self.assertEqual(files_schema["type"], "array")
                    self.assertEqual(files_schema["items"]["type"], "string")
                    self.assertEqual(files_schema["items"]["format"], "binary")
                    self.assertIn("one or more", files_schema["description"])
                    properties = openapi["components"]["schemas"][component_name][
                        "properties"
                    ]
                    self.assertEqual(
                        properties["extraction_mode"]["default"],
                        "bbox_vlm",
                    )
                    optional_properties = set(properties) - {
                        "files",
                        "cost_profile",
                        "extraction_mode",
                    }
                    for property_name in optional_properties:
                        property_schema = openapi["components"]["schemas"][
                            component_name
                        ]["properties"][property_name]
                        self.assertEqual(property_schema["example"], "")

                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={
                        "extraction_mode": "",
                        "effort": "",
                        "method": "",
                        "lang": "",
                        "temperature": "",
                        "top_p": "",
                        "seed": "",
                        "max_tokens": "",
                        "repetition_penalty": "",
                        "recovery_max_tables": "",
                        "recovery_max_proposals": "",
                        "recovery_min_confidence": "",
                        "page_sorting_llm_enabled": "",
                    },
                )
                self.assertEqual(response.status_code, 202)

    @unittest.skipUnless(PDF_RENDERING_AVAILABLE, "PDF rendering dependencies missing")
    def test_preview_endpoint_regenerates_missing_span_from_origin_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=self._successful_runner_without_preview,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                )
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)
                self.assertEqual(status["status"], "completed")
                preview = client.get(f"/tasks/{task_id}/preview")
                self.assertEqual(preview.status_code, 200)
                self.assertTrue(preview.content.startswith(b"%PDF"))
                fused_root = root / "tasks" / task_id / "output" / "fused"
                self.assertTrue(next(fused_root.rglob("*_span.pdf")).is_file())

    def test_preview_endpoint_attempts_on_demand_regeneration(self):
        def runner_without_preview(_config, input_path, output_path):
            input_names = sorted(item.name for item in Path(input_path).iterdir())
            output_root = Path(output_path)
            (output_root / "fused").mkdir(parents=True)
            (output_root / "fusion_summary.json").write_text(
                json.dumps({"documents": input_names, "failed": {}}),
                encoding="utf-8",
            )
            return 0

        def generate_preview(fused_root, _input_root):
            preview = Path(fused_root) / "document" / "document_span.pdf"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"generated-preview")
            return (preview,)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=runner_without_preview,
            )
            with mock.patch(
                "projects.custom_hybrid.api.regenerate_fused_visualizations",
                side_effect=generate_preview,
            ) as regenerate:
                with TestClient(app) as client:
                    response = client.post(
                        "/tasks",
                        files={"files": ("invoice.pdf", b"pdf")},
                    )
                    task_id = response.json()["task_id"]
                    for _attempt in range(100):
                        status = client.get(f"/tasks/{task_id}").json()
                        if status["status"] in {"completed", "failed"}:
                            break
                        time.sleep(0.01)
                    preview = client.get(f"/tasks/{task_id}/preview")

        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.content, b"generated-preview")
        regenerate.assert_called_once()

    def test_failed_workflow_is_reported_without_result_download(self):
        def failing_runner(_config, _input, _output):
            raise RuntimeError("upstream unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=failing_runner,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                )
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}").json()
                    if status["status"] == "failed":
                        break
                    time.sleep(0.01)
                self.assertIn("upstream unavailable", status["error"])
                self.assertEqual(
                    client.get(f"/tasks/{task_id}/result").status_code,
                    409,
                )

    def test_task_parameters_override_mineru_and_generation_rules(self):
        captured_configs = []

        def capturing_runner(config, input_path, output_path):
            captured_configs.append(copy.deepcopy(config))
            return self._successful_runner(config, input_path, output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=capturing_runner,
            )
            with TestClient(app) as client:
                health = client.get("/health").json()
                self.assertIn("task_parameter_defaults", health)
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={
                        "cost_profile": "balanced",
                        "extraction_mode": "hybrid_fusion",
                        "effort": "medium",
                        "method": "ocr",
                        "lang": "en",
                        "temperature": "0.25",
                        "top_p": "0.9",
                        "seed": "123",
                        "max_tokens": "1536",
                        "repetition_penalty": "1.03",
                    },
                )
                self.assertEqual(response.status_code, 202)
                task = response.json()
                self.assertEqual(task["parameters"]["mineru"]["effort"], "medium")
                self.assertEqual(
                    task["parameters"]["generation"]["temperature"],
                    0.25,
                )
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task['task_id']}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)
                self.assertEqual(status["status"], "completed")

            self.assertEqual(captured_configs[0]["mineru"]["effort"], "medium")
            self.assertFalse(captured_configs[0]["mineru"]["formula"])
            self.assertFalse(captured_configs[0]["mineru"]["image_analysis"])
            self.assertEqual(captured_configs[0]["mineru"]["method"], "ocr")
            self.assertEqual(captured_configs[0]["mineru"]["lang"], "en")
            self.assertEqual(
                captured_configs[0]["vllm"]["generation"]["task_overrides"],
                {
                    "temperature": 0.25,
                    "top_p": 0.9,
                    "seed": 123,
                    "max_tokens": 1536,
                    "repetition_penalty": 1.03,
                },
            )
            self.assertFalse(captured_configs[0]["fusion"]["verifier"]["enabled"])
            self.assertFalse(captured_configs[0]["fusion"]["recognizer"]["enabled"])
            self.assertFalse(captured_configs[0]["fusion"]["reconciliation"]["enabled"])
            self.assertEqual(
                captured_configs[0]["fusion"]["max_verifications_per_document"],
                0,
            )

    def test_quality_profile_preserves_workflow_cost_settings(self):
        captured_configs = []

        def capturing_runner(config, input_path, output_path):
            captured_configs.append(copy.deepcopy(config))
            return self._successful_runner(config, input_path, output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["mineru"].update(
                {"effort": "high", "formula": True, "image_analysis": True}
            )
            config["fusion"]["verifier"]["enabled"] = True
            config["fusion"]["max_verifications_per_document"] = 12
            config_path.write_text(json.dumps(config), encoding="utf-8")
            app = create_app(config_path, root / "tasks", runner=capturing_runner)
            with TestClient(app) as client:
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={
                        "cost_profile": "quality",
                        "extraction_mode": "hybrid_fusion",
                    },
                )
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

            self.assertEqual(status["status"], "completed")
            self.assertEqual(captured_configs[0]["mineru"]["effort"], "high")
            self.assertTrue(captured_configs[0]["mineru"]["formula"])
            self.assertTrue(captured_configs[0]["mineru"]["image_analysis"])
            self.assertTrue(captured_configs[0]["fusion"]["verifier"]["enabled"])
            self.assertEqual(
                captured_configs[0]["fusion"]["max_verifications_per_document"],
                12,
            )

    def test_api_forces_semantic_outputs_when_server_config_is_stale(self):
        captured_configs = []

        def capturing_runner(config, input_path, output_path):
            captured_configs.append(copy.deepcopy(config))
            return self._successful_runner(config, input_path, output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["fusion"].pop("semantic_markdown")
            config["fusion"].pop("page_sorting")
            config["fusion"].pop("document_output")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            app = create_app(config_path, root / "tasks", runner=capturing_runner)
            with TestClient(app) as client:
                defaults = client.get("/health").json()["task_parameter_defaults"]
                self.assertTrue(defaults["semantic_markdown"]["enabled"])
                self.assertTrue(defaults["page_sorting"]["enabled"])
                self.assertTrue(defaults["document_output"]["enabled"])
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={
                        "cost_profile": "balanced",
                        "extraction_mode": "hybrid_fusion",
                    },
                )
                self.assertEqual(response.status_code, 202)
                task = response.json()
                self.assertTrue(
                    task["parameters"]["fusion"]["semantic_markdown"]["enabled"]
                )
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task['task_id']}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

            self.assertEqual(status["status"], "completed")
            fusion = captured_configs[0]["fusion"]
            self.assertEqual(
                fusion["semantic_markdown"],
                {
                    "enabled": True,
                    "replace_primary": True,
                    "preserve_native": True,
                },
            )
            self.assertEqual(
                fusion["page_sorting"],
                {
                    "enabled": True,
                    "mode": "report_only",
                    "include_semantic_diagnostics": True,
                },
            )
            self.assertEqual(fusion["document_output"], {"enabled": True})
            snapshot = json.loads(
                (
                    root
                    / "tasks"
                    / task["task_id"]
                    / "output"
                    / "task_parameters.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(snapshot["semantic_markdown"]["enabled"])
            self.assertTrue(snapshot["page_sorting"]["enabled"])
            self.assertTrue(snapshot["document_output"]["enabled"])

    def test_bbox_vlm_mode_overrides_balanced_profile_after_cost_settings(self):
        captured_configs = []

        def capturing_runner(config, input_path, output_path):
            captured_configs.append(copy.deepcopy(config))
            return self._successful_runner(config, input_path, output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=capturing_runner,
            )
            with TestClient(app) as client:
                health = client.get("/health").json()
                self.assertEqual(
                    health["task_parameter_defaults"]["extraction_mode"],
                    "bbox_vlm",
                )
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={
                        "cost_profile": "balanced",
                        "extraction_mode": "bbox_vlm",
                        "temperature": "0.2",
                        "top_p": "0.9",
                        "seed": "123",
                        "max_tokens": "768",
                    },
                )
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

        self.assertEqual(status["status"], "completed")
        fusion = captured_configs[0]["fusion"]
        self.assertEqual(fusion["mode"], "bbox_vlm")
        self.assertTrue(fusion["enabled"])
        self.assertEqual(
            fusion["semantic_markdown"],
            {
                "enabled": True,
                "replace_primary": True,
                "preserve_native": True,
            },
        )
        self.assertTrue(fusion["recognizer"]["enabled"])
        self.assertFalse(fusion["recognizer"]["normal_ocr_enabled"])
        self.assertTrue(fusion["recognizer"]["table_ocr_enabled"])
        self.assertEqual(
            fusion["recognizer"]["selection_policy"],
            "vlm_primary",
        )
        self.assertTrue(fusion["recognizer"]["include_row_image"])
        self.assertFalse(fusion["recognizer"]["include_table_image"])
        self.assertEqual(fusion["recognizer"]["max_images_per_request"], 8)
        self.assertEqual(fusion["recognizer"]["max_image_limit_retries"], 2)
        self.assertEqual(fusion["recognizer"]["native_min_bbox_height"], 20.0)
        self.assertEqual(fusion["recognizer"]["native_max_tokens"], 256)
        self.assertEqual(
            fusion["recognizer"]["native_max_requests_per_page"],
            12,
        )
        self.assertEqual(
            fusion["recognizer"]["native_max_candidates_per_page"],
            12,
        )
        self.assertEqual(
            fusion["recognizer"]["max_requests_per_document"],
            50,
        )
        self.assertEqual(fusion["recognizer"]["target_render_scale"], 3.0)
        self.assertEqual(fusion["recognizer"]["jpeg_quality"], 85)
        self.assertEqual(fusion["recognizer"]["native_max_concurrency"], 2)
        self.assertTrue(fusion["recognizer"]["native_cache_enabled"])
        self.assertEqual(fusion["recognizer"]["temperature"], 0.2)
        self.assertEqual(fusion["recognizer"]["top_p"], 0.9)
        self.assertEqual(fusion["recognizer"]["seed"], 123)
        self.assertEqual(fusion["recognizer"]["max_tokens"], 768)
        self.assertTrue(fusion["recovery"]["enabled"])
        self.assertEqual(fusion["recovery"]["max_tables_per_document"], 3)
        self.assertEqual(fusion["recovery"]["max_requests_per_document"], 3)
        self.assertEqual(fusion["recovery"]["temperature"], 0.2)
        self.assertEqual(fusion["recovery"]["top_p"], 0.9)
        self.assertEqual(fusion["recovery"]["seed"], 123)
        self.assertEqual(fusion["recovery"]["max_tokens"], 768)
        self.assertFalse(fusion["verifier"]["enabled"])
        self.assertFalse(fusion["reconciliation"]["enabled"])

        quality = _normalize_task_parameters(
            cost_profile="quality",
            extraction_mode="bbox_vlm",
        )
        self.assertTrue(
            quality["fusion"]["recognizer"]["include_table_image"]
        )
        self.assertEqual(
            quality["fusion"]["recognizer"]["native_min_bbox_height"],
            16.0,
        )
        self.assertEqual(
            quality["fusion"]["recognizer"]["native_max_requests_per_page"],
            24,
        )
        self.assertEqual(
            quality["fusion"]["recovery"]["max_proposals_per_document"],
            500,
        )
        self.assertEqual(
            quality["fusion"]["recovery"]["max_proposals_per_table"],
            100,
        )
        recovery = _normalize_task_parameters(
            cost_profile="balanced",
            extraction_mode="bbox_vlm_recovery",
            temperature=0.1,
            seed=7,
            recovery_max_tables=4,
            recovery_max_proposals=40,
            recovery_min_confidence=0.9,
        )
        # The old recovery value remains accepted as a compatibility alias, but
        # it resolves to the single OCR -> repair -> VLM pipeline.
        self.assertEqual(recovery["fusion"]["mode"], "bbox_vlm")
        self.assertTrue(recovery["fusion"]["recovery"]["enabled"])
        self.assertEqual(
            recovery["fusion"]["recovery"]["max_tables_per_document"],
            4,
        )
        self.assertEqual(
            recovery["fusion"]["recovery"]["max_proposals_per_document"],
            40,
        )
        self.assertEqual(
            recovery["fusion"]["recovery"]["min_confidence"],
            0.9,
        )
        self.assertEqual(recovery["fusion"]["recovery"]["temperature"], 0.1)
        self.assertEqual(recovery["fusion"]["recovery"]["seed"], 7)

    def test_task_llm_toggle_preserves_server_side_endpoint_configuration(self):
        captured_configs = []

        def capturing_runner(config, input_path, output_path):
            captured_configs.append(copy.deepcopy(config))
            return self._successful_runner(config, input_path, output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["fusion"]["page_sorting"]["llm"].update(
                {
                    "enabled": False,
                    "base_url": "https://llm.example.test/v1",
                    "model": "qwen3-instruct",
                    "api_key_env": "SORTING_LLM_API_KEY",
                }
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            app = create_app(config_path, root / "tasks", runner=capturing_runner)
            with TestClient(app) as client:
                defaults = client.get("/health").json()["task_parameter_defaults"]
                self.assertTrue(defaults["page_sorting"]["llm"]["available"])
                self.assertFalse(defaults["page_sorting"]["llm"]["enabled"])
                self.assertEqual(
                    defaults["page_sorting"]["llm"]["model"],
                    "qwen3-instruct",
                )
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={"page_sorting_llm_enabled": "true"},
                )
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}").json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

        self.assertEqual(status["status"], "completed")
        llm = captured_configs[0]["fusion"]["page_sorting"]["llm"]
        self.assertTrue(llm["enabled"])
        self.assertEqual(llm["base_url"], "https://llm.example.test/v1")
        self.assertEqual(llm["model"], "qwen3-instruct")
        self.assertEqual(llm["api_key_env"], "SORTING_LLM_API_KEY")

    def test_task_parameter_validation_rejects_out_of_range_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                runner=self._successful_runner,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={"temperature": "3"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("temperature", response.json()["detail"])
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={"cost_profile": "unknown"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("cost_profile", response.json()["detail"])
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={"extraction_mode": "unknown"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("extraction_mode", response.json()["detail"])
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    data={"recovery_min_confidence": "1.2"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("recovery_min_confidence", response.json()["detail"])

    def test_client_collects_supported_inputs_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.pdf").write_bytes(b"pdf")
            (root / "b.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual(collect_inputs(root), [(root / "a.pdf").resolve()])

    def test_client_accepts_bbox_vlm_extraction_mode(self):
        args = build_parser().parse_args(
            [
                "--url",
                "http://127.0.0.1:6108",
                "--input",
                "invoice.pdf",
                "--output",
                "result.zip",
                "--cost-profile",
                "balanced",
                "--extraction-mode",
                "bbox_vlm",
            ]
        )

        self.assertEqual(args.cost_profile, "balanced")
        self.assertEqual(args.extraction_mode, "bbox_vlm")

        bbox_args = build_parser().parse_args(
            [
                "--url",
                "http://127.0.0.1:6108",
                "--input",
                "invoice.pdf",
                "--output",
                "result.zip",
                "--extraction-mode",
                "bbox_vlm",
                "--recovery-max-tables",
                "4",
                "--recovery-min-confidence",
                "0.9",
                "--page-sorting-llm",
            ]
        )
        self.assertEqual(bbox_args.extraction_mode, "bbox_vlm")
        self.assertEqual(bbox_args.recovery_max_tables, 4)
        self.assertEqual(bbox_args.recovery_min_confidence, 0.9)
        self.assertTrue(bbox_args.page_sorting_llm)


if __name__ == "__main__":
    unittest.main()

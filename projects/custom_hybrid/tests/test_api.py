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

from projects.custom_hybrid.api import create_app
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
        (fused / "images").mkdir()
        (fused / "images" / "page.png").write_bytes(b"image-data")
        (fused / "document_span.pdf").write_bytes(b"bbox-pdf")
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

    def test_authenticated_async_task_returns_fused_zip_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                self._write_config(root),
                root / "tasks",
                api_key="secret",
                runner=self._successful_runner,
            )
            headers = {"Authorization": "Bearer secret"}
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(
                    client.post(
                        "/tasks",
                        files={"files": ("invoice.pdf", b"pdf")},
                    ).status_code,
                    401,
                )
                response = client.post(
                    "/tasks",
                    files={"files": ("invoice.pdf", b"pdf")},
                    headers=headers,
                )
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _attempt in range(100):
                    status = client.get(f"/tasks/{task_id}", headers=headers).json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertEqual(status["status"], "completed")
                result = client.get(f"/tasks/{task_id}/result", headers=headers)
                self.assertEqual(result.status_code, 200)
                archive_path = root / "result.zip"
                archive_path.write_bytes(result.content)
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertIn("fused/document/document.md", archive.namelist())
                    self.assertIn("fusion_summary.json", archive.namelist())
                    self.assertIn("task_parameters.json", archive.namelist())
                    task_parameters = json.loads(
                        archive.read("task_parameters.json").decode("utf-8")
                    )
                    self.assertEqual(task_parameters["cost_profile"], "balanced")
                    self.assertEqual(task_parameters["extraction_mode"], "hybrid_fusion")
                    self.assertEqual(task_parameters["mineru"]["effort"], "medium")
                report = client.get(f"/tasks/{task_id}/report", headers=headers)
                self.assertEqual(report.json()["documents"], ["invoice.pdf"])
                preview = client.get(f"/tasks/{task_id}/preview", headers=headers)
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(preview.content, b"bbox-pdf")
                self.assertEqual(preview.headers["content-type"], "application/pdf")
                markdown = client.get(
                    f"/tasks/{task_id}/markdown",
                    headers=headers,
                )
                self.assertEqual(markdown.status_code, 200)
                self.assertEqual(markdown.json()["selected"], "document/document.md")
                self.assertIn("# Fused result", markdown.json()["markdown"])
                self.assertIn("Fused result", markdown.json()["content_list"])
                asset = client.get(
                    f"/tasks/{task_id}/asset",
                    params={
                        "document": "document/document.md",
                        "path": "images/page.png",
                    },
                    headers=headers,
                )
                self.assertEqual(asset.status_code, 200)
                self.assertEqual(asset.content, b"image-data")
                traversal = client.get(
                    f"/tasks/{task_id}/asset",
                    params={
                        "document": "document/document.md",
                        "path": "../../task_parameters.json",
                    },
                    headers=headers,
                )
                self.assertEqual(traversal.status_code, 404)
                deleted = client.delete(f"/tasks/{task_id}", headers=headers)
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
                    data={"cost_profile": "quality"},
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
                    "hybrid_fusion",
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
        self.assertTrue(fusion["recognizer"]["enabled"])
        self.assertFalse(fusion["recognizer"]["normal_ocr_enabled"])
        self.assertTrue(fusion["recognizer"]["table_ocr_enabled"])
        self.assertEqual(
            fusion["recognizer"]["selection_policy"],
            "vlm_primary",
        )
        self.assertTrue(fusion["recognizer"]["include_row_image"])
        self.assertTrue(fusion["recognizer"]["include_table_image"])
        self.assertEqual(fusion["recognizer"]["temperature"], 0.2)
        self.assertEqual(fusion["recognizer"]["top_p"], 0.9)
        self.assertEqual(fusion["recognizer"]["seed"], 123)
        self.assertEqual(fusion["recognizer"]["max_tokens"], 768)
        self.assertFalse(fusion["verifier"]["enabled"])
        self.assertFalse(fusion["reconciliation"]["enabled"])

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


if __name__ == "__main__":
    unittest.main()

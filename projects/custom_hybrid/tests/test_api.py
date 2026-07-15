import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.api import create_app
from projects.custom_hybrid.api_client import collect_inputs


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
        (fused / "document.md").write_text("fused", encoding="utf-8")
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
                report = client.get(f"/tasks/{task_id}/report", headers=headers)
                self.assertEqual(report.json()["documents"], ["invoice.pdf"])
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

    def test_client_collects_supported_inputs_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.pdf").write_bytes(b"pdf")
            (root / "b.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual(collect_inputs(root), [(root / "a.pdf").resolve()])


if __name__ == "__main__":
    unittest.main()

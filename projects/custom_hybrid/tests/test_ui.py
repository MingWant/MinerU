import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.ui import create_ui_app


class CustomHybridUiTests(unittest.TestCase):
    def test_ui_proxies_auth_upload_status_report_and_download(self):
        seen = []
        archive_buffer = BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("fused/document/document.md", "done")

        async def handler(request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            seen.append((request, body))
            self.assertEqual(request.headers.get("authorization"), "Bearer secret")
            self.assertEqual(request.url.params.get("token"), "jupyter")
            path = request.url.path
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "task_parameter_defaults": {
                            "mineru": {"effort": "high", "method": "auto", "lang": "ch"},
                            "generation": {"temperature": 0.0, "seed": 42},
                        },
                    },
                )
            if path == "/tasks" and request.method == "POST":
                self.assertIn(b"invoice.pdf", body)
                self.assertIn(b'name="effort"', body)
                self.assertIn(b"medium", body)
                self.assertIn(b'name="temperature"', body)
                self.assertIn(b"0.25", body)
                return httpx.Response(
                    202,
                    json={"task_id": "abc123", "status": "queued"},
                )
            if path == "/tasks/abc123" and request.method == "GET":
                return httpx.Response(
                    200,
                    json={"task_id": "abc123", "status": "completed"},
                )
            if path == "/tasks/abc123/report":
                return httpx.Response(
                    200,
                    json={"documents": {"invoice": {"counts": {"targets": 1}}}},
                )
            if path == "/tasks/abc123/result":
                return httpx.Response(
                    200,
                    content=archive_buffer.getvalue(),
                    headers={
                        "content-type": "application/zip",
                        "content-disposition": 'attachment; filename="result.zip"',
                    },
                )
            if path == "/tasks/abc123/preview":
                return httpx.Response(
                    200,
                    content=b"bbox-preview",
                    headers={
                        "content-type": "application/pdf",
                        "content-disposition": 'inline; filename="document_span.pdf"',
                    },
                )
            if path == "/tasks/abc123" and request.method == "DELETE":
                return httpx.Response(200, json={"deleted": True})
            return httpx.Response(404, json={"detail": "missing"})

        app = create_ui_app(
            "http://remote.example",
            remote_api_key="secret",
            jupyter_token="jupyter",
            transport=httpx.MockTransport(handler),
        )
        with TestClient(app) as client:
            page = client.get("/")
            self.assertIn("Custom Hybrid MinerU", page.text)
            self.assertIn("dropZone", page.text)
            self.assertIn("Upload & Settings", page.text)
            self.assertIn("Document Preview", page.text)
            self.assertIn("Fusion Report", page.text)
            self.assertIn("OCR Spatial Coverage", page.text)
            self.assertIn("Key–Value Pairs", page.text)
            self.assertIn("vLLM Generation", page.text)
            self.assertIn('id="effortInput"', page.text)
            self.assertIn('id="temperatureInput"', page.text)
            self.assertIn('id="seedInput"', page.text)
            self.assertNotIn("拖拽", page.text)
            self.assertEqual(client.get("/api/health").json()["status"], "ok")
            submitted = client.post(
                "/api/tasks",
                files={"files": ("invoice.pdf", b"pdf")},
                data={"effort": "medium", "temperature": "0.25", "seed": "123"},
            )
            self.assertEqual(submitted.status_code, 202)
            self.assertEqual(submitted.json()["task_id"], "abc123")
            self.assertEqual(
                client.get("/api/tasks/abc123").json()["status"],
                "completed",
            )
            report = client.get("/api/tasks/abc123/report")
            self.assertEqual(report.json()["documents"]["invoice"]["counts"], {"targets": 1})
            result = client.get("/api/tasks/abc123/result")
            self.assertEqual(result.status_code, 200)
            with zipfile.ZipFile(BytesIO(result.content)) as archive:
                self.assertEqual(
                    archive.read("fused/document/document.md").decode(),
                    "done",
                )
            preview = client.get("/api/tasks/abc123/preview")
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.content, b"bbox-preview")
            self.assertEqual(preview.headers["content-type"], "application/pdf")
            self.assertIn("inline", preview.headers["content-disposition"])
            self.assertEqual(
                client.delete("/api/tasks/abc123").json(),
                {"deleted": True},
            )
            self.assertEqual(client.get("/api/tasks/not.valid").status_code, 400)

        self.assertGreaterEqual(len(seen), 6)

    def test_ui_turns_remote_connection_failure_into_502(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        app = create_ui_app(
            "http://remote.example",
            transport=httpx.MockTransport(handler),
        )
        with TestClient(app) as client:
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 502)
            self.assertIn("Cannot reach", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

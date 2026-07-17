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
                            "cost_profile": "balanced",
                            "extraction_mode": "hybrid_fusion",
                            "mineru": {"effort": "medium", "method": "auto", "lang": "ch"},
                            "generation": {"temperature": 0.0, "seed": 42},
                        },
                    },
                )
            if path == "/tasks" and request.method == "POST":
                self.assertIn(b"invoice.pdf", body)
                self.assertIn(b'name="cost_profile"', body)
                self.assertIn(b"balanced", body)
                self.assertIn(b'name="extraction_mode"', body)
                self.assertIn(b"bbox_vlm_recovery", body)
                self.assertIn(b'name="recovery_max_tables"', body)
                self.assertIn(b"4", body)
                self.assertIn(b'name="recovery_min_confidence"', body)
                self.assertIn(b"0.9", body)
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
            if path == "/tasks/abc123/markdown":
                self.assertEqual(request.url.params.get("document"), "document/document.md")
                return httpx.Response(
                    200,
                    json={
                        "documents": [
                            {"id": "document/document.md", "name": "document"}
                        ],
                        "selected": "document/document.md",
                        "name": "document",
                        "markdown": "# Done",
                        "content_list": "[]",
                    },
                )
            if path == "/tasks/abc123/asset":
                self.assertEqual(request.url.params.get("document"), "document/document.md")
                self.assertEqual(request.url.params.get("path"), "images/page.png")
                return httpx.Response(200, content=b"image")
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
            self.assertIn("VLM Recognized", page.text)
            self.assertIn("Protocol Echoes", page.text)
            self.assertIn("Invalid VLM Outputs", page.text)
            self.assertIn("Recognition Errors", page.text)
            self.assertIn("Image-Limit Rebatches", page.text)
            self.assertIn("Native MinerU Requests", page.text)
            self.assertIn("Native Cache Hits", page.text)
            self.assertIn("Native Budget Skips", page.text)
            self.assertIn("Local Pixel Proposals", page.text)
            self.assertIn("Recovery Cells Analyzed", page.text)
            self.assertIn("Recovery Pixel Time", page.text)
            self.assertIn("Recovery Tables Reviewed", page.text)
            self.assertIn("Recovered Boxes Added", page.text)
            self.assertIn("Advanced vLLM Settings", page.text)
            self.assertIn("Markdown Rendering", page.text)
            self.assertIn("Markdown Text", page.text)
            self.assertIn("Content List JSON", page.text)
            self.assertIn("Convert Again", page.text)
            self.assertIn('id="costProfileInput"', page.text)
            self.assertIn('id="extractionModeInput"', page.text)
            self.assertIn("OCR BBox + VLM", page.text)
            self.assertIn("OCR BBox + VLM Recovery", page.text)
            self.assertIn('id="effortInput"', page.text)
            self.assertIn('id="temperatureInput"', page.text)
            self.assertIn('id="seedInput"', page.text)
            self.assertIn('id="recoveryMaxTablesInput"', page.text)
            self.assertIn("Advanced BBox Recovery Settings", page.text)
            self.assertNotIn("拖拽", page.text)
            self.assertEqual(client.get("/api/health").json()["status"], "ok")
            submitted = client.post(
                "/api/tasks",
                files={"files": ("invoice.pdf", b"pdf")},
                data={
                    "cost_profile": "balanced",
                    "extraction_mode": "bbox_vlm_recovery",
                    "effort": "medium",
                    "temperature": "0.25",
                    "seed": "123",
                    "recovery_max_tables": "4",
                    "recovery_min_confidence": "0.9",
                },
            )
            self.assertEqual(submitted.status_code, 202)
            self.assertEqual(submitted.json()["task_id"], "abc123")
            self.assertEqual(
                client.get("/api/tasks/abc123").json()["status"],
                "completed",
            )
            report = client.get("/api/tasks/abc123/report")
            self.assertEqual(report.json()["documents"]["invoice"]["counts"], {"targets": 1})
            markdown = client.get(
                "/api/tasks/abc123/markdown",
                params={"document": "document/document.md"},
            )
            self.assertEqual(markdown.json()["markdown"], "# Done")
            asset = client.get(
                "/api/tasks/abc123/asset",
                params={
                    "document": "document/document.md",
                    "path": "images/page.png",
                },
            )
            self.assertEqual(asset.content, b"image")
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

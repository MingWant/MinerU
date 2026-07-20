import base64
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.fusion import (
    FusionSettings,
    PageCropProvider,
    fuse_middle_json,
)
from projects.custom_hybrid.recognition import OpenAIBBoxRecognizer
from projects.custom_hybrid.recognition_eval import evaluate_bbox_recognition


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class ImageLimitError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__("400 Bad Request")


class ImageLimitResponse(Response):
    text = "At most 8 image(s) may be provided in one prompt."

    def raise_for_status(self):
        raise ImageLimitError(self)


class FakeHttpx:
    def __init__(self):
        self.requests = []

    def get(self, *_args, **_kwargs):
        return Response(
            {
                "data": [
                    {
                        "id": "vision-recognizer",
                        "max_model_len": 8192,
                    }
                ]
            }
        )

    def post(self, *_args, json=None, **_kwargs):
        self.requests.append(json)
        prompt = json["messages"][0]["content"][0]["text"]
        evidence = json_module.loads(prompt.split("evidence_data=", 1)[1])
        schema = (
            json.get("response_format", {})
            .get("json_schema", {})
            .get("schema", {})
        )
        if not schema:
            schema = json.get("structured_outputs", {}).get("json", {})
        constrained_ids = (
            schema.get("properties", {})
            .get("items", {})
            .get("items", {})
            .get("properties", {})
            .get("id", {})
            .get("enum", [])
        )
        items = [
            {
                "id": item["id"] if "id" in item else constrained_ids[index],
                "text": item["ocr_text"].replace("1", "l"),
            }
            for index, item in enumerate(evidence["candidates"])
        ]
        items.append({"id": "unknown-id", "text": "hallucination"})
        return Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps({"items": items})
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            }
        )


class StrictFakeHttpx(FakeHttpx):
    def post(self, *_args, json=None, **_kwargs):
        self.requests.append(json)
        prompt = json["messages"][0]["content"][0]["text"]
        evidence = json_module.loads(prompt.split("evidence_data=", 1)[1])
        schema = json["response_format"]["json_schema"]["schema"]
        candidate_ids = (
            schema["properties"]["items"]["items"]["properties"]["id"][
                "enum"
            ]
        )
        items = [
            {
                "id": candidate_id,
                "text": candidate["ocr_text"].replace("1", "l"),
            }
            for candidate_id, candidate in zip(
                candidate_ids,
                evidence["candidates"],
            )
        ]
        return Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps({"items": items})
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )


class ImageLimitFakeHttpx(FakeHttpx):
    def post(self, *_args, json=None, **kwargs):
        image_count = len(json["messages"][0]["content"]) - 1
        if image_count > 8:
            self.requests.append(json)
            return ImageLimitResponse({"detail": ImageLimitResponse.text})
        return super().post(*_args, json=json, **kwargs)


class NativeMinerUFakeHttpx:
    def __init__(self):
        self.requests = []

    def get(self, *_args, **_kwargs):
        return Response(
            {
                "data": [
                    {"id": "mineru-claim-forms", "max_model_len": 8192}
                ]
            }
        )

    def post(self, *_args, json=None, headers=None, **_kwargs):
        self.requests.append({"json": json, "headers": headers})
        return Response(
            {
                "choices": [
                    {
                        "message": {"content": "Blaine<|im_end|>"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 4},
            }
        )


class InvalidNativeMinerUFakeHttpx(NativeMinerUFakeHttpx):
    def post(self, *_args, json=None, headers=None, **_kwargs):
        self.requests.append({"json": json, "headers": headers})
        return Response(
            {
                "choices": [
                    {
                        "message": {"content": "repeated output"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 512},
            }
        )


class InvalidSchemaFakeHttpx(FakeHttpx):
    def post(self, *_args, json=None, **_kwargs):
        self.requests.append(json)
        return Response(
            {
                "choices": [
                    {
                        "message": {"content": "not valid json"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


class ConcurrentNativeFakeHttpx(NativeMinerUFakeHttpx):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def post(self, *_args, json=None, headers=None, **_kwargs):
        with self.lock:
            self.requests.append({"json": json, "headers": headers})
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.04)
        with self.lock:
            self.active -= 1
        return Response(
            {
                "choices": [
                    {
                        "message": {"content": "Recognized<|im_end|>"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


json_module = json


def middle(text):
    bbox = [10, 10, 100, 30]
    return {
        "pdf_info": [
            {
                "page_size": [200, 100],
                "preproc_blocks": [
                    {
                        "type": "text",
                        "bbox": bbox,
                        "lines": [
                            {
                                "bbox": bbox,
                                "spans": [
                                    {
                                        "type": "text",
                                        "bbox": bbox,
                                        "content": text,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


class BBoxRecognitionTests(unittest.TestCase):
    def test_recognizer_bearer_header_is_opt_in(self):
        with mock.patch.dict(
            "os.environ",
            {"VLLM_API_KEY": "local-secret"},
            clear=False,
        ):
            unauthenticated = OpenAIBBoxRecognizer(
                "http://vision.test",
                "unused.pdf",
                {},
            )
            authenticated = OpenAIBBoxRecognizer(
                "http://vision.test",
                "unused.pdf",
                {"api_key_env": "VLLM_API_KEY"},
            )
            try:
                self.assertNotIn("Authorization", unauthenticated.headers)
                self.assertEqual(
                    authenticated.headers["Authorization"],
                    "Bearer local-secret",
                )
            finally:
                unauthenticated.close()
                authenticated.close()

    def test_compliant_endpoint_improves_cer_through_full_fusion_gate(self):
        from PIL import Image

        fake_httpx = StrictFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (400, 200), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "max_context_tokens": 8192,
                    "structured_output_mode": "json_schema",
                    "include_row_image": False,
                },
            )
            recognizer.httpx = fake_httpx
            try:
                fused, report = fuse_middle_json(
                    middle("B1aine"),
                    middle("B1aine"),
                    FusionSettings(bbox_recognition_enabled=True),
                    bbox_recognizer=recognizer,
                )
            finally:
                recognizer.close()

        evaluation = evaluate_bbox_recognition(
            report,
            {
                "items": [
                    {
                        "page": 0,
                        "bbox": [10, 10, 100, 30],
                        "text": "Blaine",
                    }
                ]
            },
        )
        fused_span = fused["pdf_info"][0]["preproc_blocks"][0]["lines"][0][
            "spans"
        ][0]
        self.assertEqual(fused_span["bbox"], [10, 10, 100, 30])
        self.assertEqual(report["counts"]["bbox_recognition_vlm_selected"], 1)
        self.assertTrue(report["recognition_invariants"]["bbox_unchanged"])
        self.assertTrue(report["recognition_invariants"]["table_structure_unchanged"])
        self.assertEqual(evaluation["sources"]["selected_text"]["cer"], 0.0)
        self.assertTrue(evaluation["operational_health"]["healthy"])
        self.assertTrue(evaluation["recommended_default_enabled"])

    def test_pre_rendered_page_directory_uses_middle_json_coordinates(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            page_dir = Path(temp_dir)
            page = Image.new("RGB", (400, 200), "white")
            ImageDraw.Draw(page).rectangle((100, 50, 200, 150), fill="red")
            page.save(page_dir / "page-1.png")
            page.close()
            provider = PageCropProvider(page_dir, scale=4.0)
            try:
                data_url = provider.crop_data_url(
                    0,
                    [200, 100],
                    [50, 25, 100, 75],
                    padding_ratio=0.0,
                    jpeg_quality=100,
                )
            finally:
                provider.close()

        encoded = data_url.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as crop:
            self.assertEqual(crop.size, (108, 108))
            red, green, blue = crop.convert("RGB").getpixel((54, 54))
        self.assertGreater(red, 200)
        self.assertLess(green, 50)
        self.assertLess(blue, 50)

    def test_batches_multiscale_crops_and_filters_unknown_ids(self):
        from PIL import Image

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": None,
                    "max_batch_size": 2,
                    "max_images_per_request": 12,
                    "include_row_image": True,
                    "include_column_image": True,
                    "include_table_image": True,
                    "target_render_scale": 1.0,
                    "context_render_scale": 1.0,
                    "max_tokens": 9000,
                    "context_reserve_tokens": 2048,
                    "timeout_seconds": 5,
                },
            )
            recognizer.httpx = fake_httpx
            try:
                candidates = [
                    {
                        "id": f"p0-bbox-{index}",
                        "bbox": [10, 10 + index * 30, 100, 30 + index * 30],
                        "ocr_text": "B1aine" if index == 0 else f"Value {index}",
                        "type": "table_ocr",
                        "contexts": {
                            "target": [10, 10 + index * 30, 100, 30 + index * 30],
                            "row": [5, 5 + index * 30, 200, 35 + index * 30],
                            "column": [5, 5, 110, 150],
                            "table": [0, 0, 220, 180],
                        },
                    }
                    for index in range(3)
                ]
                result = recognizer(0, [400, 300], candidates)
            finally:
                recognizer.close()

        self.assertEqual(len(fake_httpx.requests), 2)
        self.assertEqual(
            [item["id"] for item in result["items"]],
            ["p0-bbox-0", "p0-bbox-1", "p0-bbox-2"],
        )
        self.assertEqual(result["items"][0]["text"], "Blaine")
        self.assertEqual(fake_httpx.requests[0]["model"], "vision-recognizer")
        self.assertEqual(fake_httpx.requests[0]["max_tokens"], 1024)
        self.assertEqual(
            fake_httpx.requests[0]["response_format"],
            {"type": "json_object"},
        )
        first_content = fake_httpx.requests[0]["messages"][0]["content"]
        self.assertGreater(len(first_content), 3)
        first_prompt = first_content[0]["text"]
        self.assertIn('"row": "image-', first_prompt)
        self.assertIn('"column": "image-', first_prompt)
        self.assertIn('"table": "image-', first_prompt)
        self.assertTrue(
            all(batch["status"] == "partial_schema" for batch in result["batches"])
        )
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["invalid_outputs"], 2)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["batches"][0]["finish_reason"], "stop")
        self.assertEqual(result["batches"][0]["completion_tokens"], 30)

    def test_auto_protocol_uses_native_mineru_text_recognition_and_filters_small_print(self):
        from PIL import Image

        fake_httpx = NativeMinerUFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (400, 200), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": None,
                    "protocol": "auto",
                    "native_min_bbox_height": 16.0,
                    "native_max_tokens": 256,
                    "target_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": "p0-bbox-0",
                    "bbox": [10, 10, 100, 35],
                    "ocr_text": "B1aine",
                    "type": "table_ocr",
                    "contexts": {"target": [10, 10, 100, 35]},
                },
                {
                    "id": "p0-bbox-1",
                    "bbox": [10, 40, 100, 50],
                    "ocr_text": "Printed label",
                    "type": "table_ocr",
                    "contexts": {"target": [10, 40, 100, 50]},
                },
            ]
            try:
                result = recognizer(0, [200, 100], candidates)
            finally:
                recognizer.close()

        self.assertEqual(result["items"], [{"id": "p0-bbox-0", "text": "Blaine"}])
        self.assertEqual(result["native_requests"], 1)
        self.assertEqual(result["native_skipped"], 1)
        request = fake_httpx.requests[0]
        self.assertEqual(
            request["json"]["messages"][1]["content"][1]["text"],
            "\nText Recognition:",
        )
        self.assertNotIn("response_format", request["json"])
        self.assertEqual(request["json"]["max_tokens"], 256)
        self.assertEqual(
            request["headers"]["X-Custom-Hybrid-Max-Tokens"],
            "256",
        )
        self.assertEqual(
            request["headers"]["X-Custom-Hybrid-Protocol"],
            "mineru_native",
        )

    def test_structured_protocol_stops_after_repeated_invalid_schema(self):
        from PIL import Image

        fake_httpx = InvalidSchemaFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "protocol": "structured",
                    "max_images_per_request": 1,
                    "max_consecutive_invalid_schema": 2,
                    "include_row_image": False,
                    "target_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": f"p0-bbox-{index}",
                    "bbox": [10, 10 + index * 30, 100, 30 + index * 30],
                    "ocr_text": f"Value {index}",
                    "type": "text",
                    "contexts": {
                        "target": [10, 10 + index * 30, 100, 30 + index * 30]
                    },
                }
                for index in range(4)
            ]
            try:
                result = recognizer(0, [400, 300], candidates)
            finally:
                recognizer.close()

        self.assertEqual(len(fake_httpx.requests), 2)
        self.assertEqual(result["circuit_breaker_trips"], 1)
        self.assertTrue(
            any(
                batch["status"] == "structured_circuit_breaker"
                for batch in result["batches"]
            )
        )

    def test_native_cache_reuses_crop_across_recognizer_instances(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "page.png"
            cache_dir = root / "cache"
            Image.new("RGB", (200, 100), "white").save(image_path)
            config = {
                "model": "mineru-claim-forms",
                "max_context_tokens": 8192,
                "protocol": "auto",
                "native_min_bbox_height": 0,
                "native_cache_enabled": True,
                "native_cache_dir": str(cache_dir),
                "native_max_concurrency": 1,
                "target_render_scale": 1.0,
            }
            candidate = {
                "id": "p0-bbox-0",
                "bbox": [10, 10, 100, 30],
                "ocr_text": "B1aine",
                "type": "table_ocr",
                "contexts": {"target": [10, 10, 100, 30]},
            }
            first_httpx = NativeMinerUFakeHttpx()
            first = OpenAIBBoxRecognizer(
                "http://vision.test", image_path, config
            )
            first.httpx = first_httpx
            try:
                first_result = first(0, [200, 100], [candidate])
            finally:
                first.close()
            second_httpx = NativeMinerUFakeHttpx()
            second = OpenAIBBoxRecognizer(
                "http://vision.test", image_path, config
            )
            second.httpx = second_httpx
            try:
                second_result = second(0, [200, 100], [candidate])
            finally:
                second.close()

            cache_files = list(cache_dir.rglob("*.json"))

        self.assertEqual(len(first_httpx.requests), 1)
        self.assertEqual(first_result["native_cache_writes"], 1)
        self.assertEqual(second_httpx.requests, [])
        self.assertEqual(second_result["native_cache_hits"], 1)
        self.assertEqual(
            second_result["items"],
            [{"id": "p0-bbox-0", "text": "Blaine"}],
        )
        self.assertEqual(len(cache_files), 1)

    def test_native_memory_cache_honors_ttl(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "page.png"
            Image.new("RGB", (20, 20), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "max_context_tokens": 8192,
                    "native_cache_enabled": True,
                    "native_cache_dir": str(root / "cache"),
                    "native_cache_ttl_seconds": 1,
                    "target_render_scale": 1.0,
                },
            )
            try:
                self.assertTrue(recognizer._native_cache_put("a" * 64, "value"))
                recognizer._native_cache_memory["a" * 64] = (
                    time.time() - 2,
                    "stale",
                )
                recognizer._native_cache_path("a" * 64).unlink()
                self.assertIsNone(recognizer._native_cache_get("a" * 64))
            finally:
                recognizer.close()

    def test_native_requests_use_bounded_concurrency_and_page_budget(self):
        from PIL import Image, ImageDraw

        fake_httpx = ConcurrentNativeFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (400, 200), "white")
            draw = ImageDraw.Draw(image)
            for index, shade in enumerate((20, 70, 120, 170)):
                draw.rectangle(
                    [10 + index * 90, 20, 80 + index * 90, 80],
                    fill=(shade, shade, shade),
                )
            image.save(image_path)
            image.close()
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "max_context_tokens": 8192,
                    "protocol": "auto",
                    "native_min_bbox_height": 0,
                    "native_max_requests_per_page": 3,
                    "native_max_concurrency": 2,
                    "target_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": f"p0-bbox-{index}",
                    "bbox": [5 + index * 45, 5, 45 + index * 45, 45],
                    "ocr_text": f"Value {index}",
                    "type": "table_ocr",
                    "contexts": {
                        "target": [5 + index * 45, 5, 45 + index * 45, 45]
                    },
                }
                for index in range(4)
            ]
            try:
                result = recognizer(0, [200, 100], candidates)
            finally:
                recognizer.close()

        self.assertEqual(result["native_requests"], 3)
        self.assertEqual(result["native_budget_skipped"], 1)
        self.assertEqual(fake_httpx.max_active, 2)
        self.assertEqual(len(result["items"]), 3)

    def test_native_recovered_bbox_bypasses_exhausted_request_limits(self):
        from PIL import Image, ImageDraw

        fake_httpx = NativeMinerUFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).rectangle([20, 20, 100, 50], fill="black")
            image.save(image_path)
            image.close()
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "max_context_tokens": 8192,
                    "native_min_bbox_height": 0,
                    "native_max_candidates_per_page": 1,
                    "native_max_requests_per_page": 1,
                    "max_requests_per_document": 1,
                    "target_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            ordinary = {
                "id": "p0-bbox-0",
                "bbox": [10, 10, 110, 60],
                "ocr_text": "Existing",
                "recovered": False,
                "type": "table_ocr",
                "contexts": {"target": [10, 10, 110, 60]},
            }
            recovered = {
                "id": "p0-bbox-1",
                "bbox": [20, 20, 100, 50],
                "ocr_text": "",
                "recovered": True,
                "type": "table_ocr",
                "contexts": {"target": [20, 20, 100, 50]},
            }
            try:
                first = recognizer(0, [200, 100], [ordinary])
                second = recognizer(0, [200, 100], [recovered])
                malformed = dict(recovered)
                malformed["id"] = "p0-bbox-2"
                malformed["contexts"] = {"target": [20, 20]}
                third = recognizer(0, [200, 100], [malformed])
            finally:
                recognizer.close()

        self.assertEqual(first["native_requests"], 1)
        self.assertEqual(second["native_requests"], 1)
        self.assertEqual(second["native_budget_skipped"], 0)
        self.assertEqual(second["native_recovered_limit_bypasses"], 1)
        self.assertEqual(second["native_recovered_unsent"], 0)
        self.assertEqual(second["items"][0]["id"], "p0-bbox-1")
        self.assertEqual(third["native_requests"], 0)
        self.assertEqual(third["native_recovered_unsent"], 1)
        self.assertEqual(len(fake_httpx.requests), 2)

    def test_native_circuit_breaker_does_not_leave_recovered_bbox_unsent(self):
        from PIL import Image, ImageDraw

        fake_httpx = InvalidNativeMinerUFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (240, 120), "white")
            draw = ImageDraw.Draw(image)
            for index, shade in enumerate((20, 70, 120, 170)):
                draw.rectangle(
                    [12 + index * 50, 12, 48 + index * 50, 48],
                    fill=(shade, shade, shade),
                )
            image.save(image_path)
            image.close()
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "max_context_tokens": 8192,
                    "native_all_candidates": True,
                    "native_min_bbox_height": 0,
                    "native_max_consecutive_failures": 1,
                    "native_max_concurrency": 1,
                    "target_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": f"p0-bbox-{index}",
                    "bbox": [10 + index * 50, 10, 50 + index * 50, 50],
                    "ocr_text": "" if index < 3 else "Existing",
                    "recovered": index < 3,
                    "type": "table_ocr",
                    "contexts": {
                        "target": [10 + index * 50, 10, 50 + index * 50, 50]
                    },
                }
                for index in range(4)
            ]
            try:
                result = recognizer(0, [240, 120], candidates)
            finally:
                recognizer.close()

        self.assertEqual(result["native_requests"], 3)
        self.assertEqual(result["native_recovered_unsent"], 0)
        self.assertEqual(result["native_budget_skipped"], 1)
        self.assertEqual(result["circuit_breaker_trips"], 1)
        self.assertEqual(len(fake_httpx.requests), 3)
        self.assertTrue(
            any(
                batch["status"] == "native_circuit_breaker_recovery_bypass"
                for batch in result["batches"]
            )
        )

    def test_json_schema_mode_constrains_ids_and_item_count(self):
        from PIL import Image

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "max_context_tokens": 8192,
                    "structured_output_mode": "json_schema",
                    "include_row_image": False,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": "p0-bbox-0",
                    "bbox": [10, 10, 100, 30],
                    "ocr_text": "B1aine",
                    "type": "text",
                    "contexts": {"target": [10, 10, 100, 30]},
                }
            ]
            try:
                result = recognizer(0, [200, 100], candidates)
            finally:
                recognizer.close()

        response_format = fake_httpx.requests[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["items"]["minItems"], 1)
        self.assertEqual(
            schema["properties"]["items"]["items"]["properties"]["id"][
                "enum"
            ],
            ["p0-bbox-0"],
        )
        self.assertEqual(result["items"], [{"id": "p0-bbox-0", "text": "Blaine"}])
        prompt = fake_httpx.requests[0]["messages"][0]["content"][0]["text"]
        evidence = json.loads(prompt.split("evidence_data=", 1)[1])
        self.assertNotIn("id", evidence["candidates"][0])
        self.assertEqual(evidence["candidates"][0]["slot"], 0)

    def test_structured_outputs_mode_uses_vllm_native_schema(self):
        from PIL import Image

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "max_context_tokens": 8192,
                    "structured_output_mode": "structured_outputs",
                    "include_row_image": False,
                },
            )
            recognizer.httpx = fake_httpx
            candidate = {
                "id": "p0-bbox-0",
                "bbox": [10, 10, 100, 30],
                "ocr_text": "B1aine",
                "type": "text",
                "contexts": {"target": [10, 10, 100, 30]},
            }
            try:
                result = recognizer(0, [200, 100], [candidate])
            finally:
                recognizer.close()

        structured = fake_httpx.requests[0]["structured_outputs"]
        self.assertTrue(structured["disable_additional_properties"])
        self.assertTrue(structured["disable_any_whitespace"])
        self.assertEqual(
            structured["json"]["properties"]["items"]["maxItems"],
            1,
        )
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["items"], [{"id": "p0-bbox-0", "text": "Blaine"}])

    def test_regex_mode_forces_ordered_ids_and_json_strings(self):
        regex = OpenAIBBoxRecognizer._response_regex(
            ["p0-bbox-0", "p0-bbox-1"]
        )

        self.assertIsNotNone(
            __import__("re").fullmatch(
                regex,
                '{"items":[{"id":"p0-bbox-0","text":"Blaine Bai"},'
                '{"id":"p0-bbox-1","text":"20,000.00"}]}',
            )
        )
        self.assertIsNone(
            __import__("re").fullmatch(
                regex,
                '{"items":[{"id":"p0-bbox-1","text":"wrong order"},'
                '{"id":"p0-bbox-0","text":"wrong order"}]}',
            )
        )

    def test_image_budget_rebatches_instead_of_dropping_targets(self):
        from PIL import Image

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (300, 200), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "max_context_tokens": 8192,
                    "max_batch_size": 3,
                    "max_images_per_request": 1,
                    "include_row_image": False,
                    "target_render_scale": 1.0,
                    "context_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": f"p0-bbox-{index}",
                    "bbox": [10, 10 + index * 30, 100, 30 + index * 30],
                    "ocr_text": f"Value {index}",
                    "type": "text",
                    "contexts": {
                        "target": [10, 10 + index * 30, 100, 30 + index * 30]
                    },
                }
                for index in range(3)
            ]
            try:
                result = recognizer(0, [300, 200], candidates)
            finally:
                recognizer.close()

        self.assertEqual(len(fake_httpx.requests), 3)
        self.assertEqual(result["requests"], 3)
        self.assertEqual(
            [item["id"] for item in result["items"]],
            ["p0-bbox-0", "p0-bbox-1", "p0-bbox-2"],
        )

    def test_context_aware_packing_keeps_row_and_table_images_under_limit(self):
        from PIL import Image

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "max_context_tokens": 8192,
                    "max_batch_size": 8,
                    "max_images_per_request": 8,
                    "include_row_image": True,
                    "include_table_image": True,
                    "target_render_scale": 1.0,
                    "context_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": f"p0-bbox-{index}",
                    "bbox": [10, 10 + index * 30, 100, 30 + index * 30],
                    "ocr_text": f"Value {index}",
                    "type": "table_ocr",
                    "contexts": {
                        "target": [10, 10 + index * 30, 100, 30 + index * 30],
                        "row": [5, 5 + index * 30, 200, 35 + index * 30],
                        "table": [0, 0, 220, 180],
                    },
                }
                for index in range(4)
            ]
            try:
                result = recognizer(0, [400, 300], candidates)
            finally:
                recognizer.close()

        self.assertEqual(len(fake_httpx.requests), 2)
        self.assertEqual(
            [len(request["messages"][0]["content"]) - 1 for request in fake_httpx.requests],
            [7, 3],
        )
        self.assertTrue(
            all(batch["contexts_dropped"] == [] for batch in result["batches"])
        )
        for request in fake_httpx.requests:
            prompt = request["messages"][0]["content"][0]["text"]
            self.assertIn('"row": "image-', prompt)
            self.assertIn('"table": "image-', prompt)

    def test_server_image_limit_error_rebatches_and_retries(self):
        from PIL import Image

        fake_httpx = ImageLimitFakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            recognizer = OpenAIBBoxRecognizer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-recognizer",
                    "max_context_tokens": 8192,
                    "max_batch_size": 8,
                    "max_images_per_request": 24,
                    "max_image_limit_retries": 2,
                    "include_row_image": True,
                    "include_table_image": True,
                    "target_render_scale": 1.0,
                    "context_render_scale": 1.0,
                },
            )
            recognizer.httpx = fake_httpx
            candidates = [
                {
                    "id": f"p0-bbox-{index}",
                    "bbox": [10, 10 + index * 30, 100, 30 + index * 30],
                    "ocr_text": f"Value {index}",
                    "type": "table_ocr",
                    "contexts": {
                        "target": [10, 10 + index * 30, 100, 30 + index * 30],
                        "row": [5, 5 + index * 30, 200, 35 + index * 30],
                        "table": [0, 0, 220, 180],
                    },
                }
                for index in range(4)
            ]
            try:
                result = recognizer(0, [400, 300], candidates)
            finally:
                recognizer.close()

        self.assertEqual(len(fake_httpx.requests), 3)
        self.assertGreater(
            len(fake_httpx.requests[0]["messages"][0]["content"]) - 1,
            8,
        )
        self.assertTrue(
            all(
                len(request["messages"][0]["content"]) - 1 <= 8
                for request in fake_httpx.requests[1:]
            )
        )
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["rebatches"], 1)
        self.assertEqual(result["errors"], 0)
        rebatch = next(
            batch
            for batch in result["batches"]
            if batch["status"] == "image_limit_rebatch"
        )
        self.assertEqual(rebatch["attempted_images"], 9)
        self.assertEqual(rebatch["server_image_limit"], 8)
        successful = [batch for batch in result["batches"] if "ids" in batch]
        self.assertTrue(all(batch["contexts_dropped"] == [] for batch in successful))


if __name__ == "__main__":
    unittest.main()

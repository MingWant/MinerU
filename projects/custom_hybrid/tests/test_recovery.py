import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.recovery import OpenAIBBoxRecoveryReviewer


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHttpx:
    def __init__(self):
        self.requests = []

    def get(self, *_args, **_kwargs):
        return Response({"data": [{"id": "vision-reviewer"}]})

    def post(self, *_args, json=None, **_kwargs):
        self.requests.append(json)
        schema = json["response_format"]["json_schema"]["schema"]
        cell_id = schema["properties"]["items"]["items"]["properties"][
            "cell_id"
        ]["enum"][0]
        return Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "items": [
                                        {
                                            "action": "add",
                                            "cell_id": cell_id,
                                            "target_id": "",
                                            "bbox": [50, 50, 450, 450],
                                            "confidence": 0.96,
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            }
        )


json_module = json


class BBoxRecoveryReviewerTests(unittest.TestCase):
    def test_shared_page_provider_is_not_closed_by_reviewer(self):
        provider = mock.Mock()
        reviewer = OpenAIBBoxRecoveryReviewer(
            "http://vision.test",
            "unused.pdf",
            {"model": "mineru-claim-forms"},
            page_provider=provider,
        )

        reviewer.close()

        provider.close.assert_not_called()

    def test_reviews_only_visible_missing_content_and_refines_to_pixels(self):
        from PIL import Image, ImageDraw

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (400, 200), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([40, 30, 140, 60], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-reviewer",
                    "render_scale": 2.0,
                    "min_ink_ratio": 0.002,
                    "min_uncovered_ink_ratio": 0.15,
                    "local_missing_enabled": False,
                },
            )
            reviewer.httpx = fake_httpx
            tables = [
                {
                    "id": "p0-table-0",
                    "bbox": [0, 0, 200, 100],
                    "cells": [
                        {
                            "id": "p0-t0-c0",
                            "bbox": [0, 0, 100, 50],
                            "text": "",
                            "existing": [],
                            "reasons": ["missing_content_bbox"],
                        },
                        {
                            "id": "p0-t0-c1",
                            "bbox": [100, 0, 200, 50],
                            "text": "",
                            "existing": [],
                            "reasons": ["missing_content_bbox"],
                        },
                    ],
                }
            ]
            try:
                result = reviewer(0, [200, 100], tables)
            finally:
                reviewer.close()

        self.assertEqual(result["tables_reviewed"], 1)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(len(result["items"]), 1)
        recovered = result["items"][0]
        self.assertEqual(recovered["cell_id"], "p0-t0-c0")
        self.assertGreaterEqual(recovered["bbox"][0], 18)
        self.assertLessEqual(recovered["bbox"][2], 72)
        self.assertGreaterEqual(recovered["bbox"][1], 13)
        self.assertLessEqual(recovered["bbox"][3], 32)
        request = fake_httpx.requests[0]
        self.assertEqual(len(request["messages"][0]["content"]), 2)
        prompt = request["messages"][0]["content"][0]["text"]
        evidence = json.loads(prompt.split("evidence_data=", 1)[1])
        self.assertEqual(len(evidence["cells"]), 1)
        self.assertEqual(evidence["cells"][0]["cell_id"], "p0-t0-c0")

    def test_visible_missing_content_uses_local_pixel_recovery_without_vlm(self):
        from PIL import Image, ImageDraw

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).rectangle([20, 20, 80, 40], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "local_missing_enabled": True,
                },
            )
            reviewer.httpx = fake_httpx
            try:
                result = reviewer(
                    0,
                    [200, 100],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 200, 100],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 100, 50],
                                    "text": "",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                },
                                {
                                    "id": "p0-t0-c1",
                                    "bbox": [100, 0, 200, 50],
                                    "text": "Already covered",
                                    "existing": [
                                        {
                                            "id": "p0-t0-c1-b0",
                                            "bbox": [110, 10, 190, 40],
                                            "text": "Already covered",
                                        }
                                    ],
                                    "reasons": [],
                                }
                            ],
                        }
                    ],
                )
            finally:
                reviewer.close()

        self.assertEqual(fake_httpx.requests, [])
        self.assertEqual(result["requests"], 0)
        self.assertEqual(result["local_proposals"], 1)
        self.assertEqual(result["pixel_cells_analyzed"], 1)
        self.assertEqual(result["pixel_cells_skipped"], 1)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["recovery_source"], "local_pixel_ink")
        self.assertEqual(result["batches"][0]["status"], "local_pixel_recovery")

    def test_blank_cells_do_not_trigger_vlm_request(self):
        from PIL import Image

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {"model": "vision-reviewer", "render_scale": 1.0},
            )
            reviewer.httpx = fake_httpx
            try:
                result = reviewer(
                    0,
                    [200, 100],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 200, 100],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 100, 50],
                                    "text": "",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                }
                            ],
                        }
                    ],
                )
            finally:
                reviewer.close()

        self.assertEqual(fake_httpx.requests, [])
        self.assertEqual(result["tables_reviewed"], 0)
        self.assertEqual(result["items"], [])

    def test_table_budget_blocks_suspicious_review_request(self):
        from PIL import Image, ImageDraw

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).rectangle([20, 20, 80, 40], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-reviewer",
                    "render_scale": 1.0,
                    "max_tables_per_document": 0,
                    "local_missing_enabled": False,
                },
            )
            reviewer.httpx = fake_httpx
            try:
                result = reviewer(
                    0,
                    [200, 100],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 200, 100],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 100, 50],
                                    "text": "",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                }
                            ],
                        }
                    ],
                )
            finally:
                reviewer.close()

        self.assertEqual(fake_httpx.requests, [])
        self.assertEqual(result["table_budget_skips"], 1)
        self.assertEqual(result["batches"][0]["status"], "table_limit")

    def test_document_proposal_budget_blocks_vlm_request(self):
        from PIL import Image, ImageDraw

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).rectangle([20, 20, 80, 40], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "vision-reviewer",
                    "render_scale": 1.0,
                    "max_proposals_per_document": 0,
                },
            )
            reviewer.httpx = fake_httpx
            try:
                result = reviewer(
                    0,
                    [200, 100],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 200, 100],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 100, 50],
                                    "text": "",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                }
                            ],
                        }
                    ],
                )
            finally:
                reviewer.close()

        self.assertEqual(fake_httpx.requests, [])
        self.assertEqual(result["proposal_budget_skips"], 1)
        self.assertEqual(result["batches"][0]["status"], "proposal_limit")


if __name__ == "__main__":
    unittest.main()

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
    def test_recovery_bearer_header_is_opt_in(self):
        with mock.patch.dict(
            "os.environ",
            {"VLLM_API_KEY": "local-secret"},
            clear=False,
        ):
            unauthenticated = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                "unused.pdf",
                {},
            )
            authenticated = OpenAIBBoxRecoveryReviewer(
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
        self.assertEqual(result["pixel_cells_analyzed"], 2)
        self.assertEqual(result["pixel_cells_skipped"], 0)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["recovery_source"], "local_pixel_ink")
        self.assertEqual(result["batches"][0]["status"], "local_pixel_recovery")

    def test_local_pixel_recovery_adds_uncovered_line_in_partially_boxed_cell(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (160, 100), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 20, 75, 30], fill="black")
            draw.rectangle([25, 55, 105, 65], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "local_uncovered_enabled": True,
                    "checkbox_recovery_enabled": False,
                },
            )
            try:
                result = reviewer(
                    0,
                    [160, 100],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 150, 90],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 140, 80],
                                    "text": "Already boxed",
                                    "existing": [
                                        {
                                            "id": "p0-t0-c0-b0",
                                            "bbox": [18, 18, 77, 32],
                                            "text": "Already boxed",
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

        uncovered = [
            item
            for item in result["items"]
            if item.get("recovery_source") == "local_uncovered_pixel_ink"
        ]
        self.assertEqual(len(uncovered), 1)
        self.assertGreater(uncovered[0]["bbox"][1], 45)
        self.assertIn("uncovered_ink", uncovered[0]["recovery_reasons"])

    def test_table_orphan_recovery_accepts_short_amount(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).text((155, 60), "50.00", fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "checkbox_recovery_enabled": False,
                    "table_fringe_recovery_enabled": False,
                },
            )
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
                                    "bbox": [0, 0, 150, 45],
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

        orphan = [
            item for item in result["items"] if item.get("action") == "add_orphan"
        ]
        self.assertEqual(len(orphan), 1)
        self.assertLess(orphan[0]["bbox"][2] - orphan[0]["bbox"][0], 40)

    def test_table_fringe_recovery_masks_existing_page_text(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (220, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.text((10, 76), "Already OCR", fill="black")
            draw.text((105, 76), "Date Admitted", fill="black")
            draw.text((105, 92), "Date Discharged", fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "checkbox_recovery_enabled": False,
                    "table_fringe_bottom_extension": 60.0,
                },
            )
            try:
                result = reviewer(
                    0,
                    [220, 140],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 220, 60],
                            "page_existing": [
                                {
                                    "bbox": [8, 73, 78, 86],
                                    "text": "Already OCR",
                                }
                            ],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 220, 55],
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

        fringe = [
            item for item in result["items"] if item.get("action") == "add_fringe"
        ]
        self.assertGreaterEqual(len(fringe), 2)
        self.assertTrue(all(item["bbox"][0] > 90 for item in fringe))
        self.assertEqual(result["fringe_proposals"], len(fringe))

    def test_checkbox_recovery_uses_existing_checkbox_label_bbox(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 20, 31, 31], outline="black", width=2)
            draw.text((40, 21), "Covered option", fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "checkbox_recovery_enabled": True,
                    "checkbox_merge_label_enabled": False,
                    "checkbox_min_size": 8.0,
                    "checkbox_max_size": 18.0,
                },
            )
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
                                    "bbox": [0, 0, 190, 60],
                                    "text": "Covered option",
                                    "existing": [
                                        {
                                            "id": "p0-t0-c0-b0",
                                            "bbox": [18, 18, 120, 35],
                                            "text": "Covered option",
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

        self.assertEqual(result["checkbox_candidates"], 1)
        self.assertEqual(result["checkbox_proposals"], 0)
        self.assertFalse(
            any(item.get("action") == "add_checkbox" for item in result["items"])
        )

    def test_checkbox_recovery_merges_separate_right_label_bbox(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (200, 100), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 20, 31, 31], outline="black", width=2)
            draw.text((40, 21), "Separate option", fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "local_uncovered_enabled": False,
                    "checkbox_recovery_enabled": True,
                    "checkbox_min_size": 8.0,
                    "checkbox_max_size": 18.0,
                },
            )
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
                                    "bbox": [0, 0, 190, 60],
                                    "text": "Separate option",
                                    "existing": [
                                        {
                                            "id": "p0-t0-c0-b0",
                                            "bbox": [39, 18, 120, 35],
                                            "text": "Separate option",
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

        merged = [
            item
            for item in result["items"]
            if item.get("action") == "merge_checkbox"
        ]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["target_id"], "p0-t0-c0-b0")
        self.assertLessEqual(merged[0]["bbox"][0], 20)
        self.assertGreaterEqual(merged[0]["bbox"][2], 120)
        self.assertEqual(result["checkbox_merged"], 1)

    def test_local_pixel_recovery_removes_long_table_rules_from_content_bbox(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (240, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.line([100, 10, 100, 120], fill="black", width=2)
            draw.line([10, 70, 220, 70], fill="black", width=2)
            draw.rectangle([30, 35, 80, 50], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "cell_horizontal_rule_ratio": 0.6,
                    "cell_vertical_rule_ratio": 0.6,
                    "cell_rule_min_length": 18.0,
                },
            )
            try:
                result = reviewer(
                    0,
                    [240, 140],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [10, 10, 220, 120],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [10, 10, 220, 120],
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

        self.assertEqual(result["requests"], 0)
        self.assertEqual(len(result["items"]), 1)
        bbox = result["items"][0]["bbox"]
        self.assertLess(bbox[2], 90)
        self.assertGreater(bbox[1], 25)
        self.assertLess(bbox[3], 60)

    def test_local_pixel_recovery_removes_long_diagonal_rules_across_cells(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (240, 180), "white")
            draw = ImageDraw.Draw(image)
            draw.line([30, 20, 210, 160], fill="black", width=2)
            draw.rectangle([120, 35, 180, 50], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "table_diagonal_rule_min_length": 60.0,
                    "table_diagonal_rule_max_gap": 4.0,
                    "table_orphan_recovery_enabled": False,
                    "checkbox_recovery_enabled": False,
                },
            )
            try:
                result = reviewer(
                    0,
                    [240, 180],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [10, 10, 220, 170],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [10, 10, 220, 90],
                                    "text": "",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                },
                                {
                                    "id": "p0-t0-c1",
                                    "bbox": [10, 90, 220, 170],
                                    "text": "",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                },
                            ],
                        }
                    ],
                )
            finally:
                reviewer.close()

        self.assertGreaterEqual(result["diagonal_rules_removed"], 1)
        self.assertEqual(result["requests"], 0)
        self.assertEqual(len(result["items"]), 1)
        bbox = result["items"][0]["bbox"]
        self.assertGreater(bbox[0], 100)
        self.assertLess(bbox[2], 195)
        self.assertGreater(bbox[1], 25)
        self.assertLess(bbox[3], 60)

    def test_local_pixel_recovery_splits_multiline_cell_into_tight_boxes(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (120, 100), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([20, 20, 80, 30], fill="black")
            draw.rectangle([20, 55, 90, 65], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "cell_line_gap": 1.5,
                },
            )
            try:
                result = reviewer(
                    0,
                    [120, 100],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 120, 90],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 110, 80],
                                    "text": "Line one\nLine two",
                                    "existing": [],
                                    "reasons": ["missing_content_bbox"],
                                }
                            ],
                        }
                    ],
                )
            finally:
                reviewer.close()

        self.assertEqual(len(result["items"]), 2)
        first, second = result["items"]
        self.assertLess(first["bbox"][3], second["bbox"][1])
        self.assertLess(first["bbox"][3] - first["bbox"][1], 20)
        self.assertLess(second["bbox"][3] - second["bbox"][1], 20)

    def test_local_pixel_recovery_rejects_thin_dashed_separator(self):
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (120, 60), "white")
            draw = ImageDraw.Draw(image)
            for left in range(10, 105, 12):
                draw.line([left, 30, min(left + 6, 110), 30], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "cell_line_min_dark_height": 3.0,
                },
            )
            try:
                result = reviewer(
                    0,
                    [120, 60],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [0, 0, 120, 60],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [0, 0, 120, 60],
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

        self.assertEqual(result["items"], [])
        self.assertEqual(result["requests"], 0)

    def test_table_orphan_recovery_finds_text_below_all_cells(self):
        from PIL import Image, ImageDraw

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (240, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([10, 10, 230, 130], outline="black")
            draw.line([10, 60, 230, 60], fill="black")
            draw.text((20, 82), "Delivery Option", fill="black")
            draw.text((20, 103), "Via Consultant", fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "table_orphan_recovery_enabled": True,
                },
            )
            reviewer.httpx = fake_httpx
            try:
                result = reviewer(
                    0,
                    [240, 140],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [10, 10, 230, 130],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [10, 10, 230, 60],
                                    "row_end": 0,
                                    "text": "Covered header",
                                    "existing": [
                                        {
                                            "id": "p0-t0-c0-b0",
                                            "bbox": [20, 20, 120, 32],
                                            "text": "Covered header",
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
        self.assertEqual(result["orphan_tables_analyzed"], 1)
        self.assertEqual(result["orphan_proposals"], 2)
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(all(item["action"] == "add_orphan" for item in result["items"]))
        self.assertTrue(all(item["bbox"][1] > 60 for item in result["items"]))

    def test_checkbox_recovery_adds_missing_checked_and_unchecked_boxes(self):
        from PIL import Image, ImageDraw

        fake_httpx = FakeHttpx()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (240, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle([10, 10, 230, 130], outline="black")
            draw.rectangle([20, 25, 32, 37], outline="black", width=2)
            draw.text((42, 25), "Already boxed", fill="black")
            draw.rectangle([20, 60, 32, 72], outline="black", width=2)
            draw.text((42, 60), "Unchecked option", fill="black")
            draw.rectangle([20, 95, 32, 107], outline="black", width=2)
            draw.line([23, 100, 26, 104, 30, 98], fill="black", width=2)
            draw.text((42, 95), "Checked option", fill="black")
            # Square-looking capital glyphs can satisfy the contour and
            # four-side checks. Keep their following stroke close enough to
            # model the D/Q-to-next-letter spacing seen in real forms.
            draw.rectangle([150, 60, 162, 72], outline="black", width=2)
            draw.rectangle([164, 60, 170, 72], fill="black")
            draw.rectangle([150, 95, 162, 107], outline="black", width=2)
            draw.line([160, 105, 165, 110], fill="black", width=2)
            draw.rectangle([164, 95, 170, 107], fill="black")
            image.save(image_path)
            image.close()
            reviewer = OpenAIBBoxRecoveryReviewer(
                "http://vision.test",
                image_path,
                {
                    "model": "mineru-claim-forms",
                    "render_scale": 1.0,
                    "table_orphan_recovery_enabled": False,
                    "checkbox_recovery_enabled": True,
                    "checkbox_merge_label_enabled": False,
                    "checkbox_min_size": 8.0,
                    "checkbox_max_size": 18.0,
                },
            )
            reviewer.httpx = fake_httpx
            try:
                result = reviewer(
                    0,
                    [240, 140],
                    [
                        {
                            "id": "p0-table-0",
                            "bbox": [10, 10, 230, 130],
                            "cells": [
                                {
                                    "id": "p0-t0-c0",
                                    "bbox": [10, 10, 230, 130],
                                    "row_end": 0,
                                    "text": "Options",
                                    "existing": [
                                        {
                                            "id": "p0-t0-c0-b0",
                                            "bbox": [19, 24, 33, 38],
                                            "text": "☐",
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

        checkbox_items = [
            item for item in result["items"] if item["action"] == "add_checkbox"
        ]
        self.assertEqual(fake_httpx.requests, [])
        self.assertEqual(result["checkbox_tables_analyzed"], 1)
        self.assertEqual(len(checkbox_items), 2)
        self.assertEqual(
            {item["checkbox_state"] for item in checkbox_items},
            {"checked", "unchecked"},
        )
        self.assertTrue(all(item["bbox"][1] >= 60 for item in checkbox_items))
        self.assertTrue(all(item["bbox"][0] < 100 for item in checkbox_items))

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

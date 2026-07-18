import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from projects.custom_hybrid.form_detection import (
    FormDetectionSettings,
    annotate_form_regions,
    detect_form_regions,
)


def _ruled_form_image(width=600, height=800):
    image = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (550, 760), 0, 3)
    for y in range(120, 720, 80):
        cv2.line(image, (70, y), (530, y), 0, 3)
    return image


class FormDetectionTests(unittest.TestCase):
    def test_detects_synthetic_ruled_form(self):
        regions = detect_form_regions(
            _ruled_form_image(),
            [300, 400],
        )

        self.assertEqual(len(regions), 1)
        self.assertGreaterEqual(regions[0]["confidence"], 0.9)
        self.assertGreaterEqual(
            regions[0]["evidence"]["horizontal_rules"],
            3,
        )
        self.assertEqual(regions[0]["evidence"]["vertical_borders"], 2)
        self.assertAlmostEqual(regions[0]["bbox"][0], 25, delta=2)
        self.assertAlmostEqual(regions[0]["bbox"][1], 20, delta=2)

    def test_rejects_plain_text_page(self):
        image = np.full((800, 600), 255, dtype=np.uint8)
        for row, text in enumerate(("Plain text", "without a grid", "or form rules")):
            cv2.putText(
                image,
                text,
                (70, 150 + row * 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                0,
                2,
                cv2.LINE_AA,
            )

        self.assertEqual(detect_form_regions(image, [300, 400]), [])

    def test_excludes_region_already_covered_by_mineru_table(self):
        regions = detect_form_regions(
            _ruled_form_image(),
            [300, 400],
            existing_table_bboxes=[[20, 15, 280, 385]],
        )

        self.assertEqual(regions, [])

    def test_annotation_preserves_original_layout_and_table_html(self):
        middle = {
            "pdf_info": [
                {
                    "page_size": [300, 400],
                    "preproc_blocks": [
                        {
                            "type": "text",
                            "bbox": [10, 10, 100, 30],
                            "lines": [],
                        },
                        {
                            "type": "table",
                            "bbox": [20, 300, 280, 390],
                            "blocks": [
                                {
                                    "type": "table_body",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "type": "table",
                                                    "html": "<table><tr><td>A</td></tr></table>",
                                                }
                                            ]
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                    "para_blocks": [{"type": "text", "lines": []}],
                }
            ]
        }
        original_blocks = copy.deepcopy(
            (
                middle["pdf_info"][0]["preproc_blocks"],
                middle["pdf_info"][0]["para_blocks"],
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            source.write_bytes(b"placeholder")
            with mock.patch(
                "projects.custom_hybrid.form_detection.render_document_pages",
                return_value=iter([_ruled_form_image()]),
            ):
                report = annotate_form_regions(
                    middle,
                    source,
                    FormDetectionSettings(),
                )

        self.assertEqual(report["detected_pages"], [0])
        self.assertTrue(report["changed"])
        self.assertEqual(
            (
                middle["pdf_info"][0]["preproc_blocks"],
                middle["pdf_info"][0]["para_blocks"],
            ),
            original_blocks,
        )
        self.assertEqual(len(middle["pdf_info"][0]["form_regions"]), 1)


if __name__ == "__main__":
    unittest.main()

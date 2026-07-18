import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from projects.custom_hybrid.form_segmentation import (
    FormSegmentationSettings,
    TextSpanGeometry,
    _add_ocr_gap_boundaries,
    _merge_boundary_intervals_without_vertical_support,
    _tiled_column_intervals,
    annotate_form_structure,
    segment_form_region,
)


def _blank_form(width=600, height=800):
    image = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (550, 760), 0, 3)
    return image


class FormSegmentationTests(unittest.TestCase):
    def test_columns_tile_the_complete_parent_row_without_gaps(self):
        result = _tiled_column_intervals(
            [[10.0, 50.0], [49.0, 90.0]],
            100.0,
        )

        self.assertEqual(result[0][0], 0.0)
        self.assertEqual(result[0][1], result[1][0])
        self.assertEqual(result[-1][1], 100.0)

    def test_short_rule_bounded_band_is_not_split_by_ocr_gap(self):
        boundaries = [
            {"y": 0.0, "source": "rule", "strength": 1.0},
            {"y": 80.0, "source": "rule", "strength": 1.0},
        ]
        pixel_spans = [
            {"bbox": [10.0, 8.0, 90.0, 20.0]},
            {"bbox": [10.0, 60.0, 90.0, 72.0]},
        ]

        result = _add_ocr_gap_boundaries(
            boundaries,
            pixel_spans,
            1.0,
            FormSegmentationSettings(),
        )

        self.assertEqual(result, boundaries)

    def test_boundary_fragments_require_local_vertical_rule_to_stay_split(self):
        intervals = [[20.0, 150.0], [180.0, 300.0], [320.0, 480.0]]
        vertical_rules = [(309, 0, 3, 100)]

        result = _merge_boundary_intervals_without_vertical_support(
            intervals,
            vertical_rules,
            0.0,
            100.0,
            500,
            FormSegmentationSettings(),
        )

        self.assertEqual(result, [[20.0, 300.0], [320.0, 480.0]])

    def test_segments_synthetic_ruled_grid_into_cells(self):
        image = _blank_form()
        cv2.line(image, (50, 400), (550, 400), 0, 3)
        cv2.line(image, (300, 40), (300, 760), 0, 3)

        rows, cells = segment_form_region(
            image,
            [300, 400],
            {"bbox": [25, 20, 275, 380]},
            [],
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cells), 4)
        self.assertEqual({cell["kind"] for cell in cells}, {"grid_cell"})
        self.assertEqual(
            [(cell["row_index"], cell["column_index"]) for cell in cells],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )

    def test_handwriting_crossing_cell_boundary_expands_recognition_crop(self):
        image = np.full((400, 600), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 20), (550, 380), 0, 3)
        cv2.line(image, (300, 20), (300, 380), 0, 3)
        cv2.line(image, (220, 180), (330, 220), 0, 4)

        _rows, cells = segment_form_region(
            image,
            [300, 200],
            {"bbox": [25, 10, 275, 190]},
            [],
        )

        left_cell = cells[0]
        self.assertTrue(left_cell["recognition_overflow"])
        self.assertGreater(
            left_cell["recognition_bbox"][2],
            left_cell["bbox"][2],
        )
        self.assertEqual(left_cell["bbox"][2], cells[1]["bbox"][0])

    def test_segments_sparse_form_into_semantic_rows(self):
        image = _blank_form()
        for y in (220, 420, 610):
            cv2.line(image, (70, y), (530, y), 0, 3)

        rows, cells = segment_form_region(
            image,
            [300, 400],
            {"bbox": [25, 20, 275, 380]},
            [],
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(cells), 4)
        self.assertEqual({cell["kind"] for cell in cells}, {"semantic_row"})

    def test_large_ocr_gap_splits_unruled_form_band(self):
        image = _blank_form()
        spans = [
            TextSpanGeometry((40, 60, 180, 80), "First question", 0),
            TextSpanGeometry((40, 250, 180, 270), "Second question", 1),
        ]

        rows, cells = segment_form_region(
            image,
            [300, 400],
            {"bbox": [25, 20, 275, 380]},
            spans,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(row["bottom_boundary"] == "ocr_gap" for row in rows[:1]))

    def test_underlined_columns_keep_header_and_create_field_cells(self):
        image = _blank_form()
        cv2.line(image, (80, 700), (280, 700), 0, 3)
        cv2.line(image, (320, 700), (520, 700), 0, 3)
        spans = [
            TextSpanGeometry((35, 60, 265, 80), "Full width question", 0),
            TextSpanGeometry((45, 90, 130, 110), "Left field", 1),
            TextSpanGeometry((170, 90, 255, 110), "Right field", 2),
        ]

        _rows, cells = segment_form_region(
            image,
            [300, 400],
            {"bbox": [25, 20, 275, 380]},
            spans,
        )

        kinds = [cell["kind"] for cell in cells]
        self.assertIn("row_header", kinds)
        self.assertEqual(kinds.count("field_cell"), 2)

    def test_combined_annotation_preserves_original_mineru_structures(self):
        middle = {
            "pdf_info": [
                {
                    "page_size": [300, 400],
                    "preproc_blocks": [
                        {
                            "type": "text",
                            "bbox": [35, 60, 180, 80],
                            "lines": [
                                {
                                    "spans": [
                                        {
                                            "type": "text",
                                            "bbox": [35, 60, 180, 80],
                                            "content": "Question",
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                    "para_blocks": [],
                }
            ]
        }
        original_blocks = copy.deepcopy(middle["pdf_info"][0]["preproc_blocks"])
        image = _blank_form()
        for y in (220, 420, 610):
            cv2.line(image, (70, y), (530, y), 0, 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            source.write_bytes(b"placeholder")
            with mock.patch(
                "projects.custom_hybrid.form_segmentation.render_document_pages",
                return_value=iter([image]),
            ):
                report = annotate_form_structure(middle, source)

        self.assertEqual(
            middle["pdf_info"][0]["preproc_blocks"],
            original_blocks,
        )
        self.assertTrue(middle["pdf_info"][0]["form_regions"])
        self.assertTrue(middle["pdf_info"][0]["form_rows"])
        self.assertTrue(middle["pdf_info"][0]["form_cells"])
        self.assertTrue(report["changed"])


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.semantic_markdown_benchmark import (
    benchmark_semantic_markdown,
)


class SemanticMarkdownBenchmarkTests(unittest.TestCase):
    def test_directory_replay_summarizes_documents_and_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, text in (("first", "Alpha"), ("second", "Beta")):
                (root / f"{name}_middle.json").write_text(
                    json.dumps(
                        {
                            "pdf_info": [
                                {
                                    "page_size": [200, 300],
                                    "preproc_blocks": [
                                        {
                                            "type": "text",
                                            "bbox": [10, 10, 100, 30],
                                            "lines": [
                                                {
                                                    "bbox": [10, 10, 100, 30],
                                                    "spans": [
                                                        {
                                                            "type": "text",
                                                            "bbox": [10, 10, 100, 30],
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
                    ),
                    encoding="utf-8",
                )

            report = benchmark_semantic_markdown([root])

            self.assertEqual(report["summary"]["documents"], 2)
            self.assertEqual(report["summary"]["pages"], 2)
            self.assertEqual(report["summary"]["pages_emitted"], 2)
            self.assertEqual(report["summary"]["text_bearing_pages"], 2)
            self.assertEqual(report["summary"]["text_bearing_pages_emitted"], 2)
            self.assertEqual(report["summary"]["unmatched_source_records"], 0)
            self.assertEqual(report["summary"]["region_ordered_pages"], 0)
            self.assertEqual(report["summary"]["ownership_suppressions"], 0)
            self.assertEqual(
                report["summary"]["output_duplicate_suppressions"],
                0,
            )
            self.assertEqual(report["summary"]["potential_duplicate_groups"], 0)
            self.assertEqual(report["failures"], [])
            self.assertTrue(
                all(
                    document["unmatched_source_trace"] == []
                    for document in report["documents"]
                )
            )


if __name__ == "__main__":
    unittest.main()

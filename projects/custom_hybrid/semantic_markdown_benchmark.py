"""Batch replay fused middle JSON files through semantic Markdown v5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.semantic_markdown import (
    generate_semantic_markdown,
    generate_semantic_markdown_report,
)


def _middle_files(inputs: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in inputs:
        if path.is_file() and path.name.endswith("_middle.json"):
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*_middle.json"))
    return sorted(files)


def benchmark_semantic_markdown(inputs: Sequence[Path]) -> dict[str, Any]:
    documents = []
    failures = []
    for path in _middle_files(inputs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("middle JSON root must be an object")
            markdown = generate_semantic_markdown(payload)
            report = generate_semantic_markdown_report(payload, markdown)
            documents.append(
                {
                    "path": str(path),
                    **{key: value for key, value in report.items() if key != "source_trace"},
                    "unmatched_source_trace": [
                        record
                        for record in report["source_trace"]
                        if record["status"] == "unmatched"
                    ],
                }
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return {
        "documents": documents,
        "failures": failures,
        "summary": {
            "documents": len(documents),
            "failures": len(failures),
            "pages": sum(item["pages"] for item in documents),
            "pages_emitted": sum(item["pages_emitted"] for item in documents),
            "text_bearing_pages": sum(
                item["text_bearing_pages"] for item in documents
            ),
            "text_bearing_pages_emitted": sum(
                item["text_bearing_pages_emitted"] for item in documents
            ),
            "unmatched_source_records": sum(
                item["unmatched_source_records"] for item in documents
            ),
            "fragment_heavy_pages": sum(
                len(item["fragment_heavy_pages"]) for item in documents
            ),
            "unstructured_table_fallback_blocks": sum(
                item["unstructured_table_fallback_blocks"] for item in documents
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = benchmark_semantic_markdown(args.inputs)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

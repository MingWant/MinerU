"""Run a reviewed bbox subset through the configured recognizer for A/B evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.fusion import (
    FusionSettings,
    _page_recognition_invariant_snapshot,
    apply_bbox_recognition,
    collect_table_ocr_lines,
    collect_text_lines,
)
from projects.custom_hybrid.recognition import OpenAIBBoxRecognizer
from projects.custom_hybrid.recognition_eval import evaluate_bbox_recognition


def _bbox_key(page: int, bbox: Any) -> tuple[int, tuple[float, ...]] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return page, tuple(round(float(value), 3) for value in bbox)
    except (TypeError, ValueError):
        return None


def run_trial(
    config: Mapping[str, Any],
    middle: Mapping[str, Any],
    document_path: str | Path,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    fusion_config = config.get("fusion", {})
    recognizer_config = fusion_config.get("recognizer", {})
    if not recognizer_config.get("enabled", False):
        raise ValueError("fusion.recognizer.enabled must be true for a recognition trial")
    base_url = recognizer_config.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("fusion.recognizer.base_url is required for a standalone trial")
    pages = middle.get("pdf_info")
    if not isinstance(pages, list):
        raise ValueError("middle JSON must contain pdf_info")
    requested = {
        key
        for item in reference.get("items", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("page"), int)
        and (key := _bbox_key(item["page"], item.get("bbox"))) is not None
    }
    recognizer = OpenAIBBoxRecognizer(
        base_url,
        document_path,
        recognizer_config,
    )
    settings = FusionSettings.from_mapping(fusion_config)
    decisions = []
    batches = []
    counts = {
        "candidates": 0,
        "requests": 0,
        "responses": 0,
        "vlm_selected": 0,
        "ocr_kept": 0,
        "invalid_outputs": 0,
        "errors": 0,
        "high_risk_fallbacks": 0,
        "script_guard_fallbacks": 0,
        "protocol_echoes": 0,
        "batch_quality_fallbacks": 0,
        "empty_ocr_recoveries": 0,
        "empty_ocr_context_fallbacks": 0,
        "empty_ocr_density_fallbacks": 0,
        "empty_ocr_quality_fallbacks": 0,
    }
    bbox_unchanged = True
    table_structure_unchanged = True
    try:
        for page_index, page in enumerate(pages):
            page_lines = collect_text_lines(page, page_index)
            page_lines.extend(collect_table_ocr_lines(page, page_index))
            selected_lines = [
                line
                for line in page_lines
                if (page_index, tuple(round(value, 3) for value in line.bbox))
                in requested
            ]
            if not selected_lines:
                continue
            invariant_before = _page_recognition_invariant_snapshot(page)
            page_stats, page_decisions, page_batches = apply_bbox_recognition(
                page_index,
                page.get("page_size", [0, 0]),
                selected_lines,
                settings,
                recognizer,
            )
            invariant_after = _page_recognition_invariant_snapshot(page)
            bbox_unchanged &= invariant_before[0] == invariant_after[0]
            table_structure_unchanged &= invariant_before[1] == invariant_after[1]
            for key in counts:
                counts[key] += page_stats[key]
            decisions.extend(page_decisions)
            batches.extend(page_batches)
    finally:
        recognizer.close()
    report = {
        "version": 1,
        "counts": counts,
        "recognition_invariants": {
            "enabled": True,
            "bbox_unchanged": bbox_unchanged,
            "table_structure_unchanged": table_structure_unchanged,
        },
        "recognition_batches": batches,
        "decisions": decisions,
    }
    return {
        "report": report,
        "evaluation": evaluate_bbox_recognition(report, reference),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bbox recognizer A/B trial")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--middle", required=True, type=Path)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded_config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_trial(
        (
            loaded_config
            if "fusion" in loaded_config
            else {"fusion": {"recognizer": loaded_config}}
        ),
        json.loads(args.middle.read_text(encoding="utf-8")),
        args.document,
        json.loads(args.reference.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["evaluation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

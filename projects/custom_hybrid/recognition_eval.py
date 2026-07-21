"""Ground-truth A/B evaluation for bbox-conditioned recognition reports."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.fusion import (
    FusionSettings,
    TextLine,
    apply_recognition_batch_quality_guard,
    recognition_quality_passed_ids,
    select_bbox_recognition_candidate,
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _bbox_key(page: Any, bbox: Any) -> tuple[int, tuple[float, ...]] | None:
    if not isinstance(page, int) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        normalized_bbox = tuple(round(float(value), 3) for value in bbox)
    except (TypeError, ValueError):
        return None
    return page, normalized_bbox


def _recognition_count(
    fusion_report: Mapping[str, Any],
    name: str,
) -> int | None:
    counts = fusion_report.get("counts")
    if not isinstance(counts, Mapping):
        return None
    for key in (name, f"bbox_recognition_{name}"):
        value = counts.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _operational_health(fusion_report: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "candidates",
        "requests",
        "responses",
        "invalid_outputs",
        "errors",
        "protocol_echoes",
        "batch_quality_fallbacks",
        "script_guard_fallbacks",
        "empty_ocr_recoveries",
        "empty_ocr_context_fallbacks",
        "empty_ocr_density_fallbacks",
        "empty_ocr_quality_fallbacks",
        "candidate_limit",
    )
    counts = {name: _recognition_count(fusion_report, name) for name in names}
    raw_batches = fusion_report.get("recognition_batches", [])
    batches = raw_batches if isinstance(raw_batches, list) else []
    bad_batches = [
        {
            "page": batch.get("page"),
            "ids": batch.get("ids"),
            "status": batch.get("status"),
        }
        for batch in batches
        if isinstance(batch, Mapping) and batch.get("status") != "ok"
    ]
    required_names = (
        "candidates",
        "requests",
        "responses",
        "invalid_outputs",
        "errors",
    )
    evidence_complete = all(counts[name] is not None for name in required_names)
    candidates = counts["candidates"] or 0
    responses = counts["responses"] or 0
    healthy = bool(
        evidence_complete
        and candidates > 0
        and (counts["requests"] or 0) > 0
        and responses == candidates
        and counts["invalid_outputs"] == 0
        and counts["errors"] == 0
        and (counts["protocol_echoes"] in (None, 0))
        and (counts["batch_quality_fallbacks"] in (None, 0))
        and (counts["candidate_limit"] in (None, 0))
        and not bad_batches
    )
    return {
        "healthy": healthy,
        "evidence_complete": evidence_complete,
        "counts": counts,
        "bad_batches": bad_batches,
    }


def _source_metrics(items: Sequence[Mapping[str, Any]], source: str) -> dict[str, Any]:
    exact = 0
    edits = 0
    reference_characters = 0
    missing = 0
    for item in items:
        reference = _normalize(str(item["reference_text"]))
        candidate_value = item.get(source)
        if candidate_value is None:
            missing += 1
            candidate = ""
        else:
            candidate = _normalize(str(candidate_value))
        exact += int(candidate == reference)
        edits += _edit_distance(candidate, reference)
        reference_characters += max(len(reference), 1)
    total = len(items)
    return {
        "items": total,
        "exact": exact,
        "exact_accuracy": round(exact / total, 6) if total else 0.0,
        "character_errors": edits,
        "reference_characters": reference_characters,
        "cer": round(edits / reference_characters, 6)
        if reference_characters
        else 0.0,
        "missing": missing,
    }


def reselect_bbox_recognition_report(
    fusion_report: Mapping[str, Any],
    fusion_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay only OCR/VLM selection policy over previously audited candidates."""
    replayed = copy.deepcopy(dict(fusion_report))
    settings = FusionSettings.from_mapping(fusion_config)
    vlm_selected = 0
    ocr_kept = 0
    high_risk_fallbacks = 0
    protocol_echoes = 0
    batch_quality_fallbacks = 0
    empty_ocr_recoveries = 0
    empty_ocr_context_fallbacks = 0
    replay_entries = []
    by_id = {}
    returned = {}
    for sequence_index, decision in enumerate(replayed.get("decisions", [])):
        if not isinstance(decision, dict) or decision.get("kind") != "bbox_recognition":
            continue
        ocr_text = decision.get("ocr_text")
        bbox = decision.get("bbox")
        page = decision.get("page")
        if (
            not isinstance(ocr_text, str)
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not isinstance(page, int)
        ):
            continue
        try:
            normalized_bbox = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        confidence = decision.get("ocr_confidence")
        if not isinstance(confidence, (int, float)):
            confidence = None
        line = TextLine(
            page_index=page,
            bbox=normalized_bbox,
            text=ocr_text,
            confidence=float(confidence) if confidence is not None else None,
            spans=[],
            block_type=str(decision.get("block_type", "text")),
            sequence_index=sequence_index,
        )
        vlm_value = decision.get("vlm_text")
        candidate_id = decision.get("id")
        if isinstance(candidate_id, str):
            by_id[candidate_id] = line
            if isinstance(vlm_value, str):
                returned[candidate_id] = vlm_value.strip()
        replay_entries.append((decision, line, vlm_value, candidate_id))
    raw_batches = replayed.get("recognition_batches", [])
    batches = (
        [batch for batch in raw_batches if isinstance(batch, dict)]
        if isinstance(raw_batches, list)
        else []
    )
    batch_guarded_ids = apply_recognition_batch_quality_guard(
        batches,
        returned,
        by_id,
        settings,
    )
    quality_passed_ids = recognition_quality_passed_ids(batches)
    for decision, line, vlm_value, candidate_id in replay_entries:
        if isinstance(vlm_value, str):
            source, reason, field_type, similarity, length_ratio = (
                select_bbox_recognition_candidate(
                    line,
                    vlm_value,
                    settings,
                )
            )
        else:
            source = "ocr"
            reason = "no_vlm_response"
            field_type = "general"
            similarity = 0.0
            length_ratio = 0.0
        candidate_reason = reason
        if candidate_id in batch_guarded_ids:
            source = "ocr"
            reason = "recognizer_batch_quality_guard"
            batch_quality_fallbacks += 1
        elif (
            candidate_reason == "empty_ocr_vlm_recovery"
            and candidate_id not in quality_passed_ids
        ):
            source = "ocr"
            reason = "empty_ocr_context_guard"
            empty_ocr_context_fallbacks += 1
        decision.update(
            {
                "selected_source": source,
                "selected_text": vlm_value.strip()
                if source == "vlm" and isinstance(vlm_value, str)
                else line.text,
                "reason": reason,
                "field_type": field_type,
                "similarity": round(similarity, 6),
                "length_ratio": round(length_ratio, 6),
            }
        )
        if candidate_reason != reason:
            decision["candidate_reason"] = candidate_reason
        else:
            decision.pop("candidate_reason", None)
        if source == "vlm":
            vlm_selected += 1
            empty_ocr_recoveries += int(
                candidate_reason == "empty_ocr_vlm_recovery"
            )
        else:
            ocr_kept += 1
            high_risk_fallbacks += int(candidate_reason.startswith("high_risk_"))
            protocol_echoes += int(candidate_reason == "bbox_protocol_id_echo")
    counts = replayed.get("counts")
    if isinstance(counts, dict):
        counts["bbox_recognition_vlm_selected"] = vlm_selected
        counts["bbox_recognition_ocr_kept"] = ocr_kept
        counts["bbox_recognition_high_risk_fallbacks"] = high_risk_fallbacks
        counts["bbox_recognition_protocol_echoes"] = protocol_echoes
        counts["bbox_recognition_batch_quality_fallbacks"] = (
            batch_quality_fallbacks
        )
        counts["bbox_recognition_empty_ocr_recoveries"] = empty_ocr_recoveries
        counts["bbox_recognition_empty_ocr_context_fallbacks"] = (
            empty_ocr_context_fallbacks
        )
    replayed["selection_policy_replayed"] = True
    return replayed


def evaluate_bbox_recognition(
    fusion_report: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    decisions = {
        key: decision
        for decision in fusion_report.get("decisions", [])
        if isinstance(decision, Mapping)
        and decision.get("kind") == "bbox_recognition"
        and (key := _bbox_key(decision.get("page"), decision.get("bbox")))
        is not None
    }
    evaluated = []
    missing_reference_matches = []
    for reference_item in reference.get("items", []):
        if not isinstance(reference_item, Mapping):
            continue
        key = _bbox_key(reference_item.get("page"), reference_item.get("bbox"))
        reference_text = reference_item.get("text")
        if key is None or not isinstance(reference_text, str):
            continue
        decision = decisions.get(key)
        if decision is None:
            missing_reference_matches.append(
                {"page": key[0], "bbox": list(key[1]), "text": reference_text}
            )
            continue
        evaluated.append(
            {
                "page": key[0],
                "bbox": list(key[1]),
                "reference_text": reference_text,
                "ocr_text": decision.get("ocr_text"),
                "vlm_text": decision.get("vlm_text"),
                "selected_text": decision.get("selected_text"),
                "selected_source": decision.get("selected_source"),
                "reason": decision.get("reason"),
            }
        )
    sources = {
        source: _source_metrics(evaluated, source)
        for source in ("ocr_text", "vlm_text", "selected_text")
    }
    improved = []
    regressed = []
    for item in evaluated:
        reference_text = _normalize(str(item["reference_text"]))
        ocr_error = _edit_distance(_normalize(str(item.get("ocr_text") or "")), reference_text)
        selected_error = _edit_distance(
            _normalize(str(item.get("selected_text") or "")),
            reference_text,
        )
        if selected_error < ocr_error:
            improved.append(item)
        elif selected_error > ocr_error:
            regressed.append(item)
    reported_invariants = fusion_report.get("recognition_invariants", {})
    if not isinstance(reported_invariants, Mapping):
        reported_invariants = {}
    bbox_unchanged = bool(
        reported_invariants.get("bbox_unchanged", not missing_reference_matches)
    )
    table_structure_unchanged = bool(
        reported_invariants.get("table_structure_unchanged", True)
    )
    reference_bbox_coverage_complete = not missing_reference_matches
    operational_health = _operational_health(fusion_report)
    recommendation = (
        bool(evaluated)
        and reference_bbox_coverage_complete
        and bbox_unchanged
        and table_structure_unchanged
        and operational_health["healthy"]
        and sources["selected_text"]["cer"] < sources["ocr_text"]["cer"]
        and not regressed
    )
    return {
        "version": 1,
        "reference_items": len(reference.get("items", [])),
        "matched_items": len(evaluated),
        "missing_reference_matches": missing_reference_matches,
        "sources": sources,
        "improved_items": improved,
        "regressed_items": regressed,
        "reference_bbox_coverage_complete": reference_bbox_coverage_complete,
        "bbox_keys_unchanged": bbox_unchanged and reference_bbox_coverage_complete,
        "table_structure_unchanged": table_structure_unchanged,
        "operational_health": operational_health,
        "recommended_default_enabled": recommendation,
        "items": evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR, VLM, and selected bbox text against references"
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reselect-config",
        type=Path,
        help="Replay current fusion.recognizer selection over audited candidates",
    )
    parser.add_argument(
        "--require-default-enable-evidence",
        action="store_true",
        help="Exit non-zero unless all accuracy, invariant, and health gates pass",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.reselect_config is not None:
        loaded_config = json.loads(args.reselect_config.read_text(encoding="utf-8"))
        fusion_config = loaded_config.get("fusion", loaded_config)
        if "recognizer" not in fusion_config:
            fusion_config = {"recognizer": fusion_config}
        report = reselect_bbox_recognition_report(report, fusion_config)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    result = evaluate_bbox_recognition(report, reference)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_default_enable_evidence and not result[
        "recommended_default_enabled"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

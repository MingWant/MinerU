"""Configurable vLLM proxy, MinerU runner, and extraction quality evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.fusion import (
    FusionSettings,
    OpenAIVisionVerifier,
    demote_narrative_false_tables,
    fuse_middle_json,
    normalize_fusion_mode,
    recover_table_cell_geometry,
)
from projects.custom_hybrid.form_detection import (
    FORM_DETECTOR_VERSION,
    FormDetectionSettings,
)
from projects.custom_hybrid.form_segmentation import (
    FORM_SEGMENTER_VERSION,
    annotate_form_structure,
)
from projects.custom_hybrid.recognition import OpenAIBBoxRecognizer
from projects.custom_hybrid.recovery import OpenAIBBoxRecoveryReviewer
from projects.custom_hybrid.table_fusion import extract_table_snapshots


DEFAULT_CONFIG_PATH = Path(__file__).with_name("workflow.example.json")
GENERATION_ENDPOINT_SUFFIXES = ("/chat/completions", "/completions")
INTERNAL_MAX_TOKENS_HEADER = "x-custom-hybrid-max-tokens"
INTERNAL_PROTOCOL_HEADER = "x-custom-hybrid-protocol"
FORWARDED_RESPONSE_HEADERS = {
    "cache-control",
    "content-disposition",
    "content-encoding",
    "content-type",
    "x-request-id",
}
IGNORED_REQUEST_HEADERS = {
    "content-length",
    "host",
    INTERNAL_MAX_TOKENS_HEADER,
    INTERNAL_PROTOCOL_HEADER,
}
AUDITED_GENERATION_PARAMETERS = {
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "vllm_xargs",
}


class WorkflowConfigError(ValueError):
    """Raised when a workflow configuration is invalid."""


@dataclass(frozen=True)
class AppliedGenerationPolicy:
    body: dict[str, Any]
    matched_rules: tuple[str, ...]
    changed_parameters: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError as exc:
        raise WorkflowConfigError(f"Workflow config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowConfigError(
            f"Invalid JSON in workflow config {config_path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise WorkflowConfigError("Workflow config root must be a JSON object")
    if config.get("version") != 1:
        raise WorkflowConfigError("Workflow config version must be 1")
    _require_mapping(config, "vllm")
    mineru_config = _require_mapping(config, "mineru")
    evaluation_config = _require_mapping(config, "evaluation")
    _validate_proxy_config(config["vllm"])
    if mineru_config.get("backend", "hybrid-http-client") != "hybrid-http-client":
        raise WorkflowConfigError(
            "mineru.backend must be hybrid-http-client so requests pass through the parameter proxy"
        )
    if mineru_config.get("effort", "high") not in {"medium", "high"}:
        raise WorkflowConfigError("mineru.effort must be medium or high")
    if mineru_config.get("method", "auto") not in {"auto", "txt", "ocr"}:
        raise WorkflowConfigError("mineru.method must be auto, txt, or ocr")
    _validate_fusion_config(config.get("fusion", {}))
    _validate_sweep_config(config.get("sweep", {}))
    regression_tolerance = evaluation_config.get("regression_tolerance", 0.0)
    if not isinstance(regression_tolerance, (int, float)) or regression_tolerance < 0:
        raise WorkflowConfigError("evaluation.regression_tolerance must be non-negative")
    return config


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise WorkflowConfigError(f"{key} must be a JSON object")
    return value


def _validate_proxy_config(vllm_config: Mapping[str, Any]) -> None:
    upstream_url = vllm_config.get("upstream_url")
    if not isinstance(upstream_url, str) or not upstream_url.startswith(("http://", "https://")):
        raise WorkflowConfigError("vllm.upstream_url must be an http(s) URL")
    proxy = _require_mapping(vllm_config, "proxy")
    host = proxy.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise WorkflowConfigError("vllm.proxy.host must be a non-empty string")
    if host not in {"127.0.0.1", "localhost", "::1"} and not proxy.get(
        "allow_public_bind", False
    ):
        raise WorkflowConfigError(
            "Public proxy binding requires vllm.proxy.allow_public_bind=true"
        )
    port = proxy.get("port", 30001)
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise WorkflowConfigError("vllm.proxy.port must be an integer from 0 to 65535")
    prompt_preview_chars = vllm_config.get("audit_prompt_preview_chars", 0)
    if not isinstance(prompt_preview_chars, int) or prompt_preview_chars < 0:
        raise WorkflowConfigError("vllm.audit_prompt_preview_chars must be a non-negative integer")
    timeout = vllm_config.get("request_timeout_seconds", 600)
    if not isinstance(timeout, (int, float)) or float(timeout) <= 0:
        raise WorkflowConfigError("vllm.request_timeout_seconds must be positive")
    generation = _require_mapping(vllm_config, "generation")
    for key in ("defaults", "overrides", "task_overrides"):
        if not isinstance(generation.get(key, {}), dict):
            raise WorkflowConfigError(f"vllm.generation.{key} must be a JSON object")
    if not isinstance(generation.get("remove", []), list):
        raise WorkflowConfigError("vllm.generation.remove must be a JSON array")
    max_context_tokens = generation.get("max_context_tokens")
    if max_context_tokens is not None and (
        not isinstance(max_context_tokens, int) or max_context_tokens <= 1
    ):
        raise WorkflowConfigError(
            "vllm.generation.max_context_tokens must be an integer greater than 1"
        )
    context_reserve_tokens = generation.get("context_reserve_tokens")
    if context_reserve_tokens is not None and (
        not isinstance(context_reserve_tokens, int) or context_reserve_tokens < 1
    ):
        raise WorkflowConfigError(
            "vllm.generation.context_reserve_tokens must be a positive integer"
        )
    if (
        isinstance(max_context_tokens, int)
        and isinstance(context_reserve_tokens, int)
        and context_reserve_tokens >= max_context_tokens
    ):
        raise WorkflowConfigError(
            "vllm.generation.context_reserve_tokens must be smaller than "
            "max_context_tokens"
        )
    rules = generation.get("rules", [])
    if not isinstance(rules, list):
        raise WorkflowConfigError("vllm.generation.rules must be a JSON array")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise WorkflowConfigError(f"vllm.generation.rules[{index}] must be an object")
        if not isinstance(rule.get("name"), str) or not rule["name"]:
            raise WorkflowConfigError(f"vllm.generation.rules[{index}].name is required")
        match = rule.get("match", {})
        if not isinstance(match, dict):
            raise WorkflowConfigError(f"vllm.generation.rules[{index}].match must be an object")
        for regex_key in ("path_regex", "text_regex", "model_regex"):
            pattern = match.get(regex_key)
            if pattern is not None:
                try:
                    re.compile(pattern)
                except (re.error, TypeError) as exc:
                    raise WorkflowConfigError(
                        f"Invalid {regex_key} in generation rule {rule['name']}: {exc}"
                    ) from exc


def _validate_fusion_config(fusion_config: Any) -> None:
    if not isinstance(fusion_config, dict):
        raise WorkflowConfigError("fusion must be a JSON object")
    normalized_mode = normalize_fusion_mode(
        fusion_config.get("mode", "hybrid_fusion")
    )
    if normalized_mode not in {"hybrid_fusion", "bbox_vlm"}:
        raise WorkflowConfigError(
            "fusion.mode must be hybrid_fusion or bbox_vlm"
        )
    if normalized_mode == "bbox_vlm" and not fusion_config.get("enabled", False):
        raise WorkflowConfigError(
            "fusion.enabled must be true for BBox VLM extraction modes"
        )
    form_detection = fusion_config.get("form_detection", {})
    if not isinstance(form_detection, dict):
        raise WorkflowConfigError("fusion.form_detection must be a JSON object")
    if not isinstance(form_detection.get("enabled", True), bool):
        raise WorkflowConfigError("fusion.form_detection.enabled must be a boolean")
    render_scale = form_detection.get("render_scale", 2.0)
    if (
        isinstance(render_scale, bool)
        or not isinstance(render_scale, (int, float))
        or float(render_scale) <= 0
    ):
        raise WorkflowConfigError("fusion.form_detection.render_scale must be positive")
    table_coverage = form_detection.get(
        "existing_table_coverage_threshold",
        0.75,
    )
    if (
        isinstance(table_coverage, bool)
        or not isinstance(table_coverage, (int, float))
        or not 0 <= float(table_coverage) <= 1
    ):
        raise WorkflowConfigError(
            "fusion.form_detection.existing_table_coverage_threshold must be "
            "between 0 and 1"
        )
    page_sorting = fusion_config.get("page_sorting", {})
    if not isinstance(page_sorting, dict):
        raise WorkflowConfigError("fusion.page_sorting must be a JSON object")
    if not isinstance(page_sorting.get("enabled", False), bool):
        raise WorkflowConfigError("fusion.page_sorting.enabled must be a boolean")
    if page_sorting.get("mode", "report_only") != "report_only":
        raise WorkflowConfigError(
            "fusion.page_sorting.mode must be report_only"
        )
    if not isinstance(
        page_sorting.get("include_semantic_diagnostics", True),
        bool,
    ):
        raise WorkflowConfigError(
            "fusion.page_sorting.include_semantic_diagnostics must be a boolean"
        )
    if not fusion_config.get("enabled", False):
        return
    bounded_values = {
        "min_overlap": (0.0, 1.0),
        "min_ocr_confidence": (0.0, 1.0),
        "consensus_similarity": (0.0, 1.0),
        "candidate_guard_similarity": (0.0, 1.0),
        "table_consensus_similarity": (0.0, 1.0),
        "formula_consensus_similarity": (0.0, 1.0),
        "table_cell_consensus_similarity": (0.0, 1.0),
        "table_cell_min_ocr_confidence": (0.0, 1.0),
        "table_cell_metadata_text_similarity": (0.0, 1.0),
        "missing_ocr_min_confidence": (0.0, 1.0),
    }
    for key, (minimum, maximum) in bounded_values.items():
        value = fusion_config.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum
        ):
            raise WorkflowConfigError(f"fusion.{key} must be between {minimum} and {maximum}")
    verifier = fusion_config.get("verifier", {})
    if not isinstance(verifier, dict):
        raise WorkflowConfigError("fusion.verifier must be a JSON object")
    recognizer = fusion_config.get("recognizer", {})
    if not isinstance(recognizer, dict):
        raise WorkflowConfigError("fusion.recognizer must be a JSON object")
    recovery = fusion_config.get("recovery", {})
    if not isinstance(recovery, dict):
        raise WorkflowConfigError("fusion.recovery must be a JSON object")
    if not isinstance(recovery.get("enabled", False), bool):
        raise WorkflowConfigError("fusion.recovery.enabled must be a boolean")
    for key in (
        "local_missing_enabled",
        "local_uncovered_enabled",
        "review_after_local_recovery",
        "skip_vlm_for_mineru_models",
        "disable_after_invalid_schema",
        "share_recognizer_page_cache",
        "page_recovery_enabled",
        "table_orphan_recovery_enabled",
        "table_fringe_recovery_enabled",
        "checkbox_recovery_enabled",
        "checkbox_accept_existing_label_bbox",
        "checkbox_merge_label_enabled",
        "checkbox_protruding_tick_enabled",
        "checkbox_embedded_glyph_guard_enabled",
        "list_marker_merge_enabled",
        "table_diagonal_rule_enabled",
    ):
        field = recovery.get(key)
        if field is not None and not isinstance(field, bool):
            raise WorkflowConfigError(f"fusion.recovery.{key} must be a boolean")
    recovery_base_url = recovery.get("base_url")
    if recovery_base_url is not None and (
        not isinstance(recovery_base_url, str)
        or not recovery_base_url.startswith(("http://", "https://"))
    ):
        raise WorkflowConfigError(
            "fusion.recovery.base_url must be null or an http(s) URL"
        )
    for key in ("model", "api_key_env"):
        field = recovery.get(key)
        if field is not None and (not isinstance(field, str) or not field.strip()):
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be null or a non-empty string"
            )
    for key in (
        "min_confidence",
        "min_area_ratio",
        "max_area_ratio",
        "duplicate_iou",
        "cross_cell_duplicate_overlap",
        "adjust_min_iou",
        "min_ink_ratio",
        "max_cell_ink_ratio",
        "min_uncovered_ink_ratio",
        "cell_horizontal_rule_ratio",
        "cell_vertical_rule_ratio",
        "local_pixel_confidence",
        "table_orphan_confidence",
        "table_orphan_max_cell_overlap",
        "table_orphan_horizontal_line_ratio",
        "table_orphan_vertical_line_ratio",
        "table_orphan_max_line_height_ratio",
        "table_orphan_min_ink_density",
        "table_orphan_max_ink_density",
        "checkbox_confidence",
        "checkbox_ambiguous_confidence",
        "checkbox_min_side_density",
        "checkbox_max_left_ink_ratio",
        "checkbox_min_right_ink_ratio",
        "checkbox_right_column_ink_ratio",
        "checkbox_unchecked_interior_ratio",
        "checkbox_checked_interior_ratio",
        "checkbox_label_min_vertical_overlap",
        "checkbox_tick_min_square_coverage",
        "form_full_cell_recovery_max_width_ratio",
        "list_marker_confidence",
        "list_marker_min_vertical_overlap",
        "page_recovery_graphic_min_ink_density",
        "page_recovery_graphic_max_component_ratio",
    ):
        field = recovery.get(key)
        if field is not None and (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or not 0 <= float(field) <= 1
        ):
            raise WorkflowConfigError(f"fusion.recovery.{key} must be between 0 and 1")
    for key in (
        "table_diagonal_rule_min_angle",
        "table_diagonal_rule_max_angle",
    ):
        field = recovery.get(key)
        if field is not None and (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or not 0 <= float(field) <= 90
        ):
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be between 0 and 90"
            )
    if float(recovery.get("table_diagonal_rule_max_angle", 88.0)) < float(
        recovery.get("table_diagonal_rule_min_angle", 12.0)
    ):
        raise WorkflowConfigError(
            "fusion.recovery diagonal rule angles must be ordered"
        )
    for key, default in (
        ("table_orphan_horizontal_line_ratio", 0.5),
        ("table_orphan_vertical_line_ratio", 0.5),
        ("table_orphan_max_line_height_ratio", 0.07),
        ("table_orphan_min_ink_density", 0.01),
    ):
        if float(recovery.get(key, default)) <= 0:
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be greater than 0"
            )
    if float(recovery.get("max_area_ratio", 0.95)) < float(
        recovery.get("min_area_ratio", 0.001)
    ):
        raise WorkflowConfigError("fusion.recovery area ratios must be ordered")
    if float(recovery.get("max_cell_ink_ratio", 0.65)) < float(
        recovery.get("min_ink_ratio", 0.002)
    ):
        raise WorkflowConfigError(
            "fusion.recovery Cell ink-ratio limits must be ordered"
        )
    if float(recovery.get("table_orphan_max_ink_density", 0.7)) < float(
        recovery.get("table_orphan_min_ink_density", 0.01)
    ):
        raise WorkflowConfigError(
            "fusion.recovery Table orphan ink-density limits must be ordered"
        )
    for key in (
        "max_tables_per_document",
        "max_proposals_per_document",
        "max_proposals_per_table",
        "max_requests_per_document",
        "table_orphan_max_boxes_per_table",
        "checkbox_max_boxes_per_table",
        "list_marker_max_merges_per_table",
    ):
        field = recovery.get(key)
        if field is not None and (
            isinstance(field, bool) or not isinstance(field, int) or field < 0
        ):
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be a non-negative integer"
            )
    for key in ("timeout_seconds", "render_scale"):
        field = recovery.get(key)
        if field is not None and (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or float(field) <= 0
        ):
            raise WorkflowConfigError(f"fusion.recovery.{key} must be positive")
    for key in ("max_tokens", "cache_pages"):
        field = recovery.get(key)
        if field is not None and (
            isinstance(field, bool) or not isinstance(field, int) or field < 1
        ):
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be a positive integer"
            )
    orphan_min_row_ink = recovery.get("table_orphan_min_row_ink_pixels", 4)
    if (
        isinstance(orphan_min_row_ink, bool)
        or not isinstance(orphan_min_row_ink, int)
        or orphan_min_row_ink < 1
    ):
        raise WorkflowConfigError(
            "fusion.recovery.table_orphan_min_row_ink_pixels must be a "
            "positive integer"
        )
    cell_min_row_ink = recovery.get("cell_line_min_row_ink_pixels", 2)
    if (
        isinstance(cell_min_row_ink, bool)
        or not isinstance(cell_min_row_ink, int)
        or cell_min_row_ink < 1
    ):
        raise WorkflowConfigError(
            "fusion.recovery.cell_line_min_row_ink_pixels must be a positive "
            "integer"
        )
    minimum_uncovered_ink = recovery.get("min_uncovered_ink_pixels", 12)
    if (
        isinstance(minimum_uncovered_ink, bool)
        or not isinstance(minimum_uncovered_ink, int)
        or minimum_uncovered_ink < 1
    ):
        raise WorkflowConfigError(
            "fusion.recovery.min_uncovered_ink_pixels must be a positive integer"
        )
    for key, default, allow_zero in (
        ("cell_rule_min_length", 18.0, False),
        ("cell_rule_padding", 1.0, True),
        ("cell_rule_max_thickness", 2.5, False),
        ("cell_line_gap", 1.5, True),
        ("cell_line_min_height", 2.0, False),
        ("cell_line_min_dark_height", 3.0, False),
        ("cell_line_min_width", 3.0, False),
        ("existing_bbox_padding", 1.0, True),
        ("table_diagonal_rule_min_length", 80.0, False),
        ("table_diagonal_rule_max_gap", 8.0, True),
        ("table_diagonal_rule_padding", 1.5, True),
        ("table_diagonal_rule_extension", 20.0, True),
        ("table_orphan_cell_padding", 1.5, True),
        ("table_orphan_line_gap", 0.75, True),
        ("table_orphan_min_line_height", 3.0, False),
        ("table_orphan_min_dark_height", 2.0, False),
        ("table_orphan_min_line_width", 8.0, False),
        ("table_orphan_horizontal_gap", 12.0, False),
        ("table_fringe_bottom_extension", 72.0, True),
        ("date_range_field_bottom_extension", 6.0, True),
        ("form_full_cell_recovery_max_width_ratio", 0.6, False),
        ("same_cell_fragment_max_vertical_gap", 2.5, True),
        ("same_cell_fragment_max_center_delta_ratio", 0.8, False),
        ("same_cell_fragment_max_union_height_ratio", 2.2, False),
        ("checkbox_label_max_gap", 24.0, True),
        ("list_marker_max_gap", 24.0, True),
        ("page_recovery_graphic_min_height", 14.0, False),
        ("page_recovery_max_line_height", 24.0, False),
        ("checkbox_embedded_glyph_max_size", 7.5, False),
        ("checkbox_embedded_glyph_max_left_offset", 16.0, True),
    ):
        field = recovery.get(key, default)
        if (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or (float(field) < 0 if allow_zero else float(field) <= 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be {qualifier}"
            )
    for key, default in (
        ("same_cell_fragment_min_horizontal_overlap", 0.75),
        ("split_content_min_horizontal_overlap", 0.8),
        ("split_content_min_vertical_overlap", 0.35),
    ):
        field = recovery.get(key, default)
        if (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or not 0 <= float(field) <= 1
        ):
            raise WorkflowConfigError(
                f"fusion.recovery.{key} must be between 0 and 1"
            )
    checkbox_max_vertices = recovery.get("checkbox_max_vertices", 5)
    if (
        isinstance(checkbox_max_vertices, bool)
        or not isinstance(checkbox_max_vertices, int)
        or checkbox_max_vertices < 4
    ):
        raise WorkflowConfigError(
            "fusion.recovery.checkbox_max_vertices must be an integer of at "
            "least 4"
        )
    checkbox_ambiguous_max_vertices = recovery.get(
        "checkbox_ambiguous_max_vertices", 4
    )
    if (
        isinstance(checkbox_ambiguous_max_vertices, bool)
        or not isinstance(checkbox_ambiguous_max_vertices, int)
        or checkbox_ambiguous_max_vertices < 4
        or checkbox_ambiguous_max_vertices > checkbox_max_vertices
    ):
        raise WorkflowConfigError(
            "fusion.recovery.checkbox_ambiguous_max_vertices must be an "
            "integer between 4 and checkbox_max_vertices"
        )
    for key, default in (
        ("checkbox_min_size", 5.5),
        ("checkbox_max_size", 16.0),
        ("checkbox_min_aspect", 0.75),
        ("checkbox_max_aspect", 1.25),
        ("checkbox_left_clearance", 8.0),
        ("checkbox_table_edge_allowance", 12.0),
        ("checkbox_right_context", 30.0),
        ("checkbox_right_separator_search", 6.0),
        ("checkbox_min_right_separator", 1.5),
        ("checkbox_existing_tight_scale", 2.0),
        ("checkbox_protruding_tick_min_size", 4.5),
        ("checkbox_tick_max_width", 28.0),
        ("checkbox_tick_max_height", 24.0),
        ("checkbox_tick_min_square_coverage", 0.75),
        ("checkbox_apply_min_size", 4.0),
        ("checkbox_apply_max_size", 20.0),
        ("checkbox_apply_min_aspect", 0.65),
        ("checkbox_apply_max_aspect", 1.4),
    ):
        field = recovery.get(key, default)
        if (
            isinstance(field, bool)
            or not isinstance(field, (int, float))
            or float(field) <= 0
        ):
            raise WorkflowConfigError(f"fusion.recovery.{key} must be positive")
    for lower, upper, defaults in (
        ("checkbox_min_size", "checkbox_max_size", (5.5, 16.0)),
        ("checkbox_min_aspect", "checkbox_max_aspect", (0.75, 1.25)),
        ("checkbox_apply_min_size", "checkbox_apply_max_size", (4.0, 20.0)),
        ("checkbox_apply_min_aspect", "checkbox_apply_max_aspect", (0.65, 1.4)),
        (
            "checkbox_unchecked_interior_ratio",
            "checkbox_checked_interior_ratio",
            (0.03, 0.12),
        ),
    ):
        if float(recovery.get(lower, defaults[0])) >= float(
            recovery.get(upper, defaults[1])
        ):
            raise WorkflowConfigError(
                f"fusion.recovery {lower}/{upper} limits must be ordered"
            )
    if float(recovery.get("checkbox_protruding_tick_min_size", 4.5)) > float(
        recovery.get("checkbox_min_size", 5.5)
    ):
        raise WorkflowConfigError(
            "fusion.recovery.checkbox_protruding_tick_min_size must not "
            "exceed checkbox_min_size"
        )
    recovery_structured_cap = recovery.get("structured_max_tokens_cap", 768)
    if (
        isinstance(recovery_structured_cap, bool)
        or not isinstance(recovery_structured_cap, int)
        or recovery_structured_cap < 1
    ):
        raise WorkflowConfigError(
            "fusion.recovery.structured_max_tokens_cap must be a positive integer"
        )
    recovery_temperature = recovery.get("temperature", 0.0)
    if (
        isinstance(recovery_temperature, bool)
        or not isinstance(recovery_temperature, (int, float))
        or not 0 <= float(recovery_temperature) <= 2
    ):
        raise WorkflowConfigError(
            "fusion.recovery.temperature must be between 0 and 2"
        )
    recovery_top_p = recovery.get("top_p", 1.0)
    if (
        isinstance(recovery_top_p, bool)
        or not isinstance(recovery_top_p, (int, float))
        or not 0 < float(recovery_top_p) <= 1
    ):
        raise WorkflowConfigError(
            "fusion.recovery.top_p must be greater than 0 and at most 1"
        )
    recovery_seed = recovery.get("seed")
    if recovery_seed is not None and (
        isinstance(recovery_seed, bool)
        or not isinstance(recovery_seed, int)
        or not -(2**63) <= recovery_seed < 2**63
    ):
        raise WorkflowConfigError(
            "fusion.recovery.seed must be a signed 64-bit integer"
        )
    recovery_jpeg_quality = recovery.get("jpeg_quality", 92)
    if (
        not isinstance(recovery_jpeg_quality, int)
        or not 1 <= recovery_jpeg_quality <= 100
    ):
        raise WorkflowConfigError(
            "fusion.recovery.jpeg_quality must be from 1 to 100"
        )
    selection_policy = recognizer.get("selection_policy", "conservative")
    if selection_policy not in {"conservative", "vlm_primary"}:
        raise WorkflowConfigError(
            "fusion.recognizer.selection_policy must be conservative or "
            "vlm_primary"
        )
    recognizer_protocol = recognizer.get("protocol", "auto")
    if recognizer_protocol not in {"auto", "structured", "mineru_native"}:
        raise WorkflowConfigError(
            "fusion.recognizer.protocol must be auto, structured, or mineru_native"
        )
    recognizer_base_url = recognizer.get("base_url")
    if recognizer_base_url is not None and (
        not isinstance(recognizer_base_url, str)
        or not recognizer_base_url.startswith(("http://", "https://"))
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.base_url must be null or an http(s) URL"
        )
    for key in ("model", "api_key_env"):
        value = recognizer.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise WorkflowConfigError(
                f"fusion.recognizer.{key} must be null or a non-empty string"
            )
    for key in (
        "enabled",
        "normal_ocr_enabled",
        "table_ocr_enabled",
        "prefer_vlm_for_unscored",
        "json_mode",
        "include_row_image",
        "include_column_image",
        "include_table_image",
        "disable_structured_output_whitespace",
        "hide_ids_in_prompt_when_constrained",
        "batch_guard_enabled",
        "empty_ocr_enabled",
        "script_guard_enabled",
        "native_all_candidates",
        "native_cache_enabled",
    ):
        if not isinstance(recognizer.get(key, False if key == "enabled" else True), bool):
            raise WorkflowConfigError(f"fusion.recognizer.{key} must be a boolean")
    structured_output_mode = recognizer.get("structured_output_mode")
    if structured_output_mode is not None and structured_output_mode not in {
        "json_schema",
        "structured_outputs",
        "regex",
        "json_object",
        "none",
    }:
        raise WorkflowConfigError(
            "fusion.recognizer.structured_output_mode must be json_schema, "
            "structured_outputs, regex, json_object, or none"
        )
    for key in (
        "min_similarity",
        "high_confidence_threshold",
        "assumed_vlm_confidence",
        "min_batch_acceptable_ratio",
        "vlm_primary_min_quality",
        "recovered_empty_min_confidence",
    ):
        value = recognizer.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or not 0 <= float(value) <= 1
        ):
            raise WorkflowConfigError(f"fusion.recognizer.{key} must be between 0 and 1")
    min_length_ratio = recognizer.get("min_length_ratio", 0.4)
    max_length_ratio = recognizer.get("max_length_ratio", 2.5)
    if (
        not isinstance(min_length_ratio, (int, float))
        or float(min_length_ratio) <= 0
        or not isinstance(max_length_ratio, (int, float))
        or float(max_length_ratio) < float(min_length_ratio)
    ):
        raise WorkflowConfigError(
            "fusion.recognizer length ratios must be positive and ordered"
        )
    for key in (
        "timeout_seconds",
        "target_render_scale",
        "context_render_scale",
        "empty_max_chars_per_em",
        "empty_thin_line_max_chars_per_em",
    ):
        value = recognizer.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) <= 0
        ):
            raise WorkflowConfigError(f"fusion.recognizer.{key} must be positive")
    for key in (
        "max_tokens",
        "cache_pages",
        "max_batch_size",
        "max_images_per_request",
        "max_prompt_chars",
        "candidate_text_max_chars",
        "max_candidates_per_document",
        "max_text_chars",
        "batch_guard_min_candidates",
    ):
        value = recognizer.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise WorkflowConfigError(
                f"fusion.recognizer.{key} must be a positive integer"
            )
    preview_chars = recognizer.get("audit_response_preview_chars", 0)
    if (
        isinstance(preview_chars, bool)
        or not isinstance(preview_chars, int)
        or preview_chars < 0
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.audit_response_preview_chars must be a "
            "non-negative integer"
        )
    max_requests = recognizer.get("max_requests_per_document", 80)
    if (
        isinstance(max_requests, bool)
        or not isinstance(max_requests, int)
        or max_requests < 0
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.max_requests_per_document must be a "
            "non-negative integer"
        )
    max_image_limit_retries = recognizer.get("max_image_limit_retries", 2)
    if (
        isinstance(max_image_limit_retries, bool)
        or not isinstance(max_image_limit_retries, int)
        or max_image_limit_retries < 0
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.max_image_limit_retries must be a "
            "non-negative integer"
        )
    for key, default in (
        ("native_max_tokens", 512),
        ("structured_max_tokens_cap", 1024),
        ("max_consecutive_invalid_schema", 2),
        ("native_max_concurrency", 2),
        ("native_max_consecutive_failures", 3),
    ):
        value = recognizer.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkflowConfigError(
                f"fusion.recognizer.{key} must be a positive integer"
            )
    native_min_bbox_height = recognizer.get("native_min_bbox_height", 16.0)
    if (
        isinstance(native_min_bbox_height, bool)
        or not isinstance(native_min_bbox_height, (int, float))
        or float(native_min_bbox_height) < 0
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.native_min_bbox_height must be non-negative"
        )
    for key, default in (
        ("native_max_candidates_per_page", 0),
        ("native_max_requests_per_page", 0),
        ("native_cache_ttl_seconds", 2592000),
    ):
        value = recognizer.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkflowConfigError(
                f"fusion.recognizer.{key} must be a non-negative integer"
            )
    native_cache_dir = recognizer.get("native_cache_dir")
    if native_cache_dir is not None and (
        not isinstance(native_cache_dir, str) or not native_cache_dir.strip()
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.native_cache_dir must be null or a non-empty string"
        )
    temperature = recognizer.get("temperature", 0.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= float(temperature) <= 2
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.temperature must be between 0 and 2"
        )
    top_p = recognizer.get("top_p", 1.0)
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0 < float(top_p) <= 1
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.top_p must be greater than 0 and at most 1"
        )
    seed = recognizer.get("seed")
    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not -(2**63) <= seed < 2**63
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.seed must be a signed 64-bit integer"
        )
    for key in ("target_padding_ratio", "context_padding_ratio"):
        value = recognizer.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or float(value) < 0
        ):
            raise WorkflowConfigError(f"fusion.recognizer.{key} must be non-negative")
    recognizer_jpeg_quality = recognizer.get("jpeg_quality", 92)
    if not isinstance(recognizer_jpeg_quality, int) or not 1 <= recognizer_jpeg_quality <= 100:
        raise WorkflowConfigError("fusion.recognizer.jpeg_quality must be from 1 to 100")
    max_context_tokens = recognizer.get("max_context_tokens")
    reserve_tokens = recognizer.get("context_reserve_tokens", 2048)
    if (
        isinstance(reserve_tokens, bool)
        or not isinstance(reserve_tokens, int)
        or reserve_tokens <= 0
    ):
        raise WorkflowConfigError(
            "fusion.recognizer.context_reserve_tokens must be a positive integer"
        )
    if max_context_tokens is not None and (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens <= 1
        or reserve_tokens >= max_context_tokens
    ):
        raise WorkflowConfigError(
            "fusion.recognizer context token limit must exceed its positive reserve"
        )
    reconciliation = fusion_config.get("reconciliation", {})
    if not isinstance(reconciliation, dict):
        raise WorkflowConfigError("fusion.reconciliation must be a JSON object")
    if not isinstance(reconciliation.get("enabled", False), bool):
        raise WorkflowConfigError("fusion.reconciliation.enabled must be a boolean")
    for key, default in (
        ("max_pages_per_document", 20),
        ("max_candidates_per_page", 120),
    ):
        value = reconciliation.get(key, default)
        if not isinstance(value, int) or value < 0:
            raise WorkflowConfigError(
                f"fusion.reconciliation.{key} must be a non-negative integer"
            )
    suspicious_ratio = fusion_config.get("suspicious_length_ratio", 1.8)
    if not isinstance(suspicious_ratio, (int, float)) or suspicious_ratio <= 1:
        raise WorkflowConfigError("fusion.suspicious_length_ratio must be greater than 1")
    max_verifications = fusion_config.get("max_verifications_per_document", 80)
    if not isinstance(max_verifications, int) or max_verifications < 0:
        raise WorkflowConfigError(
            "fusion.max_verifications_per_document must be a non-negative integer"
        )
    for key in (
        "table_fallback_enabled",
        "formula_fallback_enabled",
        "table_visual_verification_enabled",
        "formula_visual_verification_enabled",
        "table_cell_fusion_enabled",
        "table_cell_empty_fallback_enabled",
        "table_cell_suspicious_fallback_enabled",
        "table_cell_allow_unscored_ocr",
        "recover_missing_ocr_blocks",
        "unreliable_table_recovery_enabled",
        "unreliable_table_allow_unscored_ocr",
    ):
        value = fusion_config.get(key, True)
        if not isinstance(value, bool):
            raise WorkflowConfigError(f"fusion.{key} must be a boolean")
    max_structured = fusion_config.get("max_structured_verifications_per_document", 20)
    if not isinstance(max_structured, int) or max_structured < 0:
        raise WorkflowConfigError(
            "fusion.max_structured_verifications_per_document must be a non-negative integer"
        )
    for key, default in (
        ("max_table_cells_per_table", 500),
        ("max_table_cell_verifications_per_document", 80),
        ("max_missing_ocr_blocks_per_document", 200),
        ("max_unreliable_table_ocr_blocks_per_document", 1000),
    ):
        value = fusion_config.get(key, default)
        minimum = 1 if key == "max_table_cells_per_table" else 0
        if not isinstance(value, int) or value < minimum:
            qualifier = "positive" if minimum else "non-negative"
            raise WorkflowConfigError(f"fusion.{key} must be a {qualifier} integer")
    if verifier.get("enabled", False):
        positive_fields = (
            "timeout_seconds",
            "render_scale",
            "table_cell_render_scale",
            "max_tokens",
            "cache_pages",
            "structured_candidate_max_chars",
            "table_cell_context_max_chars",
            "table_cell_max_context_cells",
        )
        for key in positive_fields:
            value = verifier.get(key)
            if value is not None and (
                not isinstance(value, (int, float)) or float(value) <= 0
            ):
                raise WorkflowConfigError(f"fusion.verifier.{key} must be positive")
        padding = verifier.get("padding_ratio", 0.08)
        if not isinstance(padding, (int, float)) or float(padding) < 0:
            raise WorkflowConfigError("fusion.verifier.padding_ratio must be non-negative")
        jpeg_quality = verifier.get("jpeg_quality", 92)
        if not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
            raise WorkflowConfigError("fusion.verifier.jpeg_quality must be from 1 to 100")
        for key in (
            "table_cell_include_table_image",
            "table_cell_include_row_image",
            "table_cell_include_column_image",
        ):
            if not isinstance(verifier.get(key, True), bool):
                raise WorkflowConfigError(f"fusion.verifier.{key} must be a boolean")


def _validate_sweep_config(sweep_config: Any) -> None:
    if not sweep_config:
        return
    if not isinstance(sweep_config, dict):
        raise WorkflowConfigError("sweep must be a JSON object")
    generation = sweep_config.get("generation")
    if not isinstance(generation, dict) or not generation:
        raise WorkflowConfigError("sweep.generation must be a non-empty JSON object")
    for key, values in generation.items():
        if not isinstance(key, str) or not key:
            raise WorkflowConfigError("sweep.generation parameter names must be non-empty")
        if not isinstance(values, list) or not values:
            raise WorkflowConfigError(f"sweep.generation.{key} must be a non-empty array")
    max_runs = sweep_config.get("max_runs", 12)
    if not isinstance(max_runs, int) or max_runs < 1:
        raise WorkflowConfigError("sweep.max_runs must be a positive integer")


def extract_request_text(value: Any) -> str:
    """Collect textual prompt data while excluding images and binary data URIs."""
    parts: list[str] = []

    def visit(item: Any, parent_key: str | None = None) -> None:
        if isinstance(item, str):
            if parent_key in {"image", "image_url", "input_audio"}:
                return
            if item.startswith("data:"):
                return
            parts.append(item)
        elif isinstance(item, list):
            for child in item:
                visit(child, parent_key)
        elif isinstance(item, dict):
            for key, child in item.items():
                if key in {"image", "image_url", "input_audio"}:
                    continue
                visit(child, key)

    visit(value)
    return "\n".join(parts)


def _rule_matches(rule: Mapping[str, Any], path: str, body: Mapping[str, Any], text: str) -> bool:
    match = rule.get("match", {})
    predicates = []
    if "path_regex" in match:
        predicates.append(re.search(str(match["path_regex"]), path) is not None)
    if "text_regex" in match:
        predicates.append(
            re.search(str(match["text_regex"]), text, flags=re.IGNORECASE | re.DOTALL) is not None
        )
    if "model_regex" in match:
        predicates.append(
            re.search(str(match["model_regex"]), str(body.get("model", ""))) is not None
        )
    return all(predicates) if predicates else True


def _apply_policy_section(
    effective: dict[str, Any],
    section: Mapping[str, Any],
) -> None:
    defaults = section.get("defaults", {})
    overrides = section.get("overrides", {})
    remove = section.get("remove", [])
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            effective.setdefault(key, value)
    if isinstance(overrides, dict):
        effective.update(overrides)
    if isinstance(remove, list):
        for key in remove:
            if isinstance(key, str):
                effective.pop(key, None)


def apply_generation_policy(
    path: str,
    body: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    request_max_tokens_cap: int | None = None,
    request_protocol: str | None = None,
) -> AppliedGenerationPolicy:
    original = dict(body)
    effective = dict(body)
    _apply_policy_section(effective, generation_config)
    text = extract_request_text(body)
    matched_rules = []
    for rule in generation_config.get("rules", []):
        if _rule_matches(rule, path, body, text):
            _apply_policy_section(effective, rule)
            matched_rules.append(rule["name"])
    task_overrides = generation_config.get("task_overrides", {})
    if isinstance(task_overrides, dict):
        effective.update(task_overrides)
    if request_protocol == "mineru_native":
        effective["temperature"] = 0.0
        effective["top_p"] = 0.01
    if (
        isinstance(request_max_tokens_cap, int)
        and not isinstance(request_max_tokens_cap, bool)
        and request_max_tokens_cap > 0
    ):
        for token_key in ("max_tokens", "max_completion_tokens"):
            requested_tokens = effective.get(token_key)
            if isinstance(requested_tokens, (int, float)):
                effective[token_key] = min(
                    int(requested_tokens),
                    request_max_tokens_cap,
                )
    max_context_tokens = generation_config.get(
        "_resolved_max_context_tokens",
        generation_config.get("max_context_tokens"),
    )
    if isinstance(max_context_tokens, int) and max_context_tokens > 1:
        configured_reserve = generation_config.get("context_reserve_tokens")
        reserve_tokens = (
            configured_reserve
            if isinstance(configured_reserve, int) and configured_reserve > 0
            else max(1024, max_context_tokens // 2)
        )
        output_cap = max(1, max_context_tokens - reserve_tokens)
        for token_key in ("max_tokens", "max_completion_tokens"):
            requested_tokens = effective.get(token_key)
            if (
                isinstance(requested_tokens, (int, float))
                and requested_tokens > output_cap
            ):
                effective[token_key] = output_cap

    tracked_keys = set(generation_config.get("defaults", {}))
    tracked_keys.update(generation_config.get("overrides", {}))
    tracked_keys.update(generation_config.get("task_overrides", {}))
    tracked_keys.update({"max_tokens", "max_completion_tokens"})
    if request_protocol == "mineru_native":
        tracked_keys.update({"temperature", "top_p"})
    tracked_keys.update(generation_config.get("remove", []))
    for rule in generation_config.get("rules", []):
        if rule.get("name") in matched_rules:
            tracked_keys.update(rule.get("defaults", {}))
            tracked_keys.update(rule.get("overrides", {}))
            tracked_keys.update(rule.get("remove", []))
    changed = {
        key: effective.get(key)
        for key in sorted(key for key in tracked_keys if isinstance(key, str))
        if original.get(key) != effective.get(key) or (key in original) != (key in effective)
    }
    return AppliedGenerationPolicy(effective, tuple(matched_rules), changed)


class JsonlAuditLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser().resolve() if path else None
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _optional_bearer_headers(
    *configs: Mapping[str, Any],
) -> dict[str, str]:
    """Use Bearer auth only when an environment variable is explicitly named."""
    for config in configs:
        api_key_env = config.get("api_key_env")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            continue
        api_key = os.getenv(api_key_env.strip())
        if api_key:
            return {"authorization": f"Bearer {api_key}"}
    return {}


class ParameterProxyASGI:
    """Small ASGI reverse proxy that rewrites OpenAI generation requests."""

    def __init__(self, config: Mapping[str, Any]):
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Proxy mode requires httpx") from exc
        self.httpx = httpx
        vllm_config = config["vllm"]
        self.upstream_url = str(vllm_config["upstream_url"]).rstrip("/") + "/"
        self.timeout = float(vllm_config.get("request_timeout_seconds", 600))
        self.generation = vllm_config["generation"]
        self.prompt_preview_chars = int(vllm_config.get("audit_prompt_preview_chars", 0))
        self.audit = JsonlAuditLogger(vllm_config.get("audit_log"))
        self.upstream_headers = _optional_bearer_headers(vllm_config)
        self.client = None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            raise RuntimeError(f"Unsupported ASGI scope: {scope['type']}")
        await self._handle_http(scope, receive, send)

    async def _handle_lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                self.client = self.httpx.AsyncClient(timeout=self.timeout)
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self.client is not None:
                    await self.client.aclose()
                    self.client = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _read_request_body(self, receive: Any) -> bytes | None:
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            if message["type"] != "http.request":
                continue
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                return b"".join(chunks)

    @staticmethod
    async def _send_json(send: Any, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _handle_http(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        method = str(scope.get("method", "GET")).upper()
        endpoint_path = str(scope.get("path", "/"))
        raw_body = await self._read_request_body(receive)
        if raw_body is None:
            return
        incoming_headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        request_max_tokens_cap = None
        raw_token_cap = incoming_headers.get(INTERNAL_MAX_TOKENS_HEADER)
        if isinstance(raw_token_cap, str) and raw_token_cap.isdigit():
            parsed_token_cap = int(raw_token_cap)
            if parsed_token_cap > 0:
                request_max_tokens_cap = parsed_token_cap
        request_protocol = incoming_headers.get(INTERNAL_PROTOCOL_HEADER)
        outgoing_body = raw_body
        applied = AppliedGenerationPolicy({}, (), {})
        request_json: dict[str, Any] | None = None
        if method in {"POST", "PUT", "PATCH"} and endpoint_path.endswith(
            GENERATION_ENDPOINT_SUFFIXES
        ):
            try:
                parsed = json.loads(raw_body or b"{}")
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                request_json = parsed
                applied = apply_generation_policy(
                    endpoint_path,
                    parsed,
                    self.generation,
                    request_max_tokens_cap=request_max_tokens_cap,
                    request_protocol=request_protocol,
                )
                outgoing_body = json.dumps(applied.body, ensure_ascii=False).encode("utf-8")

        headers = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in scope.get("headers", [])
            if key.decode("latin-1").lower() not in IGNORED_REQUEST_HEADERS
        }
        headers.update(self.upstream_headers)
        target_url = urljoin(self.upstream_url, endpoint_path.lstrip("/"))
        query = scope.get("query_string", b"")
        if query:
            target_url += "?" + query.decode("ascii")
        client = self.client
        if client is None:
            client = self.httpx.AsyncClient(timeout=self.timeout)
            self.client = client
        upstream_request = client.build_request(
            method,
            target_url,
            headers=headers,
            content=outgoing_body,
        )
        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except self.httpx.RequestError as exc:
            self._write_audit(
                started,
                request_id,
                method,
                endpoint_path,
                502,
                applied,
                error_type=type(exc).__name__,
            )
            await self._send_json(
                send,
                502,
                {
                    "error": {
                        "message": "The configured vLLM upstream is unavailable",
                        "type": "upstream_unavailable",
                        "request_id": request_id,
                    }
                },
            )
            return

        prompt_text = extract_request_text(request_json) if request_json else ""
        self._write_audit(
            started,
            request_id,
            method,
            endpoint_path,
            upstream_response.status_code,
            applied,
            prompt_text=prompt_text,
        )
        response_headers = [
            (key.encode("latin-1"), value.encode("latin-1"))
            for key, value in upstream_response.headers.items()
            if key.lower() in FORWARDED_RESPONSE_HEADERS
        ]
        await send(
            {
                "type": "http.response.start",
                "status": upstream_response.status_code,
                "headers": response_headers,
            }
        )
        try:
            async for chunk in upstream_response.aiter_raw():
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
            await send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )
        finally:
            await upstream_response.aclose()

    def _write_audit(
        self,
        started: float,
        request_id: str,
        method: str,
        path: str,
        status: int,
        applied: AppliedGenerationPolicy,
        *,
        prompt_text: str = "",
        error_type: str | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "matched_rules": applied.matched_rules,
            "changed_parameters": applied.changed_parameters,
            "effective_generation_parameters": {
                key: applied.body[key]
                for key in sorted(AUDITED_GENERATION_PARAMETERS)
                if key in applied.body
            },
            "prompt_chars": len(prompt_text),
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if prompt_text
            else None,
        }
        if self.prompt_preview_chars and prompt_text:
            record["prompt_preview"] = prompt_text[: self.prompt_preview_chars]
        if error_type:
            record["error_type"] = error_type
        self.audit.write(record)


def create_proxy_app(config: Mapping[str, Any]) -> ParameterProxyASGI:
    return ParameterProxyASGI(config)


def resolve_upstream_max_model_len(config: Mapping[str, Any]) -> int | None:
    generation = config["vllm"]["generation"]
    explicit = generation.get("max_context_tokens")
    if isinstance(explicit, int) and explicit > 1:
        return explicit
    try:
        import httpx

        vllm_config = config["vllm"]
        headers = _optional_bearer_headers(vllm_config)
        response = httpx.get(
            str(vllm_config["upstream_url"]).rstrip("/") + "/v1/models",
            headers=headers,
            timeout=3.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    models = payload.get("data", []) if isinstance(payload, dict) else []
    context_lengths = [
        item["max_model_len"]
        for item in models
        if isinstance(item, dict)
        and isinstance(item.get("max_model_len"), int)
        and item["max_model_len"] > 1
    ]
    return min(context_lengths) if context_lengths else None


def prepare_parameter_proxy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(dict(config))
    max_model_len = resolve_upstream_max_model_len(prepared)
    if max_model_len is not None:
        prepared["vllm"]["generation"][
            "_resolved_max_context_tokens"
        ] = max_model_len
    return prepared


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def run_proxy(config: Mapping[str, Any], host: str | None = None, port: int | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Proxy mode requires MinerU's uvicorn dependency") from exc
    prepared_config = prepare_parameter_proxy_config(config)
    proxy_config = prepared_config["vllm"]["proxy"]
    resolved_host = host or str(proxy_config.get("host", "127.0.0.1"))
    if resolved_host not in {"127.0.0.1", "localhost", "::1"} and not proxy_config.get(
        "allow_public_bind", False
    ):
        raise WorkflowConfigError(
            "Public proxy binding requires vllm.proxy.allow_public_bind=true"
        )
    resolved_port = int(proxy_config.get("port", 30001) if port is None else port)
    if resolved_port == 0:
        resolved_port = find_free_port(resolved_host)
    uvicorn.run(
        create_proxy_app(prepared_config),
        host=resolved_host,
        port=resolved_port,
    )


def _option_args(options: Mapping[str, Any]) -> list[str]:
    args = []
    for key, value in options.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, list):
            for item in value:
                args.extend([option, str(item)])
        elif isinstance(value, dict):
            args.extend(
                [option, json.dumps(value, ensure_ascii=True, separators=(",", ":"))]
            )
        elif value is not None:
            args.extend([option, str(value)])
    return args


def build_vllm_server_command(config: Mapping[str, Any]) -> list[str]:
    vllm_config = config["vllm"]
    server_args = vllm_config.get("server_args", {})
    if not isinstance(server_args, dict):
        raise WorkflowConfigError("vllm.server_args must be a JSON object")
    return [
        sys.executable,
        "-m",
        "mineru.cli.vlm_server",
        "--engine",
        "vllm",
        *_option_args(server_args),
    ]


def build_mineru_command(
    config: Mapping[str, Any], input_path: str | Path, output_path: str | Path, proxy_url: str
) -> list[str]:
    mineru_config = config["mineru"]
    command = [
        sys.executable,
        "-m",
        "mineru.cli.client",
        "--path",
        str(Path(input_path).expanduser().resolve()),
        "--output",
        str(Path(output_path).expanduser().resolve()),
        "--backend",
        str(mineru_config.get("backend", "hybrid-http-client")),
        "--effort",
        str(mineru_config.get("effort", "high")),
        "--method",
        str(mineru_config.get("method", "auto")),
        "--lang",
        str(mineru_config.get("lang", "ch")),
        "--url",
        proxy_url,
        "--formula",
        str(bool(mineru_config.get("formula", True))),
        "--table",
        str(bool(mineru_config.get("table", True))),
        "--image-analysis",
        str(bool(mineru_config.get("image_analysis", True))),
    ]
    extra_args = mineru_config.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise WorkflowConfigError("mineru.extra_args must be an array of strings")
    command.extend(extra_args)
    return command


def build_pipeline_command(
    config: Mapping[str, Any], input_path: str | Path, output_path: str | Path
) -> list[str]:
    mineru_config = config["mineru"]
    return [
        sys.executable,
        "-m",
        "mineru.cli.client",
        "--path",
        str(Path(input_path).expanduser().resolve()),
        "--output",
        str(Path(output_path).expanduser().resolve()),
        "--backend",
        "pipeline",
        "--method",
        "ocr",
        "--lang",
        str(mineru_config.get("lang", "ch")),
        "--formula",
        str(bool(mineru_config.get("formula", True))),
        "--table",
        str(bool(mineru_config.get("table", True))),
    ]


def _wait_for_proxy(host: str, port: int, timeout_seconds: float = 15) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"vLLM parameter proxy did not start on {host}:{port}")


def _start_parameter_proxy(config: Mapping[str, Any]):
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("This command requires MinerU's uvicorn dependency") from exc
    prepared_config = prepare_parameter_proxy_config(config)
    proxy_config = prepared_config["vllm"]["proxy"]
    host = str(proxy_config.get("host", "127.0.0.1"))
    configured_port = int(proxy_config.get("port", 30001))
    port = configured_port or find_free_port(host)
    server = uvicorn.Server(
        uvicorn.Config(
            create_proxy_app(prepared_config),
            host=host,
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, name="mineru-vllm-proxy", daemon=True)
    thread.start()
    try:
        _wait_for_proxy(host, port)
    except Exception:
        server.should_exit = True
        thread.join(timeout=10)
        raise
    return server, thread, f"http://{host}:{port}"


def _stop_parameter_proxy(server: Any, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("vLLM parameter proxy did not stop within 10 seconds")


def _run_mineru_command(command: Sequence[str]) -> None:
    result = subprocess.run(command, check=False, cwd=REPOSITORY_ROOT)
    if result.returncode:
        raise RuntimeError(
            f"MinerU command failed with exit code {result.returncode}: "
            + " ".join(command[2:])
        )


def _index_middle_json(root: Path) -> dict[str, Path]:
    suffix = "_middle.json"
    indexed: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        stem = path.name[: -len(suffix)]
        if stem in indexed:
            raise RuntimeError(
                f"Duplicate middle JSON document stem {stem!r} under {root}"
            )
        indexed[stem] = path
    return indexed


def _index_input_documents(input_path: str | Path) -> dict[str, Path]:
    path = Path(input_path).expanduser().resolve()
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    candidates = [path] if path.is_file() else [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in supported
    ]
    raw_stems = [_truncate_utf8(item.stem, 200) or item.stem for item in candidates]
    effective_stems = _uniquify_stems(raw_stems)
    indexed: dict[str, Path] = {}
    for candidate, effective_stem in zip(candidates, effective_stems):
        indexed[effective_stem] = candidate
    return indexed


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError as exc:
            truncated = truncated[: exc.start]
    return ""


def _uniquify_stems(stems: Sequence[str]) -> list[str]:
    raw_keys = {stem.casefold() for stem in stems}
    occurrences: dict[str, int] = {}
    assigned = set()
    result = []
    for stem in stems:
        key = stem.casefold()
        seen = occurrences.get(key, 0)
        occurrences[key] = seen + 1
        if seen == 0 and key not in assigned:
            candidate = stem
        else:
            suffix_number = seen + 1
            while True:
                suffix = f"_{suffix_number}"
                candidate = _truncate_utf8(
                    stem,
                    200 - len(suffix.encode("utf-8")),
                ) + suffix
                candidate_key = candidate.casefold()
                if candidate_key not in raw_keys and candidate_key not in assigned:
                    break
                suffix_number += 1
        assigned.add(candidate.casefold())
        result.append(candidate)
    return result


def _require_empty_output(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RuntimeError(
            f"{label} output is not empty: {path}. Use a new directory or the fuse command."
        )


def _regenerate_fused_outputs(
    parse_dir: Path,
    document_stem: str,
    source_document: Path | None = None,
    semantic_markdown_config: Mapping[str, Any] | None = None,
    page_sorting_config: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from mineru.cli.client_side_output import regenerate_client_side_outputs

    generated = list(regenerate_client_side_outputs(parse_dir, document_stem))
    generated.extend(
        _generate_semantic_markdown_outputs(
            parse_dir,
            document_stem,
            semantic_markdown_config or {},
        )
    )
    generated.extend(
        _generate_page_sorting_outputs(
            parse_dir,
            document_stem,
            page_sorting_config or {},
        )
    )
    generated.extend(
        _generate_fused_visualizations(
            parse_dir,
            document_stem,
            source_document,
        )
    )
    return tuple(generated)


def _generate_page_sorting_outputs(
    parse_dir: Path,
    document_stem: str,
    config: Mapping[str, Any],
) -> tuple[Path, ...]:
    if not config.get("enabled", False):
        return ()
    from projects.custom_hybrid.page_sorting import write_page_sorting_reports

    middle_path = parse_dir / f"{document_stem}_middle.json"
    semantic_report_path = parse_dir / f"{document_stem}_semantic_report.json"
    try:
        return write_page_sorting_reports(
            middle_path,
            config=config,
            semantic_report_path=(
                semantic_report_path if semantic_report_path.is_file() else None
            ),
        )
    except Exception as exc:
        # This optional report must not invalidate otherwise usable fused output.
        error_path = parse_dir / f"{document_stem}_sorting_report.json"
        error_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "mode": "report_only",
                    "status": "error",
                    "can_auto_sort": False,
                    "report_only": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return (error_path,)


def _generate_semantic_markdown_outputs(
    parse_dir: Path,
    document_stem: str,
    config: Mapping[str, Any],
) -> tuple[Path, ...]:
    if not config.get("enabled", False):
        return ()
    from projects.custom_hybrid.semantic_markdown import write_semantic_markdown

    middle_path = parse_dir / f"{document_stem}_middle.json"
    primary_path = parse_dir / f"{document_stem}.md"
    replace_primary = bool(config.get("replace_primary", True))
    preserve_native = bool(config.get("preserve_native", True))
    report_path = parse_dir / f"{document_stem}_semantic_report.json"
    generated = []
    if replace_primary:
        if preserve_native and primary_path.is_file():
            native_path = parse_dir / f"{document_stem}_native.md"
            shutil.copy2(primary_path, native_path)
            generated.append(native_path)
        write_semantic_markdown(
            middle_path,
            primary_path,
            report_path,
        )
    else:
        semantic_path = parse_dir / f"{document_stem}_semantic.md"
        write_semantic_markdown(
            middle_path,
            semantic_path,
            report_path,
        )
        generated.append(semantic_path)
    generated.append(report_path)
    return tuple(generated)


def _resolve_visualization_pdf(
    parse_dir: Path,
    document_stem: str,
    source_document: Path | None,
) -> Path | None:
    candidates = [parse_dir / f"{document_stem}_origin.pdf"]
    if source_document is not None and source_document.suffix.lower() == ".pdf":
        candidates.append(source_document)
    return next((path for path in candidates if path.is_file()), None)


def _validate_visualization_pdf(path: Path, expected_pages: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Visualization renderer did not create {path.name}")
    from pypdf import PdfReader

    actual_pages = len(PdfReader(str(path)).pages)
    if actual_pages != expected_pages:
        raise RuntimeError(
            f"Visualization page count mismatch for {path.name}: "
            f"expected {expected_pages}, got {actual_pages}"
        )


def _generate_fused_visualizations(
    parse_dir: Path,
    document_stem: str,
    source_document: Path | None = None,
) -> tuple[Path, ...]:
    source_pdf = _resolve_visualization_pdf(
        parse_dir,
        document_stem,
        source_document,
    )
    if source_pdf is None:
        return ()

    from mineru.utils.draw_bbox import (
        BBOX_RENDERER_VERSION,
        draw_form_cell_bbox,
        draw_form_region_bbox,
        draw_layout_bbox,
        draw_span_bbox,
    )

    middle_json_path = parse_dir / f"{document_stem}_middle.json"
    middle_json = json.loads(middle_json_path.read_text(encoding="utf-8"))
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("Fused middle JSON must contain pdf_info for visualization")
    pdf_bytes = source_pdf.read_bytes()
    layout_path = parse_dir / f"{document_stem}_layout.pdf"
    span_path = parse_dir / f"{document_stem}_span.pdf"
    form_path = parse_dir / f"{document_stem}_forms.pdf"
    form_cell_path = parse_dir / f"{document_stem}_form_cells.pdf"
    span_temp = parse_dir / f".{document_stem}-{uuid.uuid4().hex}-span.pdf"
    layout_temp = parse_dir / f".{document_stem}-{uuid.uuid4().hex}-layout.pdf"
    form_temp = parse_dir / f".{document_stem}-{uuid.uuid4().hex}-forms.pdf"
    form_cell_temp = (
        parse_dir / f".{document_stem}-{uuid.uuid4().hex}-form-cells.pdf"
    )

    try:
        draw_span_bbox(pdf_info, pdf_bytes, str(parse_dir), span_temp.name)
        _validate_visualization_pdf(span_temp, len(pdf_info))
        span_temp.replace(span_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate {span_path.name} from {source_pdf.name}: {exc}"
        ) from exc
    finally:
        span_temp.unlink(missing_ok=True)

    generated = [span_path]
    try:
        draw_form_region_bbox(
            pdf_info,
            pdf_bytes,
            str(parse_dir),
            form_temp.name,
        )
        _validate_visualization_pdf(form_temp, len(pdf_info))
        form_temp.replace(form_path)
        generated.append(form_path)
    except Exception:
        # The normal span preview remains usable if this diagnostic overlay fails.
        form_temp.unlink(missing_ok=True)
    finally:
        form_temp.unlink(missing_ok=True)
    try:
        draw_form_cell_bbox(
            pdf_info,
            pdf_bytes,
            str(parse_dir),
            form_cell_temp.name,
        )
        _validate_visualization_pdf(form_cell_temp, len(pdf_info))
        form_cell_temp.replace(form_cell_path)
        generated.append(form_cell_path)
    except Exception:
        form_cell_temp.unlink(missing_ok=True)
    finally:
        form_cell_temp.unlink(missing_ok=True)
    try:
        draw_layout_bbox(pdf_info, pdf_bytes, str(parse_dir), layout_temp.name)
        _validate_visualization_pdf(layout_temp, len(pdf_info))
        layout_temp.replace(layout_path)
        generated.insert(0, layout_path)
    except Exception:
        # span.pdf is the required UI artifact; layout.pdf remains best-effort.
        layout_temp.unlink(missing_ok=True)
    finally:
        layout_temp.unlink(missing_ok=True)
    marker_path = parse_dir / f"{document_stem}_visualization.json"
    marker_path.write_text(
        json.dumps(
            {
                "bbox_renderer_version": BBOX_RENDERER_VERSION,
                "form_detector_version": FORM_DETECTOR_VERSION,
                "form_segmenter_version": FORM_SEGMENTER_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tuple(generated)


def _visualization_renderer_is_current(parse_dir: Path, stem: str) -> bool:
    from mineru.utils.draw_bbox import BBOX_RENDERER_VERSION

    marker_path = parse_dir / f"{stem}_visualization.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        marker.get("bbox_renderer_version") == BBOX_RENDERER_VERSION
        and marker.get("form_detector_version") == FORM_DETECTOR_VERSION
        and marker.get("form_segmenter_version") == FORM_SEGMENTER_VERSION
    )


def regenerate_fused_visualizations(
    fused_root: str | Path,
    input_path: str | Path,
) -> tuple[Path, ...]:
    root = Path(fused_root).expanduser().resolve()
    documents = _index_input_documents(input_path)
    ocr_root = root.parent / "ocr"
    ocr_files = _index_middle_json(ocr_root) if ocr_root.is_dir() else {}
    generated = []
    for stem, middle_path in _index_middle_json(root).items():
        metadata_changed = False
        fused_middle = None
        ocr_path = ocr_files.get(stem)
        if ocr_path is not None:
            fused_middle = json.loads(middle_path.read_text(encoding="utf-8"))
            ocr_middle = json.loads(ocr_path.read_text(encoding="utf-8"))
            _attached_cells, metadata_changed = recover_table_cell_geometry(
                fused_middle,
                ocr_middle,
            )
        document_path = documents.get(stem)
        if document_path is not None:
            if fused_middle is None:
                fused_middle = json.loads(middle_path.read_text(encoding="utf-8"))
            try:
                structure_report = annotate_form_structure(
                    fused_middle,
                    document_path,
                )
                metadata_changed = metadata_changed or structure_report["changed"]
            except Exception:
                # Old tasks can still repair their normal previews if detection fails.
                pass
        if metadata_changed and fused_middle is not None:
            middle_path.write_text(
                json.dumps(fused_middle, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
        span_path = middle_path.parent / f"{stem}_span.pdf"
        layout_path = middle_path.parent / f"{stem}_layout.pdf"
        form_path = middle_path.parent / f"{stem}_forms.pdf"
        form_cell_path = middle_path.parent / f"{stem}_form_cells.pdf"
        if (
            metadata_changed
            or not span_path.is_file()
            or not form_path.is_file()
            or not form_cell_path.is_file()
            or not _visualization_renderer_is_current(middle_path.parent, stem)
        ):
            generated.extend(
                _generate_fused_visualizations(
                    middle_path.parent,
                    stem,
                    documents.get(stem),
                )
            )
        else:
            if layout_path.is_file():
                generated.append(layout_path)
            generated.append(span_path)
            if form_path.is_file():
                generated.append(form_path)
            if form_cell_path.is_file():
                generated.append(form_cell_path)
    return tuple(generated)


def fuse_output_trees(
    config: Mapping[str, Any],
    input_path: str | Path,
    hybrid_root: Path,
    ocr_root: Path,
    fused_root: Path,
    proxy_url: str,
) -> dict[str, Any]:
    _require_empty_output(fused_root, "Fused")
    shutil.copytree(hybrid_root, fused_root, dirs_exist_ok=True)
    hybrid_files = _index_middle_json(hybrid_root)
    ocr_files = _index_middle_json(ocr_root)
    fused_files = _index_middle_json(fused_root)
    documents = _index_input_documents(input_path)
    fusion_config = config.get("fusion", {})
    settings = FusionSettings.from_mapping(fusion_config)
    verifier_config = fusion_config.get("verifier", {})
    recognizer_config = fusion_config.get("recognizer", {})
    recovery_config = fusion_config.get("recovery", {})
    form_detection_config = fusion_config.get("form_detection", {})
    form_detection_enabled = form_detection_config.get("enabled", True)
    form_detection_settings = FormDetectionSettings.from_mapping(
        form_detection_config
    )
    summary = {"documents": {}, "failed": {}}

    for stem, hybrid_path in hybrid_files.items():
        ocr_path = ocr_files.get(stem)
        fused_path = fused_files.get(stem)
        document_path = documents.get(stem)
        if ocr_path is None or fused_path is None:
            summary["failed"][stem] = "missing matching OCR or fused middle JSON"
            continue
        vision_document_path = (
            _resolve_visualization_pdf(
                fused_path.parent,
                stem,
                document_path,
            )
            or document_path
        )
        verifier = None
        recognizer = None
        recovery_reviewer = None
        try:
            if (
                verifier_config.get("enabled", False)
                and vision_document_path is not None
            ):
                verifier_base_url = str(verifier_config.get("base_url") or proxy_url)
                verifier = OpenAIVisionVerifier(
                    verifier_base_url,
                    vision_document_path,
                    verifier_config,
                )
            if (
                settings.bbox_recognition_enabled
                and vision_document_path is not None
            ):
                recognizer_base_url = str(
                    recognizer_config.get("base_url") or proxy_url
                )
                recognizer = OpenAIBBoxRecognizer(
                    recognizer_base_url,
                    vision_document_path,
                    recognizer_config,
                )
            if settings.bbox_recovery_enabled and vision_document_path is not None:
                effective_recovery_config = dict(recognizer_config)
                effective_recovery_config.update(recovery_config)
                recovery_base_url = str(
                    recovery_config.get("base_url")
                    or recognizer_config.get("base_url")
                    or proxy_url
                )
                recovery_kwargs = {}
                if (
                    recognizer is not None
                    and recovery_config.get("share_recognizer_page_cache", True)
                ):
                    shared_provider = getattr(
                        recognizer,
                        "target_crop_provider",
                        None,
                    )
                    if shared_provider is not None:
                        recovery_kwargs["page_provider"] = shared_provider
                recovery_reviewer = OpenAIBBoxRecoveryReviewer(
                    recovery_base_url,
                    vision_document_path,
                    effective_recovery_config,
                    **recovery_kwargs,
                )
            hybrid_middle = json.loads(hybrid_path.read_text(encoding="utf-8"))
            ocr_middle = json.loads(ocr_path.read_text(encoding="utf-8"))
            structure_report = None
            structure_error = None
            if (
                form_detection_enabled
                and settings.mode == "bbox_vlm"
                and vision_document_path is not None
            ):
                try:
                    demote_narrative_false_tables(ocr_middle)
                    structure_report = annotate_form_structure(
                        ocr_middle,
                        vision_document_path,
                        form_detection_settings,
                    )
                except Exception as exc:
                    structure_error = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            fused_middle, report = fuse_middle_json(
                hybrid_middle,
                ocr_middle,
                settings,
                verifier=verifier,
                candidate_chooser=verifier.choose_candidate if verifier else None,
                table_cell_candidate_chooser=(
                    verifier.choose_table_cell if verifier else None
                ),
                page_reconciler=verifier.reconcile_page if verifier else None,
                bbox_recognizer=recognizer if recognizer else None,
                bbox_recovery_reviewer=(
                    recovery_reviewer if recovery_reviewer else None
                ),
            )
            if form_detection_enabled:
                if vision_document_path is None:
                    report["form_detection"] = {
                        "status": "skipped",
                        "reason": "source_document_unavailable",
                    }
                    report["form_segmentation"] = {
                        "status": "skipped",
                        "reason": "source_document_unavailable",
                    }
                elif structure_report is not None:
                    report["form_detection"] = structure_report[
                        "form_detection"
                    ]
                    report["form_segmentation"] = structure_report[
                        "form_segmentation"
                    ]
                elif structure_error is not None:
                    report["form_detection"] = structure_error
                    report["form_segmentation"] = dict(structure_error)
                else:
                    try:
                        structure_report = annotate_form_structure(
                            fused_middle,
                            vision_document_path,
                            form_detection_settings,
                        )
                        report["form_detection"] = structure_report[
                            "form_detection"
                        ]
                        report["form_segmentation"] = structure_report[
                            "form_segmentation"
                        ]
                    except Exception as exc:
                        error_report = {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        report["form_detection"] = error_report
                        report["form_segmentation"] = dict(error_report)
            fused_path.write_text(
                json.dumps(fused_middle, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            report_path = fused_path.with_name(f"{stem}_fusion.json")
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            generated_files = _regenerate_fused_outputs(
                fused_path.parent,
                stem,
                vision_document_path,
                fusion_config.get("semantic_markdown", {}),
                fusion_config.get("page_sorting", {}),
            )
            document_summary = {
                "middle_json": str(fused_path),
                "report": str(report_path),
                "counts": report["counts"],
                "form_detection": report.get("form_detection"),
                "form_segmentation": report.get("form_segmentation"),
                "artifacts": [str(path) for path in generated_files],
            }
            sorting_report_path = fused_path.with_name(
                f"{stem}_sorting_report.json"
            )
            sorting_manifest_path = fused_path.with_name(
                f"{stem}_sorting_manifest.json"
            )
            if (
                fusion_config.get("page_sorting", {}).get("enabled", False)
                and sorting_report_path.is_file()
            ):
                sorting_summary = json.loads(
                    sorting_report_path.read_text(encoding="utf-8")
                )
                document_summary["page_sorting"] = {
                    "status": sorting_summary.get("status"),
                    "can_auto_sort": sorting_summary.get("can_auto_sort", False),
                    "can_auto_group": sorting_summary.get("can_auto_group", False),
                    "grouping_strategy": sorting_summary.get("grouping_strategy"),
                    "grouping_status": sorting_summary.get("grouping_status"),
                    "ordering_status": sorting_summary.get("ordering_status"),
                    "packet_pagination_status": (
                        sorting_summary.get("packet_pagination", {}).get("status")
                        if isinstance(
                            sorting_summary.get("packet_pagination"),
                            Mapping,
                        )
                        else None
                    ),
                    "packet_wrapper_role": (
                        sorting_summary.get("packet_wrapper_policy", {}).get("role")
                        if isinstance(
                            sorting_summary.get("packet_wrapper_policy"),
                            Mapping,
                        )
                        else None
                    ),
                    "manifest": (
                        str(sorting_manifest_path)
                        if sorting_manifest_path.is_file()
                        else None
                    ),
                    "report": str(sorting_report_path),
                }
            summary["documents"][stem] = document_summary
        except Exception as exc:
            summary["failed"][stem] = f"{type(exc).__name__}: {exc}"
        finally:
            if verifier is not None:
                verifier.close()
            if recognizer is not None:
                recognizer.close()
            if recovery_reviewer is not None:
                recovery_reviewer.close()
    return summary


def run_extract(config: Mapping[str, Any], input_path: str | Path, output_path: str | Path) -> int:
    fusion_config = config.get("fusion", {})
    fusion_enabled = bool(fusion_config.get("enabled", False))
    fusion_mode = normalize_fusion_mode(
        fusion_config.get("mode", "hybrid_fusion")
    )
    output_root = Path(output_path).expanduser().resolve()
    if fusion_enabled:
        child_names = (
            ("ocr", "fused")
            if fusion_mode == "bbox_vlm"
            else ("hybrid", "ocr", "fused")
        )
        for child_name in child_names:
            _require_empty_output(output_root / child_name, child_name.capitalize())
    server, thread, proxy_url = _start_parameter_proxy(config)
    try:
        if fusion_enabled and fusion_mode == "bbox_vlm":
            ocr_output = output_root / "ocr"
            fused_output = output_root / "fused"
            _run_mineru_command(build_pipeline_command(config, input_path, ocr_output))
            summary = fuse_output_trees(
                config,
                input_path,
                ocr_output,
                ocr_output,
                fused_output,
                proxy_url,
            )
            summary_path = output_root / "fusion_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if summary["failed"]:
                raise RuntimeError(
                    f"Fusion failed for {len(summary['failed'])} document(s); see {summary_path}"
                )
            return 0

        hybrid_output = output_root / "hybrid" if fusion_enabled else output_root
        _run_mineru_command(
            build_mineru_command(config, input_path, hybrid_output, proxy_url)
        )
        if not fusion_enabled:
            return 0

        ocr_output = output_root / "ocr"
        fused_output = output_root / "fused"
        _run_mineru_command(build_pipeline_command(config, input_path, ocr_output))
        summary = fuse_output_trees(
            config,
            input_path,
            hybrid_output,
            ocr_output,
            fused_output,
            proxy_url,
        )
        summary_path = output_root / "fusion_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if summary["failed"]:
            raise RuntimeError(
                f"Fusion failed for {len(summary['failed'])} document(s); see {summary_path}"
            )
        return 0
    finally:
        _stop_parameter_proxy(server, thread)


def normalize_markdown(text: str, options: Mapping[str, Any]) -> str:
    if options.get("unicode_nfkc", True):
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if options.get("trim_trailing_whitespace", True):
        text = "\n".join(line.rstrip() for line in text.split("\n"))
    if options.get("collapse_blank_lines", True):
        text = re.sub(r"\n{3,}", "\n\n", text)
    if options.get("collapse_whitespace", False):
        text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def levenshtein_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Return exact edit distance using Myers' bit-parallel algorithm."""
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left
    pattern_masks: dict[Any, int] = {}
    for index, item in enumerate(right):
        pattern_masks[item] = pattern_masks.get(item, 0) | (1 << index)

    positive = ~0
    negative = 0
    score = len(right)
    last_bit = 1 << (len(right) - 1)
    for item in left:
        matches = pattern_masks.get(item, 0)
        combined = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & last_bit:
            score += 1
        elif negative_horizontal & last_bit:
            score -= 1
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(combined | positive_horizontal)
        negative = positive_horizontal & combined
    return score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _counter_f1(reference: Iterable[str], candidate: Iterable[str]) -> dict[str, float]:
    reference_counter = Counter(reference)
    candidate_counter = Counter(candidate)
    common = sum((reference_counter & candidate_counter).values())
    reference_total = sum(reference_counter.values())
    candidate_total = sum(candidate_counter.values())
    precision = common / candidate_total if candidate_total else float(reference_total == 0)
    recall = common / reference_total if reference_total else float(candidate_total == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _normalize_table_cell(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def evaluate_table_texts(reference: str, candidate: str) -> dict[str, Any]:
    reference_tables = extract_table_snapshots(reference)
    candidate_tables = extract_table_snapshots(candidate)
    count_scores = _counter_f1(
        ["table"] * len(reference_tables),
        ["table"] * len(candidate_tables),
    )
    structure_scores = _counter_f1(
        [repr(table.structure_signature) for table in reference_tables],
        [repr(table.structure_signature) for table in candidate_tables],
    )

    candidate_by_structure: dict[Any, list[int]] = {}
    for index, table in enumerate(candidate_tables):
        candidate_by_structure.setdefault(table.structure_signature, []).append(index)
    used_candidates = set()
    text_distance = 0
    text_denominator = 0
    exact_cells = 0
    reference_cells = sum(len(table.cells) for table in reference_tables)
    candidate_cells = sum(len(table.cells) for table in candidate_tables)

    for reference_table in reference_tables:
        matching_indices = candidate_by_structure.get(reference_table.structure_signature, [])
        candidate_index = next(
            (index for index in matching_indices if index not in used_candidates),
            None,
        )
        if candidate_index is None:
            for _key, text in reference_table.cells:
                normalized = _normalize_table_cell(text)
                text_distance += max(len(normalized), 1)
                text_denominator += max(len(normalized), 1)
            continue
        used_candidates.add(candidate_index)
        candidate_map = candidate_tables[candidate_index].cell_map
        for key, reference_text in reference_table.cells:
            normalized_reference = _normalize_table_cell(reference_text)
            normalized_candidate = _normalize_table_cell(candidate_map.get(key, ""))
            text_distance += levenshtein_distance(
                normalized_reference,
                normalized_candidate,
            )
            text_denominator += max(
                len(normalized_reference),
                len(normalized_candidate),
                1,
            )
            exact_cells += int(normalized_reference == normalized_candidate)

    for index, table in enumerate(candidate_tables):
        if index in used_candidates:
            continue
        for _key, text in table.cells:
            normalized = _normalize_table_cell(text)
            text_distance += max(len(normalized), 1)
            text_denominator += max(len(normalized), 1)

    text_similarity = (
        max(0.0, 1.0 - text_distance / text_denominator)
        if text_denominator
        else 1.0
    )
    cell_exact_match = exact_cells / max(reference_cells, candidate_cells, 1)
    table_quality = (
        0.15 * count_scores["f1"]
        + 0.4 * structure_scores["f1"]
        + 0.45 * text_similarity
    )
    return {
        "reference_table_count": len(reference_tables),
        "candidate_table_count": len(candidate_tables),
        "table_count_precision": round(count_scores["precision"], 6),
        "table_count_recall": round(count_scores["recall"], 6),
        "table_count_f1": round(count_scores["f1"], 6),
        "table_structure_precision": round(structure_scores["precision"], 6),
        "table_structure_recall": round(structure_scores["recall"], 6),
        "table_structure_f1": round(structure_scores["f1"], 6),
        "table_cell_exact_match": round(cell_exact_match, 6),
        "table_cell_text_similarity": round(text_similarity, 6),
        "table_quality_score": round(table_quality, 6),
    }


def markdown_structure_tokens(text: str) -> list[str]:
    tokens = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            tokens.append("code_fence")
        if in_code:
            continue
        heading = re.match(r"^(#{1,6})\s+", stripped)
        if heading:
            tokens.append(f"heading_{len(heading.group(1))}")
        if re.match(r"^([-*+]\s+|\d+[.)]\s+)", stripped):
            tokens.append("list_item")
        if stripped.startswith(">"):
            tokens.append("blockquote")
        if stripped.count("|") >= 2:
            tokens.append("table_row")
        tokens.extend("display_math" for _ in re.finditer(r"\$\$", stripped))
        tokens.extend("image" for _ in re.finditer(r"!\[[^]]*]\([^)]+\)", stripped))
    return tokens


def evaluate_texts(reference: str, candidate: str, options: Mapping[str, Any]) -> dict[str, Any]:
    normalized_reference = normalize_markdown(reference, options)
    normalized_candidate = normalize_markdown(candidate, options)
    edit_distance = levenshtein_distance(normalized_reference, normalized_candidate)
    denominator = max(len(normalized_reference), 1)
    char_error_rate = edit_distance / denominator
    token_scores = _counter_f1(
        _tokenize(normalized_reference), _tokenize(normalized_candidate)
    )
    structure_scores = _counter_f1(
        markdown_structure_tokens(normalized_reference),
        markdown_structure_tokens(normalized_candidate),
    )
    reference_lines = [line for line in normalized_reference.splitlines() if line.strip()]
    candidate_lines = [line for line in normalized_candidate.splitlines() if line.strip()]
    line_order_similarity = SequenceMatcher(None, reference_lines, candidate_lines).ratio()
    char_similarity = max(0.0, 1.0 - char_error_rate)
    quality_score = (
        0.45 * char_similarity
        + 0.25 * token_scores["f1"]
        + 0.15 * line_order_similarity
        + 0.15 * structure_scores["f1"]
    )
    table_scores = evaluate_table_texts(normalized_reference, normalized_candidate)
    return {
        "quality_score": round(quality_score, 6),
        "char_error_rate": round(char_error_rate, 6),
        "edit_distance": edit_distance,
        "reference_chars": len(normalized_reference),
        "candidate_chars": len(normalized_candidate),
        "token_precision": round(token_scores["precision"], 6),
        "token_recall": round(token_scores["recall"], 6),
        "token_f1": round(token_scores["f1"], 6),
        "line_order_similarity": round(line_order_similarity, 6),
        "structure_f1": round(structure_scores["f1"], 6),
        **table_scores,
    }


def _markdown_files(path: Path) -> dict[str, Path]:
    if path.is_file():
        return {path.name: path}
    files: dict[str, Path] = {}
    for file_path in path.rglob("*.md"):
        name = file_path.name
        if name in files:
            raise ValueError(
                f"Duplicate Markdown filename {name!r} under {path}; use unique document stems"
            )
        files[name] = file_path
    return files


def evaluate_paths(
    reference_path: str | Path,
    candidate_path: str | Path,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    reference = Path(reference_path).expanduser().resolve()
    candidate = Path(candidate_path).expanduser().resolve()
    if not reference.exists() or not candidate.exists():
        raise FileNotFoundError("Reference and candidate paths must both exist")
    if reference.is_file() != candidate.is_file():
        raise ValueError("Reference and candidate must both be files or both be directories")
    if reference.is_file():
        pairs = [(reference.name, reference, candidate)]
        missing_reference: list[str] = []
        missing_candidate: list[str] = []
    else:
        reference_files = _markdown_files(reference)
        candidate_files = _markdown_files(candidate)
        shared = sorted(reference_files.keys() & candidate_files.keys())
        pairs = [(name, reference_files[name], candidate_files[name]) for name in shared]
        missing_reference = sorted(candidate_files.keys() - reference_files.keys())
        missing_candidate = sorted(reference_files.keys() - candidate_files.keys())
    if not pairs:
        raise ValueError("No matching Markdown files found")

    documents = []
    for name, reference_file, candidate_file in pairs:
        scores = evaluate_texts(
            reference_file.read_text(encoding="utf-8"),
            candidate_file.read_text(encoding="utf-8"),
            options,
        )
        documents.append({"name": name, **scores})
    aggregate_keys = [
        "quality_score",
        "char_error_rate",
        "token_precision",
        "token_recall",
        "token_f1",
        "line_order_similarity",
        "structure_f1",
    ]
    table_metric_keys = [
        "table_count_f1",
        "table_structure_f1",
        "table_cell_exact_match",
        "table_cell_text_similarity",
        "table_quality_score",
    ]
    aggregate = {
        key: round(sum(document[key] for document in documents) / len(documents), 6)
        for key in aggregate_keys
    }
    table_documents = [
        document
        for document in documents
        if document["reference_table_count"] or document["candidate_table_count"]
    ]
    aggregate.update(
        {
            key: round(
                sum(document[key] for document in table_documents)
                / len(table_documents),
                6,
            )
            if table_documents
            else 1.0
            for key in table_metric_keys
        }
    )
    return {
        "document_count": len(documents),
        "table_document_count": len(table_documents),
        "aggregate": aggregate,
        "documents": documents,
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
    }


def load_benchmark_manifest(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Benchmark manifest must be an object with version=1")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Benchmark manifest documents must be a non-empty array")
    seen_ids = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"Benchmark documents[{index}] must be an object")
        document_id = document.get("id")
        reference = document.get("reference")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"Benchmark documents[{index}].id is required")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate benchmark document id: {document_id}")
        seen_ids.add(document_id)
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"Benchmark documents[{index}].reference is required")
        weight = document.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or float(weight) <= 0:
            raise ValueError(f"Benchmark document {document_id} weight must be positive")
        tags = document.get("tags", ["all"])
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError(f"Benchmark document {document_id} tags must be strings")
    return manifest_path, documents


def _weighted_metric_aggregate(documents: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metric_keys = [
        "quality_score",
        "char_error_rate",
        "token_precision",
        "token_recall",
        "token_f1",
        "line_order_similarity",
        "structure_f1",
    ]
    table_metric_keys = [
        "table_count_f1",
        "table_structure_f1",
        "table_cell_exact_match",
        "table_cell_text_similarity",
        "table_quality_score",
    ]
    total_weight = sum(float(document["weight"]) for document in documents)
    aggregate = {
        key: round(
            sum(float(document[key]) * float(document["weight"]) for document in documents)
            / total_weight,
            6,
        )
        for key in metric_keys
    }
    table_documents = [
        document
        for document in documents
        if document["reference_table_count"] or document["candidate_table_count"]
    ]
    table_weight = sum(float(document["weight"]) for document in table_documents)
    aggregate.update(
        {
            key: round(
                sum(
                    float(document[key]) * float(document["weight"])
                    for document in table_documents
                )
                / table_weight,
                6,
            )
            if table_weight
            else 1.0
            for key in table_metric_keys
        }
    )
    return aggregate


def evaluate_benchmark_run(
    manifest_path: str | Path,
    candidate_path: str | Path,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_manifest, manifest_documents = load_benchmark_manifest(manifest_path)
    candidate_root = Path(candidate_path).expanduser().resolve()
    if not candidate_root.exists():
        raise FileNotFoundError(f"Candidate path does not exist: {candidate_root}")
    candidate_files = _markdown_files(candidate_root)
    results = []
    missing = []
    expected_weight = sum(float(document.get("weight", 1.0)) for document in manifest_documents)
    expected_category_weights: dict[str, float] = {}
    for document in manifest_documents:
        weight = float(document.get("weight", 1.0))
        for tag in document.get("tags", ["all"]):
            expected_category_weights[tag] = expected_category_weights.get(tag, 0.0) + weight
    for document in manifest_documents:
        document_id = document["id"]
        reference_path = (resolved_manifest.parent / document["reference"]).resolve()
        candidate_filename = str(document.get("candidate_filename", f"{document_id}.md"))
        candidate_file = candidate_files.get(candidate_filename)
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Benchmark reference does not exist for {document_id}: {reference_path}"
            )
        if candidate_file is None:
            missing.append(
                {
                    "id": document_id,
                    "candidate_filename": candidate_filename,
                    "weight": float(document.get("weight", 1.0)),
                    "tags": list(document.get("tags", ["all"])),
                }
            )
            continue
        scores = evaluate_texts(
            reference_path.read_text(encoding="utf-8"),
            candidate_file.read_text(encoding="utf-8"),
            options,
        )
        results.append(
            {
                "id": document_id,
                "weight": float(document.get("weight", 1.0)),
                "tags": list(document.get("tags", ["all"])),
                **scores,
            }
        )
    if not results:
        raise ValueError("Benchmark has no matched reference/candidate Markdown documents")
    categories: dict[str, list[dict[str, Any]]] = {}
    for document in results:
        for tag in document["tags"]:
            categories.setdefault(tag, []).append(document)
    aggregate = _weighted_metric_aggregate(results)
    matched_weight = sum(float(document["weight"]) for document in results)
    coverage = matched_weight / expected_weight
    category_reports = {}
    for tag, expected_category_weight in sorted(expected_category_weights.items()):
        tag_documents = categories.get(tag, [])
        category_aggregate = (
            _weighted_metric_aggregate(tag_documents)
            if tag_documents
            else {
                "quality_score": 0.0,
                "char_error_rate": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "token_f1": 0.0,
                "line_order_similarity": 0.0,
                "structure_f1": 0.0,
                "table_count_f1": 0.0,
                "table_structure_f1": 0.0,
                "table_cell_exact_match": 0.0,
                "table_cell_text_similarity": 0.0,
                "table_quality_score": 0.0,
            }
        )
        category_matched_weight = sum(float(document["weight"]) for document in tag_documents)
        category_coverage = category_matched_weight / expected_category_weight
        category_reports[tag] = {
            "document_count": len(tag_documents),
            "table_document_count": sum(
                1
                for document in tag_documents
                if document["reference_table_count"]
                or document["candidate_table_count"]
            ),
            "coverage": round(category_coverage, 6),
            "aggregate": category_aggregate,
            "coverage_adjusted_quality_score": round(
                category_aggregate["quality_score"] * category_coverage, 6
            ),
        }
    return {
        "candidate": str(candidate_root),
        "expected_document_count": len(manifest_documents),
        "document_count": len(results),
        "table_document_count": sum(
            1
            for document in results
            if document["reference_table_count"] or document["candidate_table_count"]
        ),
        "coverage": round(coverage, 6),
        "aggregate": aggregate,
        "coverage_adjusted_quality_score": round(
            aggregate["quality_score"] * coverage, 6
        ),
        "categories": category_reports,
        "documents": results,
        "missing": missing,
    }


def evaluate_benchmark_runs(
    manifest_path: str | Path,
    candidates: Sequence[str],
    options: Mapping[str, Any],
    regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one benchmark candidate is required")
    runs = {}
    for candidate_spec in candidates:
        if "=" in candidate_spec:
            label, raw_path = candidate_spec.split("=", 1)
        else:
            raw_path = candidate_spec
            label = Path(raw_path).name or "run"
        if not label or label in runs:
            raise ValueError(f"Candidate labels must be non-empty and unique: {label!r}")
        runs[label] = evaluate_benchmark_run(manifest_path, raw_path, options)
    leaderboard = sorted(
        (
            {
                "label": label,
                "quality_score": report["aggregate"]["quality_score"],
                "coverage_adjusted_quality_score": report[
                    "coverage_adjusted_quality_score"
                ],
                "char_error_rate": report["aggregate"]["char_error_rate"],
                "table_quality_score": report["aggregate"]["table_quality_score"],
                "coverage": report["coverage"],
                "document_count": report["document_count"],
            }
            for label, report in runs.items()
        ),
        key=lambda item: (
            -item["coverage_adjusted_quality_score"],
            item["char_error_rate"],
            item["label"],
        ),
    )
    baseline_label = next(iter(runs))
    baseline = runs[baseline_label]
    baseline_documents = {document["id"]: document for document in baseline["documents"]}
    comparisons = {}
    for label, report in runs.items():
        if label == baseline_label:
            continue
        document_deltas = []
        regressions = []
        report_documents = {document["id"]: document for document in report["documents"]}
        missing_ids = {item["id"] for item in report["missing"]}
        for document_id, baseline_document in baseline_documents.items():
            document = report_documents.get(document_id)
            if document is None and document_id not in missing_ids:
                continue
            if document is None:
                delta = round(-baseline_document["quality_score"], 6)
            else:
                delta = round(
                    document["quality_score"] - baseline_document["quality_score"], 6
                )
            item = {"id": document_id, "quality_score_delta": delta}
            document_deltas.append(item)
            if delta < -regression_tolerance:
                regressions.append(item)
        category_deltas = {}
        for tag, category in report["categories"].items():
            baseline_category = baseline["categories"].get(tag)
            if baseline_category is None:
                continue
            category_deltas[tag] = round(
                category["coverage_adjusted_quality_score"]
                - baseline_category["coverage_adjusted_quality_score"],
                6,
            )
        comparisons[label] = {
            "aggregate_quality_score_delta": round(
                report["coverage_adjusted_quality_score"]
                - baseline["coverage_adjusted_quality_score"],
                6,
            ),
            "category_quality_score_deltas": category_deltas,
            "document_deltas": document_deltas,
            "regressions": regressions,
        }
    return {
        "version": 1,
        "baseline": baseline_label,
        "regression_tolerance": regression_tolerance,
        "leaderboard": leaderboard,
        "comparisons": comparisons,
        "runs": runs,
    }


def build_sweep_variants(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    sweep_config = config.get("sweep", {})
    _validate_sweep_config(sweep_config)
    generation = sweep_config["generation"]
    parameter_names = list(generation)
    combinations = list(itertools.product(*(generation[name] for name in parameter_names)))
    max_runs = int(sweep_config.get("max_runs", 12))
    if len(combinations) > max_runs:
        raise WorkflowConfigError(
            f"Sweep expands to {len(combinations)} runs, exceeding sweep.max_runs={max_runs}"
        )
    variants = []
    for index, values in enumerate(combinations, 1):
        parameters = dict(zip(parameter_names, values))
        variant = copy.deepcopy(config)
        variant["vllm"]["generation"].setdefault("overrides", {}).update(parameters)
        variants.append((f"run-{index:03d}", parameters, variant))
    return variants


def run_sweep(
    config: Mapping[str, Any],
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    variants = build_sweep_variants(config)
    output_root = Path(output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fusion_enabled = bool(config.get("fusion", {}).get("enabled", False))
    shared_ocr_root = output_root / "shared-ocr"
    if fusion_enabled:
        _require_empty_output(shared_ocr_root, "Shared OCR")
        _run_mineru_command(build_pipeline_command(config, input_path, shared_ocr_root))

    summary: dict[str, Any] = {"version": 1, "runs": {}, "failed": {}}
    benchmark_candidates = []
    for run_id, parameters, variant in variants:
        run_root = output_root / run_id
        _require_empty_output(run_root, run_id)
        hybrid_root = run_root / "hybrid"
        fused_root = run_root / "fused"
        variant["vllm"]["audit_log"] = str(run_root / "vllm_requests.jsonl")
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "workflow.json").write_text(
            json.dumps(variant, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        server = thread = None
        try:
            server, thread, proxy_url = _start_parameter_proxy(variant)
            _run_mineru_command(
                build_mineru_command(variant, input_path, hybrid_root, proxy_url)
            )
            candidate_root = hybrid_root
            fusion_summary = None
            if fusion_enabled:
                fusion_summary = fuse_output_trees(
                    variant,
                    input_path,
                    hybrid_root,
                    shared_ocr_root,
                    fused_root,
                    proxy_url,
                )
                if fusion_summary["failed"]:
                    raise RuntimeError(
                        f"Fusion failed for {len(fusion_summary['failed'])} document(s)"
                    )
                candidate_root = fused_root
            summary["runs"][run_id] = {
                "parameters": parameters,
                "candidate": str(candidate_root),
                "fusion": fusion_summary,
            }
            benchmark_candidates.append(f"{run_id}={candidate_root}")
        except Exception as exc:
            summary["failed"][run_id] = {
                "parameters": parameters,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if server is not None and thread is not None:
                try:
                    _stop_parameter_proxy(server, thread)
                except Exception as exc:
                    summary["failed"].setdefault(
                        run_id,
                        {
                            "parameters": parameters,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

    if not summary["runs"]:
        raise RuntimeError("All sweep runs failed")
    if manifest_path is not None:
        summary["benchmark"] = evaluate_benchmark_runs(
            manifest_path,
            benchmark_candidates,
            config["evaluation"].get("normalization", {}),
            float(config["evaluation"].get("regression_tolerance", 0.0)),
        )
    summary_path = output_root / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_doctor(config: Mapping[str, Any]) -> dict[str, Any]:
    dependency_names = (
        "httpx",
        "uvicorn",
        "mineru_vl_utils",
        "PIL",
        "pypdf",
        "pypdfium2",
        "reportlab",
        "torch",
        "cv2",
        "numpy",
        "six",
    )
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in dependency_names
    }
    local_mineru_source = (REPOSITORY_ROOT / "mineru" / "__init__.py").is_file()
    upstream_url = str(config["vllm"]["upstream_url"]).rstrip("/")
    upstream_status: dict[str, Any] = {
        "url": upstream_url,
        "reachable": False,
        "status_code": None,
    }
    try:
        import httpx

        headers = _optional_bearer_headers(config["vllm"])
        response = httpx.get(
            upstream_url + "/v1/models", headers=headers, timeout=3.0
        )
        upstream_status["status_code"] = response.status_code
        upstream_status["reachable"] = response.is_success
        if response.is_success:
            payload = response.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            context_lengths = [
                item["max_model_len"]
                for item in models
                if isinstance(item, dict)
                and isinstance(item.get("max_model_len"), int)
                and item["max_model_len"] > 1
            ]
            upstream_status["max_model_len"] = (
                min(context_lengths) if context_lengths else None
            )
    except Exception as exc:
        upstream_status["error"] = type(exc).__name__

    fusion_config = config.get("fusion", {})
    fusion_mode = normalize_fusion_mode(
        fusion_config.get("mode", "hybrid_fusion")
        if isinstance(fusion_config, Mapping)
        else "hybrid_fusion"
    )
    recognizer_config = (
        fusion_config.get("recognizer", {})
        if isinstance(fusion_config, Mapping)
        else {}
    )
    recovery_config = (
        fusion_config.get("recovery", {})
        if isinstance(fusion_config, Mapping)
        else {}
    )
    recognizer_enabled = bool(
        isinstance(recognizer_config, Mapping)
        and (
            recognizer_config.get("enabled", False)
            or (
                isinstance(fusion_config, Mapping)
                and fusion_mode == "bbox_vlm"
            )
        )
    )
    recognizer_status: dict[str, Any] = {
        "enabled": recognizer_enabled,
        "ready": not recognizer_enabled,
        "capability_trial_required": recognizer_enabled,
    }
    if recognizer_enabled:
        recognizer_url = str(
            recognizer_config.get("base_url") or upstream_url
        ).rstrip("/")
        recognizer_status.update(
            {
                "url": recognizer_url,
                "reachable": False,
                "status_code": None,
                "configured_model": recognizer_config.get("model"),
                "structured_output_mode": recognizer_config.get(
                    "structured_output_mode",
                    "json_object",
                ),
                "structured_output_supported": False,
            }
        )
        try:
            import httpx

            recognizer_headers = _optional_bearer_headers(
                recognizer_config,
                config["vllm"],
            )
            models_response = httpx.get(
                recognizer_url + "/v1/models",
                headers=recognizer_headers,
                timeout=3.0,
            )
            recognizer_status["status_code"] = models_response.status_code
            recognizer_status["reachable"] = models_response.is_success
            model_ids = []
            context_lengths = []
            if models_response.is_success:
                payload = models_response.json()
                models = payload.get("data", []) if isinstance(payload, dict) else []
                model_ids = [
                    item["id"]
                    for item in models
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                context_lengths = [
                    item["max_model_len"]
                    for item in models
                    if isinstance(item, dict)
                    and isinstance(item.get("max_model_len"), int)
                    and item["max_model_len"] > 1
                ]
            recognizer_status["available_models"] = model_ids
            configured_model = recognizer_config.get("model")
            recognizer_status["model_available"] = (
                bool(model_ids)
                if not isinstance(configured_model, str) or not configured_model
                else configured_model in model_ids
            )
            recognizer_status["max_model_len"] = (
                min(context_lengths) if context_lengths else None
            )
            effective_model = (
                configured_model
                if isinstance(configured_model, str) and configured_model
                else model_ids[0] if model_ids else ""
            )
            protocol = str(recognizer_config.get("protocol", "auto"))
            if protocol == "auto":
                protocol = (
                    "mineru_native"
                    if "mineru" in effective_model.casefold()
                    else "structured"
                )
            recognizer_status["protocol"] = protocol
            recognizer_status["native_runtime"] = {
                "max_concurrency": recognizer_config.get(
                    "native_max_concurrency",
                    2,
                ),
                "max_requests_per_page": recognizer_config.get(
                    "native_max_requests_per_page",
                    0,
                ),
                "max_candidates_per_page": recognizer_config.get(
                    "native_max_candidates_per_page",
                    0,
                ),
                "max_requests_per_document": recognizer_config.get(
                    "max_requests_per_document",
                    80,
                ),
                "cache_enabled": recognizer_config.get(
                    "native_cache_enabled",
                    False,
                ),
                "cache_dir": recognizer_config.get(
                    "native_cache_dir",
                    "~/.cache/mineru/custom-hybrid/native-recognition",
                ),
                "cache_ttl_seconds": recognizer_config.get(
                    "native_cache_ttl_seconds",
                    2592000,
                ),
            }
            if protocol == "mineru_native":
                recognizer_status["structured_output_mode"] = "mineru_native"
                structured_supported = True
            else:
                openapi_response = httpx.get(
                    recognizer_url + "/openapi.json",
                    headers=recognizer_headers,
                    timeout=3.0,
                )
                openapi = (
                    openapi_response.json() if openapi_response.is_success else {}
                )
                schemas = (
                    openapi.get("components", {}).get("schemas", {})
                    if isinstance(openapi, dict)
                    else {}
                )
                request_properties = (
                    schemas.get("ChatCompletionRequest", {}).get("properties", {})
                    if isinstance(schemas, dict)
                    else {}
                )
                mode = recognizer_status["structured_output_mode"]
                if mode == "none":
                    structured_supported = True
                elif mode in {"structured_outputs", "regex"}:
                    structured_supported = "structured_outputs" in request_properties
                elif mode == "json_object":
                    structured_supported = "response_format" in request_properties
                else:
                    response_format_enum = (
                        schemas.get("ResponseFormat", {})
                        .get("properties", {})
                        .get("type", {})
                        .get("enum", [])
                    )
                    structured_supported = "json_schema" in response_format_enum
            recognizer_status["structured_output_supported"] = bool(
                structured_supported
            )
            recognizer_status["ready"] = bool(
                recognizer_status["reachable"]
                and recognizer_status["model_available"]
                and recognizer_status["structured_output_supported"]
            )
        except Exception as exc:
            recognizer_status["error"] = type(exc).__name__

    recovery_enabled = bool(
        isinstance(recovery_config, Mapping)
        and (
            recovery_config.get("enabled", False)
            or (
                isinstance(fusion_config, Mapping)
                and fusion_mode == "bbox_vlm"
            )
        )
    )
    recovery_status: dict[str, Any] = {
        "enabled": recovery_enabled,
        "ready": not recovery_enabled,
        "capability_trial_required": recovery_enabled,
        "share_recognizer_page_cache": bool(
            recovery_config.get("share_recognizer_page_cache", True)
        ),
        "table_orphan_recovery_enabled": bool(
            recovery_config.get("table_orphan_recovery_enabled", True)
        ),
        "checkbox_recovery_enabled": bool(
            recovery_config.get("checkbox_recovery_enabled", True)
        ),
    }
    if recovery_enabled:
        recovery_url = str(
            recovery_config.get("base_url")
            or recognizer_config.get("base_url")
            or upstream_url
        ).rstrip("/")
        configured_recovery_model = (
            recovery_config.get("model") or recognizer_config.get("model")
        )
        recovery_status.update(
            {
                "url": recovery_url,
                "reachable": False,
                "status_code": None,
                "configured_model": configured_recovery_model,
                "structured_output_mode": "json_schema",
                "structured_output_supported": False,
            }
        )
        try:
            import httpx

            recovery_headers = _optional_bearer_headers(
                recovery_config,
                recognizer_config,
                config["vllm"],
            )
            models_response = httpx.get(
                recovery_url + "/v1/models",
                headers=recovery_headers,
                timeout=3.0,
            )
            recovery_status["status_code"] = models_response.status_code
            recovery_status["reachable"] = models_response.is_success
            model_ids = []
            context_lengths = []
            if models_response.is_success:
                payload = models_response.json()
                models = payload.get("data", []) if isinstance(payload, dict) else []
                model_ids = [
                    item["id"]
                    for item in models
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                context_lengths = [
                    item["max_model_len"]
                    for item in models
                    if isinstance(item, dict)
                    and isinstance(item.get("max_model_len"), int)
                    and item["max_model_len"] > 1
                ]
            recovery_status["available_models"] = model_ids
            recovery_status["model_available"] = (
                bool(model_ids)
                if not isinstance(configured_recovery_model, str)
                or not configured_recovery_model
                else configured_recovery_model in model_ids
            )
            recovery_status["max_model_len"] = (
                min(context_lengths) if context_lengths else None
            )
            effective_recovery_model = (
                configured_recovery_model
                if isinstance(configured_recovery_model, str)
                and configured_recovery_model
                else model_ids[0] if model_ids else ""
            )
            local_mineru_recovery = bool(
                "mineru" in effective_recovery_model.casefold()
                and recovery_config.get("local_missing_enabled", True)
                and recovery_config.get("skip_vlm_for_mineru_models", True)
            )
            if local_mineru_recovery:
                recovery_status["structured_output_mode"] = "local_pixel"
                recovery_status["protocol"] = "local_pixel"
                recovery_status["structured_output_supported"] = True
            else:
                openapi_response = httpx.get(
                    recovery_url + "/openapi.json",
                    headers=recovery_headers,
                    timeout=3.0,
                )
                openapi = (
                    openapi_response.json() if openapi_response.is_success else {}
                )
                schemas = (
                    openapi.get("components", {}).get("schemas", {})
                    if isinstance(openapi, dict)
                    else {}
                )
                response_format_enum = (
                    schemas.get("ResponseFormat", {})
                    .get("properties", {})
                    .get("type", {})
                    .get("enum", [])
                    if isinstance(schemas, dict)
                    else []
                )
                recovery_status["protocol"] = "json_schema"
                recovery_status["structured_output_supported"] = (
                    "json_schema" in response_format_enum
                )
            recovery_status["ready"] = bool(
                recovery_status["reachable"]
                and recovery_status["model_available"]
                and recovery_status["structured_output_supported"]
            )
        except Exception as exc:
            recovery_status["error"] = type(exc).__name__

    missing_dependencies = [name for name, available in dependencies.items() if not available]
    ready = (
        local_mineru_source
        and not missing_dependencies
        and upstream_status["reachable"]
        and recognizer_status["ready"]
        and recovery_status["ready"]
    )
    return {
        "ready": ready,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repository_root": str(REPOSITORY_ROOT),
        "local_mineru_source": local_mineru_source,
        "dependencies": dependencies,
        "missing_dependencies": missing_dependencies,
        "upstream": upstream_status,
        "recognizer": recognizer_status,
        "recovery": recovery_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to workflow JSON config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    proxy_parser = subparsers.add_parser("proxy", help="Run the vLLM parameter proxy")
    proxy_parser.add_argument("--host")
    proxy_parser.add_argument("--port", type=int)

    subparsers.add_parser("serve-vllm", help="Start vLLM with configured engine arguments")

    subparsers.add_parser("doctor", help="Check local dependencies and the configured vLLM upstream")

    extract_parser = subparsers.add_parser("extract", help="Run MinerU through the parameter proxy")
    extract_parser.add_argument("--input", required=True)
    extract_parser.add_argument("--output", required=True)

    fuse_parser = subparsers.add_parser(
        "fuse", help="Re-run OCR/VLM fusion from existing output trees"
    )
    fuse_parser.add_argument("--input", required=True)
    fuse_parser.add_argument("--hybrid-root", required=True)
    fuse_parser.add_argument("--ocr-root", required=True)
    fuse_parser.add_argument("--output", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare Markdown against references")
    evaluate_parser.add_argument("--reference", required=True)
    evaluate_parser.add_argument("--candidate", required=True)
    evaluate_parser.add_argument("--report")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Score and rank one or more extraction output trees"
    )
    benchmark_parser.add_argument("--manifest", required=True)
    benchmark_parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate output path, optionally labeled as name=path; repeat for multiple runs",
    )
    benchmark_parser.add_argument("--report")

    sweep_parser = subparsers.add_parser(
        "sweep", help="Run a generation-parameter grid while reusing one OCR parse"
    )
    sweep_parser.add_argument("--input", required=True)
    sweep_parser.add_argument("--output", required=True)
    sweep_parser.add_argument("--manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "proxy":
            run_proxy(config, args.host, args.port)
            return 0
        if args.command == "serve-vllm":
            return subprocess.run(
                build_vllm_server_command(config), check=False, cwd=REPOSITORY_ROOT
            ).returncode
        if args.command == "doctor":
            report = run_doctor(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready"] else 1
        if args.command == "extract":
            return run_extract(config, args.input, args.output)
        if args.command == "fuse":
            server, thread, proxy_url = _start_parameter_proxy(config)
            try:
                summary = fuse_output_trees(
                    config,
                    args.input,
                    Path(args.hybrid_root).expanduser().resolve(),
                    Path(args.ocr_root).expanduser().resolve(),
                    Path(args.output).expanduser().resolve(),
                    proxy_url,
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 0 if not summary["failed"] else 2
            finally:
                _stop_parameter_proxy(server, thread)
        if args.command == "evaluate":
            report = evaluate_paths(
                args.reference,
                args.candidate,
                config["evaluation"].get("normalization", {}),
            )
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            if args.report:
                report_path = Path(args.report).expanduser().resolve()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            return 0
        if args.command == "benchmark":
            report = evaluate_benchmark_runs(
                args.manifest,
                args.candidate,
                config["evaluation"].get("normalization", {}),
                float(config["evaluation"].get("regression_tolerance", 0.0)),
            )
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            if args.report:
                report_path = Path(args.report).expanduser().resolve()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            return 0
        if args.command == "sweep":
            report = run_sweep(config, args.input, args.output, args.manifest)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if not report["failed"] else 2
    except (WorkflowConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

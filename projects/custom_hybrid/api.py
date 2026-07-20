"""HTTP task service for the custom OCR + Hybrid + Cell Fusion workflow."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hmac
import json
import os
import shutil
import sys
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.workflow import (
    load_config,
    regenerate_fused_visualizations,
    run_extract,
)


SUPPORTED_INPUT_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
MARKDOWN_ASSET_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
}
TERMINAL_STATUSES = {"completed", "failed"}
Runner = Callable[[Mapping[str, Any], str | Path, str | Path], int]
GENERATION_PARAMETER_LIMITS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "repetition_penalty": (0.01, 2.0),
}
COST_PROFILES = {"balanced", "quality"}
EXTRACTION_MODES = {"hybrid_fusion", "bbox_vlm"}
LEGACY_EXTRACTION_MODE_ALIASES = {"bbox_vlm_recovery": "bbox_vlm"}
DEFAULT_COST_PROFILE = "balanced"
BALANCED_FUSION_OVERRIDES = {
    "max_verifications_per_document": 0,
    "formula_fallback_enabled": False,
    "table_visual_verification_enabled": False,
    "formula_visual_verification_enabled": False,
    "max_structured_verifications_per_document": 0,
    "max_table_cell_verifications_per_document": 0,
    "verifier": {"enabled": False},
    "recognizer": {"enabled": False},
    "recovery": {"enabled": False},
    "reconciliation": {"enabled": False},
}
BBOX_VLM_FUSION_OVERRIDES = {
    "enabled": True,
    "mode": "bbox_vlm",
    "max_verifications_per_document": 0,
    "table_fallback_enabled": False,
    "formula_fallback_enabled": False,
    "table_visual_verification_enabled": False,
    "formula_visual_verification_enabled": False,
    "max_structured_verifications_per_document": 0,
    "table_cell_fusion_enabled": False,
    "max_table_cell_verifications_per_document": 0,
    "recover_missing_ocr_blocks": False,
    "unreliable_table_recovery_enabled": False,
    "recognizer": {
        "enabled": True,
        "normal_ocr_enabled": False,
        "table_ocr_enabled": True,
        "selection_policy": "vlm_primary",
        "include_row_image": True,
        "include_table_image": False,
        "max_images_per_request": 8,
        "max_image_limit_retries": 2,
    },
    "verifier": {"enabled": False},
    # BBox repair is an OCR post-processing stage, not a separate extraction
    # mode. Keep it on whenever the bbox-conditioned pipeline is selected.
    "recovery": {
        "enabled": True,
        "max_tables_per_document": 3,
        "max_proposals_per_document": 100,
        "max_proposals_per_table": 30,
        "max_requests_per_document": 3,
    },
    "reconciliation": {"enabled": False},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_upload_name(raw_name: str | None, used: set[str]) -> str:
    name = Path((raw_name or "upload").replace("\\", "/")).name
    if not name or name in {".", ".."}:
        name = "upload"
    stem = Path(name).stem or "upload"
    suffix = Path(name).suffix.lower()
    candidate = stem + suffix
    counter = 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used.add(candidate.casefold())
    return candidate


@dataclass
class TaskRecord:
    task_id: str
    input_names: list[str]
    task_root: Path
    input_root: Path
    output_root: Path
    archive_path: Path
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class CustomHybridTaskManager:
    """Single-worker task manager because each extraction owns a local proxy port."""

    def __init__(
        self,
        config_path: Path,
        output_root: Path,
        runner: Runner = run_extract,
    ) -> None:
        self.config_path = config_path
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self._records: dict[str, TaskRecord] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._queue: list[str] = []
        self._queue_event = threading.Event()
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="custom-hybrid-api-worker",
            daemon=True,
        )
        self._worker.start()

    def create(
        self,
        task_id: str,
        input_names: list[str],
        parameters: Mapping[str, Any] | None = None,
    ) -> TaskRecord:
        task_root = self.output_root / task_id
        record = TaskRecord(
            task_id=task_id,
            input_names=input_names,
            task_root=task_root,
            input_root=task_root / "input",
            output_root=task_root / "output",
            archive_path=task_root / "fused-result.zip",
            parameters=copy.deepcopy(dict(parameters or {})),
        )
        with self._lock:
            self._records[task_id] = record
            self._events[task_id] = threading.Event()
        return record

    def enqueue(self, task_id: str) -> None:
        with self._lock:
            if task_id not in self._records:
                raise KeyError(task_id)
            self._queue.append(task_id)
            self._queue_event.set()

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            record = self._records.get(task_id)
            return copy.copy(record) if record is not None else None

    def wait(self, task_id: str) -> TaskRecord:
        with self._lock:
            event = self._events.get(task_id)
        if event is None:
            raise KeyError(task_id)
        event.wait()
        record = self.get(task_id)
        if record is None:
            raise KeyError(task_id)
        return record

    def delete(self, task_id: str) -> bool:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return False
            if record.status not in TERMINAL_STATUSES:
                raise RuntimeError("Only completed or failed tasks can be deleted")
            self._records.pop(task_id, None)
            self._events.pop(task_id, None)
        shutil.rmtree(record.task_root, ignore_errors=True)
        return True

    def close(self) -> None:
        self._closed = True
        self._queue_event.set()
        self._worker.join(timeout=10)

    def _next_task_id(self) -> str | None:
        with self._lock:
            if not self._queue:
                self._queue_event.clear()
                return None
            return self._queue.pop(0)

    def _worker_loop(self) -> None:
        while not self._closed:
            self._queue_event.wait(timeout=1)
            if self._closed:
                return
            task_id = self._next_task_id()
            if task_id is None:
                continue
            self._run_task(task_id)

    def _run_task(self, task_id: str) -> None:
        with self._lock:
            record = self._records[task_id]
            record.status = "processing"
            record.started_at = _now()
        try:
            config = load_config(self.config_path)
            _apply_task_parameters(config, record.parameters)
            config["vllm"]["audit_log"] = str(
                record.output_root / "vllm_requests.jsonl"
            )
            record.output_root.mkdir(parents=True, exist_ok=True)
            (record.output_root / "task_parameters.json").write_text(
                json.dumps(
                    _task_parameter_defaults(
                        config,
                        cost_profile=str(
                            record.parameters.get(
                                "cost_profile",
                                DEFAULT_COST_PROFILE,
                            )
                        ),
                        apply_profile=False,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            exit_code = self.runner(config, record.input_root, record.output_root)
            if exit_code:
                raise RuntimeError(f"Custom Hybrid workflow exited with code {exit_code}")
            _create_fused_archive(record.output_root, record.archive_path)
            with self._lock:
                record.status = "completed"
        except Exception as exc:
            with self._lock:
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                record.completed_at = _now()
                event = self._events[task_id]
            event.set()


def _normalize_task_parameters(
    *,
    cost_profile: str | None = None,
    extraction_mode: str | None = None,
    effort: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
    repetition_penalty: float | None = None,
    recovery_max_tables: int | None = None,
    recovery_max_proposals: int | None = None,
    recovery_min_confidence: float | None = None,
) -> dict[str, Any]:
    mineru: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    recognizer_generation: dict[str, Any] = {}
    fusion: dict[str, Any] = {}
    if cost_profile is not None:
        if cost_profile not in COST_PROFILES:
            raise HTTPException(
                status_code=400,
                detail="cost_profile must be balanced or quality",
            )
        if cost_profile == "balanced":
            mineru.update(
                {
                    "effort": "medium",
                    "formula": False,
                    "image_analysis": False,
                }
            )
            generation["max_tokens"] = 2048
            fusion.update(copy.deepcopy(BALANCED_FUSION_OVERRIDES))
    if extraction_mode is not None:
        extraction_mode = LEGACY_EXTRACTION_MODE_ALIASES.get(
            extraction_mode,
            extraction_mode,
        )
        if extraction_mode not in EXTRACTION_MODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "extraction_mode must be hybrid_fusion or bbox_vlm"
                ),
            )
        if extraction_mode == "bbox_vlm":
            bbox_vlm_overrides = copy.deepcopy(BBOX_VLM_FUSION_OVERRIDES)
            bbox_vlm_overrides["recognizer"]["include_table_image"] = (
                cost_profile == "quality"
            )
            bbox_vlm_overrides["recognizer"].update(
                {
                    "native_min_bbox_height": 16.0
                    if cost_profile == "quality"
                    else 20.0,
                    "native_max_tokens": 512
                    if cost_profile == "quality"
                    else 256,
                    "native_max_requests_per_page": 24
                    if cost_profile == "quality"
                    else 12,
                    "native_max_candidates_per_page": 24
                    if cost_profile == "quality"
                    else 12,
                    "max_requests_per_document": 80
                    if cost_profile == "quality"
                    else 50,
                    "target_render_scale": 4.0
                    if cost_profile == "quality"
                    else 3.0,
                    "jpeg_quality": 92 if cost_profile == "quality" else 85,
                    "native_max_concurrency": 2,
                    "native_cache_enabled": True,
                }
            )
            bbox_vlm_overrides["recovery"].update(
                {
                    "max_tables_per_document": 10
                    if cost_profile == "quality"
                    else 3,
                    "max_proposals_per_document": 100,
                    "max_proposals_per_table": 30,
                    "max_requests_per_document": 10
                    if cost_profile == "quality"
                    else 3,
                }
            )
            fusion.update(bbox_vlm_overrides)
        else:
            fusion["mode"] = "hybrid_fusion"
    if effort is not None:
        if effort not in {"medium", "high"}:
            raise HTTPException(status_code=400, detail="effort must be medium or high")
        mineru["effort"] = effort
    if method is not None:
        if method not in {"auto", "txt", "ocr"}:
            raise HTTPException(status_code=400, detail="method must be auto, txt, or ocr")
        mineru["method"] = method
    if lang is not None:
        normalized_lang = lang.strip()
        if not normalized_lang or len(normalized_lang) > 32:
            raise HTTPException(
                status_code=400,
                detail="lang must be a non-empty value up to 32 characters",
            )
        mineru["lang"] = normalized_lang
    for name, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("repetition_penalty", repetition_penalty),
    ):
        if value is None:
            continue
        minimum, maximum = GENERATION_PARAMETER_LIMITS[name]
        outside_range = not minimum <= float(value) <= maximum
        if name == "top_p" and value <= 0:
            outside_range = True
        if outside_range:
            qualifier = (
                "greater than 0 and at most 1"
                if name == "top_p"
                else f"between {minimum} and {maximum}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be {qualifier}",
            )
        generation[name] = float(value)
        if name in {"temperature", "top_p"}:
            recognizer_generation[name] = float(value)
    if seed is not None:
        if not -(2**63) <= seed < 2**63:
            raise HTTPException(status_code=400, detail="seed must be a signed 64-bit integer")
        generation["seed"] = seed
        recognizer_generation["seed"] = seed
    if max_tokens is not None:
        if not 1 <= max_tokens <= 131072:
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be between 1 and 131072",
            )
        generation["max_tokens"] = max_tokens
        recognizer_generation["max_tokens"] = max_tokens
    if extraction_mode == "bbox_vlm" and recognizer_generation:
        recognizer_overrides = fusion.setdefault("recognizer", {})
        if isinstance(recognizer_overrides, dict):
            recognizer_overrides.update(recognizer_generation)
        recovery_overrides = fusion.setdefault("recovery", {})
        if isinstance(recovery_overrides, dict):
            recovery_overrides.update(recognizer_generation)
    recovery_overrides = fusion.setdefault("recovery", {}) if any(
        value is not None
        for value in (
            recovery_max_tables,
            recovery_max_proposals,
            recovery_min_confidence,
        )
    ) else None
    if isinstance(recovery_overrides, dict):
        for name, value, maximum in (
            ("max_tables_per_document", recovery_max_tables, 1000),
            ("max_proposals_per_document", recovery_max_proposals, 10000),
        ):
            if value is None:
                continue
            if not 0 <= value <= maximum:
                raise HTTPException(
                    status_code=400,
                    detail=f"recovery_{name} must be between 0 and {maximum}",
                )
            recovery_overrides[name] = value
        if recovery_min_confidence is not None:
            if not 0 <= recovery_min_confidence <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="recovery_min_confidence must be between 0 and 1",
                )
            recovery_overrides["min_confidence"] = float(
                recovery_min_confidence
            )
    parameters: dict[str, Any] = {}
    if cost_profile is not None:
        parameters["cost_profile"] = cost_profile
    if mineru:
        parameters["mineru"] = mineru
    if generation:
        parameters["generation"] = generation
    if fusion:
        parameters["fusion"] = fusion
    return parameters


def _apply_task_parameters(config: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    mineru = parameters.get("mineru", {})
    if isinstance(mineru, Mapping):
        config["mineru"].update(mineru)
    generation = parameters.get("generation", {})
    if isinstance(generation, Mapping):
        config["vllm"]["generation"]["task_overrides"] = dict(generation)
    fusion = parameters.get("fusion", {})
    if isinstance(fusion, Mapping):
        for key, value in fusion.items():
            if isinstance(value, Mapping):
                nested = config["fusion"].setdefault(key, {})
                if isinstance(nested, dict):
                    nested.update(value)
                else:
                    config["fusion"][key] = dict(value)
            else:
                config["fusion"][key] = value


def _task_parameter_defaults(
    config: Mapping[str, Any],
    *,
    cost_profile: str = DEFAULT_COST_PROFILE,
    apply_profile: bool = True,
) -> dict[str, Any]:
    effective_config = copy.deepcopy(dict(config))
    if apply_profile:
        _apply_task_parameters(
            effective_config,
            _normalize_task_parameters(cost_profile=cost_profile),
        )
    mineru_config = effective_config["mineru"]
    generation_config = effective_config["vllm"]["generation"]
    defaults = generation_config.get("defaults", {})
    overrides = generation_config.get("overrides", {})
    task_overrides = generation_config.get("task_overrides", {})

    def generation_value(name: str) -> Any:
        if name in task_overrides:
            return task_overrides[name]
        if name in overrides:
            return overrides[name]
        return defaults.get(name)

    configured_mode = effective_config.get("fusion", {}).get(
        "mode",
        "hybrid_fusion",
    )
    extraction_mode = LEGACY_EXTRACTION_MODE_ALIASES.get(
        configured_mode,
        configured_mode,
    )
    return {
        "cost_profile": cost_profile,
        "extraction_mode": extraction_mode,
        "mineru": {
            "effort": mineru_config.get("effort", "medium"),
            "method": mineru_config.get("method", "auto"),
            "lang": mineru_config.get("lang", "ch"),
        },
        "generation": {
            name: generation_value(name)
            for name in (
                "temperature",
                "top_p",
                "seed",
                "max_tokens",
                "repetition_penalty",
            )
        },
        "recovery": {
            "max_tables_per_document": effective_config.get("fusion", {})
            .get("recovery", {})
            .get("max_tables_per_document", 10),
            "max_proposals_per_document": effective_config.get("fusion", {})
            .get("recovery", {})
            .get("max_proposals_per_document", 100),
            "min_confidence": effective_config.get("fusion", {})
            .get("recovery", {})
            .get("min_confidence", 0.85),
        },
    }


def _create_fused_archive(output_root: Path, archive_path: Path) -> None:
    fused_root = output_root / "fused"
    if not fused_root.is_dir():
        raise RuntimeError(f"Fused output was not created: {fused_root}")
    selected = [fused_root]
    for optional_name in (
        "fusion_summary.json",
        "task_parameters.json",
        "vllm_requests.jsonl",
    ):
        optional_path = output_root / optional_name
        if optional_path.is_file():
            selected.append(optional_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            if path.is_file():
                archive.write(path, path.relative_to(output_root).as_posix())
                continue
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    archive.write(child, child.relative_to(output_root).as_posix())


def _task_markdown_documents(record: TaskRecord) -> dict[str, Path]:
    fused_root = record.output_root / "fused"
    if not fused_root.is_dir():
        return {}
    return {
        path.relative_to(fused_root).as_posix(): path
        for path in sorted(fused_root.rglob("*.md"))
        if path.is_file()
    }


def _select_task_markdown(record: TaskRecord, document: str | None) -> tuple[str, Path]:
    documents = _task_markdown_documents(record)
    if not documents:
        raise HTTPException(status_code=404, detail="No Markdown output is available")
    selected = document or next(iter(documents))
    path = documents.get(selected)
    if path is None:
        raise HTTPException(status_code=404, detail="Markdown document not found")
    return selected, path


def _content_list_text(markdown_path: Path) -> str | None:
    candidates = (
        markdown_path.with_name(f"{markdown_path.stem}_content_list.json"),
        markdown_path.with_name("content_list.json"),
    )
    content_path = next((path for path in candidates if path.is_file()), None)
    return content_path.read_text(encoding="utf-8") if content_path else None


async def _save_uploads(
    files: Sequence[UploadFile],
    input_root: Path,
    max_upload_bytes: int,
) -> list[str]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one input file is required")
    input_root.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    saved_names = []
    total_bytes = 0
    try:
        for upload in files:
            name = _safe_upload_name(upload.filename, used_names)
            if Path(name).suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported input type: {name}",
                )
            destination = input_root / name
            with destination.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Uploaded files exceed the configured size limit",
                        )
                    output.write(chunk)
            saved_names.append(name)
    except Exception:
        shutil.rmtree(input_root, ignore_errors=True)
        raise
    finally:
        for upload in files:
            await upload.close()
    return saved_names


def _task_payload(record: TaskRecord, request: Request) -> dict[str, Any]:
    payload = {
        "task_id": record.task_id,
        "status": record.status,
        "input_names": record.input_names,
        "parameters": record.parameters,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "error": record.error,
        "status_url": str(request.url_for("get_task", task_id=record.task_id)),
        "result_url": str(request.url_for("get_task_result", task_id=record.task_id)),
        "report_url": str(request.url_for("get_task_report", task_id=record.task_id)),
        "preview_url": str(
            request.url_for("get_task_preview", task_id=record.task_id)
        ),
    }
    return payload


def create_app(
    config_path: str | Path,
    output_root: str | Path,
    *,
    api_key: str | None = None,
    max_upload_mb: int = 200,
    runner: Runner = run_extract,
) -> FastAPI:
    resolved_config = Path(config_path).expanduser().resolve()
    load_config(resolved_config)
    manager = CustomHybridTaskManager(
        resolved_config,
        Path(output_root).expanduser().resolve(),
        runner=runner,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.close()

    app = FastAPI(
        title="Custom Hybrid MinerU API",
        version="1",
        lifespan=lifespan,
    )
    app.state.task_manager = manager
    max_upload_bytes = max_upload_mb * 1024 * 1024

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        prefix = "Bearer "
        supplied = (
            authorization[len(prefix) :]
            if authorization and authorization.startswith(prefix)
            else ""
        )
        if not hmac.compare_digest(supplied, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    def require_task(task_id: str) -> TaskRecord:
        record = manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return record

    def require_completed_task(task_id: str) -> TaskRecord:
        record = require_task(task_id)
        if record.status == "failed":
            raise HTTPException(status_code=409, detail=record.error or "Task failed")
        if record.status != "completed":
            raise HTTPException(status_code=409, detail=f"Task is {record.status}")
        return record

    async def create_uploaded_task(
        files: Sequence[UploadFile],
        parameters: Mapping[str, Any] | None = None,
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        task_root = manager.output_root / task_id
        input_root = task_root / "input"
        try:
            names = await _save_uploads(files, input_root, max_upload_bytes)
            record = manager.create(task_id, names, parameters)
            manager.enqueue(task_id)
            return record
        except Exception:
            shutil.rmtree(task_root, ignore_errors=True)
            raise

    @app.get("/health", name="health")
    async def health() -> dict[str, Any]:
        config = load_config(resolved_config)
        return {
            "status": "ok",
            "service": "custom-hybrid-mineru",
            "upstream_url": config["vllm"]["upstream_url"],
            "authentication_required": api_key is not None,
            "max_upload_mb": max_upload_mb,
            "task_parameter_defaults": _task_parameter_defaults(config),
        }

    @app.post("/tasks", status_code=202, dependencies=[Depends(authorize)])
    async def submit_task(
        request: Request,
        files: list[UploadFile] = File(...),
        cost_profile: str = Form(default=DEFAULT_COST_PROFILE),
        extraction_mode: str | None = Form(default=None),
        effort: str | None = Form(default=None),
        method: str | None = Form(default=None),
        lang: str | None = Form(default=None),
        temperature: float | None = Form(default=None),
        top_p: float | None = Form(default=None),
        seed: int | None = Form(default=None),
        max_tokens: int | None = Form(default=None),
        repetition_penalty: float | None = Form(default=None),
        recovery_max_tables: int | None = Form(default=None),
        recovery_max_proposals: int | None = Form(default=None),
        recovery_min_confidence: float | None = Form(default=None),
    ) -> dict[str, Any]:
        parameters = _normalize_task_parameters(
            cost_profile=cost_profile,
            extraction_mode=extraction_mode,
            effort=effort,
            method=method,
            lang=lang,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            recovery_max_tables=recovery_max_tables,
            recovery_max_proposals=recovery_max_proposals,
            recovery_min_confidence=recovery_min_confidence,
        )
        record = await create_uploaded_task(files, parameters)
        return _task_payload(record, request)

    @app.get("/tasks/{task_id}", name="get_task", dependencies=[Depends(authorize)])
    async def get_task(task_id: str, request: Request) -> dict[str, Any]:
        return _task_payload(require_task(task_id), request)

    @app.get(
        "/tasks/{task_id}/result",
        name="get_task_result",
        dependencies=[Depends(authorize)],
    )
    async def get_task_result(task_id: str):
        record = require_completed_task(task_id)
        return FileResponse(
            record.archive_path,
            media_type="application/zip",
            filename=f"{task_id}-fused.zip",
        )

    @app.get(
        "/tasks/{task_id}/preview",
        name="get_task_preview",
        dependencies=[Depends(authorize)],
    )
    async def get_task_preview(task_id: str, kind: str = "span"):
        record = require_completed_task(task_id)
        suffixes = {
            "span": "*_span.pdf",
            "form_cells": "*_form_cells.pdf",
        }
        suffix = suffixes.get(kind)
        if suffix is None:
            raise HTTPException(
                status_code=400,
                detail="Preview kind must be span or form_cells",
            )
        try:
            await asyncio.to_thread(
                regenerate_fused_visualizations,
                record.output_root / "fused",
                record.input_root,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Bounding Box PDF generation failed: {exc}",
            ) from exc
        previews = sorted((record.output_root / "fused").rglob(suffix))
        if not previews:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No {kind} preview PDF is available; the task has no "
                    "matching PDF source or generated artifact"
                ),
            )
        return FileResponse(
            previews[0],
            media_type="application/pdf",
            filename=previews[0].name,
            content_disposition_type="inline",
        )

    @app.get(
        "/tasks/{task_id}/markdown",
        dependencies=[Depends(authorize)],
    )
    async def get_task_markdown(task_id: str, document: str | None = None):
        record = require_completed_task(task_id)
        selected, markdown_path = _select_task_markdown(record, document)
        documents = _task_markdown_documents(record)
        return JSONResponse(
            content={
                "documents": [
                    {"id": item_id, "name": path.stem}
                    for item_id, path in documents.items()
                ],
                "selected": selected,
                "name": markdown_path.stem,
                "markdown": markdown_path.read_text(encoding="utf-8"),
                "content_list": _content_list_text(markdown_path),
            }
        )

    @app.get(
        "/tasks/{task_id}/asset",
        dependencies=[Depends(authorize)],
    )
    async def get_task_markdown_asset(
        task_id: str,
        document: str,
        path: str,
    ):
        record = require_completed_task(task_id)
        _, markdown_path = _select_task_markdown(record, document)
        fused_root = (record.output_root / "fused").resolve()
        asset_path = (markdown_path.parent / path).resolve()
        try:
            asset_path.relative_to(fused_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Asset not found") from exc
        if (
            not asset_path.is_file()
            or asset_path.suffix.lower() not in MARKDOWN_ASSET_SUFFIXES
        ):
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(
            asset_path,
            filename=asset_path.name,
            content_disposition_type="inline",
        )

    @app.get(
        "/tasks/{task_id}/report",
        name="get_task_report",
        dependencies=[Depends(authorize)],
    )
    async def get_task_report(task_id: str):
        record = require_task(task_id)
        report_path = record.output_root / "fusion_summary.json"
        if record.status == "failed":
            raise HTTPException(status_code=409, detail=record.error or "Task failed")
        if record.status != "completed" or not report_path.is_file():
            raise HTTPException(status_code=409, detail=f"Task is {record.status}")
        return JSONResponse(
            content=json.loads(report_path.read_text(encoding="utf-8"))
        )

    @app.post("/file_parse", dependencies=[Depends(authorize)])
    async def file_parse(
        files: list[UploadFile] = File(...),
        cost_profile: str = Form(default=DEFAULT_COST_PROFILE),
        extraction_mode: str | None = Form(default=None),
        effort: str | None = Form(default=None),
        method: str | None = Form(default=None),
        lang: str | None = Form(default=None),
        temperature: float | None = Form(default=None),
        top_p: float | None = Form(default=None),
        seed: int | None = Form(default=None),
        max_tokens: int | None = Form(default=None),
        repetition_penalty: float | None = Form(default=None),
        recovery_max_tables: int | None = Form(default=None),
        recovery_max_proposals: int | None = Form(default=None),
        recovery_min_confidence: float | None = Form(default=None),
    ):
        parameters = _normalize_task_parameters(
            cost_profile=cost_profile,
            extraction_mode=extraction_mode,
            effort=effort,
            method=method,
            lang=lang,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            recovery_max_tables=recovery_max_tables,
            recovery_max_proposals=recovery_max_proposals,
            recovery_min_confidence=recovery_min_confidence,
        )
        record = await create_uploaded_task(files, parameters)
        completed = await asyncio.to_thread(manager.wait, record.task_id)
        if completed.status == "failed":
            raise HTTPException(status_code=409, detail=completed.error or "Task failed")
        return FileResponse(
            completed.archive_path,
            media_type="application/zip",
            filename=f"{completed.task_id}-fused.zip",
        )

    @app.delete("/tasks/{task_id}", dependencies=[Depends(authorize)])
    async def delete_task(task_id: str):
        require_task(task_id)
        try:
            manager.delete(task_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"task_id": task_id, "deleted": True}

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Workflow JSON configuration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--output-root", default="./output/custom_hybrid_api")
    parser.add_argument("--api-key-env", default="CUSTOM_HYBRID_API_KEY")
    parser.add_argument("--max-upload-mb", type=int, default=200)
    parser.add_argument("--allow-unauthenticated-public", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    public_host = args.host in {"0.0.0.0", "::"}
    if public_host and not api_key and not args.allow_unauthenticated_public:
        raise SystemExit(
            f"Set {args.api_key_env} or pass --allow-unauthenticated-public explicitly"
        )
    app = create_app(
        args.config,
        args.output_root,
        api_key=api_key,
        max_upload_mb=args.max_upload_mb,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

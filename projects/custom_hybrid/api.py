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
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.custom_hybrid.workflow import load_config, run_extract


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
TERMINAL_STATUSES = {"completed", "failed"}
Runner = Callable[[Mapping[str, Any], str | Path, str | Path], int]


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

    def create(self, task_id: str, input_names: list[str]) -> TaskRecord:
        task_root = self.output_root / task_id
        record = TaskRecord(
            task_id=task_id,
            input_names=input_names,
            task_root=task_root,
            input_root=task_root / "input",
            output_root=task_root / "output",
            archive_path=task_root / "fused-result.zip",
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
            config["vllm"]["audit_log"] = str(
                record.output_root / "vllm_requests.jsonl"
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


def _create_fused_archive(output_root: Path, archive_path: Path) -> None:
    fused_root = output_root / "fused"
    if not fused_root.is_dir():
        raise RuntimeError(f"Fused output was not created: {fused_root}")
    selected = [fused_root]
    for optional_name in ("fusion_summary.json", "vllm_requests.jsonl"):
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
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "error": record.error,
        "status_url": str(request.url_for("get_task", task_id=record.task_id)),
        "result_url": str(request.url_for("get_task_result", task_id=record.task_id)),
        "report_url": str(request.url_for("get_task_report", task_id=record.task_id)),
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
        supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
        if not hmac.compare_digest(supplied, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    def require_task(task_id: str) -> TaskRecord:
        record = manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return record

    async def create_uploaded_task(files: Sequence[UploadFile]) -> TaskRecord:
        task_id = uuid.uuid4().hex
        task_root = manager.output_root / task_id
        input_root = task_root / "input"
        try:
            names = await _save_uploads(files, input_root, max_upload_bytes)
            record = manager.create(task_id, names)
            manager.enqueue(task_id)
            return record
        except Exception:
            shutil.rmtree(task_root, ignore_errors=True)
            raise

    @app.get("/health", name="health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "custom-hybrid-mineru",
            "upstream_url": load_config(resolved_config)["vllm"]["upstream_url"],
            "authentication_required": api_key is not None,
            "max_upload_mb": max_upload_mb,
        }

    @app.post("/tasks", status_code=202, dependencies=[Depends(authorize)])
    async def submit_task(
        request: Request,
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        record = await create_uploaded_task(files)
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
        record = require_task(task_id)
        if record.status == "failed":
            raise HTTPException(status_code=409, detail=record.error or "Task failed")
        if record.status != "completed":
            raise HTTPException(status_code=409, detail=f"Task is {record.status}")
        return FileResponse(
            record.archive_path,
            media_type="application/zip",
            filename=f"{task_id}-fused.zip",
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
    async def file_parse(files: list[UploadFile] = File(...)):
        record = await create_uploaded_task(files)
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

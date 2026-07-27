"""Local browser UI that proxies requests to the Custom Hybrid MinerU API."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


UI_HTML_PATH = Path(__file__).with_name("ui.html")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def create_ui_app(
    remote_url: str,
    *,
    remote_api_key: str | None = None,
    jupyter_token: str | None = None,
    timeout_seconds: float = 3600,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    # Retained for compatibility with existing launch scripts. The trusted-LAN
    # Custom Hybrid API no longer accepts or requires a bearer token.
    _ = remote_api_key
    remote_base = remote_url.rstrip("/")
    headers: dict[str, str] = {}
    params = {"token": jupyter_token} if jupyter_token else None
    app = FastAPI(title="Custom Hybrid MinerU Local UI", docs_url=None, redoc_url=None)

    def make_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=True,
        )

    def task_url(task_id: str, suffix: str = "") -> str:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(status_code=400, detail="Invalid task id")
        return f"{remote_base}/tasks/{task_id}{suffix}"

    def request_params(extra: Mapping[str, str] | None = None) -> dict[str, str] | None:
        merged = dict(params or {})
        if extra:
            merged.update(extra)
        return merged or None

    async def buffered_request(
        method: str,
        url: str,
        *,
        extra_params: Mapping[str, str] | None = None,
    ) -> Response:
        try:
            async with make_client() as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=request_params(extra_params),
                )
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={"detail": f"Cannot reach Custom Hybrid API: {exc}"},
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json").split(
                ";", 1
            )[0],
        )

    async def streamed_task_request(
        task_id: str,
        suffix: str,
        *,
        default_media_type: str,
        default_disposition: str,
        extra_params: Mapping[str, str] | None = None,
    ):
        client = make_client()
        try:
            request = client.build_request(
                "GET",
                task_url(task_id, suffix),
                headers=headers,
                params=request_params(extra_params),
            )
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            return JSONResponse(
                status_code=502,
                content={"detail": f"Remote artifact request failed: {exc}"},
            )
        if response.is_error:
            content = await response.aread()
            media_type = response.headers.get("content-type", "application/json")
            await response.aclose()
            await client.aclose()
            return Response(
                content=content,
                status_code=response.status_code,
                media_type=media_type,
            )

        async def close_remote() -> None:
            await response.aclose()
            await client.aclose()

        return StreamingResponse(
            response.aiter_bytes(),
            media_type=response.headers.get("content-type", default_media_type),
            headers={
                "Content-Disposition": response.headers.get(
                    "content-disposition",
                    default_disposition,
                )
            },
            background=BackgroundTask(close_remote),
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(UI_HTML_PATH.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> Response:
        return await buffered_request("GET", f"{remote_base}/health")

    @app.post("/api/tasks")
    async def submit_task(
        files: list[UploadFile] = File(...),
        cost_profile: str | None = Form(default=None),
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
        page_sorting_llm_enabled: bool | None = Form(default=None),
    ) -> Response:
        if not files:
            raise HTTPException(status_code=400, detail="Select at least one file")
        payload = [
            (
                "files",
                (
                    upload.filename or "upload",
                    upload.file,
                    upload.content_type or "application/octet-stream",
                ),
            )
            for upload in files
        ]
        task_parameters = {
            key: value
            for key, value in {
                "cost_profile": cost_profile,
                "extraction_mode": extraction_mode,
                "effort": effort,
                "method": method,
                "lang": lang,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "recovery_max_tables": recovery_max_tables,
                "recovery_max_proposals": recovery_max_proposals,
                "recovery_min_confidence": recovery_min_confidence,
                "page_sorting_llm_enabled": page_sorting_llm_enabled,
            }.items()
            if value is not None
        }
        try:
            async with make_client() as client:
                response = await client.post(
                    f"{remote_base}/tasks",
                    files=payload,
                    data=task_parameters,
                    headers=headers,
                    params=params,
                )
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={"detail": f"Upload failed: {exc}"},
            )
        finally:
            for upload in files:
                await upload.close()
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json").split(
                ";", 1
            )[0],
        )

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str) -> Response:
        return await buffered_request("GET", task_url(task_id))

    @app.get("/api/tasks/{task_id}/report")
    async def get_report(task_id: str) -> Response:
        return await buffered_request("GET", task_url(task_id, "/report"))

    @app.get("/api/tasks/{task_id}/sorting")
    async def get_sorting(task_id: str) -> Response:
        return await buffered_request("GET", task_url(task_id, "/sorting"))

    @app.get("/api/tasks/{task_id}/markdown")
    async def get_markdown(task_id: str, document: str | None = None) -> Response:
        extra = {"document": document} if document else None
        return await buffered_request(
            "GET",
            task_url(task_id, "/markdown"),
            extra_params=extra,
        )

    @app.get("/api/tasks/{task_id}/asset")
    async def get_markdown_asset(task_id: str, document: str, path: str):
        return await streamed_task_request(
            task_id,
            "/asset",
            default_media_type="application/octet-stream",
            default_disposition="inline",
            extra_params={"document": document, "path": path},
        )

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: str) -> Response:
        return await buffered_request("DELETE", task_url(task_id))

    @app.get("/api/tasks/{task_id}/result")
    async def get_result(task_id: str):
        return await streamed_task_request(
            task_id,
            "/result",
            default_media_type="application/zip",
            default_disposition=f'attachment; filename="{task_id}-fused.zip"',
        )

    @app.get("/api/tasks/{task_id}/preview")
    async def get_preview(task_id: str, kind: str = "span"):
        return await streamed_task_request(
            task_id,
            "/preview",
            default_media_type="application/pdf",
            default_disposition=f'inline; filename="{task_id}-{kind}.pdf"',
            extra_params={"kind": kind},
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote-url",
        default=os.getenv("CUSTOM_HYBRID_API_URL", "http://127.0.0.1:8010"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--remote-api-key-env",
        default="CUSTOM_HYBRID_API_KEY",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--jupyter-token-env", default="JUPYTER_TOKEN")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument(
        "--allow-public-bind",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_ui_app(
        args.remote_url,
        jupyter_token=os.getenv(args.jupyter_token_env),
        timeout_seconds=args.timeout_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

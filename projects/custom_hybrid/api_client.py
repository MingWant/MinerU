"""Upload documents to a Custom Hybrid MinerU API and save the fused ZIP."""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

import httpx


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


def collect_inputs(input_path: str | Path) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = [item for item in sorted(path.iterdir()) if item.is_file()]
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")
    supported = [
        item for item in candidates if item.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    ]
    if not supported:
        raise ValueError(f"No supported PDF/image inputs found under {path}")
    return supported


def parse_remote(
    base_url: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    jupyter_token: str | None = None,
    cost_profile: str | None = None,
    extraction_mode: str | None = None,
    timeout_seconds: float = 3600,
) -> Path:
    inputs = collect_inputs(input_path)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    task_parameters = {
        name: value
        for name, value in {
            "cost_profile": cost_profile,
            "extraction_mode": extraction_mode,
        }.items()
        if value is not None
    }
    with ExitStack() as stack:
        files = [
            (
                "files",
                (
                    item.name,
                    stack.enter_context(item.open("rb")),
                    "application/octet-stream",
                ),
            )
            for item in inputs
        ]
        with httpx.Client(timeout=timeout_seconds) as client:
            with client.stream(
                "POST",
                base_url.rstrip("/") + "/file_parse",
                files=files,
                data=task_parameters or None,
                headers=headers,
                params={"token": jupyter_token} if jupyter_token else None,
            ) as response:
                if response.is_error:
                    detail = response.read().decode("utf-8", errors="replace")[:2000]
                    raise RuntimeError(
                        f"Custom Hybrid API returned {response.status_code}: {detail}"
                    )
                with destination.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Custom Hybrid API base URL")
    parser.add_argument("--input", required=True, help="PDF/image file or directory")
    parser.add_argument("--output", required=True, help="Destination ZIP path")
    parser.add_argument("--api-key-env", default="CUSTOM_HYBRID_API_KEY")
    parser.add_argument("--jupyter-token-env", default="JUPYTER_TOKEN")
    parser.add_argument("--cost-profile", choices=("balanced", "quality"))
    parser.add_argument(
        "--extraction-mode",
        choices=("hybrid_fusion", "bbox_vlm"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    jupyter_token = (
        os.getenv(args.jupyter_token_env) if args.jupyter_token_env else None
    )
    destination = parse_remote(
        args.url,
        args.input,
        args.output,
        api_key=api_key,
        jupyter_token=jupyter_token,
        cost_profile=args.cost_profile,
        extraction_mode=args.extraction_mode,
        timeout_seconds=args.timeout_seconds,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

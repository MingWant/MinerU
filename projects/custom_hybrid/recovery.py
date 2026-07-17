"""Selective Table content-box recovery with VLM review and pixel refinement."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from projects.custom_hybrid.fusion import PageCropProvider


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _clip_bbox(
    bbox: Sequence[float],
    outer: Sequence[float],
) -> tuple[float, float, float, float] | None:
    return _valid_bbox(
        (
            max(float(bbox[0]), float(outer[0])),
            max(float(bbox[1]), float(outer[1])),
            min(float(bbox[2]), float(outer[2])),
            min(float(bbox[3]), float(outer[3])),
        )
    )


def _bbox_overlap_ratio(
    bbox: Sequence[float],
    existing: Sequence[float],
) -> float:
    width = max(0.0, min(bbox[2], existing[2]) - max(bbox[0], existing[0]))
    height = max(0.0, min(bbox[3], existing[3]) - max(bbox[1], existing[1]))
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return width * height / area if area > 0 else 0.0


def _parse_json_object(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, Mapping) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.casefold().startswith("json"):
            stripped = stripped[4:].lstrip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class OpenAIBBoxRecoveryReviewer:
    """Review suspicious Table Cells and return pixel-refined page bboxes."""

    def __init__(
        self,
        base_url: str,
        document_path: str | Path,
        config: Mapping[str, Any],
        *,
        page_provider: PageCropProvider | None = None,
    ):
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("BBox VLM recovery requires httpx") from exc
        self.httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.config = config
        self.timeout = float(config.get("timeout_seconds", 120))
        self.headers = {"Content-Type": "application/json"}
        api_key = os.getenv(str(config.get("api_key_env", "VLLM_API_KEY")))
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = config.get("model")
        self.requests_made = 0
        self.tables_reviewed = 0
        self.proposals_returned = 0
        self.protocol_disabled = False
        self.pixel_cells_analyzed = 0
        self.pixel_cells_skipped = 0
        self.orphan_tables_analyzed = 0
        self.orphan_boxes_proposed = 0
        self._owns_provider = page_provider is None
        self.provider = page_provider or PageCropProvider(
            document_path,
            scale=float(config.get("render_scale", 2.0)),
            cache_pages=int(config.get("cache_pages", 2)),
        )

    def _resolve_model(self) -> str:
        if isinstance(self.model, str) and self.model:
            return self.model
        response = self.httpx.get(
            self.base_url + "/v1/models",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models or not isinstance(models[0], Mapping):
            raise RuntimeError("Recovery endpoint returned no model metadata")
        model = models[0].get("id")
        if not isinstance(model, str) or not model:
            raise RuntimeError("Recovery endpoint returned no model id")
        self.model = model
        return model

    @staticmethod
    def _page_scale(image, page_size: Sequence[float]) -> tuple[float, float]:
        width = float(page_size[0]) if len(page_size) >= 2 else 0.0
        height = float(page_size[1]) if len(page_size) >= 2 else 0.0
        if width <= 0 or height <= 0:
            raise ValueError("Invalid page_size for bbox recovery")
        return image.width / width, image.height / height

    def _ink_analysis(
        self,
        image,
        page_size: Sequence[float],
        region_bbox: Sequence[float],
        existing: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        scale_x, scale_y = self._page_scale(image, page_size)
        bbox = _valid_bbox(region_bbox)
        if bbox is None:
            return {"ink_ratio": 0.0, "uncovered_ratio": 0.0, "ink_bbox": None}
        inset_x = max((bbox[2] - bbox[0]) * 0.025, 2 / scale_x)
        inset_y = max((bbox[3] - bbox[1]) * 0.06, 2 / scale_y)
        crop_box = (
            max(0, int((bbox[0] + inset_x) * scale_x)),
            max(0, int((bbox[1] + inset_y) * scale_y)),
            min(image.width, int(math.ceil((bbox[2] - inset_x) * scale_x))),
            min(image.height, int(math.ceil((bbox[3] - inset_y) * scale_y))),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return {"ink_ratio": 0.0, "uncovered_ratio": 0.0, "ink_bbox": None}
        crop = image.crop(crop_box).convert("L")
        try:
            histogram = crop.histogram()
            total_pixels = max(crop.width * crop.height, 1)
            target = total_pixels * 0.9
            cumulative = 0
            background = 255
            for value, count in enumerate(histogram):
                cumulative += count
                if cumulative >= target:
                    background = value
                    break
            threshold = max(40, min(220, background - 25))
            pixels = np.asarray(crop, dtype=np.uint8)
            dark_mask = pixels < threshold
            dark_y, dark_x = np.nonzero(dark_mask)
            dark_count = int(dark_x.size)
            if dark_count == 0:
                return {"ink_ratio": 0.0, "uncovered_ratio": 0.0, "ink_bbox": None}
            existing_bboxes = [
                valid
                for item in existing
                if isinstance(item, Mapping)
                for valid in [_valid_bbox(item.get("bbox"))]
                if valid is not None
            ]
            covered_mask = np.zeros_like(dark_mask, dtype=bool)
            for existing_bbox in existing_bboxes:
                left = max(
                    0,
                    int(math.floor(existing_bbox[0] * scale_x)) - crop_box[0],
                )
                top = max(
                    0,
                    int(math.floor(existing_bbox[1] * scale_y)) - crop_box[1],
                )
                right = min(
                    crop.width,
                    int(math.ceil(existing_bbox[2] * scale_x)) - crop_box[0] + 1,
                )
                bottom = min(
                    crop.height,
                    int(math.ceil(existing_bbox[3] * scale_y)) - crop_box[1] + 1,
                )
                if right > left and bottom > top:
                    covered_mask[top:bottom, left:right] = True
            uncovered = int(np.count_nonzero(dark_mask & ~covered_mask))
            min_x = float((crop_box[0] + int(dark_x.min())) / scale_x)
            max_x = float((crop_box[0] + int(dark_x.max())) / scale_x)
            min_y = float((crop_box[1] + int(dark_y.min())) / scale_y)
            max_y = float((crop_box[1] + int(dark_y.max())) / scale_y)
            padding_x = max(1.5 / scale_x, (max_x - min_x) * 0.02)
            padding_y = max(1.5 / scale_y, (max_y - min_y) * 0.08)
            ink_bbox = _clip_bbox(
                (
                    min_x - padding_x,
                    min_y - padding_y,
                    max_x + padding_x,
                    max_y + padding_y,
                ),
                bbox,
            )
            return {
                "ink_ratio": dark_count / total_pixels,
                "uncovered_ratio": uncovered / dark_count,
                "ink_bbox": list(ink_bbox) if ink_bbox is not None else None,
            }
        finally:
            crop.close()

    def _suspicious_cells(
        self,
        image,
        page_size: Sequence[float],
        table: Mapping[str, Any],
        *,
        missing_only: bool = False,
    ) -> list[dict[str, Any]]:
        min_ink_ratio = float(self.config.get("min_ink_ratio", 0.002))
        min_uncovered_ratio = float(self.config.get("min_uncovered_ink_ratio", 0.15))
        suspicious = []
        for cell in table.get("cells", []):
            if not isinstance(cell, Mapping):
                continue
            existing = cell.get("existing", [])
            if missing_only and existing:
                self.pixel_cells_skipped += 1
                continue
            self.pixel_cells_analyzed += 1
            analysis = self._ink_analysis(
                image,
                page_size,
                cell.get("bbox", []),
                cell.get("existing", []),
            )
            reasons = list(
                reason
                for reason in cell.get("reasons", [])
                if isinstance(reason, str) and reason != "missing_content_bbox"
            )
            if (
                not existing
                and analysis["ink_ratio"] >= min_ink_ratio
                and analysis["ink_bbox"] is not None
            ):
                reasons.append("missing_content_bbox")
                reasons.append("visible_ink_without_bbox")
            if (
                existing
                and analysis["uncovered_ratio"] >= min_uncovered_ratio
                and analysis["ink_bbox"] is not None
            ):
                reasons.append("uncovered_ink")
            if not reasons:
                continue
            suspicious.append(
                {
                    **dict(cell),
                    "reasons": sorted(set(reasons)),
                    "pixel_ink_bbox": analysis["ink_bbox"],
                    "ink_ratio": round(float(analysis["ink_ratio"]), 6),
                    "uncovered_ink_ratio": round(
                        float(analysis["uncovered_ratio"]),
                        6,
                    ),
                }
            )
        return suspicious

    def _table_orphan_proposals(
        self,
        image,
        page_size: Sequence[float],
        table: Mapping[str, Any],
        max_items: int,
    ) -> list[dict[str, Any]]:
        """Find text-line ink inside a Table but outside every OCR Cell bbox."""
        if (
            not self.config.get("table_orphan_recovery_enabled", True)
            or max_items <= 0
        ):
            return []
        table_bbox = _valid_bbox(table.get("bbox"))
        cells = [
            cell
            for cell in table.get("cells", [])
            if isinstance(cell, Mapping)
            and _valid_bbox(cell.get("bbox")) is not None
        ]
        if table_bbox is None or not cells:
            return []
        scale_x, scale_y = self._page_scale(image, page_size)
        crop_box = (
            max(0, int(math.floor(table_bbox[0] * scale_x))),
            max(0, int(math.floor(table_bbox[1] * scale_y))),
            min(image.width, int(math.ceil(table_bbox[2] * scale_x))),
            min(image.height, int(math.ceil(table_bbox[3] * scale_y))),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return []
        crop = image.crop(crop_box).convert("L")
        self.orphan_tables_analyzed += 1
        try:
            histogram = crop.histogram()
            total_pixels = max(crop.width * crop.height, 1)
            target = total_pixels * 0.9
            cumulative = 0
            background = 255
            for value, count in enumerate(histogram):
                cumulative += count
                if cumulative >= target:
                    background = value
                    break
            threshold = max(40, min(220, background - 25))
            dark_mask = np.asarray(crop, dtype=np.uint8) < threshold
            cell_mask = np.zeros_like(dark_mask, dtype=bool)
            padding_points = max(
                float(self.config.get("table_orphan_cell_padding", 1.5)),
                0.0,
            )
            padding_x = int(math.ceil(padding_points * scale_x))
            padding_y = int(math.ceil(padding_points * scale_y))

            def mark_bbox(mask, raw_bbox: Any, extra_x: int, extra_y: int) -> None:
                bbox = _valid_bbox(raw_bbox)
                if bbox is None:
                    return
                left = max(
                    0,
                    int(math.floor(bbox[0] * scale_x)) - crop_box[0] - extra_x,
                )
                top = max(
                    0,
                    int(math.floor(bbox[1] * scale_y)) - crop_box[1] - extra_y,
                )
                right = min(
                    crop.width,
                    int(math.ceil(bbox[2] * scale_x)) - crop_box[0] + extra_x,
                )
                bottom = min(
                    crop.height,
                    int(math.ceil(bbox[3] * scale_y)) - crop_box[1] + extra_y,
                )
                if right > left and bottom > top:
                    mask[top:bottom, left:right] = True

            existing_mask = np.zeros_like(dark_mask, dtype=bool)
            existing_bboxes = []
            for cell in cells:
                mark_bbox(cell_mask, cell.get("bbox"), padding_x, padding_y)
                for box in cell.get("existing", []):
                    if not isinstance(box, Mapping):
                        continue
                    bbox = _valid_bbox(box.get("bbox"))
                    if bbox is not None:
                        existing_bboxes.append(bbox)
                        mark_bbox(existing_mask, bbox, padding_x, padding_y)

            orphan_mask = dark_mask & ~cell_mask & ~existing_mask
            border_px = max(1, int(round(min(scale_x, scale_y))))
            orphan_mask[:border_px, :] = False
            orphan_mask[-border_px:, :] = False
            orphan_mask[:, :border_px] = False
            orphan_mask[:, -border_px:] = False

            horizontal_ratio = float(
                self.config.get("table_orphan_horizontal_line_ratio", 0.5)
            )
            vertical_ratio = float(
                self.config.get("table_orphan_vertical_line_ratio", 0.5)
            )
            long_rows = np.flatnonzero(
                np.count_nonzero(orphan_mask, axis=1)
                >= max(int(crop.width * horizontal_ratio), 1)
            )
            long_columns = np.flatnonzero(
                np.count_nonzero(orphan_mask, axis=0)
                >= max(int(crop.height * vertical_ratio), 1)
            )
            for row in long_rows:
                orphan_mask[max(0, row - 1) : min(crop.height, row + 2), :] = False
            for column in long_columns:
                orphan_mask[:, max(0, column - 1) : min(crop.width, column + 2)] = False

            minimum_row_ink = max(
                int(self.config.get("table_orphan_min_row_ink_pixels", 4)),
                int(crop.width * 0.002),
            )
            active_rows = np.flatnonzero(
                np.count_nonzero(orphan_mask, axis=1) >= minimum_row_ink
            )
            if active_rows.size == 0:
                return []
            max_gap = max(
                int(
                    round(
                        float(self.config.get("table_orphan_line_gap", 0.75))
                        * scale_y
                    )
                ),
                1,
            )
            bands: list[tuple[int, int]] = []
            start = previous = int(active_rows[0])
            for raw_row in active_rows[1:]:
                row = int(raw_row)
                if row - previous > max_gap + 1:
                    bands.append((start, previous + 1))
                    start = row
                previous = row
            bands.append((start, previous + 1))

            proposals = []
            minimum_height = max(
                float(self.config.get("table_orphan_min_line_height", 4.0)),
                0.0,
            )
            maximum_height_ratio = float(
                self.config.get("table_orphan_max_line_height_ratio", 0.07)
            )
            minimum_width = max(
                float(self.config.get("table_orphan_min_line_width", 20.0)),
                0.0,
            )
            minimum_ink_density = float(
                self.config.get("table_orphan_min_ink_density", 0.01)
            )
            maximum_ink_density = float(
                self.config.get("table_orphan_max_ink_density", 0.7)
            )
            cells_with_bbox = [
                (cell, _valid_bbox(cell.get("bbox"))) for cell in cells
            ]
            for top, bottom in bands:
                band_mask = orphan_mask[top:bottom, :]
                dark_y, dark_x = np.nonzero(band_mask)
                if dark_x.size < minimum_row_ink:
                    continue
                min_x = int(dark_x.min())
                max_x = int(dark_x.max()) + 1
                min_y = top + int(dark_y.min())
                max_y = top + int(dark_y.max()) + 1
                raw_height = (max_y - min_y) / scale_y
                if raw_height < float(
                    self.config.get("table_orphan_min_dark_height", 2.0)
                ):
                    continue
                pixel_area = max((max_x - min_x) * (max_y - min_y), 1)
                ink_density = dark_x.size / pixel_area
                if not minimum_ink_density <= ink_density <= maximum_ink_density:
                    continue
                bbox = _clip_bbox(
                    (
                        (crop_box[0] + min_x) / scale_x - 1.0,
                        (crop_box[1] + min_y) / scale_y - 0.8,
                        (crop_box[0] + max_x) / scale_x + 1.0,
                        (crop_box[1] + max_y) / scale_y + 0.8,
                    ),
                    table_bbox,
                )
                if bbox is None:
                    continue
                height = bbox[3] - bbox[1]
                maximum_height = max(
                    (table_bbox[3] - table_bbox[1]) * maximum_height_ratio,
                    minimum_height * 3,
                )
                if height < minimum_height or height > maximum_height:
                    continue
                if bbox[2] - bbox[0] < minimum_width:
                    continue
                if any(
                    _bbox_overlap_ratio(bbox, existing_bbox) >= 0.5
                    for existing_bbox in existing_bboxes
                ):
                    continue

                def cell_distance(
                    item: tuple[Mapping[str, Any], tuple[float, float, float, float] | None]
                ) -> tuple[float, float]:
                    cell, cell_bbox = item
                    if cell_bbox is None:
                        return (float("inf"), float("inf"))
                    if bbox[1] >= cell_bbox[3]:
                        vertical = bbox[1] - cell_bbox[3]
                    elif cell_bbox[1] >= bbox[3]:
                        vertical = cell_bbox[1] - bbox[3]
                    else:
                        vertical = 0.0
                    horizontal_overlap = max(
                        0.0,
                        min(bbox[2], cell_bbox[2]) - max(bbox[0], cell_bbox[0]),
                    )
                    horizontal = 0.0 if horizontal_overlap > 0 else min(
                        abs(bbox[0] - cell_bbox[2]),
                        abs(cell_bbox[0] - bbox[2]),
                    )
                    row = cell.get("row_end")
                    return (vertical + horizontal, -float(row or 0))

                nearest_cell, _nearest_bbox = min(
                    cells_with_bbox,
                    key=cell_distance,
                )
                cell_id = nearest_cell.get("id")
                table_id = table.get("id")
                if not isinstance(cell_id, str) or not isinstance(table_id, str):
                    continue
                proposals.append(
                    {
                        "action": "add_orphan",
                        "table_id": table_id,
                        "cell_id": cell_id,
                        "target_id": "",
                        "bbox": list(bbox),
                        "confidence": float(
                            self.config.get("table_orphan_confidence", 0.92)
                        ),
                        "recovery_reasons": ["table_ink_outside_cells"],
                        "recovery_source": "local_table_orphan_ink",
                    }
                )
                if len(proposals) >= max_items:
                    break
            self.orphan_boxes_proposed += len(proposals)
            return proposals
        finally:
            crop.close()

    def _overlay_data_url(
        self,
        image,
        page_size: Sequence[float],
        table: Mapping[str, Any],
        suspicious: Sequence[Mapping[str, Any]],
    ) -> str:
        try:
            from PIL import ImageDraw
        except ImportError as exc:
            raise RuntimeError("BBox recovery overlays require Pillow") from exc
        table_bbox = _valid_bbox(table.get("bbox"))
        if table_bbox is None:
            raise ValueError("Recovery Table is missing a valid bbox")
        scale_x, scale_y = self._page_scale(image, page_size)
        crop_box = (
            max(0, int(table_bbox[0] * scale_x)),
            max(0, int(table_bbox[1] * scale_y)),
            min(image.width, int(math.ceil(table_bbox[2] * scale_x))),
            min(image.height, int(math.ceil(table_bbox[3] * scale_y))),
        )
        overlay = image.crop(crop_box).convert("RGB")
        try:
            draw = ImageDraw.Draw(overlay)

            def local_box(raw_bbox):
                bbox = _valid_bbox(raw_bbox)
                if bbox is None:
                    return None
                return (
                    int(bbox[0] * scale_x) - crop_box[0],
                    int(bbox[1] * scale_y) - crop_box[1],
                    int(bbox[2] * scale_x) - crop_box[0],
                    int(bbox[3] * scale_y) - crop_box[1],
                )

            suspicious_ids = {str(item.get("id")) for item in suspicious}
            for cell in table.get("cells", []):
                if not isinstance(cell, Mapping):
                    continue
                bbox = local_box(cell.get("bbox"))
                if bbox is None:
                    continue
                color = (220, 55, 55) if str(cell.get("id")) in suspicious_ids else (140, 145, 155)
                draw.rectangle(bbox, outline=color, width=2)
                label = str(cell.get("id", "")).rsplit("-", 1)[-1].upper()
                draw.text((bbox[0] + 2, bbox[1] + 2), label, fill=color)
                for existing in cell.get("existing", []):
                    if not isinstance(existing, Mapping):
                        continue
                    content_box = local_box(existing.get("bbox"))
                    if content_box is not None:
                        draw.rectangle(content_box, outline=(0, 190, 210), width=3)
            buffer = io.BytesIO()
            overlay.save(
                buffer,
                format="JPEG",
                quality=int(self.config.get("jpeg_quality", 92)),
                optimize=True,
            )
            return "data:image/jpeg;base64," + base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
        finally:
            overlay.close()

    @staticmethod
    def _response_schema(
        cell_ids: Sequence[str],
        box_ids: Sequence[str],
        max_items: int,
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "adjust"]},
                            "cell_id": {"type": "string", "enum": list(cell_ids)},
                            "target_id": {"type": "string", "enum": ["", *box_ids]},
                            "bbox": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["action", "cell_id", "target_id", "bbox", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }

    @staticmethod
    def _normalized_to_page_bbox(
        raw_bbox: Any,
        table_bbox: Sequence[float],
    ) -> tuple[float, float, float, float] | None:
        normalized = _valid_bbox(raw_bbox)
        if normalized is None or any(value < 0 or value > 1000 for value in normalized):
            return None
        width = table_bbox[2] - table_bbox[0]
        height = table_bbox[3] - table_bbox[1]
        return _valid_bbox(
            (
                table_bbox[0] + normalized[0] / 1000 * width,
                table_bbox[1] + normalized[1] / 1000 * height,
                table_bbox[0] + normalized[2] / 1000 * width,
                table_bbox[1] + normalized[3] / 1000 * height,
            )
        )

    def _review_table(
        self,
        page_index: int,
        page_size: Sequence[float],
        table: Mapping[str, Any],
        suspicious: Sequence[Mapping[str, Any]],
        image,
        max_items: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        table_bbox = _valid_bbox(table.get("bbox"))
        if table_bbox is None:
            return [], {"status": "invalid_table_bbox"}
        if max_items <= 0:
            return [], {"status": "proposal_limit"}
        overlay_url = self._overlay_data_url(image, page_size, table, suspicious)
        cell_ids = [str(cell.get("id")) for cell in suspicious]
        box_ids = [
            str(box.get("id"))
            for cell in suspicious
            for box in cell.get("existing", [])
            if isinstance(box, Mapping)
        ]
        evidence = {
            "table_id": table.get("id"),
            "coordinate_system": "bbox values are integers from 0 to 1000 relative to the Table image",
            "cells": [
                {
                    "cell_id": cell.get("id"),
                    "ocr_text": str(cell.get("text", ""))[:256],
                    "reasons": cell.get("reasons", []),
                    "existing_box_ids": [
                        box.get("id")
                        for box in cell.get("existing", [])
                        if isinstance(box, Mapping)
                    ],
                }
                for cell in suspicious
            ],
        }
        prompt = (
            "Inspect only the red suspicious Table Cells in the overlay. Cyan boxes are "
            "existing OCR content boxes. Propose add only when visible text lacks a cyan "
            "box. Propose adjust only when an existing cyan box misses visible characters "
            "or includes substantial neighboring content. Return no item when the existing "
            "geometry is acceptable. Coordinates are rough normalized Table-image values; "
            "a deterministic pixel refiner will produce the final bbox. Never change Table "
            "or Cell boundaries.\n"
            f"evidence_data={json.dumps(evidence, ensure_ascii=False)}"
        )
        token_cap = max(int(self.config.get("structured_max_tokens_cap", 768)), 1)
        payload = {
            "model": self._resolve_model(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": overlay_url}},
                    ],
                }
            ],
            "temperature": float(self.config.get("temperature", 0.0)),
            "top_p": float(self.config.get("top_p", 1.0)),
            "max_tokens": min(
                int(self.config.get("max_tokens", 1024)),
                token_cap,
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "bbox_recovery_review",
                    "schema": self._response_schema(
                        cell_ids,
                        box_ids,
                        min(
                            len(suspicious),
                            int(self.config.get("max_proposals_per_table", 30)),
                            max_items,
                        ),
                    ),
                    "strict": True,
                },
            },
        }
        seed = self.config.get("seed")
        if isinstance(seed, int):
            payload["seed"] = seed
        started = time.monotonic()
        request_headers = dict(self.headers)
        request_headers["X-Custom-Hybrid-Max-Tokens"] = str(
            payload["max_tokens"]
        )
        response = self.httpx.post(
            self.base_url + "/v1/chat/completions",
            headers=request_headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_payload = response.json()
        choice = response_payload["choices"][0]
        content = choice["message"]["content"]
        parsed = _parse_json_object(content)
        raw_items = parsed.get("items", []) if isinstance(parsed, Mapping) else []
        by_cell = {str(cell.get("id")): cell for cell in suspicious}
        accepted = []
        invalid = 0
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, Mapping):
                invalid += 1
                continue
            cell_id = item.get("cell_id")
            cell = by_cell.get(cell_id) if isinstance(cell_id, str) else None
            action = item.get("action")
            target_id = item.get("target_id")
            confidence = item.get("confidence")
            rough_bbox = self._normalized_to_page_bbox(item.get("bbox"), table_bbox)
            cell_bbox = _valid_bbox(cell.get("bbox")) if cell is not None else None
            if (
                cell is None
                or cell_bbox is None
                or action not in {"add", "adjust"}
                or not isinstance(target_id, str)
                or not isinstance(confidence, (int, float))
                or rough_bbox is None
            ):
                invalid += 1
                continue
            rough_bbox = _clip_bbox(rough_bbox, cell_bbox)
            if rough_bbox is None:
                invalid += 1
                continue
            refined = self._ink_analysis(image, page_size, rough_bbox)["ink_bbox"]
            if refined is None:
                refined = cell.get("pixel_ink_bbox")
            refined_bbox = _clip_bbox(refined, cell_bbox) if refined is not None else None
            if refined_bbox is None:
                invalid += 1
                continue
            accepted.append(
                {
                    "action": action,
                    "cell_id": cell_id,
                    "target_id": target_id,
                    "bbox": list(refined_bbox),
                    "confidence": float(confidence),
                    "reviewer_bbox_normalized": item.get("bbox"),
                    "recovery_reasons": cell.get("reasons", []),
                }
            )
        usage = response_payload.get("usage", {})
        return accepted, {
            "status": "ok" if parsed is not None else "invalid_schema",
            "table_id": table.get("id"),
            "suspicious_cells": len(suspicious),
            "proposals": len(accepted),
            "invalid_outputs": invalid + int(parsed is None),
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }

    def _local_missing_proposals(
        self,
        suspicious: Sequence[Mapping[str, Any]],
        max_items: int,
    ) -> list[dict[str, Any]]:
        if not self.config.get("local_missing_enabled", True) or max_items <= 0:
            return []
        proposals = []
        for cell in suspicious:
            reasons = cell.get("reasons", [])
            if not isinstance(reasons, list) or "missing_content_bbox" not in reasons:
                continue
            bbox = _valid_bbox(cell.get("pixel_ink_bbox"))
            cell_id = cell.get("id")
            if bbox is None or not isinstance(cell_id, str):
                continue
            proposals.append(
                {
                    "action": "add",
                    "cell_id": cell_id,
                    "target_id": "",
                    "bbox": list(bbox),
                    "confidence": float(
                        self.config.get("local_pixel_confidence", 0.95)
                    ),
                    "recovery_reasons": sorted(set(reasons)),
                    "recovery_source": "local_pixel_ink",
                }
            )
            if len(proposals) >= max_items:
                break
        return proposals

    def __call__(
        self,
        page_index: int,
        page_size: Sequence[float],
        tables: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        analyzed_before = self.pixel_cells_analyzed
        skipped_before = self.pixel_cells_skipped
        orphan_tables_before = self.orphan_tables_analyzed
        orphan_boxes_before = self.orphan_boxes_proposed
        pixel_analysis_seconds = 0.0
        image = self.provider.get_page(page_index)
        resolved_model = None
        model_resolution_error = None
        try:
            resolved_model = self._resolve_model().casefold()
        except Exception as exc:
            model_resolution_error = type(exc).__name__
        mineru_local_only = bool(
            isinstance(resolved_model, str)
            and "mineru" in resolved_model
            and self.config.get("local_missing_enabled", True)
            and self.config.get("skip_vlm_for_mineru_models", True)
            and not self.config.get("review_after_local_recovery", False)
        )
        max_requests = max(int(self.config.get("max_requests_per_document", 10)), 0)
        max_tables = max(int(self.config.get("max_tables_per_document", 10)), 0)
        max_proposals = max(
            int(self.config.get("max_proposals_per_document", 100)),
            0,
        )
        items = []
        batches = []
        requests = 0
        tables_reviewed = 0
        errors = 0
        table_budget_skips = 0
        proposal_budget_skips = 0
        local_proposals = 0
        protocol_failures = 0
        protocol_skips = 0
        local_only_tables = 0
        for table in tables:
            if mineru_local_only:
                local_only_tables += 1
            pixel_started = time.monotonic()
            try:
                suspicious = self._suspicious_cells(
                    image,
                    page_size,
                    table,
                    missing_only=mineru_local_only,
                )
            finally:
                pixel_analysis_seconds += time.monotonic() - pixel_started
            remaining_proposals = max(max_proposals - self.proposals_returned, 0)
            if remaining_proposals <= 0:
                proposal_budget_skips += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "proposal_limit",
                        "suspicious_cells": len(suspicious),
                    }
                )
                continue
            per_table_limit = max(
                int(self.config.get("max_proposals_per_table", 30)),
                0,
            )
            missing_items = self._local_missing_proposals(
                suspicious,
                min(remaining_proposals, per_table_limit),
            )
            orphan_limit = min(
                max(remaining_proposals - len(missing_items), 0),
                max(per_table_limit - len(missing_items), 0),
                max(
                    int(
                        self.config.get(
                            "table_orphan_max_boxes_per_table",
                            12,
                        )
                    ),
                    0,
                ),
            )
            pixel_started = time.monotonic()
            try:
                orphan_items = self._table_orphan_proposals(
                    image,
                    page_size,
                    table,
                    orphan_limit,
                )
            finally:
                pixel_analysis_seconds += time.monotonic() - pixel_started
            local_items = missing_items + orphan_items
            if not suspicious and not local_items:
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "not_suspicious",
                    }
                )
                continue
            reviewed_locally = bool(local_items)
            if local_items:
                items.extend(local_items)
                local_count = len(local_items)
                local_proposals += local_count
                self.proposals_returned += local_count
                tables_reviewed += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "local_pixel_recovery",
                        "suspicious_cells": len(suspicious),
                        "proposals": local_count,
                        "orphan_proposals": len(orphan_items),
                    }
                )
                if not self.config.get("review_after_local_recovery", False):
                    continue
                remaining_proposals = max(
                    remaining_proposals - local_count,
                    0,
                )
                if remaining_proposals <= 0:
                    continue
            if self.tables_reviewed >= max_tables:
                table_budget_skips += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "table_limit",
                        "suspicious_cells": len(suspicious),
                    }
                )
                continue
            if self.protocol_disabled:
                protocol_skips += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "protocol_circuit_breaker",
                        "suspicious_cells": len(suspicious),
                    }
                )
                continue
            if model_resolution_error is not None:
                errors += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "model_resolution_error",
                        "error": model_resolution_error,
                    }
                )
                continue
            if (
                isinstance(resolved_model, str)
                and "mineru" in resolved_model
                and self.config.get("skip_vlm_for_mineru_models", True)
            ):
                protocol_skips += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "mineru_geometry_protocol_skipped",
                        "suspicious_cells": len(suspicious),
                    }
                )
                continue
            if self.requests_made >= max_requests:
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "request_limit",
                        "suspicious_cells": len(suspicious),
                    }
                )
                continue
            try:
                self.requests_made += 1
                requests += 1
                proposals, audit = self._review_table(
                    page_index,
                    page_size,
                    table,
                    suspicious,
                    image,
                    remaining_proposals,
                )
                proposals = proposals[:remaining_proposals]
                self.proposals_returned += len(proposals)
                self.tables_reviewed += 1
                if not reviewed_locally:
                    tables_reviewed += 1
                items.extend(proposals)
                batches.append({"page": page_index, **audit})
                if audit.get("status") == "invalid_schema":
                    protocol_failures += 1
                    if self.config.get(
                        "disable_after_invalid_schema",
                        True,
                    ):
                        self.protocol_disabled = True
            except Exception as exc:
                errors += 1
                batches.append(
                    {
                        "page": page_index,
                        "table_id": table.get("id"),
                        "status": "error",
                        "error": type(exc).__name__,
                    }
                )
        return {
            "items": items,
            "batches": batches,
            "requests": requests,
            "tables_reviewed": tables_reviewed,
            "errors": errors,
            "table_budget_skips": table_budget_skips,
            "proposal_budget_skips": proposal_budget_skips,
            "local_proposals": local_proposals,
            "protocol_failures": protocol_failures,
            "protocol_skips": protocol_skips,
            "local_only_tables": local_only_tables,
            "orphan_tables_analyzed": (
                self.orphan_tables_analyzed - orphan_tables_before
            ),
            "orphan_proposals": self.orphan_boxes_proposed - orphan_boxes_before,
            "pixel_cells_analyzed": (
                self.pixel_cells_analyzed - analyzed_before
            ),
            "pixel_cells_skipped": self.pixel_cells_skipped - skipped_before,
            "pixel_analysis_ms": round(
                pixel_analysis_seconds * 1000,
                3,
            ),
        }

    def close(self) -> None:
        if self._owns_provider:
            self.provider.close()

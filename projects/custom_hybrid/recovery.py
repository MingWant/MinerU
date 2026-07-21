"""Selective Table content-box recovery with VLM review and pixel refinement."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from projects.custom_hybrid.fusion import PageCropProvider


_LIST_MARKER_RE = re.compile(
    r"^(?:\(\s*\d{1,3}\s*\)|\d{1,3}\s*[.)、:：])$"
)
_TERMINAL_FIELD_RE = re.compile(
    r"(?:signature|signed|date|ID\s*/\s*Passport|簽署|签署|日期|身份[證证])",
    flags=re.IGNORECASE,
)
_TERMINAL_DATE_FIELD_RE = re.compile(r"(?:\bdate\b|日期)", flags=re.IGNORECASE)
_TERMINAL_IDENTIFIER_FIELD_RE = re.compile(
    r"(?:ID\s*/\s*Passport|身份[證证]|護照|护照)",
    flags=re.IGNORECASE,
)
_TERMINAL_SIGNATURE_FIELD_RE = re.compile(
    r"(?:signature|signed|簽署|签署)",
    flags=re.IGNORECASE,
)


def _terminal_field_kind(text: str) -> str | None:
    if _TERMINAL_DATE_FIELD_RE.search(text):
        return "date"
    if _TERMINAL_IDENTIFIER_FIELD_RE.search(text):
        return "identifier"
    if _TERMINAL_SIGNATURE_FIELD_RE.search(text):
        return "signature"
    return None


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


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


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
        api_key_env = config.get("api_key_env")
        api_key = (
            os.getenv(api_key_env.strip())
            if isinstance(api_key_env, str) and api_key_env.strip()
            else None
        )
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = config.get("model")
        self.requests_made = 0
        self.tables_reviewed = 0
        self.proposals_returned = 0
        self.protocol_disabled = False
        self.pixel_cells_analyzed = 0
        self.pixel_cells_skipped = 0
        self.diagonal_rules_removed = 0
        self.orphan_tables_analyzed = 0
        self.orphan_boxes_proposed = 0
        self.fringe_boxes_proposed = 0
        self.checkbox_tables_analyzed = 0
        self.checkbox_candidates = 0
        self.checkbox_boxes_proposed = 0
        self.checkbox_labels_merged = 0
        self.list_marker_labels_merged = 0
        self.ink_marker_labels_merged = 0
        self.checkbox_checked = 0
        self.checkbox_unchecked = 0
        self.checkbox_ambiguous = 0
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

    def _table_diagonal_rules(
        self,
        image,
        page_size: Sequence[float],
        table_bbox: Sequence[float],
    ) -> list[tuple[float, float, float, float]]:
        """Detect long non-axis-aligned rules once for the whole Table."""
        if not self.config.get("table_diagonal_rule_enabled", True):
            return []
        bbox = _valid_bbox(table_bbox)
        if bbox is None:
            return []
        try:
            import cv2
        except ImportError:
            return []
        scale_x, scale_y = self._page_scale(image, page_size)
        crop_box = (
            max(0, int(math.floor(bbox[0] * scale_x))),
            max(0, int(math.floor(bbox[1] * scale_y))),
            min(image.width, int(math.ceil(bbox[2] * scale_x))),
            min(image.height, int(math.ceil(bbox[3] * scale_y))),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return []
        crop = image.crop(crop_box).convert("L")
        try:
            pixels = np.asarray(crop, dtype=np.uint8)
            histogram = crop.histogram()
            target = max(crop.width * crop.height, 1) * 0.9
            cumulative = 0
            background = 255
            for value, count in enumerate(histogram):
                cumulative += count
                if cumulative >= target:
                    background = value
                    break
            dark_threshold = max(40, min(220, background - 25))
            binary = (pixels < dark_threshold).astype(np.uint8) * 255
            minimum_length = max(
                float(self.config.get("table_diagonal_rule_min_length", 80.0)),
                0.0,
            )
            minimum_length_pixels = max(
                int(round(minimum_length * min(scale_x, scale_y))),
                1,
            )
            maximum_gap = max(
                float(self.config.get("table_diagonal_rule_max_gap", 8.0)),
                0.0,
            )
            lines = cv2.HoughLinesP(
                binary,
                1,
                np.pi / 180,
                threshold=max(int(round(minimum_length_pixels * 0.35)), 12),
                minLineLength=minimum_length_pixels,
                maxLineGap=max(
                    int(round(maximum_gap * min(scale_x, scale_y))),
                    0,
                ),
            )
            if lines is None:
                return []
            minimum_angle = float(
                self.config.get("table_diagonal_rule_min_angle", 12.0)
            )
            maximum_angle = float(
                self.config.get("table_diagonal_rule_max_angle", 88.0)
            )
            rules: list[tuple[float, float, float, float]] = []
            for raw_line in lines:
                coordinates = np.asarray(raw_line).reshape(-1)
                if coordinates.size < 4:
                    continue
                x1, y1, x2, y2 = (int(value) for value in coordinates[:4])
                delta_x = (x2 - x1) / scale_x
                delta_y = (y2 - y1) / scale_y
                if math.hypot(delta_x, delta_y) < minimum_length:
                    continue
                angle = abs(math.degrees(math.atan2(delta_y, delta_x)))
                angle = min(angle, 180.0 - angle)
                if not minimum_angle <= angle <= maximum_angle:
                    continue
                rules.append(
                    (
                        (crop_box[0] + x1) / scale_x,
                        (crop_box[1] + y1) / scale_y,
                        (crop_box[0] + x2) / scale_x,
                        (crop_box[1] + y2) / scale_y,
                    )
                )
            return rules
        finally:
            crop.close()

    def _mask_diagonal_rules(
        self,
        mask: np.ndarray,
        crop_box: Sequence[int],
        scale_x: float,
        scale_y: float,
        rules: Sequence[Sequence[float]],
    ) -> None:
        if not rules:
            return
        try:
            import cv2
        except ImportError:
            return
        padding = max(
            float(self.config.get("table_diagonal_rule_padding", 1.5)),
            0.0,
        )
        extension = max(
            float(self.config.get("table_diagonal_rule_extension", 20.0)),
            0.0,
        )
        rule_mask = np.zeros_like(mask, dtype=np.uint8)
        thickness = max(
            int(round((2.0 * padding + 1.0) * min(scale_x, scale_y))),
            1,
        )
        for rule in rules:
            if len(rule) != 4:
                continue
            start_x, start_y, end_x, end_y = (float(value) for value in rule)
            delta_x = end_x - start_x
            delta_y = end_y - start_y
            length = math.hypot(delta_x, delta_y)
            if length > 0 and extension > 0:
                scale = extension / length
                start_x -= delta_x * scale
                start_y -= delta_y * scale
                end_x += delta_x * scale
                end_y += delta_y * scale
            cv2.line(
                rule_mask,
                (
                    int(round(start_x * scale_x)) - int(crop_box[0]),
                    int(round(start_y * scale_y)) - int(crop_box[1]),
                ),
                (
                    int(round(end_x * scale_x)) - int(crop_box[0]),
                    int(round(end_y * scale_y)) - int(crop_box[1]),
                ),
                1,
                thickness=thickness,
            )
        mask[rule_mask.astype(bool)] = False

    def _ink_analysis(
        self,
        image,
        page_size: Sequence[float],
        region_bbox: Sequence[float],
        existing: Sequence[Mapping[str, Any]] = (),
        diagonal_rules: Sequence[Sequence[float]] = (),
        minimum_row_ink_pixels: int | None = None,
        maximum_line_gap_points: float | None = None,
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

            # Cell borders can survive the inset when MinerU's Cell geometry
            # overlaps a neighboring row/column. Remove long, straight runs
            # before deriving a content bbox; character strokes are shorter
            # than the minimum rule length and remain intact.
            cleaned_mask = dark_mask.copy()
            horizontal_ratio = min(
                max(float(self.config.get("cell_horizontal_rule_ratio", 0.6)), 0.0),
                1.0,
            )
            vertical_ratio = min(
                max(float(self.config.get("cell_vertical_rule_ratio", 0.6)), 0.0),
                1.0,
            )
            minimum_rule_length = max(
                float(self.config.get("cell_rule_min_length", 18.0)),
                0.0,
            )
            rule_padding = max(
                float(self.config.get("cell_rule_padding", 1.0)),
                0.0,
            )
            maximum_rule_thickness = max(
                float(self.config.get("cell_rule_max_thickness", 2.5)),
                0.0,
            )

            def thin_rule_indices(
                indices: np.ndarray,
                maximum_pixels: int,
            ) -> np.ndarray:
                if indices.size == 0:
                    return indices
                accepted: list[int] = []
                start = previous = int(indices[0])
                for raw_index in indices[1:]:
                    index = int(raw_index)
                    if index != previous + 1:
                        if previous - start + 1 <= maximum_pixels:
                            accepted.extend(range(start, previous + 1))
                        start = index
                    previous = index
                if previous - start + 1 <= maximum_pixels:
                    accepted.extend(range(start, previous + 1))
                return np.asarray(accepted, dtype=int)

            horizontal_length = crop.width / scale_x
            vertical_length = crop.height / scale_y
            horizontal_rows = np.flatnonzero(
                np.count_nonzero(dark_mask, axis=1)
                >= max(int(round(crop.width * horizontal_ratio)), 1)
            )
            if horizontal_length < minimum_rule_length:
                horizontal_rows = np.asarray([], dtype=int)
            else:
                horizontal_rows = thin_rule_indices(
                    horizontal_rows,
                    max(int(math.ceil(maximum_rule_thickness * scale_y)), 1),
                )
            vertical_columns = np.flatnonzero(
                np.count_nonzero(dark_mask, axis=0)
                >= max(int(round(crop.height * vertical_ratio)), 1)
            )
            if vertical_length < minimum_rule_length:
                vertical_columns = np.asarray([], dtype=int)
            else:
                vertical_columns = thin_rule_indices(
                    vertical_columns,
                    max(int(math.ceil(maximum_rule_thickness * scale_x)), 1),
                )
            row_padding = max(int(round(rule_padding * scale_y)), 0)
            column_padding = max(int(round(rule_padding * scale_x)), 0)
            for row in horizontal_rows:
                start = max(0, int(row) - row_padding)
                stop = min(crop.height, int(row) + row_padding + 1)
                cleaned_mask[start:stop, :] = False
            for column in vertical_columns:
                start = max(0, int(column) - column_padding)
                stop = min(crop.width, int(column) + column_padding + 1)
                cleaned_mask[:, start:stop] = False
            self._mask_diagonal_rules(
                cleaned_mask,
                crop_box,
                scale_x,
                scale_y,
                diagonal_rules,
            )

            dark_y, dark_x = np.nonzero(cleaned_mask)
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
            existing_padding = max(
                float(self.config.get("existing_bbox_padding", 1.0)),
                0.0,
            )
            existing_padding_x = int(math.ceil(existing_padding * scale_x))
            existing_padding_y = int(math.ceil(existing_padding * scale_y))
            for existing_bbox in existing_bboxes:
                left = max(
                    0,
                    int(math.floor(existing_bbox[0] * scale_x))
                    - crop_box[0]
                    - existing_padding_x,
                )
                top = max(
                    0,
                    int(math.floor(existing_bbox[1] * scale_y))
                    - crop_box[1]
                    - existing_padding_y,
                )
                right = min(
                    crop.width,
                    int(math.ceil(existing_bbox[2] * scale_x))
                    - crop_box[0]
                    + existing_padding_x
                    + 1,
                )
                bottom = min(
                    crop.height,
                    int(math.ceil(existing_bbox[3] * scale_y))
                    - crop_box[1]
                    + existing_padding_y
                    + 1,
                )
                if right > left and bottom > top:
                    covered_mask[top:bottom, left:right] = True
            visible_mask = cleaned_mask & ~covered_mask
            visible_y, visible_x = np.nonzero(visible_mask)
            visible_count = int(visible_x.size)
            if visible_count == 0:
                return {
                    "ink_ratio": dark_count / total_pixels,
                    "uncovered_ratio": 0.0,
                    "ink_bbox": None,
                    "ink_bboxes": [],
                    "uncovered_ink_pixels": 0,
                }
            min_x = float((crop_box[0] + int(visible_x.min())) / scale_x)
            max_x = float((crop_box[0] + int(visible_x.max())) / scale_x)
            min_y = float((crop_box[1] + int(visible_y.min())) / scale_y)
            max_y = float((crop_box[1] + int(visible_y.max())) / scale_y)
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
            minimum_row_ink = max(
                int(
                    minimum_row_ink_pixels
                    if minimum_row_ink_pixels is not None
                    else self.config.get("cell_line_min_row_ink_pixels", 2)
                ),
                1,
            )
            active_rows = np.flatnonzero(
                np.count_nonzero(visible_mask, axis=1) >= minimum_row_ink
            )
            line_bboxes: list[list[float]] = []
            ink_lines: list[dict[str, Any]] = []
            ink_components: list[dict[str, Any]] = []
            if active_rows.size:
                maximum_gap = max(
                    int(
                        round(
                            (
                                float(self.config.get("cell_line_gap", 1.5))
                                if maximum_line_gap_points is None
                                else float(maximum_line_gap_points)
                            )
                            * scale_y
                        )
                    ),
                    0,
                )
                bands: list[tuple[int, int]] = []
                start = previous = int(active_rows[0])
                for raw_row in active_rows[1:]:
                    row = int(raw_row)
                    if row - previous > maximum_gap + 1:
                        bands.append((start, previous + 1))
                        start = row
                    previous = row
                bands.append((start, previous + 1))
                minimum_line_height = max(
                    float(self.config.get("cell_line_min_height", 2.0)),
                    0.0,
                )
                minimum_dark_height = max(
                    float(self.config.get("cell_line_min_dark_height", 3.0)),
                    0.0,
                )
                minimum_line_width = max(
                    float(self.config.get("cell_line_min_width", 3.0)),
                    0.0,
                )
                for top, bottom in bands:
                    band_mask = visible_mask[top:bottom, :]
                    band_y, band_x = np.nonzero(band_mask)
                    if band_x.size == 0:
                        continue
                    line_min_x = float(
                        (crop_box[0] + int(band_x.min())) / scale_x
                    )
                    line_max_x = float(
                        (crop_box[0] + int(band_x.max())) / scale_x
                    )
                    line_min_y = float(
                        (crop_box[1] + top + int(band_y.min())) / scale_y
                    )
                    line_max_y = float(
                        (crop_box[1] + top + int(band_y.max())) / scale_y
                    )
                    dark_height = line_max_y - line_min_y + 1 / scale_y
                    if dark_height < minimum_dark_height:
                        continue
                    line_padding_x = max(
                        1.5 / scale_x,
                        (line_max_x - line_min_x) * 0.02,
                    )
                    line_padding_y = max(
                        1.5 / scale_y,
                        (line_max_y - line_min_y) * 0.08,
                    )
                    line_bbox = _clip_bbox(
                        (
                            line_min_x - line_padding_x,
                            line_min_y - line_padding_y,
                            line_max_x + line_padding_x,
                            line_max_y + line_padding_y,
                        ),
                        bbox,
                    )
                    if line_bbox is None:
                        continue
                    if (
                        line_bbox[3] - line_bbox[1] < minimum_line_height
                        or line_bbox[2] - line_bbox[0] < minimum_line_width
                    ):
                        continue
                    line_bboxes.append(list(line_bbox))
                    pixel_area = max(
                        (int(band_x.max()) - int(band_x.min()) + 1)
                        * (int(band_y.max()) - int(band_y.min()) + 1),
                        1,
                    )
                    ink_lines.append(
                        {
                            "bbox": list(line_bbox),
                            "ink_density": round(float(band_x.size / pixel_area), 6),
                            "dark_height": round(float(dark_height), 6),
                        }
                    )
                    active_columns = np.flatnonzero(
                        np.count_nonzero(band_mask, axis=0) > 0
                    )
                    if active_columns.size:
                        maximum_component_gap = max(
                            int(
                                round(
                                    float(
                                        self.config.get(
                                            "cell_component_horizontal_gap",
                                            16.0,
                                        )
                                    )
                                    * scale_x
                                )
                            ),
                            1,
                        )
                        column_bands: list[tuple[int, int]] = []
                        left = previous_column = int(active_columns[0])
                        for raw_column in active_columns[1:]:
                            column = int(raw_column)
                            if column - previous_column > maximum_component_gap + 1:
                                column_bands.append((left, previous_column + 1))
                                left = column
                            previous_column = column
                        column_bands.append((left, previous_column + 1))
                        for left, right in column_bands:
                            component_mask = band_mask[:, left:right]
                            component_y, component_x = np.nonzero(component_mask)
                            if component_x.size == 0:
                                continue
                            component_min_x = left + int(component_x.min())
                            component_max_x = left + int(component_x.max())
                            component_min_y = top + int(component_y.min())
                            component_max_y = top + int(component_y.max())
                            component_dark_height = (
                                component_max_y - component_min_y + 1
                            ) / scale_y
                            component_width = (
                                component_max_x - component_min_x + 1
                            ) / scale_x
                            if (
                                component_dark_height < minimum_dark_height
                                or component_width < minimum_line_width
                            ):
                                continue
                            component_bbox = _clip_bbox(
                                (
                                    (crop_box[0] + component_min_x) / scale_x
                                    - 1.0,
                                    (crop_box[1] + component_min_y) / scale_y
                                    - 0.8,
                                    (crop_box[0] + component_max_x) / scale_x
                                    + 1.0,
                                    (crop_box[1] + component_max_y) / scale_y
                                    + 0.8,
                                ),
                                bbox,
                            )
                            if component_bbox is None:
                                continue
                            component_area = max(
                                (component_max_x - component_min_x + 1)
                                * (component_max_y - component_min_y + 1),
                                1,
                            )
                            ink_components.append(
                                {
                                    "bbox": list(component_bbox),
                                    "ink_density": round(
                                        float(component_x.size / component_area),
                                        6,
                                    ),
                                    "dark_height": round(
                                        float(component_dark_height),
                                        6,
                                    ),
                                }
                            )
            return {
                "ink_ratio": dark_count / total_pixels,
                "uncovered_ratio": visible_count / dark_count,
                "ink_bbox": list(ink_bbox) if ink_bbox is not None else None,
                "ink_bboxes": line_bboxes,
                "ink_lines": ink_lines,
                "ink_components": ink_components,
                "uncovered_ink_pixels": visible_count,
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
        diagonal_rules: Sequence[Sequence[float]] = (),
    ) -> list[dict[str, Any]]:
        min_ink_ratio = float(self.config.get("min_ink_ratio", 0.002))
        max_ink_ratio = float(self.config.get("max_cell_ink_ratio", 0.65))
        min_uncovered_ratio = float(self.config.get("min_uncovered_ink_ratio", 0.15))
        minimum_uncovered_pixels = max(
            int(self.config.get("min_uncovered_ink_pixels", 12)),
            1,
        )
        local_uncovered_enabled = bool(
            self.config.get("local_uncovered_enabled", True)
        )
        suspicious = []
        cells = [
            cell
            for cell in table.get("cells", [])
            if isinstance(cell, Mapping)
        ]
        table_existing = [
            item
            for related_cell in cells
            for item in related_cell.get("existing", [])
            if isinstance(item, Mapping)
        ]
        table_existing.extend(
            item
            for item in table.get("page_existing", [])
            if isinstance(item, Mapping)
        )
        table_bbox = _valid_bbox(table.get("bbox"))
        numeric_rows = [
            int(cell["row_end"])
            for cell in cells
            if isinstance(cell.get("row_end"), int)
        ]
        terminal_row = max(numeric_rows) if numeric_rows else None
        terminal_cells = [
            cell
            for cell in cells
            if terminal_row is not None and cell.get("row_end") == terminal_row
        ]
        terminal_field_row = bool(
            terminal_cells
            and any(
                _TERMINAL_FIELD_RE.search(str(cell.get("text", "")))
                for cell in terminal_cells
            )
        )
        terminal_bottom_gap = (
            table_bbox[3]
            - max(
                (
                    bbox[3]
                    for cell in terminal_cells
                    for bbox in [_valid_bbox(cell.get("bbox"))]
                    if bbox is not None
                ),
                default=table_bbox[3] if table_bbox is not None else 0.0,
            )
            if table_bbox is not None
            else 0.0
        )
        terminal_content_overflow = any(
            existing_bbox[3] > cell_bbox[3] + 1.0
            for cell in terminal_cells
            for cell_bbox in [_valid_bbox(cell.get("bbox"))]
            if cell_bbox is not None
            for item in cell.get("existing", [])
            if isinstance(item, Mapping)
            for existing_bbox in [_valid_bbox(item.get("bbox"))]
            if existing_bbox is not None
        )
        incomplete_terminal_row = bool(
            self.config.get("terminal_incomplete_row_extension_enabled", True)
            and terminal_content_overflow
            and float(
                self.config.get("terminal_field_min_bottom_extension", 8.0)
            )
            <= terminal_bottom_gap
            <= float(
                self.config.get("terminal_field_max_bottom_extension", 80.0)
            )
        )
        previous_row_bottom = max(
            (
                bbox[3]
                for cell in cells
                if isinstance(cell.get("row_end"), int)
                and terminal_row is not None
                and int(cell["row_end"]) < terminal_row
                for bbox in [_valid_bbox(cell.get("bbox"))]
                if bbox is not None
            ),
            default=None,
        )
        ordered_terminal = sorted(
            (
                (cell, bbox)
                for cell in terminal_cells
                for bbox in [_valid_bbox(cell.get("bbox"))]
                if bbox is not None
            ),
            key=lambda item: item[1][0],
        )
        terminal_right_edges = {
            id(cell): (
                ordered_terminal[index + 1][1][0]
                if index + 1 < len(ordered_terminal)
                else table_bbox[2]
                if table_bbox is not None
                else bbox[2]
            )
            for index, (cell, bbox) in enumerate(ordered_terminal)
        }
        for cell in cells:
            existing = cell.get("existing", [])
            if missing_only and existing and not local_uncovered_enabled:
                self.pixel_cells_skipped += 1
                continue
            analysis_bbox = _valid_bbox(cell.get("bbox"))
            cell_bottom_overflow_extended = False
            if (
                analysis_bbox is not None
                and table_bbox is not None
                and not cell.get("form_region")
                and isinstance(cell.get("row_start"), int)
                and isinstance(cell.get("row_end"), int)
                and int(cell["row_end"]) > int(cell["row_start"])
                and analysis_bbox[3] - analysis_bbox[1]
                <= float(
                    self.config.get(
                        "cell_bottom_overflow_max_cell_height",
                        40.0,
                    )
                )
            ):
                bottom_extension = max(
                    float(
                        self.config.get(
                            "cell_bottom_overflow_extension",
                            12.0,
                        )
                    ),
                    0.0,
                )
                extended_bottom = min(
                    table_bbox[3],
                    analysis_bbox[3] + bottom_extension,
                )
                if extended_bottom > analysis_bbox[3]:
                    analysis_bbox = (
                        analysis_bbox[0],
                        analysis_bbox[1],
                        analysis_bbox[2],
                        extended_bottom,
                    )
                    cell_bottom_overflow_extended = True
            terminal_field_extended = False
            if (
                analysis_bbox is not None
                and table_bbox is not None
                and (terminal_field_row or incomplete_terminal_row)
                and terminal_row is not None
                and cell.get("row_end") == terminal_row
            ):
                extension = table_bbox[3] - analysis_bbox[3]
                minimum_extension = max(
                    float(
                        self.config.get(
                            "terminal_field_min_bottom_extension",
                            8.0,
                        )
                    ),
                    0.0,
                )
                maximum_extension = max(
                    float(
                        self.config.get(
                            "terminal_field_max_bottom_extension",
                            80.0,
                        )
                    ),
                    minimum_extension,
                )
                if minimum_extension <= extension <= maximum_extension:
                    analysis_bbox = (
                        analysis_bbox[0],
                        max(
                            analysis_bbox[1],
                            previous_row_bottom
                            if previous_row_bottom is not None
                            else analysis_bbox[1],
                        ),
                        max(
                            analysis_bbox[2],
                            terminal_right_edges.get(id(cell), analysis_bbox[2]),
                        ),
                        table_bbox[3],
                    )
                    terminal_field_extended = True
            self.pixel_cells_analyzed += 1
            analysis = self._ink_analysis(
                image,
                page_size,
                analysis_bbox or cell.get("bbox", []),
                table_existing,
                diagonal_rules,
                minimum_row_ink_pixels=(
                    int(self.config.get("form_line_min_row_ink_pixels", 6))
                    if cell.get("form_region") or terminal_field_extended
                    else None
                ),
                maximum_line_gap_points=(
                    float(self.config.get("terminal_field_line_gap", 0.25))
                    if terminal_field_extended
                    else None
                ),
            )
            reasons = list(
                reason
                for reason in cell.get("reasons", [])
                if isinstance(reason, str) and reason != "missing_content_bbox"
            )
            if (
                not existing
                and analysis["ink_ratio"] >= min_ink_ratio
                and analysis["ink_ratio"] <= max_ink_ratio
                and analysis["ink_bbox"] is not None
            ):
                reasons.append("missing_content_bbox")
                reasons.append("visible_ink_without_bbox")
            if (
                existing
                and local_uncovered_enabled
                and analysis["ink_ratio"] <= max_ink_ratio
                and analysis["uncovered_ratio"]
                >= (
                    min(
                        min_uncovered_ratio,
                        float(
                            self.config.get(
                                "form_min_uncovered_ink_ratio",
                                0.05,
                            )
                        ),
                    )
                    if cell.get("form_region")
                    else min(
                        min_uncovered_ratio,
                        float(
                            self.config.get(
                                "small_field_min_uncovered_ink_ratio",
                                0.12,
                            )
                        ),
                    )
                    if analysis_bbox is not None
                    and analysis_bbox[3] - analysis_bbox[1] <= 35.0
                    and analysis.get("uncovered_ink_pixels", 0) >= 48
                    else min_uncovered_ratio
                )
                and analysis.get("uncovered_ink_pixels", 0)
                >= minimum_uncovered_pixels
                and analysis["ink_bbox"] is not None
            ):
                reasons.append("uncovered_ink")
            if not reasons:
                continue
            suspicious.append(
                {
                    **dict(cell),
                    "terminal_field_extended": terminal_field_extended,
                    "cell_bottom_overflow_extended": (
                        cell_bottom_overflow_extended
                    ),
                    "reasons": sorted(set(reasons)),
                    "pixel_ink_bbox": analysis["ink_bbox"],
                    "pixel_ink_bboxes": analysis.get("ink_bboxes", []),
                    "pixel_ink_lines": analysis.get("ink_lines", []),
                    "pixel_ink_components": analysis.get("ink_components", []),
                    "ink_ratio": round(float(analysis["ink_ratio"]), 6),
                    "uncovered_ink_ratio": round(
                        float(analysis["uncovered_ratio"]),
                        6,
                    ),
                    "uncovered_ink_pixels": int(
                        analysis.get("uncovered_ink_pixels", 0)
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
        diagonal_rules: Sequence[Sequence[float]] = (),
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
        page_height = float(page_size[1]) if len(page_size) >= 2 else table_bbox[3]
        fringe_enabled = bool(
            self.config.get("table_fringe_recovery_enabled", True)
        )
        fringe_bottom_extension = (
            max(
                float(
                    self.config.get(
                        "table_fringe_bottom_extension",
                        72.0,
                    )
                ),
                0.0,
            )
            if fringe_enabled
            else 0.0
        )
        search_bbox = (
            table_bbox[0],
            table_bbox[1],
            table_bbox[2],
            min(page_height, table_bbox[3] + fringe_bottom_extension),
        )
        crop_box = (
            max(0, int(math.floor(search_bbox[0] * scale_x))),
            max(0, int(math.floor(search_bbox[1] * scale_y))),
            min(image.width, int(math.ceil(search_bbox[2] * scale_x))),
            min(image.height, int(math.ceil(search_bbox[3] * scale_y))),
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
            for box in table.get("page_existing", []):
                if not isinstance(box, Mapping):
                    continue
                bbox = _valid_bbox(box.get("bbox"))
                if bbox is not None:
                    existing_bboxes.append(bbox)
                    mark_bbox(existing_mask, bbox, padding_x, padding_y)
            for raw_bbox in table.get("page_exclusions", []):
                mark_bbox(existing_mask, raw_bbox, padding_x, padding_y)

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
                np.count_nonzero(dark_mask, axis=1)
                >= max(int(crop.width * horizontal_ratio), 1)
            )
            long_columns = np.flatnonzero(
                np.count_nonzero(dark_mask, axis=0)
                >= max(int(crop.height * vertical_ratio), 1)
            )
            for row in long_rows:
                orphan_mask[max(0, row - 1) : min(crop.height, row + 2), :] = False
            for column in long_columns:
                orphan_mask[:, max(0, column - 1) : min(crop.width, column + 2)] = False
            self._mask_diagonal_rules(
                orphan_mask,
                crop_box,
                scale_x,
                scale_y,
                diagonal_rules,
            )

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
            candidate_limit = max(max_items * 8, max_items)
            minimum_height = max(
                float(self.config.get("table_orphan_min_line_height", 3.0)),
                0.0,
            )
            maximum_height_ratio = float(
                self.config.get("table_orphan_max_line_height_ratio", 0.07)
            )
            minimum_width = max(
                float(self.config.get("table_orphan_min_line_width", 8.0)),
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
            horizontal_gap = max(
                int(
                    round(
                        float(
                            self.config.get(
                                "table_orphan_horizontal_gap",
                                12.0,
                            )
                        )
                        * scale_x
                    )
                ),
                1,
            )

            def cell_distance(
                item: tuple[
                    Mapping[str, Any],
                    tuple[float, float, float, float] | None,
                ],
                candidate_bbox: Sequence[float],
            ) -> tuple[float, float]:
                cell, cell_bbox = item
                if cell_bbox is None:
                    return (float("inf"), float("inf"))
                if candidate_bbox[1] >= cell_bbox[3]:
                    vertical = candidate_bbox[1] - cell_bbox[3]
                elif cell_bbox[1] >= candidate_bbox[3]:
                    vertical = cell_bbox[1] - candidate_bbox[3]
                else:
                    vertical = 0.0
                horizontal_overlap = max(
                    0.0,
                    min(candidate_bbox[2], cell_bbox[2])
                    - max(candidate_bbox[0], cell_bbox[0]),
                )
                horizontal = (
                    0.0
                    if horizontal_overlap > 0
                    else min(
                        abs(candidate_bbox[0] - cell_bbox[2]),
                        abs(cell_bbox[0] - candidate_bbox[2]),
                    )
                )
                row = cell.get("row_end")
                return (vertical + horizontal, -float(row or 0))

            for top, bottom in bands:
                band_mask = orphan_mask[top:bottom, :]
                active_columns = np.flatnonzero(
                    np.count_nonzero(band_mask, axis=0) > 0
                )
                if active_columns.size == 0:
                    continue
                column_bands: list[tuple[int, int]] = []
                left = previous_column = int(active_columns[0])
                for raw_column in active_columns[1:]:
                    column = int(raw_column)
                    if column - previous_column > horizontal_gap + 1:
                        column_bands.append((left, previous_column + 1))
                        left = column
                    previous_column = column
                column_bands.append((left, previous_column + 1))
                for left, right in column_bands:
                    segment_mask = band_mask[:, left:right]
                    dark_y, dark_x = np.nonzero(segment_mask)
                    if dark_x.size < minimum_row_ink:
                        continue
                    min_x = left + int(dark_x.min())
                    max_x = left + int(dark_x.max()) + 1
                    min_y = top + int(dark_y.min())
                    max_y = top + int(dark_y.max()) + 1
                    raw_height = (max_y - min_y) / scale_y
                    if raw_height < float(
                        self.config.get("table_orphan_min_dark_height", 2.0)
                    ):
                        continue
                    pixel_area = max(
                        (max_x - min_x) * (max_y - min_y),
                        1,
                    )
                    ink_density = dark_x.size / pixel_area
                    if not (
                        minimum_ink_density
                        <= ink_density
                        <= maximum_ink_density
                    ):
                        continue
                    bbox = _clip_bbox(
                        (
                            (crop_box[0] + min_x) / scale_x - 1.0,
                            (crop_box[1] + min_y) / scale_y - 0.8,
                            (crop_box[0] + max_x) / scale_x + 1.0,
                            (crop_box[1] + max_y) / scale_y + 0.8,
                        ),
                        search_bbox,
                    )
                    if bbox is None:
                        continue
                    height = bbox[3] - bbox[1]
                    maximum_height = max(
                        (table_bbox[3] - table_bbox[1])
                        * maximum_height_ratio,
                        minimum_height * 4,
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

                    nearest_cell, _nearest_bbox = min(
                        cells_with_bbox,
                        key=lambda item: cell_distance(item, bbox),
                    )
                    cell_id = nearest_cell.get("id")
                    table_id = table.get("id")
                    if not isinstance(cell_id, str) or not isinstance(
                        table_id,
                        str,
                    ):
                        continue
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    is_fringe = not (
                        table_bbox[0] <= center_x <= table_bbox[2]
                        and table_bbox[1] <= center_y <= table_bbox[3]
                    )
                    proposals.append(
                        {
                            "action": (
                                "add_fringe" if is_fringe else "add_orphan"
                            ),
                            "table_id": table_id,
                            "cell_id": cell_id,
                            "target_id": "",
                            "bbox": list(bbox),
                            "confidence": float(
                                self.config.get(
                                    "table_orphan_confidence",
                                    0.92,
                                )
                            ),
                            "recovery_reasons": [
                                "table_fringe_ink_without_bbox"
                                if is_fringe
                                else "table_ink_outside_cells"
                            ],
                            "recovery_source": (
                                "local_table_fringe_ink"
                                if is_fringe
                                else "local_table_orphan_ink"
                            ),
                            "_ink_pixels": int(dark_x.size),
                        }
                    )
                    if len(proposals) >= candidate_limit:
                        break
                if len(proposals) >= candidate_limit:
                    break
            preferred_height = max(
                float(
                    self.config.get(
                        "table_orphan_preferred_line_height",
                        6.5,
                    )
                ),
                minimum_height,
            )

            def proposal_priority(item: Mapping[str, Any]) -> tuple[Any, ...]:
                bbox = _valid_bbox(item.get("bbox"))
                if bbox is None:
                    return (False, 0, 0.0, 0.0)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return (
                    height >= preferred_height,
                    int(item.get("_ink_pixels", 0)),
                    height,
                    width,
                )

            proposals = sorted(
                sorted(proposals, key=proposal_priority, reverse=True)[:max_items],
                key=lambda item: (
                    item.get("bbox", [0, 0, 0, 0])[1],
                    item.get("bbox", [0, 0, 0, 0])[0],
                ),
            )
            for proposal in proposals:
                proposal.pop("_ink_pixels", None)
            fringe_count = sum(
                1 for proposal in proposals if proposal["action"] == "add_fringe"
            )
            self.fringe_boxes_proposed += fringe_count
            self.orphan_boxes_proposed += len(proposals) - fringe_count
            return proposals
        finally:
            crop.close()

    def _table_checkbox_proposals(
        self,
        image,
        page_size: Sequence[float],
        table: Mapping[str, Any],
        max_items: int,
    ) -> list[dict[str, Any]]:
        """Detect small square form controls independently of OCR text boxes."""
        if (
            not self.config.get("checkbox_recovery_enabled", True)
            or max_items <= 0
        ):
            return []
        try:
            import cv2
        except ImportError:
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
        self.checkbox_tables_analyzed += 1
        try:
            pixels = np.asarray(crop, dtype=np.uint8)
            _threshold, binary = cv2.threshold(
                pixels,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            contours, _hierarchy = cv2.findContours(
                binary,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            minimum_size = float(self.config.get("checkbox_min_size", 5.5))
            maximum_size = float(self.config.get("checkbox_max_size", 16.0))
            minimum_aspect = float(self.config.get("checkbox_min_aspect", 0.75))
            maximum_aspect = float(self.config.get("checkbox_max_aspect", 1.25))
            minimum_side_density = float(
                self.config.get("checkbox_min_side_density", 0.35)
            )
            maximum_vertices = max(
                int(self.config.get("checkbox_max_vertices", 5)),
                4,
            )
            ambiguous_maximum_vertices = max(
                int(self.config.get("checkbox_ambiguous_max_vertices", 4)),
                4,
            )
            left_clearance = float(
                self.config.get("checkbox_left_clearance", 8.0)
            )
            left_ink_limit = float(
                self.config.get("checkbox_max_left_ink_ratio", 0.08)
            )
            left_edge_allowance = float(
                self.config.get("checkbox_table_edge_allowance", 12.0)
            )
            right_context = float(
                self.config.get("checkbox_right_context", 30.0)
            )
            right_ink_minimum = float(
                self.config.get("checkbox_min_right_ink_ratio", 0.01)
            )
            right_separator_search = float(
                self.config.get("checkbox_right_separator_search", 6.0)
            )
            minimum_right_separator = float(
                self.config.get("checkbox_min_right_separator", 1.5)
            )
            right_column_ink_ratio = float(
                self.config.get("checkbox_right_column_ink_ratio", 0.05)
            )
            unchecked_threshold = float(
                self.config.get("checkbox_unchecked_interior_ratio", 0.03)
            )
            checked_threshold = float(
                self.config.get("checkbox_checked_interior_ratio", 0.12)
            )
            raw_candidates = []
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                width_points = width / scale_x
                height_points = height / scale_y
                if not (
                    minimum_size <= width_points <= maximum_size
                    and minimum_size <= height_points <= maximum_size
                ):
                    continue
                aspect = width_points / height_points if height_points > 0 else 0.0
                if not minimum_aspect <= aspect <= maximum_aspect:
                    continue
                perimeter = cv2.arcLength(contour, True)
                vertices = len(
                    cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                )
                if vertices < 3 or vertices > maximum_vertices:
                    continue
                roi = binary[y : y + height, x : x + width] > 0
                edge = max(1, int(min(width, height) * 0.18))
                side_densities = (
                    float(roi[:edge, :].mean()),
                    float(roi[-edge:, :].mean()),
                    float(roi[:, :edge].mean()),
                    float(roi[:, -edge:].mean()),
                )
                if min(side_densities) < minimum_side_density:
                    continue
                page_left = (crop_box[0] + x) / scale_x
                page_top = (crop_box[1] + y) / scale_y
                page_right = (crop_box[0] + x + width) / scale_x
                page_bottom = (crop_box[1] + y + height) / scale_y
                context_top = max(0, y - height // 2)
                context_bottom = min(crop.height, y + height + height // 2)
                left_end = max(0, x - 1)
                left_start = max(0, x - int(left_clearance * scale_x))
                left_region = binary[
                    context_top:context_bottom,
                    left_start:left_end,
                ]
                near_table_edge = page_left - table_bbox[0] <= left_edge_allowance
                if (
                    not near_table_edge
                    and left_region.size
                    and float((left_region > 0).mean()) > left_ink_limit
                ):
                    continue
                right_start = min(crop.width, x + width + 1)
                right_end = min(
                    crop.width,
                    x + width + int(right_context * scale_x),
                )
                right_region = binary[
                    context_top:context_bottom,
                    right_start:right_end,
                ]
                if (
                    not right_region.size
                    or float((right_region > 0).mean()) < right_ink_minimum
                ):
                    continue
                # A form control is visually separated from its label.  A
                # connected tick may protrude beyond the square, so looking
                # only at the first dark pixel rejects valid checked boxes.
                # Instead, require a short run of fully separating columns
                # near the control.  Capital D/Q glyphs that otherwise look
                # square remain joined to the rest of their word and fail
                # this test.
                aligned_right_end = min(
                    crop.width,
                    x + width + int(math.ceil(right_context * scale_x)),
                )
                aligned_right_region = binary[
                    y : y + height,
                    x + width : aligned_right_end,
                ]
                separator_width = min(
                    aligned_right_region.shape[1],
                    int(math.ceil(right_separator_search * scale_x)),
                )
                separator_region = aligned_right_region[:, :separator_width]
                if not separator_region.size:
                    continue
                aligned_ink_columns = (
                    (aligned_right_region > 0).mean(axis=0)
                    > right_column_ink_ratio
                )
                ink_columns = aligned_ink_columns[:separator_width]
                minimum_separator_pixels = max(
                    1,
                    int(math.ceil(minimum_right_separator * scale_x)),
                )
                has_separator_before_label = False
                separator_start = None
                for column_index, has_ink in enumerate(ink_columns):
                    if not has_ink and separator_start is None:
                        separator_start = column_index
                    elif has_ink and separator_start is not None:
                        if (
                            column_index - separator_start
                            >= minimum_separator_pixels
                            and aligned_ink_columns[column_index:].any()
                        ):
                            has_separator_before_label = True
                            break
                        separator_start = None
                if separator_start is not None:
                    separator_end = len(ink_columns)
                    if (
                        separator_end - separator_start
                        >= minimum_separator_pixels
                        and aligned_ink_columns[separator_end:].any()
                    ):
                        has_separator_before_label = True
                if not has_separator_before_label:
                    continue
                interior_edge = max(1, int(min(width, height) * 0.28))
                interior = roi[
                    interior_edge : height - interior_edge,
                    interior_edge : width - interior_edge,
                ]
                interior_density = (
                    float(interior.mean()) if interior.size else float(roi.mean())
                )
                if interior_density <= unchecked_threshold:
                    state = "unchecked"
                    text = "☐"
                elif interior_density >= checked_threshold:
                    state = "checked"
                    text = "☑"
                else:
                    state = "ambiguous"
                    text = ""
                if (
                    state == "ambiguous"
                    and vertices > ambiguous_maximum_vertices
                ):
                    continue
                bbox = _clip_bbox(
                    (page_left, page_top, page_right, page_bottom),
                    table_bbox,
                )
                if bbox is None:
                    continue
                raw_candidates.append(
                    {
                        "bbox": bbox,
                        "state": state,
                        "text": text,
                        "interior_density": interior_density,
                    }
                )

            raw_candidates.sort(
                key=lambda item: -(
                    (item["bbox"][2] - item["bbox"][0])
                    * (item["bbox"][3] - item["bbox"][1])
                )
            )
            candidates = []
            for candidate in raw_candidates:
                if any(
                    _bbox_iou(candidate["bbox"], existing["bbox"]) >= 0.5
                    for existing in candidates
                ):
                    continue
                candidates.append(candidate)
            self.checkbox_candidates += len(candidates)

            existing_bboxes = [
                valid
                for cell in cells
                for box in cell.get("existing", [])
                if isinstance(box, Mapping)
                for valid in [_valid_bbox(box.get("bbox"))]
                if valid is not None
            ]
            merge_with_label = bool(
                self.config.get("checkbox_merge_label_enabled", True)
            ) and not bool(table.get("form_region"))
            label_max_gap = max(
                float(self.config.get("checkbox_label_max_gap", 24.0)),
                0.0,
            )
            label_min_vertical_overlap = min(
                max(
                    float(
                        self.config.get(
                            "checkbox_label_min_vertical_overlap",
                            0.25,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )
            proposals = []
            cells_with_bbox = [
                (cell, _valid_bbox(cell.get("bbox"))) for cell in cells
            ]
            tight_scale = float(
                self.config.get("checkbox_existing_tight_scale", 2.0)
            )
            accept_existing_label_bbox = bool(
                self.config.get("checkbox_accept_existing_label_bbox", True)
            ) and not bool(table.get("form_region"))
            for candidate in sorted(
                candidates,
                key=lambda item: (item["bbox"][1], item["bbox"][0]),
            ):
                bbox = candidate["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                already_boxed = any(
                    _bbox_overlap_ratio(bbox, existing) >= 0.7
                    and (
                        accept_existing_label_bbox
                        or (
                            existing[2] - existing[0] <= width * tight_scale
                            and existing[3] - existing[1] <= height * tight_scale
                        )
                    )
                    for existing in existing_bboxes
                )
                if already_boxed:
                    continue
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                def cell_distance(
                    item: tuple[Mapping[str, Any], tuple[float, float, float, float] | None]
                ) -> tuple[int, float, float]:
                    cell, cell_bbox = item
                    if cell_bbox is None:
                        return (1, float("inf"), float("inf"))
                    contains = (
                        cell_bbox[0] <= center_x <= cell_bbox[2]
                        and cell_bbox[1] <= center_y <= cell_bbox[3]
                    )
                    distance_x = max(
                        cell_bbox[0] - center_x,
                        0.0,
                        center_x - cell_bbox[2],
                    )
                    distance_y = max(
                        cell_bbox[1] - center_y,
                        0.0,
                        center_y - cell_bbox[3],
                    )
                    area = (cell_bbox[2] - cell_bbox[0]) * (
                        cell_bbox[3] - cell_bbox[1]
                    )
                    return (0 if contains else 1, distance_x + distance_y, area)

                nearest_cell, _nearest_bbox = min(
                    cells_with_bbox,
                    key=cell_distance,
                )
                cell_id = nearest_cell.get("id")
                table_id = table.get("id")
                if not isinstance(cell_id, str) or not isinstance(table_id, str):
                    continue
                state = str(candidate["state"])
                confidence = float(
                    self.config.get(
                        "checkbox_ambiguous_confidence"
                        if state == "ambiguous"
                        else "checkbox_confidence",
                        0.9 if state == "ambiguous" else 0.95,
                    )
                )
                label_targets = []
                if merge_with_label:
                    for existing in nearest_cell.get("existing", []):
                        if not isinstance(existing, Mapping):
                            continue
                        existing_bbox = _valid_bbox(existing.get("bbox"))
                        target_id = existing.get("id")
                        if existing_bbox is None or not isinstance(target_id, str):
                            continue
                        gap = existing_bbox[0] - bbox[2]
                        vertical_overlap = max(
                            0.0,
                            min(bbox[3], existing_bbox[3])
                            - max(bbox[1], existing_bbox[1]),
                        )
                        minimum_height = min(
                            bbox[3] - bbox[1],
                            existing_bbox[3] - existing_bbox[1],
                        )
                        overlap_ratio = (
                            vertical_overlap / minimum_height
                            if minimum_height > 0
                            else 0.0
                        )
                        if (
                            0.0 <= gap <= label_max_gap
                            and overlap_ratio >= label_min_vertical_overlap
                            and str(existing.get("text", "")).strip()
                        ):
                            label_targets.append(
                                (gap, -overlap_ratio, target_id, existing_bbox)
                            )
                label_target = min(label_targets, default=None)
                action = "merge_checkbox" if label_target is not None else "add_checkbox"
                target_id = label_target[2] if label_target is not None else ""
                proposal_bbox = (
                    (
                        min(bbox[0], label_target[3][0]),
                        min(bbox[1], label_target[3][1]),
                        max(bbox[2], label_target[3][2]),
                        max(bbox[3], label_target[3][3]),
                    )
                    if label_target is not None
                    else bbox
                )
                proposals.append(
                    {
                        "action": action,
                        "table_id": table_id,
                        "cell_id": cell_id,
                        "target_id": target_id,
                        "bbox": list(proposal_bbox),
                        "confidence": confidence,
                        "text": candidate["text"],
                        "checkbox_state": state,
                        "checkbox_interior_density": round(
                            float(candidate["interior_density"]),
                            6,
                        ),
                        "recovery_reasons": ["missing_checkbox_bbox"],
                        "recovery_source": "local_checkbox_detector",
                    }
                )
                if label_target is not None:
                    self.checkbox_labels_merged += 1
                if state == "checked":
                    self.checkbox_checked += 1
                elif state == "unchecked":
                    self.checkbox_unchecked += 1
                else:
                    self.checkbox_ambiguous += 1
                if len(proposals) >= max_items:
                    break
            self.checkbox_boxes_proposed += len(proposals)
            return proposals
        finally:
            crop.close()

    def _table_ink_marker_proposals(
        self,
        image,
        page_size: Sequence[float],
        table: Mapping[str, Any],
        max_items: int,
        diagonal_rules: Sequence[Sequence[float]] = (),
        excluded_target_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Extend a tall handwritten bbox over an unboxed circled/list prefix."""
        if (
            not self.config.get("ink_marker_merge_enabled", True)
            or max_items <= 0
        ):
            return []
        table_bbox = _valid_bbox(table.get("bbox"))
        table_id = table.get("id")
        cells = [
            cell
            for cell in table.get("cells", [])
            if isinstance(cell, Mapping)
        ]
        if table_bbox is None or not isinstance(table_id, str) or not cells:
            return []
        excluded = {
            target_id
            for target_id in excluded_target_ids
            if isinstance(target_id, str)
        }
        all_existing = [
            item
            for cell in cells
            for item in cell.get("existing", [])
            if isinstance(item, Mapping)
        ]
        search_width = max(
            float(self.config.get("ink_marker_search_width", 48.0)),
            0.0,
        )
        maximum_gap = max(
            float(self.config.get("ink_marker_max_gap", 20.0)),
            0.0,
        )
        vertical_padding = max(
            float(self.config.get("ink_marker_vertical_padding", 3.0)),
            0.0,
        )
        minimum_target_height = max(
            float(self.config.get("ink_marker_target_min_height", 14.0)),
            0.0,
        )
        minimum_marker_width = max(
            float(self.config.get("ink_marker_min_width", 7.0)),
            0.0,
        )
        maximum_marker_width = max(
            float(self.config.get("ink_marker_max_width", 36.0)),
            minimum_marker_width,
        )
        minimum_marker_height = max(
            float(self.config.get("ink_marker_min_height", 7.0)),
            0.0,
        )
        maximum_marker_height = max(
            float(self.config.get("ink_marker_max_height", 36.0)),
            minimum_marker_height,
        )
        minimum_vertical_overlap = min(
            max(
                float(
                    self.config.get(
                        "ink_marker_min_vertical_overlap",
                        0.45,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        confidence = float(
            self.config.get("ink_marker_confidence", 0.97)
        )
        proposals = []
        for cell in cells:
            cell_id = cell.get("id")
            if not isinstance(cell_id, str):
                continue
            for target in cell.get("existing", []):
                if not isinstance(target, Mapping):
                    continue
                target_id = target.get("id")
                target_bbox = _valid_bbox(target.get("bbox"))
                target_text = str(target.get("text", "")).strip()
                if (
                    not isinstance(target_id, str)
                    or target_id in excluded
                    or target_bbox is None
                    or target.get("source") != "content_span"
                    or target.get("checkbox")
                    or target.get("orphan")
                    or target.get("grouped_list_marker")
                    or target.get("grouped_list_item")
                    or target_bbox[3] - target_bbox[1] < minimum_target_height
                    or not any(character.isalnum() for character in target_text)
                ):
                    continue
                search_bbox = _clip_bbox(
                    (
                        target_bbox[0] - search_width,
                        target_bbox[1] - vertical_padding,
                        target_bbox[0],
                        target_bbox[3] + vertical_padding,
                    ),
                    table_bbox,
                )
                if search_bbox is None:
                    continue
                analysis = self._ink_analysis(
                    image,
                    page_size,
                    search_bbox,
                    [
                        item
                        for item in all_existing
                        if item.get("id") != target_id
                    ],
                    diagonal_rules,
                )
                raw_candidates = analysis.get("ink_bboxes")
                if not isinstance(raw_candidates, list) or not raw_candidates:
                    raw_candidates = [analysis.get("ink_bbox")]
                candidates = []
                for raw_bbox in raw_candidates:
                    marker_bbox = _valid_bbox(raw_bbox)
                    if marker_bbox is None:
                        continue
                    marker_width = marker_bbox[2] - marker_bbox[0]
                    marker_height = marker_bbox[3] - marker_bbox[1]
                    gap = target_bbox[0] - marker_bbox[2]
                    vertical_overlap = max(
                        0.0,
                        min(marker_bbox[3], target_bbox[3])
                        - max(marker_bbox[1], target_bbox[1]),
                    )
                    minimum_height = min(
                        marker_height,
                        target_bbox[3] - target_bbox[1],
                    )
                    overlap_ratio = (
                        vertical_overlap / minimum_height
                        if minimum_height > 0
                        else 0.0
                    )
                    if not (
                        minimum_marker_width
                        <= marker_width
                        <= maximum_marker_width
                        and minimum_marker_height
                        <= marker_height
                        <= maximum_marker_height
                        and 0.0 <= gap <= maximum_gap
                        and overlap_ratio >= minimum_vertical_overlap
                    ):
                        continue
                    center_delta = abs(
                        (marker_bbox[1] + marker_bbox[3]) / 2
                        - (target_bbox[1] + target_bbox[3]) / 2
                    )
                    candidates.append(
                        (
                            gap,
                            -overlap_ratio,
                            center_delta,
                            marker_bbox,
                        )
                    )
                selected = min(candidates, default=None)
                if selected is None:
                    continue
                marker_bbox = selected[3]
                proposals.append(
                    {
                        "action": "merge_ink_marker",
                        "table_id": table_id,
                        "cell_id": cell_id,
                        "target_id": target_id,
                        "bbox": [
                            min(marker_bbox[0], target_bbox[0]),
                            min(marker_bbox[1], target_bbox[1]),
                            max(marker_bbox[2], target_bbox[2]),
                            max(marker_bbox[3], target_bbox[3]),
                        ],
                        "ink_marker_bbox": list(marker_bbox),
                        "confidence": confidence,
                        "recovery_reasons": ["unboxed_ink_marker_prefix"],
                        "recovery_source": "local_ink_marker_merge",
                    }
                )
                excluded.add(target_id)
                self.ink_marker_labels_merged += 1
                if len(proposals) >= max_items:
                    return proposals
        return proposals

    def _table_list_marker_proposals(
        self,
        table: Mapping[str, Any],
        max_items: int,
    ) -> list[dict[str, Any]]:
        """Merge a standalone numbered-list marker with its same-line label."""
        if (
            not self.config.get("list_marker_merge_enabled", True)
            or max_items <= 0
        ):
            return []
        table_id = table.get("id")
        if not isinstance(table_id, str):
            return []
        maximum_gap = max(
            float(self.config.get("list_marker_max_gap", 24.0)),
            0.0,
        )
        minimum_vertical_overlap = min(
            max(
                float(
                    self.config.get(
                        "list_marker_min_vertical_overlap",
                        0.5,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        confidence = float(
            self.config.get("list_marker_confidence", 0.98)
        )
        proposals = []
        used_targets: set[str] = set()
        for cell in table.get("cells", []):
            if not isinstance(cell, Mapping):
                continue
            cell_id = cell.get("id")
            if not isinstance(cell_id, str):
                continue
            records = []
            for item in cell.get("existing", []):
                if not isinstance(item, Mapping):
                    continue
                item_id = item.get("id")
                bbox = _valid_bbox(item.get("bbox"))
                text = str(item.get("text", "")).strip()
                if (
                    not isinstance(item_id, str)
                    or bbox is None
                    or not text
                    or item.get("checkbox")
                    or item.get("orphan")
                    or item.get("grouped_list_marker")
                    or item.get("grouped_list_item")
                ):
                    continue
                records.append((item_id, bbox, text))
            for marker_id, marker_bbox, marker_text in records:
                if _LIST_MARKER_RE.fullmatch(marker_text) is None:
                    continue
                candidates = []
                for target_id, target_bbox, target_text in records:
                    if (
                        target_id == marker_id
                        or target_id in used_targets
                        or _LIST_MARKER_RE.fullmatch(target_text) is not None
                        or not any(character.isalpha() for character in target_text)
                    ):
                        continue
                    gap = target_bbox[0] - marker_bbox[2]
                    vertical_overlap = max(
                        0.0,
                        min(marker_bbox[3], target_bbox[3])
                        - max(marker_bbox[1], target_bbox[1]),
                    )
                    minimum_height = min(
                        marker_bbox[3] - marker_bbox[1],
                        target_bbox[3] - target_bbox[1],
                    )
                    overlap_ratio = (
                        vertical_overlap / minimum_height
                        if minimum_height > 0
                        else 0.0
                    )
                    if not (
                        0.0 <= gap <= maximum_gap
                        and overlap_ratio >= minimum_vertical_overlap
                    ):
                        continue
                    union_bbox = (
                        min(marker_bbox[0], target_bbox[0]),
                        min(marker_bbox[1], target_bbox[1]),
                        max(marker_bbox[2], target_bbox[2]),
                        max(marker_bbox[3], target_bbox[3]),
                    )
                    already_combined = any(
                        other_id not in {marker_id, target_id}
                        and _bbox_overlap_ratio(union_bbox, other_bbox) >= 0.9
                        for other_id, other_bbox, _other_text in records
                    )
                    if already_combined:
                        continue
                    center_delta = abs(
                        (marker_bbox[1] + marker_bbox[3]) / 2
                        - (target_bbox[1] + target_bbox[3]) / 2
                    )
                    candidates.append(
                        (
                            gap,
                            -overlap_ratio,
                            center_delta,
                            target_id,
                            target_bbox,
                            target_text,
                            union_bbox,
                        )
                    )
                target = min(candidates, default=None)
                if target is None:
                    continue
                target_id = target[3]
                target_text = target[5]
                union_bbox = target[6]
                proposals.append(
                    {
                        "action": "merge_list_marker",
                        "table_id": table_id,
                        "cell_id": cell_id,
                        "target_id": target_id,
                        "marker_id": marker_id,
                        "bbox": list(union_bbox),
                        "confidence": confidence,
                        "text": f"{marker_text} {target_text}",
                        "list_marker_text": marker_text,
                        "recovery_reasons": ["split_list_marker_bbox"],
                        "recovery_source": "local_list_marker_merge",
                    }
                )
                used_targets.add(target_id)
                self.list_marker_labels_merged += 1
                if len(proposals) >= max_items:
                    return proposals
        return proposals

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
        candidate_limit = max(max_items * 3, max_items)
        for cell in suspicious:
            reasons = cell.get("reasons", [])
            if not isinstance(reasons, list) or not {
                "missing_content_bbox",
                "uncovered_ink",
            }.intersection(reasons):
                continue
            cell_id = cell.get("id")
            if not isinstance(cell_id, str):
                continue
            is_form_region = bool(cell.get("form_region"))
            if is_form_region and not cell.get("form_recover_text"):
                continue
            raw_ink_lines = (
                cell.get("pixel_ink_components")
                if is_form_region
                else cell.get("pixel_ink_lines")
            )
            has_rich_line_analysis = isinstance(raw_ink_lines, list)
            line_candidates = [
                (bbox, raw_line)
                for raw_line in raw_ink_lines
                if isinstance(raw_line, Mapping)
                for bbox in [_valid_bbox(raw_line.get("bbox"))]
                if bbox is not None
            ] if has_rich_line_analysis else []
            raw_line_bboxes = cell.get("pixel_ink_bboxes")
            has_line_analysis = has_rich_line_analysis or isinstance(
                raw_line_bboxes,
                list,
            )
            if not has_rich_line_analysis and isinstance(raw_line_bboxes, list):
                line_candidates = [
                    (bbox, {})
                    for raw_bbox in raw_line_bboxes
                    for bbox in [_valid_bbox(raw_bbox)]
                    if bbox is not None
                ]
            if not has_line_analysis:
                fallback_bbox = _valid_bbox(cell.get("pixel_ink_bbox"))
                if fallback_bbox is not None:
                    line_candidates = [(fallback_bbox, {})]
            for bbox, line_analysis in line_candidates:
                cell_bbox = _valid_bbox(cell.get("bbox"))
                cell_bottom_overflow = bool(
                    cell.get("cell_bottom_overflow_extended")
                    and cell_bbox is not None
                    and bbox[3] > cell_bbox[3] + 0.5
                )
                if is_form_region:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    density = line_analysis.get("ink_density")
                    maximum_density = float(
                        self.config.get("form_recovery_max_ink_density", 0.72)
                    )
                    maximum_height = float(
                        self.config.get("form_recovery_max_line_height", 24.0)
                    )
                    maximum_width_ratio = float(
                        self.config.get(
                            "demoted_form_recovery_max_width_ratio"
                            if cell.get("demoted_form_cell")
                            else "form_recovery_max_width_ratio",
                            0.8 if cell.get("demoted_form_cell") else 0.5,
                        )
                    )
                    looks_like_checkbox = (
                        4.0 <= width <= 20.0
                        and 4.0 <= height <= 20.0
                        and 0.65 <= width / height <= 1.4
                    )
                    if (
                        isinstance(density, (int, float))
                        and float(density) > maximum_density
                    ) or (
                        isinstance(density, (int, float))
                        and float(density) < 0.02
                    ) or (
                        cell_bbox is not None
                        and width
                        > (cell_bbox[2] - cell_bbox[0]) * maximum_width_ratio
                    ) or (
                        width < 8.0 or height < 6.0
                    ) or (
                        cell_bbox is not None
                        and cell.get("form_recover_text")
                        and not cell.get("demoted_form_cell")
                        and not cell.get("form_recover_full_cell")
                        and (bbox[1] + bbox[3]) / 2
                        < (cell_bbox[1] + cell_bbox[3]) / 2
                    ) or height > maximum_height or looks_like_checkbox:
                        continue
                elif (
                    cell.get("terminal_field_extended")
                    and bbox[2] - bbox[0] < 10.0
                ):
                    continue
                terminal_field_extended = bool(
                    cell.get("terminal_field_extended")
                )
                terminal_field_kind = (
                    _terminal_field_kind(str(cell.get("text", "")))
                    if terminal_field_extended
                    else None
                )
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
                        "recovery_source": (
                            "local_terminal_field_ink"
                            if terminal_field_extended
                            else "local_uncovered_pixel_ink"
                            if "uncovered_ink" in reasons
                            else "local_pixel_ink"
                        ),
                        "terminal_field_extension": terminal_field_extended,
                        "terminal_field_kind": terminal_field_kind,
                        "cell_bottom_overflow": cell_bottom_overflow,
                    }
                )
                if len(proposals) >= candidate_limit:
                    break
            if len(proposals) >= candidate_limit:
                break
        return self._merge_split_local_proposals(proposals)[:max_items]

    def _merge_split_local_proposals(
        self,
        proposals: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge one handwritten line detected by overlapping adjacent Cells."""
        if not self.config.get("merge_split_content_boxes_enabled", True):
            return [dict(item) for item in proposals]
        minimum_horizontal_overlap = min(
            max(
                float(
                    self.config.get(
                        "split_content_min_horizontal_overlap",
                        0.8,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        minimum_vertical_overlap = min(
            max(
                float(
                    self.config.get(
                        "split_content_min_vertical_overlap",
                        0.35,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        maximum_union_height_ratio = max(
            float(
                self.config.get(
                    "split_content_max_union_height_ratio",
                    1.7,
                )
            ),
            1.0,
        )
        merged: list[dict[str, Any]] = []
        for raw_item in sorted(
            proposals,
            key=lambda item: tuple(item.get("bbox", (0, 0, 0, 0))),
        ):
            item = dict(raw_item)
            bbox = _valid_bbox(item.get("bbox"))
            cell_id = item.get("cell_id")
            if bbox is None or not isinstance(cell_id, str):
                merged.append(item)
                continue
            match = None
            for index, existing in enumerate(merged):
                existing_bbox = _valid_bbox(existing.get("bbox"))
                existing_cells = existing.get("merged_cell_ids", [])
                existing_cell_id = existing.get("cell_id")
                cell_ids = {
                    value
                    for value in (
                        list(existing_cells)
                        if isinstance(existing_cells, list)
                        else []
                    )
                    if isinstance(value, str)
                }
                if isinstance(existing_cell_id, str):
                    cell_ids.add(existing_cell_id)
                if existing_bbox is None or cell_id in cell_ids:
                    continue
                left_width = bbox[2] - bbox[0]
                right_width = existing_bbox[2] - existing_bbox[0]
                horizontal_overlap = max(
                    0.0,
                    min(bbox[2], existing_bbox[2])
                    - max(bbox[0], existing_bbox[0]),
                )
                horizontal_ratio = horizontal_overlap / min(
                    left_width,
                    right_width,
                )
                left_height = bbox[3] - bbox[1]
                right_height = existing_bbox[3] - existing_bbox[1]
                vertical_overlap = max(
                    0.0,
                    min(bbox[3], existing_bbox[3])
                    - max(bbox[1], existing_bbox[1]),
                )
                vertical_ratio = vertical_overlap / min(
                    left_height,
                    right_height,
                )
                union_height = max(bbox[3], existing_bbox[3]) - min(
                    bbox[1],
                    existing_bbox[1],
                )
                if (
                    horizontal_ratio >= minimum_horizontal_overlap
                    and vertical_ratio >= minimum_vertical_overlap
                    and union_height
                    <= max(left_height, right_height) * maximum_union_height_ratio
                ):
                    match = index
                    break
            if match is None:
                merged.append(item)
                continue
            existing = merged[match]
            existing_bbox = _valid_bbox(existing.get("bbox"))
            existing_cells = existing.get("merged_cell_ids", [])
            merged_cells = {
                value
                for value in (
                    list(existing_cells)
                    if isinstance(existing_cells, list)
                    else []
                )
                if isinstance(value, str)
            }
            if isinstance(existing.get("cell_id"), str):
                merged_cells.add(existing["cell_id"])
            merged_cells.add(cell_id)
            existing["bbox"] = [
                min(bbox[0], existing_bbox[0]),
                min(bbox[1], existing_bbox[1]),
                max(bbox[2], existing_bbox[2]),
                max(bbox[3], existing_bbox[3]),
            ]
            existing["merged_cell_ids"] = sorted(merged_cells)
            existing["spanning_cells"] = True
            # Spanning proposals already use the Table bbox as their bounded
            # outer geometry. Do not retain a single Cell's overflow flag,
            # which would incorrectly route the merged line through the much
            # narrower short-Cell overflow guard.
            existing["cell_bottom_overflow"] = False
            existing["confidence"] = min(
                float(existing.get("confidence", 0.0)),
                float(item.get("confidence", 0.0)),
            )
            existing["recovery_source"] = "local_split_pixel_ink_merge"
            reasons = {
                reason
                for candidate in (existing, item)
                for reason in candidate.get("recovery_reasons", [])
                if isinstance(reason, str)
            }
            reasons.add("split_content_bbox")
            existing["recovery_reasons"] = sorted(reasons)
        return merged

    def __call__(
        self,
        page_index: int,
        page_size: Sequence[float],
        tables: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        analyzed_before = self.pixel_cells_analyzed
        skipped_before = self.pixel_cells_skipped
        diagonal_rules_before = self.diagonal_rules_removed
        orphan_tables_before = self.orphan_tables_analyzed
        orphan_boxes_before = self.orphan_boxes_proposed
        fringe_boxes_before = self.fringe_boxes_proposed
        checkbox_tables_before = self.checkbox_tables_analyzed
        checkbox_candidates_before = self.checkbox_candidates
        checkbox_boxes_before = self.checkbox_boxes_proposed
        checkbox_merged_before = self.checkbox_labels_merged
        list_marker_merged_before = self.list_marker_labels_merged
        ink_marker_merged_before = self.ink_marker_labels_merged
        checkbox_checked_before = self.checkbox_checked
        checkbox_unchecked_before = self.checkbox_unchecked
        checkbox_ambiguous_before = self.checkbox_ambiguous
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
            is_form_region = bool(table.get("form_region"))
            if mineru_local_only:
                local_only_tables += 1
            pixel_started = time.monotonic()
            try:
                diagonal_rules = self._table_diagonal_rules(
                    image,
                    page_size,
                    table.get("bbox", []),
                )
                self.diagonal_rules_removed += len(diagonal_rules)
                suspicious = self._suspicious_cells(
                    image,
                    page_size,
                    table,
                    missing_only=mineru_local_only,
                    diagonal_rules=diagonal_rules,
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
            list_marker_limit = min(
                remaining_proposals,
                per_table_limit,
                max(
                    int(
                        self.config.get(
                            "list_marker_max_merges_per_table",
                            40,
                        )
                    ),
                    0,
                ),
            )
            list_marker_items = (
                []
                if is_form_region or table.get("disable_marker_merges")
                else self._table_list_marker_proposals(table, list_marker_limit)
            )
            ink_marker_limit = min(
                max(remaining_proposals - len(list_marker_items), 0),
                max(per_table_limit - len(list_marker_items), 0),
                max(
                    int(
                        self.config.get(
                            "ink_marker_max_merges_per_table",
                            20,
                        )
                    ),
                    0,
                ),
            )
            pixel_started = time.monotonic()
            try:
                ink_marker_items = (
                    []
                    if is_form_region or table.get("disable_marker_merges")
                    else self._table_ink_marker_proposals(
                        image,
                        page_size,
                        table,
                        ink_marker_limit,
                        diagonal_rules,
                        [
                            str(item.get("target_id"))
                            for item in list_marker_items
                            if isinstance(item.get("target_id"), str)
                        ],
                    )
                )
            finally:
                pixel_analysis_seconds += time.monotonic() - pixel_started
            missing_items = self._local_missing_proposals(
                suspicious,
                min(
                    max(
                        remaining_proposals
                        - len(list_marker_items)
                        - len(ink_marker_items),
                        0,
                    ),
                    max(
                        per_table_limit
                        - len(list_marker_items)
                        - len(ink_marker_items),
                        0,
                    ),
                ),
            )
            checkbox_limit = min(
                max(
                    remaining_proposals
                    - len(list_marker_items)
                    - len(ink_marker_items)
                    - len(missing_items),
                    0,
                ),
                max(
                    per_table_limit
                    - len(list_marker_items)
                    - len(ink_marker_items)
                    - len(missing_items),
                    0,
                ),
                max(
                    int(
                        self.config.get(
                            "checkbox_max_boxes_per_table",
                            40,
                        )
                    ),
                    0,
                ),
            )
            pixel_started = time.monotonic()
            try:
                checkbox_items = self._table_checkbox_proposals(
                    image,
                    page_size,
                    table,
                    checkbox_limit,
                )
            finally:
                pixel_analysis_seconds += time.monotonic() - pixel_started
            orphan_limit = min(
                max(
                    remaining_proposals
                    - len(list_marker_items)
                    - len(ink_marker_items)
                    - len(missing_items)
                    - len(checkbox_items),
                    0,
                ),
                max(
                    per_table_limit
                    - len(list_marker_items)
                    - len(ink_marker_items)
                    - len(missing_items)
                    - len(checkbox_items),
                    0,
                ),
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
                orphan_items = (
                    []
                    if is_form_region or table.get("disable_orphan_recovery")
                    else self._table_orphan_proposals(
                        image,
                        page_size,
                        table,
                        orphan_limit,
                        diagonal_rules,
                    )
                )
            finally:
                pixel_analysis_seconds += time.monotonic() - pixel_started
            local_items = (
                list_marker_items
                + ink_marker_items
                + missing_items
                + checkbox_items
                + orphan_items
            )
            fringe_items = [
                item for item in orphan_items if item.get("action") == "add_fringe"
            ]
            table_orphan_items = [
                item for item in orphan_items if item.get("action") == "add_orphan"
            ]
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
                        "orphan_proposals": len(table_orphan_items),
                        "fringe_proposals": len(fringe_items),
                        "checkbox_proposals": len(checkbox_items),
                        "list_marker_merge_proposals": len(
                            list_marker_items
                        ),
                        "ink_marker_merge_proposals": len(
                            ink_marker_items
                        ),
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
            "fringe_proposals": self.fringe_boxes_proposed - fringe_boxes_before,
            "checkbox_tables_analyzed": (
                self.checkbox_tables_analyzed - checkbox_tables_before
            ),
            "checkbox_candidates": self.checkbox_candidates - checkbox_candidates_before,
            "checkbox_proposals": self.checkbox_boxes_proposed - checkbox_boxes_before,
            "checkbox_merged": self.checkbox_labels_merged - checkbox_merged_before,
            "list_marker_merged": (
                self.list_marker_labels_merged - list_marker_merged_before
            ),
            "ink_marker_merged": (
                self.ink_marker_labels_merged - ink_marker_merged_before
            ),
            "checkbox_checked": self.checkbox_checked - checkbox_checked_before,
            "checkbox_unchecked": self.checkbox_unchecked - checkbox_unchecked_before,
            "checkbox_ambiguous": self.checkbox_ambiguous - checkbox_ambiguous_before,
            "pixel_cells_analyzed": (
                self.pixel_cells_analyzed - analyzed_before
            ),
            "pixel_cells_skipped": self.pixel_cells_skipped - skipped_before,
            "diagonal_rules_removed": (
                self.diagonal_rules_removed - diagonal_rules_before
            ),
            "pixel_analysis_ms": round(
                pixel_analysis_seconds * 1000,
                3,
            ),
        }

    def close(self) -> None:
        if self._owns_provider:
            self.provider.close()

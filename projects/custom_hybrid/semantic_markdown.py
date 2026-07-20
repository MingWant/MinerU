"""Deterministic semantic Markdown preview built from fused middle JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC_MARKDOWN_VERSION = 1

SECTION_RE = re.compile(
    r"^(?:PART\s+(?:[IVXLC]+|\d+|[A-Z])\b|POINTS? TO NOTE\b|IMPORTANT NOTES?\b|"
    r"CONSULTANT(?:'S)? INFORMATION\b|INSURED(?:'S)? INFORMATION\b|"
    r"PAYMENT INSTRUCTION\b|DETAILS? OF\b|STATEMENT OF ACCOUNT\b|"
    r"HOSPITAL BILL\b|個人資料|注意事項|顧問資料|受保人資料|支付方式)",
    flags=re.IGNORECASE,
)
FIELD_LABEL_RE = re.compile(
    r"(?:\b(?:policy|claim|patient|hospital|account|invoice|bill|room|"
    r"name|date|age|sex|gender|address|phone|telephone|mobile|email|"
    r"code|occupation|diagnosis|result|amount|balance|doctor|physician)"
    r"\b|保單|索償|病人|醫院|帳戶|賬單|房號|姓名|日期|年齡|性別|"
    r"地址|電話|電郵|編號|職業|診斷|結果|金額|結餘|醫生)",
    flags=re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(r"^(?:\d{1,3}[.)、:：]|\(\d{1,3}\))\s*")
ISOLATED_LIST_MARKER_RE = re.compile(
    r"^(?:\d{1,3}[.)、:：]?|\([a-z0-9]{1,3}\)|[a-z][.)])$",
    flags=re.IGNORECASE,
)
FOOTER_RE = re.compile(
    r"(?:\bpage\s*\d+\s*(?:of|/)\s*\d+\b|"
    r"\bP\.\s*\d+\s*/\s*\d+\b|"
    r"^[-–—]?\s*\d{2}/\d{4}\s+SL\b)",
    flags=re.IGNORECASE,
)
NOISE_TEXT_RE = re.compile(
    r"^(?:PAID|HKSH|E\.?\s*&\s*O\.?\s*E\.?|"
    r"[=_\-–—·.]{3,}|\d*[引二州推指]{2,}\d*)$",
    flags=re.IGNORECASE,
)
DATE_VALUE_RE = re.compile(
    r"^\s*\d{1,2}(?:[/.-]\d{1,2}[/.-]\d{2,4}|-[A-Za-z]{3}-\d{2,4})"
    r"(?:\s*[: ]?\s*\d{1,2}:\d{2})?\s*$",
    flags=re.IGNORECASE,
)
DATE_TOKEN_RE = re.compile(
    r"\d{1,2}(?:[/.-]\d{1,2}[/.-]\d{2,4}|-[A-Za-z]{3}-\d{2,4})"
    r"(?::?\d{1,2}:\d{2})?",
    flags=re.IGNORECASE,
)
TIME_VALUE_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*$")
MONEY_VALUE_RE = re.compile(
    r"^\s*[$(]?[-+]?\s*[0-9OoIl][0-9OoIl, .]*-?"
    r"\s*(?:\([^)]{1,40}\))?\s*\)?$",
    flags=re.IGNORECASE,
)
LEDGER_TAIL_RE = re.compile(
    r"\[DISCHARGED\]|\bDISCOUNT\b|\bDate\s+(?:Admitted|Discharged)\b|"
    r"入院日期|出院日期|附注|附註",
    flags=re.IGNORECASE,
)
CONTACT_FOOTER_RE = re.compile(
    r"(?:\bTel(?:ephone)?\b.*\b(?:Fax|Web|E-?mail)\b|"
    r"電話.*傳真|\bWeb\s*:|\bE-?mail\s*:|^[a-z]{1,3}\.[a-z]{2,8}$)",
    flags=re.IGNORECASE,
)
BOTTOM_ORGANIZATION_RE = re.compile(
    r"Hong Kong Sanator(?:ium|lum)|香港.*醫院有限公司|"
    r"養和.*(?:醫院|醫療).*成員|member of HKSH",
    flags=re.IGNORECASE,
)
TITLE_HINT_RE = re.compile(
    r"\b(?:CLAIM|FORM|BILL|RECEIPT|STATEMENT|INFORMATION|INSTRUCTION|NOTES?)\b|"
    r"申請表|賬單|帳單|收據|資料|指示|注意事項",
    flags=re.IGNORECASE,
)

LEDGER_PATTERNS = {
    "date": re.compile(r"(?:^|\b)DATE\b|日期", re.IGNORECASE),
    "code": re.compile(r"CODE|代號|代码|編號", re.IGNORECASE),
    "particulars": re.compile(
        r"PARTICULARS?|項目|项目|DESCRIPTION",
        re.IGNORECASE,
    ),
    "amount": re.compile(r"AMOUNT|金額|金额", re.IGNORECASE),
    "balance": re.compile(
        r"BALANCE|SUB-?TOTAL|結餘|结余|分項金額|分项金额",
        re.IGNORECASE,
    ),
}
LEDGER_LABELS = {
    "date": "Date / 日期",
    "code": "Code / 代號",
    "particulars": "Particulars / 項目",
    "amount": "Amount / 金額",
    "balance": "Balance / 結餘",
}


@dataclass(frozen=True)
class SemanticLine:
    bbox: tuple[float, float, float, float]
    text: str

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(item) for item in bbox)
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        return None
    return bbox


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = html.unescape(value).replace("\u00a0", " ")
    text = re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:eq|span|div|p)\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)PAID(?!\w)", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*E\.?\s*&\s*O\.?\s*E\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _span_text(span: Mapping[str, Any]) -> str:
    for key in ("text", "content"):
        text = _clean_text(span.get(key))
        if text:
            return text
    return ""


def _normalized(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()


def _iter_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_objects(child)


def _table_span(block: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for item in _iter_objects(block):
        if item.get("type") == "table" and isinstance(
            item.get("table_cells"), list
        ):
            return item
    return None


def _checkbox_text(span: Mapping[str, Any], text: str) -> str:
    if not span.get("fusion_checkbox_grouped"):
        return text
    state = str(span.get("fusion_checkbox_state", "ambiguous"))
    prefix = "☑" if state == "checked" else "☐" if state == "unchecked" else "◫"
    return (
        text
        if text.startswith(("☑", "☐", "◫", "☒", "✓", "✔", "■", "□"))
        else f"{prefix} {text}"
    )


def _lines_from_text(
    bbox: tuple[float, float, float, float],
    text: str,
) -> list[SemanticLine]:
    parts = [line.strip() for line in text.splitlines() if line.strip()]
    if not parts:
        return []
    line_height = (bbox[3] - bbox[1]) / len(parts)
    return [
        SemanticLine(
            (
                bbox[0],
                bbox[1] + index * line_height,
                bbox[2],
                bbox[1] + (index + 1) * line_height,
            ),
            part,
        )
        for index, part in enumerate(parts)
    ]


def _visible_cell_lines(
    cell: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> list[SemanticLine]:
    result = []
    excluded = excluded_keys or set()
    raw_spans = cell.get("content_spans", [])
    if isinstance(raw_spans, list):
        for span in raw_spans:
            if (
                not isinstance(span, Mapping)
                or span.get("fusion_visualization_hidden")
                or span.get("fusion_grouped_list_marker")
                or span.get("type") in {"image", "table"}
                or (
                    span.get("fusion_recovery_fringe")
                    and CONTACT_FOOTER_RE.search(_span_text(span))
                )
            ):
                continue
            bbox = _valid_bbox(span.get("bbox"))
            text = _checkbox_text(span, _span_text(span))
            if bbox is not None and text:
                result.extend(
                    line
                    for line in _lines_from_text(bbox, text)
                    if not FOOTER_RE.search(line.text)
                    and not NOISE_TEXT_RE.fullmatch(line.text)
                    and _normalized(line.text) not in excluded
                )
    if not result:
        bbox = _valid_bbox(cell.get("content_bbox")) or _valid_bbox(cell.get("bbox"))
        text = _clean_text(cell.get("text"))
        if bbox is not None and text:
            result.extend(
                line
                for line in _lines_from_text(bbox, text)
                if not FOOTER_RE.search(line.text)
                and not NOISE_TEXT_RE.fullmatch(line.text)
                and _normalized(line.text) not in excluded
            )
    return _deduplicate_lines(result)


def _deduplicate_lines(lines: Sequence[SemanticLine]) -> list[SemanticLine]:
    result = []
    seen = set()
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        key = (
            _normalized(line.text),
            tuple(round(value, 1) for value in line.bbox),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(line)
    return result


def _vertical_overlap(left: SemanticLine, right: SemanticLine) -> float:
    overlap = max(
        0.0,
        min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]),
    )
    minimum = min(left.height, right.height)
    return overlap / minimum if minimum > 0 else 0.0


def _group_visual_rows(lines: Sequence[SemanticLine]) -> list[list[SemanticLine]]:
    if not lines:
        return []
    heights = [line.height for line in lines if line.height > 0]
    tolerance = max(2.0, (statistics.median(heights) if heights else 8.0) * 0.45)
    groups: list[list[SemanticLine]] = []
    for line in sorted(lines, key=lambda item: (item.center_y, item.bbox[0])):
        best_index = None
        best_distance = float("inf")
        for index, group in enumerate(groups[-4:]):
            actual_index = len(groups) - len(groups[-4:]) + index
            group_center = statistics.mean(item.center_y for item in group)
            distance = abs(line.center_y - group_center)
            if distance <= tolerance:
                if distance < best_distance:
                    best_index = actual_index
                    best_distance = distance
        if best_index is None:
            groups.append([line])
        else:
            groups[best_index].append(line)
    return [sorted(group, key=lambda item: item.bbox[0]) for group in groups]


def _join_row_lines(lines: Sequence[SemanticLine]) -> str:
    parts = []
    for line in sorted(lines, key=lambda item: item.bbox[0]):
        text = line.text.strip()
        if not text or any(
            _normalized(text) == _normalized(existing) for existing in parts
        ):
            continue
        parts.append(text)
    return " ".join(parts)


def _table_lines(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> list[SemanticLine]:
    return _deduplicate_lines(
        [
            line
            for cell in table.get("table_cells", [])
            if isinstance(cell, Mapping)
            for line in _visible_cell_lines(cell, excluded_keys)
        ]
    )


def _ledger_category(text: str) -> str | None:
    matches = [name for name, pattern in LEDGER_PATTERNS.items() if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


def _ledger_header(
    lines: Sequence[SemanticLine],
) -> tuple[list[tuple[str, SemanticLine]], float, float] | None:
    candidates = [
        (category, line)
        for line in lines
        for category in [_ledger_category(line.text)]
        if category is not None and len(line.text) <= 100
    ]
    best: tuple[int, float, list[tuple[str, SemanticLine]]] | None = None
    for _category, seed in candidates:
        cluster = [
            item for item in candidates if abs(item[1].center_y - seed.center_y) <= 24.0
        ]
        by_category: dict[str, SemanticLine] = {}
        for category, line in cluster:
            current = by_category.get(category)
            if current is None or abs(line.center_y - seed.center_y) < abs(
                current.center_y - seed.center_y
            ):
                by_category[category] = line
        score = len(by_category)
        if score < 3:
            continue
        ordered = sorted(by_category.items(), key=lambda item: item[1].center_x)
        candidate = (score, -seed.center_y, ordered)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    ordered = best[2]
    return (
        ordered,
        min(line.bbox[1] for _category, line in ordered),
        max(line.bbox[3] for _category, line in ordered),
    )


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not headers:
        return ""
    output = [
        "| " + " | ".join(_escape_table_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _item in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(_escape_table_cell(item) for item in row) + " |"
        for row in rows
        if any(item.strip() for item in row)
    )
    return "\n".join(output)


def _ledger_column_for_line(
    line: SemanticLine,
    columns: Sequence[tuple[str, SemanticLine]],
) -> int:
    centers = [anchor.center_x for _category, anchor in columns]
    by_category = {category: index for index, (category, _line) in enumerate(columns)}
    text = line.text.strip()
    if DATE_VALUE_RE.fullmatch(text) and "date" in by_category:
        date_index = by_category["date"]
        if abs(line.center_x - centers[date_index]) <= 90:
            return date_index
    money_columns = [
        by_category[category]
        for category in ("amount", "balance")
        if category in by_category
    ]
    money_index = (
        min(money_columns, key=lambda index: abs(line.center_x - centers[index]))
        if money_columns
        else None
    )
    if (
        MONEY_VALUE_RE.fullmatch(text)
        and money_index is not None
        and (
            re.search(r"[.,]", text)
            or text.rstrip().endswith("-")
            or "(" in text
        )
        and abs(line.center_x - centers[money_index]) <= 100
    ):
        return money_index
    code_text = re.sub(r"^[|\\\s]+", "", text)
    if (
        "code" in by_category
        and len(code_text) <= 32
        and re.fullmatch(
            r"(?:[A-Z]?\d[\w.-]*|[A-Z][A-Z0-9.-]{1,9})"
            r"(?:\s+[A-Z]?\d[\w.-]*)*",
            code_text,
        )
    ):
        code_index = by_category["code"]
        if abs(line.center_x - centers[code_index]) <= 70:
            return code_index
    if "particulars" in by_category and any(
        character.isalpha() for character in text
    ):
        return by_category["particulars"]
    return min(range(len(centers)), key=lambda index: abs(line.center_x - centers[index]))


def _ledger_date_values(lines: Sequence[SemanticLine]) -> list[str]:
    result = []
    times = [line for line in lines if TIME_VALUE_RE.fullmatch(line.text)]
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        match = DATE_TOKEN_RE.search(line.text)
        if match is None:
            continue
        value = re.sub(r"(?<=\d{4}):(?=\d{1,2}:\d{2}$)", " ", match.group(0))
        if not re.search(r"\d{1,2}:\d{2}$", value):
            close_time = min(
                times,
                key=lambda item: abs(item.center_y - line.center_y),
                default=None,
            )
            if close_time is not None and abs(close_time.center_y - line.center_y) <= 15:
                value += " " + close_time.text.strip()
        if value not in result:
            result.append(value)
    return result


def _render_ledger_tail(lines: Sequence[SemanticLine]) -> str:
    if not lines:
        return ""
    joined = " ".join(line.text for line in lines)
    has_admitted = bool(re.search(r"Date\s+Admitted|入院日期", joined, re.IGNORECASE))
    has_discharged = bool(
        re.search(r"Date\s+Discharged|出院日期", joined, re.IGNORECASE)
    )
    date_values = _ledger_date_values(lines)
    parts = []
    if any(re.search(r"\[DISCHARGED\]", line.text, re.IGNORECASE) for line in lines):
        parts.append("### Discharged / 出院")
    remaining = [
        line
        for line in lines
        if not re.search(
            r"\[DISCHARGED\]|Date\s+(?:Admitted|Discharged)|入院日期|出院日期",
            line.text,
            flags=re.IGNORECASE,
        )
        and DATE_TOKEN_RE.search(line.text) is None
        and not TIME_VALUE_RE.fullmatch(line.text)
    ]
    for group in _group_visual_rows(remaining):
        text = _join_row_lines(group)
        if text and len(_normalized(text)) >= 2 and not NOISE_TEXT_RE.fullmatch(text):
            parts.append(text)
    if has_admitted:
        value = date_values[0] if date_values else ""
        parts.append("- **Date Admitted / 入院日期**" + (f": {value}" if value else ""))
    if has_discharged:
        value = date_values[1] if len(date_values) >= 2 else ""
        parts.append("- **Date Discharged / 出院日期**" + (f": {value}" if value else ""))
    return "\n\n".join(parts)


def _render_ledger_table(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> str | None:
    lines = _table_lines(table, excluded_keys)
    header = _ledger_header(lines)
    if header is None:
        return None
    columns, header_top, header_bottom = header
    headers = [LEDGER_LABELS[category] for category, _line in columns]
    preheader_lines = [line for line in lines if line.bbox[3] < header_top - 1.0]
    data_lines = [
        line
        for line in lines
        if line.bbox[1] > header_bottom + 0.5
        and not (
            _ledger_category(line.text) is not None
            and line.bbox[1] <= header_bottom + 30.0
        )
        and not NOISE_TEXT_RE.fullmatch(line.text)
    ]
    tail_candidates = [line for line in data_lines if LEDGER_TAIL_RE.search(line.text)]
    tail_lines = []
    if tail_candidates:
        tail_top = min(line.bbox[1] for line in tail_candidates)
        tail_lines = [line for line in data_lines if line.bbox[1] >= tail_top - 1.0]
        data_lines = [line for line in data_lines if line.bbox[1] < tail_top - 1.0]
    rows = []
    for group in _group_visual_rows(data_lines):
        values: list[list[SemanticLine]] = [[] for _column in columns]
        for line in group:
            column = _ledger_column_for_line(line, columns)
            values[column].append(line)
        row = [_join_row_lines(items) for items in values]
        if any(row):
            rows.append(row)
    parts = []
    if preheader_lines:
        preheader = [
            _join_row_lines(group)
            for group in _group_visual_rows(preheader_lines)
        ]
        parts.extend(line for line in preheader if line and not FOOTER_RE.search(line))
    parts.append(_markdown_table(headers, rows))
    tail = _render_ledger_tail(tail_lines)
    if tail:
        parts.append(tail)
    return "\n\n".join(part for part in parts if part)


def _is_section(text: str) -> bool:
    stripped = text.strip()
    if SECTION_RE.search(stripped):
        return True
    letters = [character for character in stripped if character.isalpha()]
    return bool(
        len(letters) >= 5
        and len(stripped) <= 100
        and sum(character.isupper() for character in letters) / len(letters) >= 0.8
    )


def _task_line(text: str) -> str | None:
    if text.startswith(("☑", "☒", "✓", "✔", "■")):
        return "- [x] " + text[1:].strip()
    if text.startswith(("☐", "□")):
        return "- [ ] " + text[1:].strip()
    if text.startswith("◫"):
        return "- [?] " + text[1:].strip()
    return None


def _merge_isolated_list_markers(
    lines: Sequence[SemanticLine],
) -> list[SemanticLine]:
    result = []
    for group in _group_visual_rows(lines):
        ordered = sorted(group, key=lambda item: item.bbox[0])
        marker = ordered[0]
        followers = ordered[1:]
        gap = followers[0].bbox[0] - marker.bbox[2] if followers else float("inf")
        if (
            followers
            and ISOLATED_LIST_MARKER_RE.fullmatch(marker.text.strip())
            and marker.bbox[2] - marker.bbox[0] <= max(20.0, marker.height * 2.5)
            and gap <= max(30.0, marker.height * 4.0)
        ):
            marker_text = marker.text.strip()
            if marker_text.isdigit():
                marker_text += "."
            text = f"{marker_text} {_join_row_lines(followers)}".strip()
            result.append(
                SemanticLine(
                    (
                        min(item.bbox[0] for item in ordered),
                        min(item.bbox[1] for item in ordered),
                        max(item.bbox[2] for item in ordered),
                        max(item.bbox[3] for item in ordered),
                    ),
                    text,
                )
            )
        else:
            result.extend(ordered)
    return sorted(result, key=lambda item: (item.bbox[1], item.bbox[0]))


def _looks_like_field_label(text: str) -> bool:
    match = FIELD_LABEL_RE.search(text)
    return bool(match and match.start() <= 35 and len(text) <= 120)


def _render_form_cell(
    cell: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> list[str]:
    lines = [
        line.text
        for line in _merge_isolated_list_markers(
            _visible_cell_lines(cell, excluded_keys)
        )
    ]
    if not lines:
        return []
    result = []
    first = lines[0]
    if _is_section(first):
        result.append(f"### {first}")
        lines = lines[1:]
    elif (
        len(lines) <= 4
        and _task_line(first) is None
        and _looks_like_field_label(first)
        and not any(_looks_like_field_label(item) for item in lines[1:])
        and not any(_task_line(item) is not None for item in lines[1:])
        and not LIST_ITEM_RE.match(first)
    ):
        value = " / ".join(lines[1:])
        result.append(f"- **{first}**" + (f": {value}" if value else ""))
        return result
    for text in lines:
        if FOOTER_RE.search(text) or NOISE_TEXT_RE.fullmatch(text):
            continue
        task = _task_line(text)
        if task is not None:
            result.append(task)
        elif LIST_ITEM_RE.match(text):
            result.append(text)
        elif _is_section(text):
            result.append(f"### {text}")
        else:
            result.append(text)
    return result


def _render_form_table(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> str:
    rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    fallback_index = 0
    for cell in table.get("table_cells", []):
        if not isinstance(cell, Mapping):
            continue
        row = cell.get("row_start")
        if not isinstance(row, int):
            row = 100000 + fallback_index
            fallback_index += 1
        rows[row].append(cell)
    parts = []
    seen = set()
    for row in sorted(rows):
        cells = sorted(
            rows[row],
            key=lambda cell: (
                int(cell.get("col_start", 0))
                if isinstance(cell.get("col_start"), int)
                else 0,
                (_valid_bbox(cell.get("bbox")) or (0, 0, 0, 0))[0],
            ),
        )
        for cell in cells:
            for item in _render_form_cell(cell, excluded_keys):
                key = _normalized(re.sub(r"^#+\s*", "", item))
                if not key or (key in seen and not _repeatable_value(item)):
                    continue
                if not _repeatable_value(item):
                    seen.add(key)
                parts.append(item)
    return "\n\n".join(parts)


def _repeatable_value(text: str) -> bool:
    normalized = _normalized(text)
    return bool(
        any(character.isdigit() for character in text)
        and len(normalized) <= 32
        and len(text.split()) <= 5
    )


def _block_lines(block: Mapping[str, Any]) -> list[SemanticLine]:
    result = []
    raw_lines = block.get("lines", [])
    if not isinstance(raw_lines, list):
        return result
    for line in raw_lines:
        if not isinstance(line, Mapping):
            continue
        bbox = _valid_bbox(line.get("bbox"))
        parts = []
        for span in line.get("spans", []):
            if (
                not isinstance(span, Mapping)
                or span.get("fusion_visualization_hidden")
                or span.get("type") in {"image", "table"}
            ):
                continue
            text = _span_text(span)
            if text:
                parts.append(text)
        text = " ".join(parts).strip()
        if bbox is not None and text:
            result.append(SemanticLine(bbox, text))
    return result


def _margin_repetitions(pages: Sequence[Mapping[str, Any]]) -> set[str]:
    occurrences: dict[str, set[int]] = defaultdict(set)
    for page_index, page in enumerate(pages):
        page_size = page.get("page_size", [])
        height = (
            float(page_size[1])
            if isinstance(page_size, (list, tuple)) and len(page_size) >= 2
            else 0.0
        )
        if height <= 0:
            continue
        for block in page.get("preproc_blocks", []):
            if not isinstance(block, Mapping):
                continue
            table = _table_span(block) if block.get("type") == "table" else None
            lines = _table_lines(table) if table is not None else _block_lines(block)
            for line in lines:
                if line.bbox[1] <= height * 0.1 or line.bbox[3] >= height * 0.9:
                    key = _normalized(line.text)
                    if len(key) >= 5:
                        occurrences[key].add(page_index)
    return {key for key, page_ids in occurrences.items() if len(page_ids) >= 2}


def generate_semantic_markdown(middle_json: Mapping[str, Any]) -> str:
    """Build a conservative alternative Markdown view from fused geometry."""
    pages = middle_json.get("pdf_info", [])
    if not isinstance(pages, list):
        raise ValueError("middle_json must contain a pdf_info list")
    repeated_margins = _margin_repetitions(pages)
    emitted_margins = set()
    document_parts = [f"<!-- semantic-markdown-v{SEMANTIC_MARKDOWN_VERSION} -->"]
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            continue
        page_size = page.get("page_size", [])
        page_height = (
            float(page_size[1])
            if isinstance(page_size, (list, tuple)) and len(page_size) >= 2
            else 0.0
        )
        segments: list[tuple[float, float, str]] = []
        for block in page.get("preproc_blocks", []):
            if not isinstance(block, Mapping):
                continue
            bbox = _valid_bbox(block.get("bbox")) or (0.0, 0.0, 0.0, 0.0)
            table = _table_span(block) if block.get("type") == "table" else None
            if table is not None:
                rendered = _render_ledger_table(
                    table,
                    repeated_margins,
                ) or _render_form_table(table, repeated_margins)
                if rendered:
                    segments.append((bbox[1], bbox[0], rendered))
                continue
            if block.get("type") in {"image", "chart", "interline_equation"}:
                continue
            for line in _block_lines(block):
                text = line.text
                key = _normalized(text)
                if FOOTER_RE.search(text) or NOISE_TEXT_RE.fullmatch(text):
                    continue
                if (
                    page_height > 0
                    and line.bbox[3] >= page_height * 0.9
                    and (
                        CONTACT_FOOTER_RE.search(text)
                        or BOTTOM_ORGANIZATION_RE.search(text)
                    )
                ):
                    continue
                if key in repeated_margins:
                    if key in emitted_margins:
                        continue
                    emitted_margins.add(key)
                is_short_title = (
                    len(text) <= 80
                    and len(text.split()) <= 12
                    and not re.search(r"https?://|[.!?。！？]$", text, re.IGNORECASE)
                    and TITLE_HINT_RE.search(text)
                    and (
                        bool(re.search(r"[\u3400-\u9fff]", text))
                        or bool(re.match(r"^[^a-z]*[A-Z]", text))
                    )
                )
                prefix = (
                    "## "
                    if block.get("type") == "title"
                    and (_is_section(text) or is_short_title)
                    else ""
                )
                segments.append((line.bbox[1], line.bbox[0], prefix + text))
        if not segments:
            continue
        document_parts.append(f"<!-- Page {page_index + 1} -->")
        seen = set()
        for _top, _left, text in sorted(segments):
            key = _normalized(re.sub(r"^#+\s*", "", text))
            if not key or (key in seen and not _repeatable_value(text)):
                continue
            if not _repeatable_value(text):
                seen.add(key)
            document_parts.append(text)
    return "\n\n".join(document_parts).strip() + "\n"


def write_semantic_markdown(
    middle_json_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    middle_path = Path(middle_json_path)
    payload = json.loads(middle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("middle_json must be a JSON object")
    if output_path is None:
        suffix = "_middle.json"
        stem = middle_path.name[: -len(suffix)] if middle_path.name.endswith(suffix) else middle_path.stem
        output = middle_path.with_name(f"{stem}_semantic.md")
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_semantic_markdown(payload), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("middle_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(write_semantic_markdown(args.middle_json, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

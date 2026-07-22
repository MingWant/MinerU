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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC_MARKDOWN_VERSION = 5

SECTION_RE = re.compile(
    r"^(?:PART\s+(?:[IVXLC]+|\d+|[A-Z])\b|POINTS? TO NOTE\b|IMPORTANT NOTES?\b|"
    r"CONSULTANT(?:'S)? INFORMATION\b|INSURED(?:'S)? INFORMATION\b|"
    r"PAYMENT INSTRUCTION\b|DETAILS? OF\b|STATEMENT OF ACCOUNT\b|"
    r"HOSPITAL BILL\b|DECLARATION AND AUTHORIZATION\b|"
    r"PERSONAL INFORMATION COLLECTION STATEMENT\b|"
    r"(?:APPLICANT|PATIENT|CONTACT|MEMBER|PROVIDER) INFORMATION\b|"
    r"MEDICAL HISTORY\b|PAYMENT DETAILS\b|DECLARATION\b|AUTHORIZATION\b|CONSENT\b|"
    r"個人資料|注意事項|顧問資料|受保人資料|支付方式|聲明及授權)",
    flags=re.IGNORECASE,
)
FIELD_LABEL_RE = re.compile(
    r"(?:\b(?:policy(?:\s*(?:no|number|id))?|"
    r"claim(?:\s*(?:no|number|id|date|reference|amount)|ed\s+benefit)|"
    r"case\s+type|patient\s*(?:no|number|name)|hospital\s*(?:no|number|name)|"
    r"account\s*(?:no|number)|invoice\s*(?:no|number|date)|bill\s*(?:no|number|date)|"
    r"(?:member|subscriber|customer|provider|employee|group)\s*(?:id|no|number|code)|"
    r"(?:reference|certificate|application|order)\s*(?:id|no|number)|"
    r"social\s+security(?:\s*(?:no|number))?|ssn|date\s+of\s+birth|dob|"
    r"postal\s*code|zip\s*code|"
    r"room\s*(?:no|number)|name|date|time|age|sex|gender|address|phone|telephone|"
    r"mobile|email|code|occupation|diagnosis|result|amount|balance|doctor|physician|"
    r"district|branch|id\s*/?\s*passport|i\.?d\.?\s*(?:no|number)|"
    r"test|investigation|qualification)\b|保單|索償|病人|醫院|帳戶|賬單|房號|姓名|日期|年齡|性別|"
    r"地址|電話|電郵|編號|身份[證証]|護照|職業|診斷|結果|金額|結餘|分行|區域)",
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
    r"^\s*(?:\d{1,2}(?:[/.-]\d{1,2}[/.-]\d{2,4}|-[A-Za-z]{3}-\d{2,4})|"
    r"\d{4}-\d{1,2}-\d{1,2})"
    r"(?:\s*[: ]?\s*\d{1,2}:\d{2})?\s*$",
    flags=re.IGNORECASE,
)
DATE_TOKEN_RE = re.compile(
    r"(?:\d{1,2}(?:[/.-]\d{1,2}[/.-]\d{2,4}|-[A-Za-z]{3}-\d{2,4})|"
    r"\d{4}-\d{1,2}-\d{1,2})"
    r"(?:\s*:?\s*\d{1,2}:\d{2})?",
    flags=re.IGNORECASE,
)
TIME_VALUE_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*$")
MONEY_VALUE_RE = re.compile(
    r"^\s*(?:(?:HKD|USD|EUR|GBP|RMB|CNY)\s*)?[$€£¥(]?[-+]?\s*"
    r"[0-9OoIl][0-9OoIl, .]*-?"
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
INSTRUCTION_TEXT_RE = re.compile(
    r"\b(?:if|please|must|shall|will|may|agree|authorize|authorise|"
    r"subject to|according to|required|reserve the right)\b|"
    r"(?:如有|若|必須|同意|授權|根據|要求|保留)",
    flags=re.IGNORECASE,
)
FORM_PROMPT_RE = re.compile(
    r"^(?:\d{1,3}[.)、:]|\([A-Za-z0-9]{1,3}\)|[ivx]{1,4}[.)])\s*",
    flags=re.IGNORECASE,
)
QUESTION_PROMPT_RE = re.compile(
    r"^(?:are|did|do|does|have|has|was|were|is|can|could|will|would)\b|[?？]\s*$",
    flags=re.IGNORECASE,
)
LABEL_QUALIFIER_RE = re.compile(
    r"^\(?\s*(?:with\s+stamp|dd\s*/\s*mm\s*/\s*yy|mm\s*/\s*yy|"
    r"day\s*/\s*month\s*/\s*year)\s*\)?\s*[:：]?\s*$",
    flags=re.IGNORECASE,
)
PARTIAL_DATE_VALUE_RE = re.compile(r"^\s*\d{1,2}\s*[/.-]\s*\d{2,4}\s*$")
IDENTIFIER_LABEL_RE = re.compile(
    r"\b(?:id|passport|member|subscriber|customer|provider|employee|group|"
    r"reference|certificate|application|order)\s*(?:no|number|id|code)?\b|"
    r"social\s+security|\bssn\b|身份[證証]|護照",
    flags=re.IGNORECASE,
)
EMAIL_LABEL_RE = re.compile(r"\bemail\b|電郵", flags=re.IGNORECASE)
POSTAL_LABEL_RE = re.compile(
    r"\b(?:postal|zip)\s*code\b", flags=re.IGNORECASE
)
AMOUNT_LABEL_RE = re.compile(
    r"\b(?:amount|total|charge|price|fee|balance)\b|金額|總計|結餘",
    flags=re.IGNORECASE,
)
OPTION_TEXT_RE = re.compile(
    r"^(?:new|further|mail|via|pay|credit|please|other|china unionpay|"
    r"yes|no|by cheque|others?)\b",
    flags=re.IGNORECASE,
)
POLICY_OWNER_SIGNATURE_RE = re.compile(
    r"Signature\s+of\s+Policy\s+Owner|保單主[權权橙榷]人.*簽署",
    flags=re.IGNORECASE,
)
INSURED_SIGNATURE_RE = re.compile(
    r"Signature\s+of\s+Insured|受保人[簽签]署",
    flags=re.IGNORECASE,
)
BLOCK_NAME_LABEL_RE = re.compile(
    r"Name\s*\(\s*in\s+block\s+letters\s*\)|姓名\s*[（(].*?[大人].*?[）)]",
    flags=re.IGNORECASE,
)
COMPACT_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*/\s*(\d{2})(\d{2})(?!\d)"
)
PIPE_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*\|\s*(\d{1,2})\s*\|\s*(\d{2,4})(?!\d)"
)
REPEATED_GROUPED_AMOUNT_RE = re.compile(
    r"(?<!\d)(\d{1,3})[.,](\d{3})[.,](\d{2})(?!\d)"
)
FORMULA_TEXT_RE = re.compile(
    r"\\(?:begin|end|frac|sqrt|cdot|mathrm|mathsf|ldots|cos|sin|alpha|beta)\b|"
    r"[_^]\s*\{",
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
LEDGER_CURRENCY_UNIT_RE = re.compile(
    r"\(?\s*(?:HKD|HKS|USD|EUR|GBP|RMB|CNY|港元)\s*\)?",
    re.IGNORECASE,
)
MEDICAL_QUALIFICATION_RE = re.compile(
    r"\b(?:MBBS|MBBCh|MD|DO|FRCS(?:Ed)?|FCSHK|FHKAM|MRCP|MRCS|FRCPath|"
    r"FRCR|FCOphth|DCH|DFM|Dip\.?\s*Med)\b",
    re.IGNORECASE,
)
SIGNATURE_VALUE_PREFIX_RE = re.compile(
    r"^(?:(?:signature|signed)\b|[簽签](?:署|名))\s*"
    r"(?:[:：]\s*(?:[xX✓✔✗]\s*)?|[xX✓✔✗]\s+)",
    re.IGNORECASE,
)

MEDICAL_GRID_SCHEMAS = (
    frozenset(("date", "test", "result")),
    frozenset(("date", "investigation_result", "medical_treatment")),
    frozenset(("date", "conditions", "treatment", "recovery")),
)


@dataclass(frozen=True)
class SemanticLine:
    bbox: tuple[float, float, float, float]
    text: str
    cell_row: int | None = None
    cell_col: int | None = None
    cell_bbox: tuple[float, float, float, float] | None = None

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


def _repair_compact_date(match: re.Match[str]) -> str:
    day, month, year = match.groups()
    if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
        return f"{int(day)}/{month}/{year}"
    return match.group(0)


def _repair_pipe_date(match: re.Match[str]) -> str:
    day, month, year = match.groups()
    if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
        return f"{int(day)}/{int(month):02d}/{year}"
    return match.group(0)


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
    text = COMPACT_DATE_RE.sub(_repair_compact_date, text)
    text = PIPE_DATE_RE.sub(_repair_pipe_date, text)
    text = REPEATED_GROUPED_AMOUNT_RE.sub(
        lambda match: f"{match.group(1)},{match.group(2)}.{match.group(3)}",
        text,
    )
    text = re.sub(
        r"(\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})[.;,。]+$",
        r"\1",
        text,
    )
    text = re.sub(r"(?<=\d{4})(?=\d{1,2}:\d{2}\b)", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _span_text(span: Mapping[str, Any]) -> str:
    original_ocr = _clean_text(span.get("fusion_recognition_original_ocr"))
    selected_text = ""
    for key in ("text", "content"):
        text = _clean_text(span.get(key))
        if text:
            selected_text = text
            break
    if original_ocr and selected_text:
        ocr_compact = re.sub(r"\s+", "", original_ocr)
        selected_compact = re.sub(r"\s+", "", selected_text)
        if (
            re.fullmatch(r"[A-Z][A-Z0-9._/-]*", ocr_compact)
            and any(character.isdigit() for character in ocr_compact[1:])
            and re.fullmatch(r"\d[\d._/-]*", selected_compact)
        ):
            return original_ocr
    return selected_text


def _semantic_cell_text(cell: Mapping[str, Any]) -> str:
    text = _clean_text(cell.get("text"))
    raw_spans = cell.get("content_spans", [])
    if not text or not isinstance(raw_spans, list):
        return text
    for span in raw_spans:
        if not isinstance(span, Mapping):
            continue
        raw_text = ""
        for key in ("text", "content"):
            raw_text = _clean_text(span.get(key))
            if raw_text:
                break
        semantic_text = _span_text(span)
        if raw_text and semantic_text and raw_text != semantic_text:
            text = text.replace(raw_text, semantic_text)
    return text


def _normalized(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()


def _looks_like_formula_text(text: str) -> bool:
    return bool(FORMULA_TEXT_RE.search(text))


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


def _table_has_cells(table: Mapping[str, Any] | None) -> bool:
    if table is None:
        return False
    cells = table.get("table_cells", [])
    return bool(
        isinstance(cells, list)
        and any(isinstance(cell, Mapping) for cell in cells)
    )


def _direct_table_spans(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return structured Tables directly owned by a block, not its siblings."""
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    if block.get("type") == "table" and isinstance(block.get("table_cells"), list):
        result.append(block)
        seen.add(id(block))
    raw_lines = block.get("lines", [])
    for line in raw_lines if isinstance(raw_lines, list) else []:
        if not isinstance(line, Mapping):
            continue
        raw_spans = line.get("spans", [])
        for span in raw_spans if isinstance(raw_spans, list) else []:
            if (
                isinstance(span, Mapping)
                and span.get("type") == "table"
                and isinstance(span.get("table_cells"), list)
                and id(span) not in seen
            ):
                result.append(span)
                seen.add(id(span))
    return result


def _layout_blocks(value: Any) -> list[Mapping[str, Any]]:
    """Return layout blocks in tree order without descending into media."""
    result: list[Mapping[str, Any]] = []
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if not isinstance(item, Mapping) or id(item) in visited:
            return
        visited.add(id(item))
        result.append(item)
        if item.get("type") in {"image", "chart", "interline_equation"}:
            return
        raw_blocks = item.get("blocks", [])
        if isinstance(raw_blocks, list):
            for child in raw_blocks:
                visit(child)

    if isinstance(value, list):
        for item in value:
            visit(item)
    else:
        visit(value)
    return result


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


def _implausible_recovered_text_density(
    span: Mapping[str, Any],
    bbox: tuple[float, float, float, float],
    text: str,
) -> bool:
    if (
        not span.get("fusion_recovery_source")
        or span.get("fusion_recognition_original_ocr") not in {None, ""}
    ):
        return False
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    visible = len(re.sub(r"\s+", "", text))
    line_count = max(text.count("\n") + 1, 1)
    maximum = max(16, int(width / max(height, 1.0) * 2.5))
    repeated_parts = [
        _normalized(part)
        for part in re.split(r"[\s,，。:：;；/|]+", text)
        if _normalized(part)
    ]
    if repeated_parts and re.fullmatch(r"\d+[.)、]?", repeated_parts[0]):
        repeated_parts = repeated_parts[1:]
    repeated = bool(
        len(repeated_parts) >= 3
        and len(set(repeated_parts)) == 1
        and len(repeated_parts[0]) >= 2
    )
    return repeated or bool(
        line_count >= 3
        and height / line_count < 5.5
        and visible >= 24
    ) or bool(
        height <= 12.0
        and width <= 240.0
        and line_count == 1
        and visible >= 24
        and visible > maximum
    )


def _lines_from_text(
    bbox: tuple[float, float, float, float],
    text: str,
    *,
    cell_row: int | None = None,
    cell_col: int | None = None,
    cell_bbox: tuple[float, float, float, float] | None = None,
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
            cell_row,
            cell_col,
            cell_bbox,
        )
        for index, part in enumerate(parts)
    ]


def _prune_aggregate_span_records(
    records: Sequence[
        tuple[Mapping[str, Any], tuple[float, float, float, float], str]
    ],
) -> list[tuple[Mapping[str, Any], tuple[float, float, float, float], str]]:
    """Prefer content-tight child spans over a duplicated aggregate OCR span."""
    skipped: set[int] = set()
    for index, (_span, bbox, text) in enumerate(records):
        normalized = _normalized(text)
        if len(normalized) < 12:
            continue
        represented = []
        for other_index, (_other_span, other_bbox, other_text) in enumerate(records):
            if other_index == index:
                continue
            other_normalized = _normalized(other_text)
            center_x = (other_bbox[0] + other_bbox[2]) / 2
            center_y = (other_bbox[1] + other_bbox[3]) / 2
            if (
                len(other_normalized) >= 3
                and other_normalized in normalized
                and bbox[0] - 2.0 <= center_x <= bbox[2] + 2.0
                and bbox[1] - 2.0 <= center_y <= bbox[3] + 2.0
            ):
                represented.append(other_normalized)
        if (
            len(set(represented)) >= 2
            and sum(len(item) for item in set(represented))
            >= len(normalized) * 0.45
        ):
            skipped.add(index)
    return [record for index, record in enumerate(records) if index not in skipped]


def _visible_cell_lines(
    cell: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> list[SemanticLine]:
    result = []
    excluded = excluded_keys or set()
    cell_bbox = _valid_bbox(cell.get("bbox"))
    cell_row = cell.get("row_start") if isinstance(cell.get("row_start"), int) else None
    cell_col = cell.get("col_start") if isinstance(cell.get("col_start"), int) else None
    raw_spans = cell.get("content_spans", [])
    if isinstance(raw_spans, list):
        records = []
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
            if (
                bbox is not None
                and text
                and not _looks_like_formula_text(text)
                and not _implausible_recovered_text_density(span, bbox, text)
            ):
                records.append((span, bbox, text))
        for _span, bbox, text in _prune_aggregate_span_records(records):
            result.extend(
                    line
                    for line in _lines_from_text(
                        bbox,
                        text,
                        cell_row=cell_row,
                        cell_col=cell_col,
                        cell_bbox=cell_bbox,
                    )
                    if not FOOTER_RE.search(line.text)
                    and not NOISE_TEXT_RE.fullmatch(line.text)
                    and _normalized(line.text) not in excluded
                )
    if not result:
        bbox = _valid_bbox(cell.get("content_bbox")) or cell_bbox
        text = _semantic_cell_text(cell)
        if bbox is not None and text and not _looks_like_formula_text(text):
            result.extend(
                line
                for line in _lines_from_text(
                    bbox,
                    text,
                    cell_row=cell_row,
                    cell_col=cell_col,
                    cell_bbox=cell_bbox,
                )
                if not FOOTER_RE.search(line.text)
                and not NOISE_TEXT_RE.fullmatch(line.text)
                and _normalized(line.text) not in excluded
            )
    return _deduplicate_lines(result)


def _same_visual_position(left: SemanticLine, right: SemanticLine) -> bool:
    horizontal = max(
        0.0,
        min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]),
    )
    vertical = max(
        0.0,
        min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]),
    )
    minimum_width = min(
        left.bbox[2] - left.bbox[0],
        right.bbox[2] - right.bbox[0],
    )
    minimum_height = min(left.height, right.height)
    return bool(
        minimum_width > 0
        and minimum_height > 0
        and horizontal / minimum_width >= 0.45
        and vertical / minimum_height >= 0.45
    )


def _equivalent_semantic_text(left: str, right: str) -> bool:
    normalized_left = _normalized(left)
    normalized_right = _normalized(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    shorter, longer = sorted(
        (normalized_left, normalized_right),
        key=len,
    )
    if len(shorter) >= 6 and shorter in longer and len(shorter) / len(longer) >= 0.8:
        return True
    similarity = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    same_field = _same_field_label(left, right)
    return similarity >= (0.72 if same_field else 0.92)


def _same_field_label(left: str, right: str) -> bool:
    left_fields = {
        _normalized(match.group(0)) for match in FIELD_LABEL_RE.finditer(left)
    }
    right_fields = {
        _normalized(match.group(0)) for match in FIELD_LABEL_RE.finditer(right)
    }
    return bool(left_fields & right_fields)


def _uses_cell_bbox(line: SemanticLine) -> bool:
    return bool(
        line.cell_bbox is not None
        and all(
            abs(value - cell_value) <= 1.0
            for value, cell_value in zip(line.bbox, line.cell_bbox)
        )
    )


def _same_row_adjacent_field_duplicate(
    left: SemanticLine,
    right: SemanticLine,
) -> bool:
    if (
        left.cell_row is None
        or right.cell_row is None
        or left.cell_row != right.cell_row
        or left.cell_col is None
        or right.cell_col is None
        or abs(left.cell_col - right.cell_col) > 1
        or not _same_field_label(left.text, right.text)
    ):
        return False
    horizontal_gap = max(
        left.bbox[0] - right.bbox[2],
        right.bbox[0] - left.bbox[2],
        0.0,
    )
    return bool(
        horizontal_gap <= max(12.0, left.height, right.height)
        and abs(left.center_y - right.center_y)
        <= max(left.height, right.height) * 0.75
    )


def _preferred_duplicate_line(
    existing: SemanticLine,
    candidate: SemanticLine,
) -> SemanticLine:
    existing_fallback = _uses_cell_bbox(existing)
    candidate_fallback = _uses_cell_bbox(candidate)
    if existing_fallback != candidate_fallback:
        return existing if not existing_fallback else candidate
    if len(_normalized(candidate.text)) != len(_normalized(existing.text)):
        return (
            candidate
            if len(_normalized(candidate.text)) > len(_normalized(existing.text))
            else existing
        )
    existing_area = (existing.bbox[2] - existing.bbox[0]) * existing.height
    candidate_area = (candidate.bbox[2] - candidate.bbox[0]) * candidate.height
    return candidate if candidate_area < existing_area else existing


def _line_has_semantic_content(text: str) -> bool:
    return bool(_normalized(text) or _isolated_checkbox_marker(text))


def _equivalent_line_text(left: str, right: str) -> bool:
    if _isolated_checkbox_marker(left) or _isolated_checkbox_marker(right):
        return left.strip() == right.strip()
    return _equivalent_semantic_text(left, right)


def _deduplicate_lines(lines: Sequence[SemanticLine]) -> list[SemanticLine]:
    result: list[SemanticLine] = []
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        if not _line_has_semantic_content(line.text):
            continue
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(result)
                if (
                    _same_visual_position(line, existing)
                    or _same_row_adjacent_field_duplicate(line, existing)
                    or (
                        _same_field_label(line.text, existing.text)
                        and abs(line.center_y - existing.center_y)
                        <= max(line.height, existing.height) * 1.5
                        and max(
                            0.0,
                            min(line.bbox[2], existing.bbox[2])
                            - max(line.bbox[0], existing.bbox[0]),
                        )
                        / min(
                            line.bbox[2] - line.bbox[0],
                            existing.bbox[2] - existing.bbox[0],
                        )
                        >= 0.25
                    )
                )
                and _equivalent_line_text(line.text, existing.text)
            ),
            None,
        )
        if duplicate_index is not None:
            result[duplicate_index] = _preferred_duplicate_line(
                result[duplicate_index],
                line,
            )
            continue
        result.append(line)
    return sorted(result, key=lambda item: (item.bbox[1], item.bbox[0]))


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


def _reading_order_lines(lines: Sequence[SemanticLine]) -> list[SemanticLine]:
    """Return scale-independent row-major order for unstructured text.

    Sorting every fragment by its exact top coordinate makes tiny baseline
    differences swap left/right text.  Group by visual rows first, then sort
    within a row from left to right.  This is deliberately conservative: it
    does not guess columns or rewrite the source text.
    """
    return [line for row in _group_visual_rows(lines) for line in row]


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


def _cell_key(cell: Mapping[str, Any]) -> tuple[int, int] | None:
    row = cell.get("row_start")
    col = cell.get("col_start")
    if not isinstance(row, int) or not isinstance(col, int):
        return None
    return row, col


def _line_for_cell(
    line: SemanticLine,
    cell: Mapping[str, Any],
    *,
    text: str | None = None,
) -> SemanticLine:
    key = _cell_key(cell)
    cell_bbox = _valid_bbox(cell.get("bbox"))
    bbox = line.bbox
    if cell_bbox is not None:
        clipped = (
            max(line.bbox[0], cell_bbox[0]),
            max(line.bbox[1], cell_bbox[1]),
            min(line.bbox[2], cell_bbox[2]),
            min(line.bbox[3], cell_bbox[3]),
        )
        bbox = clipped if _valid_bbox(clipped) is not None else cell_bbox
    return SemanticLine(
        bbox,
        line.text if text is None else text,
        key[0] if key is not None else None,
        key[1] if key is not None else None,
        cell_bbox,
    )


def _reassign_lines_to_matching_cells(
    lines: Sequence[SemanticLine],
    cells: Sequence[Mapping[str, Any]],
) -> list[SemanticLine]:
    """Move a span when only another Cell's authoritative text contains it."""
    cell_texts = [(_normalized(_semantic_cell_text(cell)), cell) for cell in cells]
    result = []
    for line in lines:
        line_key = (line.cell_row, line.cell_col)
        normalized = _normalized(line.text)
        if len(normalized) < 4 or _looks_like_field_label(line.text):
            result.append(line)
            continue
        own_text = next(
            (
                text
                for text, cell in cell_texts
                if _cell_key(cell) == line_key
            ),
            "",
        )
        if normalized in own_text:
            result.append(line)
            continue
        matches = [
            cell
            for text, cell in cell_texts
            if normalized in text and _cell_key(cell) != line_key
        ]
        if len(matches) == 1:
            result.append(_line_for_cell(line, matches[0]))
        else:
            result.append(line)
    return result


def _cell_text_suffix(cell_text: str, label_text: str) -> str:
    clean_cell = _clean_text(cell_text)
    clean_label = _clean_text(label_text).rstrip(":：")
    if not clean_cell or not clean_label:
        return ""
    start = clean_cell.casefold().find(clean_label.casefold())
    if start < 0:
        return ""
    return clean_cell[start + len(clean_label) :].lstrip(" :：\n\t")


def _line_spills_outside_cell(line: SemanticLine) -> bool:
    if line.cell_bbox is None:
        return False
    left, top, right, bottom = line.cell_bbox
    tolerance = max(4.0, min(right - left, bottom - top) * 0.08)
    return bool(
        line.bbox[0] < left - tolerance
        or line.bbox[1] < top - tolerance
        or line.bbox[2] > right + tolerance
        or line.bbox[3] > bottom + tolerance
    )


def _restore_authoritative_cell_values(
    lines: Sequence[SemanticLine],
    cells: Sequence[Mapping[str, Any]],
) -> list[SemanticLine]:
    """Recover per-Cell values hidden inside a cross-Cell aggregate OCR span."""
    result = list(lines)
    for cell in cells:
        key = _cell_key(cell)
        cell_bbox = _valid_bbox(cell.get("bbox"))
        cell_text = _semantic_cell_text(cell)
        if key is None or cell_bbox is None or not cell_text:
            continue
        owned = [
            (index, line)
            for index, line in enumerate(result)
            if (line.cell_row, line.cell_col) == key
        ]
        labels = [
            (index, line)
            for index, line in owned
            if _looks_like_field_label(line.text) and not _is_section(line.text)
        ]
        if len(labels) != 1:
            continue
        _label_index, label = labels[0]
        value = _cell_text_suffix(cell_text, label.text)
        if not _form_value_candidate(value) or not _field_value_compatible(
            label.text,
            value,
        ):
            continue
        normalized_value = _normalized(value)
        value_lines = [
            (index, line)
            for index, line in owned
            if index != _label_index
            and _form_value_candidate(line.text)
            and _field_value_compatible(label.text, line.text)
        ]
        represented = False
        for index, line in value_lines:
            normalized_line = _normalized(line.text)
            if normalized_line == normalized_value:
                represented = True
                break
            if normalized_value in normalized_line and _line_spills_outside_cell(line):
                result[index] = _line_for_cell(line, cell, text=value)
                represented = True
                break
            if normalized_line in normalized_value:
                represented = True
                break
        if represented:
            continue
        value_top = max(cell_bbox[1], min(label.bbox[3] + 1.0, cell_bbox[3] - 1.0))
        value_bbox = (
            cell_bbox[0],
            value_top,
            cell_bbox[2],
            cell_bbox[3],
        )
        result.append(
            SemanticLine(
                value_bbox,
                value,
                key[0],
                key[1],
                cell_bbox,
            )
        )
    return result


def _prune_cross_cell_aggregate_labels(
    lines: Sequence[SemanticLine],
) -> list[SemanticLine]:
    """Drop a wide OCR label that duplicates multiple tighter field labels."""
    skipped: set[int] = set()
    for index, aggregate in enumerate(lines):
        if not _looks_like_field_label(aggregate.text):
            continue
        aggregate_text = _normalized(aggregate.text)
        represented = []
        for other_index, other in enumerate(lines):
            if other_index == index or not _looks_like_field_label(other.text):
                continue
            other_text = _normalized(other.text)
            if (
                len(other_text) < 5
                or other_text == aggregate_text
                or other_text not in aggregate_text
            ):
                continue
            horizontal_margin = max(12.0, aggregate.height * 2.0)
            vertical_margin = max(12.0, aggregate.height * 1.5)
            if (
                aggregate.bbox[0] - horizontal_margin
                <= other.center_x
                <= aggregate.bbox[2] + horizontal_margin
                and aggregate.bbox[1] - vertical_margin
                <= other.center_y
                <= aggregate.bbox[3] + vertical_margin
            ):
                represented.append(other_text)
        distinct = set(represented)
        if len(distinct) >= 2 and sum(map(len, distinct)) >= len(aggregate_text) * 0.45:
            skipped.add(index)
    return [line for index, line in enumerate(lines) if index not in skipped]


def _table_lines(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> list[SemanticLine]:
    cells = [
        cell
        for cell in table.get("table_cells", [])
        if isinstance(cell, Mapping)
    ]
    lines = [
        line
        for cell in cells
        for line in _visible_cell_lines(cell, excluded_keys)
    ]
    lines = _reassign_lines_to_matching_cells(lines, cells)
    lines = _restore_authoritative_cell_values(lines, cells)
    lines = _prune_cross_cell_aggregate_labels(lines)
    return _deduplicate_lines(
        lines
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
    best: tuple[int, float, float, float, list[tuple[str, SemanticLine]]] | None = None
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
        centers = [line.center_y for _category, line in ordered]
        spread = max(centers) - min(centers)
        median_center = statistics.median(centers)
        deviation = statistics.mean(abs(center - median_center) for center in centers)
        candidate = (score, -spread, -deviation, -seed.center_y, ordered)
        if best is None or candidate[:4] > best[:4]:
            best = candidate
    if best is None:
        return None
    ordered = best[4]
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


def _ledger_date_entries(
    lines: Sequence[SemanticLine],
) -> list[tuple[SemanticLine, str]]:
    result: list[tuple[SemanticLine, str]] = []
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
        if not any(
            value == existing_value
            and abs(line.center_y - existing_line.center_y) <= 2.0
            for existing_line, existing_value in result
        ):
            result.append((line, value))
    return result


def _ledger_date_values(lines: Sequence[SemanticLine]) -> list[str]:
    return [value for _line, value in _ledger_date_entries(lines)]


def _ledger_labeled_date_values(
    lines: Sequence[SemanticLine],
) -> dict[str, str]:
    label_patterns = {
        "admitted": re.compile(r"Date\s+Admitted|入院日期", re.IGNORECASE),
        "discharged": re.compile(r"Date\s+Discharged|出院日期", re.IGNORECASE),
    }
    label_centers = {
        name: statistics.mean(line.center_y for line in matches)
        for name, pattern in label_patterns.items()
        if (matches := [line for line in lines if pattern.search(line.text)])
    }
    entries = _ledger_date_entries(lines)
    pair_candidates = sorted(
        (
            abs(line.center_y - center_y),
            name,
            index,
            value,
        )
        for name, center_y in label_centers.items()
        for index, (line, value) in enumerate(entries)
    )
    assigned_fields: dict[str, str] = {}
    assigned_entries: set[int] = set()
    for distance, name, index, value in pair_candidates:
        if distance > 60.0 or name in assigned_fields or index in assigned_entries:
            continue
        assigned_fields[name] = value
        assigned_entries.add(index)
    return assigned_fields


def _append_unique_ledger_text(
    parts: list[str],
    seen: list[str] | None,
    text: str,
) -> None:
    key = _output_text_key(text)
    if not key:
        return
    if seen is not None:
        duplicate = (
            key in seen
            if text.lstrip().startswith("- **")
            else any(_equivalent_semantic_text(key, existing) for existing in seen)
        )
        if duplicate:
            return
        seen.append(key)
    parts.append(text)


def _render_ledger_tail(
    lines: Sequence[SemanticLine],
    seen: list[str] | None = None,
) -> str:
    if not lines:
        return ""
    joined = " ".join(line.text for line in lines)
    has_admitted = bool(re.search(r"Date\s+Admitted|入院日期", joined, re.IGNORECASE))
    has_discharged = bool(
        re.search(r"Date\s+Discharged|出院日期", joined, re.IGNORECASE)
    )
    labeled_dates = _ledger_labeled_date_values(lines)
    parts = []
    if any(re.search(r"\[DISCHARGED\]", line.text, re.IGNORECASE) for line in lines):
        _append_unique_ledger_text(parts, seen, "### Discharged / 出院")
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
            _append_unique_ledger_text(parts, seen, text)
    if has_admitted and (value := labeled_dates.get("admitted")):
        _append_unique_ledger_text(
            parts,
            seen,
            f"- **Date Admitted / 入院日期**: {value}",
        )
    if has_discharged and (value := labeled_dates.get("discharged")):
        _append_unique_ledger_text(
            parts,
            seen,
            f"- **Date Discharged / 出院日期**: {value}",
        )
    return "\n\n".join(parts)


def _render_ledger_table(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
    seen_preheaders: set[str] | None = None,
    seen_tail: list[str] | None = None,
) -> str | None:
    lines = _table_lines(table)
    header = _ledger_header(lines)
    if header is None:
        return None
    if excluded_keys:
        lines = [
            line
            for line in lines
            if _normalized(line.text) not in excluded_keys
            or _ledger_category(line.text) is not None
        ]
        header = _ledger_header(lines)
        if header is None:
            return None
    columns, header_top, header_bottom = header
    headers = [LEDGER_LABELS[category] for category, _line in columns]
    header_line_ids = {id(line) for _category, line in columns}
    header_band_extras = [
        line
        for line in lines
        if id(line) not in header_line_ids
        and line.bbox[1] <= header_bottom
        and line.bbox[3] >= header_top
        and _ledger_category(line.text) is None
        and LEDGER_CURRENCY_UNIT_RE.fullmatch(line.text.strip()) is None
    ]
    preheader_lines = _deduplicate_lines(
        [line for line in lines if line.bbox[3] < header_top - 1.0]
        + header_band_extras
    )
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
        for line in preheader:
            key = _output_text_key(line)
            if not line or FOOTER_RE.search(line) or not key:
                continue
            if seen_preheaders is not None:
                if key in seen_preheaders:
                    continue
                seen_preheaders.add(key)
            parts.append(line)
    parts.append(_markdown_table(headers, rows))
    tail = _render_ledger_tail(tail_lines, seen_tail)
    if tail:
        parts.append(tail)
    return "\n\n".join(part for part in parts if part)


def _is_section(text: str) -> bool:
    stripped = text.strip()
    section_candidate = re.sub(
        r"^(?:\d{1,3}[.)、:]|\([A-Za-z0-9]{1,3}\))\s*",
        "",
        stripped,
    )
    if SECTION_RE.search(section_candidate):
        return True
    letters = [character for character in stripped if character.isalpha()]
    return bool(
        len(letters) >= 5
        and len(stripped) <= 100
        and TITLE_HINT_RE.search(stripped) is not None
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


def _isolated_checkbox_marker(text: str) -> bool:
    return text.strip() in {"☑", "☒", "✓", "✔", "■", "☐", "□", "◫"}


def _merge_isolated_checkbox_markers(
    lines: Sequence[SemanticLine],
) -> list[SemanticLine]:
    """Join each standalone control to its own nearby right-hand label."""
    result: list[SemanticLine] = []
    for group in _group_visual_rows(lines):
        ordered = sorted(group, key=lambda item: item.bbox[0])
        index = 0
        while index < len(ordered):
            marker = ordered[index]
            if not _isolated_checkbox_marker(marker.text):
                result.append(marker)
                index += 1
                continue
            next_marker = next(
                (
                    candidate_index
                    for candidate_index in range(index + 1, len(ordered))
                    if _isolated_checkbox_marker(ordered[candidate_index].text)
                ),
                len(ordered),
            )
            followers = ordered[index + 1 : next_marker]
            first_gap = (
                followers[0].bbox[0] - marker.bbox[2]
                if followers
                else float("inf")
            )
            if (
                not followers
                or first_gap > max(30.0, marker.height * 4.0)
                or _vertical_overlap(marker, followers[0]) < 0.25
            ):
                result.append(marker)
                index += 1
                continue
            attached = [followers[0]]
            for follower in followers[1:]:
                previous = attached[-1]
                gap = follower.bbox[0] - previous.bbox[2]
                if gap > max(30.0, previous.height * 4.0):
                    break
                attached.append(follower)
            merged_items = [marker, *attached]
            result.append(
                SemanticLine(
                    (
                        min(item.bbox[0] for item in merged_items),
                        min(item.bbox[1] for item in merged_items),
                        max(item.bbox[2] for item in merged_items),
                        max(item.bbox[3] for item in merged_items),
                    ),
                    f"{marker.text.strip()} {_join_row_lines(attached)}".strip(),
                    marker.cell_row,
                    marker.cell_col,
                    marker.cell_bbox,
                )
            )
            index += 1 + len(attached)
    return _reading_order_lines(result)


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
    return result


def _merge_control_markers(
    lines: Sequence[SemanticLine],
) -> list[SemanticLine]:
    return _merge_isolated_list_markers(
        _merge_isolated_checkbox_markers(lines)
    )


def _looks_like_field_label(text: str) -> bool:
    stripped = text.strip()
    if re.search(
        r"diagnostic procedures?.*medication.*treatment.*operation|"
        r"診斷程序.*藥物.*治療.*手術",
        stripped,
        re.IGNORECASE,
    ):
        return False
    match = FIELD_LABEL_RE.search(text)
    if not match or match.start() > 35 or len(stripped) > 120:
        return False
    if stripped.startswith(("☑", "☐", "☒", "□", "✓", "✔", "■", "◫")):
        return False
    if OPTION_TEXT_RE.search(stripped) and not stripped.rstrip().endswith((":", "：")):
        return False
    words = re.findall(r"[A-Za-z0-9]+", stripped)
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", stripped))
    if len(words) > 18:
        return False
    if len(stripped) > 80 and FORM_PROMPT_RE.match(stripped) is None:
        return False
    if (
        cjk_count > 30
        and FORM_PROMPT_RE.match(stripped) is None
        and not stripped.rstrip().endswith((":", "："))
    ):
        return False
    if stripped.endswith(("。", "!", "！", "?", "？")) or (
        stripped.endswith(".")
        and re.search(r"\b(?:No|Nº)\.$", stripped, re.IGNORECASE) is None
    ):
        return False
    if INSTRUCTION_TEXT_RE.search(stripped) and (
        len(words) > 8 or len(stripped) > 45
    ):
        return False
    return True


def _form_value_candidate(text: str) -> bool:
    stripped = text.strip()
    if (
        not stripped
        or len(stripped) > 140
        or _looks_like_field_label(stripped)
        or _is_section(stripped)
        or _task_line(stripped) is not None
        or ISOLATED_LIST_MARKER_RE.fullmatch(stripped)
        or FORM_PROMPT_RE.match(stripped)
        or QUESTION_PROMPT_RE.search(stripped)
        or LABEL_QUALIFIER_RE.fullmatch(stripped)
        or OPTION_TEXT_RE.search(stripped)
        or FOOTER_RE.search(stripped)
        or NOISE_TEXT_RE.fullmatch(stripped)
    ):
        return False
    words = re.findall(r"[A-Za-z0-9]+", stripped)
    return not INSTRUCTION_TEXT_RE.search(stripped) and len(words) <= 14


def _field_value_compatible(label_text: str, value_text: str) -> bool:
    label = label_text.casefold()
    value = value_text.strip()
    date_value = bool(DATE_VALUE_RE.fullmatch(value))
    qualification_value = bool(
        re.search(r"\b(?:MBBS|FRCS|FHKAM|qualification)\b", value, re.IGNORECASE)
    )
    address_value = bool(
        re.search(
            r"\b(?:road|street|avenue|building|tower|floor|flat|room|hong kong|h\.k\.)\b|"
            r"\d+\s*/?F\b|地址|香港|九龍",
            value,
            re.IGNORECASE,
        )
    )
    if re.search(r"\baddress\b|地址", label):
        return not date_value and not qualification_value
    if re.search(r"\bdate\b|日期", label):
        return date_value or bool(TIME_VALUE_RE.fullmatch(value))
    if re.search(r"\b(?:phone|telephone|mobile)\b|電話", label):
        return any(character.isdigit() for character in value)
    if IDENTIFIER_LABEL_RE.search(label):
        return bool(
            len(value) <= 50
            and any(character.isdigit() for character in value)
            and DATE_VALUE_RE.fullmatch(value) is None
            and re.search(r"身份[證証]|護照", value) is None
        )
    if EMAIL_LABEL_RE.search(label):
        return "@" in value and not QUESTION_PROMPT_RE.search(value)
    if POSTAL_LABEL_RE.search(label):
        return bool(
            2 <= len(value) <= 16
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 -]*", value) is not None
        )
    if AMOUNT_LABEL_RE.search(label):
        return MONEY_VALUE_RE.fullmatch(value) is not None
    if re.search(r"\b(?:age|sex|gender)\b|年齡|年龄|性別|性别", label):
        return bool(
            PARTIAL_DATE_VALUE_RE.fullmatch(value) is None
            and (
                re.search(r"\b(?:M|F|MALE|FEMALE)\b", value, re.IGNORECASE)
                or re.fullmatch(r"\s*\d{1,3}\s*", value)
                or re.fullmatch(
                    r"\s*(?:M|F)\s*[/ -]?\s*\d{1,3}\s*",
                    value,
                    re.IGNORECASE,
                )
                or re.fullmatch(
                    r"\s*\d{1,3}\s*[/ -]?\s*(?:M|F)\s*",
                    value,
                    re.IGNORECASE,
                )
            )
        )
    if re.search(r"\b(?:name|occupation|diagnosis)\b|姓名|職業|診斷", label):
        return bool(
            not date_value
            and MONEY_VALUE_RE.fullmatch(value) is None
            and not address_value
            and not qualification_value
            and re.search(r"\b(?:tel|fax|phone)\b", value, re.IGNORECASE) is None
        )
    return True


def _field_concepts(text: str) -> frozenset[str]:
    patterns = {
        "name": re.compile(r"\bname\b|姓名", re.IGNORECASE),
        "date": re.compile(r"\bdate\b|日期", re.IGNORECASE),
        "address": re.compile(r"\baddress\b|地址", re.IGNORECASE),
        "phone": re.compile(
            r"\b(?:phone|telephone|mobile)\b|電話|手机|手機",
            re.IGNORECASE,
        ),
        "id": IDENTIFIER_LABEL_RE,
        "diagnosis": re.compile(r"\bdiagnosis\b|診斷", re.IGNORECASE),
        "result": re.compile(r"\bresult\b|結果", re.IGNORECASE),
    }
    return frozenset(name for name, pattern in patterns.items() if pattern.search(text))


def _merge_stacked_field_labels(
    lines: Sequence[SemanticLine],
) -> list[SemanticLine]:
    result: list[SemanticLine] = []
    consumed: set[int] = set()
    for index, line in enumerate(lines):
        if index in consumed or not _looks_like_field_label(line.text):
            if index not in consumed:
                result.append(line)
            continue
        concepts = _field_concepts(line.text)
        has_cjk = bool(re.search(r"[\u3400-\u9fff]", line.text))
        match_index = None
        for other_index in range(index + 1, len(lines)):
            other = lines[other_index]
            if (
                other_index in consumed
                or line.cell_row is None
                or line.cell_row != other.cell_row
                or line.cell_col is None
                or line.cell_col != other.cell_col
                or not _looks_like_field_label(other.text)
                or not concepts
                or _field_concepts(other.text) != concepts
                or has_cjk
                == bool(re.search(r"[\u3400-\u9fff]", other.text))
            ):
                continue
            vertical_gap = max(
                other.bbox[1] - line.bbox[3],
                line.bbox[1] - other.bbox[3],
                0.0,
            )
            if vertical_gap <= max(20.0, line.height * 2.0, other.height * 2.0):
                match_index = other_index
                break
        if match_index is None:
            result.append(line)
            continue
        other = lines[match_index]
        consumed.add(match_index)
        result.append(
            SemanticLine(
                (
                    min(line.bbox[0], other.bbox[0]),
                    min(line.bbox[1], other.bbox[1]),
                    max(line.bbox[2], other.bbox[2]),
                    max(line.bbox[3], other.bbox[3]),
                ),
                f"{line.text.rstrip(':：')} / {other.text.rstrip(':：')}",
                line.cell_row,
                line.cell_col,
                line.cell_bbox,
            )
        )
    return sorted(result, key=lambda item: (item.bbox[1], item.bbox[0]))


def _field_assignment_score(
    label: SemanticLine,
    value: SemanticLine,
) -> tuple[int, float, float] | None:
    if not _field_value_compatible(label.text, value.text):
        return None
    same_cell_row = bool(
        label.cell_row is not None
        and value.cell_row is not None
        and label.cell_row == value.cell_row
    )
    same_value_cell = bool(
        same_cell_row
        and label.cell_col is not None
        and value.cell_col is not None
        and label.cell_col == value.cell_col
    )
    adjacent_value_cell = bool(
        same_cell_row
        and label.cell_col is not None
        and value.cell_col is not None
        and value.cell_col == label.cell_col + 1
        and value.center_x > label.center_x
    )
    if (
        label.cell_row is not None
        and value.cell_row is not None
        and abs(value.cell_row - label.cell_row) > 1
    ):
        return None
    if (
        not same_cell_row
        and value.center_y < label.center_y - max(label.height, value.height) * 0.35
    ):
        return None
    horizontal_gap = value.bbox[0] - label.bbox[2]
    overlap = _vertical_overlap(label, value)
    center_delta = abs(value.center_x - label.center_x)
    likely_filled = (
        value.height >= max(label.height * 1.2, 13.0)
        or bool(
            DATE_TOKEN_RE.search(value.text)
            or TIME_VALUE_RE.fullmatch(value.text.strip())
            or MONEY_VALUE_RE.fullmatch(value.text.strip())
            or re.search(r"\d|@", value.text)
        )
    )
    if same_value_cell:
        cell_height = max(
            (
                label.cell_bbox[3] - label.cell_bbox[1]
                if label.cell_bbox is not None
                else label.height
            ),
            (
                value.cell_bbox[3] - value.cell_bbox[1]
                if value.cell_bbox is not None
                else value.height
            ),
        )
        if abs(value.center_y - label.center_y) <= cell_height + 12.0:
            return (-2, abs(value.center_y - label.center_y), center_delta)
    if adjacent_value_cell:
        row_height = max(
            (
                label.cell_bbox[3] - label.cell_bbox[1]
                if label.cell_bbox is not None
                else label.height
            ),
            (
                value.cell_bbox[3] - value.cell_bbox[1]
                if value.cell_bbox is not None
                else value.height
            ),
        )
        if abs(value.center_y - label.center_y) <= row_height + 12.0:
            return (-1, abs(value.center_y - label.center_y), center_delta)
    if (
        same_cell_row
        and likely_filled
        and -3.0 <= horizontal_gap <= 160.0
        and abs(value.center_y - label.center_y)
        <= max(
            35.0,
            (label.cell_bbox[3] - label.cell_bbox[1])
            if label.cell_bbox is not None
            else 35.0,
        )
    ):
        return (0, max(horizontal_gap, 0.0), center_delta)
    if (
        likely_filled
        and overlap >= 0.35
        and -3.0 <= horizontal_gap <= (160.0 if same_cell_row else 80.0)
    ):
        return (0, max(horizontal_gap, 0.0), center_delta)
    vertical_gap = value.bbox[1] - label.bbox[3]
    label_width = label.bbox[2] - label.bbox[0]
    value_width = value.bbox[2] - value.bbox[0]
    horizontal_overlap = max(
        0.0,
        min(label.bbox[2], value.bbox[2])
        - max(label.bbox[0], value.bbox[0]),
    )
    minimum_width = min(label_width, value_width)
    overlap_ratio = horizontal_overlap / minimum_width if minimum_width > 0 else 0.0
    if (
        -max(6.0, label.height * 0.6) <= vertical_gap <= 55.0
        and (
            overlap_ratio >= 0.15
            or center_delta <= max(label_width, value_width) * 0.65 + 15.0
        )
    ):
        if not likely_filled:
            return None
        return (1, max(vertical_gap, 0.0), center_delta)
    return None


def _has_assignment_boundary(
    lines: Sequence[SemanticLine],
    label_index: int,
    value_index: int,
) -> bool:
    label = lines[label_index]
    value = lines[value_index]
    top = min(label.center_y, value.center_y)
    bottom = max(label.center_y, value.center_y)
    column_left = min(label.bbox[0], value.bbox[0]) - 4.0
    column_right = max(label.bbox[2], value.bbox[2]) + 4.0
    column_width = max(column_right - column_left, 1.0)
    for index, boundary in enumerate(lines):
        if index in {label_index, value_index} or not (
            top < boundary.center_y < bottom
        ):
            continue
        if not (
            _looks_like_field_label(boundary.text)
            or FORM_PROMPT_RE.match(boundary.text.strip())
            or _is_section(boundary.text)
        ):
            continue
        horizontal_overlap = max(
            0.0,
            min(column_right, boundary.bbox[2])
            - max(column_left, boundary.bbox[0]),
        )
        boundary_width = boundary.bbox[2] - boundary.bbox[0]
        if (
            column_left <= boundary.center_x <= column_right
            or horizontal_overlap / min(column_width, boundary_width) >= 0.35
        ):
            return True
    return False


def _merge_adjacent_value_text(left: str, right: str, gap: float) -> str:
    left = left.rstrip()
    right = right.lstrip()
    maximum_overlap = min(len(left), len(right), 16)
    overlap = 0
    for size in range(maximum_overlap, 0, -1):
        if left[-size:].casefold() == right[:size].casefold() and (
            size >= 2 or gap <= 4.0
        ):
            overlap = size
            break
    right = right[overlap:].lstrip()
    if not right:
        return left
    cjk_boundary = bool(
        re.search(r"[\u3400-\u9fff]$", left)
        or re.match(r"^[\u3400-\u9fff]", right)
    )
    separator = "" if cjk_boundary else " "
    return f"{left}{separator}{right}".strip()


def _join_field_values(values: Sequence[SemanticLine]) -> str:
    groups = _group_visual_rows(values)
    rendered_groups = []
    for group in groups:
        ordered = sorted(group, key=lambda item: item.bbox[0])
        chunks: list[tuple[str, SemanticLine]] = []
        for line in ordered:
            if not chunks:
                chunks.append((line.text.strip(), line))
                continue
            previous_text, previous_line = chunks[-1]
            gap = line.bbox[0] - previous_line.bbox[2]
            if gap <= max(6.0, previous_line.height * 0.55, line.height * 0.55):
                chunks[-1] = (
                    _merge_adjacent_value_text(previous_text, line.text, gap),
                    SemanticLine(
                        (
                            previous_line.bbox[0],
                            min(previous_line.bbox[1], line.bbox[1]),
                            max(previous_line.bbox[2], line.bbox[2]),
                            max(previous_line.bbox[3], line.bbox[3]),
                        ),
                        "",
                    ),
                )
            else:
                chunks.append((line.text.strip(), line))
        rendered_groups.append(" / ".join(text for text, _line in chunks if text))
    return " / ".join(group for group in rendered_groups if group)


def _adjacent_value_continues_own_fragment(
    label: SemanticLine,
    value: SemanticLine,
    own_values: Sequence[SemanticLine],
    labels: Sequence[tuple[int, SemanticLine]],
) -> bool:
    if not own_values or any(
        other.cell_row == value.cell_row
        and other.cell_col == value.cell_col
        for _index, other in labels
    ):
        return False
    for own in own_values:
        gap = value.bbox[0] - own.bbox[2]
        if (
            len(_normalized(own.text)) <= 12
            and -4.0 <= gap <= 8.0
            and _vertical_overlap(own, value) >= 0.35
            and value.center_x > label.center_x
        ):
            return True
    return False


def _pair_form_fields(lines: Sequence[SemanticLine]) -> list[SemanticLine]:
    lines = _merge_stacked_field_labels(lines)
    labels = [
        (index, line)
        for index, line in enumerate(lines)
        if _looks_like_field_label(line.text) and not _is_section(line.text)
    ]
    label_indices = {index for index, _line in labels}
    own_values_by_label = {
        label_index: [
            value
            for value_index, value in enumerate(lines)
            if value_index not in label_indices
            and label.cell_row is not None
            and value.cell_row == label.cell_row
            and label.cell_col is not None
            and value.cell_col == label.cell_col
            and _form_value_candidate(value.text)
            and _field_value_compatible(label.text, value.text)
        ]
        for label_index, label in labels
    }
    labels_with_own_values = {
        label_index
        for label_index, values in own_values_by_label.items()
        if values
    }
    assignments: dict[int, list[int]] = defaultdict(list)
    assigned_values: set[int] = set()
    for value_index, value in enumerate(lines):
        if value_index in label_indices or not _form_value_candidate(value.text):
            continue
        candidates = []
        for label_index, label in labels:
            adjacent_value_cell = bool(
                label.cell_row is not None
                and value.cell_row is not None
                and label.cell_row == value.cell_row
                and label.cell_col is not None
                and value.cell_col is not None
                and value.cell_col == label.cell_col + 1
            )
            if (
                adjacent_value_cell
                and label_index in labels_with_own_values
                and not _adjacent_value_continues_own_fragment(
                    label,
                    value,
                    own_values_by_label[label_index],
                    labels,
                )
            ):
                continue
            if (
                label_index >= value_index
                and _vertical_overlap(label, value) < 0.35
                and not adjacent_value_cell
            ):
                continue
            if _has_assignment_boundary(
                lines,
                label_index,
                value_index,
            ):
                continue
            score = _field_assignment_score(label, value)
            if score is not None:
                candidates.append((score, label_index))
        selected = min(candidates, default=None)
        if selected is None:
            continue
        assignments[selected[1]].append(value_index)
        assigned_values.add(value_index)

    for value_index, value in enumerate(lines):
        if value_index in assigned_values or value_index in label_indices:
            continue
        if not _form_value_candidate(value.text) or any(
            label.cell_row == value.cell_row and label.cell_col == value.cell_col
            for _label_index, label in labels
        ):
            continue
        continuations = []
        for label_index, label in labels:
            if not _field_value_compatible(label.text, value.text):
                continue
            for assigned_index in assignments.get(label_index, []):
                assigned = lines[assigned_index]
                gap = value.bbox[0] - assigned.bbox[2]
                if (
                    len(_normalized(assigned.text)) <= 12
                    and -4.0 <= gap <= 8.0
                    and _vertical_overlap(assigned, value) >= 0.35
                ):
                    continuations.append((abs(gap), label_index))
        selected_continuation = min(continuations, default=None)
        if selected_continuation is not None:
            assignments[selected_continuation[1]].append(value_index)
            assigned_values.add(value_index)

    result = []
    for index, line in enumerate(lines):
        if index in assigned_values:
            continue
        values = assignments.get(index, [])
        if values:
            related = [line, *(lines[value_index] for value_index in values)]
            label = line.text.strip().rstrip(":：")
            value = _join_field_values(
                [lines[value_index] for value_index in values]
            )
            result.append(
                SemanticLine(
                    (
                        min(item.bbox[0] for item in related),
                        min(item.bbox[1] for item in related),
                        max(item.bbox[2] for item in related),
                        max(item.bbox[3] for item in related),
                    ),
                    f"- **{label}**: {value}",
                )
            )
            continue
        inline = re.match(r"^(.{1,100}?)[：:]\s*(\S.+)$", line.text.strip())
        if (
            inline is not None
            and _looks_like_field_label(inline.group(1))
            and _form_value_candidate(inline.group(2))
        ):
            result.append(
                SemanticLine(
                    line.bbox,
                    f"- **{inline.group(1).strip()}**: {inline.group(2).strip()}",
                )
            )
        else:
            result.append(line)
    return result


def _render_form_line(text: str) -> str | None:
    stripped = text.strip()
    if (
        not stripped
        or FOOTER_RE.search(stripped)
        or NOISE_TEXT_RE.fullmatch(stripped)
        or LABEL_QUALIFIER_RE.fullmatch(stripped)
    ):
        return None
    if stripped.startswith("- **"):
        return stripped
    task = _task_line(stripped)
    if task is not None:
        return task
    if _is_section(stripped):
        return f"### {stripped}"
    if _looks_like_field_label(stripped):
        return f"- **{stripped.rstrip(':：')}**"
    return stripped


def _output_text_key(text: str) -> str:
    return _normalized(
        re.sub(r"^(?:#+\s*|-\s*(?:\[[x ?]\]\s*)?|\*+)", "", text).replace(
            "**",
            "",
        )
    )


def _text_represented_in_markdown(text: str, markdown_key: str) -> bool:
    normalized = _normalized(text)
    if not normalized:
        return False
    if normalized in markdown_key:
        return True
    markerless = _normalized(
        re.sub(
            r"^(?:\d{1,3}[.)、:]|\([A-Za-z0-9]{1,3}\))\s*",
            "",
            text.strip(),
        )
    )
    if len(markerless) >= 5 and markerless in markdown_key:
        return True
    signature_markerless = _normalized(
        SIGNATURE_VALUE_PREFIX_RE.sub("", text.strip(), count=1)
    )
    if len(signature_markerless) >= 4 and signature_markerless in markdown_key:
        return True
    if "\n" in text or "\r" in text:
        return False
    return bool(
        len(normalized) >= 10
        and any(
            len(_normalized(fragment)) >= 8
            and _normalized(fragment) in markdown_key
            for fragment in re.findall(r"[A-Za-z0-9]{4,}|[\u3400-\u9fff]{4,}", text)
        )
    )


def _aggregate_text_represented_in_markdown(
    text: str,
    markdown_key: str,
) -> bool:
    fragments = [
        fragment.strip()
        for fragment in re.split(r"[\r\n]+", text)
        if len(_normalized(fragment)) >= 4
    ]
    if len(fragments) < 2:
        return False
    unique_fragments = list(
        dict.fromkeys(_normalized(fragment) for fragment in fragments)
    )
    represented = [
        fragment
        for fragment in fragments
        if _text_represented_in_markdown(fragment, markdown_key)
    ]
    represented_keys = {_normalized(fragment) for fragment in represented}
    represented_characters = sum(len(key) for key in represented_keys)
    total_characters = sum(len(key) for key in unique_fragments)
    return bool(
        len(represented_keys) >= 2
        and total_characters > 0
        and represented_characters / total_characters >= 0.8
    )


def _structured_value_represented_in_markdown(
    text: str,
    markdown_key: str,
) -> bool:
    return any(
        len(_normalized(match.group(0))) >= 4
        and _normalized(match.group(0)) in markdown_key
        for match in DATE_TOKEN_RE.finditer(text)
    )


def _append_unique_output(
    parts: list[str],
    seen: list[Any],
    text: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    key = _output_text_key(text)
    if not key:
        return
    if not _repeatable_value(text):
        duplicate = False
        for existing in seen:
            if isinstance(existing, tuple) and len(existing) == 2:
                existing_key, existing_bbox = existing
            else:
                existing_key, existing_bbox = existing, None
            if not _equivalent_semantic_text(key, str(existing_key)):
                continue
            if bbox is None or existing_bbox is None:
                duplicate = True
                break
            if _same_visual_position(
                SemanticLine(bbox, key),
                SemanticLine(existing_bbox, str(existing_key)),
            ):
                duplicate = True
                break
        if duplicate:
            return
        seen.append((key, bbox) if bbox is not None else key)
    parts.append(text)


def _active_page_objects(page: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield output-owning page evidence, excluding diagnostic/discarded copies."""
    yielded: set[int] = set()
    for key in ("preproc_blocks",):
        for item in _iter_objects(page.get(key, [])):
            if id(item) not in yielded:
                yielded.add(id(item))
                yield item


def _is_table_cell(item: Mapping[str, Any]) -> bool:
    return bool(
        "row_start" in item
        and "col_start" in item
        and ("text" in item or "content_spans" in item)
    )


def _source_text_candidate(
    item: Mapping[str, Any],
) -> tuple[str, tuple[float, float, float, float] | None] | None:
    if _is_table_cell(item):
        raw_spans = item.get("content_spans", [])
        has_visible_span = bool(
            isinstance(raw_spans, list)
            and any(
                isinstance(span, Mapping)
                and _span_text(span)
                and not span.get("fusion_visualization_hidden")
                for span in raw_spans
            )
        )
        # A Cell with visible child spans is represented by those span records.
        # Use the Cell text only as a fallback when the children are absent or
        # entirely hidden, matching the renderer's authoritative-text fallback.
        if has_visible_span:
            return None
        text = _semantic_cell_text(item)
        bbox = _valid_bbox(item.get("content_bbox")) or _valid_bbox(
            item.get("bbox")
        )
        return text, bbox
    span_type = item.get("type")
    untyped_leaf = bool(
        span_type is None
        and not any(
            key in item
            for key in ("blocks", "lines", "spans", "content_spans", "table_cells")
        )
    )
    if span_type in {"text", "hyperlink"} or untyped_leaf:
        text = _span_text(item)
        return text, _valid_bbox(item.get("bbox"))
    return None


def _page_has_source_text(page: Mapping[str, Any]) -> bool:
    for item in _active_page_objects(page):
        candidate = _source_text_candidate(item)
        if candidate is not None and _normalized(candidate[0]):
            return True
    return False


def _text_source_stats(middle_json: Mapping[str, Any]) -> tuple[int, int]:
    seen: set[int] = set()
    count = 0
    characters = 0
    pages = middle_json.get("pdf_info", [])
    for page in pages if isinstance(pages, list) else []:
        if not isinstance(page, Mapping):
            continue
        for item in _active_page_objects(page):
            if id(item) in seen:
                continue
            seen.add(id(item))
            candidate = _source_text_candidate(item)
            if candidate is None:
                continue
            text, _bbox = candidate
            if not _normalized(text):
                continue
            count += 1
            characters += len(_normalized(text))
    return count, characters


def _trace_source_records(
    middle_json: Mapping[str, Any],
    markdown: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pages = middle_json.get("pdf_info", [])
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    if not isinstance(pages, list):
        return records, dict(counts)
    markdown_key = _normalized(markdown)
    repeated_margins = _margin_repetitions(pages)
    seen: set[int] = set()
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            continue
        page_size = page.get("page_size", [])
        page_height = (
            float(page_size[1])
            if isinstance(page_size, (list, tuple)) and len(page_size) >= 2
            else 0.0
        )
        for item in _active_page_objects(page):
            if id(item) in seen:
                continue
            seen.add(id(item))
            candidate = _source_text_candidate(item)
            if candidate is None:
                continue
            text, bbox = candidate
            normalized = _normalized(text)
            if not text or bbox is None:
                continue
            at_page_bottom = bool(page_height > 0 and bbox[3] >= page_height * 0.9)
            if FOOTER_RE.search(text) or (
                at_page_bottom
                and (
                    CONTACT_FOOTER_RE.search(text)
                    or BOTTOM_ORGANIZATION_RE.search(text)
                )
            ):
                status = "filtered_footer"
            elif NOISE_TEXT_RE.fullmatch(text):
                status = "filtered_noise"
            elif _looks_like_formula_text(text):
                status = "filtered_formula"
            elif _implausible_recovered_text_density(item, bbox, text):
                status = "filtered_recovery_quality"
            elif not normalized:
                status = "filtered_punctuation"
            elif normalized in repeated_margins:
                status = "deduplicated_repeated_margin"
            elif (
                (category := _ledger_category(text)) is not None
                and _normalized(LEDGER_LABELS[category]) in markdown_key
            ):
                status = "represented_schema"
            elif (
                LEDGER_CURRENCY_UNIT_RE.fullmatch(text.strip())
                and _normalized(LEDGER_LABELS["amount"]) in markdown_key
            ):
                status = "represented_schema"
            elif _isolated_checkbox_marker(text) and re.search(
                r"^- \[[x ?]\] ",
                markdown,
                re.MULTILINE,
            ):
                status = "represented_control"
            elif item.get("fusion_visualization_hidden"):
                status = "represented_grouped_marker"
            elif _text_represented_in_markdown(text, markdown_key):
                status = "represented"
            elif _structured_value_represented_in_markdown(text, markdown_key):
                status = "represented_value"
            elif _aggregate_text_represented_in_markdown(text, markdown_key):
                status = "represented_aggregate"
            elif len(normalized) >= 6 and any(
                _equivalent_semantic_text(text, line)
                for line in markdown.splitlines()
                if line.strip() and not line.lstrip().startswith("<!--")
            ):
                status = "represented_equivalent"
            else:
                status = "unmatched"
            counts[status] += 1
            records.append(
                {
                    "page": page_index + 1,
                    "bbox": [round(value, 4) for value in bbox],
                    "text": text,
                    "status": status,
                    "source": (
                        item.get("fusion_recognition_source")
                        or item.get("fusion_source")
                        or item.get("fusion_recovery_source")
                        or "ocr"
                    ),
                }
            )
    return records, dict(sorted(counts.items()))


def _markdown_body_stats(markdown: str) -> tuple[int, int]:
    lines = [
        line.strip()
        for line in markdown.splitlines()
        if line.strip() and not line.lstrip().startswith("<!--")
    ]
    return len(lines), sum(len(_normalized(line)) for line in lines)


def _signature_row_bounds(
    anchors: Sequence[SemanticLine],
    lines: Sequence[SemanticLine],
) -> list[tuple[float, float]]:
    centers = [anchor.center_y for anchor in anchors]
    top = min(line.bbox[1] for line in lines)
    bottom = max(line.bbox[3] for line in lines)
    bounds = []
    for index, center in enumerate(centers):
        row_top = (
            (centers[index - 1] + center) / 2
            if index
            else max(top, center - max(16.0, anchors[index].height * 2.0))
        )
        row_bottom = (
            (center + centers[index + 1]) / 2
            if index + 1 < len(centers)
            else bottom
        )
        bounds.append((row_top, row_bottom))
    return bounds


def _signature_name_value(text: str) -> str:
    value = _clean_text(text)
    value = re.split(
        r"\b(?:ID\s*/?\s*P\w*|Passport|Date)\b|身份[證证詮]",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = SIGNATURE_VALUE_PREFIX_RE.sub("", value, count=1).strip()
    value = re.sub(r"^[^A-Za-z\u3400-\u9fff]+", "", value).strip()
    value = re.sub(r"^[\u3400-\u9fff]\s*[)）]\s*", "", value).strip()
    if (
        len(_normalized(value)) < 4
        or any(character.isdigit() for character in value)
        or _looks_like_field_label(value)
        or MEDICAL_QUALIFICATION_RE.search(value)
        or POLICY_OWNER_SIGNATURE_RE.search(value)
        or INSURED_SIGNATURE_RE.search(value)
        or BLOCK_NAME_LABEL_RE.search(value)
    ):
        return ""
    words = re.findall(r"[A-Za-z]+|[\u3400-\u9fff]+", value)
    return value if words and len(words) <= 6 else ""


def _signature_identifier_value(text: str) -> str:
    value = _clean_text(text)
    value = re.sub(
        r"^.*?(?:ID\s*/\s*Passport\s*No\.?|ID\s*/?P\w*|"
        r"身份[證证詮]\s*/?\s*(?:護照)?號碼)",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" :：/")
    if DATE_TOKEN_RE.search(value) or not any(character.isdigit() for character in value):
        return ""
    match = re.search(
        r"/?[A-Za-z]?[A-Za-z0-9/.-]*\d[A-Za-z0-9/().-]*",
        value,
    )
    if match is None:
        return ""
    candidate = match.group(0).lstrip("/")
    return candidate if 4 <= len(_normalized(candidate)) <= 24 else ""


def _signature_label(
    lines: Sequence[SemanticLine],
    pattern: re.Pattern[str],
    fallback: str,
) -> str:
    matches = [line.text.strip().rstrip(":：") for line in lines if pattern.search(line.text)]
    if not matches:
        return fallback
    return min(matches, key=len)


def _signature_identifier_quality(value: str) -> int:
    compact = value.strip()
    score = int("/" not in compact and "." not in compact)
    score += int(compact.count("(") == compact.count(")"))
    score += 3 * int(
        re.fullmatch(r"[A-Za-z]\d{6}\(\d\)", compact) is not None
    )
    return score


def _render_signature_table(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> str | None:
    lines = _table_lines(table, excluded_keys)
    policy_anchors = [line for line in lines if POLICY_OWNER_SIGNATURE_RE.search(line.text)]
    insured_anchors = [line for line in lines if INSURED_SIGNATURE_RE.search(line.text)]
    if not policy_anchors or not insured_anchors:
        return None
    anchors = [
        min(policy_anchors, key=lambda line: line.center_y),
        min(insured_anchors, key=lambda line: line.center_y),
    ]
    anchors.sort(key=lambda line: line.center_y)
    zone_top = min(anchor.bbox[1] for anchor in anchors) - 8.0
    pre_lines = [line for line in lines if line.center_y < zone_top]
    zone_lines = [line for line in lines if line.center_y >= zone_top]
    global_id_lefts = [
        line.bbox[0]
        for line in zone_lines
        if re.search(r"\bID\s*/\s*Passport", line.text, re.IGNORECASE)
    ]
    global_date_lefts = [
        line.bbox[0]
        for line in zone_lines
        if re.search(r"\bDate\s*\(", line.text, re.IGNORECASE)
    ]
    global_id_left = (
        statistics.median(global_id_lefts) if global_id_lefts else float("inf")
    )
    global_date_left = (
        statistics.median(global_date_lefts)
        if global_date_lefts
        else float("inf")
    )

    parts: list[str] = []
    seen: list[str] = []
    for line in _pair_form_fields(pre_lines):
        rendered = _render_form_line(line.text)
        if rendered is not None:
            _append_unique_output(parts, seen, rendered)

    row_bounds = _signature_row_bounds(anchors, zone_lines)
    parsed_rows = []
    for anchor, (row_top, row_bottom) in zip(anchors, row_bounds):
        row_lines = [
            line
            for line in zone_lines
            if row_top <= line.center_y < row_bottom
        ]
        id_left = global_id_left
        date_left = global_date_left
        name_labels = [line for line in row_lines if BLOCK_NAME_LABEL_RE.search(line.text)]
        name_values = []
        id_values = []
        dates = []
        for line in row_lines:
            if not (
                POLICY_OWNER_SIGNATURE_RE.search(line.text)
                or INSURED_SIGNATURE_RE.search(line.text)
                or BLOCK_NAME_LABEL_RE.search(line.text)
            ):
                name = _signature_name_value(line.text)
                if name and line.center_x < id_left:
                    name_values.append((abs(line.center_y - anchor.center_y), line.bbox[0], name))
            identifier = _signature_identifier_value(line.text)
            if identifier and line.center_x >= id_left - 20.0 and line.center_x < date_left:
                id_values.append((abs(line.center_y - anchor.center_y), identifier))
            for date in DATE_TOKEN_RE.findall(line.text):
                dates.append((abs(line.center_y - anchor.center_y), _clean_text(date)))
        parsed_rows.append(
            {
                "anchor": anchor,
                "row_lines": row_lines,
                "name_label": _signature_label(
                    name_labels,
                    BLOCK_NAME_LABEL_RE,
                    "Name (in block letters)",
                ),
                "name": min(name_values, default=(0.0, 0.0, ""))[2],
                "id_label": "ID / Passport No. / 身份證/護照號碼",
                "identifier": min(id_values, default=(0.0, ""))[1],
                "date_label": "Date (DD/MM/YY) / 日期(日/月/年)",
                "date": min(dates, default=(0.0, ""))[1],
            }
        )

    known_dates = [row["date"] for row in parsed_rows if row["date"]]
    if len(set(known_dates)) == 1:
        for row in parsed_rows:
            if not row["date"]:
                row["date"] = known_dates[0]

    identifiers = [row["identifier"] for row in parsed_rows]
    if len(identifiers) == 2 and all(identifiers):
        normalized = [_normalized(value) for value in identifiers]
        if (
            normalized[0][:1] == normalized[1][:1]
            and SequenceMatcher(None, *normalized).ratio() >= 0.84
        ):
            best = max(identifiers, key=_signature_identifier_quality)
            if _signature_identifier_quality(best) > min(
                _signature_identifier_quality(value) for value in identifiers
            ):
                for row in parsed_rows:
                    row["identifier"] = best

    for row in parsed_rows:
        signature_label = row["anchor"].text.strip().rstrip(":：")
        parts.append(f"- **{signature_label}**")
        name = f"- **{row['name_label']}**"
        if row["name"]:
            name += f": {row['name']}"
        parts.append(name)
        identifier = f"- **{row['id_label']}**"
        if row["identifier"]:
            identifier += f": {row['identifier']}"
        parts.append(identifier)
        date = f"- **{row['date_label']}**"
        if row["date"]:
            date += f": {row['date']}"
        parts.append(date)
    return "\n\n".join(parts)


def _medical_grid_category(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) > 100:
        return None
    patterns = (
        ("recovery", r"%\s*of\s*recovery|康復程度"),
        ("conditions", r"Conditions?\s*/\s*Impairment|情況\s*/?\s*身體缺陷"),
        (
            "medical_treatment",
            r"Medication\s*/\s*Treatment\s*/\s*Operation|藥物\s*/\s*治療\s*/\s*手術",
        ),
        (
            "investigation_result",
            r"Investigation\s*/\s*Result|檢查\s*/\s*結果",
        ),
        ("test", r"Test\s*/\s*Investigation|化驗\s*/\s*檢查"),
        ("result", r"^Result\b|^結果\b"),
        ("treatment", r"^Treatment\b|^治療\b"),
        ("date", r"^Date\s*日期\s*[:：]?$"),
    )
    matches = [
        category
        for category, pattern in patterns
        if re.search(pattern, stripped, re.IGNORECASE)
    ]
    return matches[0] if len(matches) == 1 else None


def _medical_grid_header(
    lines: Sequence[SemanticLine],
) -> list[tuple[str, SemanticLine]] | None:
    candidates = [
        (category, line)
        for line in lines
        for category in [_medical_grid_category(line.text)]
        if category is not None
    ]
    best: tuple[int, float, list[tuple[str, SemanticLine]]] | None = None
    for _category, seed in candidates:
        cluster = [
            item
            for item in candidates
            if abs(item[1].center_y - seed.center_y) <= 18.0
        ]
        by_category: dict[str, SemanticLine] = {}
        for category, line in cluster:
            current = by_category.get(category)
            if current is None or abs(line.center_y - seed.center_y) < abs(
                current.center_y - seed.center_y
            ):
                by_category[category] = line
        categories = frozenset(by_category)
        if categories not in MEDICAL_GRID_SCHEMAS:
            continue
        ordered = sorted(by_category.items(), key=lambda item: item[1].bbox[0])
        candidate = (len(ordered), -seed.center_y, ordered)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else None


def _medical_data_rows(
    lines: Sequence[SemanticLine],
) -> list[list[SemanticLine]]:
    visual_rows = _group_visual_rows(lines)
    if not visual_rows:
        return []
    result: list[list[SemanticLine]] = []
    current: list[SemanticLine] = []
    current_bottom = 0.0
    for visual_row in visual_rows:
        row_top = min(line.bbox[1] for line in visual_row)
        row_bottom = max(line.bbox[3] for line in visual_row)
        if current and row_top - current_bottom > 24.0:
            result.append(current)
            current = []
        current.extend(visual_row)
        current_bottom = max(current_bottom, row_bottom) if current else row_bottom
    if current:
        result.append(current)
    return result


def _render_plain_form_lines(lines: Sequence[SemanticLine]) -> list[str]:
    parts: list[str] = []
    seen: list[str] = []
    for line in _pair_form_fields(lines):
        rendered = _render_form_line(line.text)
        if rendered is not None:
            _append_unique_output(parts, seen, rendered)
    return parts


def _physician_footer_value_lines(
    row_lines: Sequence[SemanticLine],
    *,
    left: float,
    right: float | None = None,
) -> list[SemanticLine]:
    result = []
    for line in row_lines:
        if line.center_x < left or (right is not None and line.center_x >= right):
            continue
        if (
            re.search(
                r"^\s*(?:Signed|Qualifications|Date|Name of physician|Address|"
                r"Telephone Number)\b|^(?:簽名|資歷|日期|醫生的姓名|地址|電話號碼)",
                line.text,
                re.IGNORECASE,
            )
            or LABEL_QUALIFIER_RE.fullmatch(line.text.strip())
            or FOOTER_RE.search(line.text)
            or (
                len(_normalized(line.text)) <= 1
                and line.bbox[2] - line.bbox[0] <= max(20.0, line.height * 2.5)
            )
        ):
            continue
        result.append(line)
    return result


def _render_physician_footer_lines(
    lines: Sequence[SemanticLine],
) -> list[str] | None:
    signed = [line for line in lines if re.search(r"^Signed\b|^簽名", line.text, re.IGNORECASE)]
    qualifications = [
        line
        for line in lines
        if re.search(r"^Qualifications\b|^資歷", line.text, re.IGNORECASE)
    ]
    dates = [
        line
        for line in lines
        if re.search(r"^Date\s*日期", line.text, re.IGNORECASE)
    ]
    physician_labels = [
        line
        for line in lines
        if re.search(r"Name of physician|醫生的姓名", line.text, re.IGNORECASE)
    ]
    address_labels = [
        line for line in lines if re.search(r"^Address\b|^地址", line.text, re.IGNORECASE)
    ]
    telephone_labels = [
        line
        for line in lines
        if re.search(r"Telephone Number|電話號碼", line.text, re.IGNORECASE)
    ]
    if not all(
        (
            signed,
            qualifications,
            dates,
            physician_labels,
            address_labels,
            telephone_labels,
        )
    ):
        return None
    label_patterns = re.compile(
        r"^\s*(?:Signed|Qualifications|Date|Name of physician|Address|"
        r"Telephone Number)\b|^(?:簽名|資歷|日期|醫生的姓名|地址|電話號碼)",
        re.IGNORECASE,
    )
    all_labels = [
        *signed,
        *qualifications,
        *dates,
        *physician_labels,
        *address_labels,
        *telephone_labels,
    ]
    label_keys = {
        (line.cell_row, line.cell_col)
        for line in all_labels
        if line.cell_row is not None and line.cell_col is not None
    }
    use_cell_layout = bool(
        len(label_keys) >= 4
        and len({row for row, _col in label_keys}) >= 2
    )
    if use_cell_layout:
        label_ids = {id(line) for line in all_labels}

        def related_values(field_labels: Sequence[SemanticLine]) -> list[SemanticLine]:
            scored: list[tuple[int, float, SemanticLine]] = []
            for line in lines:
                if (
                    id(line) in label_ids
                    or label_patterns.search(line.text)
                    or LABEL_QUALIFIER_RE.fullmatch(line.text.strip())
                    or re.fullmatch(r"\(?\s*with stamp\s*\)?", line.text, re.IGNORECASE)
                    or line.cell_row is None
                    or line.cell_col is None
                ):
                    continue
                scores: list[tuple[int, float]] = []
                for key, label in (
                    ((candidate.cell_row, candidate.cell_col), candidate)
                    for candidate in field_labels
                    if candidate.cell_row is not None
                    and candidate.cell_col is not None
                ):
                    column_delta = line.cell_col - key[1]
                    row_delta = abs(line.cell_row - key[0])
                    right_boundaries = [
                        other.cell_col
                        for other in all_labels
                        if other.cell_row == key[0]
                        and other.cell_col is not None
                        and other.cell_col > key[1]
                    ]
                    right_boundary = min(right_boundaries, default=None)
                    relation = (
                        0
                        if (line.cell_row, line.cell_col) == key
                        else 1
                        if (
                            row_delta == 0
                            and 0 < column_delta <= 2
                            and (
                                right_boundary is None
                                or line.cell_col < right_boundary
                            )
                        )
                        else 2
                        if (
                            row_delta == 1
                            and 0 < column_delta <= 2
                            and (
                                right_boundary is None
                                or line.cell_col < right_boundary
                            )
                        )
                        else 99
                    )
                    if relation < 99:
                        scores.append(
                            (relation, abs(line.center_y - label.center_y))
                        )
                score = min(scores, default=(99, float("inf")))
                if score[0] < 99:
                    scored.append((score[0], score[1], line))
            if not scored:
                return []
            if any(relation <= 1 for relation, _distance, _line in scored):
                scored = [
                    item for item in scored if item[0] <= 1
                ]
            minimum_distance = min(distance for _relation, distance, _line in scored)
            typical_height = statistics.median(
                line.height for _relation, _distance, line in scored
            )
            maximum_distance = minimum_distance + max(12.0, typical_height)
            return _deduplicate_lines(
                [
                    line
                    for _relation, distance, line in sorted(
                        scored,
                        key=lambda item: (
                            item[2].bbox[1],
                            item[0],
                            item[2].bbox[0],
                        ),
                    )
                    if distance <= maximum_distance
                ]
            )

        grid_top = min(
            line.bbox[1]
            for line in (
                *qualifications,
                *dates,
                *physician_labels,
                *address_labels,
                *telephone_labels,
            )
        )
        prefix = [
            line
            for line in lines
            if line.center_y < grid_top and id(line) not in label_ids
        ]
        parts = _render_plain_form_lines(prefix)
        physician_values = related_values(physician_labels)
        qualification_values = _deduplicate_lines(
            [
                *related_values(qualifications),
                *[
                    line
                    for line in lines
                    if line.center_y >= grid_top
                    and MEDICAL_QUALIFICATION_RE.search(line.text)
                    and id(line) not in label_ids
                ],
            ]
        )
        address_values = [
            line
            for line in related_values(address_labels)
            if MEDICAL_QUALIFICATION_RE.search(line.text) is None
            and re.search(r"^\s*(?:Tel(?:ephone)?|Fax)\s*[:：]", line.text, re.IGNORECASE)
            is None
        ]
        date_values = related_values(dates)
        telephone_values = related_values(telephone_labels)
        physician_names = [
            value
            for line in physician_values
            for value in [_signature_name_value(line.text)]
            if value
        ]
        signature_left = min(line.bbox[0] for line in physician_labels)
        physician_rows = {
            line.cell_row for line in physician_labels if line.cell_row is not None
        }
        physician_columns = [
            line.cell_col for line in physician_labels if line.cell_col is not None
        ]
        first_physician_column = min(physician_columns, default=0)
        signature_present = any(
            (
                (
                    line.cell_row in physician_rows
                    and line.cell_col is not None
                    and line.cell_col < first_physician_column
                )
                or (
                    line.center_x < signature_left
                    and line.bbox[1] >= min(anchor.bbox[1] for anchor in signed)
                    and line.bbox[3]
                    <= max(
                        anchor.cell_bbox[3]
                        if anchor.cell_bbox
                        else anchor.bbox[3]
                        for anchor in physician_labels
                    )
                )
            )
            and id(line) not in label_ids
            and not LABEL_QUALIFIER_RE.fullmatch(line.text.strip())
            and len(_normalized(line.text)) <= 20
            for line in lines
        )
        parts.append(
            f"- **{min(signed, key=lambda line: line.center_y).text.strip().rstrip(':：')}**"
            + (": [Signature]" if signature_present else "")
        )
        parts.append(
            "- **Name of physician (with stamp) / 醫生的姓名(蓋印)**"
            + (
                f": {' / '.join(dict.fromkeys(physician_names))}"
                if physician_names
                else ""
            )
        )
        qualification_text = _join_field_values(qualification_values)
        parts.append(
            "- **Qualifications / 資歷**"
            + (f": {qualification_text}" if qualification_text else "")
        )
        address_text = _join_field_values(address_values)
        parts.append(
            "- **Address 地址**" + (f": {address_text}" if address_text else "")
        )
        date_text = next(
            (
                _clean_text(match.group(0))
                for line in date_values
                for match in [DATE_TOKEN_RE.search(line.text)]
                if match is not None
            ),
            _join_field_values(date_values),
        )
        parts.append(
            "- **Date / 日期**" + (f": {date_text}" if date_text else "")
        )
        telephone_text = _join_field_values(telephone_values)
        parts.append(
            "- **Telephone Number / 電話號碼**"
            + (f": {telephone_text}" if telephone_text else "")
        )
        return parts
    anchors = [
        min(signed, key=lambda line: line.center_y),
        min(qualifications, key=lambda line: line.center_y),
        min(dates, key=lambda line: line.center_y),
    ]
    anchors.sort(key=lambda line: line.center_y)
    zone_top = anchors[0].bbox[1] - 6.0
    prefix = [line for line in lines if line.center_y < zone_top]
    zone_lines = [line for line in lines if line.center_y >= zone_top]
    right_label_left = statistics.median(
        line.bbox[0]
        for line in (*physician_labels, *address_labels, *telephone_labels)
    )
    right_value_left = max(
        right_label_left + 75.0,
        statistics.median(
            line.bbox[2]
            for line in (*physician_labels, *address_labels, *telephone_labels)
        ),
    )
    left_value_left = max(anchor.bbox[2] for anchor in anchors) - 4.0
    bounds = _signature_row_bounds(anchors, zone_lines)
    parsed = []
    for anchor, (row_top, row_bottom) in zip(anchors, bounds):
        row_lines = [line for line in zone_lines if row_top <= line.center_y < row_bottom]
        left_values = _physician_footer_value_lines(
            row_lines,
            left=left_value_left,
            right=right_label_left,
        )
        right_values = _physician_footer_value_lines(
            row_lines,
            left=right_value_left,
        )
        parsed.append((anchor, left_values, right_values))

    parts = _render_plain_form_lines(prefix)
    signed_anchor, signed_values, physician_values = parsed[0]
    signature_present = bool(signed_values)
    parts.append(
        f"- **{signed_anchor.text.strip().rstrip(':：')}**"
        + (": [Signature]" if signature_present else "")
    )
    physician_names = [
        value
        for line in physician_values
        for value in [_signature_name_value(line.text)]
        if value
    ]
    parts.append(
        "- **Name of physician (with stamp) / 醫生的姓名(蓋印)**"
        + (f": {physician_names[0]}" if physician_names else "")
    )

    qualifications_anchor, qualification_values, address_values = parsed[1]
    qualification_text = _join_field_values(qualification_values)
    parts.append(
        "- **Qualifications / 資歷**"
        + (f": {qualification_text}" if qualification_text else "")
    )
    address_text = _join_field_values(address_values)
    parts.append(
        "- **Address 地址**" + (f": {address_text}" if address_text else "")
    )

    date_anchor, date_values, telephone_values = parsed[2]
    date_text = next(
        (
            _clean_text(match.group(0))
            for line in date_values
            for match in [DATE_TOKEN_RE.search(line.text)]
            if match is not None
        ),
        _join_field_values(date_values),
    )
    parts.append(
        "- **Date / 日期**" + (f": {date_text}" if date_text else "")
    )
    telephone_text = _join_field_values(telephone_values)
    parts.append(
        "- **Telephone Number / 電話號碼**"
        + (f": {telephone_text}" if telephone_text else "")
    )
    return parts


def _render_generic_form_lines(lines: Sequence[SemanticLine]) -> list[str]:
    physician_footer = _render_physician_footer_lines(lines)
    if physician_footer is not None:
        return physician_footer
    return _render_plain_form_lines(lines)


def _render_medical_grid_table(
    table: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> str | None:
    lines = _table_lines(table, excluded_keys)
    header = _medical_grid_header(lines)
    if header is None:
        return None
    header_indices = {id(line) for _category, line in header}
    header_top = min(line.bbox[1] for _category, line in header)
    header_bottom = max(line.bbox[3] for _category, line in header)
    left_anchor = min(line.bbox[0] for _category, line in header)
    boundary_candidates = []
    for line in lines:
        stripped = line.text.strip()
        if FORM_PROMPT_RE.match(stripped) is None:
            continue
        numbered_question = bool(
            QUESTION_PROMPT_RE.search(FORM_PROMPT_RE.sub("", stripped))
        )
        starts_below_header = line.bbox[1] > header_bottom + 6.0
        spanning_question = bool(
            numbered_question
            and line.bbox[1] <= header_bottom + 6.0
            and line.bbox[3] > header_bottom + 30.0
        )
        if (
            starts_below_header
            and (line.bbox[0] <= left_anchor + 35.0 or numbered_question)
        ) or spanning_question:
            boundary_candidates.append(line)
    boundary_positions = {
        id(line): (
            line.bbox[1]
            if line.bbox[1] > header_bottom + 6.0
            else line.bbox[3]
        )
        for line in boundary_candidates
    }
    boundary_top = min(
        boundary_positions.values(),
        default=float("inf"),
    )
    boundary_ids = {id(line) for line in boundary_candidates}
    spanning_boundary = [
        line
        for line in boundary_candidates
        if line.bbox[1] < boundary_top
        and boundary_positions[id(line)] == boundary_top
    ]
    spanning_cells = {
        (line.cell_row, line.cell_col)
        for line in spanning_boundary
        if line.cell_row is not None and line.cell_col is not None
    }
    spanning_companions = [
        line
        for line in lines
        if id(line) not in boundary_ids
        and (line.cell_row, line.cell_col) in spanning_cells
        and _uses_cell_bbox(line)
        and _medical_grid_category(line.text) is None
    ]
    spanning_ids = {
        id(line) for line in (*spanning_boundary, *spanning_companions)
    }
    prefix = [line for line in lines if line.center_y < header_top]
    data_lines = [
        line
        for line in lines
        if id(line) not in header_indices
        and id(line) not in boundary_ids
        and id(line) not in spanning_ids
        and line.center_y > header_bottom
        and line.bbox[1] < boundary_top
        and not FOOTER_RE.search(line.text)
        and not NOISE_TEXT_RE.fullmatch(line.text)
        and not (
            ISOLATED_LIST_MARKER_RE.fullmatch(line.text.strip())
            and line.bbox[2] - line.bbox[0] <= max(20.0, line.height * 2.5)
        )
    ]
    suffix = _deduplicate_lines(
        [*spanning_boundary, *spanning_companions]
        + [
            line
            for line in lines
            if line.bbox[1] >= boundary_top
            and id(line) not in spanning_ids
        ]
    )
    if not data_lines:
        return None

    headers = [line.text.strip() for _category, line in header]
    column_lefts = [line.bbox[0] for _category, line in header]
    rows = []
    for row_lines in _medical_data_rows(data_lines):
        columns: list[list[SemanticLine]] = [[] for _header in header]
        for line in row_lines:
            column = min(
                range(len(header)),
                key=lambda index: abs(line.bbox[0] - column_lefts[index]),
            )
            columns[column].append(line)
        rows.append(
            [
                "\n".join(
                    line.text.strip()
                    for line in sorted(column, key=lambda item: (item.bbox[1], item.bbox[0]))
                    if line.text.strip()
                )
                for column in columns
            ]
        )

    parts = _render_generic_form_lines(prefix)
    parts.append(_markdown_table(headers, rows))
    parts.extend(_render_generic_form_lines(suffix))
    return "\n\n".join(part for part in parts if part)


def _render_form_cell(
    cell: Mapping[str, Any],
    excluded_keys: set[str] | None = None,
) -> list[str]:
    lines = [
        line.text
        for line in _merge_control_markers(
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
    signature = _render_signature_table(table, excluded_keys)
    if signature is not None:
        return signature
    medical_grid = _render_medical_grid_table(table, excluded_keys)
    if medical_grid is not None:
        return medical_grid
    lines = _merge_control_markers(_table_lines(table, excluded_keys))
    return "\n\n".join(_render_generic_form_lines(lines))


def _repeatable_value(text: str) -> bool:
    normalized = _normalized(text)
    return bool(
        any(character.isdigit() for character in text)
        and len(normalized) <= 32
        and len(text.split()) <= 5
    )


def _direct_block_lines(block: Mapping[str, Any]) -> list[SemanticLine]:
    """Return only the lines owned directly by one layout block."""
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
            text = _checkbox_text(span, _span_text(span))
            if text:
                parts.append(text)
        text = " ".join(parts).strip()
        if bbox is not None and text:
            result.append(SemanticLine(bbox, text))
    return result


def _direct_semantic_lines(block: Mapping[str, Any]) -> list[SemanticLine]:
    return _merge_control_markers(
        _reading_order_lines(_deduplicate_lines(_direct_block_lines(block)))
    )


def _block_lines(block: Mapping[str, Any]) -> list[SemanticLine]:
    """Collect visible text recursively from ordinary or demoted blocks.

    A page-sized Table can be demoted after OCR and retain its evidence in
    nested ``blocks`` instead of ``table_cells``.  Treating such a container as
    empty silently loses the page, so nested layout blocks are a conservative
    geometry-backed fallback.  Each mapping is visited once to avoid aggregate
    and child traversal duplicating the same object.
    """
    result: list[SemanticLine] = []
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if not isinstance(item, Mapping) or id(item) in visited:
            return
        visited.add(id(item))
        result.extend(_direct_block_lines(item))
        raw_blocks = item.get("blocks", [])
        if isinstance(raw_blocks, list):
            for child in raw_blocks:
                visit(child)

    visit(block)
    return _merge_control_markers(_reading_order_lines(_deduplicate_lines(result)))


def _equivalent_margin_text(left: str, right: str) -> bool:
    left_key = _normalized(left)
    right_key = _normalized(right)
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) < 20 or len(shorter) / len(longer) < 0.75:
        return False
    left_numbers = re.findall(r"\d+(?:[.,:/-]\d+)*", left)
    right_numbers = re.findall(r"\d+(?:[.,:/-]\d+)*", right)
    if left_numbers != right_numbers:
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.88


def _margin_repetitions(pages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    occurrences: dict[tuple[str, str], tuple[str, set[int]]] = {}
    for page_index, page in enumerate(pages):
        page_size = page.get("page_size", [])
        height = (
            float(page_size[1])
            if isinstance(page_size, (list, tuple)) and len(page_size) >= 2
            else 0.0
        )
        if height <= 0:
            continue
        seen_tables: set[int] = set()
        for block in _layout_blocks(page.get("preproc_blocks", [])):
            lines = list(_direct_semantic_lines(block))
            for table in _direct_table_spans(block):
                if id(table) in seen_tables:
                    continue
                seen_tables.add(id(table))
                lines.extend(_table_lines(table))
            for line in _deduplicate_lines(lines):
                side = (
                    "top"
                    if line.bbox[1] <= height * 0.1
                    else "bottom"
                    if line.bbox[3] >= height * 0.82
                    else ""
                )
                key = _normalized(line.text)
                if not side or len(key) < 5:
                    continue
                occurrence_key = (side, key)
                if occurrence_key not in occurrences:
                    occurrences[occurrence_key] = (line.text, set())
                occurrences[occurrence_key][1].add(page_index)

    nodes = list(occurrences)
    parents = list(range(len(nodes)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = root(left_index)
        right_root = root(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, (left_side, left_key) in enumerate(nodes):
        left_text, left_pages = occurrences[(left_side, left_key)]
        for right_index in range(left_index + 1, len(nodes)):
            right_side, right_key = nodes[right_index]
            if left_side != right_side:
                continue
            right_text, right_pages = occurrences[(right_side, right_key)]
            if len(left_pages | right_pages) < 2:
                continue
            if _equivalent_margin_text(left_text, right_text):
                union(left_index, right_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(nodes)):
        clusters[root(index)].append(index)
    result: dict[str, str] = {}
    for indices in clusters.values():
        page_ids = set().union(
            *(occurrences[nodes[index]][1] for index in indices)
        )
        if len(page_ids) < 2:
            continue
        keys = sorted(nodes[index][1] for index in indices)
        canonical = keys[0]
        for key in keys:
            result[key] = canonical
    return result


def _fragment_noise_page(parts: Sequence[str]) -> bool:
    lines = [
        line.strip()
        for part in parts
        for line in part.splitlines()
        if line.strip() and not line.lstrip().startswith("<!--")
    ]
    if len(lines) < 12:
        return False
    if any(
        line.startswith(("#", "- **", "- ["))
        or line.startswith("| ---")
        or DATE_TOKEN_RE.search(line)
        or re.search(r"https?://", line, re.IGNORECASE)
        for line in lines
    ):
        return False
    fragment_count = sum(len(_normalized(line)) <= 8 for line in lines)
    lexical_count = sum(
        len(_normalized(line)) >= 18
        or len(re.findall(r"[A-Za-z]{3,}", line)) >= 3
        or len(re.findall(r"[\u3400-\u9fff]", line)) >= 8
        for line in lines
    )
    return fragment_count / len(lines) >= 0.75 and lexical_count <= 2


def _page_noise_lines(parts: Sequence[str]) -> list[str]:
    """Keep probable OCR noise auditable without discarding an entire page."""
    return [
        line.strip()
        for part in parts
        for line in part.splitlines()
        if line.strip() and not line.lstrip().startswith("<!--")
    ]


def _sort_page_segments(
    segments: Sequence[
        tuple[float, float, str, tuple[float, float, float, float]]
    ],
) -> list[tuple[float, float, str, tuple[float, float, float, float]]]:
    """Sort page segments by tolerant top rows and then left position."""
    if not segments:
        return []
    heights = [
        bbox[3] - bbox[1]
        for _top, _left, _text, bbox in segments
        if bbox[3] > bbox[1]
    ]
    typical_height = statistics.median(heights) if heights else 8.0
    tolerance = max(2.0, min(typical_height, 24.0) * 0.45)
    rows: list[list[tuple[float, float, str, tuple[float, float, float, float]]]] = []
    for segment in sorted(segments, key=lambda item: (item[0], item[1])):
        best_index = None
        best_distance = float("inf")
        for index, row in enumerate(rows[-4:]):
            actual_index = len(rows) - len(rows[-4:]) + index
            row_top = statistics.mean(item[0] for item in row)
            distance = abs(segment[0] - row_top)
            if distance <= tolerance and distance < best_distance:
                best_index = actual_index
                best_distance = distance
        if best_index is None:
            rows.append([segment])
        else:
            rows[best_index].append(segment)
    return [
        segment
        for row in rows
        for segment in sorted(row, key=lambda item: (item[1], item[0]))
    ]


def generate_semantic_markdown(middle_json: Mapping[str, Any]) -> str:
    """Build a conservative alternative Markdown view from fused geometry."""
    pages = middle_json.get("pdf_info", [])
    if not isinstance(pages, list):
        raise ValueError("middle_json must contain a pdf_info list")
    repeated_margins = _margin_repetitions(pages)
    emitted_margins = set()
    seen_ledger_preheaders: set[str] = set()
    seen_ledger_tail: list[str] = []
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
        segments: list[
            tuple[float, float, str, tuple[float, float, float, float]]
        ] = []
        seen_tables: set[int] = set()
        for block in _layout_blocks(page.get("preproc_blocks", [])):
            block_bbox = _valid_bbox(block.get("bbox")) or (0.0, 0.0, 0.0, 0.0)
            for table in _direct_table_spans(block):
                if id(table) in seen_tables:
                    continue
                seen_tables.add(id(table))
                table_bbox = _valid_bbox(table.get("bbox")) or block_bbox
                rendered = _render_ledger_table(
                    table,
                    set(repeated_margins),
                    seen_ledger_preheaders,
                    seen_ledger_tail,
                ) or _render_form_table(table, set(repeated_margins))
                if rendered:
                    segments.append(
                        (table_bbox[1], table_bbox[0], rendered, table_bbox)
                    )
            if block.get("type") in {"image", "chart", "interline_equation"}:
                continue
            for line in _direct_semantic_lines(block):
                text = line.text
                key = _normalized(text)
                if FOOTER_RE.search(text) or NOISE_TEXT_RE.fullmatch(text):
                    continue
                if _looks_like_formula_text(text):
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
                margin_group = repeated_margins.get(key)
                if margin_group is not None:
                    if margin_group in emitted_margins:
                        continue
                    emitted_margins.add(margin_group)
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
                    if block.get("type") in {"title", "table_caption"}
                    and (_is_section(text) or is_short_title)
                    else ""
                )
                task = _task_line(text)
                if task is not None:
                    text = task
                    prefix = ""
                segments.append(
                    (line.bbox[1], line.bbox[0], prefix + text, line.bbox)
                )
        if not segments:
            continue
        page_parts: list[str] = []
        seen: list[Any] = []
        for _top, _left, text, source_bbox in _sort_page_segments(segments):
            _append_unique_output(
                page_parts,
                seen,
                text,
                bbox=source_bbox,
            )
        if _fragment_noise_page(page_parts):
            # A noisy page is still source evidence.  Omitting it makes a
            # false-positive page classifier indistinguishable from a blank
            # page, so preserve the conservative reading-order fallback and
            # mark it for downstream/manual review.
            page_parts = [
                "<!-- semantic-warning: fragment-heavy-page -->",
                *_page_noise_lines(page_parts),
            ]
        document_parts.append(f"<!-- Page {page_index + 1} -->")
        document_parts.extend(page_parts)
    return "\n\n".join(document_parts).strip() + "\n"


def generate_semantic_markdown_report(
    middle_json: Mapping[str, Any],
    markdown: str | None = None,
) -> dict[str, Any]:
    """Summarize source evidence and conservative fallback decisions.

    The report intentionally avoids claiming character accuracy without ground
    truth.  It exposes coverage proxies and warning pages so regressions can be
    compared across arbitrary documents and reviewed against source bboxes.
    """
    rendered = (
        markdown
        if markdown is not None
        else generate_semantic_markdown(middle_json)
    )
    pages = middle_json.get("pdf_info", [])
    page_count = len(pages) if isinstance(pages, list) else 0
    emitted_page_numbers = {
        int(value) for value in re.findall(r"<!-- Page (\d+) -->", rendered)
    }
    text_bearing_page_numbers = [
        page_index + 1
        for page_index, page in enumerate(pages if isinstance(pages, list) else [])
        if isinstance(page, Mapping) and _page_has_source_text(page)
    ]
    text_bearing_page_set = set(text_bearing_page_numbers)
    source_records, source_characters = _text_source_stats(middle_json)
    emitted_records, emitted_characters = _markdown_body_stats(rendered)
    trace_records, trace_counts = _trace_source_records(middle_json, rendered)
    fragment_pages = [
        int(match.group(1))
        for match in re.finditer(
            r"<!-- Page (\d+) -->\s*\n\n"
            r"<!-- semantic-warning: fragment-heavy-page -->",
            rendered,
        )
    ]
    fallback_blocks = 0
    fallback_pages: list[int] = []
    if isinstance(pages, list):
        for page_index, page in enumerate(pages):
            if not isinstance(page, Mapping):
                continue
            page_fallbacks = 0
            for block in page.get("preproc_blocks", []):
                if not isinstance(block, Mapping):
                    continue
                if (
                    block.get("type") == "table"
                    and not _table_has_cells(_table_span(block))
                    and _block_lines(block)
                ):
                    page_fallbacks += 1
            fallback_blocks += page_fallbacks
            if page_fallbacks:
                fallback_pages.append(page_index + 1)
    return {
        "semantic_markdown_version": SEMANTIC_MARKDOWN_VERSION,
        "pages": page_count,
        "pages_emitted": len(emitted_page_numbers),
        "text_bearing_pages": len(text_bearing_page_numbers),
        "text_bearing_pages_emitted": len(
            text_bearing_page_set & emitted_page_numbers
        ),
        "text_bearing_page_numbers": text_bearing_page_numbers,
        "text_bearing_pages_not_emitted": sorted(
            text_bearing_page_set - emitted_page_numbers
        ),
        "blank_or_image_only_pages": [
            page_number
            for page_number in range(1, page_count + 1)
            if page_number not in text_bearing_page_set
        ],
        "source_text_records": source_records,
        "source_text_characters": source_characters,
        "markdown_records": emitted_records,
        "markdown_characters": emitted_characters,
        "markdown_to_source_character_ratio": (
            round(emitted_characters / source_characters, 6)
            if source_characters
            else 1.0
        ),
        "trace_accounting_ratio": (
            round(
                (len(trace_records) - trace_counts.get("unmatched", 0))
                / len(trace_records),
                6,
            )
            if trace_records
            else 1.0
        ),
        "fragment_heavy_pages": fragment_pages,
        "unstructured_table_fallback_blocks": fallback_blocks,
        "unstructured_table_fallback_pages": fallback_pages,
        "trace_counts": trace_counts,
        "unmatched_source_records": trace_counts.get("unmatched", 0),
        "source_trace": trace_records,
    }


def write_semantic_markdown(
    middle_json_path: str | Path,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
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
    markdown = generate_semantic_markdown(payload)
    output.write_text(markdown, encoding="utf-8")
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                generate_semantic_markdown_report(payload, markdown),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("middle_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    print(write_semantic_markdown(args.middle_json, args.output, args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

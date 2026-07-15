"""HTML table grid parsing, cell metadata alignment, and guarded reconstruction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable, Mapping


CellKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class TableCell:
    index: int
    tag: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    content_start: int
    content_end: int
    inner_html: str
    text: str

    @property
    def key(self) -> CellKey:
        return self.row_start, self.row_end, self.col_start, self.col_end


@dataclass(frozen=True)
class ParsedTable:
    html: str
    cells: tuple[TableCell, ...]
    row_count: int
    col_count: int
    valid: bool
    coverage_complete: bool
    errors: tuple[str, ...]
    table_start: int | None = None
    table_end: int | None = None

    @property
    def structure_signature(self) -> tuple[int, int, tuple[CellKey, ...]]:
        return (
            self.row_count,
            self.col_count,
            tuple(sorted(cell.key for cell in self.cells)),
        )

    @property
    def cell_map(self) -> dict[CellKey, TableCell]:
        return {cell.key: cell for cell in self.cells}

    def structure_report(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "coverage_complete": self.coverage_complete,
            "rows": self.row_count,
            "columns": self.col_count,
            "cells": len(self.cells),
            "grid": [list(cell.key) for cell in sorted(self.cells, key=lambda item: item.key)],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class TableCellEvidence:
    key: CellKey
    bbox: tuple[float, float, float, float] | None
    text: str
    confidence: float | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TableCellContext:
    key: CellKey
    cell_bbox: tuple[float, float, float, float]
    row_bbox: tuple[float, float, float, float] | None
    column_bbox: tuple[float, float, float, float] | None
    row_cells: tuple[dict[str, Any], ...]
    column_cells: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TableSnapshot:
    row_count: int
    col_count: int
    cells: tuple[tuple[CellKey, str], ...]

    @property
    def structure_signature(self) -> tuple[int, int, tuple[CellKey, ...]]:
        return (
            self.row_count,
            self.col_count,
            tuple(sorted(key for key, _text in self.cells)),
        )

    @property
    def cell_map(self) -> dict[CellKey, str]:
        return dict(self.cells)


@dataclass
class _PendingCell:
    tag: str
    physical_row: int
    rowspan: int
    colspan: int
    content_start: int
    content_end: int | None = None


class _TableGridParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = _line_offsets(source)
        self.table_depth = 0
        self.completed = False
        self.table_start: int | None = None
        self.table_end: int | None = None
        self.physical_row = -1
        self.in_top_level_row = False
        self.pending: list[_PendingCell] = []
        self.active: _PendingCell | None = None
        self.errors: list[str] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        if line < 1 or line > len(self.line_offsets):
            return len(self.source)
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag == "table":
            if self.completed:
                self.errors.append("multiple_top_level_tables")
                return
            if self.table_depth == 0:
                self.table_start = self._offset()
            self.table_depth += 1
            return
        if self.completed or self.table_depth != 1:
            return
        if tag == "tr":
            if self.in_top_level_row:
                self.errors.append("nested_top_level_row")
            self.physical_row += 1
            self.in_top_level_row = True
            return
        if tag not in {"td", "th"}:
            return
        if not self.in_top_level_row:
            self.errors.append("cell_outside_row")
            return
        if self.active is not None:
            self.errors.append("nested_cell")
            return
        raw_tag = self.get_starttag_text() or ""
        attributes = {name.casefold(): value for name, value in attrs}
        cell = _PendingCell(
            tag=tag,
            physical_row=self.physical_row,
            rowspan=_positive_span(attributes.get("rowspan"), "rowspan", self.errors),
            colspan=_positive_span(attributes.get("colspan"), "colspan", self.errors),
            content_start=self._offset() + len(raw_tag),
        )
        self.pending.append(cell)
        self.active = cell

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.completed:
            return
        if tag in {"td", "th"} and self.table_depth == 1:
            if self.active is None:
                self.errors.append("unmatched_cell_end")
            elif self.active.tag != tag:
                self.errors.append("mismatched_cell_end")
            else:
                self.active.content_end = self._offset()
                self.active = None
            return
        if tag == "tr" and self.table_depth == 1:
            if self.active is not None:
                self.errors.append("unclosed_cell")
                self.active = None
            self.in_top_level_row = False
            return
        if tag != "table" or self.table_depth <= 0:
            return
        self.table_depth -= 1
        if self.table_depth == 0:
            self.table_end = self._offset() + len("</table>")
            self.completed = True


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        elif not self.hidden_depth and tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif not self.hidden_depth and tag in {"p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


class _CellSafetyParser(HTMLParser):
    UNSAFE_TAGS = {
        "script",
        "style",
        "iframe",
        "object",
        "embed",
        "form",
        "input",
        "button",
        "textarea",
        "select",
        "meta",
        "base",
    }

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.safe = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in self.UNSAFE_TAGS:
            self.safe = False
        for name, value in attrs:
            lowered_name = name.casefold()
            lowered_value = (value or "").strip().casefold()
            if lowered_name.startswith("on"):
                self.safe = False
            if lowered_name in {"href", "src", "xlink:href", "formaction"} and re.match(
                r"(?:javascript|vbscript|data):",
                lowered_value,
            ):
                self.safe = False

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)


class _TableFragmentParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = _line_offsets(source)
        self.depth = 0
        self.start: int | None = None
        self.fragments: list[str] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "table":
            return
        if self.depth == 0:
            self.start = self._offset()
        self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "table" or self.depth <= 0:
            return
        self.depth -= 1
        if self.depth == 0 and self.start is not None:
            end = self._offset() + len("</table>")
            self.fragments.append(self.source[self.start:end])
            self.start = None


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r"\n", source):
        offsets.append(match.end())
    return offsets


def _positive_span(value: str | None, name: str, errors: list[str]) -> int:
    if value is None:
        return 1
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"invalid_{name}")
        return 1
    if parsed < 1 or parsed > 1000:
        errors.append(f"invalid_{name}")
        return 1
    return parsed


def visible_cell_text(value: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return ""
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def cell_content_safe(value: str) -> bool:
    parser = _CellSafetyParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return False
    return parser.safe


def parse_table_html(value: Any) -> ParsedTable:
    if not isinstance(value, str) or not value.strip():
        return ParsedTable(str(value or ""), (), 0, 0, False, False, ("empty_html",))
    parser = _TableGridParser(value)
    try:
        parser.feed(value)
        parser.close()
    except Exception as exc:
        return ParsedTable(value, (), 0, 0, False, False, (type(exc).__name__,))

    errors = list(parser.errors)
    if parser.table_start is None:
        errors.append("missing_table")
    if not parser.completed:
        errors.append("unclosed_table")
    if parser.active is not None or any(item.content_end is None for item in parser.pending):
        errors.append("unclosed_cell")

    occupied: set[tuple[int, int]] = set()
    cells: list[TableCell] = []
    for pending in parser.pending:
        if pending.content_end is None:
            continue
        col_start = 0
        while (pending.physical_row, col_start) in occupied:
            col_start += 1
        row_end = pending.physical_row + pending.rowspan - 1
        col_end = col_start + pending.colspan - 1
        coordinates = {
            (row, column)
            for row in range(pending.physical_row, row_end + 1)
            for column in range(col_start, col_end + 1)
        }
        if occupied & coordinates:
            errors.append("overlapping_cells")
            continue
        occupied.update(coordinates)
        inner_html = value[pending.content_start:pending.content_end]
        cells.append(
            TableCell(
                index=len(cells),
                tag=pending.tag,
                row_start=pending.physical_row,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                content_start=pending.content_start,
                content_end=pending.content_end,
                inner_html=inner_html,
                text=visible_cell_text(inner_html),
            )
        )

    row_count = max((cell.row_end + 1 for cell in cells), default=0)
    row_count = max(row_count, parser.physical_row + 1)
    col_count = max((cell.col_end + 1 for cell in cells), default=0)
    expected = {(row, column) for row in range(row_count) for column in range(col_count)}
    coverage_complete = bool(expected) and occupied == expected
    if expected and not coverage_complete:
        errors.append("incomplete_grid")
    if not cells:
        errors.append("missing_cells")
    fatal_errors = {
        "missing_table",
        "unclosed_table",
        "unclosed_cell",
        "cell_outside_row",
        "nested_cell",
        "mismatched_cell_end",
        "overlapping_cells",
        "missing_cells",
        "invalid_rowspan",
        "invalid_colspan",
        "unmatched_cell_end",
        "nested_top_level_row",
        "multiple_top_level_tables",
    }
    valid = not any(error in fatal_errors for error in errors)
    return ParsedTable(
        html=value,
        cells=tuple(cells),
        row_count=row_count,
        col_count=col_count,
        valid=valid,
        coverage_complete=coverage_complete,
        errors=tuple(dict.fromkeys(errors)),
        table_start=parser.table_start,
        table_end=parser.table_end,
    )


def align_table_cells(
    hybrid: ParsedTable,
    pipeline: ParsedTable,
) -> list[tuple[TableCell, TableCell]] | None:
    if not hybrid.valid or not pipeline.valid:
        return None
    if not hybrid.coverage_complete or not pipeline.coverage_complete:
        return None
    if hybrid.structure_signature != pipeline.structure_signature:
        return None
    pipeline_cells = pipeline.cell_map
    return [(cell, pipeline_cells[cell.key]) for cell in hybrid.cells]


def rebuild_table_html(
    baseline: ParsedTable,
    replacements: Mapping[CellKey, str],
) -> tuple[str | None, str | None]:
    if not baseline.valid:
        return None, "invalid_baseline"
    known = baseline.cell_map
    if any(key not in known for key in replacements):
        return None, "unknown_cell"
    rebuilt = baseline.html
    cells = sorted(
        (known[key] for key in replacements),
        key=lambda cell: cell.content_start,
        reverse=True,
    )
    for cell in cells:
        content = replacements[cell.key]
        if not isinstance(content, str):
            return None, "non_string_content"
        if not cell_content_safe(content):
            return None, "unsafe_cell_content"
        rebuilt = rebuilt[:cell.content_start] + content + rebuilt[cell.content_end:]
    reparsed = parse_table_html(rebuilt)
    if not reparsed.valid:
        return None, "invalid_rebuilt_html"
    if reparsed.structure_signature != baseline.structure_signature:
        return None, "structure_changed"
    return rebuilt, None


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


def _confidence(item: Mapping[str, Any]) -> float | None:
    for key in ("score", "confidence", "ocr_confidence"):
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, min(1.0, float(value)))
    return None


def collect_table_cell_evidence(span: Mapping[str, Any]) -> dict[CellKey, TableCellEvidence]:
    raw_cells = span.get("table_cells", [])
    if not isinstance(raw_cells, list):
        return {}
    result: dict[CellKey, TableCellEvidence] = {}
    duplicates: set[CellKey] = set()
    for item in raw_cells:
        if not isinstance(item, dict):
            continue
        try:
            key = tuple(
                int(item[name])
                for name in ("row_start", "row_end", "col_start", "col_end")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if key[0] < 0 or key[2] < 0 or key[1] < key[0] or key[3] < key[2]:
            continue
        if key in result:
            duplicates.add(key)
            continue
        text = item.get("text", "")
        result[key] = TableCellEvidence(
            key=key,
            bbox=_valid_bbox(item.get("bbox")),
            text=text if isinstance(text, str) else "",
            confidence=_confidence(item),
            raw=item,
        )
    for key in duplicates:
        result.pop(key, None)
    return result


def _union_bbox(items: Iterable[TableCellEvidence]) -> tuple[float, float, float, float] | None:
    boxes = [item.bbox for item in items if item.bbox is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _context_item(
    cell: TableCell,
    hybrid: TableCell,
    pipeline: TableCell,
) -> dict[str, Any]:
    return {
        "row_start": cell.row_start,
        "row_end": cell.row_end,
        "col_start": cell.col_start,
        "col_end": cell.col_end,
        "hybrid": hybrid.text,
        "pipeline": pipeline.text,
    }


def build_table_cell_context(
    target: TableCell,
    hybrid: ParsedTable,
    pipeline: ParsedTable,
    evidence: Mapping[CellKey, TableCellEvidence],
) -> TableCellContext | None:
    target_evidence = evidence.get(target.key)
    if target_evidence is None or target_evidence.bbox is None:
        return None
    hybrid_map = hybrid.cell_map
    pipeline_map = pipeline.cell_map
    row_cells = [
        cell
        for cell in hybrid.cells
        if cell.row_start <= target.row_start <= cell.row_end
    ]
    column_cells = [
        cell
        for cell in hybrid.cells
        if cell.col_start <= target.col_start <= cell.col_end
    ]
    row_evidence = [evidence[cell.key] for cell in row_cells if cell.key in evidence]
    column_evidence = [evidence[cell.key] for cell in column_cells if cell.key in evidence]
    return TableCellContext(
        key=target.key,
        cell_bbox=target_evidence.bbox,
        row_bbox=_union_bbox(row_evidence),
        column_bbox=_union_bbox(column_evidence),
        row_cells=tuple(
            _context_item(cell, hybrid_map[cell.key], pipeline_map[cell.key])
            for cell in row_cells
        ),
        column_cells=tuple(
            _context_item(cell, hybrid_map[cell.key], pipeline_map[cell.key])
            for cell in column_cells
        ),
    )


def extract_html_tables(value: str) -> list[ParsedTable]:
    parser = _TableFragmentParser(value)
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return []
    return [parse_table_html(fragment) for fragment in parser.fragments]


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    parts: list[str] = []
    current: list[str] = []
    in_code = False
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if char == "\\" and index + 1 < len(stripped) and stripped[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "`":
            in_code = not in_code
            current.append(char)
        elif char == "|" and not in_code:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current).strip())
    if parts and not parts[0]:
        parts.pop(0)
    if parts and not parts[-1]:
        parts.pop()
    return parts


def _markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) is not None
        for cell in cells
    )


def extract_markdown_tables(value: str) -> list[TableSnapshot]:
    lines = value.splitlines()
    code_lines: set[int] = set()
    in_code = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
            code_lines.add(index)
        elif in_code:
            code_lines.add(index)

    result = []
    consumed: set[int] = set()
    for index in range(1, len(lines)):
        if index in code_lines or index - 1 in code_lines or index in consumed:
            continue
        separator = _split_markdown_row(lines[index])
        header = _split_markdown_row(lines[index - 1])
        if not _markdown_separator(separator) or len(header) != len(separator):
            continue
        rows = [header]
        consumed.update({index - 1, index})
        cursor = index + 1
        while cursor < len(lines) and cursor not in code_lines:
            if "|" not in lines[cursor] or not lines[cursor].strip():
                break
            row = _split_markdown_row(lines[cursor])
            if len(row) != len(header):
                break
            rows.append(row)
            consumed.add(cursor)
            cursor += 1
        cells = tuple(
            ((row_index, row_index, col_index, col_index), text)
            for row_index, row in enumerate(rows)
            for col_index, text in enumerate(row)
        )
        result.append(TableSnapshot(len(rows), len(header), cells))
    return result


def extract_table_snapshots(value: str) -> list[TableSnapshot]:
    html_tables = [
        TableSnapshot(
            table.row_count,
            table.col_count,
            tuple((cell.key, cell.text) for cell in table.cells),
        )
        for table in extract_html_tables(value)
        if table.valid
    ]
    return [*html_tables, *extract_markdown_tables(value)]

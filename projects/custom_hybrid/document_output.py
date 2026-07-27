"""Stable document-oriented output for the Custom Hybrid MinerU workflow.

MinerU's ``middle.json`` remains the lossless internal representation and
``content_list_v2.json`` remains the backend-neutral rendered representation.
This module combines both into a smaller integration contract shaped as
``documents -> pages -> blocks``.  It also consumes the optional deterministic
page-sorting report when that report proves a complete document partition.

All public ``bbox`` values use normalized ``[x, y, width, height]`` coordinates
with a top-left origin.  This differs deliberately from MinerU's internal
pixel-based ``[x0, y0, x1, y1]`` coordinates and is declared in every output.
"""

from __future__ import annotations

import argparse
import json
import math
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DOCUMENT_OUTPUT_SCHEMA_VERSION = "1.0"
DOCUMENT_OUTPUT_SUFFIX = "_document.json"


class ExtensibleSchemaModel(BaseModel):
    """Accept additive fields so downstream schema extensions remain compatible."""

    model_config = ConfigDict(extra="allow")


class CoordinateSystem(ExtensibleSchemaModel):
    bbox_format: str = Field(
        default="xywh",
        description="Bounding-box order: [x, y, width, height].",
    )
    unit: str = Field(
        default="normalized",
        description="Coordinates are normalized independently by page width and height.",
    )
    origin: str = Field(default="top_left")


class TableCellPosition(ExtensibleSchemaModel):
    table_id: str = Field(min_length=1)
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=0)
    col_start: int = Field(ge=0)
    col_end: int = Field(ge=0)
    is_header: bool | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "TableCellPosition":
        if self.row_end < self.row_start:
            raise ValueError("row_end must be greater than or equal to row_start")
        if self.col_end < self.col_start:
            raise ValueError("col_end must be greater than or equal to col_start")
        return self


class DocumentBlock(ExtensibleSchemaModel):
    block_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = ""
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Normalized [x, y, width, height] bounding box.",
    )
    type: str = Field(
        min_length=1,
        description="Stable type such as text, table, figure, or equation.",
    )
    raw_type: str | None = Field(
        default=None,
        description="Original MinerU content_list_v2 type.",
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    asset_path: str | None = None
    table_cell: TableCellPosition | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls,
        value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x, y, width, height = (float(item) for item in value)
        if not all(math.isfinite(item) for item in (x, y, width, height)):
            raise ValueError("bbox coordinates must be finite")
        tolerance = 1e-6
        if min(x, y, width, height) < -tolerance:
            raise ValueError("bbox coordinates must be non-negative")
        if x + width > 1 + tolerance or y + height > 1 + tolerance:
            raise ValueError("bbox must fit inside the normalized page")
        return tuple(round(max(0.0, min(1.0, item)), 6) for item in (x, y, width, height))


class DocumentTable(ExtensibleSchemaModel):
    table_id: str = Field(min_length=1)
    bbox: tuple[float, float, float, float] | None = None
    html: str = ""
    table_type: str | None = None
    table_nest_level: int | None = Field(default=None, ge=1)
    image_path: str | None = None
    caption: str = ""
    footnote: str = ""
    cell_block_ids: list[str] = Field(default_factory=list)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls,
        value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        return DocumentBlock.validate_bbox(value)


class DocumentPage(ExtensibleSchemaModel):
    page_number: int = Field(ge=1)
    source_page_index: int | None = Field(default=None, ge=0)
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    tables: list[DocumentTable] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_members(self) -> "DocumentPage":
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block_id values must be unique within a page")
        if any(block.page_number != self.page_number for block in self.blocks):
            raise ValueError("block page_number must match its containing page")
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError("table_id values must be unique within a page")
        return self


class ParsedDocument(ExtensibleSchemaModel):
    document_index: int = Field(ge=0)
    page_range: str = Field(min_length=1)
    pages: list[DocumentPage] = Field(default_factory=list)
    group_id: str | None = None
    document_kind: str | None = None
    document_title: str | None = None
    grouping_status: str | None = None
    ordering_status: str | None = None
    needs_review: bool = False


class DocumentOutput(ExtensibleSchemaModel):
    schema_version: str = DOCUMENT_OUTPUT_SCHEMA_VERSION
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    documents: list[ParsedDocument] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output_members(self) -> "DocumentOutput":
        indexes = [document.document_index for document in self.documents]
        if len(indexes) != len(set(indexes)):
            raise ValueError("document_index values must be unique")
        page_numbers = [
            page.page_number
            for document in self.documents
            for page in document.pages
        ]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("a source page may appear in only one document")
        return self


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _xyxy_to_normalized_xywh(
    bbox: Any,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    values = [_as_finite_float(value) for value in bbox[:4]]
    if any(value is None for value in values):
        return None
    x0, y0, x1, y1 = (float(value) for value in values)
    if page_width <= 0 or page_height <= 0 or x1 < x0 or y1 < y0:
        return None
    normalized = (
        x0 / page_width,
        y0 / page_height,
        (x1 - x0) / page_width,
        (y1 - y0) / page_height,
    )
    tolerance = 1e-6
    if min(normalized) < -tolerance:
        return None
    if normalized[0] + normalized[2] > 1 + tolerance:
        return None
    if normalized[1] + normalized[3] > 1 + tolerance:
        return None
    return tuple(round(max(0.0, min(1.0, value)), 6) for value in normalized)


def _content_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    """Convert content_list_v2's 0-1000 xyxy box into normalized xywh."""

    return _xyxy_to_normalized_xywh(bbox, 1000.0, 1000.0)


def _page_size(page: Mapping[str, Any]) -> tuple[float | None, float | None]:
    value = page.get("page_size")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    width = _as_finite_float(value[0])
    height = _as_finite_float(value[1])
    if width is None or height is None or width <= 0 or height <= 0:
        return None, None
    return width, height


def _source_page_index(page: Mapping[str, Any], physical_index: int) -> int:
    value = page.get("page_idx")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return physical_index


def _inline_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_inline_text(item) for item in value)
    if not isinstance(value, Mapping):
        return ""
    children = value.get("children")
    if isinstance(children, list):
        child_text = _inline_text(children)
        if child_text:
            return child_text
    content = value.get("content")
    if isinstance(content, str):
        return content
    item_content = value.get("item_content")
    if item_content is not None:
        return _inline_text(item_content)
    return ""


def _joined_text(*values: Any) -> str:
    return "\n".join(
        text.strip()
        for text in (_inline_text(value) for value in values)
        if text.strip()
    )


def _html_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


class _HTMLTextExtractor(HTMLParser):
    """Small dependency-free text view for fallback table blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _block_text(raw_type: str, content: Mapping[str, Any]) -> str:
    if raw_type == "title":
        return _inline_text(content.get("title_content")).strip()
    if raw_type == "paragraph":
        return _inline_text(content.get("paragraph_content")).strip()
    if raw_type == "equation_interline":
        return str(content.get("math_content") or "").strip()
    if raw_type == "image":
        return _joined_text(
            content.get("content"),
            content.get("image_caption"),
            content.get("image_footnote"),
        )
    if raw_type == "chart":
        return _joined_text(
            content.get("content"),
            content.get("chart_caption"),
            content.get("chart_footnote"),
        )
    if raw_type == "table":
        return _joined_text(
            content.get("table_caption"),
            _html_text(content.get("html")),
            content.get("table_footnote"),
        )
    if raw_type in {"code", "algorithm"}:
        prefix = raw_type
        return _joined_text(
            content.get(f"{prefix}_caption"),
            content.get(f"{prefix}_content"),
            content.get(f"{prefix}_footnote"),
        )
    if raw_type in {"list", "index"}:
        items = content.get("list_items")
        if not isinstance(items, list):
            return ""
        return "\n".join(
            text for text in (_inline_text(item).strip() for item in items) if text
        )
    for key, value in content.items():
        if key.endswith("_content"):
            text = _inline_text(value).strip()
            if text:
                return text
    return _inline_text(content).strip()


def _stable_block_type(raw_type: str) -> str:
    if raw_type in {"image", "chart"}:
        return "figure"
    if raw_type == "table":
        return "table"
    if raw_type == "equation_interline":
        return "equation"
    return "text"


def _asset_path(content: Mapping[str, Any]) -> str | None:
    source = content.get("image_source")
    if not isinstance(source, Mapping):
        return None
    path = source.get("path")
    return path.strip() if isinstance(path, str) and path.strip() else None


def _iter_middle_tables(page: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tables: list[Mapping[str, Any]] = []
    for bucket in ("para_blocks", "discarded_blocks"):
        blocks = page.get(bucket)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, Mapping) and block.get("type") == "table":
                tables.append(block)
    return tables


def _middle_table_cells(table: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for block in table.get("blocks", []):
        if not isinstance(block, Mapping) or block.get("type") != "table_body":
            continue
        for line in block.get("lines", []):
            if not isinstance(line, Mapping):
                continue
            for span in line.get("spans", []):
                if not isinstance(span, Mapping) or span.get("type") != "table":
                    continue
                cells = span.get("table_cells")
                if isinstance(cells, list):
                    return [cell for cell in cells if isinstance(cell, Mapping)]
    return []


def _cell_position(
    cell: Mapping[str, Any],
    table_id: str,
) -> TableCellPosition | None:
    try:
        return TableCellPosition(
            table_id=table_id,
            row_start=int(cell.get("row_start", 0)),
            row_end=int(cell.get("row_end", cell.get("row_start", 0))),
            col_start=int(cell.get("col_start", 0)),
            col_end=int(cell.get("col_end", cell.get("col_start", 0))),
            is_header=(
                cell["is_header"] if isinstance(cell.get("is_header"), bool) else None
            ),
        )
    except (TypeError, ValueError):
        return None


def _unique_cell_block_id(
    table_id: str,
    position: TableCellPosition,
    seen: set[str],
) -> str:
    base = (
        f"{table_id}-r{position.row_start}-c{position.col_start}"
    )
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}-{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _table_blocks(
    page_number: int,
    table_index: int,
    item: Mapping[str, Any],
    middle_table: Mapping[str, Any] | None,
    page_width: float | None,
    page_height: float | None,
) -> tuple[list[DocumentBlock], DocumentTable]:
    table_id = f"p{page_number}-t{table_index}"
    content = item.get("content")
    content = content if isinstance(content, Mapping) else {}
    table_bbox = _content_bbox(item.get("bbox"))
    raw_cells = _middle_table_cells(middle_table or {})
    use_middle_coordinates = bool(raw_cells and page_width and page_height)
    if not raw_cells:
        candidate_cells = content.get("table_cells")
        if isinstance(candidate_cells, list):
            raw_cells = [
                cell for cell in candidate_cells if isinstance(cell, Mapping)
            ]

    blocks: list[DocumentBlock] = []
    seen_ids: set[str] = set()
    for cell in raw_cells:
        position = _cell_position(cell, table_id)
        if position is None:
            continue
        if use_middle_coordinates:
            cell_bbox = _xyxy_to_normalized_xywh(
                cell.get("bbox"),
                float(page_width),
                float(page_height),
            )
        else:
            cell_bbox = _content_bbox(cell.get("bbox"))
        if cell_bbox is None:
            continue
        confidence = _as_finite_float(cell.get("confidence"))
        if confidence is not None and not 0 <= confidence <= 1:
            confidence = None
        blocks.append(
            DocumentBlock(
                block_id=_unique_cell_block_id(table_id, position, seen_ids),
                page_number=page_number,
                text=str(cell.get("text") or ""),
                bbox=cell_bbox,
                type="table",
                raw_type="table_cell",
                confidence=confidence,
                table_cell=position,
            )
        )

    html = content.get("html")
    html = html if isinstance(html, str) else ""
    if not blocks:
        blocks.append(
            DocumentBlock(
                block_id=table_id,
                page_number=page_number,
                text=_block_text("table", content),
                bbox=table_bbox,
                type="table",
                raw_type="table",
                asset_path=_asset_path(content),
            )
        )
    table = DocumentTable(
        table_id=table_id,
        bbox=table_bbox,
        html=html,
        table_type=(
            str(content["table_type"]) if content.get("table_type") is not None else None
        ),
        table_nest_level=(
            int(content["table_nest_level"])
            if isinstance(content.get("table_nest_level"), int)
            else None
        ),
        image_path=_asset_path(content),
        caption=_inline_text(content.get("table_caption")).strip(),
        footnote=_inline_text(content.get("table_footnote")).strip(),
        cell_block_ids=[block.block_id for block in blocks if block.table_cell],
    )
    return blocks, table


def _build_pages(
    middle_pages: Sequence[Mapping[str, Any]],
    content_pages: Sequence[Any],
) -> list[DocumentPage]:
    if len(middle_pages) != len(content_pages):
        raise ValueError(
            "middle_json.pdf_info and content_list_v2 must contain the same number of pages"
        )
    pages: list[DocumentPage] = []
    for physical_index, (middle_page, raw_items) in enumerate(
        zip(middle_pages, content_pages)
    ):
        if not isinstance(raw_items, list):
            raise ValueError(f"content_list_v2 page {physical_index + 1} must be a list")
        source_page_index = _source_page_index(middle_page, physical_index)
        page_number = source_page_index + 1
        page_width, page_height = _page_size(middle_page)
        middle_tables = _iter_middle_tables(middle_page)
        blocks: list[DocumentBlock] = []
        tables: list[DocumentTable] = []
        regular_block_index = 0
        table_index = 0
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            raw_type = str(item.get("type") or "unknown")
            content = item.get("content")
            content = content if isinstance(content, Mapping) else {}
            if raw_type == "table":
                middle_table = (
                    middle_tables[table_index]
                    if table_index < len(middle_tables)
                    else None
                )
                table_blocks, table = _table_blocks(
                    page_number,
                    table_index,
                    item,
                    middle_table,
                    page_width,
                    page_height,
                )
                blocks.extend(table_blocks)
                tables.append(table)
                table_index += 1
                continue
            regular_block_index += 1
            blocks.append(
                DocumentBlock(
                    block_id=f"p{page_number}-b{regular_block_index}",
                    page_number=page_number,
                    text=_block_text(raw_type, content),
                    bbox=_content_bbox(item.get("bbox")),
                    type=_stable_block_type(raw_type),
                    raw_type=raw_type,
                    asset_path=_asset_path(content),
                )
            )
        pages.append(
            DocumentPage(
                page_number=page_number,
                source_page_index=source_page_index,
                width=page_width,
                height=page_height,
                blocks=blocks,
                tables=tables,
            )
        )
    return pages


def _physical_index(page_id: Any) -> int | None:
    if not isinstance(page_id, str) or not page_id.startswith("p"):
        return None
    suffix = page_id[1:]
    return int(suffix) if suffix.isdigit() else None


def _single_document_group(
    page_count: int,
    reason: str,
    *,
    needs_review: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return (
        [
            {
                "page_indexes": list(range(page_count)),
                "grouping_status": "single_document_fallback",
                "ordering_status": "physical_order",
                "needs_review": needs_review,
            }
        ]
        if page_count
        else [],
        {
            "mode": "single_document_fallback",
            "reason": reason,
        },
    )


def _document_groups(
    page_count: int,
    sorting_report: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(sorting_report, Mapping):
        return _single_document_group(page_count, "sorting_report_unavailable")
    grouping_complete = sorting_report.get("grouping_status") == "complete" or bool(
        sorting_report.get("can_auto_group")
    )
    groups = sorting_report.get("groups")
    if not grouping_complete or not isinstance(groups, list) or not groups:
        return _single_document_group(
            page_count,
            "grouping_not_proven",
            needs_review=True,
        )

    resolved_groups: list[dict[str, Any]] = []
    assigned: list[int] = []
    for group in groups:
        if not isinstance(group, Mapping):
            return _single_document_group(
                page_count,
                "invalid_group_entry",
                needs_review=True,
            )
        member_ids = group.get("member_page_ids")
        if not isinstance(member_ids, list) or not member_ids:
            return _single_document_group(
                page_count,
                "group_without_pages",
                needs_review=True,
            )
        member_indexes = [_physical_index(page_id) for page_id in member_ids]
        if any(index is None for index in member_indexes):
            return _single_document_group(
                page_count,
                "invalid_page_id",
                needs_review=True,
            )
        member_indexes = [int(index) for index in member_indexes]
        if any(index < 0 or index >= page_count for index in member_indexes):
            return _single_document_group(
                page_count,
                "unknown_page_id",
                needs_review=True,
            )
        resolved_ids = group.get("resolved_order")
        if isinstance(resolved_ids, list):
            resolved_indexes = [_physical_index(page_id) for page_id in resolved_ids]
            resolved_order_valid = (
                not any(index is None for index in resolved_indexes)
                and sorted(int(index) for index in resolved_indexes) == sorted(member_indexes)
            )
            if resolved_order_valid:
                page_indexes = [int(index) for index in resolved_indexes]
            else:
                page_indexes = sorted(member_indexes)
        else:
            resolved_order_valid = False
            page_indexes = sorted(member_indexes)
        assigned.extend(member_indexes)
        ordering_status = str(
            group.get("ordering_status")
            or sorting_report.get("ordering_status")
            or "unknown"
        )
        needs_review = ordering_status == "needs_review" or not resolved_order_valid
        if needs_review and ordering_status != "needs_review":
            ordering_status = "needs_review"
        resolved_groups.append(
            {
                "page_indexes": page_indexes,
                "group_id": group.get("group_id"),
                "document_kind": group.get("document_kind"),
                "document_title": group.get("document_title"),
                "grouping_status": str(group.get("grouping_status") or "complete"),
                "ordering_status": ordering_status,
                "needs_review": needs_review,
            }
        )

    if sorted(assigned) != list(range(page_count)) or len(assigned) != len(set(assigned)):
        return _single_document_group(
            page_count,
            "incomplete_or_overlapping_partition",
            needs_review=True,
        )
    return resolved_groups, {
        "mode": "sorting_report",
        "grouping_strategy": sorting_report.get("grouping_strategy"),
        "grouping_status": sorting_report.get("grouping_status"),
        "ordering_status": sorting_report.get("ordering_status"),
        "can_auto_group": sorting_report.get("can_auto_group"),
        "can_auto_sort": sorting_report.get("can_auto_sort"),
    }


def _format_page_range(page_numbers: Sequence[int]) -> str:
    if not page_numbers:
        return "empty"
    ranges: list[str] = []
    start = previous = page_numbers[0]
    for page_number in page_numbers[1:]:
        if page_number == previous + 1:
            previous = page_number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page_number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def build_document_output(
    middle_json: Mapping[str, Any],
    content_list_v2: Sequence[Any],
    sorting_report: Mapping[str, Any] | None = None,
    raw_metadata: Mapping[str, Any] | None = None,
) -> DocumentOutput:
    """Merge MinerU output artifacts into the stable document integration schema."""

    raw_pages = middle_json.get("pdf_info")
    if not isinstance(raw_pages, list):
        raise ValueError("middle_json must contain a pdf_info list")
    if not isinstance(content_list_v2, (list, tuple)):
        raise ValueError("content_list_v2 must be a page-grouped list")
    middle_pages = [page for page in raw_pages if isinstance(page, Mapping)]
    if len(middle_pages) != len(raw_pages):
        raise ValueError("every middle_json.pdf_info entry must be an object")
    pages = _build_pages(middle_pages, content_list_v2)
    groups, grouping_metadata = _document_groups(len(pages), sorting_report)
    documents = []
    for document_index, group in enumerate(groups):
        document_pages = [pages[index] for index in group["page_indexes"]]
        documents.append(
            ParsedDocument(
                document_index=document_index,
                page_range=_format_page_range(
                    [page.page_number for page in document_pages]
                ),
                pages=document_pages,
                group_id=group.get("group_id"),
                document_kind=group.get("document_kind"),
                document_title=group.get("document_title"),
                grouping_status=group.get("grouping_status"),
                ordering_status=group.get("ordering_status"),
                needs_review=bool(group.get("needs_review", False)),
            )
        )
    metadata: dict[str, Any] = {
        "source": "mineru-custom-hybrid",
        "pipeline": "MinerU Custom Hybrid",
        "backend": middle_json.get("_backend"),
        "mineru_version": middle_json.get("_version_name"),
        "document_grouping": grouping_metadata,
    }
    if raw_metadata:
        metadata.update(dict(raw_metadata))
    return DocumentOutput(documents=documents, raw_metadata=metadata)


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object: {path}")
    return value


def _default_artifact_path(middle_path: Path, suffix: str) -> Path:
    middle_suffix = "_middle.json"
    stem = (
        middle_path.name[: -len(middle_suffix)]
        if middle_path.name.endswith(middle_suffix)
        else middle_path.stem
    )
    return middle_path.with_name(f"{stem}{suffix}")


def write_document_output(
    middle_json_path: str | Path,
    content_list_v2_path: str | Path | None = None,
    sorting_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
    raw_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Read MinerU artifacts, validate the merged output, and write JSON."""

    middle_path = Path(middle_json_path)
    content_path = (
        Path(content_list_v2_path)
        if content_list_v2_path is not None
        else _default_artifact_path(middle_path, "_content_list_v2.json")
    )
    report_path = (
        Path(sorting_report_path)
        if sorting_report_path is not None
        else _default_artifact_path(middle_path, "_sorting_report.json")
    )
    target_path = (
        Path(output_path)
        if output_path is not None
        else _default_artifact_path(middle_path, DOCUMENT_OUTPUT_SUFFIX)
    )
    middle_json = _load_mapping(middle_path, "middle_json")
    content_list_v2 = json.loads(content_path.read_text(encoding="utf-8"))
    sorting_report = _load_mapping(report_path, "sorting_report") if report_path.is_file() else None
    output = build_document_output(
        middle_json,
        content_list_v2,
        sorting_report=sorting_report,
        raw_metadata=raw_metadata,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(
            output.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("middle_json", nargs="?", type=Path)
    parser.add_argument("--content-list-v2", type=Path)
    parser.add_argument("--sorting-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the JSON Schema without reading any artifacts.",
    )
    args = parser.parse_args(argv)
    if args.print_schema:
        print(json.dumps(DocumentOutput.model_json_schema(), ensure_ascii=False, indent=2))
        return 0
    if args.middle_json is None:
        parser.error("middle_json is required unless --print-schema is used")
    output_path = write_document_output(
        args.middle_json,
        args.content_list_v2,
        args.sorting_report,
        args.output,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

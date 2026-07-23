"""Deterministic, report-only grouping and ordering for MinerU middle JSON.

The input to this module is the fused ``*_middle.json`` document.  It never
rewrites that document or the source PDF.  A proposed order is considered
safe only when page-number evidence, document grouping evidence, and global
consistency checks all agree; otherwise the report retains the unresolved
pages for review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PAGE_SORTING_VERSION = 1
REPORT_ONLY_MODE = "report_only"

PAGE_WORD_RE = re.compile(
    r"\bpage\s*([0-9A-Za-z|]+)\s*['’`·,.:;-]*\s*"
    r"(?:of|/)\s*([0-9A-Za-z|]+)\b",
    re.IGNORECASE,
)
P_SHORT_RE = re.compile(
    r"(?:^|[\s(\[])p\s*[.,]?\s*([0-9A-Za-z|]+)\s*/\s*"
    r"([0-9A-Za-z|]+)\b",
    re.IGNORECASE,
)
CJK_PAGE_RE = re.compile(
    r"\u7b2c\s*([0-9A-Za-z|]+)\s*[\u9801\u9875]\s*"
    r"(?:[/,\uff0c]|\u4e4b)?\s*(?:\u5171\s*)?"
    r"([0-9A-Za-z|]+)\s*[\u9801\u9875]",
    re.IGNORECASE,
)
CJK_TOTAL_FIRST_RE = re.compile(
    r"\u5171\s*([0-9A-Za-z|]+)\s*[\u9801\u9875]\s*"
    r"[,\uff0c]?\s*\u7b2c\s*([0-9A-Za-z|]+)\s*[\u9801\u9875]",
    re.IGNORECASE,
)
FRACTION_RE = re.compile(
    r"(?:^|[\s(\[])\s*([0-9A-Za-z|]{1,4})\s*/\s*"
    r"([0-9A-Za-z|]{1,4})(?:$|[\s)\].,;:-])",
    re.IGNORECASE,
)
BARE_NUMBER_RE = re.compile(
    r"^[\s\-.,:()\[\]]*([0-9A-Za-z|]{1,4})"
    r"[\s\-.,:()\[\]]*$"
)
IDENTIFIER_RE = re.compile(
    r"\b(policy|claim|invoice|patient|member|case|reference|application|"
    r"account|contract|certificate|document)\s*"
    r"(?:no\.?|number|id|#)\s*[:.]?\s*"
    r"([A-Z0-9][A-Z0-9<>/()\-]{3,})",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]{2,}|[\u3400-\u9fff]{2,}")
PAGE_MARKER_RE = re.compile(
    r"\bpage\s*[0-9A-Za-z|]+\s*['’`·,.:;-]*\s*"
    r"(?:of|/)\s*[0-9A-Za-z|]+\b|"
    r"\bp\s*[.,]?\s*[0-9A-Za-z|]+\s*/\s*[0-9A-Za-z|]+\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d{1,4}\b")

FAMILY_HINTS = (
    "claim",
    "invoice",
    "medical",
    "report",
    "statement",
    "benefit",
    "receipt",
    "application",
    "certificate",
    "policy",
    "form",
)
FAMILY_STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "page",
    "pages",
    "of",
    "no",
    "number",
    "id",
    "form",
}
DOCUMENT_TITLE_BLOCK_TYPES = frozenset({"title", "heading", "table_caption"})
DOCUMENT_KIND_PATTERNS = (
    (
        "letter_of_guarantee",
        re.compile(r"\bletter\s+of\s+guarantee\b|保證書|擔保信", re.IGNORECASE),
    ),
    (
        "statement_summary",
        re.compile(
            r"\bsummary\s+of\s+(?:the\s+)?statement\s+of\s+account\b|"
            r"賬單摘要|帳單摘要",
            re.IGNORECASE,
        ),
    ),
    (
        "statement_of_account",
        re.compile(
            r"\bstatement\s+of\s+account\b|留醫賬單|住院賬單|住院帳單",
            re.IGNORECASE,
        ),
    ),
    ("invoice", re.compile(r"^\s*invoice\s*$|發票", re.IGNORECASE)),
    ("receipt", re.compile(r"^\s*receipt\s*$|收據", re.IGNORECASE)),
    (
        "discharge_summary",
        re.compile(r"\bdischarge\s+summary\b|出院總結|出院摘要", re.IGNORECASE),
    ),
    (
        "medical_report",
        re.compile(r"\bmedical\s+report\b|醫療報告|醫事報告", re.IGNORECASE),
    ),
    (
        "claim_form",
        re.compile(r"\bclaim\s+form\b|索償.*申請表|理賠.*申請表", re.IGNORECASE),
    ),
)
PACKET_PAGINATION_MIN_COVERAGE = 0.60
OCR_IDENTIFIER_CONFUSABLE_PAIRS = frozenset(
    {
        frozenset(("0", "o")),
        frozenset(("1", "i")),
        # OCR can split a printed K into ``1<``; identifier cleanup removes <.
        frozenset(("1", "k")),
        frozenset(("1", "l")),
        frozenset(("2", "z")),
        frozenset(("5", "s")),
        frozenset(("6", "g")),
        frozenset(("8", "b")),
        frozenset(("9", "q")),
    }
)
MIN_OCR_NEAR_IDENTIFIER_LENGTH = 8
MIN_OCR_NEAR_PAGE_CONFIDENCE = 0.90
MIN_OCR_NEAR_SHARED_FAMILY_TOKENS = 2
OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }
)
ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)


@dataclass(frozen=True)
class TextObservation:
    text: str
    block_type: str
    bucket: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class PageCandidate:
    current: int
    total: int | None
    raw: str
    source_type: str
    block_type: str
    bucket: str
    bbox: tuple[float, float, float, float] | None
    confidence: float
    explicit: bool
    corrected: bool = False
    supporting_observations: int = 1


@dataclass
class PageRecord:
    page_id: str
    physical_page_idx: int
    source_page_idx: int | None
    page_size: tuple[float, float] | None
    candidates: list[PageCandidate] = field(default_factory=list)


@dataclass
class PageEvidence:
    page_id: str
    physical_page_idx: int
    source_page_idx: int | None
    page_size: tuple[float, float] | None
    record: PageRecord
    evidence_text_count: int
    family_tokens: frozenset[str]
    identifiers: dict[str, tuple[str, ...]]
    document_kind: str | None
    document_title: str | None

    def best_candidate(
        self,
        expected_total: int | None = None,
    ) -> tuple[PageCandidate | None, str]:
        """Choose one candidate while refusing close conflicting evidence."""

        candidates = [
            candidate
            for candidate in self.record.candidates
            if candidate.current > 0
            and (candidate.total is None or candidate.current <= candidate.total)
        ]
        if expected_total is not None:
            matching_explicit = [
                candidate
                for candidate in candidates
                if candidate.explicit and candidate.total == expected_total
            ]
            conflicting_explicit = [
                candidate
                for candidate in candidates
                if candidate.explicit
                and candidate.total is not None
                and candidate.total != expected_total
            ]
            if matching_explicit and conflicting_explicit:
                best_matching = max(
                    candidate.confidence for candidate in matching_explicit
                )
                best_conflicting = max(
                    candidate.confidence for candidate in conflicting_explicit
                )
                if best_matching - best_conflicting < 0.12:
                    return None, "conflicting_candidates"
            elif conflicting_explicit:
                return None, "total_conflict"
            compatible = [
                candidate
                for candidate in candidates
                if candidate.total in (None, expected_total)
            ]
            if compatible:
                candidates = compatible
        if not candidates:
            return None, "no_candidate"

        ranked: list[tuple[float, PageCandidate]] = []
        for candidate in candidates:
            score = candidate.confidence
            if expected_total is not None:
                if candidate.total == expected_total:
                    score += 0.10
                elif candidate.total is not None:
                    score -= 0.30
            if candidate.explicit:
                score += 0.03
            score += min(candidate.supporting_observations, 3) * 0.01
            ranked.append((score, candidate))
        ranked.sort(
            key=lambda item: (
                -item[0],
                -int(item[1].explicit),
                item[1].current,
                item[1].total if item[1].total is not None else 10**9,
                item[1].raw,
            )
        )
        best_score, best = ranked[0]
        if len(ranked) > 1:
            second_score, second = ranked[1]
            different = (
                best.current != second.current or best.total != second.total
            )
            if different and best_score - second_score < 0.08:
                return None, "conflicting_candidates"
        return best, "selected"


@dataclass
class DocumentGroup:
    group_id: str
    seed_page_id: str
    expected_total: int
    members: list[PageEvidence] = field(default_factory=list)
    family_tokens: set[str] = field(default_factory=set)
    identifiers: dict[str, set[str]] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SortingReport:
    version: int
    mode: str
    status: str
    page_count: int
    anchor_count: int
    group_count: int
    can_auto_sort: bool
    physical_order: list[str]
    proposed_document_order: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    semantic_quality: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        values = tuple(float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(value == value for value in values):
        return None
    if values[2] < values[0] or values[3] < values[1]:
        return None
    return values[0], values[1], values[2], values[3]


def _page_size(page: Mapping[str, Any]) -> tuple[float, float] | None:
    value = page.get("page_size")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _span_text(span: Mapping[str, Any]) -> str:
    for key in ("content", "text"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _iter_text_observations(
    node: Mapping[str, Any],
    bucket: str,
    inherited_type: str = "",
) -> Iterable[TextObservation]:
    """Yield line-level text without depending on rendered Markdown."""

    node_type = str(node.get("type") or inherited_type or "unknown").casefold()
    node_bbox = _valid_bbox(node.get("bbox"))
    direct_text = []
    for key in ("text", "content"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            direct_text.append(value.strip())
    if direct_text:
        yield TextObservation(
            text=" ".join(dict.fromkeys(direct_text)),
            block_type=node_type,
            bucket=bucket,
            bbox=node_bbox,
        )

    lines = node.get("lines")
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, Mapping):
                continue
            line_bbox = _valid_bbox(line.get("bbox")) or node_bbox
            line_text = []
            for key in ("text", "content"):
                value = line.get(key)
                if isinstance(value, str) and value.strip():
                    line_text.append(value.strip())
            spans = line.get("spans")
            if isinstance(spans, list):
                for span in spans:
                    if isinstance(span, Mapping):
                        value = _span_text(span)
                        if value:
                            line_text.append(value)
            if line_text:
                yield TextObservation(
                    text=" ".join(dict.fromkeys(line_text)),
                    block_type=node_type,
                    bucket=bucket,
                    bbox=line_bbox,
                )

    # Table cells and nested blocks can carry text outside the normal lines.
    for key in ("blocks", "table_cells", "content_spans", "spans"):
        children = node.get(key)
        if not isinstance(children, list):
            continue
        for child in children:
            if isinstance(child, Mapping):
                yield from _iter_text_observations(child, bucket, node_type)


def _page_observations(page: Mapping[str, Any]) -> list[TextObservation]:
    observations: list[TextObservation] = []
    seen: set[tuple[str, str, str, tuple[float, ...] | None]] = set()
    for bucket in ("discarded_blocks", "preproc_blocks", "para_blocks"):
        blocks = page.get(bucket)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            for observation in _iter_text_observations(block, bucket):
                text = " ".join(observation.text.split())
                if not text:
                    continue
                key = (
                    text.casefold(),
                    observation.block_type,
                    observation.bucket,
                    observation.bbox,
                )
                if key in seen:
                    continue
                seen.add(key)
                observations.append(replace(observation, text=text))
    return observations


def _roman_to_int(value: str) -> int | None:
    token = value.upper()
    if not token or not ROMAN_RE.fullmatch(token):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(token):
        current = values[char]
        total += -current if current < previous else current
        previous = current
    return total if 0 < total <= 999 else None


def _parse_number(value: str) -> tuple[int | None, bool]:
    token = re.sub(r"[^0-9A-Za-z|]", "", value)
    if not token:
        return None, False
    # A lower-case OCR ``l`` is much more likely to be a one than Roman L.
    if token in {"l", "|"}:
        return 1, True
    if token in {"i", "I"}:
        return 1, token == "i"
    roman = _roman_to_int(token)
    if roman is not None:
        return roman, False
    normalized = token.translate(OCR_DIGIT_TRANSLATION)
    if not normalized.isdigit():
        return None, False
    number = int(normalized)
    if not 0 < number <= 999:
        return None, False
    return number, normalized != token


def _candidate(
    current_token: str,
    total_token: str | None,
    raw: str,
    observation: TextObservation,
    explicit: bool,
    page_size: tuple[float, float] | None,
) -> PageCandidate | None:
    current, current_corrected = _parse_number(current_token)
    total, total_corrected = (
        _parse_number(total_token) if total_token is not None else (None, False)
    )
    if current is None:
        return None
    block_type = observation.block_type
    source_type = (
        "page_number"
        if block_type == "page_number"
        else block_type
        if block_type in {"header", "footer"}
        else "text"
    )
    confidence = 0.90 if explicit else 0.64
    if source_type == "page_number":
        confidence += 0.05
    elif source_type in {"header", "footer"}:
        confidence += 0.02
    if page_size is not None and observation.bbox is not None:
        height = page_size[1]
        top, bottom = observation.bbox[1], observation.bbox[3]
        if bottom <= height * 0.18 or top >= height * 0.82:
            confidence += 0.02
    if current_corrected or total_corrected:
        confidence -= 0.04
    if total is not None and current > total:
        confidence -= 0.45
    return PageCandidate(
        current=current,
        total=total,
        raw=raw.strip(),
        source_type=source_type,
        block_type=block_type,
        bucket=observation.bucket,
        bbox=observation.bbox,
        confidence=round(max(0.05, min(confidence, 0.99)), 4),
        explicit=explicit,
        corrected=current_corrected or total_corrected,
    )


def _extract_candidates(
    observations: Sequence[TextObservation],
    page_size: tuple[float, float] | None,
) -> list[PageCandidate]:
    candidates: dict[tuple[int, int | None], PageCandidate] = {}
    for observation in observations:
        text = observation.text
        patterns: list[tuple[re.Pattern[str], int, int]] = [
            (PAGE_WORD_RE, 1, 2),
            (P_SHORT_RE, 1, 2),
            (CJK_PAGE_RE, 1, 2),
            (CJK_TOTAL_FIRST_RE, 2, 1),
        ]
        for pattern, current_group, total_group in patterns:
            for match in pattern.finditer(text):
                candidate = _candidate(
                    match.group(current_group),
                    match.group(total_group),
                    match.group(0),
                    observation,
                    True,
                    page_size,
                )
                if candidate is not None:
                    _merge_candidate(candidates, candidate)

        # A bare number is accepted only from MinerU's page_number block.  A
        # random number in body text or a table must not become page evidence.
        if observation.block_type == "page_number":
            match = BARE_NUMBER_RE.fullmatch(text)
            if match:
                candidate = _candidate(
                    match.group(1),
                    None,
                    match.group(0),
                    observation,
                    False,
                    page_size,
                )
                if candidate is not None:
                    _merge_candidate(candidates, candidate)
        # Fractions without a Page/P prefix are only considered in page
        # metadata or a header/footer margin.  This avoids dates and amounts.
        if observation.block_type in {"page_number", "header", "footer"}:
            match = FRACTION_RE.search(text)
            short_margin_text = (
                observation.block_type == "page_number" or len(text.strip()) <= 24
            )
            if (
                match
                and short_margin_text
                and not PAGE_WORD_RE.search(text)
                and not P_SHORT_RE.search(text)
            ):
                candidate = _candidate(
                    match.group(1),
                    match.group(2),
                    match.group(0),
                    observation,
                    observation.block_type == "page_number",
                    page_size,
                )
                if (
                    candidate is not None
                    and candidate.total is not None
                    and candidate.current <= candidate.total
                ):
                    _merge_candidate(candidates, candidate)
    return sorted(
        candidates.values(),
        key=lambda item: (-item.confidence, -int(item.explicit), item.current, item.raw),
    )


def _merge_candidate(
    candidates: dict[tuple[int, int | None], PageCandidate],
    candidate: PageCandidate,
) -> None:
    key = (candidate.current, candidate.total)
    previous = candidates.get(key)
    if previous is None:
        candidates[key] = candidate
        return
    if candidate.confidence > previous.confidence:
        candidates[key] = replace(
            candidate,
            supporting_observations=previous.supporting_observations + 1,
        )
    else:
        candidates[key] = replace(
            previous,
            supporting_observations=previous.supporting_observations + 1,
        )


def _normalize_text(value: str, remove_numbers: bool = False) -> str:
    value = PAGE_MARKER_RE.sub(" ", value.casefold())
    value = CJK_PAGE_RE.sub(" ", value)
    value = CJK_TOTAL_FIRST_RE.sub(" ", value)
    value = IDENTIFIER_RE.sub(" ", value)
    if remove_numbers:
        value = NUMBER_RE.sub(" ", value)
    value = re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", value)
    return " ".join(value.split())


def _identifier_values(observations: Sequence[TextObservation]) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {}
    for observation in observations:
        for match in IDENTIFIER_RE.finditer(observation.text):
            key = match.group(1).casefold()
            value = re.sub(r"[^a-z0-9]", "", match.group(2).casefold())
            if not value or not any(char.isdigit() for char in value):
                continue
            values.setdefault(key, set()).add(value)
    return {key: tuple(sorted(items)) for key, items in sorted(values.items())}


def _family_tokens(
    observations: Sequence[TextObservation],
    page_size: tuple[float, float] | None,
) -> frozenset[str]:
    tokens: set[str] = set()
    for observation in observations:
        normalized = _normalize_text(observation.text, remove_numbers=True)
        if not normalized:
            continue
        is_margin = False
        if page_size is not None and observation.bbox is not None:
            is_margin = (
                observation.bbox[3] <= page_size[1] * 0.20
                or observation.bbox[1] >= page_size[1] * 0.80
            )
        has_hint = any(hint in normalized for hint in FAMILY_HINTS)
        is_title = observation.block_type in {"title", "heading"}
        top_title_hint = bool(
            has_hint
            and len(normalized) <= 160
            and observation.bbox is not None
            and page_size is not None
            and observation.bbox[1] <= page_size[1] * 0.35
        )
        if (
            observation.block_type in {"header", "footer"}
            or is_margin
            or is_title
            or top_title_hint
        ):
            tokens.update(
                token
                for token in TOKEN_RE.findall(normalized)
                if token not in FAMILY_STOPWORDS and len(token) >= 2
            )
    return frozenset(tokens)


def _document_profile(
    observations: Sequence[TextObservation],
) -> tuple[str | None, str | None]:
    """Classify strong document titles without using body-text semantics."""

    seen: set[str] = set()
    for observation in observations:
        if observation.block_type not in DOCUMENT_TITLE_BLOCK_TYPES:
            continue
        title = " ".join(observation.text.split())
        normalized = title.casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        for kind, pattern in DOCUMENT_KIND_PATTERNS:
            if pattern.search(title):
                return kind, title
    return None, None


def build_manifest(
    middle_json: Mapping[str, Any],
    page_order: Sequence[Mapping[str, Any]] | None = None,
) -> list[PageRecord]:
    pages = page_order if page_order is not None else middle_json.get("pdf_info", [])
    if not isinstance(pages, list):
        raise ValueError("middle_json.pdf_info must be a list")
    records: list[PageRecord] = []
    for physical_idx, raw_page in enumerate(pages):
        page = raw_page if isinstance(raw_page, Mapping) else {}
        raw_idx = page.get("page_idx")
        source_idx = raw_idx if isinstance(raw_idx, int) and not isinstance(raw_idx, bool) else None
        size = _page_size(page)
        observations = _page_observations(page)
        records.append(
            PageRecord(
                page_id=f"p{physical_idx:04d}",
                physical_page_idx=physical_idx,
                source_page_idx=source_idx,
                page_size=size,
                candidates=_extract_candidates(observations, size),
            )
        )
    return records


def build_evidence(
    middle_json: Mapping[str, Any],
    page_order: Sequence[Mapping[str, Any]] | None = None,
) -> list[PageEvidence]:
    pages = page_order if page_order is not None else middle_json.get("pdf_info", [])
    if not isinstance(pages, list):
        raise ValueError("middle_json.pdf_info must be a list")
    result: list[PageEvidence] = []
    for physical_idx, raw_page in enumerate(pages):
        page = raw_page if isinstance(raw_page, Mapping) else {}
        raw_idx = page.get("page_idx")
        source_idx = (
            raw_idx
            if isinstance(raw_idx, int) and not isinstance(raw_idx, bool)
            else None
        )
        size = _page_size(page)
        observations = _page_observations(page)
        record = PageRecord(
            page_id=f"p{physical_idx:04d}",
            physical_page_idx=physical_idx,
            source_page_idx=source_idx,
            page_size=size,
            candidates=_extract_candidates(observations, size),
        )
        document_kind, document_title = _document_profile(observations)
        provisional = PageEvidence(
            page_id=record.page_id,
            physical_page_idx=record.physical_page_idx,
            source_page_idx=record.source_page_idx,
            page_size=record.page_size,
            record=record,
            evidence_text_count=len(observations),
            family_tokens=_family_tokens(observations, record.page_size),
            identifiers=_identifier_values(observations),
            document_kind=document_kind,
            document_title=document_title,
        )
        result.append(provisional)
    return result


def _jaccard(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _add_to_group(group: DocumentGroup, page: PageEvidence) -> None:
    group.members.append(page)
    group.family_tokens.update(page.family_tokens)
    for key, values in page.identifiers.items():
        group.identifiers.setdefault(key, set()).update(values)


def _is_ocr_near_identifier(left: str, right: str) -> bool:
    """Return true only for one known OCR-confusable substitution."""

    if (
        left == right
        or len(left) != len(right)
        or len(left) < MIN_OCR_NEAR_IDENTIFIER_LENGTH
    ):
        return False
    differences = [
        (left_char, right_char)
        for left_char, right_char in zip(left, right)
        if left_char != right_char
    ]
    return bool(
        len(differences) == 1
        and frozenset(differences[0]) in OCR_IDENTIFIER_CONFUSABLE_PAIRS
    )


def _supports_ocr_near_identifier(
    page: PageEvidence,
    group: DocumentGroup,
    candidate: PageCandidate | None,
) -> bool:
    """Require independent structure before relaxing an identifier conflict."""

    if (
        candidate is None
        or not candidate.explicit
        or candidate.total != group.expected_total
        or candidate.confidence < MIN_OCR_NEAR_PAGE_CONFIDENCE
    ):
        return False

    occupied_slots: set[int] = set()
    for member in group.members:
        selected, status = member.best_candidate(group.expected_total)
        if selected is None or status != "selected":
            return False
        occupied_slots.add(selected.current)
    missing_slots = set(range(1, group.expected_total + 1)) - occupied_slots
    if (
        len(group.members) != group.expected_total - 1
        or len(occupied_slots) != len(group.members)
        or missing_slots != {candidate.current}
    ):
        return False

    shared_family = page.family_tokens & group.family_tokens
    return bool(
        len(shared_family) >= MIN_OCR_NEAR_SHARED_FAMILY_TOKENS
        and any(token in FAMILY_HINTS for token in shared_family)
    )


def _identifier_match_evidence(
    page: PageEvidence,
    group: DocumentGroup,
    candidate: PageCandidate | None,
) -> tuple[list[str], list[str], list[str]]:
    exact: list[str] = []
    ocr_near: list[str] = []
    conflicts: list[str] = []
    structural_support: bool | None = None
    for key, values in page.identifiers.items():
        existing = group.identifiers.get(key)
        if not existing:
            continue
        if not set(values).isdisjoint(existing):
            exact.append(key)
            continue
        if structural_support is None:
            structural_support = _supports_ocr_near_identifier(
                page,
                group,
                candidate,
            )
        near_match = bool(
            structural_support
            and len(values) == 1
            and len(existing) == 1
            and _is_ocr_near_identifier(values[0], next(iter(existing)))
        )
        if near_match:
            ocr_near.append(key)
        else:
            conflicts.append(key)
    return exact, ocr_near, conflicts


def _group_score(
    page: PageEvidence,
    group: DocumentGroup,
) -> tuple[float, list[str], PageCandidate | None]:
    reasons: list[str] = []
    candidate, candidate_status = page.best_candidate(group.expected_total)
    if candidate_status == "total_conflict":
        return float("-inf"), ["total_conflict"], None
    if candidate_status == "conflicting_candidates":
        reasons.append("conflicting_page_candidates")
    if candidate is not None and candidate.total is not None:
        if candidate.total != group.expected_total:
            return float("-inf"), ["total_conflict"], candidate
        reasons.append("total_match")
    exact_ids, ocr_near_ids, identifier_conflicts = _identifier_match_evidence(
        page,
        group,
        candidate,
    )
    if identifier_conflicts:
        return (
            float("-inf"),
            [f"{key}_conflict" for key in identifier_conflicts],
            candidate,
        )

    score = 0.0
    if exact_ids:
        score += 105.0 + 12.0 * len(exact_ids)
        reasons.extend(f"{key}_match" for key in exact_ids)
    if ocr_near_ids:
        score += 72.0 + 8.0 * len(ocr_near_ids)
        reasons.extend(f"{key}_ocr_near_match" for key in ocr_near_ids)

    if page.family_tokens and group.family_tokens:
        similarity = _jaccard(page.family_tokens, group.family_tokens)
        if similarity == 0:
            score -= 18.0
            reasons.append("family_conflict")
        else:
            score += 55.0 * similarity
            reasons.append(f"family_similarity={similarity:.2f}")
    elif page.family_tokens:
        score += 8.0
        reasons.append("family_observed")

    if candidate is not None:
        score += 30.0 if candidate.total == group.expected_total else 8.0
        score += 14.0 if candidate.explicit else 4.0
        reasons.append("page_number_observed")
    elif candidate_status == "conflicting_candidates":
        score -= 20.0
    elif page.identifiers:
        reasons.append("identifier_observed")

    occupied_slots = {
        selected.current
        for member in group.members
        if (selected := member.best_candidate(group.expected_total)[0]) is not None
    }
    if candidate is not None and candidate.current in occupied_slots:
        score -= 42.0
        reasons.append("duplicate_logical_slot")
    return score, reasons, candidate


def _page_strength(page: PageEvidence) -> tuple[int, int, int, int]:
    explicit = sum(1 for item in page.record.candidates if item.explicit)
    identifier_count = sum(len(values) for values in page.identifiers.values())
    return (
        identifier_count,
        explicit,
        len(page.family_tokens),
        -page.physical_page_idx,
    )


def _is_packet_margin_candidate(
    page: PageEvidence,
    candidate: PageCandidate,
) -> bool:
    if candidate.source_type in {"header", "footer", "page_number"}:
        return True
    if page.page_size is None or candidate.bbox is None:
        return False
    height = page.page_size[1]
    return candidate.bbox[3] <= height * 0.15 or candidate.bbox[1] >= height * 0.85


def _detect_packet_pagination(
    evidence: Sequence[PageEvidence],
) -> dict[str, Any] | None:
    """Detect a wrapper page sequence that validates the physical packet order."""

    page_count = len(evidence)
    if page_count < 3:
        return None

    observed: dict[str, PageCandidate] = {}
    source_counts: dict[str, int] = {}
    for page in evidence:
        packet_candidates = [
            candidate
            for candidate in page.record.candidates
            if candidate.explicit
            and candidate.total == page_count
            and _is_packet_margin_candidate(page, candidate)
        ]
        expected_current = page.physical_page_idx + 1
        if any(candidate.current != expected_current for candidate in packet_candidates):
            return None
        matching = [
            candidate
            for candidate in packet_candidates
            if candidate.current == expected_current
        ]
        if not matching:
            continue
        selected = max(matching, key=lambda item: item.confidence)
        observed[page.page_id] = selected
        source_counts[selected.source_type] = source_counts.get(selected.source_type, 0) + 1

    coverage = len(observed) / page_count
    if (
        coverage < PACKET_PAGINATION_MIN_COVERAGE
        or evidence[0].page_id not in observed
        or evidence[-1].page_id not in observed
    ):
        return None
    dominant_source, dominant_count = max(
        source_counts.items(),
        key=lambda item: (item[1], item[0]),
    )
    if dominant_count / len(observed) < 0.80:
        return None

    minimum_confidence = min(candidate.confidence for candidate in observed.values())
    inferred_confidence = round(max(0.70, minimum_confidence - 0.10), 4)
    pages: dict[str, dict[str, Any]] = {}
    inferred_page_ids: list[str] = []
    for page in evidence:
        candidate = observed.get(page.page_id)
        inferred = candidate is None
        if inferred:
            inferred_page_ids.append(page.page_id)
        pages[page.page_id] = {
            "current": page.physical_page_idx + 1,
            "total": page_count,
            "raw": candidate.raw if candidate is not None else None,
            "source_type": (
                candidate.source_type if candidate is not None else dominant_source
            ),
            "confidence": (
                candidate.confidence if candidate is not None else inferred_confidence
            ),
            "inferred": inferred,
        }
    return {
        "status": "recovered" if inferred_page_ids else "complete",
        "total": page_count,
        "observed_count": len(observed),
        "coverage": round(coverage, 6),
        "dominant_source_type": dominant_source,
        "aligned_with_physical_order": True,
        "inferred_page_ids": inferred_page_ids,
        "resolved_order": [page.page_id for page in evidence],
        "pages": pages,
    }


def _segment_identifier_conflict(
    page: PageEvidence,
    identifiers: Mapping[str, set[str]],
) -> bool:
    for key, values in page.identifiers.items():
        existing = identifiers.get(key)
        if not existing or not set(values).isdisjoint(existing):
            continue
        if (
            len(values) == 1
            and len(existing) == 1
            and _is_ocr_near_identifier(values[0], next(iter(existing)))
        ):
            continue
        return True
    return False


def _segment_packet_documents(
    evidence: Sequence[PageEvidence],
) -> list[dict[str, Any]]:
    """Split an ordered packet only at strong title or identifier boundaries."""

    segments: list[dict[str, Any]] = []
    for page in evidence:
        if not segments:
            segments.append(
                {
                    "document_kind": page.document_kind,
                    "document_title": page.document_title,
                    "members": [page],
                    "identifiers": {
                        key: set(values) for key, values in page.identifiers.items()
                    },
                    "boundary_reason": "packet_start",
                }
            )
            continue

        current = segments[-1]
        kind_change = bool(
            page.document_kind
            and current["document_kind"]
            and page.document_kind != current["document_kind"]
        )
        identifier_change = _segment_identifier_conflict(
            page,
            current["identifiers"],
        )
        if kind_change or identifier_change:
            reason = (
                f"document_kind_change:{current['document_kind']}"
                f"->{page.document_kind}"
                if kind_change
                else "identifier_change"
            )
            segments.append(
                {
                    "document_kind": page.document_kind,
                    "document_title": page.document_title,
                    "members": [page],
                    "identifiers": {
                        key: set(values) for key, values in page.identifiers.items()
                    },
                    "boundary_reason": reason,
                }
            )
            continue

        current["members"].append(page)
        if current["document_kind"] is None and page.document_kind is not None:
            current["document_kind"] = page.document_kind
            current["document_title"] = page.document_title
        for key, values in page.identifiers.items():
            current["identifiers"].setdefault(key, set()).update(values)

    known_kinds = {
        segment["document_kind"]
        for segment in segments
        if segment["document_kind"] is not None
    }
    if len(segments) < 2 or len(known_kinds) < 2:
        return []
    return segments


def _packet_segment_group(
    segment: Mapping[str, Any],
    group_index: int,
    packet_pagination: Mapping[str, Any],
) -> dict[str, Any]:
    members = list(segment["members"])
    group_id = f"doc-{group_index:03d}"
    resolved_order = [page.page_id for page in members]
    family_tokens = sorted(
        set().union(*(set(page.family_tokens) for page in members))
    )
    pages: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for logical_position, page in enumerate(members, start=1):
        packet_page = packet_pagination["pages"][page.page_id]
        reasons = [
            "packet_order_validated",
            f"document_kind={segment['document_kind'] or 'unknown'}",
        ]
        if packet_page["inferred"]:
            reasons.append("packet_page_inferred")
        decisions.append(
            {
                "page_id": page.page_id,
                "group_id": group_id,
                "score": round(100.0 * float(packet_page["confidence"]), 3),
                "reasons": reasons,
                "selected_page_number": logical_position,
            }
        )
        pages.append(
            {
                "page_id": page.page_id,
                "physical_page_idx": page.physical_page_idx,
                "source_page_idx": page.source_page_idx,
                "candidates": [asdict(item) for item in page.record.candidates],
                "selected": None,
                "status": "ordered_by_packet_pagination",
                "logical_position": logical_position,
                "packet_page": packet_page,
                "document_kind": page.document_kind,
                "document_title": page.document_title,
            }
        )
    return {
        "group_id": group_id,
        "seed_page_id": members[0].page_id,
        "expected_total": len(members),
        "member_page_ids": resolved_order,
        "identifiers": {
            key: sorted(values)
            for key, values in sorted(segment["identifiers"].items())
        },
        "family_tokens": family_tokens,
        "document_kind": segment["document_kind"],
        "document_title": segment["document_title"],
        "boundary_reason": segment["boundary_reason"],
        "status": "complete",
        "resolved_order": resolved_order,
        "duplicates": {},
        "missing_numbers": [],
        "unexpected_numbers": [],
        "ambiguous_pages": {},
        "pages": pages,
        "decisions": decisions,
    }


def _analyze_packet_segments(
    evidence: Sequence[PageEvidence],
    segments: Sequence[Mapping[str, Any]],
    packet_pagination: Mapping[str, Any],
    semantic_report: Mapping[str, Any] | None,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = [
        _packet_segment_group(segment, index, packet_pagination)
        for index, segment in enumerate(segments, start=1)
    ]
    proposed_document_order = [
        {
            "group_id": group["group_id"],
            "seed_page_id": group["seed_page_id"],
            "document_kind": group["document_kind"],
            "resolved_order": group["resolved_order"],
            "status": group["status"],
        }
        for group in groups
    ]
    report = SortingReport(
        version=PAGE_SORTING_VERSION,
        mode=mode,
        status="complete",
        page_count=len(evidence),
        anchor_count=len(groups),
        group_count=len(groups),
        can_auto_sort=True,
        physical_order=[page.page_id for page in evidence],
        proposed_document_order=proposed_document_order,
        unresolved=[],
        groups=groups,
        semantic_quality=_semantic_quality_summary(semantic_report),
    ).to_dict()
    report.update(
        {
            "assignment_count": len(evidence) - len(groups),
            "unresolved_count": 0,
            "report_only": True,
            "grouping_strategy": "packet_document_segmentation",
            "packet_pagination": dict(packet_pagination),
            "packet_groups": [
                {
                    "group_id": "packet-001",
                    "member_document_group_ids": [
                        group["group_id"] for group in groups
                    ],
                    "resolved_document_order": [
                        group["group_id"] for group in groups
                    ],
                }
            ],
        }
    )

    group_by_page = {
        page_id: group["group_id"]
        for group in groups
        for page_id in group["member_page_ids"]
    }
    manifest_pages: list[dict[str, Any]] = []
    for page in evidence:
        entry = _page_manifest_entry(page, semantic_report)
        entry["packet_page"] = packet_pagination["pages"][page.page_id]
        entry["document_group_id"] = group_by_page[page.page_id]
        manifest_pages.append(entry)
    manifest = {
        "version": PAGE_SORTING_VERSION,
        "mode": mode,
        "page_count": len(evidence),
        "anchor_count": len(groups),
        "grouping_strategy": "packet_document_segmentation",
        "packet_pagination": dict(packet_pagination),
        "pages": manifest_pages,
        "semantic_quality": _semantic_quality_summary(semantic_report),
    }
    return manifest, report


def _candidate_dict(candidate: PageCandidate | None) -> dict[str, Any] | None:
    return asdict(candidate) if candidate is not None else None


def _sort_group(group: DocumentGroup) -> dict[str, Any]:
    selected: dict[str, PageCandidate] = {}
    page_details: list[dict[str, Any]] = []
    ambiguous: dict[str, str] = {}
    for page in sorted(group.members, key=lambda item: item.physical_page_idx):
        candidate, reason = page.best_candidate(group.expected_total)
        detail = {
            "page_id": page.page_id,
            "physical_page_idx": page.physical_page_idx,
            "source_page_idx": page.source_page_idx,
            "candidates": [asdict(item) for item in page.record.candidates],
            "selected": _candidate_dict(candidate) if reason == "selected" else None,
            "status": reason,
        }
        page_details.append(detail)
        if candidate is not None and reason == "selected":
            selected[page.page_id] = candidate
        elif reason != "selected":
            ambiguous[page.page_id] = reason

    by_number: dict[int, list[str]] = {}
    for page_id, candidate in selected.items():
        by_number.setdefault(candidate.current, []).append(page_id)
    duplicates = {
        str(number): sorted(page_ids)
        for number, page_ids in by_number.items()
        if len(page_ids) > 1
    }
    missing = [
        number
        for number in range(1, group.expected_total + 1)
        if number not in by_number
    ]
    unexpected = sorted(
        number
        for number in by_number
        if number < 1 or number > group.expected_total
    )
    complete = bool(
        group.members
        and len(group.members) == group.expected_total
        and len(selected) == len(group.members)
        and not ambiguous
        and not duplicates
        and not unexpected
        and not missing
    )
    if complete:
        status = "complete"
        resolved_order = [
            page_id
            for _number, page_id in sorted(
                (candidate.current, page_id) for page_id, candidate in selected.items()
            )
        ]
    elif not selected:
        status = "no_page_numbers"
        resolved_order = None
    elif duplicates or ambiguous:
        status = "conflict"
        resolved_order = None
    else:
        status = "incomplete"
        resolved_order = None
    return {
        "group_id": group.group_id,
        "seed_page_id": group.seed_page_id,
        "expected_total": group.expected_total,
        "member_page_ids": [
            page.page_id for page in sorted(group.members, key=lambda item: item.physical_page_idx)
        ],
        "identifiers": {
            key: sorted(values) for key, values in sorted(group.identifiers.items())
        },
        "family_tokens": sorted(group.family_tokens),
        "status": status,
        "resolved_order": resolved_order,
        "duplicates": duplicates,
        "missing_numbers": missing,
        "unexpected_numbers": unexpected,
        "ambiguous_pages": ambiguous,
        "pages": page_details,
        "decisions": list(group.decisions),
    }


def _semantic_quality_summary(
    semantic_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(semantic_report, Mapping):
        return None
    keys = (
        "semantic_markdown_version",
        "pages",
        "pages_emitted",
        "text_bearing_pages",
        "text_bearing_pages_not_emitted",
        "trace_accounting_ratio",
        "unmatched_source_records",
        "fragment_heavy_pages",
        "unstructured_table_fallback_pages",
        "trace_counts",
    )
    return {key: semantic_report[key] for key in keys if key in semantic_report}


def _page_manifest_entry(page: PageEvidence, semantic_report: Mapping[str, Any] | None) -> dict[str, Any]:
    semantic_flags: list[str] = []
    if isinstance(semantic_report, Mapping):
        page_number = page.physical_page_idx + 1
        fragment_pages = semantic_report.get("fragment_heavy_pages", [])
        fallback_pages = semantic_report.get("unstructured_table_fallback_pages", [])
        omitted_pages = semantic_report.get("text_bearing_pages_not_emitted", [])
        if isinstance(fragment_pages, list) and page_number in fragment_pages:
            semantic_flags.append("fragment_heavy")
        if isinstance(fallback_pages, list) and page_number in fallback_pages:
            semantic_flags.append("unstructured_table_fallback")
        if isinstance(omitted_pages, list) and page_number in omitted_pages:
            semantic_flags.append("text_bearing_page_not_emitted")
    selected, selection_status = page.best_candidate()
    return {
        "page_id": page.page_id,
        "physical_page_idx": page.physical_page_idx,
        "source_page_idx": page.source_page_idx,
        "page_size": list(page.page_size) if page.page_size is not None else None,
        "evidence_text_count": page.evidence_text_count,
        "family_tokens": sorted(page.family_tokens),
        "identifiers": {
            key: list(values) for key, values in sorted(page.identifiers.items())
        },
        "document_kind": page.document_kind,
        "document_title": page.document_title,
        "selection_status": selection_status,
        "selected_candidate": (
            asdict(selected) if selection_status == "selected" else None
        ),
        "candidates": [asdict(candidate) for candidate in page.record.candidates],
        "semantic_flags": semantic_flags,
    }


def analyze_middle_json(
    middle_json: Mapping[str, Any],
    page_order: Sequence[Mapping[str, Any]] | None = None,
    semantic_report: Mapping[str, Any] | None = None,
    mode: str = REPORT_ONLY_MODE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(manifest, report)`` without changing the input payload."""

    if mode != REPORT_ONLY_MODE:
        raise ValueError(f"Unsupported page sorting mode: {mode}")
    pages = page_order if page_order is not None else middle_json.get("pdf_info", [])
    if not isinstance(pages, list):
        raise ValueError("middle_json.pdf_info must be a list")
    evidence = build_evidence(middle_json, page_order)

    # A fused packet can carry one continuous physical page sequence while
    # containing several adjacent documents.  Validate that wrapper sequence
    # first; only then use strong document titles/identifiers to segment it.
    packet_pagination = _detect_packet_pagination(evidence)
    if packet_pagination is not None:
        packet_segments = _segment_packet_documents(evidence)
        if packet_segments:
            return _analyze_packet_segments(
                evidence,
                packet_segments,
                packet_pagination,
                semantic_report,
                mode,
            )

    anchors: list[tuple[PageEvidence, PageCandidate]] = []
    for page in evidence:
        candidate, reason = page.best_candidate()
        if (
            candidate is not None
            and reason == "selected"
            and candidate.explicit
            and candidate.current == 1
            and candidate.total is not None
        ):
            anchors.append((page, candidate))

    groups: list[DocumentGroup] = []
    for index, (page, candidate) in enumerate(
        sorted(anchors, key=lambda item: item[0].physical_page_idx),
        start=1,
    ):
        group = DocumentGroup(
            group_id=f"doc-{index:03d}",
            seed_page_id=page.page_id,
            expected_total=candidate.total or 0,
        )
        _add_to_group(group, page)
        groups.append(group)

    anchor_ids = {page.page_id for page, _candidate in anchors}
    remaining = [page for page in evidence if page.page_id not in anchor_ids]
    unresolved: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []

    if not groups:
        unresolved.update(
            {
                page.page_id: {
                    "page_id": page.page_id,
                    "physical_page_idx": page.physical_page_idx,
                    "reason": "no_page_one_anchor",
                }
                for page in remaining
            }
        )

    # Process stronger identifiers and explicit page markers first.  Revisit
    # unresolved pages after each pass so family evidence can accumulate.
    for _round in range(max(1, len(remaining))):
        if not remaining or not groups:
            break
        progress = False
        ordered = sorted(remaining, key=_page_strength, reverse=True)
        next_remaining: list[PageEvidence] = []
        for page in ordered:
            scored: list[tuple[float, DocumentGroup, list[str], PageCandidate | None]] = []
            for group in groups:
                score, reasons, candidate = _group_score(page, group)
                if score != float("-inf"):
                    scored.append((score, group, reasons, candidate))
            scored.sort(key=lambda item: (-item[0], item[1].group_id))
            if not scored:
                unresolved[page.page_id] = {
                    "page_id": page.page_id,
                    "physical_page_idx": page.physical_page_idx,
                    "reason": "no_compatible_group",
                }
                next_remaining.append(page)
                continue
            best_score, best_group, reasons, candidate = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else float("-inf")
            ocr_near_id = any(
                reason.endswith("_ocr_near_match") for reason in reasons
            )
            exact_id = any(
                reason.endswith("_match") and not reason.endswith("_ocr_near_match")
                for reason in reasons
            )
            minimum_score = 25.0
            if exact_id:
                margin_required = 8.0
            elif ocr_near_id:
                margin_required = 10.0
            else:
                margin_required = 12.0
            ambiguous = best_score < minimum_score or (
                len(scored) > 1 and best_score - second_score < margin_required
            )
            duplicate_slot = "duplicate_logical_slot" in reasons
            if ambiguous or (duplicate_slot and not (exact_id or ocr_near_id)):
                unresolved[page.page_id] = {
                    "page_id": page.page_id,
                    "physical_page_idx": page.physical_page_idx,
                    "reason": "ambiguous_group",
                    "candidates": [
                        {
                            "group_id": group.group_id,
                            "score": round(score, 3),
                            "reasons": why,
                        }
                        for score, group, why, _item in scored
                    ],
                }
                next_remaining.append(page)
                continue

            _add_to_group(best_group, page)
            decision = {
                "page_id": page.page_id,
                "group_id": best_group.group_id,
                "score": round(best_score, 3),
                "reasons": reasons,
                "selected_page_number": candidate.current if candidate else None,
            }
            best_group.decisions.append(decision)
            decisions.append(decision)
            progress = True
            unresolved.pop(page.page_id, None)
        remaining = next_remaining
        if not progress:
            break

    # Pages left after the final pass retain the most recent ambiguity detail.
    for page in remaining:
        unresolved.setdefault(
            page.page_id,
            {
                "page_id": page.page_id,
                "physical_page_idx": page.physical_page_idx,
                "reason": "unresolved",
            },
        )

    group_reports = [_sort_group(group) for group in groups]
    if not evidence:
        status = "empty"
    elif not anchors:
        status = "no_page_anchors"
    elif not unresolved and all(item["status"] == "complete" for item in group_reports):
        status = "complete"
    else:
        status = "needs_review"

    proposed_document_order = [
        {
            "group_id": item["group_id"],
            "seed_page_id": item["seed_page_id"],
            "resolved_order": item["resolved_order"],
            "status": item["status"],
        }
        for item in group_reports
    ]
    report = SortingReport(
        version=PAGE_SORTING_VERSION,
        mode=mode,
        status=status,
        page_count=len(evidence),
        anchor_count=len(anchors),
        group_count=len(groups),
        can_auto_sort=status == "complete",
        physical_order=[page.page_id for page in evidence],
        proposed_document_order=proposed_document_order,
        unresolved=sorted(
            unresolved.values(), key=lambda item: item["physical_page_idx"]
        ),
        groups=group_reports,
        semantic_quality=_semantic_quality_summary(semantic_report),
    ).to_dict()
    report["assignment_count"] = len(decisions)
    report["unresolved_count"] = len(unresolved)
    report["report_only"] = True

    manifest = {
        "version": PAGE_SORTING_VERSION,
        "mode": mode,
        "page_count": len(evidence),
        "anchor_count": len(anchors),
        "pages": [_page_manifest_entry(page, semantic_report) for page in evidence],
        "semantic_quality": _semantic_quality_summary(semantic_report),
    }
    return manifest, report


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_page_sorting_reports(
    middle_json_path: str | Path,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    semantic_report_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Analyze a fused middle JSON and write only new report artifacts."""

    middle_path = Path(middle_json_path)
    settings = config or {}
    mode = str(settings.get("mode", REPORT_ONLY_MODE))
    semantic_report = None
    if settings.get("include_semantic_diagnostics", True):
        candidate = (
            Path(semantic_report_path)
            if semantic_report_path is not None
            else middle_path.with_name(
                middle_path.name.replace("_middle.json", "_semantic_report.json")
            )
        )
        if candidate.is_file():
            try:
                semantic_report = _load_json(candidate)
            except (OSError, json.JSONDecodeError, ValueError):
                semantic_report = None
    manifest, report = analyze_middle_json(
        _load_json(middle_path),
        semantic_report=semantic_report,
        mode=mode,
    )
    suffix = "_middle.json"
    stem = middle_path.name[: -len(suffix)] if middle_path.name.endswith(suffix) else middle_path.stem
    output_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else middle_path.with_name(f"{stem}_sorting_manifest.json")
    )
    output_report = (
        Path(report_path)
        if report_path is not None
        else middle_path.with_name(f"{stem}_sorting_report.json")
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_manifest, output_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("middle_json", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--semantic-report", type=Path)
    args = parser.parse_args(argv)
    manifest, report = write_page_sorting_reports(
        args.middle_json,
        args.manifest,
        args.report,
        semantic_report_path=args.semantic_report,
    )
    print(json.dumps({"manifest": str(manifest), "report": str(report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Section-Aware, Token-Budgeted Document Chunker (Module M-18, Ingestion Pipeline).

Splits parsed guideline documents into retrieval chunks:

* **Token budget measured with the embedder's tokenizer** (default 400 tokens,
  60-token overlap). ``bge-small-en-v1.5`` and ``ms-marco-MiniLM-L-6-v2`` both
  truncate at 512 word pieces, so chunks must stay well below that once the
  query is added for reranking (see :mod:`medbridge.ingestion.tokenizer`).
* **Section-aware**: a heading closes the running chunk (no overlap across
  sections); tiny sections are merged forward instead of producing fragments.
  Generic headings ("Synopsis", "Recommendation-Specific Supportive Text") are
  qualified with the last numbered/major heading.
* **Page-aware**: every chunk records ``page_number`` (first page) and
  ``page_end`` (last page) of the text it contains.
* **Table-aware**: tables become markdown ``content_type="table"`` chunks (rows
  split with the header repeated); their text is not duplicated in prose.
* **Clean**: bibliography entries and table-of-contents lines are dropped before
  packing (so a short section next to them is not lost), reference-list / TOC
  chunks are dropped as a backstop, and chunks are de-duplicated by a hash of
  their normalised text.

Technical Specification Part V §3.3, ADL-003 (superseded sizes: see ADL corpus).
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field, model_validator

from medbridge.ingestion.parser import ParsedDocument, ParsedPage, ParsedTable, TextBlock
from medbridge.ingestion.tokenizer import get_token_counter, whitespace_token_count

logger = structlog.get_logger(__name__)

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_MIN_SECTION_TOKENS = 40
# Chunks below this size (e.g. an orphan heading) are not emitted.
MIN_EMIT_TOKENS = 8
MAX_TITLE_CHARS = 160

ContentType = Literal["text", "table"]

# Legacy line-based section patterns (used when a page carries plain text only).
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:Section|SECTION)\s+\d+(?:\.\d+)*[:.]?\s+\S.*"),
    re.compile(r"^(?:Chapter|CHAPTER)\s+\d+[:.]?\s+\S.*"),
    re.compile(r"^\d+\.\d+(?:\.\d+)*\.?\s+[A-Z].{0,120}$"),
    re.compile(r"^(?=(?:.*[A-Z]{2,}){2})[A-Z][A-Z0-9 ,\-:&/()]{3,80}$"),
]
_GENERIC_HEADINGS = frozenset(
    {
        "synopsis", "recommendation-specific supportive text", "supportive text",
        "recommendations", "recommendation", "background", "summary", "overview",
        "introduction", "key points", "rationale", "rationale and impact",
        "why the committee made the recommendations",
        "how the recommendations might affect practice", "evidence", "remarks",
        "practice points", "implementation remarks", "evidence-to-decision considerations",
        "definition", "definitions", "treatment", "diagnosis", "management", "tables",
    }
)
_NUMBERED_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+\S")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9“\"(\[•])|\s+(?=•\s)")
_NORMALISE_RE = re.compile(r"[^a-z0-9]+")
_DOT_LEADER_RE = re.compile(r"(?:\.\s?){5,}")
_TRAILING_NUMBER_RE = re.compile(r"[\s\d.:]+$")
_TOC_TITLES = frozenset({"contents", "table of contents"})
_TOC_LINE_RE = re.compile(r"(?:\.\s?){5,}\s*[A-Za-z]?\d+\s*$")
_REF_ENTRY_RE = re.compile(r"^\s*(?:\[?\d{1,4}[\].]?\s*)?[A-Z][A-Za-z'\-]+(?:\s[A-Z]{1,3})?,\s")
_REF_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}\s*;\s*\d+"),
    re.compile(r"\bdoi\b|doi\.org", re.IGNORECASE),
    re.compile(r"\bet al\b"),
    re.compile(r"\bPMID\b|\bPMCID\b"),
)


class RawChunk(BaseModel):
    """An unindexed guideline chunk with the metadata stored in its Qdrant payload."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'aha_acc_htn_2025_p12_c40'")
    guideline_id: str = Field(..., description="Guideline source identifier, e.g. 'AHA_ACC_HTN_2025'")
    section_title: str = Field(default="General", description="Section or chapter title")
    page_number: int = Field(default=1, description="First source page (1-indexed)")
    chunk_text: str = Field(..., description="Full text content of the chunk")
    source_url: str = Field(default="", description="Reference document or guideline URL")
    page_end: int = Field(default=0, description="Last source page covered by the chunk")
    chunk_index: int = Field(default=0, ge=0, description="0-based position within the document")
    content_type: ContentType = "text"
    source_format: Literal["pdf", "html"] = Field(
        default="pdf", description="'html': page_number/page_end are section indices"
    )
    doc_title: str = ""
    org: str = ""
    year: int = 0
    topics: list[str] = Field(default_factory=list)
    doc_sha256: str = ""
    token_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _default_page_end(self) -> RawChunk:
        if self.page_end < self.page_number:
            self.page_end = self.page_number
        return self

    def to_payload(self) -> dict[str, Any]:
        """Qdrant payload: interface contract §3.5 keys plus the additive ``source_format``."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "guideline_id": self.guideline_id,
            "doc_title": self.doc_title,
            "org": self.org,
            "year": self.year,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "source_url": self.source_url,
            "topics": list(self.topics),
            "doc_sha256": self.doc_sha256,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
            "source_format": self.source_format,
        }


@dataclass(frozen=True)
class DocumentMetadata:
    """Document-level metadata copied onto every chunk (from the corpus manifest)."""

    doc_title: str = ""
    org: str = ""
    year: int = 0
    topics: tuple[str, ...] = ()
    doc_sha256: str = ""


@dataclass
class ChunkingStats:
    """Counters describing one chunking run (reported by ``--stats``)."""

    text_chunks: int = 0
    table_chunks: int = 0
    duplicates_dropped: int = 0
    reference_chunks_dropped: int = 0
    toc_chunks_dropped: int = 0
    boilerplate_paragraphs_dropped: int = 0
    tiny_chunks_dropped: int = 0


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int
    section: str
    para_start: bool = False
    heading: bool = False


@dataclass
class _Draft:
    text: str
    section: str
    page: int
    page_end: int
    content_type: ContentType
    tokens: int


@dataclass
class _DocState:
    section: str = "General"
    last_major: str = ""
    buffer: list[_Unit] = field(default_factory=list)
    overlap_units: int = 0
    drafts: list[_Draft] = field(default_factory=list)


def count_tokens(text: str) -> int:
    """Whitespace word count (kept for backwards compatibility; not the chunk budget)."""
    return whitespace_token_count(text)


def normalised_text_hash(text: str) -> str:
    """Hash of lower-cased alphanumeric content, used for chunk de-duplication."""
    normalised = _NORMALISE_RE.sub(" ", text.lower()).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def looks_like_reference_list(text: str) -> bool:
    """True when a chunk is dominated by bibliographic citations."""
    hits = sum(len(p.findall(text)) for p in _REF_PATTERNS)
    words = max(len(text.split()), 1)
    return hits >= 4 and hits * 100 / words >= 2.5


def is_generic_heading(heading: str) -> bool:
    """True for boilerplate sub-headings, e.g. "Recommendation-Specific Supportive Text 1."."""
    return _TRAILING_NUMBER_RE.sub("", heading.strip().lower()) in _GENERIC_HEADINGS


def looks_like_reference_entry(text: str) -> bool:
    """True for a single bibliography entry ("12. Smith JA, ... 2019;37:100-10.")."""
    if len(text.split()) > 150 or not _REF_ENTRY_RE.match(text):
        return False
    return bool(_REF_PATTERNS[0].search(text) or _REF_PATTERNS[1].search(text))


def is_boilerplate_paragraph(text: str) -> bool:
    """Paragraph-level filter: reference entries/lists and table-of-contents lines."""
    return (
        looks_like_reference_list(text)
        or looks_like_reference_entry(text)
        or bool(_TOC_LINE_RE.search(text))
        or len(_DOT_LEADER_RE.findall(text)) >= 2
    )


def looks_like_table_of_contents(text: str, section: str = "") -> bool:
    """True when a chunk is a table of contents (dot leaders before page numbers)."""
    leaders = len(_DOT_LEADER_RE.findall(text))
    in_contents = section.strip().lower() in _TOC_TITLES
    return leaders >= 4 or (in_contents and leaders >= 1)


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences / bullet items (packing granularity)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


class SectionAwareChunker:
    """Section-, page- and table-aware chunker with a model-token budget."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        token_counter: Callable[[str], int] | None = None,
        min_section_tokens: int = DEFAULT_MIN_SECTION_TOKENS,
        drop_reference_lists: bool = True,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Maximum model tokens per chunk (including the section prefix).
            chunk_overlap: Tokens carried over between consecutive chunks of a section.
            token_counter: ``count(text) -> int``; defaults to the embedder's tokenizer.
                Counts must be additive over whitespace-separated words (true for
                BERT WordPiece tokenizers).
            min_section_tokens: Sections smaller than this are merged into the next one.
            drop_reference_lists: Drop bibliography entries / reference-list chunks and
                table-of-contents lines.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        if min_section_tokens < 0:
            raise ValueError("min_section_tokens must be non-negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_tokens = min_section_tokens
        self.drop_reference_lists = drop_reference_lists
        counter = token_counter or get_token_counter()
        self._word_tokens: Callable[[str], int] = lru_cache(maxsize=262_144)(counter)
        self.stats = ChunkingStats()

    # ------------------------------------------------------------------ tokens
    def count(self, text: str) -> int:
        """Model tokens in ``text`` (sum over whitespace-separated words)."""
        return sum(self._word_tokens(word) for word in text.split())

    def _prefix(self, section: str) -> str:
        if not section or section == "General":
            return ""
        words = section.split()
        limit = max(self.chunk_size // 4, 1)
        while words and self.count(" ".join(words)) > limit:
            words.pop()
        return " ".join(words)

    # ---------------------------------------------------------------- headings
    def is_section_header(self, line: str) -> bool:
        """Legacy check: does a plain-text line look like a section heading?"""
        stripped = line.strip()
        if len(stripped) < 4:
            return False
        return any(pattern.match(stripped) is not None for pattern in _SECTION_PATTERNS)

    @staticmethod
    def _qualify(heading: str, last_major: str) -> str:
        title = heading.strip().rstrip(":")
        if is_generic_heading(title) and last_major:
            title = f"{last_major} > {title}"
        return title[:MAX_TITLE_CHARS]

    # ------------------------------------------------------------------ public
    def chunk_document(
        self,
        document: ParsedDocument,
        metadata: DocumentMetadata | None = None,
        *,
        seen_hashes: set[str] | None = None,
        start_index: int = 0,
    ) -> list[RawChunk]:
        """Chunk a parsed document into validated :class:`RawChunk` items.

        Args:
            document: Parsed document (pages with structured blocks or plain text).
            metadata: Manifest metadata copied onto every chunk.
            seen_hashes: Shared normalised-text hashes for cross-document de-duplication.
            start_index: First ``chunk_index`` to assign.
        """
        drafts = self._draft_document(document.pages)
        return self._finalise(document.guideline_id, document.source_url, drafts,
                              metadata or DocumentMetadata(), seen_hashes, start_index,
                              source_format=document.source_format)

    def chunk_text(
        self,
        text: str,
        guideline_id: str = "GUIDELINE",
        page_number: int = 1,
        source_url: str = "",
        start_index: int = 1,
        initial_section: str = "General",
    ) -> list[RawChunk]:
        """Chunk a plain-text string (legacy API, single page)."""
        if not text or not text.strip():
            return []
        page = ParsedPage(page_number=page_number, text=text)
        drafts = self._draft_document([page], initial_section=initial_section)
        return self._finalise(guideline_id, source_url, drafts, DocumentMetadata(), None,
                              start_index)

    # ----------------------------------------------------------------- drafting
    def _blocks_from_text(self, page: ParsedPage) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        paragraph: list[str] = []

        def close() -> None:
            if paragraph:
                blocks.append(TextBlock(kind="paragraph", text=" ".join(paragraph),
                                        page_number=page.page_number))
                paragraph.clear()

        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped:
                close()
            elif self.is_section_header(stripped):
                close()
                blocks.append(TextBlock(kind="heading", text=stripped,
                                        page_number=page.page_number))
            else:
                paragraph.append(stripped)
        close()
        return blocks

    def _draft_document(
        self, pages: Iterable[ParsedPage], initial_section: str = "General"
    ) -> list[_Draft]:
        state = _DocState(section=initial_section)
        for page in pages:
            blocks = page.blocks or self._blocks_from_text(page)
            for block in blocks:
                page_no = block.page_number or page.page_number
                if block.kind == "heading":
                    self._on_heading(state, block.text, page_no)
                elif block.kind == "table" and block.table is not None:
                    state.drafts.extend(self._table_drafts(block.table, state.section, page_no))
                elif self.drop_reference_lists and is_boilerplate_paragraph(block.text):
                    self.stats.boilerplate_paragraphs_dropped += 1
                else:
                    for i, sentence in enumerate(split_sentences(block.text)):
                        self._add_unit(state, _Unit(
                            text=sentence, tokens=self.count(sentence), page=page_no,
                            section=state.section, para_start=i == 0,
                        ))
        self._emit(state)
        return state.drafts

    def _on_heading(self, state: _DocState, heading: str, page_no: int) -> None:
        title = self._qualify(heading, state.last_major)
        if not is_generic_heading(heading):
            state.last_major = heading.strip().rstrip(":")[:MAX_TITLE_CHARS]
        body_tokens = sum(u.tokens for u in state.buffer[state.overlap_units:])
        if body_tokens >= self.min_section_tokens:
            self._emit(state)
        elif state.overlap_units:
            # Overlap only continues a section; never carry it into a new one.
            del state.buffer[: state.overlap_units]
            state.overlap_units = 0
        state.section = title
        self._add_unit(state, _Unit(text=heading.strip(), tokens=self.count(heading),
                                    page=page_no, section=title, para_start=True,
                                    heading=True))

    def _budget(self, section: str) -> int:
        prefix = self._prefix(section)
        return self.chunk_size - (self.count(prefix) if prefix else 0)

    def _add_unit(self, state: _DocState, unit: _Unit) -> None:
        max_unit = max(self._budget(unit.section) - self.chunk_overlap, 8)
        if unit.tokens > max_unit:
            for piece in self._split_words(unit, max_unit):
                self._add_unit(state, piece)
            return
        budget = self._budget(self._dominant_section([*state.buffer, unit]))
        used = sum(u.tokens for u in state.buffer)
        if used + unit.tokens > budget:
            if state.overlap_units == len(state.buffer):
                # Only carried-over context left: trim it rather than emit a duplicate.
                while state.buffer and used + unit.tokens > budget:
                    used -= state.buffer.pop(0).tokens
                    state.overlap_units -= 1
            else:
                self._emit(state, carry_overlap=True)
        state.buffer.append(unit)

    def _split_words(self, unit: _Unit, max_tokens: int) -> list[_Unit]:
        pieces: list[_Unit] = []
        words: list[str] = []
        tokens = 0
        for word in unit.text.split():
            cost = self._word_tokens(word)
            if words and tokens + cost > max_tokens:
                pieces.append(_Unit(text=" ".join(words), tokens=tokens, page=unit.page,
                                    section=unit.section, para_start=not pieces and
                                    unit.para_start))
                words, tokens = [], 0
            words.append(word)
            tokens += cost
        if words:
            pieces.append(_Unit(text=" ".join(words), tokens=tokens, page=unit.page,
                                section=unit.section, para_start=not pieces and unit.para_start))
        return pieces

    @staticmethod
    def _dominant_section(units: list[_Unit]) -> str:
        weights: dict[str, int] = {}
        for u in units:
            weights[u.section] = weights.get(u.section, 0) + u.tokens
        return max(weights, key=lambda k: weights[k]) if weights else "General"

    def _emit(self, state: _DocState, carry_overlap: bool = False) -> None:
        units = state.buffer
        if not units or state.overlap_units == len(units):
            state.buffer, state.overlap_units = [], 0
            return
        section = self._dominant_section(units)
        body = units[0].text
        for prev, u in pairwise(units):
            separator = "\n" if (u.heading or u.para_start or prev.heading) else " "
            body = f"{body}{separator}{u.text}"
        prefix = "" if units[0].heading else self._prefix(section)
        text = f"{prefix}\n\n{body}" if prefix else body
        state.drafts.append(_Draft(
            text=text, section=section, page=min(u.page for u in units),
            page_end=max(u.page for u in units), content_type="text",
            tokens=sum(u.tokens for u in units) + (self.count(prefix) if prefix else 0),
        ))
        state.buffer, state.overlap_units = [], 0
        if carry_overlap and self.chunk_overlap:
            tail: list[_Unit] = []
            budget = self.chunk_overlap
            for u in reversed(units):
                if u.heading or u.tokens > budget:
                    break
                tail.insert(0, u)
                budget -= u.tokens
            if not tail and not units[-1].heading:
                last = units[-1]
                words = last.text.split()
                kept: list[str] = []
                tokens = 0
                for word in reversed(words):
                    cost = self._word_tokens(word)
                    if tokens + cost > self.chunk_overlap:
                        break
                    kept.insert(0, word)
                    tokens += cost
                if kept:
                    tail = [_Unit(text=" ".join(kept), tokens=tokens, page=last.page,
                                  section=last.section)]
            state.buffer = tail
            state.overlap_units = len(tail)

    def _table_drafts(self, table: ParsedTable, section: str, page_no: int) -> list[_Draft]:
        header_lines = [line for line in (self._prefix(section), table.caption) if line]
        header = "\n".join(header_lines)
        base = self.count(header) + self.count(table.to_markdown(rows=[]))
        drafts: list[_Draft] = []
        batch: list[list[str]] = []
        batch_tokens = 0

        def close() -> None:
            nonlocal batch, batch_tokens
            if batch:
                md = table.to_markdown(rows=batch)
                text = f"{header}\n\n{md}" if header else md
                drafts.append(_Draft(text=text, section=section, page=page_no,
                                     page_end=page_no, content_type="table",
                                     tokens=base + batch_tokens))
            batch, batch_tokens = [], 0

        for row in table.rows[1:]:
            # Exact: tokens are additive over the newline-joined markdown rows.
            row_tokens = self.count(table.to_markdown(rows=[row])) - self.count(
                table.to_markdown(rows=[]))
            if base + row_tokens > self.chunk_size:
                close()
                text = f"{header}\n\n" if header else ""
                for piece in self._split_words(
                    _Unit(text=" ; ".join(c for c in row if c), tokens=row_tokens, page=page_no,
                          section=section), self.chunk_size - self.count(header)):
                    drafts.append(_Draft(text=f"{text}{piece.text}", section=section,
                                         page=page_no, page_end=page_no, content_type="table",
                                         tokens=piece.tokens + self.count(header)))
                continue
            if batch and base + batch_tokens + row_tokens > self.chunk_size:
                close()
            batch.append(row)
            batch_tokens += row_tokens
        close()
        return drafts

    # --------------------------------------------------------------- finalising
    def _finalise(
        self,
        guideline_id: str,
        source_url: str,
        drafts: list[_Draft],
        metadata: DocumentMetadata,
        seen_hashes: set[str] | None,
        start_index: int,
        source_format: Literal["pdf", "html"] = "pdf",
    ) -> list[RawChunk]:
        seen = seen_hashes if seen_hashes is not None else set()
        chunks: list[RawChunk] = []
        index = start_index
        for draft in drafts:
            if draft.tokens < MIN_EMIT_TOKENS:
                self.stats.tiny_chunks_dropped += 1
                continue
            if draft.content_type == "text" and looks_like_table_of_contents(draft.text,
                                                                             draft.section):
                self.stats.toc_chunks_dropped += 1
                continue
            if (self.drop_reference_lists and draft.content_type == "text"
                    and looks_like_reference_list(draft.text)):
                self.stats.reference_chunks_dropped += 1
                continue
            digest = normalised_text_hash(draft.text)
            if digest in seen:
                self.stats.duplicates_dropped += 1
                continue
            seen.add(digest)
            if draft.content_type == "table":
                self.stats.table_chunks += 1
            else:
                self.stats.text_chunks += 1
            chunks.append(RawChunk(
                chunk_id=f"{guideline_id.lower()}_p{draft.page}_c{index}",
                guideline_id=guideline_id,
                section_title=draft.section or "General",
                page_number=max(draft.page, 1),
                page_end=max(draft.page_end, draft.page, 1),
                chunk_text=draft.text,
                source_url=source_url,
                chunk_index=index,
                content_type=draft.content_type,
                source_format=source_format,
                doc_title=metadata.doc_title,
                org=metadata.org,
                year=metadata.year,
                topics=list(metadata.topics),
                doc_sha256=metadata.doc_sha256,
                token_count=draft.tokens,
            ))
            index += 1
        logger.info(
            "Chunked document",
            guideline_id=guideline_id,
            chunks=len(chunks),
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        return chunks

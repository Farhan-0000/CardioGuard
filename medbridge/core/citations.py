"""
Citation-marker validation (ADL-028, contract §3.4).

The generator is shown the reranked evidence as numbered chunks
(``<chunk index="1" …>`` … ``index="5"``) and cites them inline as ``[n]``.
This module makes those citations trustworthy:

* a marker ``[n]`` maps to the *n*-th chunk passed to the generator (1-based,
  prompt order); markers that do not map to a chunk are removed;
* the surviving markers are renumbered ``[1], [2], …`` in order of first
  appearance, and ``[1, 3]`` style groups are normalised to ``[1][2]``;
* the citation list is rebuilt from the chunk metadata (source label, section,
  page, URL and the first ~350 characters of the chunk text). Excerpts or
  sources written by the LLM are never trusted.
* HTML sources (payload ``source_format == "html"``) have no physical pages —
  their ``page_number`` is a section index — so the citation's ``page`` is
  ``None`` and its section reads ``"<section title> (section N)"``.

The orchestrator uses ``valid == 0`` on an ANSWER to downgrade it to
GENERALIZE (an "answer" with no verifiable evidence is not an answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medbridge.api.schemas.responses import CitationResponse
from medbridge.retrieval.hybrid_retriever import RetrievedChunk

EXCERPT_CHARS = 350
_MAX_RANGE_SPAN = 4

# One bracket group: "[3]", "[1, 2]", "[1–3]". Two-digit numbers at most, so
# clinical ranges such as "[130-139 mmHg]" or years such as "[2024]" are left alone.
_MARKER_GROUP = re.compile(
    r"[ \t]*\[\s*(\d{1,2}(?:\s*[-–]\s*\d{1,2})?(?:\s*,\s*\d{1,2}(?:\s*[-–]\s*\d{1,2})?)*)\s*\]"
)


@dataclass
class CitationValidation:
    """Result of validating the inline markers of one generated response."""

    text: str
    citations: list[CitationResponse] = field(default_factory=list)
    valid: int = 0  # distinct chunks cited with a valid marker
    stripped: int = 0  # marker references removed because they map to no chunk
    cited_chunk_ids: list[str] = field(default_factory=list)


def _expand(group: str) -> list[int] | None:
    """Expand ``"1, 3-4"`` → ``[1, 3, 4]``; ``None`` if a range is implausible."""
    numbers: list[int] = []
    for part in group.split(","):
        bounds = re.split(r"\s*[-–]\s*", part.strip())
        if len(bounds) == 1:
            numbers.append(int(bounds[0]))
            continue
        start, end = int(bounds[0]), int(bounds[1])
        if not 0 < end - start <= _MAX_RANGE_SPAN:
            return None
        numbers.extend(range(start, end + 1))
    return numbers


def source_label(chunk: RetrievedChunk) -> str:
    """Human-readable source ``f"{org} {year}"``, e.g. ``"AHA/ACC 2025"``.

    Falls back to the document title, then the guideline id.
    """
    if chunk.org and chunk.year:
        return f"{chunk.org} {chunk.year}"
    if chunk.doc_title:
        return chunk.doc_title
    return chunk.guideline_id or "Clinical guideline"


def is_html_source(chunk: RetrievedChunk) -> bool:
    """HTML sources have no physical pages: ``page_number`` is a section index."""
    return (chunk.source_format or "").strip().lower() == "html"


def display_page(chunk: RetrievedChunk) -> int | None:
    """Physical page to show for a chunk; ``None`` for HTML sources and unknown pages."""
    if is_html_source(chunk):
        return None
    return chunk.page_number or None


def display_section(chunk: RetrievedChunk) -> str:
    """Section label; HTML sources read ``"<section title> (section N)"``."""
    title = (chunk.section_title or "").strip()
    if not is_html_source(chunk) or not chunk.page_number or chunk.page_number <= 0:
        return title
    suffix = f"section {chunk.page_number}"
    return f"{title} ({suffix})" if title else suffix.capitalize()


def make_excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """First ``limit`` characters of ``text``, cut at a word boundary."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def build_citation(index: int, chunk: RetrievedChunk) -> CitationResponse:
    """Build the citation for marker ``[index]`` from chunk metadata only."""
    return CitationResponse(
        marker=f"[{index}]",
        chunk_id=chunk.chunk_id,
        source=source_label(chunk),
        section=display_section(chunk),
        excerpt=make_excerpt(chunk.chunk_text),
        guideline_id=chunk.guideline_id or None,
        title=chunk.doc_title,
        org=chunk.org,
        year=chunk.year,
        page=display_page(chunk),
        url=chunk.source_url or None,
    )


def validate_citations(text: str, chunks: list[RetrievedChunk]) -> CitationValidation:
    """Strip invalid ``[n]`` markers, renumber valid ones and rebuild the citation list.

    Args:
        text: Generated response text with inline markers.
        chunks: The chunks exactly as passed to the generator (prompt order).
    """
    renumber: dict[int, int] = {}  # original chunk number → new marker number
    stripped = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal stripped
        numbers = _expand(match.group(1))
        if numbers is None:
            return match.group(0)
        kept: list[int] = []
        for n in numbers:
            if not 1 <= n <= len(chunks):
                stripped += 1
                continue
            if n not in renumber:
                renumber[n] = len(renumber) + 1
            if renumber[n] not in kept:
                kept.append(renumber[n])
        if not kept:
            return ""
        leading = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip(" \t"))]
        return leading + "".join(f"[{k}]" for k in kept)

    new_text = _MARKER_GROUP.sub(_replace, text)
    ordered = sorted(renumber.items(), key=lambda item: item[1])
    citations = [build_citation(new, chunks[orig - 1]) for orig, new in ordered]
    return CitationValidation(
        text=new_text,
        citations=citations,
        valid=len(citations),
        stripped=stripped,
        cited_chunk_ids=[c.chunk_id for c in citations],
    )

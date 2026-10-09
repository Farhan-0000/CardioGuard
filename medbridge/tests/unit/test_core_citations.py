"""Unit tests for citation-marker validation (medbridge/core/citations.py, ADL-028)."""

from __future__ import annotations

import pytest

from medbridge.core.citations import (
    EXCERPT_CHARS,
    display_page,
    display_section,
    make_excerpt,
    source_label,
    validate_citations,
)
from medbridge.retrieval.hybrid_retriever import RetrievedChunk


def chunk(i: int, **extra: object) -> RetrievedChunk:
    data: dict[str, object] = {
        "chunk_id": f"c{i}",
        "chunk_text": f"Evidence text {i}. " * 40,
        "guideline_id": f"GUIDE_{i}",
        "section_title": f"Section {i}",
        "page_number": i,
        "source_url": f"https://example.org/{i}",
        "org": "ESC",
        "year": 2024,
        "doc_title": f"Guideline {i}",
    }
    data.update(extra)
    return RetrievedChunk(**data)  # type: ignore[arg-type]


CHUNKS = [chunk(i) for i in range(1, 6)]


def test_valid_markers_kept_and_metadata_used():
    result = validate_citations("Lower salt helps [1]. Exercise too [2].", CHUNKS)
    assert result.text == "Lower salt helps [1]. Exercise too [2]."
    assert result.valid == 2 and result.stripped == 0
    assert result.cited_chunk_ids == ["c1", "c2"]
    first = result.citations[0]
    assert first.marker == "[1]"
    assert first.source == "ESC 2024"
    assert first.section == "Section 1"
    assert first.page == 1
    assert first.url == "https://example.org/1"
    assert first.guideline_id == "GUIDE_1" and first.title == "Guideline 1"
    assert first.excerpt.startswith("Evidence text 1.")


def test_renumbered_in_order_of_first_appearance():
    result = validate_citations("A [4]. B [2]. C [4][2]. D [5].", CHUNKS)
    assert result.text == "A [1]. B [2]. C [1][2]. D [3]."
    assert result.cited_chunk_ids == ["c4", "c2", "c5"]
    assert [c.marker for c in result.citations] == ["[1]", "[2]", "[3]"]


def test_out_of_range_markers_stripped_with_preceding_space():
    result = validate_citations("Claim [6]. Other [0]. Fine [3].", CHUNKS)
    assert result.text == "Claim. Other. Fine [1]."
    assert result.stripped == 2
    assert result.valid == 1


def test_grouped_and_range_markers_normalised():
    result = validate_citations("Combined [1, 3]. Range [2-4]. Bad [1, 9].", CHUNKS)
    assert result.text == "Combined [1][2]. Range [3][2][4]. Bad [1]."
    assert result.stripped == 1
    assert result.cited_chunk_ids == ["c1", "c3", "c2", "c4"]


@pytest.mark.parametrize(
    "text",
    [
        "Stage 1 is [130-139 mmHg] systolic.",
        "Published in [2024].",
        "A wide range [1-20] is not a citation group.",
    ],
)
def test_non_citation_brackets_untouched(text):
    result = validate_citations(text, CHUNKS)
    assert result.text == text
    assert result.valid == 0 and result.stripped == 0


def test_no_chunks_strips_every_marker():
    result = validate_citations("Fabricated [1] and [2].", [])
    assert result.text == "Fabricated and."
    assert result.valid == 0 and result.stripped == 2
    assert result.citations == []


def test_duplicate_markers_cited_once():
    result = validate_citations("X [1]. Y [1]. Z [1, 1].", CHUNKS)
    assert result.text == "X [1]. Y [1]. Z [1]."
    assert result.valid == 1
    assert len(result.citations) == 1


def test_llm_excerpt_never_trusted():
    long_chunk = chunk(1, chunk_text="word " * 500)
    result = validate_citations("Claim [1].", [long_chunk])
    excerpt = result.citations[0].excerpt
    assert excerpt.endswith("…")
    assert len(excerpt) <= EXCERPT_CHARS + 1


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, "ESC 2024"),
        ({"year": None}, "Guideline 1"),
        ({"year": None, "doc_title": None}, "GUIDE_1"),  # org alone is not a source label
        ({"year": None, "doc_title": None, "org": None}, "GUIDE_1"),
        ({"year": None, "doc_title": None, "org": None, "guideline_id": ""}, "Clinical guideline"),
    ],
)
def test_source_label_fallbacks(extra, expected):
    assert source_label(chunk(1, **extra)) == expected


def test_missing_page_and_url_become_none():
    result = validate_citations("Claim [1].", [chunk(1, page_number=0, source_url="")])
    assert result.citations[0].page is None
    assert result.citations[0].url is None


def test_make_excerpt_collapses_whitespace():
    assert make_excerpt("a\n\n  b\tc") == "a b c"
    assert make_excerpt("x" * 10, limit=20) == "x" * 10


def test_html_source_has_no_physical_page():
    html = chunk(3, source_format="html", section_title="Treatment", page_number=7)
    result = validate_citations("Claim [1].", [html])
    citation = result.citations[0]
    assert citation.page is None
    assert citation.section == "Treatment (section 7)"
    assert citation.source == "ESC 2024"


def test_pdf_source_keeps_page_and_plain_section():
    pdf = chunk(2, source_format="pdf", section_title="8.1 BP goals", page_number=34)
    assert display_page(pdf) == 34 and display_section(pdf) == "8.1 BP goals"
    legacy = chunk(2, section_title="8.1 BP goals", page_number=34)  # no source_format
    assert display_page(legacy) == 34 and display_section(legacy) == "8.1 BP goals"


@pytest.mark.parametrize(
    "extra,section",
    [
        ({"section_title": "", "page_number": 4}, "Section 4"),
        ({"section_title": "Overview", "page_number": 0}, "Overview"),
        ({"section_title": "Overview", "page_number": 2, "source_format": "HTML"}, None),
    ],
)
def test_html_section_label_edge_cases(extra, section):
    item = chunk(1, **{"source_format": "html", **extra})
    expected = section if section is not None else "Overview (section 2)"
    assert display_section(item) == expected
    assert display_page(item) is None


def test_source_label_is_org_year_then_title_then_id():
    assert source_label(chunk(1, org="NICE", year=2026)) == "NICE 2026"
    assert source_label(chunk(1, org="NICE", year=None, doc_title="Hypertension in adults")) == (
        "Hypertension in adults"
    )
    assert source_label(chunk(1, org="NICE", year=None, doc_title=None)) == "GUIDE_1"

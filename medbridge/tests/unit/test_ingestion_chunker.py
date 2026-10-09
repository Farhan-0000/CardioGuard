"""
Unit tests for the token-budgeted, section/page/table-aware chunker.

A deterministic whitespace token counter is injected so the tests are hermetic;
one test checks the real BERT tokenizer path only through its additive contract.
"""
import pytest

from medbridge.ingestion.chunker import (
    DocumentMetadata,
    SectionAwareChunker,
    is_boilerplate_paragraph,
    is_generic_heading,
    looks_like_reference_entry,
    looks_like_reference_list,
    looks_like_table_of_contents,
    normalised_text_hash,
    split_sentences,
)
from medbridge.ingestion.parser import ParsedDocument, ParsedPage, ParsedTable, TextBlock
from medbridge.ingestion.tokenizer import approximate_wordpiece_count, whitespace_token_count


def _chunker(**kwargs) -> SectionAwareChunker:
    kwargs.setdefault("token_counter", whitespace_token_count)
    return SectionAwareChunker(**kwargs)


def _doc(pages: list[list[TextBlock]], gid: str = "TEST_DOC") -> ParsedDocument:
    return ParsedDocument(
        guideline_id=gid, file_path="x.pdf", source_url="https://example.org/doc",
        pages=[ParsedPage(page_number=i + 1, text="", blocks=blocks)
               for i, blocks in enumerate(pages)],
    )


def _h(text: str, page: int) -> TextBlock:
    return TextBlock(kind="heading", text=text, page_number=page)


def _p(text: str, page: int) -> TextBlock:
    return TextBlock(kind="paragraph", text=text, page_number=page)


def _sentences(prefix: str, n: int, words: int = 9) -> str:
    return " ".join(
        f"{prefix} sentence {i} " + " ".join(["word"] * (words - 3)) + "." for i in range(n)
    )


def test_chunks_respect_budget_and_carry_metadata() -> None:
    chunker = _chunker(chunk_size=60, chunk_overlap=10)
    doc = _doc([[_h("1. Blood pressure targets", 1), _p(_sentences("Target", 12), 1)],
                [_p(_sentences("Continued", 12), 2)]])
    meta = DocumentMetadata(doc_title="Synthetic guideline", org="WHO", year=2021,
                            topics=("hypertension",), doc_sha256="f" * 64)
    chunks = chunker.chunk_document(doc, meta)

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= 60
        assert whitespace_token_count(chunk.chunk_text) <= 60
        assert chunk.org == "WHO" and chunk.year == 2021 and chunk.topics == ["hypertension"]
        assert chunk.doc_sha256 == "f" * 64
        assert chunk.source_url == "https://example.org/doc"
        assert chunk.page_end >= chunk.page_number
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # Continuation chunks repeat the section title as context.
    assert all(c.chunk_text.startswith("1. Blood pressure targets") for c in chunks)
    # A chunk straddling the page break records both pages.
    assert any(c.page_number == 1 and c.page_end == 2 for c in chunks)


def test_overlap_within_section_only() -> None:
    chunker = _chunker(chunk_size=40, chunk_overlap=9, min_section_tokens=5)
    doc = _doc([[_h("Section A", 1), _p(_sentences("Alpha", 8), 1),
                 _h("Section B", 1), _p(_sentences("Beta", 2), 1)]])
    chunks = chunker.chunk_document(doc)
    alpha = [c for c in chunks if c.section_title == "Section A"]
    beta = [c for c in chunks if c.section_title == "Section B"]
    assert len(alpha) >= 2 and len(beta) == 1
    # Consecutive chunks of the same section share the overlap sentence.
    last_sentence = alpha[0].chunk_text.split(". ")[-1]
    assert last_sentence.rstrip(".") in alpha[1].chunk_text
    # No Alpha text leaks into the next section.
    assert "Alpha" not in beta[0].chunk_text
    assert beta[0].chunk_text.startswith("Section B")


def test_small_sections_are_merged_forward() -> None:
    chunker = _chunker(chunk_size=100, chunk_overlap=10, min_section_tokens=20)
    doc = _doc([[_h("Abstract", 1), _p("Short abstract text here.", 1),
                 _h("1. Introduction", 1), _p(_sentences("Intro", 4), 1)]])
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "Short abstract text here." in chunks[0].chunk_text
    assert chunks[0].section_title == "1. Introduction"  # dominant section


def test_generic_headings_are_qualified() -> None:
    chunker = _chunker(chunk_size=200, chunk_overlap=10, min_section_tokens=5)
    doc = _doc([[_h("5.2. Medical management", 3), _p(_sentences("Drug", 3), 3),
                 _h("Synopsis", 3), _p(_sentences("Synopsis", 3), 3),
                 _h("Recommendation-Specific Supportive Text 1.", 3), _p(_sentences("Rst", 3), 3)]])
    titles = [c.section_title for c in chunker.chunk_document(doc)]
    assert titles == [
        "5.2. Medical management",
        "5.2. Medical management > Synopsis",
        "5.2. Medical management > Recommendation-Specific Supportive Text 1.",
    ]
    assert is_generic_heading("Synopsis") and not is_generic_heading("Atrial fibrillation")


def test_tables_become_table_chunks_with_repeated_header() -> None:
    chunker = _chunker(chunk_size=45, chunk_overlap=5)
    rows = [["Drug", "Class", "Usual dose"]] + [
        [f"Drug{i}", "calcium channel blocker", f"{i} mg once daily"] for i in range(8)
    ]
    table = ParsedTable(page_number=2, rows=rows, caption="Table 3. Oral antihypertensives")
    doc = _doc([[_h("Treatment", 1), _p(_sentences("Treat", 2), 1)],
                [TextBlock(kind="table", text=table.to_markdown(), page_number=2, table=table)]])
    chunks = chunker.chunk_document(doc)
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) >= 2
    for c in tables:
        assert c.page_number == 2 and c.page_end == 2
        assert "Table 3. Oral antihypertensives" in c.chunk_text
        assert "| Drug | Class | Usual dose |" in c.chunk_text
        assert c.token_count <= 45
    body = " ".join(c.chunk_text for c in tables)
    assert all(f"Drug{i}" in body for i in range(8))
    assert chunker.stats.table_chunks == len(tables)


def test_ragged_table_rows_stay_within_budget() -> None:
    chunker = _chunker(chunk_size=40, chunk_overlap=5)
    rows = [["Term", "Key words", "Notes"]] + [[f"Term {i} alpha beta"] for i in range(12)]
    table = ParsedTable(page_number=1, rows=rows, caption="Search criteria")
    doc = _doc([[TextBlock(kind="table", text=table.to_markdown(), page_number=1, table=table)]])
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert all(whitespace_token_count(c.chunk_text) <= 40 for c in chunks)


def test_duplicate_chunks_dropped_across_documents() -> None:
    chunker = _chunker(chunk_size=80, chunk_overlap=10)
    text = _sentences("Shared", 3)
    seen: set[str] = set()
    first = chunker.chunk_document(_doc([[_p(text, 1)]], "DOC_A"), seen_hashes=seen)
    second = chunker.chunk_document(_doc([[_p(text.upper(), 1)]], "DOC_B"), seen_hashes=seen)
    assert len(first) == 1 and second == []
    assert chunker.stats.duplicates_dropped == 1
    assert normalised_text_hash("Blood  pressure!") == normalised_text_hash("blood pressure")


def test_reference_lists_and_tocs_dropped() -> None:
    refs = " ".join(
        f"{i}. Author A, Author B, et al. Trial {i}. J Hypertens 2019;37:{i}00-10. "
        f"doi:10.1097/hjh.{i}" for i in range(1, 8)
    )
    toc = " ".join(f"{i}. Topic {i} . . . . . . . . . . {i * 3}" for i in range(1, 7))
    assert looks_like_reference_list(refs)
    assert not looks_like_reference_list(_sentences("Prose", 5))
    assert looks_like_table_of_contents(toc)
    assert looks_like_table_of_contents("1.1 Measuring ........ 5", section="Contents")
    assert not looks_like_table_of_contents("Measure BP... then wait.")

    chunker = _chunker(chunk_size=500, chunk_overlap=10)
    chunks = chunker.chunk_document(_doc([[_h("1. Key messages", 1), _p(_sentences("Key", 3), 1)],
                                          [_h("Literature", 2), _p(refs, 2)],
                                          [_h("Index", 3), _p(toc, 3)]]))
    # The short "Key messages" section survives; the bibliography and TOC paragraphs
    # are removed before packing so they cannot drag it into a dropped chunk.
    assert len(chunks) == 1 and "Key sentence 0" in chunks[0].chunk_text
    assert "J Hypertens" not in chunks[0].chunk_text and ". . ." not in chunks[0].chunk_text
    assert chunker.stats.boilerplate_paragraphs_dropped == 2


def test_single_reference_entries_and_toc_lines_detected() -> None:
    assert looks_like_reference_entry(
        "12. Whelton PK, Carey RM, Aronow WS, et al. 2017 Guideline. Hypertension "
        "2018;71:e13-e115."
    )
    assert looks_like_reference_entry("Smith J, Jones K. Title. doi:10.1161/abc")
    assert not looks_like_reference_entry(
        "Hypertension, defined as BP of 130/80 mm Hg or higher, affects many adults."
    )
    assert is_boilerplate_paragraph("1.4 Treating and monitoring hypertension ........ 10")
    assert not is_boilerplate_paragraph("Offer an ACE inhibitor or ARB to adults with diabetes.")


def test_oversized_sentence_is_split_by_words() -> None:
    chunker = _chunker(chunk_size=30, chunk_overlap=5)
    long_sentence = " ".join(f"w{i}" for i in range(100))
    chunks = chunker.chunk_text(long_sentence, guideline_id="LONG")
    assert all(c.token_count <= 30 for c in chunks)
    joined = " ".join(c.chunk_text for c in chunks)
    assert all(f"w{i}" in joined.split() for i in range(100))


def test_split_sentences_and_bullets() -> None:
    assert split_sentences("First point. Second point! • bullet one • bullet two") == [
        "First point.", "Second point!", "• bullet one", "• bullet two",
    ]


def test_approximate_counter_is_additive() -> None:
    text = "Amlodipine 5 mg; BP 130/80 mm Hg."
    words = text.split()
    assert approximate_wordpiece_count(text) == sum(approximate_wordpiece_count(w) for w in words)
    assert approximate_wordpiece_count(text) > len(words)
    assert approximate_wordpiece_count("") == 0


def test_empty_text_yields_no_chunks() -> None:
    assert _chunker().chunk_text("   ") == []


@pytest.mark.parametrize("size,overlap", [(400, 60), (256, 32)])
def test_default_style_configuration(size: int, overlap: int) -> None:
    chunker = _chunker(chunk_size=size, chunk_overlap=overlap)
    assert chunker.chunk_size == size and chunker.chunk_overlap == overlap

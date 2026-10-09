"""
Unit tests for the HTML parser (``medbridge.ingestion.html_parser``).

Covers main-content selection, removal of navigation/chrome/glossary pop-ups,
link-list filtering, headings -> virtual pages, skipped reference sections,
citation superscripts and table extraction. Hermetic (synthetic HTML only).
"""
from pathlib import Path

import pytest

from medbridge.ingestion.html_parser import extract_sections_from_html
from medbridge.ingestion.parser import PDFParseError, parse_document

PAGE = """<!DOCTYPE html>
<html><head><title>High Blood Pressure</title>
<style>.x { color: red }</style><script>var tracking = "do not index";</script></head>
<body>
<header><nav><ul><li><a href="/">Home</a></li><li><a href="/topics">Health Topics</a></li></ul></nav></header>
<main>
  <div role="navigation"><a href="/menu">MENU</a></div>
  <h1>High Blood Pressure</h1>
  <p>High blood pressure is common<sup><a href="#r1">1,2</a></sup>. It often has no
     <span class="glossary-term">symptoms
       <span class="usa-modal"><span class="glossarydefinition">GLOSSARY POPUP TEXT</span></span>
     </span>.</p>
  <ul class="related">
    <li><a href="/a">Related topic link one</a></li>
    <li><a href="/b">Related topic link two</a></li>
  </ul>
  <h2>Blood pressure categories</h2>
  <p>Readings are written as two numbers, for example 120/80 mm&nbsp;Hg.</p>
  <ul><li>Eat less <a href="/salt">salt</a> and more vegetables every day.</li></ul>
  <section class="tw">
    <h4>Table 1.</h4>
    <div class="caption"><p>Blood pressure levels in adults</p></div>
    <table>
      <thead><tr><th>Category</th><th>Systolic</th><th>Diastolic</th></tr></thead>
      <tbody>
        <tr><td>Normal</td><td>&lt;120</td><td>&lt;80</td></tr>
        <tr><td>Stage 2</td><td>140 or higher<br>(confirm)</td><td>90 or higher</td></tr>
      </tbody>
    </table>
  </section>
  <h3>Treatment</h3>
  <p>Lifestyle changes and medicines lower blood pressure.</p>
  <h2>References</h2>
  <ol><li>Smith J. Hypertension. 2020;75:1-10.</li></ol>
  <h2>Living with high blood pressure</h2>
  <p>Check your blood pressure at home with a validated cuff.</p>
  <div hidden>Hidden text must not appear.</div>
</main>
<footer><p>Footer text must not appear.</p></footer>
</body></html>
"""


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "medlineplus_high_blood_pressure.html"
    path.write_text(PAGE, encoding="utf-8")
    return path


def _all_text(doc) -> str:
    return "\n".join(p.text for p in doc.pages)


def test_main_text_extracted_and_chrome_removed(html_file: Path) -> None:
    doc = extract_sections_from_html(html_file, "MEDLINEPLUS_HIGH_BP", "https://example.gov/x")
    text = _all_text(doc)
    assert doc.source_format == "html"
    assert doc.guideline_id == "MEDLINEPLUS_HIGH_BP"
    assert "High blood pressure is common" in text
    for unwanted in ("Home", "Health Topics", "MENU", "tracking", "GLOSSARY POPUP TEXT",
                     "Hidden text", "Footer text", "color: red"):
        assert unwanted not in text


def test_citation_superscripts_and_link_lists_dropped(html_file: Path) -> None:
    text = _all_text(extract_sections_from_html(html_file))
    assert "common1,2" not in text and "common 1,2" not in text
    assert "Related topic link" not in text
    # A list item that merely *contains* a link is kept.
    assert "Eat less salt and more vegetables every day." in text


def test_headings_define_virtual_pages(html_file: Path) -> None:
    doc = extract_sections_from_html(html_file)
    headings = [(b.text, b.level, b.page_number) for p in doc.pages for b in p.blocks
                if b.kind == "heading"]
    assert headings == [
        ("High Blood Pressure", 1, 1),
        ("Blood pressure categories", 2, 2),
        ("Treatment", 3, 3),
        ("Living with high blood pressure", 2, 4),
    ]
    assert [p.page_number for p in doc.pages] == [1, 2, 3, 4]


def test_reference_sections_skipped(html_file: Path) -> None:
    text = _all_text(extract_sections_from_html(html_file))
    assert "Smith J" not in text
    assert "References" not in text
    assert "Check your blood pressure at home" in text  # section after references kept


def test_tables_become_structured_blocks(html_file: Path) -> None:
    doc = extract_sections_from_html(html_file)
    tables = [t for p in doc.pages for t in p.tables]
    assert len(tables) == 1
    table = tables[0]
    assert table.caption == "Table 1. Blood pressure levels in adults"
    assert table.rows[0] == ["Category", "Systolic", "Diastolic"]
    assert table.rows[2] == ["Stage 2", "140 or higher (confirm)", "90 or higher"]
    assert table.page_number == 2
    markdown = table.to_markdown()
    assert markdown.splitlines()[1] == "| --- | --- | --- |"
    # Table cell text is not duplicated as prose.
    prose = [b.text for p in doc.pages for b in p.blocks if b.kind == "paragraph"]
    assert not any("90 or higher" in t for t in prose)


def test_explicit_root_selector(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><body><div id='other'><p>Sidebar promotional text here.</p></div>"
        "<section id='topsum_section'><h2>Summary</h2><p>Only this summary counts.</p>"
        "</section></body></html>",
        encoding="utf-8",
    )
    doc = extract_sections_from_html(path, root="#topsum_section")
    text = _all_text(doc)
    assert "Only this summary counts." in text
    assert "Sidebar" not in text


def test_class_root_and_missing_root(tmp_path: Path) -> None:
    path = tmp_path / "pmc.html"
    path.write_text(
        "<html><body><section class='front-matter'><p>Author affiliations text.</p></section>"
        "<section class='body main-article-body'><h2>1. Introduction</h2>"
        "<p>Resistant hypertension is defined as uncontrolled BP on three drugs.</p></section>"
        "</body></html>",
        encoding="utf-8",
    )
    doc = extract_sections_from_html(path, root=".main-article-body")
    assert "Resistant hypertension" in _all_text(doc)
    assert "Author affiliations" not in _all_text(doc)
    with pytest.raises(PDFParseError):
        extract_sections_from_html(path, root="#does-not-exist")


def test_html_chunks_are_marked_as_html(html_file: Path) -> None:
    from medbridge.ingestion.chunker import SectionAwareChunker
    from medbridge.ingestion.tokenizer import whitespace_token_count

    chunker = SectionAwareChunker(token_counter=whitespace_token_count)
    chunks = chunker.chunk_document(extract_sections_from_html(html_file, "MP_TEST"))
    assert chunks and all(c.to_payload()["source_format"] == "html" for c in chunks)


def test_parse_document_dispatches_on_extension(html_file: Path, tmp_path: Path) -> None:
    assert parse_document(html_file).source_format == "html"
    with pytest.raises(PDFParseError):
        parse_document(tmp_path / "notes.txt")
    with pytest.raises(PDFParseError):
        extract_sections_from_html(tmp_path / "missing.html")

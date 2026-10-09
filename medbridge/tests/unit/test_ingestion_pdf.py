"""
Unit tests for PDF structure handling in ``medbridge.ingestion.parser``.

Synthetic PDFs (built with PyMuPDF) exercise running header/footer removal,
the references-tail cut, heading detection and table detection.
"""
from pathlib import Path

import pymupdf
import pytest

from medbridge.ingestion.parser import (
    ParsedTable,
    extract_pages_from_pdf,
    is_heading_text,
)

HEADER = "Journal of Synthetic Cardiology 2024;12:100-120"
BODY = [
    "Hypertension is defined as a blood pressure of 130/80 mm Hg or higher.",
    "Lifestyle changes include sodium reduction and regular exercise.",
]
TABLE = [
    ["Category", "Systolic", "Diastolic"],
    ["Normal", "<120 mm Hg", "<80 mm Hg"],
    ["Stage 1", "130-139 mm Hg", "80-89 mm Hg"],
]


def _page(doc: pymupdf.Document, number: int, lines: list[str], *, heading: str | None = None,
          table: list[list[str]] | None = None) -> None:
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 30), HEADER, fontsize=8)
    page.insert_text((280, 825), f"Page {number}", fontsize=8)
    y = 90
    if heading:
        page.insert_text((50, y), heading, fontsize=14, fontname="hebo")
        y += 30
    for line in lines:
        page.insert_text((50, y), line, fontsize=10)
        y += 14
    if table:
        x0, y0, cw, rh = 50, y + 20, 160, 22
        page.insert_text((x0, y0 - 6), "Table 1. Blood pressure categories", fontsize=9)
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
                page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                page.insert_text((rect.x0 + 4, rect.y0 + 15), cell, fontsize=9)


@pytest.fixture
def guideline_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    _page(doc, 1, BODY, heading="1. Definition of hypertension")
    _page(doc, 2, BODY, heading="2. Blood pressure categories", table=TABLE)
    _page(doc, 3, BODY)
    _page(doc, 4, ["References", "1. Smith J, et al. Hypertension 2020;75:100-10. doi:10.1/x"])
    _page(doc, 5, ["2. Jones K, et al. Circulation 2019;140:e596. doi:10.2/y"])
    path = tmp_path / "synthetic_guideline.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_running_headers_and_page_numbers_removed(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf, "SYNTH_2024")
    text = doc.full_text
    assert "Journal of Synthetic Cardiology" not in text
    assert "Page 1" not in text and "Page 3" not in text
    assert doc.removed_header_footer_lines >= 6
    assert "130/80 mm Hg" in text


def test_header_stripping_can_be_disabled(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf, strip_headers=False, strip_references=False)
    assert "Journal of Synthetic Cardiology" in doc.full_text
    assert doc.removed_header_footer_lines == 0


def test_references_tail_cut(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf)
    assert doc.references_start_page == 4
    assert doc.total_pages == 5  # physical page numbering is preserved
    assert "Smith J" not in doc.full_text and "Jones K" not in doc.full_text
    assert doc.pages[3].blocks == [] and doc.pages[4].blocks == []


def test_references_kept_when_disabled(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf, strip_references=False)
    assert doc.references_start_page is None
    assert "Smith J" in doc.full_text


def test_headings_detected_from_font(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf)
    headings = [b.text for p in doc.pages for b in p.blocks if b.kind == "heading"]
    assert headings == ["1. Definition of hypertension", "2. Blood pressure categories"]


def test_tables_detected_with_caption_and_not_duplicated(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf)
    assert doc.table_count == 1
    table = doc.pages[1].tables[0]
    assert table.page_number == 2
    assert table.rows[0] == ["Category", "Systolic", "Diastolic"]
    assert table.rows[2] == ["Stage 1", "130-139 mm Hg", "80-89 mm Hg"]
    assert table.caption == "Table 1. Blood pressure categories"
    prose = [b.text for b in doc.pages[1].blocks if b.kind == "paragraph"]
    assert not any("130-139" in t for t in prose)


def test_table_extraction_can_be_disabled(guideline_pdf: Path) -> None:
    doc = extract_pages_from_pdf(guideline_pdf, extract_tables=False)
    assert doc.table_count == 0
    assert "130-139 mm Hg" in doc.pages[1].text


@pytest.mark.parametrize(
    ("text", "size", "bold", "expected"),
    [
        ("2.1. Definition of High BP", 10.0, True, True),
        ("Synopsis", 12.0, False, True),
        ("Section 8: Pharmacological Treatment", 10.0, False, True),
        ("Table 4. Categories of Blood Pressure in Adults", 12.0, True, False),
        ("(Class IIa)", 12.0, True, False),
        ("Continued", 12.0, True, False),
        ("and the dose should be titrated", 12.0, False, False),
        ("Hypertension is common in adults.", 10.0, False, False),
        ("130/80", 14.0, True, False),
    ],
)
def test_is_heading_text(text: str, size: float, bold: bool, expected: bool) -> None:
    assert is_heading_text(text, size, bold, body_size=10.0, size_share=0.01) is expected


def test_large_but_common_font_is_not_a_heading() -> None:
    # A size used for > 5 % of the document's characters is body text, not a heading.
    assert is_heading_text("Blood pressure targets", 12.0, False, 10.0, size_share=0.3) is False


def test_table_markdown_rendering() -> None:
    table = ParsedTable(page_number=3, rows=[["Drug", "Dose"], ["Amlodipine", "5 mg"],
                                             ["A|B", ""]])
    assert table.n_rows == 3 and table.n_cols == 2
    assert table.to_markdown().splitlines() == [
        "| Drug | Dose |", "| --- | --- |", "| Amlodipine | 5 mg |", "| A/B |  |",
    ]
    assert table.to_markdown(rows=[]).count("\n") == 1

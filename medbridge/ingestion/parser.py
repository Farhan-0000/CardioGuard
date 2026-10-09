"""
Document Parser (Module M-18, Ingestion Pipeline).

Turns curated guideline files into page-aware, structured text:

* **PDF** (PyMuPDF): text blocks in content-stream order, heading detection from
  font statistics and numbering, running headers/footers removed (lines in the
  top/bottom margin repeated on > 30 % of pages, plus rotated margin text such as
  "Downloaded from ..."), the references/bibliography tail of the document cut,
  and tables detected with ``page.find_tables()`` and kept as structured rows so
  the chunker can emit them as markdown table chunks without duplicating their
  text in prose.
* **HTML** (``.html``/``.htm``): delegated to :mod:`medbridge.ingestion.html_parser`.

Page numbers are 1-indexed physical PDF pages (HTML: section index, see the HTML
parser) and are preserved for citation generation.

Technical Specification Part V §3.2, §8.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf
import structlog

logger = structlog.get_logger(__name__)

BlockKind = Literal["heading", "paragraph", "table"]

# Lines in the top/bottom margin repeated on more than this share of pages are
# running headers/footers.
HEADER_FOOTER_MIN_SHARE = 0.30
HEADER_FOOTER_ZONE = 0.09
# The references tail is only searched for in the second half of a document.
REFERENCES_MIN_POSITION = 0.5

_REFERENCES_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:references|reference list|bibliography|literature cited|"
    r"works cited|cited literature)\s*:?$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,4}\.?|[A-Z]\.|[IVX]{1,4}\.)\s+\S")
_LEGACY_HEADING_RE = re.compile(
    r"^(?:Section|SECTION|Chapter|CHAPTER)\s+\d+(?:\.\d+)*\s*[:.]\s*\S"
)
_CAPTION_RE = re.compile(r"^(?:Table|TABLE|Figure|FIGURE|Fig\.|Box|BOX)\s*\d+[A-Za-z]?\b")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
# C0/C1 control characters (symbol-font bullets often decode to these).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_NON_HEADINGS = frozenset({"continued", "cont.", "cont'd", "continued on next page"})
# Figure/recommendation-table labels that are typeset like headings.
_FIGURE_LABEL_RE = re.compile(r"^\(|\bClass (?:I{1,3}|IV)[ab]?\b|\bLevel of\b|\bLOE\b|\bCOR\b")


class PDFParseError(Exception):
    """Raised when a document cannot be opened, read, or parsed."""

    def __init__(self, file_path: str, message: str) -> None:
        self.file_path = file_path
        super().__init__(f"Failed to parse document '{file_path}': {message}")


@dataclass
class ParsedTable:
    """A table detected on a page, kept as rows of cleaned cell strings."""

    page_number: int
    rows: list[list[str]]
    caption: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_markdown(self, rows: list[list[str]] | None = None) -> str:
        """Render (a subset of) the table as GitHub-flavoured markdown.

        Args:
            rows: Body rows to render under the header row; defaults to all rows.
        """
        header = self.rows[0]
        body = self.rows[1:] if rows is None else rows
        width = self.n_cols

        def fmt(cells: list[str]) -> str:
            padded = [c.replace("|", "/") for c in cells] + [""] * (width - len(cells))
            return "| " + " | ".join(padded) + " |"

        lines = [fmt(header), "| " + " | ".join(["---"] * width) + " |"]
        lines.extend(fmt(r) for r in body)
        return "\n".join(lines)


@dataclass
class TextBlock:
    """A structural unit of a page in reading order."""

    kind: BlockKind
    text: str
    page_number: int
    level: int = 0
    table: ParsedTable | None = None


@dataclass
class ParsedPage:
    """A single parsed page (PDF page or HTML section) with 1-indexed page number.

    ``text`` holds the cleaned page text. ``blocks`` holds the structured content
    (headings, paragraphs, tables) used by the chunker; when empty the chunker
    falls back to line-based section detection over ``text``.
    """

    page_number: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def tables(self) -> list[ParsedTable]:
        return [b.table for b in self.blocks if b.kind == "table" and b.table is not None]


@dataclass
class ParsedDocument:
    """A complete parsed guideline document."""

    guideline_id: str
    file_path: str
    pages: list[ParsedPage] = field(default_factory=list)
    source_url: str = ""
    source_format: Literal["pdf", "html"] = "pdf"
    removed_header_footer_lines: int = 0
    references_start_page: int | None = None

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def table_count(self) -> int:
        return sum(len(p.tables) for p in self.pages)


def clean_text(text: str) -> str:
    """NFKC-normalise (ligatures, odd spaces), drop control glyphs, collapse whitespace."""
    normalised = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    normalised = _CONTROL_RE.sub(" ", normalised)
    return _WS_RE.sub(" ", normalised).strip()


def deduce_guideline_id(filename: str) -> str:
    """Infer a guideline identifier from a filename (legacy ``--pdf-dir`` mode).

    Manifest-driven ingestion never uses this: ids come from
    ``data/corpus_manifest.json``.

    Examples:
        'aha_acc_2025.pdf' -> 'AHA_ACC_2025'
        'ehae178.pdf' -> 'ESC_2024'
        'hypertension-in-adults-diagnosis-and-management...pdf' -> 'NICE_NG136'
        'medlineplus_hypertension.pdf' -> 'MEDLINEPLUS'
    """
    stem = Path(filename).stem
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").upper()

    if "AHA" in clean or "JONES" in clean:
        return "AHA_ACC_2025"
    if "ESC_ESH" in clean:
        return "ESC_ESH_2024"
    if "ESC" in clean or "EHAE178" in clean or "EHAE" in clean:
        return "ESC_2024"
    if "NICE" in clean or "DIAGNOSIS_AND_MANAGEMENT" in clean or "NG136" in clean:
        return "NICE_NG136"
    if "MEDLINE" in clean:
        return "MEDLINEPLUS"

    return clean or "CLINICAL_GUIDELINE"


# ---------------------------------------------------------------------------
# PDF line model
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    page_idx: int
    block_idx: int
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    horizontal: bool


def _line_key(text: str) -> str:
    """Normalisation used to detect repeated header/footer lines (digits removed)."""
    return _WS_RE.sub(" ", _DIGITS_RE.sub("", clean_text(text).lower())).strip()


def _is_bold(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    return bool(int(span.get("flags", 0)) & 16) or "bold" in font or "black" in font


def _collect_lines(doc: pymupdf.Document) -> tuple[list[list[_Line]], list[float]]:
    """Extract text lines per page with geometry and font information."""
    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_DEHYPHENATE
    flags &= ~pymupdf.TEXT_PRESERVE_IMAGES
    pages: list[list[_Line]] = []
    heights: list[float] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        heights.append(float(page.rect.height))
        lines: list[_Line] = []
        for block_idx, block in enumerate(page.get_text("dict", flags=flags)["blocks"]):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if str(s.get("text", "")).strip()]
                if not spans:
                    continue
                text = "".join(str(s["text"]) for s in line["spans"])
                main = max(spans, key=lambda s: len(str(s["text"]).strip()))
                direction = line.get("dir", (1.0, 0.0))
                x0, y0, x1, y1 = (float(v) for v in line["bbox"])
                lines.append(
                    _Line(
                        page_idx=page_idx,
                        block_idx=block_idx,
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        size=round(float(main.get("size", 0.0)) * 2) / 2,
                        bold=_is_bold(main),
                        horizontal=abs(float(direction[1])) < 0.5,
                    )
                )
        pages.append(lines)
    return pages, heights


def _in_margin(line: _Line, height: float) -> bool:
    """True for lines in the top/bottom margin zone or running vertically."""
    return (
        line.bbox[3] <= height * HEADER_FOOTER_ZONE
        or line.bbox[1] >= height * (1 - HEADER_FOOTER_ZONE)
        or not line.horizontal
    )


def _repeated_margin_keys(pages: list[list[_Line]], heights: list[float]) -> set[str]:
    """Normalised texts of margin lines repeated on > HEADER_FOOTER_MIN_SHARE of pages."""
    n_pages = len(pages)
    if n_pages < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines, height in zip(pages, heights, strict=True):
        counts.update(
            {_line_key(ln.text) for ln in lines
             if _in_margin(ln, height) and len(ln.text.strip()) <= 200}
        )
    threshold = HEADER_FOOTER_MIN_SHARE * n_pages
    return {k for k, c in counts.items() if c > threshold}


def strip_running_headers(
    pages: list[list[_Line]], heights: list[float]
) -> tuple[list[list[_Line]], int]:
    """Remove running headers/footers, margin page numbers and off-axis margin text.

    A line is removed when it sits in the top/bottom margin zone (or runs
    perpendicular to the page's dominant text direction) and its digit-free
    normalised text repeats on more than 30 % of pages, or it is a bare page
    number in the margin.

    Returns:
        The kept lines per page and the number of removed lines.
    """
    drop_keys = _repeated_margin_keys(pages, heights)
    removed = 0
    kept_pages: list[list[_Line]] = []
    for lines, height in zip(pages, heights, strict=True):
        horiz_chars = sum(len(ln.text) for ln in lines if ln.horizontal)
        dominant = horiz_chars >= sum(len(ln.text) for ln in lines) / 2
        kept: list[_Line] = []
        for ln in lines:
            margin = _in_margin(ln, height) or ln.horizontal != dominant
            key = _line_key(ln.text)
            if ln.horizontal != dominant or (margin and (key in drop_keys or not key)):
                removed += 1
                continue
            kept.append(ln)
        kept_pages.append(kept)
    return kept_pages, removed


def find_references_cut(pages: list[list[_Line]]) -> tuple[int, int] | None:
    """Locate the references/bibliography heading that starts the document tail.

    Standalone "References"/"Bibliography" lines are collected; the final run of
    hits (consecutive hits at most two pages apart, e.g. a repeated
    "references" running title) is taken and the cut is placed at its first
    hit, provided it lies in the second half of the document. Earlier,
    per-chapter reference lists are left to the chunk-level reference filter.

    Returns:
        ``(page_index, line_index)`` of the cut, or ``None``.
    """
    hits = [
        (p_idx, l_idx)
        for p_idx, lines in enumerate(pages)
        for l_idx, ln in enumerate(lines)
        if _REFERENCES_HEADING_RE.match(clean_text(ln.text))
    ]
    if not hits:
        return None
    start = len(hits) - 1
    while start > 0 and hits[start][0] - hits[start - 1][0] <= 2:
        start -= 1
    p_idx, l_idx = hits[start]
    if len(pages) >= 4 and p_idx < len(pages) * REFERENCES_MIN_POSITION:
        return None
    return p_idx, l_idx


@dataclass(frozen=True)
class FontProfile:
    """Document-level font statistics used for heading detection."""

    body_size: float
    size_share: dict[float, float]

    @classmethod
    def from_pages(cls, pages: list[list[_Line]]) -> FontProfile:
        sizes: Counter[float] = Counter()
        for lines in pages:
            for ln in lines:
                sizes[ln.size] += len(ln.text)
        total = sum(sizes.values())
        if not total:
            return cls(body_size=10.0, size_share={})
        return cls(
            body_size=sizes.most_common(1)[0][0],
            size_share={size: count / total for size, count in sizes.items()},
        )


def is_heading_text(
    text: str, size: float, bold: bool, body_size: float, size_share: float = 0.0
) -> bool:
    """Heuristic heading classifier for a PDF line.

    A line is a heading when it is short, contains real words, is not a
    table/figure caption, and either (a) is set in a rare font size at least
    1 pt larger than the body text, or (b) is numbered ("2.1. Definition of
    High BP") and emphasised (bold or slightly larger).

    Args:
        text: Line text.
        size: Font size of the line's main span.
        bold: Whether the main span is bold.
        body_size: Most common (body) font size of the document.
        size_share: Fraction of the document's characters set in ``size``.
    """
    stripped = clean_text(text)
    if len(stripped) < 3 or len(stripped) > 160:
        return False
    if not _ALPHA_WORD_RE.search(stripped) or _CAPTION_RE.match(stripped):
        return False
    if stripped.strip("()").lower() in _NON_HEADINGS or _FIGURE_LABEL_RE.search(stripped):
        return False
    if _LEGACY_HEADING_RE.match(stripped):
        return True
    if stripped[0].islower():
        return False
    if size >= body_size + 1.0 and size_share <= 0.05 and len(stripped.split()) <= 16:
        return True
    numbered = bool(_NUMBERED_HEADING_RE.match(stripped))
    return numbered and (bold or size >= body_size + 0.5) and len(stripped) <= 120


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines of one block into a paragraph."""
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif out.endswith("\u00ad"):
            out = out[:-1] + piece
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and piece[:1].islower():
            out = out[:-1] + piece
        else:
            out = f"{out} {piece}"
    return clean_text(out)


def _clean_cell(cell: object) -> str:
    return clean_text(str(cell)) if cell is not None else ""


_BBox = tuple[float, float, float, float]


def _extract_tables(page: pymupdf.Page, page_number: int) -> list[tuple[_BBox, ParsedTable]]:
    """Detect tables on a page; drop degenerate detections (layout false positives)."""
    try:
        finder = page.find_tables()
    except (ValueError, RuntimeError, IndexError) as exc:
        logger.debug("find_tables failed", page=page_number, error=str(exc))
        return []
    found: list[tuple[_BBox, ParsedTable]] = []
    for table in finder.tables:
        rows = [[_clean_cell(c) for c in row] for row in table.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        n_cols = max(len(r) for r in rows)
        cells = [c for r in rows for c in r]
        filled = sum(1 for c in cells if c)
        text_len = sum(len(c) for c in cells)
        if n_cols < 2 or text_len < 40 or filled < 0.5 * max(len(cells), 1):
            continue
        x0, y0, x1, y1 = (float(v) for v in table.bbox)
        found.append(((x0, y0, x1, y1), ParsedTable(page_number=page_number, rows=rows)))
    return found


def _inside(bbox: _BBox, box: _BBox) -> bool:
    """True when the centre of ``bbox`` lies within ``box`` (2pt tolerance)."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return box[0] - 2 <= cx <= box[2] + 2 and box[1] - 2 <= cy <= box[3] + 2


def _table_block(table: ParsedTable, page_number: int) -> TextBlock:
    return TextBlock(kind="table", text=table.to_markdown(), page_number=page_number, table=table)


def _page_blocks(
    lines: list[_Line],
    tables: list[tuple[_BBox, ParsedTable]],
    page_number: int,
    fonts: FontProfile,
) -> list[TextBlock]:
    """Group a page's surviving lines into heading/paragraph/table blocks."""
    blocks: list[TextBlock] = []
    emitted: set[int] = set()
    # Caption lookup: "Table N ..." lines just above a table bbox.
    for box, table in tables:
        above = [
            ln for ln in lines
            if _CAPTION_RE.match(clean_text(ln.text))
            and box[1] - 90 <= ln.bbox[3] <= box[1] + 6
            and min(ln.bbox[2], box[2]) - max(ln.bbox[0], box[0]) > 0
        ]
        if above:
            table.caption = clean_text(max(above, key=lambda ln: ln.bbox[3]).text)

    by_block: dict[int, list[_Line]] = {}
    for ln in lines:
        by_block.setdefault(ln.block_idx, []).append(ln)

    for group in by_block.values():
        table_hit = next(
            (i for i, (box, _) in enumerate(tables) if all(_inside(ln.bbox, box) for ln in group)),
            None,
        )
        if table_hit is not None:
            if table_hit not in emitted:
                emitted.add(table_hit)
                blocks.append(_table_block(tables[table_hit][1], page_number))
            continue
        # Split the block into runs of heading / non-heading lines.
        run: list[_Line] = []
        run_heading = False
        for ln in group:
            # A wrapped heading continues on the next line in the same emphasised font.
            continues_heading = (
                run_heading and bool(run) and abs(ln.size - run[-1].size) <= 0.25
                and ln.bold == run[-1].bold
                and (run[-1].bold or run[-1].size >= fonts.body_size + 0.5)
                and len(clean_text(ln.text)) <= 60
                and not clean_text(ln.text).endswith(".")
                and sum(len(r.text) for r in run) <= 140
            )
            heading = continues_heading or is_heading_text(
                ln.text, ln.size, ln.bold, fonts.body_size, fonts.size_share.get(ln.size, 0.0)
            )
            same_run = bool(run) and heading == run_heading and (
                not heading or abs(ln.size - run[-1].size) <= 0.5
            )
            if run and not same_run:
                blocks.append(_run_to_block(run, run_heading, page_number))
                run = []
            run.append(ln)
            run_heading = heading
        if run:
            blocks.append(_run_to_block(run, run_heading, page_number))

    blocks.extend(
        _table_block(table, page_number)
        for t_idx, (_, table) in enumerate(tables) if t_idx not in emitted
    )
    return [b for b in blocks if b.text]


def _run_to_block(run: list[_Line], heading: bool, page_number: int) -> TextBlock:
    text = _join_lines([ln.text for ln in run])
    if heading and len(text) > 200:
        heading = False
    return TextBlock(kind="heading" if heading else "paragraph", text=text, page_number=page_number)


def page_text_from_blocks(blocks: list[TextBlock]) -> str:
    """Plain text of a page (headings, paragraphs and markdown tables)."""
    return "\n".join(b.text for b in blocks if b.text)


def extract_pages_from_pdf(
    pdf_path: Path | str,
    guideline_id: str | None = None,
    source_url: str = "",
    *,
    strip_headers: bool = True,
    strip_references: bool = True,
    extract_tables: bool = True,
) -> ParsedDocument:
    """Extract structured, page-aware text from a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF document.
        guideline_id: Guideline identifier; deduced from the filename if omitted.
        source_url: Canonical source URL recorded on every chunk.
        strip_headers: Remove running headers/footers and rotated margin text.
        strip_references: Cut the references/bibliography tail of the document.
        extract_tables: Detect tables and keep them as structured blocks.

    Returns:
        ParsedDocument with one ParsedPage per physical page.

    Raises:
        PDFParseError: If the file does not exist, is corrupted, or fails to parse.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise PDFParseError(str(pdf_path), "File does not exist")

    gid = guideline_id or deduce_guideline_id(path_obj.name)

    try:
        doc = pymupdf.open(str(path_obj))
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        logger.error("PyMuPDF failed to open file", path=str(path_obj), error=str(exc))
        raise PDFParseError(str(pdf_path), str(exc)) from exc

    try:
        raw_pages, heights = _collect_lines(doc)
        removed = 0
        if strip_headers:
            raw_pages, removed = strip_running_headers(raw_pages, heights)
        references_page: int | None = None
        cut = find_references_cut(raw_pages) if strip_references else None
        if cut is not None:
            p_cut, l_cut = cut
            references_page = p_cut + 1
            raw_pages = raw_pages[:p_cut] + [raw_pages[p_cut][:l_cut]] + [
                [] for _ in raw_pages[p_cut + 1:]
            ]
        fonts = FontProfile.from_pages(raw_pages)

        parsed_pages: list[ParsedPage] = []
        for page_idx, lines in enumerate(raw_pages):
            page_number = page_idx + 1
            tables: list[tuple[_BBox, ParsedTable]] = []
            if extract_tables and lines:
                tables = _extract_tables(doc[page_idx], page_number)
                if cut is not None and page_idx == cut[0]:
                    last_y = max(ln.bbox[3] for ln in lines)
                    tables = [t for t in tables if t[0][1] <= last_y]
            blocks = _page_blocks(lines, tables, page_number, fonts)
            parsed_pages.append(
                ParsedPage(page_number=page_number, text=page_text_from_blocks(blocks),
                           blocks=blocks)
            )
    except (RuntimeError, ValueError, IndexError, KeyError) as exc:
        logger.error("Error extracting text from PDF", path=str(path_obj), error=str(exc))
        raise PDFParseError(str(pdf_path), str(exc)) from exc
    finally:
        doc.close()

    logger.info(
        "Parsed PDF",
        file=path_obj.name,
        pages=len(parsed_pages),
        guideline_id=gid,
        header_footer_lines_removed=removed,
        references_start_page=references_page,
        body_font_size=fonts.body_size,
        tables=sum(len(p.tables) for p in parsed_pages),
    )

    return ParsedDocument(
        guideline_id=gid,
        file_path=str(path_obj.resolve()),
        pages=parsed_pages,
        source_url=source_url,
        source_format="pdf",
        removed_header_footer_lines=removed,
        references_start_page=references_page,
    )


def parse_document(
    path: Path | str,
    guideline_id: str | None = None,
    source_url: str = "",
    *,
    html_root: str = "",
    extract_tables: bool = True,
) -> ParsedDocument:
    """Parse a PDF or HTML document, dispatching on the file extension.

    Raises:
        PDFParseError: If the file is missing, unsupported or unreadable.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".pdf":
        return extract_pages_from_pdf(
            path_obj, guideline_id, source_url, extract_tables=extract_tables
        )
    if suffix in {".html", ".htm"}:
        from medbridge.ingestion.html_parser import extract_sections_from_html

        return extract_sections_from_html(path_obj, guideline_id, source_url, root=html_root)
    raise PDFParseError(str(path), f"Unsupported file type '{suffix}'")

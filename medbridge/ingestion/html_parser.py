"""
HTML Document Parser (Module M-18, Ingestion Pipeline).

Extracts the main text of saved web pages (government patient pages, PMC
full-text articles) with the standard-library :class:`html.parser.HTMLParser`
-- no third-party HTML dependency.

* Only the main content is read: an explicit root selector from the manifest
  (``#id``, ``.class`` or a tag name), else ``<main>``, ``<article>`` or ``<body>``.
* Navigation and chrome are dropped: ``nav/header/footer/aside/script/style/
  form/button/svg``, ARIA ``navigation/banner/contentinfo/dialog`` regions,
  hidden/modal/tooltip elements (e.g. glossary pop-ups) and list items that are
  pure link lists.
* Headings (``h1``-``h6``) become section titles. HTML has no physical pages, so
  each ``h1``-``h3`` section is one *virtual page*: ``page_number`` of an HTML
  chunk is the 1-based index of that section in reading order.
* Sections titled References/Footnotes/Acknowledgements/... are skipped, as are
  superscript citation markers ("12,13").
* ``<table>`` elements become structured :class:`ParsedTable` blocks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import structlog

from medbridge.ingestion.parser import (
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    PDFParseError,
    TextBlock,
    clean_text,
    deduce_guideline_id,
    page_text_from_blocks,
)

logger = structlog.get_logger(__name__)

_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
     "source", "track", "wbr"}
)
_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "button",
     "svg", "select", "option", "template", "iframe", "canvas", "math", "dialog"}
)
_SKIP_ROLES = frozenset({"navigation", "banner", "contentinfo", "search", "dialog", "tooltip"})
_SKIP_CLASS_PREFIXES = (
    "usa-modal", "modal", "tooltip", "popover", "visually-hidden", "sr-only", "usa-sr-only",
    "hide-offscreen", "breadcrumb", "share", "social", "skipnav", "cookie",
)
_SKIP_IDS = frozenset({"toc-section", "table-of-contents"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "dd", "dt", "blockquote", "pre", "figcaption", "section", "article",
     "ul", "ol", "dl", "main", "body", "tr", "caption", "address", "details", "summary"}
)
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_PAGE_BREAK_LEVEL = 3
_SKIP_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s*)?(?:references?|reference list|bibliography|footnotes?|notes|"
    r"acknowledge?ments?|disclosures?|conflicts? of interest|author contributions?|"
    r"funding|supplementary (?:material|data)|article information|associated data|"
    r"data availability|permalink|related information|footnotes and references)\s*:?$",
    re.IGNORECASE,
)
_TABLE_HEADING_RE = re.compile(r"^(?:Table|Box)\s*\d+", re.IGNORECASE)
_FIGURE_HEADING_RE = re.compile(r"^(?:Figure|Fig\.)\s*\d+", re.IGNORECASE)
_CITATION_SUP_RE = re.compile(r"^[\d,\-–—\s]+$")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{3,}")
# Interface strings that survive markup filtering on PMC / government pages.
_UI_STRINGS = frozenset(
    {"open in a new tab", "download pdf", "view in own window", "go to:", "back to top",
     "share", "print", "email", "language switcher"}
)


@dataclass
class _Element:
    tag: str
    skip: bool
    is_root: bool
    is_caption: bool = False


@dataclass
class _ListItem:
    text_len: int = 0
    link_len: int = 0


@dataclass
class _TableState:
    rows: list[list[str]] = field(default_factory=list)
    cell: list[str] | None = None
    depth: int = 1


RawBlock = tuple[str, str, int, ParsedTable | None]


class _MainTextExtractor(HTMLParser):
    """Streaming extractor producing heading/paragraph/table blocks."""

    def __init__(self, root: str) -> None:
        super().__init__(convert_charrefs=True)
        self._root_kind, self._root_value = _parse_selector(root)
        self._stack: list[_Element] = []
        self._root_depth = 0
        self._skip_depth = 0
        self._buffer: list[str] = []
        self._heading_level = 0
        self._heading_buffer: list[str] = []
        self._sup_buffer: list[str] | None = None
        self._caption_depth = 0
        self._caption_buffer: list[str] = []
        self._pending_caption = ""
        self._table: _TableState | None = None
        self._items: list[_ListItem] = []
        self._link_depth = 0
        self._skip_section_level = 0
        self.blocks: list[RawBlock] = []

    # -- helpers ---------------------------------------------------------------
    @property
    def _in_content(self) -> bool:
        """Inside the root element and not inside a skipped (chrome) element."""
        return self._root_depth > 0 and self._skip_depth == 0

    def _is_root(self, tag: str, attrs: dict[str, str]) -> bool:
        if self._root_kind == "id":
            return attrs.get("id", "") == self._root_value
        if self._root_kind == "class":
            return self._root_value in attrs.get("class", "").split()
        return tag == self._root_value

    @staticmethod
    def _should_skip(tag: str, attrs: dict[str, str]) -> bool:
        if tag in _SKIP_TAGS:
            return True
        if attrs.get("role", "") in _SKIP_ROLES or attrs.get("aria-hidden") == "true":
            return True
        if "hidden" in attrs or attrs.get("id", "") in _SKIP_IDS:
            return True
        classes = attrs.get("class", "").lower().split()
        return any(c.startswith(_SKIP_CLASS_PREFIXES) for c in classes)

    def _flush(self) -> None:
        text = clean_text("".join(self._buffer))
        self._buffer = []
        if not text or not _ALPHA_WORD_RE.search(text) or text.lower() in _UI_STRINGS:
            return
        if self._items:
            self._items[-1].text_len += len(text)
        self.blocks.append(("paragraph", text, 0, None))

    # -- HTMLParser API ----------------------------------------------------------
    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag in _VOID_TAGS:
            if tag == "br" and self._in_content:
                if self._table is not None and self._table.cell is not None:
                    self._table.cell.append(" ")
                else:
                    self._buffer.append(" ")
            return
        is_root = self._root_depth == 0 and self._is_root(tag, attrs)
        skip = self._should_skip(tag, attrs)
        is_caption = tag == "caption" or "caption" in attrs.get("class", "").split()
        self._stack.append(_Element(tag=tag, skip=skip, is_root=is_root, is_caption=is_caption))
        if is_root:
            self._root_depth = 1
        if skip:
            self._skip_depth += 1
            return
        if not self._in_content:
            return
        if tag in _HEADING_TAGS and self._table is None:
            self._flush()
            self._heading_level = _HEADING_TAGS[tag]
            self._heading_buffer = []
            return
        if self._skip_section_level:
            return
        if self._table is not None:
            self._table_start(tag)
            return
        if is_caption:
            self._flush()
            self._caption_depth += 1
        elif tag == "table":
            self._flush()
            self._table = _TableState()
        elif tag == "sup":
            self._sup_buffer = []
        elif tag == "a":
            self._link_depth += 1
        elif tag == "li":
            self._flush()
            self._items.append(_ListItem())
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or not any(el.tag == tag for el in self._stack):
            return
        while self._stack:
            element = self._stack.pop()
            self._close(element)
            if element.tag == tag:
                break

    def _close(self, element: _Element) -> None:
        if element.skip:
            self._skip_depth -= 1
            return
        if element.is_root:
            self._flush()
            self._root_depth = 0
            return
        if not self._in_content:
            return
        tag = element.tag
        if tag in _HEADING_TAGS and self._heading_level and self._table is None:
            self._end_heading()
            return
        if self._skip_section_level:
            return
        if self._table is not None:
            self._table_end(tag)
        elif element.is_caption and self._caption_depth:
            self._caption_depth -= 1
            caption = clean_text(" ".join(self._caption_buffer))
            self._caption_buffer = []
            self._pending_caption = clean_text(f"{self._pending_caption} {caption}")[:300]
        elif tag == "sup" and self._sup_buffer is not None:
            text = "".join(self._sup_buffer)
            self._sup_buffer = None
            if not _CITATION_SUP_RE.match(text):
                self._emit_text(text)
        elif tag == "a":
            self._link_depth = max(0, self._link_depth - 1)
        elif tag == "li":
            self._flush()
            item = self._items.pop() if self._items else None
            if item and item.text_len and item.link_len >= 0.85 * item.text_len:
                self._drop_link_list_item(item)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def _drop_link_list_item(self, item: _ListItem) -> None:
        """Remove trailing paragraph blocks belonging to a pure link-list item."""
        removed = 0
        while self.blocks and removed < item.text_len and self.blocks[-1][0] == "paragraph":
            removed += len(self.blocks.pop()[1])

    def _end_heading(self) -> None:
        text = clean_text("".join(self._heading_buffer))
        level = self._heading_level
        self._heading_level = 0
        self._heading_buffer = []
        if not text:
            return
        if self._skip_section_level and level <= self._skip_section_level:
            self._skip_section_level = 0
        if self._skip_section_level:
            return
        if _SKIP_SECTION_RE.match(text):
            self._skip_section_level = level
            return
        if _TABLE_HEADING_RE.match(text):
            self._pending_caption = text
            return
        if _FIGURE_HEADING_RE.match(text):
            self.blocks.append(("paragraph", text, 0, None))
            return
        self.blocks.append(("heading", text, level, None))

    def handle_data(self, data: str) -> None:
        if not self._in_content:
            return
        if self._heading_level and self._table is None:
            self._heading_buffer.append(data)
            return
        if self._skip_section_level:
            return
        if self._table is not None:
            if self._table.cell is not None:
                self._table.cell.append(data)
            return
        if self._sup_buffer is not None:
            self._sup_buffer.append(data)
            return
        if self._caption_depth:
            self._caption_buffer.append(data)
            return
        self._emit_text(data)

    def _emit_text(self, data: str) -> None:
        self._buffer.append(data)
        if self._link_depth and self._items:
            self._items[-1].link_len += len(data.strip())

    # -- tables ----------------------------------------------------------------
    def _table_start(self, tag: str) -> None:
        table = self._table
        if table is None:
            return
        if tag == "table":
            table.depth += 1
        elif tag == "tr" and table.depth == 1:
            table.rows.append([])
        elif tag in {"td", "th"} and table.depth == 1:
            if not table.rows:
                table.rows.append([])
            table.cell = []
        elif tag in {"p", "li", "div"} and table.cell is not None:
            table.cell.append(" ")

    def _table_end(self, tag: str) -> None:
        table = self._table
        if table is None:
            return
        if tag == "table":
            table.depth -= 1
            if table.depth == 0:
                self._finish_table(table)
        elif tag in {"td", "th"} and table.depth == 1 and table.cell is not None:
            table.rows[-1].append(clean_text("".join(table.cell)))
            table.cell = None

    def _finish_table(self, table: _TableState) -> None:
        rows = [r for r in table.rows if any(r)]
        self._table = None
        caption = self._pending_caption
        self._pending_caption = ""
        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            text = clean_text(" ".join(" ".join(r) for r in rows))
            if text:
                self.blocks.append(("paragraph", text, 0, None))
            return
        self.blocks.append(
            ("table", "", 0, ParsedTable(page_number=0, rows=rows, caption=caption))
        )


def _parse_selector(root: str) -> tuple[str, str]:
    root = root.strip()
    if root.startswith("#"):
        return "id", root[1:]
    if root.startswith("."):
        return "class", root[1:]
    return "tag", root.lower()


def _default_root(html: str) -> str:
    lowered = html.lower()
    if "<main" in lowered:
        return "main"
    if "<article" in lowered:
        return "article"
    return "body"


def blocks_to_pages(raw_blocks: list[RawBlock]) -> list[ParsedPage]:
    """Group extracted blocks into virtual pages, one per ``h1``-``h3`` section."""
    pages: list[ParsedPage] = []
    current: list[TextBlock] = []

    def close_page() -> None:
        if current:
            number = len(pages) + 1
            for block in current:
                block.page_number = number
                if block.table is not None:
                    block.table.page_number = number
            pages.append(ParsedPage(page_number=number, text=page_text_from_blocks(current),
                                    blocks=list(current)))
            current.clear()

    for kind, text, level, table in raw_blocks:
        if kind == "heading" and level <= _PAGE_BREAK_LEVEL and any(
            b.kind != "heading" for b in current
        ):
            close_page()
        if kind == "table" and table is not None:
            current.append(TextBlock(kind="table", text=table.to_markdown(), page_number=0,
                                     table=table))
        elif kind == "heading":
            current.append(TextBlock(kind="heading", text=text, page_number=0, level=level))
        else:
            current.append(TextBlock(kind="paragraph", text=text, page_number=0))
    close_page()
    return pages


def extract_sections_from_html(
    html_path: Path | str,
    guideline_id: str | None = None,
    source_url: str = "",
    *,
    root: str = "",
) -> ParsedDocument:
    """Parse a saved HTML page into a page-aware :class:`ParsedDocument`.

    Args:
        html_path: Path to the ``.html``/``.htm`` file.
        guideline_id: Guideline identifier; deduced from the filename if omitted.
        source_url: Canonical page URL recorded on every chunk.
        root: Optional main-content selector (``#id``, ``.class`` or tag name).

    Raises:
        PDFParseError: If the file is missing, unreadable, or has no main text.
    """
    path_obj = Path(html_path)
    if not path_obj.exists():
        raise PDFParseError(str(html_path), "File does not exist")
    try:
        html = path_obj.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PDFParseError(str(html_path), str(exc)) from exc

    selector = root or _default_root(html)
    extractor = _MainTextExtractor(selector)
    extractor.feed(html)
    extractor.close()
    pages = blocks_to_pages(extractor.blocks)
    if not pages:
        raise PDFParseError(str(html_path), f"no main text found (root '{selector}')")

    gid = guideline_id or deduce_guideline_id(path_obj.name)
    logger.info(
        "Parsed HTML",
        file=path_obj.name,
        sections=len(pages),
        guideline_id=gid,
        root=selector,
        tables=sum(len(p.tables) for p in pages),
    )
    return ParsedDocument(
        guideline_id=gid,
        file_path=str(path_obj.resolve()),
        pages=pages,
        source_url=source_url,
        source_format="html",
    )

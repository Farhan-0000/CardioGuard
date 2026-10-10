"""
Shared loading for the IEEE paper builders (``build_paper.py`` -> DOCX, ``build_latex.py`` -> LaTeX).

The paper is written in the same markup as the report (``report/lib/content.py``, see
``report/README.md``) and reuses the report library for parsing, results values,
generated tables and the verified reference list. This module adds what a conference
paper needs on top of it:

* **Content discovery** (``report/paper/content/``): ``00_*.md`` is the title block (first
  ``#`` heading = title, first pipe table = authors), ``01_*.md`` holds the abstract
  (``#* Abstract``) and the index terms (``#* Index Terms``), and every other ``NN_*.md`` is
  a body file, assembled in name order. Files without a two-digit prefix are include-only
  (``[[include:results.md]]``).
* **Labels and cross-references.** A caption, heading or algorithm title may end with
  ``{#fig:name}``, ``{#tab:name}``, ``{#alg:name}`` or ``{#sec:name}``. In the text,
  ``[@fig:name]`` becomes "Fig. 1", ``[@tab:name]`` "Table II", ``[@alg:name]`` "Algorithm 1"
  and ``[@sec:name]`` "Section III" (LaTeX: ``Fig.~\\ref{fig:name}`` ...). Generated tables
  get the label ``tab:<KEY>`` unless their caption sets one. Adding ``.wide`` to a label
  (``{#tab:name .wide}``) sets a figure or table across both columns; figures wider than
  3.6 in and tables that cannot fit one column are set across both columns automatically.
* **Numbering** in IEEE style: sections I, II, ... (subsections A, B, ...), figures 1, 2, ...,
  tables I, II, ..., algorithms 1, 2, ...
* **Citations** are numbered in order of first appearance in the body (IEEE). Only keys of
  ``report/references/references.json`` and ``patents.json`` resolve; unknown keys are
  reported and rendered as a highlighted marker, never invented.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parent
REPORT_DIR = PAPER_DIR.parent
REPO_ROOT = REPORT_DIR.parent
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

from lib.content import (  # noqa: E402
    Algorithm,
    Block,
    Figure,
    GeneratedTable,
    Heading,
    ListBlock,
    Paragraph,
    ParseContext,
    PendingTable,
    Table,
    parse_file,
    tokenize,
)
from lib.references import CitationRegistry, load_references  # noqa: E402
from lib.results_context import ResultsContext  # noqa: E402
from lib.tables import DEFAULT_CAPTIONS, generate  # noqa: E402

DEFAULT_CONTENT = PAPER_DIR / "content"

#: IEEE column width (both US Letter and A4 templates) and the width above which a
#: figure is set across both columns.
COLUMN_WIDTH_IN = 3.5
WIDE_FIGURE_IN = 3.6

LABEL_RE = re.compile(
    r"\s*\{#((?:fig|tab|alg|sec):[A-Za-z0-9_.:\-]+)((?:\s+\.[A-Za-z-]+)*)\s*\}\s*$"
)
XREF_KINDS: dict[str, str] = {"fig": "Fig.", "tab": "Table", "alg": "Algorithm", "sec": "Section"}
BODY_FILE = re.compile(r"^(\d{2})_[\w.-]+\.md$")
UNNUMBERED_SECTIONS = {"acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}
ANONYMOUS_DROP = UNNUMBERED_SECTIONS


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def to_roman(n: int) -> str:
    """1 -> I, 4 -> IV, 14 -> XIV (IEEE section and table numbers)."""
    out = ""
    for value, sym in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while n >= value:
            out += sym
            n -= value
    return out


def split_label(text: str) -> tuple[str, str | None, set[str]]:
    """``"Caption {#fig:x .wide}"`` -> ``("Caption", "fig:x", {"wide"})``."""
    m = LABEL_RE.search(text or "")
    if not m:
        return text, None, set()
    classes = {c.lstrip(".") for c in m.group(2).split()} if m.group(2) else set()
    return text[: m.start()].rstrip(), m.group(1), classes


def is_xref(key: str) -> bool:
    kind, sep, rest = key.partition(":")
    return bool(sep and rest) and kind in XREF_KINDS


def _plain(cell: str) -> str:
    return re.sub(r"\*\*|`|\[@[^\]]*\]", "", str(cell))


def table_min_width_in(table: Table, font_size: float = 7.5) -> float:
    """Narrowest sensible width: each column's longest unbreakable word plus padding."""
    char_in = font_size * 0.5 / 72.0
    ncols = max(len(table.header), max((len(r) for r in table.rows), default=0))
    total = 0.0
    for c in range(ncols):
        cells = [table.header[c] if c < len(table.header) else ""]
        cells += [r[c] for r in table.rows if c < len(r)]
        words = [w for cell in cells for w in re.split(r"[ \t\r\n]+", _plain(cell)) if w]
        longest = min(max((len(w) for w in words), default=1), 24)
        total += longest * char_in + 0.1
    return total


def table_is_wide(table: Table, classes: set[str] | None = None) -> bool:
    if classes and "wide" in classes:
        return True
    ncols = max(len(table.header), max((len(r) for r in table.rows), default=0))
    return ncols >= 7 or table_min_width_in(table) > COLUMN_WIDTH_IN - 0.05


# ----------------------------------------------------------------------
# Paper model
# ----------------------------------------------------------------------


@dataclass
class Author:
    name: str
    affiliation: list[str] = field(default_factory=list)
    email: str = ""


@dataclass
class Element:
    """Numbering of one labelled/numbered block."""

    kind: str  # section | subsection | figure | table | algorithm
    number: int
    display: str  # "III", "III-B", "1", "II"
    label: str | None = None
    wide: bool = False

    def xref_text(self) -> str:
        prefix = {
            "section": "Section",
            "subsection": "Section",
            "figure": "Fig.",
            "table": "Table",
            "algorithm": "Algorithm",
        }[self.kind]
        return f"{prefix} {self.display}"


@dataclass
class Paper:
    title: str
    authors: list[Author]
    abstract: list[str]
    index_terms: str
    blocks: list[Block]
    elements: dict[int, Element]
    labels: dict[str, Element]
    registry: CitationRegistry
    ctx: ResultsContext
    content_dir: Path
    repo_root: Path
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    table_errors: list[str] = field(default_factory=list)
    unresolved_xrefs: list[str] = field(default_factory=list)
    anonymous: bool = False

    def element(self, block: Any) -> Element | None:
        return self.elements.get(id(block))

    def xref(self, key: str) -> str | None:
        el = self.labels.get(key)
        return el.xref_text() if el else None

    def resolve_figure(self, path_text: str) -> Path | None:
        """Repo-relative (or content/paper-relative) image path; ``.svg`` -> ``.png`` sibling."""
        raw = Path(path_text.strip())
        roots = [self.repo_root, self.content_dir, PAPER_DIR, self.repo_root / "report"]
        candidates = [raw] if raw.is_absolute() else [root / raw for root in roots]
        for cand in candidates:
            options = [cand]
            if cand.suffix.lower() in {".svg", ".pdf", ".eps"}:
                options = [cand.with_suffix(".png"), cand.with_suffix(".jpg")]
            for opt in options:
                if opt.is_file():
                    return opt
        return None


# ----------------------------------------------------------------------
# Discovery and parsing
# ----------------------------------------------------------------------


def discover(content_dir: Path) -> tuple[Path | None, Path | None, list[Path]]:
    """(title file, abstract file, body files in order)."""
    title = abstract = None
    body: list[Path] = []
    for path in sorted(content_dir.glob("*.md")):
        m = BODY_FILE.match(path.name)
        stem = path.stem.lower()
        if (m and m.group(1) == "00") or stem == "title":
            title = title or path
        elif (m and m.group(1) == "01") or stem == "abstract":
            abstract = abstract or path
        elif m:
            body.append(path)
    return title, abstract, body


def _authors_from_table(table: Table) -> list[Author]:
    header = [h.strip().lower() for h in table.header]

    def col(*names: str) -> int | None:
        for i, h in enumerate(header):
            if any(h.startswith(n) for n in names):
                return i
        return None

    i_name, i_aff, i_mail = (
        col("author", "name"),
        col("affil", "institution"),
        col("e-mail", "email"),
    )
    out = []
    for row in table.rows:
        name = row[i_name].strip() if i_name is not None and i_name < len(row) else ""
        if not name:
            continue
        aff = row[i_aff] if i_aff is not None and i_aff < len(row) else ""
        mail = row[i_mail].strip() if i_mail is not None and i_mail < len(row) else ""
        lines = [p.strip() for p in aff.split(";") if p.strip()]
        out.append(Author(name=name, affiliation=lines, email=mail))
    return out


def _texts(block: Block) -> list[str]:
    if isinstance(block, Heading | Paragraph):
        return [block.text]
    if isinstance(block, ListBlock):
        return [item.text for item in block.items]
    if isinstance(block, Figure):
        return [block.caption]
    if isinstance(block, Table):
        out = [block.caption or ""]
        if block.markup:
            out += block.header + [c for row in block.rows for c in row]
        return out
    if isinstance(block, PendingTable):
        return [block.caption or ""]
    if isinstance(block, Algorithm):
        return [block.title]
    return []


def _drop_sections(blocks: list[Block], names: set[str]) -> list[Block]:
    out: list[Block] = []
    skipping = False
    for block in blocks:
        if isinstance(block, Heading) and block.level == 1:
            skipping = split_label(block.text)[0].strip().lower() in names
        if not skipping:
            out.append(block)
    return out


def load_paper(
    content_dir: Path = DEFAULT_CONTENT,
    repo_root: Path = REPO_ROOT,
    *,
    ctx: ResultsContext | None = None,
    anonymous: bool = False,
) -> Paper:
    """Parse the paper content, expand generated tables, number everything, register citations."""
    content_dir = Path(content_dir)
    ctx = ctx or ResultsContext.load(repo_root)
    pctx = ParseContext(content_dir=content_dir)
    title_file, abstract_file, body_files = discover(content_dir)

    title = ""
    authors: list[Author] = []
    if title_file is not None:
        for block in parse_file(title_file, pctx):
            if isinstance(block, Heading) and not title:
                title = block.text.strip()
            elif isinstance(block, Table) and not authors:
                authors = _authors_from_table(block)
    if not title:
        title = "[PENDING: paper title (content/00_title.md)]"
        pctx.warnings.append("no title: create content/00_title.md with a '# Title' heading")
    if anonymous:
        authors = [Author("Anonymous Author(s)", ["Affiliation withheld for double-blind review"])]

    abstract: list[str] = []
    index_terms = ""
    if abstract_file is not None:
        mode = ""
        for block in parse_file(abstract_file, pctx):
            if isinstance(block, Heading):
                name = block.text.strip().lower()
                mode = "terms" if ("index" in name or "keyword" in name) else "abstract"
            elif isinstance(block, Paragraph):
                if mode == "terms":
                    index_terms = (index_terms + " " + block.text).strip()
                else:
                    abstract.append(block.text)
    if not abstract:
        abstract = ["[PENDING: abstract (content/01_abstract.md)]"]

    raw: list[Block] = []
    for path in body_files:
        raw.extend(parse_file(path, pctx))
    if anonymous:
        raw = _drop_sections(raw, ANONYMOUS_DROP)

    # generated tables -> Table / PendingTable
    table_errors: list[str] = []
    blocks: list[Block] = []
    for block in raw:
        if isinstance(block, GeneratedTable):
            caption, label, classes = split_label(block.caption or "")
            table = generate(block.key, ctx, table_errors)
            caption = (
                caption
                or (table.caption if table else None)
                or DEFAULT_CAPTIONS.get(block.key, block.key)
            )
            label = label or f"tab:{block.key}"
            if table is None:
                pending = PendingTable(key=block.key, caption=f"{caption} {{#{label}}}")
                blocks.append(pending)
            else:
                suffix = " .wide" if "wide" in classes else ""
                table.caption = f"{caption} {{#{label}{suffix}}}"
                blocks.append(table)
        else:
            blocks.append(block)

    # numbering + labels
    elements: dict[int, Element] = {}
    labels: dict[str, Element] = {}
    warnings = list(pctx.warnings)
    counters = {"section": 0, "subsection": 0, "figure": 0, "table": 0, "algorithm": 0}

    def register(block: Any, el: Element) -> None:
        elements[id(block)] = el
        if el.label:
            if el.label in labels:
                warnings.append(f"duplicate label {el.label}")
            labels[el.label] = el

    for block in blocks:
        if isinstance(block, Heading):
            text, label, _ = split_label(block.text)
            block.text = text
            if block.level == 1:
                counters["subsection"] = 0
                numbered = block.numbered and text.strip().lower() not in UNNUMBERED_SECTIONS
                block.numbered = numbered
                if numbered:
                    counters["section"] += 1
                    n = counters["section"]
                    register(block, Element("section", n, to_roman(n), label))
                elif label:
                    warnings.append(f"label {label} on an unnumbered section is ignored")
            elif block.level == 2 and block.numbered and counters["section"]:
                counters["subsection"] += 1
                n = counters["subsection"]
                display = f"{to_roman(counters['section'])}-{chr(64 + n)}"
                register(block, Element("subsection", n, display, label))
        elif isinstance(block, Figure):
            text, label, classes = split_label(block.caption)
            block.caption = text
            counters["figure"] += 1
            n = counters["figure"]
            wide = "wide" in classes or (block.width or 0) > WIDE_FIGURE_IN
            register(block, Element("figure", n, str(n), label, wide))
        elif isinstance(block, Table | PendingTable):
            text, label, classes = split_label(block.caption or "")
            block.caption = text or None
            if isinstance(block, Table) and not block.caption and not label:
                continue  # an uncaptioned pipe table is not numbered
            counters["table"] += 1
            n = counters["table"]
            wide = isinstance(block, Table) and table_is_wide(block, classes)
            register(block, Element("table", n, to_roman(n), label, wide))
        elif isinstance(block, Algorithm):
            text, label, _ = split_label(block.title)
            block.title = text
            counters["algorithm"] += 1
            n = counters["algorithm"]
            register(block, Element("algorithm", n, str(n), label))

    # citations (body order) and cross-reference check
    registry = CitationRegistry(load_references(repo_root))
    unresolved: list[str] = []
    for text in abstract + [t for b in blocks for t in _texts(b)]:
        for tok in tokenize(text or ""):
            if tok.kind != "cite":
                continue
            refs = [k for k in tok.keys if not is_xref(k)]
            registry.register(refs)
            for key in tok.keys:
                if is_xref(key) and key not in labels and key not in unresolved:
                    unresolved.append(key)
    for key in unresolved:
        warnings.append(f"unresolved cross-reference {key}")

    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        index_terms=index_terms,
        blocks=blocks,
        elements=elements,
        labels=labels,
        registry=registry,
        ctx=ctx,
        content_dir=content_dir,
        repo_root=repo_root,
        files=[p.name for p in [title_file, abstract_file, *body_files] if p is not None],
        warnings=warnings,
        includes=list(pctx.includes_used),
        table_errors=table_errors,
        unresolved_xrefs=unresolved,
        anonymous=anonymous,
    )


def base_log(paper: Paper) -> dict[str, Any]:
    """Fields shared by the DOCX and LaTeX build logs."""
    return {
        "content_dir": _rel(paper.content_dir, paper.repo_root),
        "files": paper.files,
        "includes": paper.includes,
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "anonymous": paper.anonymous,
        "sections": [
            f"{el.display}. {b.text}"
            for b in paper.blocks
            if isinstance(b, Heading) and (el := paper.element(b)) and el.kind == "section"
        ],
        "counts": {
            kind: sum(1 for el in paper.elements.values() if el.kind == kind)
            for kind in ("section", "figure", "table", "algorithm")
        },
        "references": len(paper.registry.order),
        "citation_keys": list(paper.registry.order),
        "citation_keys_missing": list(paper.registry.missing),
        "unresolved_xrefs": list(paper.unresolved_xrefs),
        "values_missing": sorted(set(paper.ctx.missing)),
        "results_sources": paper.ctx.data.get("_sources", {}),
        "results_skipped": paper.ctx.data.get("_skipped", {}),
        "warnings": paper.warnings + paper.table_errors,
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name

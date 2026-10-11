#!/usr/bin/env python3
"""
Build the MedBridge IEEE conference paper as LaTeX (IEEEtran, conference mode).

Converts the paper content (``report/paper/content/*.md``, the report markup) into
``report/paper/latex/main.tex`` and ``report/paper/latex/references.bib`` and copies the
figures it uses into ``report/paper/latex/figures/``. Values, generated tables and
citations come from the same sources as the DOCX builder (``build_paper.py``), so the two
versions always agree. Nothing is compiled here (no TeX installation is assumed): upload
the ``latex/`` folder to Overleaf or run ``pdflatex``/``bibtex`` locally, see
``report/paper/latex/README.md``.

The generated source is plain ASCII: special characters are escaped and non-ASCII
symbols are mapped to LaTeX commands (``≥`` -> ``$\\geq$``, ``é`` -> ``\\'{e}``, ...). A
built-in check verifies balanced braces and matching ``\\begin``/``\\end`` pairs.
Placeholders (``[TO BE FILLED: ...]``) and pending markers are typeset in bold red via
``\\pending{...}``.

Usage::

    python report/paper/build_latex.py                 # -> report/paper/latex/{main.tex,references.bib,figures/}
    python report/paper/build_latex.py --anonymous     # double-blind review copy
    python report/paper/build_latex.py --out-dir DIR --strict

Exit codes: 0 success, 1 strict check failed (unknown citations, unresolved
cross-references or a structural problem in the generated source), 2 content
directory missing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import paper_common as pc  # noqa: E402  (also puts report/ on sys.path)
from lib.content import (  # noqa: E402
    Algorithm,
    Block,
    CodeBlock,
    Figure,
    Heading,
    ListBlock,
    PageBreak,
    Paragraph,
    Pending,
    PendingTable,
    Table,
    tokenize,
)
from lib.references import is_organisation  # noqa: E402

DEFAULT_OUT_DIR = PAPER_DIR / "latex"
DEFAULT_LOG = PAPER_DIR / "build" / "latex_build_log.json"

# ----------------------------------------------------------------------
# Escaping
# ----------------------------------------------------------------------

TEX_SPECIAL: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}
TEX_UNICODE: dict[str, str] = {
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "`",
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u2026": r"\ldots{}",
    "\u00a0": "~",
    "\u200b": "",
    "\u2009": r"\,",
    "\u2212": "$-$",
    "\u2265": r"$\geq$",
    "\u2264": r"$\leq$",
    "\u2260": r"$\neq$",
    "\u2192": r"$\rightarrow$",
    "\u2190": r"$\leftarrow$",
    "\u21d2": r"$\Rightarrow$",
    "\u00d7": r"$\times$",
    "\u00b1": r"$\pm$",
    "\u2248": r"$\approx$",
    "\u222a": r"$\cup$",
    "\u2229": r"$\cap$",
    "\u2208": r"$\in$",
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u03b8": r"$\theta$",
    "\u03ba": r"$\kappa$",
    "\u03bc": r"$\mu$",
    "\u00b5": r"$\mu$",
    "\u03c4": r"$\tau$",
    "\u03c7": r"$\chi$",
    "\u00b0": r"\textdegree{}",
    "\u2713": r"$\checkmark$",
    "\u2714": r"$\checkmark$",
    "\u00b7": r"$\cdot$",
    "\u2022": r"\textbullet{}",
    "\u00a7": r"\S{}",
    "\u00ae": r"\textregistered{}",
    "\u2122": r"\texttrademark{}",
    "\u00a9": r"\textcopyright{}",
    "\u20b9": "Rs.~",
    "\u00df": r"\ss{}",
    "\u00e6": r"\ae{}",
    "\u00f8": r"\o{}",
    "\u0131": r"\i{}",
}
#: Unicode superscripts (p-values such as 5.4×10⁻¹⁶); consecutive ones form one $^{...}$.
TEX_SUPERSCRIPT: dict[str, str] = {
    **{sup: str(d) for d, sup in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
    "\u207b": "-",
    "\u207a": "+",
}
#: Combining accents -> LaTeX accent commands (for NFD-decomposed Latin letters).
TEX_ACCENTS: dict[str, str] = {
    "\u0301": "'",
    "\u0300": "`",
    "\u0302": "^",
    "\u0308": '"',
    "\u0303": "~",
    "\u0304": "=",
    "\u0327": "c",
    "\u030a": "r",
    "\u030c": "v",
    "\u0306": "u",
    "\u0328": "k",
    "\u0307": ".",
}
#: ASCII stand-ins for code / listings (listings does not take UTF-8 under pdfLaTeX).
ASCII_CODE: dict[str, str] = {
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2260": "!=",
    "\u2192": "->",
    "\u2190": "<-",
    "\u00d7": "x",
    "\u222a": "U",
    "\u2208": "in",
    "\u03c4": "tau",
    "\u03b8": "theta",
    "\u03ba": "kappa",
    "\u2013": "-",
    "\u2014": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "\u200b": "",
}
PENDING_PREFIXES = (
    "[PENDING",
    "[TABLE PENDING",
    "[FIGURE PENDING",
    "[TO BE FILLED",
    "[TODO",
    "[TBD",
)
XREF_TEX = {"fig": "Fig.~", "tab": "Table~", "alg": "Algorithm~", "sec": "Section~"}


class Escaper:
    """Escapes text to ASCII LaTeX and records characters it could not map."""

    def __init__(self) -> None:
        self.unmapped: set[str] = set()

    def char(self, ch: str) -> str:
        if ch in TEX_SPECIAL:
            return TEX_SPECIAL[ch]
        if ord(ch) < 128:
            return ch
        if ch in TEX_UNICODE:
            return TEX_UNICODE[ch]
        decomposed = unicodedata.normalize("NFD", ch)
        base, marks = decomposed[0], decomposed[1:]
        if ord(base) < 128 and marks and all(m in TEX_ACCENTS for m in marks):
            out = base if base not in "ij" else "\\" + base
            for mark in marks:
                out = f"\\{TEX_ACCENTS[mark]}{{{out}}}"
            return out
        self.unmapped.add(ch)
        return "?"

    def text(self, text: str) -> str:
        out: list[str] = []
        sup: list[str] = []
        for ch in str(text):
            if ch in TEX_SUPERSCRIPT:  # runs such as "10⁻¹⁶" -> 10$^{-16}$
                sup.append(TEX_SUPERSCRIPT[ch])
                continue
            if sup:
                out.append("$^{" + "".join(sup) + "}$")
                sup = []
            out.append(self.char(ch))
        if sup:
            out.append("$^{" + "".join(sup) + "}$")
        return "".join(out)

    def ascii_code(self, text: str) -> str:
        out = []
        for ch in str(text):
            if ord(ch) < 128:
                out.append(ch)
            elif ch in ASCII_CODE:
                out.append(ASCII_CODE[ch])
            else:
                plain = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
                if not plain:
                    self.unmapped.add(ch)
                out.append(plain or "?")
        return "".join(out)


def smart_quotes(escaped: str) -> str:
    """Straight double quotes -> TeX quotes (`` and '')."""
    escaped = re.sub(r'(^|[\s(\[{~])"', r"\1``", escaped)
    return escaped.replace('"', "''")


def url_arg(url: str) -> str:
    """Argument for ``\\url{}`` (the url package needs %, # escaped inside arguments)."""
    return re.sub(r"([%#{}\\])", r"\\\1", url.strip())


# ----------------------------------------------------------------------
# Structural check of the generated source
# ----------------------------------------------------------------------

_VERBATIM = re.compile(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", re.DOTALL)


def validate_tex(source: str) -> list[str]:
    """Balanced braces (outside verbatim, comments and escapes), matching environments, ASCII."""
    problems: list[str] = []
    for n, line in enumerate(source.splitlines(), 1):
        bad = sorted({ch for ch in line if ord(ch) > 127})
        if bad:
            problems.append(f"line {n}: non-ASCII characters {''.join(bad)!r}")
    body = _VERBATIM.sub("", source)
    depth = 0
    stack: list[str] = []
    for n, line in enumerate(body.splitlines(), 1):
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\":
                m = re.match(r"\\(begin|end)\{([^}]*)\}", line[i:])
                if m:
                    if m.group(1) == "begin":
                        stack.append(m.group(2))
                    elif not stack or stack[-1] != m.group(2):
                        problems.append(
                            f"line {n}: \\end{{{m.group(2)}}} does not match "
                            f"{stack[-1] if stack else 'nothing'}"
                        )
                    else:
                        stack.pop()
                    i += len(m.group(0))
                    continue
                i += 2  # skip the escaped character / first letter of a command
                continue
            if ch == "%":
                break
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    problems.append(f"line {n}: unbalanced closing brace")
                    depth = 0
            i += 1
    if depth:
        problems.append(f"{depth} unclosed brace(s)")
    if stack:
        problems.append(f"unclosed environment(s): {', '.join(stack)}")
    return problems


# ----------------------------------------------------------------------
# BibTeX
# ----------------------------------------------------------------------

_SUFFIX = re.compile(r",?\s+(Jr\.?|Sr\.?|II|III|IV)$")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def _bib_text(esc: Escaper, text: Any) -> str:
    return esc.text(re.sub(r"\s+", " ", str(text or "")).strip())


def bib_author(esc: Escaper, name: str) -> str:
    name = name.strip()
    if name.lower().rstrip(".") == "et al":
        return "others"
    m = _SUFFIX.search(name)
    if m:
        rest = name[: m.start()].strip().split()
        if len(rest) >= 2:
            return f"{esc.text(rest[-1])}, {esc.text(m.group(1))}, {esc.text(' '.join(rest[:-1]))}"
    if is_organisation(name):
        return "{" + esc.text(name) + "}"
    return esc.text(name)


def bib_entry(esc: Escaper, entry: dict[str, Any]) -> str:
    """One BibTeX entry mirroring ``report/lib/references.py`` (IEEEtran.bst)."""
    kind = str(entry.get("type", "misc")).lower()
    venue = re.sub(r",?\s*https?://\S+$", "", str(entry.get("venue") or "")).strip().rstrip(",")
    ident = str(entry.get("identifier") or "")
    doi = str(entry.get("doi") or "").strip()
    url = str(entry.get("url") or "").strip()
    arxiv = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", ident)
    fields: list[tuple[str, str]] = []
    authors = [a for a in entry.get("authors") or [] if str(a).strip()]
    if authors:
        fields.append(("author", " and ".join(bib_author(esc, str(a)) for a in authors)))
    title = str(entry.get("title") or "").strip().rstrip(".")
    fields.append(("title", "{" + _bib_text(esc, title) + "}"))
    notes: list[str] = []
    with_url = False
    if kind in {"inproceedings", "conference"}:
        btype = "inproceedings"
        fields.append(("booktitle", _bib_text(esc, venue)))
        if entry.get("pages"):
            fields.append(("pages", _bib_text(esc, str(entry["pages"]).replace("-", "--"))))
        if not doi and arxiv:
            notes.append(f"arXiv:{arxiv.group(1)}")
    elif kind in {"article", "guideline"} and (entry.get("volume") or entry.get("pages")):
        btype = "article"
        fields.append(("journal", _bib_text(esc, venue)))
        for key in ("volume", "issue", "pages"):
            if entry.get(key):
                value = str(entry[key]).replace("-", "--") if key == "pages" else entry[key]
                fields.append(("number" if key == "issue" else key, _bib_text(esc, value)))
    elif kind == "book":
        btype = "book"
        fields.append(("publisher", _bib_text(esc, venue)))
    elif kind == "preprint":
        btype = "misc"
        howpublished = f"arXiv preprint arXiv:{arxiv.group(1)}" if arxiv else venue
        fields.append(("howpublished", _bib_text(esc, howpublished)))
    elif kind == "patent":
        btype = "patent"
        num = re.search(r"(US)\s*([\d,]+)\s*(B\d|A\d)", ident)
        granted = re.search(r"granted\s+(\d{4})-(\d{2})-(\d{2})", ident)
        fields.append(("nationality", "U.S."))
        fields.append(
            ("number", esc.text(f"{num.group(2)} {num.group(3)}" if num else ident.split("(")[0]))
        )
        if granted:
            fields.append(("month", _MONTHS[int(granted.group(2)) - 1]))
            fields.append(("day", str(int(granted.group(3)))))
        assignee = re.search(r"assignee:\s*([^)]+)\)", str(entry.get("venue") or ""))
        if assignee:
            fields.append(("assignee", _bib_text(esc, assignee.group(1))))
        with_url = True
    elif kind in {"web", "software"}:
        btype = "misc"
        fields.append(("howpublished", _bib_text(esc, venue)))
        accessed = re.search(r"accessed\s+(\d{4})-(\d{2})-(\d{2})", ident, re.IGNORECASE)
        if accessed:
            month = (
                "Jan.",
                "Feb.",
                "Mar.",
                "Apr.",
                "May",
                "Jun.",
                "Jul.",
                "Aug.",
                "Sep.",
                "Oct.",
                "Nov.",
                "Dec.",
            )[int(accessed.group(2)) - 1]
            notes.append(f"Accessed: {month} {int(accessed.group(3))}, {accessed.group(1)}")
        with_url = True
    else:  # report, guideline without volume, law, misc
        btype = "misc"
        fields.append(("howpublished", _bib_text(esc, venue)))
        isbn = re.match(r"ISBN[^;(]*", ident)
        if kind in {"report", "guideline"} and isbn and len(isbn.group(0)) <= 80:
            notes.append(isbn.group(0).strip().rstrip(","))
        with_url = not doi
    if entry.get("year"):
        fields.append(("year", str(entry["year"])))
    if doi:
        notes.append(f"doi: {doi}")
    if notes:
        fields.append(("note", "; ".join(_bib_text(esc, n) for n in notes)))
    if with_url and url:
        fields.append(("url", url_arg(url)))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@{btype}{{{entry['key']},\n{body}\n}}\n"


def build_bib(paper: pc.Paper, esc: Escaper) -> str:
    head = (
        "% References for main.tex -- GENERATED by report/paper/build_latex.py from the verified\n"
        "% entries in report/references/references.json and patents.json (only cited keys).\n"
        "% Edit those JSON files, not this file.\n\n"
    )
    return head + "\n".join(bib_entry(esc, paper.registry.entries[k]) for k in paper.registry.order)


# ----------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------


class LatexRenderer:
    def __init__(self, paper: pc.Paper, out_dir: Path) -> None:
        self.paper = paper
        self.out_dir = out_dir
        self.esc = Escaper()
        self.out: list[str] = []
        self.figures: dict[str, str] = {}  # source path -> latex path
        self.pending: list[str] = []
        self.stats: dict[str, int] = {
            "figures": 0,
            "figures_pending": 0,
            "tables": 0,
            "tables_pending": 0,
            "algorithms": 0,
            "to_be_filled": 0,
        }

    # -- inline ---------------------------------------------------------------

    def pend(self, text: str) -> str:
        return r"\pending{" + self.esc.text(text) + "}"

    @staticmethod
    def _wrap(s: str, bold: bool, italic: bool) -> str:
        if not s:
            return s
        if italic:
            s = r"\emph{" + s + "}"
        if bold:
            s = r"\textbf{" + s + "}"
        return s

    def inline(self, text: str, *, markup: bool = True) -> str:
        if not markup:
            raw = str(text)
            if raw.lstrip().upper().startswith(PENDING_PREFIXES):
                self.pending.append(raw)
                return self.pend(raw)
            return self.esc.text(raw)
        parts: list[str] = []
        for tok in tokenize(text or ""):
            if tok.kind == "text":
                parts.append(
                    self._wrap(smart_quotes(self.esc.text(tok.text)), tok.bold, tok.italic)
                )
            elif tok.kind == "code":
                code = "".join(
                    self.esc.char(ch) + (r"\allowbreak{}" if ch in "/._-" else "")
                    for ch in tok.text
                )
                parts.append(self._wrap(r"\texttt{" + code + "}", tok.bold, tok.italic))
            elif tok.kind == "cite":
                parts.append(self._cite(tok.keys))
            elif tok.kind == "value":
                value, ok = self.paper.ctx.render(tok.text)
                if ok:
                    parts.append(self._wrap(self.esc.text(value), tok.bold, tok.italic))
                else:
                    self.pending.append(f"value: {tok.text}")
                    parts.append(self.pend(value))
            elif tok.kind == "fill":
                if tok.text.upper().startswith("[TO BE FILLED"):
                    self.stats["to_be_filled"] += 1
                else:
                    self.pending.append(tok.text)
                parts.append(self.pend(tok.text))
            elif tok.kind == "link":
                label = self._wrap(self.esc.text(tok.text), tok.bold, tok.italic)
                parts.append(label + r"~(\url{" + url_arg(tok.url) + "})")
            elif tok.kind == "br":
                parts.append(r"\newline ")
        return "".join(parts)

    def _cite(self, keys: list[str]) -> str:
        out: list[str] = []
        for key in (k for k in keys if pc.is_xref(k)):
            if key in self.paper.labels:
                out.append(XREF_TEX[key.split(":")[0]] + r"\ref{" + key + "}")
            else:
                self.pending.append(f"cross-reference not found: {key}")
                out.append(self.pend(f"[REF NOT FOUND: {key}]"))
        refs = [k for k in keys if not pc.is_xref(k)]
        known = [k for k in refs if k in self.paper.registry.entries]
        if known:
            out.append(r"\cite{" + ",".join(known) + "}")
        for key in refs:
            if key not in self.paper.registry.entries:
                self.pending.append(f"citation not found: {key}")
                out.append(self.pend(f"[CITATION NOT FOUND: {key}]"))
        return ", ".join(out)

    # -- front ----------------------------------------------------------------

    def front(self) -> str:
        paper = self.paper
        authors = []
        for author in paper.authors or [pc.Author("[TO BE FILLED: authors]")]:
            lines = [self.inline(author.name)]
            aff = [self.inline(line) for line in author.affiliation]
            if aff:
                aff[0] = r"\textit{" + aff[0] + "}"
            if author.email:
                aff.append(self.inline(author.email))
            block = r"\IEEEauthorblockN{" + lines[0] + "}"
            if aff:
                block += "\n" + r"\IEEEauthorblockA{" + " \\\\\n".join(aff) + "}"
            authors.append(block)
        abstract = "\n\n".join(self.inline(p) for p in paper.abstract)
        terms = self.inline(paper.index_terms) if paper.index_terms else ""
        out = [
            r"\title{" + self.inline(paper.title) + "}",
            "",
            r"\author{" + "\n\\and\n".join(authors) + "}",
            "",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            abstract,
            r"\end{abstract}",
            "",
        ]
        if terms:
            out += [r"\begin{IEEEkeywords}", terms, r"\end{IEEEkeywords}", ""]
        return "\n".join(out)

    # -- blocks -----------------------------------------------------------------

    def emit(self, *lines: str) -> None:
        self.out.extend(lines)

    def heading(self, h: Heading) -> None:
        el = self.paper.element(h)
        title = self.inline(h.text)
        label = r"\label{" + el.label + "}" if el is not None and el.label else ""
        if h.level == 1:
            cmd = r"\section" if el is not None else r"\section*"
        elif h.level == 2:
            cmd = r"\subsection" if h.numbered else r"\subsection*"
        elif h.level == 3:
            cmd = r"\subsubsection" if h.numbered else r"\subsubsection*"
        else:
            cmd = r"\paragraph"
        self.emit("", cmd + "{" + title + "}" + label, "")

    def paragraph(self, p: Paragraph) -> None:
        text = self.inline(p.text)
        if p.style == "quote":
            self.emit(r"\begin{quote}", text, r"\end{quote}", "")
        elif p.style == "center":
            self.emit(r"\begin{center}", text, r"\end{center}", "")
        else:
            self.emit(text, "")

    def list_block(self, lb: ListBlock) -> None:
        stack: list[str] = []  # environment names, one per open level
        for item in lb.items:
            level = max(0, min(2, item.level))
            env = "enumerate" if item.ordered else "itemize"
            while len(stack) > level + 1:
                self.emit(r"\end{" + stack.pop() + "}")
            if len(stack) == level + 1 and stack[-1] != env:
                self.emit(r"\end{" + stack.pop() + "}")
            while len(stack) < level + 1:
                stack.append(env)
                self.emit(r"\begin{" + env + "}")
            self.emit(r"\item " + self.inline(item.text))
        while stack:
            self.emit(r"\end{" + stack.pop() + "}")
        self.emit("")

    def _copy_figure(self, src: Path) -> str:
        key = str(src.resolve())
        if key in self.figures:
            return self.figures[key]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", src.stem) or "figure"
        name = f"{stem}{src.suffix.lower()}"
        taken = set(self.figures.values())
        n = 2
        while f"figures/{name}" in taken:
            name = f"{stem}_{n}{src.suffix.lower()}"
            n += 1
        dest = self.out_dir / "figures" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self.figures[key] = f"figures/{name}"
        return self.figures[key]

    def figure(self, f: Figure) -> None:
        el = self.paper.element(f)
        wide = el is not None and el.wide
        env = "figure*" if wide else "figure"
        full = r"\textwidth" if wide else r"\columnwidth"
        path = self.paper.resolve_figure(f.path)
        lines = [r"\begin{" + env + "}[t]", r"\centering"]
        if path is None:
            self.pending.append(f"figure: {f.path}")
            self.stats["figures_pending"] += 1
            lines.append(
                r"\fbox{\parbox{0.9"
                + full
                + r"}{\centering "
                + self.pend(f"[FIGURE PENDING: {f.path}]")
                + "}}"
            )
        else:
            rel = self._copy_figure(path)
            frac = 1.0 if wide or not f.width else min(1.0, f.width / pc.COLUMN_WIDTH_IN)
            width = full if frac >= 0.99 else f"{frac:.2f}{full}"
            lines.append(r"\includegraphics[width=" + width + "]{" + rel + "}")
            self.stats["figures"] += 1
        caption = self.inline(f.caption or (path.name if path else f.path))
        label = r"\label{" + el.label + "}" if el is not None and el.label else ""
        lines += [r"\caption{" + caption + "}" + label, r"\end{" + env + "}", ""]
        self.emit(*lines)

    @staticmethod
    def _small(t: Table) -> bool:
        """A generated table with a font size of 7 pt or less (a compact table measured by
        hand, ``paper_tables.py``) is set in \\scriptsize with natural column widths and,
        if it is still wider than the column, scaled to fit with \\resizebox."""
        return bool(t.font_size) and float(t.font_size) <= 7.0

    @staticmethod
    def _natural_pt(t: Table, ncols: int, char_pt: float, pad_pt: float) -> float:
        width = 0.0
        for c in range(ncols):
            cells = [pc._plain(t.header[c]) if c < len(t.header) else ""]
            cells += [pc._plain(r[c]) for r in t.rows if c < len(r)]
            width += max((len(x) for x in cells), default=1) * char_pt + pad_pt
        return width

    def _colspec(self, t: Table, ncols: int, avail_pt: float) -> tuple[str, bool]:
        """Column spec. A table that fits keeps l/c/r columns; otherwise its wrappable
        columns become ``X`` columns of tabularx, sized in proportion to their content."""
        aligns = (t.align + ["l"] * ncols)[:ncols]
        natural: list[int] = []
        minimum: list[int] = []
        for c in range(ncols):
            cells = [pc._plain(t.header[c]) if c < len(t.header) else ""]
            cells += [pc._plain(r[c]) for r in t.rows if c < len(r)]
            natural.append(max((len(x) for x in cells), default=1))
            minimum.append(max((len(w) for x in cells for w in x.split()), default=1))
        char_pt, pad_pt = 4.0, 8.0  # \footnotesize Times, \tabcolsep = 4pt
        if sum(n * char_pt + pad_pt for n in natural) <= avail_pt:
            return "".join(a if a in "lcr" else "l" for a in aligns), False
        x_cols = [c for c in range(ncols) if natural[c] > minimum[c] + 2]
        if not x_cols:
            x_cols = [natural.index(max(natural))]
        total = sum(natural[c] for c in x_cols)
        spec = []
        for c in range(ncols):
            if c in x_cols:
                factor = len(x_cols) * natural[c] / total
                align = {"c": r"\centering", "r": r"\raggedleft"}.get(aligns[c], r"\raggedright")
                spec.append(
                    r">{" + align + r"\arraybackslash\hsize=" + f"{factor:.2f}" + r"\hsize}X"
                )
            else:
                spec.append(aligns[c] if aligns[c] in "lcr" else "l")
        return "".join(spec), True

    def table(self, t: Table) -> None:
        if not t.header and not t.rows:
            return
        el = self.paper.element(t)
        wide = el is not None and el.wide
        env = "table*" if wide else "table"
        ncols = max(len(t.header), max((len(r) for r in t.rows), default=0))
        header = t.header + [""] * (ncols - len(t.header))
        rows = [r + [""] * (ncols - len(r)) for r in t.rows]
        avail = 516.0 if wide else 252.0
        small = self._small(t)
        if small:
            aligns = (t.align + ["l"] * ncols)[:ncols]
            spec, tabularx = "".join(a if a in "lcr" else "l" for a in aligns), False
            # \scriptsize Times, \tabcolsep = 2pt
            resize = self._natural_pt(t, ncols, 3.5, 4.0) > avail
        else:
            spec, tabularx = self._colspec(t, ncols, avail)
            resize = False
        width = r"\textwidth" if wide else r"\columnwidth"
        lines = [r"\begin{" + env + "}[t]", r"\centering"]
        if t.caption or el is not None:
            label = r"\label{" + el.label + "}" if el is not None and el.label else ""
            lines.append(r"\caption{" + self.inline(t.caption or "") + "}" + label)
        lines.append(r"\scriptsize" if small else r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{2pt}" if small else r"\setlength{\tabcolsep}{4pt}")
        begin = (
            (r"\begin{tabularx}{" + width + "}{" + spec + "}")
            if tabularx
            else (r"\begin{tabular}{" + spec + "}")
        )
        if resize:
            lines.append(r"\resizebox{" + width + "}{!}{%")
        lines += [begin, r"\toprule"]
        lines.append(" & ".join(self.inline(h, markup=t.markup) for h in header) + r" \\")
        lines.append(r"\midrule")
        for row in rows:
            cells = [self.inline(str(c), markup=t.markup) for c in row]
            if row and str(row[0]).strip().lower() in {"total", "overall"}:
                lines.append(r"\midrule")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabularx}" if tabularx else r"\end{tabular}"]
        if resize:
            lines.append("}")
        if t.note:
            lines.append(
                r"\par\vspace{2pt}{\scriptsize\itshape "
                + self.inline(t.note, markup=False)
                + r"\par}"
            )
        lines += [r"\end{" + env + "}", ""]
        self.emit(*lines)
        self.stats["tables"] += 1

    def pending_table(self, t: PendingTable) -> None:
        el = self.paper.element(t)
        label = r"\label{" + el.label + "}" if el is not None and el.label else ""
        self.pending.append(f"table: {t.key}")
        self.stats["tables_pending"] += 1
        self.emit(
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{" + self.inline(t.caption or t.key) + "}" + label,
            self.pend(f"[TABLE PENDING: {t.key}]"),
            r"\end{table}",
            "",
        )

    def listing(self, lines: list[str]) -> list[str]:
        body = [self.esc.ascii_code(line.rstrip()) for line in lines]
        return [r"\begin{lstlisting}", *body, r"\end{lstlisting}"]

    def algorithm(self, a: Algorithm) -> None:
        el = self.paper.element(a)
        label = r"\label{" + el.label + "}" if el is not None and el.label else ""
        self.emit(
            r"\begin{algorithm}[t]",
            r"\caption{" + self.inline(a.title) + "}" + label,
            *self.listing(a.lines),
            r"\end{algorithm}",
            "",
        )
        self.stats["algorithms"] += 1

    def render(self, blocks: list[Block]) -> None:
        for block in blocks:
            if isinstance(block, Heading):
                self.heading(block)
            elif isinstance(block, Paragraph):
                self.paragraph(block)
            elif isinstance(block, ListBlock):
                self.list_block(block)
            elif isinstance(block, Figure):
                self.figure(block)
            elif isinstance(block, Table):
                self.table(block)
            elif isinstance(block, PendingTable):
                self.pending_table(block)
            elif isinstance(block, Algorithm):
                self.algorithm(block)
            elif isinstance(block, CodeBlock):
                self.emit(*self.listing(block.lines), "")
            elif isinstance(block, Pending):
                self.pending.append(block.text)
                self.emit(r"\noindent" + self.pend(block.text), "")
            elif isinstance(block, PageBreak):
                continue

    def document(self) -> str:
        front = self.front()
        self.render(self.paper.blocks)
        body = "\n".join(self.out).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        return PREAMBLE + "\n" + front + "\n" + body + "\n" + POSTAMBLE


PREAMBLE = r"""% MedBridge -- IEEE conference paper.
% GENERATED by report/paper/build_latex.py from report/paper/content/*.md and the results
% files of the repository. Rebuild instead of editing by hand (or stop regenerating once
% the team starts editing this file in Overleaf). Compile: pdflatex, bibtex, pdflatex x2.
% Red bold text (\pending{...}) marks placeholders and results that do not exist yet.
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{url}
\DeclareRobustCommand{\pending}[1]{\textcolor{red}{\textbf{#1}}}
\lstset{basicstyle=\scriptsize\ttfamily, columns=fullflexible, keepspaces=true,
  breaklines=true, breakatwhitespace=false, frame=single, framesep=2pt,
  xleftmargin=3pt, xrightmargin=3pt, aboveskip=2pt, belowskip=0pt, upquote=true,
  showstringspaces=false}

\begin{document}
"""

POSTAMBLE = r"""
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------


@dataclass
class LatexResult:
    main_tex: Path
    bib: Path
    figures: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    log: dict[str, Any] = field(default_factory=dict)


def build(
    content_dir: Path = pc.DEFAULT_CONTENT,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = pc.REPO_ROOT,
    anonymous: bool = False,
    log_path: Path | None = DEFAULT_LOG,
) -> LatexResult:
    paper = pc.load_paper(Path(content_dir), Path(repo_root), anonymous=anonymous)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = LatexRenderer(paper, out_dir)
    tex = renderer.document()
    bib = build_bib(paper, renderer.esc)
    main_tex = out_dir / "main.tex"
    bib_path = out_dir / "references.bib"
    main_tex.write_text(tex, encoding="utf-8")
    bib_path.write_text(bib, encoding="utf-8")
    # remove figures copied by earlier builds that the paper no longer uses
    used = {out_dir / rel for rel in renderer.figures.values()}
    for old in (out_dir / "figures").glob("*") if (out_dir / "figures").is_dir() else []:
        if old.is_file() and old not in used:
            old.unlink()
    problems = validate_tex(tex) + [f"references.bib: {p}" for p in validate_tex(bib)]
    if renderer.esc.unmapped:
        problems.append(
            "unmapped characters replaced by '?': " + " ".join(sorted(renderer.esc.unmapped))
        )
    log = pc.base_log(paper)
    log.update(
        {
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "output_dir": pc._rel(out_dir, Path(repo_root)),
            "figures_copied": sorted(renderer.figures.values()),
            "stats": renderer.stats,
            "pending_markers": sorted(set(renderer.pending)),
            "structure_problems": problems,
        }
    )
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return LatexResult(main_tex, bib_path, sorted(used), problems, log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MedBridge IEEE paper (LaTeX).")
    parser.add_argument("--content-dir", type=Path, default=pc.DEFAULT_CONTENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--anonymous", action="store_true", help="double-blind review copy")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on unknown citations, unresolved xrefs or structure problems",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.content_dir.is_dir():
        print(f"content directory not found: {args.content_dir}", file=sys.stderr)
        return 2
    result = build(
        args.content_dir,
        args.out_dir,
        anonymous=args.anonymous,
        log_path=None if args.no_log else args.log,
    )
    log = result.log
    if not args.quiet:
        print(f"LaTeX: {result.main_tex}")
        print(f"BibTeX: {result.bib} ({log['references']} entries)")
        print(f"figures: {', '.join(log['figures_copied']) or 'none'}")
        for key in (
            "citation_keys_missing",
            "unresolved_xrefs",
            "structure_problems",
            "pending_markers",
            "warnings",
        ):
            if log.get(key):
                print(f"{key}: {len(log[key])}")
                for item in log[key][:20]:
                    print(f"  - {item}")
    if args.strict and (log["citation_keys_missing"] or log["unresolved_xrefs"] or result.problems):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

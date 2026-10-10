"""
Parser for the report content markup (``report/content/*.md``).

Block level
-----------
``# Title``                 chapter (numbered "CHAPTER n" in the main matter)
``## Title`` / ``### Title`` section n.m / subsection n.m.k
``#* Title``                unnumbered heading (front matter, or any unnumbered heading)
``- item`` / ``1. item``    bullet / numbered list (indent by 2+ spaces to nest)
``> text``                  block quote
``Table: Caption``          caption for the pipe table (or ``[[table:KEY]]``) that follows
``| a | b |``               pipe table (``|---|:---:|---:|`` row sets alignment)
``[[figure:path|Caption|width=6.0]]``   figure (path relative to the repo root)
``[[table:KEY]]`` / ``[[table:KEY|Caption]]``   generated data table (report/lib/tables.py)
``[[include:file.md]]``     inline another content file
`````algorithm`` fence      bordered pseudocode box; first line = title
``\\pagebreak``             page break
``<!-- ... -->``            comment (removed)

Inline
------
``**bold**``, ``*italic*``, ```code```, ``[text](url)``, ``[@key]`` / ``[@a; @b]``
citations, ``{{path|filter}}`` result values, ``[TO BE FILLED: ...]`` placeholders
(yellow highlight; ``[PENDING ...]`` is highlighted too), ``\\`` escapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------
# Block model
# ----------------------------------------------------------------------


@dataclass
class Heading:
    level: int
    text: str
    numbered: bool = True
    source: str = ""


@dataclass
class Paragraph:
    text: str
    style: str | None = None  # None | "quote" | "center"
    source: str = ""


@dataclass
class ListItem:
    level: int
    ordered: bool
    text: str


@dataclass
class ListBlock:
    items: list[ListItem] = field(default_factory=list)


@dataclass
class Figure:
    path: str
    caption: str
    width: float | None = None
    source: str = ""


@dataclass
class Table:
    caption: str | None
    header: list[str]
    rows: list[list[str]]
    align: list[str] = field(default_factory=list)  # "l" | "c" | "r" per column
    markup: bool = True  # cells contain inline markup
    note: str | None = None  # small print under the table (data source)
    key: str | None = None  # generator key for generated tables
    font_size: float | None = None
    col_widths: list[float] | None = None  # relative weights


@dataclass
class GeneratedTable:
    key: str
    caption: str | None = None
    source: str = ""


@dataclass
class PendingTable:
    key: str
    caption: str | None = None
    reason: str = ""


@dataclass
class Algorithm:
    title: str
    lines: list[str]


@dataclass
class CodeBlock:
    lang: str
    lines: list[str]


@dataclass
class PageBreak:
    pass


@dataclass
class Pending:
    """A highlighted pending marker paragraph (missing include, unknown directive...)."""

    text: str


Block = (
    Heading
    | Paragraph
    | ListBlock
    | Figure
    | Table
    | GeneratedTable
    | PendingTable
    | Algorithm
    | CodeBlock
    | PageBreak
    | Pending
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})(\*?)\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_DIRECTIVE = re.compile(r"^\[\[\s*(figure|table|include)\s*:(.*)\]\]\s*$", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^Table\s*:\s*(.+)$")
_PAGEBREAK = re.compile(r"^\\(pagebreak|newpage|clearpage)\s*$")
PLACEHOLDER_MARK = re.compile(r"^\s*<!--\s*placeholder\b", re.IGNORECASE)


def split_args(text: str) -> list[str]:
    """Split on ``|`` outside ``{{ }}`` and ``[ ]`` (captions may contain values/citations)."""
    parts: list[str] = []
    buf = ""
    brace = bracket = 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            brace += 1
            buf += "{{"
            i += 2
            continue
        if text.startswith("}}", i) and brace:
            brace -= 1
            buf += "}}"
            i += 2
            continue
        ch = text[i]
        if ch == "[":
            bracket += 1
        elif ch == "]" and bracket:
            bracket -= 1
        if ch == "|" and not brace and not bracket:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    parts.append(buf)
    return [p.strip() for p in parts]


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells = split_args(line.replace("\\|", "\x01"))
    return [c.replace("\x01", "|").strip() for c in cells]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells if c)


def _alignment(cells: list[str]) -> list[str]:
    out = []
    for c in cells:
        c = c.replace(" ", "")
        if c.startswith(":") and c.endswith(":"):
            out.append("c")
        elif c.endswith(":"):
            out.append("r")
        else:
            out.append("l")
    return out


def parse_pipe_table(lines: list[str], caption: str | None) -> Table:
    rows = [_split_row(line) for line in lines]
    header = rows[0] if rows else []
    body = rows[1:]
    align: list[str] = []
    if body and _is_separator(body[0]):
        align = _alignment(body[0])
        body = body[1:]
    width = max([len(header)] + [len(r) for r in body]) if rows else 0
    header = header + [""] * (width - len(header))
    body = [r + [""] * (width - len(r)) for r in body]
    align = (align + ["l"] * width)[:width]
    return Table(caption=caption, header=header, rows=body, align=align, markup=True)


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------


@dataclass
class ParseContext:
    content_dir: Path
    include_stack: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    includes_used: list[str] = field(default_factory=list)


def is_placeholder_file(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8")[:200]
    except OSError:
        return False
    return bool(PLACEHOLDER_MARK.match(head))


def parse_file(path: Path, ctx: ParseContext) -> list[Block]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        ctx.warnings.append(f"cannot read {path.name}: {exc}")
        return [Pending(f"[PENDING: content file {path.name} unreadable]")]
    ctx.include_stack.append(path.resolve())
    try:
        return parse_markup(text, ctx, source=path.name, base_dir=path.parent)
    finally:
        ctx.include_stack.pop()


def _resolve_include(name: str, base_dir: Path, ctx: ParseContext) -> Path | None:
    name = name.strip()
    for root in (base_dir, ctx.content_dir):
        candidate = (root / name).resolve()
        if candidate.is_file():
            return candidate
    return None


def parse_markup(
    text: str, ctx: ParseContext, *, source: str = "", base_dir: Path | None = None
) -> list[Block]:
    """Parse one markup string into blocks (includes are expanded recursively)."""
    base_dir = base_dir or ctx.content_dir
    text = _COMMENT.sub("", text.replace("\r\n", "\n").replace("\t", "    "))
    lines = text.split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    pending_caption: str | None = None
    i = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            joined = ""
            for part in para:
                if joined.endswith("\n") or not joined:
                    joined += part.strip()
                else:
                    joined += " " + part.strip()
                if part.endswith("  ") or part.rstrip().endswith("\\"):
                    joined = joined.rstrip("\\").rstrip() + "\n"
            blocks.append(Paragraph(joined.strip("\n"), source=source))
            para = []

    def flush_caption() -> None:
        nonlocal pending_caption
        if pending_caption is not None:
            blocks.append(Paragraph(f"Table: {pending_caption}", source=source))
            pending_caption = None

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_para()
            i += 1
            continue

        # fenced block
        if stripped.startswith("```"):
            flush_para()
            flush_caption()
            lang = stripped[3:].strip().lower()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i].rstrip())
                i += 1
            i += 1  # closing fence
            if lang.startswith("algorithm"):
                while body and not body[0].strip():
                    body.pop(0)
                title = body.pop(0).strip() if body else "Algorithm"
                title = re.sub(r"^(algorithm\s*\d*\s*[:.]\s*)", "", title, flags=re.IGNORECASE)
                while body and not body[-1].strip():
                    body.pop()
                blocks.append(Algorithm(title=title or "Algorithm", lines=body))
            else:
                blocks.append(CodeBlock(lang=lang, lines=body))
            continue

        if _PAGEBREAK.match(stripped):
            flush_para()
            flush_caption()
            blocks.append(PageBreak())
            i += 1
            continue

        m = _HEADING.match(stripped)
        if m and not line.startswith(" "):
            flush_para()
            flush_caption()
            level = len(m.group(1))
            blocks.append(
                Heading(
                    level=level, text=m.group(3).strip(), numbered=not m.group(2), source=source
                )
            )
            i += 1
            continue

        m = _DIRECTIVE.match(stripped)
        if m:
            flush_para()
            kind = m.group(1).lower()
            args = split_args(m.group(2))
            if kind == "figure":
                flush_caption()
                path = args[0] if args else ""
                caption = ""
                width: float | None = None
                for arg in args[1:]:
                    wm = re.fullmatch(r"width\s*=\s*([\d.]+)\s*(in)?", arg)
                    if wm:
                        width = float(wm.group(1))
                    elif not caption:
                        caption = arg
                blocks.append(Figure(path=path, caption=caption, width=width, source=source))
            elif kind == "table":
                key = args[0].strip() if args else ""
                tcaption = args[1] if len(args) > 1 and args[1] else pending_caption
                pending_caption = None
                blocks.append(GeneratedTable(key=key, caption=tcaption, source=source))
            else:  # include
                flush_caption()
                name = args[0] if args else ""
                target = _resolve_include(name, base_dir, ctx)
                if target is None:
                    ctx.warnings.append(f"{source}: include '{name}' not found (pending)")
                    blocks.append(Pending(f"[PENDING: {name}]"))
                elif target in ctx.include_stack:
                    ctx.warnings.append(f"{source}: recursive include '{name}' skipped")
                else:
                    ctx.includes_used.append(target.name)
                    blocks.extend(parse_file(target, ctx))
            i += 1
            continue

        m = _TABLE_CAPTION.match(stripped)
        if m and not para:
            flush_caption()
            pending_caption = m.group(1).strip()
            i += 1
            continue

        if stripped.startswith("|"):
            flush_para()
            tlines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tlines.append(lines[i])
                i += 1
            blocks.append(parse_pipe_table(tlines, pending_caption))
            pending_caption = None
            continue

        om_start = _ORDERED.match(line)
        if om_start and para and not re.match(r"^\s*1[.)]\s", line):
            om_start = None  # only "1." may interrupt a paragraph (wrapped "2019. The ...")
        if _BULLET.match(line) or om_start:
            flush_para()
            flush_caption()
            lb = ListBlock()
            base_indent: int | None = None
            while i < len(lines):
                cur = lines[i].rstrip()
                if not cur.strip():
                    # a blank line ends the list unless the next line is another item
                    nxt = lines[i + 1].rstrip() if i + 1 < len(lines) else ""
                    if _BULLET.match(nxt) or _ORDERED.match(nxt):
                        i += 1
                        continue
                    break
                bm, om = _BULLET.match(cur), _ORDERED.match(cur)
                if bm or om:
                    mm = bm or om
                    assert mm is not None
                    indent = len(mm.group(1))
                    if base_indent is None:
                        base_indent = indent
                    level = max(0, min(2, (indent - base_indent + 1) // 2))
                    lb.items.append(
                        ListItem(
                            level=level,
                            ordered=om is not None and bm is None,
                            text=mm.group(2).strip(),
                        )
                    )
                elif lb.items and not _HEADING.match(cur.strip()):
                    cs = cur.strip()
                    if (
                        _DIRECTIVE.match(cs)
                        or _TABLE_CAPTION.match(cs)
                        or _PAGEBREAK.match(cs)
                        or cs.startswith(("|", "```", ">"))
                    ):
                        break
                    lb.items[-1].text += " " + cs
                else:
                    break
                i += 1
            blocks.append(lb)
            continue

        if stripped.startswith(">"):
            flush_para()
            flush_caption()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(Paragraph(" ".join(quote), style="quote", source=source))
            continue

        flush_caption()
        para.append(raw)
        i += 1

    flush_para()
    flush_caption()
    return blocks


# ----------------------------------------------------------------------
# Inline tokenizer
# ----------------------------------------------------------------------


@dataclass
class Token:
    kind: str  # text | code | cite | value | fill | link | br
    text: str = ""
    bold: bool = False
    italic: bool = False
    url: str = ""
    keys: list[str] = field(default_factory=list)


_HIGHLIGHT_PREFIXES = (
    "[TO BE FILLED",
    "[PENDING",
    "[TABLE PENDING",
    "[FIGURE PENDING",
    "[TODO",
    "[TBD",
    "[CITATION NOT FOUND",
)


def _match_bracket(text: str, start: int) -> int:
    """Index of the ``]`` closing the ``[`` at ``start`` (nested brackets allowed), or -1."""
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                return j
    return -1


def tokenize(text: str) -> list[Token]:
    """Inline markup -> tokens (bold/italic carried as flags on text tokens)."""
    tokens: list[Token] = []
    buf = ""
    bold = italic = False
    i = 0
    n = len(text)

    def flush() -> None:
        nonlocal buf
        if buf:
            tokens.append(Token("text", buf, bold=bold, italic=italic))
            buf = ""

    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] in "\\`*_{}[]()#+-.!|@":
            buf += text[i + 1]
            i += 2
            continue
        if ch == "\n":
            flush()
            tokens.append(Token("br"))
            i += 1
            continue
        if ch == "`":
            end = text.find("`", i + 1)
            if end > i:
                flush()
                tokens.append(Token("code", text[i + 1 : end], bold=bold, italic=italic))
                i = end + 1
                continue
        if ch == "[":
            upper = text[i : i + 24].upper()
            if upper.startswith(_HIGHLIGHT_PREFIXES):
                end = _match_bracket(text, i)
                if end > i:
                    flush()
                    tokens.append(Token("fill", text[i : end + 1], bold=bold, italic=italic))
                    i = end + 1
                    continue
            if text.startswith("[@", i):
                end = text.find("]", i)
                if end > i:
                    flush()
                    body = text[i + 1 : end]
                    keys = [k.strip().lstrip("@").strip() for k in re.split(r"[;,]", body)]
                    tokens.append(Token("cite", keys=[k for k in keys if k]))
                    i = end + 1
                    continue
            lm = re.compile(r"\[([^\[\]]+)\]\(([^()\s]+)\)").match(text, i)
            if lm:
                flush()
                tokens.append(Token("link", lm.group(1), bold=bold, italic=italic, url=lm.group(2)))
                i = lm.end()
                continue
        if text.startswith("{{", i):
            end = text.find("}}", i + 2)
            if end > i:
                flush()
                tokens.append(Token("value", text[i + 2 : end].strip(), bold=bold, italic=italic))
                i = end + 2
                continue
        if text.startswith("**", i):
            flush()
            bold = not bold
            i += 2
            continue
        if ch == "*":
            prev_c = text[i - 1] if i > 0 else " "
            next_c = text[i + 1] if i + 1 < n else " "
            opening = not italic and not next_c.isspace()
            closing = italic and not prev_c.isspace()
            if opening or closing:
                flush()
                italic = not italic
                i += 1
                continue
        buf += ch
        i += 1
    flush()
    return tokens


def plain_text(text: str) -> str:
    """Markup stripped to plain text (citations/values kept as their raw form)."""
    out = []
    for tok in tokenize(text):
        if tok.kind in {"text", "code", "fill", "link"}:
            out.append(tok.text)
        elif tok.kind == "br":
            out.append(" ")
        elif tok.kind == "cite":
            out.append("[" + "; ".join("@" + k for k in tok.keys) + "]")
        elif tok.kind == "value":
            out.append("{{" + tok.text + "}}")
    return "".join(out)

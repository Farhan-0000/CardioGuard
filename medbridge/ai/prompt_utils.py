"""
Prompt construction utilities (Modules M-11/M-12).

Shared, deterministic helpers that render the XML-partitioned prompt blocks
used by all four LLM calls (ADL-018, ADL-025, Constitution §7.1):

* ``untrusted_block``  — patient text inside ``<untrusted_user_input>``, with any
  tag-like sequences neutralised so the text cannot close or open XML blocks;
* ``format_snapshot``  — the patient context snapshot as compact JSON;
* ``format_history``   — recent turns in ``<conversation_history>`` (patient turns
  are themselves wrapped in ``<untrusted_user_input>``);
* ``format_chunks``    — evidence chunks as ``<chunk index="n" source=… section=… page=…>``
  so the generator can cite them as ``[n]``.
"""
from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from collections.abc import Sequence
from typing import Any, Protocol

from medbridge.config import get_settings

DEFAULT_EMPTY_CHUNKS_PLACEHOLDER = "(No guideline evidence was retrieved for this question.)"
HISTORY_ASSISTANT_MAX_CHARS = 300

# Unicode look-alikes of "<" and ">" (fullwidth, small form, angle brackets) that a
# tokenizer may read as tag delimiters; folded to ASCII before escaping.
_ANGLE_LOOKALIKES = str.maketrans({
    "＜": "<", "﹤": "<", "‹": "<", "⟨": "<", "〈": "<", "〈": "<",
    "＞": ">", "﹥": ">", "›": ">", "⟩": ">", "〉": ">", "〉": ">",
})
# Characters that render as nothing (or only modify the previous character) and
# could be slipped between "<" and a tag name: every code point whose Unicode
# category is a format character (Cf), a combining mark (Mn/Me), a separator
# (Zs/Zl/Zp) or a control character (Cc), plus the remaining
# Default_Ignorable_Code_Point ranges (Hangul fillers, reserved ignorables and the
# tag/variation-selector planes). Built once at import; only used *inside* tag
# matches, so ZWJ emoji sequences and bidi marks in ordinary text are untouched.
_INVISIBLE_CATEGORIES: frozenset[str] = frozenset({"Cf", "Mn", "Me", "Zs", "Zl", "Zp", "Cc"})
_DEFAULT_IGNORABLE_EXTRA: tuple[tuple[int, int], ...] = (
    (0x115F, 0x1160), (0x2065, 0x2065), (0x3164, 0x3164), (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8), (0xE0000, 0xE0FFF),
)


def _invisible_ranges() -> list[tuple[int, int]]:
    points = {
        cp for cp in range(sys.maxunicode + 1)
        if unicodedata.category(chr(cp)) in _INVISIBLE_CATEGORIES
    }
    for low, high in _DEFAULT_IGNORABLE_EXTRA:
        points.update(range(low, high + 1))
    ranges: list[tuple[int, int]] = []
    for cp in sorted(points):
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], cp)
        else:
            ranges.append((cp, cp))
    return ranges


def _char_class(ranges: list[tuple[int, int]]) -> str:
    return "[" + "".join(
        f"\\U{low:08x}" if low == high else f"\\U{low:08x}-\\U{high:08x}"
        for low, high in ranges
    ) + "]"


_INVISIBLE = _char_class(_invisible_ranges())
_GAP = rf"(?:\s|{_INVISIBLE})*"
# Anything that looks like an XML/HTML tag: <tag>, </tag>, <tag attr="x">.
_TAG_LIKE = re.compile(rf"<{_GAP}(/?){_GAP}([A-Za-z_][\w:.-]*)([^<>]*)>")
# A "<" that starts a tag-like token even when the tag is never closed
# ("</untrusted_user_input" at the end of the text, "<!--", "<?xml", "< /x").
_TAG_OPEN = re.compile(rf"<(?={_GAP}[/!?]?{_GAP}[A-Za-z_!?/])")


class ChatTurnLike(Protocol):
    """Structural type of ``medbridge.state.turn_store.ChatTurn``."""

    role: str
    content: str


def neutralize_tags(text: str) -> str:
    """Escape tag-like sequences, e.g. ``</untrusted_user_input>`` → ``&lt;/…&gt;``.

    Used on every piece of untrusted text placed in a prompt (patient messages,
    history, snapshot JSON, extractor intent, evidence text) so it can neither
    close nor open a prompt block:

    * complete tags (``<x>``, ``</x>``, ``< / x >``, ``<x attr="…">``) are escaped,
      also with invisible characters (format, combining, separator, control or
      other default-ignorable code points) between ``<`` and the name;
    * an unterminated tag opener (``</untrusted_user_input`` at the end of the
      text, ``<!--``, ``<?xml``) has its ``<`` escaped;
    * Unicode look-alike angle brackets (``＜``/``＞``, ``‹``/``›``, ``⟨``/``⟩``…)
      are folded to ASCII first, so they are escaped too.

    Plain comparisons such as ``BP < 140`` or ``> 90`` are left untouched (a
    lone ``>`` cannot open or close a block).
    """
    folded = (text or "").translate(_ANGLE_LOOKALIKES)
    escaped = _TAG_LIKE.sub(
        lambda m: f"&lt;{m.group(1)}{m.group(2)}{m.group(3)}&gt;", folded
    )
    return _TAG_OPEN.sub("&lt;", escaped)


def untrusted_block(text: str) -> str:
    """Wrap patient-authored text in ``<untrusted_user_input>`` tags (NON-NEGOTIABLE #5)."""
    return f"<untrusted_user_input>\n{neutralize_tags(text).strip()}\n</untrusted_user_input>"


def format_snapshot(snapshot: dict[str, Any] | None) -> str:
    """Render the patient context snapshot as compact JSON (``{}`` when nothing is known)."""
    if not snapshot:
        return "{}"
    return neutralize_tags(json.dumps(snapshot, default=str, ensure_ascii=False))


def format_history(
    history: Sequence[ChatTurnLike] | None,
    max_turns: int,
    assistant_max_chars: int = HISTORY_ASSISTANT_MAX_CHARS,
) -> str:
    """Render the last ``max_turns`` turns as ``<turn role=…>`` elements (empty string if none).

    Patient turns are wrapped in ``<untrusted_user_input>``; assistant turns are
    truncated to ``assistant_max_chars`` characters to keep prompts small.
    """
    if not history or max_turns <= 0:
        return ""
    lines: list[str] = []
    for turn in list(history)[-max_turns:]:
        content = (turn.content or "").strip()
        if not content:
            continue
        if turn.role == "user":
            lines.append(f'<turn role="patient">{untrusted_block(content)}</turn>')
        else:
            if len(content) > assistant_max_chars:
                content = content[:assistant_max_chars].rsplit(" ", 1)[0] + " …"
            lines.append(f'<turn role="assistant">{neutralize_tags(content)}</turn>')
    return "\n".join(lines)


def _field(chunk: Any, name: str) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name)
    return getattr(chunk, name, None)


def chunk_text(chunk: Any) -> str:
    """Text of a chunk in any supported shape (model, dict, (text, score) tuple, str)."""
    if isinstance(chunk, tuple) and chunk:
        return str(chunk[0])
    if isinstance(chunk, str):
        return chunk
    return str(_field(chunk, "chunk_text") or "")


def chunk_source_label(chunk: Any) -> str:
    """Human-readable source, e.g. ``"AHA/ACC 2025 — Guideline for High Blood Pressure"``.

    Falls back to ``guideline_id`` when organisation/year/title are unavailable.
    """
    org = str(_field(chunk, "org") or "").strip()
    year = str(_field(chunk, "year") or "").strip()
    title = str(_field(chunk, "doc_title") or "").strip()
    head = " ".join(part for part in (org, year) if part)
    if head and title:
        return f"{head} — {title}"
    if head or title:
        return head or title
    return str(_field(chunk, "guideline_id") or "Clinical guideline")


def _is_html(chunk: Any) -> bool:
    return str(_field(chunk, "source_format") or "").strip().lower() == "html"


def _page_label(chunk: Any) -> str:
    """Physical page(s) of a chunk; empty for HTML sources (``page_number`` is a section index)."""
    if _is_html(chunk):
        return ""
    try:
        start = int(_field(chunk, "page_number") or 0)
        end = int(_field(chunk, "page_end") or 0)
    except (TypeError, ValueError):
        return ""
    if start <= 0:
        return ""
    return f"{start}–{end}" if end > start else str(start)


def _section_label(chunk: Any) -> str:
    """Section title; HTML sources read ``"<title> (section N)"``."""
    title = str(_field(chunk, "section_title") or "").strip() or "General"
    if not _is_html(chunk):
        return title
    try:
        index = int(_field(chunk, "page_number") or 0)
    except (TypeError, ValueError):
        index = 0
    return f"{title} (section {index})" if index > 0 else title


def _attr(value: str) -> str:
    """Attribute-safe value: look-alike brackets folded, then quotes and ``<>&`` escaped."""
    return html.escape(value.translate(_ANGLE_LOOKALIKES), quote=True)


def truncate_text(text: str, max_chars: int) -> str:
    """Trim ``text`` to ``max_chars`` at a word boundary, marking the cut with an ellipsis."""
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def format_chunks(
    chunks: Sequence[Any],
    empty_placeholder: str = DEFAULT_EMPTY_CHUNKS_PLACEHOLDER,
    max_chunks: int = 5,
    max_chars: int | None = None,
) -> str:
    """Render up to ``max_chunks`` chunks as indexed ``<chunk>`` elements (ADL-025).

    Each chunk becomes ``<chunk index="n" source="ORG YEAR — title" section="…" page="…">``
    so the generator can cite it as ``[n]``. Supports ``RetrievedChunk`` objects,
    dicts, ``(text, score)`` tuples and raw strings. Text is truncated to
    ``max_chars`` (default ``settings.EVIDENCE_MAX_CHARS_PER_CHUNK``).
    """
    if not chunks:
        return empty_placeholder
    limit = get_settings().EVIDENCE_MAX_CHARS_PER_CHUNK if max_chars is None else max_chars
    pieces: list[str] = []
    for index, chunk in enumerate(list(chunks)[:max_chunks], start=1):
        is_record = not isinstance(chunk, (tuple, str))
        source = chunk_source_label(chunk) if is_record else "Clinical guideline"
        section = _section_label(chunk) if is_record else "General"
        page = _page_label(chunk) if is_record else ""
        attrs = f'index="{index}" source="{_attr(source)}" section="{_attr(section)}"'
        if page:
            attrs += f' page="{page}"'
        body = neutralize_tags(truncate_text(chunk_text(chunk), limit))
        pieces.append(f"<chunk {attrs}>\n{body}\n</chunk>")
    return "\n".join(pieces)

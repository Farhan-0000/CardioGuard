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
from collections.abc import Sequence
from typing import Any, Protocol

from medbridge.config import get_settings

DEFAULT_EMPTY_CHUNKS_PLACEHOLDER = "(No guideline evidence was retrieved for this question.)"
HISTORY_ASSISTANT_MAX_CHARS = 300

# Anything that looks like an XML/HTML tag: <tag>, </tag>, <tag attr="x">.
_TAG_LIKE = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w:.-]*)([^<>]*)>")


class ChatTurnLike(Protocol):
    """Structural type of ``medbridge.state.turn_store.ChatTurn``."""

    role: str
    content: str


def neutralize_tags(text: str) -> str:
    """Escape tag-like sequences, e.g. ``</untrusted_user_input>`` → ``&lt;/…&gt;``.

    Plain comparisons such as ``BP < 140`` are left untouched; only sequences
    that could be parsed as XML tags are escaped, which prevents prompt
    injection by closing or opening prompt blocks.
    """
    return _TAG_LIKE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}{m.group(3)}&gt;", text or "")


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


def _page_label(chunk: Any) -> str:
    try:
        start = int(_field(chunk, "page_number") or 0)
        end = int(_field(chunk, "page_end") or 0)
    except (TypeError, ValueError):
        return ""
    if start <= 0:
        return ""
    return f"{start}–{end}" if end > start else str(start)


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
        section = str(_field(chunk, "section_title") or "General") if is_record else "General"
        page = _page_label(chunk) if is_record else ""
        attrs = (
            f'index="{index}" source="{html.escape(source, quote=True)}" '
            f'section="{html.escape(section, quote=True)}"'
        )
        if page:
            attrs += f' page="{page}"'
        body = neutralize_tags(truncate_text(chunk_text(chunk), limit))
        pieces.append(f"<chunk {attrs}>\n{body}\n</chunk>")
    return "\n".join(pieces)

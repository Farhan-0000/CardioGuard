"""
Clinician summary export (Markdown / JSON).

Builds a structured, cited summary of one session from the decrypted session
data: patient-reported snapshot, BP log with ACC/AHA categories and the
numbered question/answer exchanges with their routing action and guideline
citations (source, section, page). It contains nothing beyond what the
session already holds — user messages were PII-redacted before storage — and
the session is referenced only by a short prefix of its id.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.config import Settings, get_settings
from medbridge.features.bp import bp_series
from medbridge.state import turn_store
from medbridge.state.session_manager import SessionErasedError, SessionNotFoundError
from medbridge.state.turn_store import HistoryRow

DISCLAIMER = (
    "Generated automatically from a {app} conversation. It contains patient-reported "
    "information and guideline-based educational responses only — it is not a diagnosis, a "
    "prescription or a medical record. Personal identifiers were redacted before storage. "
    "Verify all values clinically before acting on them."
)
_CITATION_KEYS = ("marker", "source", "title", "org", "year", "section", "page", "chunk_id")


def _citation(c: dict[str, Any]) -> dict[str, Any]:
    return {k: c.get(k) for k in _CITATION_KEYS if c.get(k) not in (None, "")}


def _exchanges(history: list[HistoryRow]) -> list[dict[str, Any]]:
    """Pair each user message with the latest assistant reply that follows it."""
    exchanges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in history:
        if row.role == "user":
            current = {
                "n": len(exchanges) + 1,
                "question": row.content,
                "asked_at": row.created_at.isoformat(),
                "answer": None,
                "action": None,
                "citations": [],
                "regenerated": False,
            }
            exchanges.append(current)
        elif row.role == "assistant" and current is not None:
            current["regenerated"] = current["answer"] is not None
            current["answer"] = row.content
            current["action"] = row.action
            current["citations"] = [_citation(c) for c in row.citations if isinstance(c, dict)]
    return exchanges


async def build_clinician_summary(
    db: AsyncSession, session_id: uuid.UUID, settings: Settings | None = None
) -> dict[str, Any]:
    """Assemble the summary as a JSON-serialisable dict.

    Raises:
        SessionNotFoundError / SessionErasedError.
    """
    cfg = settings or get_settings()
    state = await turn_store.load_session_state(db, session_id)
    if not state.exists:
        raise SessionNotFoundError(session_id)
    if state.erased:
        raise SessionErasedError(session_id)
    history = await turn_store.get_history(db, session_id)
    events = await turn_store.list_events(db, session_id, ["BP_READING"])
    exchanges = _exchanges(history)
    actions: dict[str, int] = {}
    for ex in exchanges:
        if ex["action"]:
            actions[ex["action"]] = actions.get(ex["action"], 0) + 1
    return {
        "app": cfg.APP_NAME,
        "session_ref": str(session_id)[:8],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER.format(app=cfg.APP_NAME),
        "snapshot": state.snapshot,
        "bp_readings": bp_series(events),
        "exchanges": exchanges,
        "action_counts": actions,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _cell(value: Any) -> str:
    text = "—" if value in (None, "") else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _format_med(med: dict[str, Any]) -> str:
    parts = [str(med.get("drug", "")).strip(), str(med.get("dosage", "")).strip(),
             str(med.get("frequency", "")).strip()]
    text = " ".join(p for p in parts if p)
    if med.get("reason"):
        text += f" (reason: {med['reason']})"
    return text


def _format_lab(lab: dict[str, Any]) -> str:
    name = lab.get("lab_type") or lab.get("lab") or lab.get("test") or lab.get("name") or "lab"
    value = " ".join(str(lab.get(k)) for k in ("value", "unit") if lab.get(k) not in (None, ""))
    return f"{name} {value}".strip()


def _snapshot_lines(snapshot: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    demo = snapshot.get("demographics") or {}
    if isinstance(demo, dict) and demo:
        lines.append("- **Demographics:** " + ", ".join(f"{k} {v}" for k, v in demo.items()))
    sections = (
        ("Current medications", snapshot.get("current_medications"), _format_med),
        ("Discontinued medications", snapshot.get("discontinued_medications"), _format_med),
        ("Lab results", snapshot.get("lab_results"), _format_lab),
    )
    for title, items, fmt in sections:
        if isinstance(items, list) and items:
            lines.append(f"- **{title}:** " + "; ".join(fmt(i) for i in items if isinstance(i, dict)))
    symptoms = snapshot.get("symptoms")
    if isinstance(symptoms, list) and symptoms:
        lines.append("- **Symptoms:** " + ", ".join(str(s) for s in symptoms))
    return lines or ["- No patient context was reported in this session."]


def _citation_line(c: dict[str, Any]) -> str:
    source = c.get("title") or c.get("source") or "Guideline"
    extras = [str(c[k]) for k in ("org", "year") if c.get(k) and str(c[k]) not in str(source)]
    text = f"{c.get('marker', '')} {source}".strip()
    if extras:
        text += f" ({', '.join(extras)})"
    if c.get("section"):
        text += f" — {c['section']}"
    if c.get("page") is not None:
        text += f", p. {c['page']}"
    return text


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a summary dict (from :func:`build_clinician_summary`) as Markdown."""
    lines = [
        f"# {summary['app']} — Clinician Summary",
        "",
        f"**Session reference:** `{summary['session_ref']}` · "
        f"**Generated:** {summary['generated_at']} · "
        f"**Exchanges:** {len(summary['exchanges'])} · "
        f"**BP readings:** {len(summary['bp_readings'])}",
        "",
        f"> **Disclaimer:** {summary['disclaimer']}",
        "",
        "## Patient-reported context",
        "",
        *_snapshot_lines(summary["snapshot"]),
        "",
        "## Blood-pressure log (ACC/AHA categories)",
        "",
    ]
    if summary["bp_readings"]:
        lines += [
            "| # | Recorded at | Reading (mmHg) | Pulse | Category | Source |",
            "|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(summary["bp_readings"], start=1):
            lines.append(
                f"| {i} | {_cell(r['recorded_at'])} | {r['systolic']}/{r['diastolic']} | "
                f"{_cell(r['pulse'])} | {_cell(r['category_label'])} | {_cell(r['source'])} |"
            )
    else:
        lines.append("No blood-pressure readings were recorded.")
    lines += ["", "## Questions and answers", ""]
    if not summary["exchanges"]:
        lines.append("No questions were asked in this session.")
    for ex in summary["exchanges"]:
        badge = f"`{ex['action']}`" if ex["action"] else "`NO REPLY`"
        lines += [f"### {ex['n']}. {badge} {_one_line(ex['question'])}", ""]
        lines += [str(ex["answer"]) if ex["answer"] else "_No answer was stored._", ""]
        if ex["regenerated"]:
            lines += ["_(Answer regenerated by the user; latest version shown.)_", ""]
        if ex["citations"]:
            lines.append("**Sources:**")
            lines += [f"- {_citation_line(c)}" for c in ex["citations"]]
            lines.append("")
    lines += ["---", f"*Generated by {summary['app']} on {summary['generated_at']}.*", ""]
    return "\n".join(lines)

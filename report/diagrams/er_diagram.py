"""
Entity-relationship diagram of the MedBridge PostgreSQL schema.

Columns, types, keys and constraints are read live from the SQLAlchemy metadata
in ``medbridge/db/models.py`` (the same models Alembic migrates to; head =
``h2b8d4f6a1c3``). Encrypted columns are declared below and cross-checked
against the ``enc:v1`` / "Encrypted envelope" markers in ``models.py`` so the
figure fails to build if the two drift apart. Append-only tables are the two
protected by the ``no_update_*`` / ``no_delete_*`` RULEs of migration
``6a69a1c21f43_initial_schema``.

Run: ``python report/diagrams/er_diagram.py`` → ``report/figures/er_diagram.{png,svg}``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FONT, PALETTE, REPO_ROOT, SUBTEXT, esc, render_dot

sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy import types as satypes
from sqlalchemy.dialects import postgresql

from medbridge.db.models import Base

NAME = "er_diagram"
MODELS = REPO_ROOT / "medbridge" / "db" / "models.py"

APPEND_ONLY = {"clinical_events", "audit_logs"}
# Fully encrypted with the per-session data key (Fernet): text "enc:v1:…", JSONB {"enc":"v1","ct":…}
ENCRYPTED = {
    ("clinical_events", "payload"),
    ("context_snapshots", "snapshot"),
    ("audit_logs", "request_message"),
    ("audit_logs", "gate_1_rationale"),
    ("audit_logs", "gate_2_rationale"),
    ("audit_logs", "response_text"),
    ("message_history", "content"),
    ("feedback", "comment"),
}
PARTIAL = {("audit_logs", "trace")}  # PHI-free JSONB; only its "private" member is encrypted
KEK_WRAPPED = {("session_keys", "wrapped_key")}  # the data key itself, wrapped by the master key

TABLE_ORDER = [
    "audit_logs",
    "clinical_events",
    "sessions",
    "context_snapshots",
    "session_keys",
    "message_history",
    "feedback",
]
NOTES = {
    "sessions": "never deleted; erased_at marks a crypto-shredded session",
    "session_keys": "deleting this row = crypto-shred (right to erasure)",
    "clinical_events": "typed delta events; snapshot = fold(events)",
    "context_snapshots": "projection cache + loop-breaker counter",
    "audit_logs": "one row per turn; trace = timings, actions, scores",
    "message_history": "user + assistant turns; citations = guideline metadata",
    "feedback": "one rating per message (upsert)",
}


def _markers_from_source() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(fully encrypted, partially encrypted) columns according to the comments in models.py."""
    full: set[tuple[str, str]] = set()
    partial: set[tuple[str, str]] = set()
    table = None
    prev = ""
    for line in MODELS.read_text(encoding="utf-8").splitlines():
        m = re.search(r'__tablename__\s*=\s*"(\w+)"', line)
        if m:
            table = m.group(1)
        col = re.match(r"\s+(\w+): Mapped\[", line)
        if col and table:
            ctx = line + " " + prev
            if "encrypted envelope" in ctx.lower() and "phi parts nested" in ctx.lower():
                partial.add((table, col.group(1)))
            elif "enc:v1" in ctx or "Encrypted envelope" in ctx:
                full.add((table, col.group(1)))
        prev = line if line.strip().startswith("#") else ""
    return full, partial


def _type_name(col) -> str:
    t = col.type
    if isinstance(t, satypes.Enum):
        return t.name or "enum"
    if isinstance(t, postgresql.ARRAY):
        return f"{_type_name_simple(t.item_type)}[]"
    return _type_name_simple(t)


def _type_name_simple(t) -> str:
    if isinstance(t, satypes.DateTime):
        return "timestamptz" if t.timezone else "timestamp"
    if isinstance(t, postgresql.JSONB):
        return "jsonb"
    if isinstance(t, satypes.Uuid | postgresql.UUID):
        return "uuid"
    if isinstance(t, satypes.SmallInteger):
        return "smallint"
    if isinstance(t, satypes.Integer):
        return "int"
    if (
        isinstance(t, satypes.String)
        and getattr(t, "length", None)
        and not isinstance(t, satypes.Text)
    ):
        return f"varchar({t.length})"
    if isinstance(t, satypes.Text | satypes.String):
        return "text"
    return str(t).lower()


def _badge(text: str, kind: str, port: str = "") -> str:
    k = PALETTE[kind]
    return (
        f'<TD {port} BGCOLOR="{k.fill}" BORDER="1" COLOR="{k.line}" CELLPADDING="1">'
        f'<FONT POINT-SIZE="7" COLOR="{k.text}"><B>{esc(text)}</B></FONT></TD>'
    )


def table_node(table) -> str:
    st = PALETTE["storage"]
    append_only = table.name in APPEND_ONLY
    head_fill = "#4C3A82" if not append_only else "#7A2E2E"
    rows = [
        f'<TR><TD COLSPAN="4" BGCOLOR="{head_fill}" ALIGN="LEFT" CELLPADDING="4">'
        f'<FONT COLOR="white" POINT-SIZE="10"><B>{esc(table.name)}</B></FONT>'
        + ('<FONT COLOR="#FFE3E0" POINT-SIZE="7.5">   APPEND-ONLY</FONT>' if append_only else "")
        + "</TD></TR>"
    ]
    fks = {fk.parent.name: fk.column.table.name for fk in table.foreign_keys}
    for col in table.columns:
        key = (table.name, col.name)
        flags = []
        if col.primary_key:
            flags.append("PK")
        if col.name in fks:
            flags.append("FK")
        flag_txt = " ".join(flags)
        name_txt = f"<B>{esc(col.name)}</B>" if col.primary_key else esc(col.name)
        null_txt = "" if col.nullable is False or col.primary_key else " ?"
        port = f'PORT="{col.name}_r"'
        if key in ENCRYPTED:
            badge = _badge("ENC", "storage", port)
        elif key in PARTIAL:
            badge = _badge("ENC*", "storage", port)
        elif key in KEK_WRAPPED:
            badge = _badge("KEK", "det", port)
        else:
            badge = f"<TD {port}></TD>"
        rows.append(
            f'<TR><TD ALIGN="LEFT" PORT="{col.name}_l"><FONT POINT-SIZE="7.5" COLOR="{SUBTEXT}">'
            f"{esc(flag_txt) or ' '}</FONT></TD>"
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="8.5">{name_txt}</FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="7.5" COLOR="{SUBTEXT}">{esc(_type_name(col) + null_txt)}'
            f"</FONT></TD>{badge}</TR>"
        )
    extras = []
    for cons in sorted(table.constraints, key=lambda k: (type(k).__name__, str(k.name))):
        if isinstance(cons, UniqueConstraint):
            extras.append("UNIQUE (" + ", ".join(c.name for c in cons.columns) + ")")
        if isinstance(cons, CheckConstraint):
            extras.append("CHECK " + str(cons.sqltext))
    if append_only:
        extras.append("RULEs: UPDATE / DELETE → DO INSTEAD NOTHING")
    extras.append(NOTES.get(table.name, ""))
    for e in [e for e in extras if e]:
        rows.append(
            f'<TR><TD COLSPAN="4" ALIGN="LEFT"><FONT POINT-SIZE="7" COLOR="{SUBTEXT}"><I>'
            f"{esc(e)}</I></FONT></TD></TR>"
        )
    label = (
        '<<TABLE BORDER="1" COLOR="' + st.line + '" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2" '
        'BGCOLOR="white">' + "".join(rows) + "</TABLE>>"
    )
    return f"  {table.name} [shape=plain, label={label}];"


def legend() -> str:
    def row(badge: str, text: str) -> str:
        return f'<TR>{badge}<TD ALIGN="LEFT"><FONT POINT-SIZE="8">{esc(text)}</FONT></TD></TR>'

    rows = (
        '<TR><TD ALIGN="LEFT" COLSPAN="2"><FONT POINT-SIZE="9"><B>Legend</B></FONT></TD></TR>'
        + row(_badge("ENC", "storage"), "encrypted, per-session data key (Fernet)")
        + row(_badge("ENC*", "storage"), "PHI-free JSONB; its private part encrypted")
        + row(_badge("KEK", "det"), "data key wrapped by the master key (env)")
        + row('<TD BGCOLOR="#7A2E2E" WIDTH="22"> </TD>', "append-only (UPDATE/DELETE RULEs)")
        + row('<TD><FONT POINT-SIZE="8">?</FONT></TD>', "nullable column")
        + row(
            '<TD><FONT POINT-SIZE="8">&gt;|  o|</FONT></TD>', "many  ·  zero-or-one (crow's foot)"
        )
    )
    return (
        '  legend [shape=plain, label=<<TABLE BORDER="1" COLOR="#CBD2DC" CELLBORDER="0" CELLSPACING="3" '
        f'CELLPADDING="2">{rows}</TABLE>>];'
    )


LEFT = ["audit_logs", "clinical_events"]  # append-only, many per session
RIGHT = ["context_snapshots", "session_keys", "message_history", "feedback"]
# arrow shape at the child end (first shape is nearest the child table)
CHILD_END = {
    "audit_logs": "crow",
    "clinical_events": "crow",
    "message_history": "crow",
    "feedback": "crow",
    "context_snapshots": "teeodot",
    "session_keys": "teeodot",
}


def build_source() -> str:
    full, partial = _markers_from_source()
    if full != ENCRYPTED or partial != PARTIAL:
        raise SystemExit(
            "Encrypted-column markers in medbridge/db/models.py changed:\n"
            f"  source full={sorted(full)}\n  declared={sorted(ENCRYPTED)}\n"
            f"  source partial={sorted(partial)} declared={sorted(PARTIAL)}"
        )
    tables = {t.name: t for t in Base.metadata.sorted_tables}
    missing = set(TABLE_ORDER) ^ set(tables)
    if missing:
        raise SystemExit(f"Table set changed: {sorted(missing)}")

    s = [
        f"digraph {NAME} {{",
        f'  graph [rankdir=LR, fontname="{FONT}", nodesep=0.22, ranksep=0.55, splines=spline, '
        'pad="0.15", bgcolor="white", newrank=true];',
        f'  node [fontname="{FONT}"];',
        f'  edge [fontname="{FONT}", fontsize=8, color="#4A5568", penwidth=1.2, arrowsize=0.8];',
    ]
    for name in LEFT:
        s.append(table_node(tables[name]))
    s.append(table_node(tables["sessions"]))
    s.append(legend())
    for name in RIGHT:
        s.append(table_node(tables[name]))
    for child in LEFT:
        s.append(
            f"  {child}:session_id_r:e -> sessions:session_id_l:w [dir=both, "
            f'arrowtail="{CHILD_END[child]}", arrowhead="tee"];'
        )
    for child in RIGHT:
        s.append(
            f'  sessions:session_id_r:e -> {child}:session_id_l:w [dir=both, arrowtail="tee", '
            f'arrowhead="{CHILD_END[child]}"];'
        )
    s.append(
        '  feedback:message_id_r:e -> message_history:message_id_r:e [style=dashed, color="#8A94A6", '
        'dir=both, arrowtail="teeodot", arrowhead="tee", constraint=false, '
        'label=<<FONT POINT-SIZE="7.5" COLOR="#5A6472">logical ref.<BR/>(no FK)</FONT>>];'
    )
    s.append("  { rank=same; " + "; ".join(LEFT) + "; }")
    s.append("  { rank=same; " + "; ".join(RIGHT) + "; }")
    s.append("  { rank=same; sessions; legend; }")
    s.append("  sessions -> legend [style=invis];")
    s.append("}")
    return "\n".join(s)


def build() -> list[Path]:
    return render_dot(NAME, build_source())


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(REPO_ROOT))

"""
Sequence diagram — one patient turn through the streaming endpoint (ANSWER path).

Source of truth:

* ``medbridge/api/routes/messages.py`` — ``handle_stream_message`` →
  ``_precheck_stream`` (404 / 410) → ``stream_turn`` (orchestrator in its own task
  and DB session, ``on_event`` → queue → ``event: stage``; then ``meta``, ``token``
  × N, ``done``);
* ``medbridge/core/orchestrator.py`` — stage order in ``_run_turn`` / ``_run_stages``,
  speculative retrieval task created before the Gate 1 call
  (``_stage_context_gate``), ``_retrieve`` = ``hybrid_search(top_k=20)`` then
  ``rerank_chunks(top_k=5)``, template bypass, ``validate_citations``,
  one ``persist_turn`` call;
* ``medbridge/retrieval/hybrid_retriever.py`` — dense + BM25 prefetch fused by
  ``RrfQuery(k=60)`` inside Qdrant.

Run: ``python report/diagrams/sequence_turn.py`` → ``report/figures/sequence_turn.{png,svg}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EDGE, PALETTE, SUBTEXT, save_matplotlib
from _mpl import DASH, Canvas
from matplotlib.patches import FancyBboxPatch, Rectangle

NAME = "sequence_turn"

BLUE = PALETTE["llm"].line
AMBER = PALETTE["gate"].line
RED = PALETTE["emergency"].line
PURPLE = PALETTE["storage"].line
GREY = EDGE

# (key, header lines, kind, tag)
PARTICIPANTS = [
    ("ui", ["Browser SPA"], "actor", "UI"),
    ("api", ["FastAPI"], "det", "SSE route"),
    ("orch", ["Orchestrator"], "det", "pipeline"),
    ("priv", ["Privacy"], "det", "redactor"),
    ("emer", ["Emergency"], "emergency", "classifier"),
    ("store", ["Turn store"], "storage", "PostgreSQL"),
    ("ext", ["Context", "extractor"], "llm", "LLM"),
    ("g1", ["Gate 1", "Context Gate"], "gate", "LLM"),
    ("ret", ["Retriever"], "storage", "+ Qdrant"),
    ("rr", ["Reranker"], "det", "cross-encoder"),
    ("g2", ["Gate 2", "Evidence Gate"], "gate", "LLM"),
    ("gen", ["Generator"], "llm", "LLM"),
    ("cv", ["Citation"], "det", "validator"),
]

W, H = 11.3, 8.25
# UI, API and orchestrator get wider gaps (most labels start there), the rest 0.8 in.
XS = [0.55, 1.85, 2.85] + [3.65 + 0.8 * i for i in range(10)]
BAR = 0.085  # activation bar width
TOP = 0.62  # bottom of the header boxes


def build() -> list[Path]:
    c = Canvas(W, H)
    ax = c.ax
    x = {key: XS[i] for i, (key, *_rest) in enumerate(PARTICIPANTS)}

    for key, lines, kind, tag in PARTICIPANTS:
        c.header(x[key], 0.36, 0.76, 0.52, lines, kind=kind, tag=tag, size=7.0, tag_size=6.0)

    y_rows: dict[str, float] = {}
    step = 0.205

    def row(name: str, y: float) -> float:
        y_rows[name] = y
        return y

    def bar(key: str, y0: float, y1: float) -> None:
        kind = next(p[2] for p in PARTICIPANTS if p[0] == key)
        k = PALETTE[kind]
        ax.add_patch(
            Rectangle(
                (x[key] - BAR / 2, y0 - 0.03),
                BAR,
                y1 - y0 + 0.06,
                facecolor=k.fill,
                edgecolor=k.line,
                lw=0.8,
                zorder=2,
            )
        )

    def msg(
        a: str,
        b: str,
        y: float,
        text: str,
        *,
        color: str = GREY,
        ret: bool = False,
        text_color: str | None = None,
        bold: bool = False,
    ) -> None:
        xa, xb = x[a], x[b]
        sgn = 1 if xb > xa else -1
        start = xa + sgn * BAR / 2
        end = xb - sgn * BAR / 2
        c.arrow([(start, y), (end, y)], color=color, dashed=ret, lw=1.0)
        # labels start just right of the left-hand lifeline (tidy column, no bar overlap)
        c.label(
            min(xa, xb) + 0.12,
            y - 0.075,
            text,
            color=text_color or color,
            size=6.5,
            bold=bold,
            ha="left",
        )

    def self_msg(a: str, y: float, text: str, color: str = GREY) -> None:
        xa = x[a] + BAR / 2
        c.arrow(
            [(xa, y), (xa + 0.22, y), (xa + 0.22, y + 0.11), (xa, y + 0.11)], color=color, lw=1.0
        )
        c.label(xa + 0.28, y + 0.055, text, color=color, size=6.5, ha="left")

    def note(x0: float, y: float, text: str, width: float, color: str = SUBTEXT) -> None:
        n = text.count("\n") + 1
        h = 0.15 + 0.105 * (n - 1)
        ax.add_patch(
            FancyBboxPatch(
                (x0, y - 0.075),
                width,
                h,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=PALETTE["note"].fill,
                edgecolor=PALETTE["note"].line,
                lw=0.8,
                zorder=4,
            )
        )
        ax.text(
            x0 + 0.06,
            y - 0.075 + h / 2,
            text,
            ha="left",
            va="center",
            fontsize=6.3,
            color=color,
            fontstyle="italic",
            zorder=5,
            linespacing=1.25,
        )

    y = 0.86
    msg("ui", "api", row("post", y), "POST …/messages/stream  (Bearer session token)")
    y += step
    self_msg(
        "api", row("check", y), "session check → 404 unknown / 410 erased (before any LLM work)"
    )
    y += step + 0.06
    msg("api", "orch", row("run", y), "process_message(text, on_event) · own task + DB session")
    y += step
    msg("orch", "priv", row("priv", y), "redact(raw text)")
    y += step
    msg("priv", "orch", row("priv_r", y), "redacted text + counts", ret=True)
    y += step
    c.arrow(
        [(x["orch"] - BAR / 2, y), (x["ui"] + BAR / 2, y)], color="#718096", dashed=True, lw=0.9
    )
    c.label(
        x["ui"] + 0.12,
        y - 0.075,
        "event: stage (start / end of every stage)",
        color="#718096",
        size=6.3,
        italic=True,
        ha="left",
    )
    row("stage", y)
    y += step
    msg("orch", "emer", row("emer", y), "classify_detailed(RAW text)", color=RED)
    y += step
    msg("emer", "orch", row("emer_r", y), "no red flag (rule ids only)", color=RED, ret=True)
    note(
        x["emer"] + 0.14,
        y,
        "alt [red flag]: ESCALATE template → persist → done · no LLM call, "
        "no wait for the session lock",
        3.95,
        color=RED,
    )
    y += step
    msg(
        "orch",
        "store",
        row("load", y),
        "per-session turn lock · load_session_state + recent turns",
        color=PURPLE,
    )
    y += step
    msg(
        "store",
        "orch",
        row("load_r", y),
        "snapshot, soft_ask_count, history (decrypted)",
        color=PURPLE,
        ret=True,
    )
    y += step
    self_msg("orch", row("small", y), "small-talk check (rules): no")
    y += step + 0.06
    msg("orch", "ext", row("ext", y), "extract(redacted text, snapshot, last turns)", color=BLUE)
    y += step
    msg(
        "ext",
        "orch",
        row("ext_r", y),
        "delta events + search query (JSON schema)",
        color=BLUE,
        ret=True,
    )
    y += step
    self_msg("orch", row("proj", y), "StateProjector.apply_events → snapshot (in memory)")
    y += step + 0.13

    # ---- par: Gate 1 ‖ speculative retrieval ---------------------------------
    par_top = y - 0.16
    msg(
        "orch",
        "g1",
        row("g1", y),
        "evaluate(snapshot, effective question) · T 0, seed 42",
        color=BLUE,
    )
    y += step
    msg(
        "g1",
        "orch",
        row("g1_r", y),
        "PROCEED → soft_ask_count = 0",
        color=AMBER,
        ret=True,
        bold=True,
    )
    y += step
    note(
        x["orch"] + 0.15,
        y,
        "alt [SOFT-ASK, count < 2]: retrieval task cancelled, generator asks for "
        "the missing facts, Gate 2 skipped\n     [SOFT-ASK, count ≥ 2]: loop-breaker → forced "
        "GENERALIZE, count reset, Gate 2 skipped",
        5.35,
        color=AMBER,
    )
    y += step + 0.11
    sep = y - 0.06
    y += 0.1
    msg(
        "orch",
        "ret",
        row("ret", y),
        "hybrid_search(query, top_k = 20) · speculative task",
        color=PURPLE,
    )
    y += step
    msg(
        "ret",
        "orch",
        row("ret_r", y),
        "20 candidates (dense + BM25 prefetch → RRF k = 60 in Qdrant)",
        color=PURPLE,
        ret=True,
    )
    y += step
    msg("orch", "rr", row("rr", y), "rerank_chunks(top_k = 5) · worker thread")
    y += step
    msg("rr", "orch", row("rr_r", y), "top-5 evidence chunks", ret=True)
    par_bot = y + 0.1
    x_par0, x_par1 = x["orch"] - 0.55, x["rr"] + 0.45
    ax.add_patch(
        Rectangle(
            (x_par0, par_top),
            x_par1 - x_par0,
            par_bot - par_top,
            facecolor="none",
            edgecolor="#4A5568",
            lw=0.9,
            zorder=1.5,
        )
    )
    ax.plot([x_par0, x_par1], [sep, sep], color="#4A5568", lw=0.8, linestyle=DASH, zorder=1.5)
    ax.add_patch(
        Rectangle(
            (x_par0, par_top),
            0.32,
            0.14,
            facecolor="#E2E8F0",
            edgecolor="#4A5568",
            lw=0.9,
            zorder=1.6,
        )
    )
    ax.text(
        x_par0 + 0.16,
        par_top + 0.07,
        "par",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        zorder=1.7,
    )
    ax.text(
        x_par0 + 0.05,
        par_top + 0.25,
        "[Gate 1]",
        ha="left",
        va="center",
        fontsize=6.2,
        fontstyle="italic",
        color=SUBTEXT,
        zorder=1.7,
    )
    ax.text(
        x_par0 + 0.05,
        sep + 0.1,
        "[retrieval,",
        ha="left",
        va="center",
        fontsize=6.2,
        fontstyle="italic",
        color=SUBTEXT,
        zorder=1.7,
    )
    ax.text(
        x_par0 + 0.05,
        sep + 0.22,
        "speculative]",
        ha="left",
        va="center",
        fontsize=6.2,
        fontstyle="italic",
        color=SUBTEXT,
        zorder=1.7,
    )
    y = par_bot + 0.17

    msg(
        "orch",
        "g2",
        row("g2", y),
        "evaluate(top-5, snapshot, query, message) · T 0, seed 42",
        color=BLUE,
    )
    y += step
    msg("g2", "orch", row("g2_r", y), "ANSWER", color=AMBER, ret=True, bold=True)
    y += step
    note(
        x["orch"] + 0.15,
        y,
        "alt [ABSTAIN / ESCALATE]: pre-vetted template, generator not called  ·  "
        "[GENERALIZE]: generator in general-information mode",
        5.4,
        color=AMBER,
    )
    y += step
    msg(
        "orch",
        "gen",
        row("gen", y),
        "generate(ANSWER, top-5, XML-isolated input) · T 0.3",
        color=BLUE,
    )
    y += step
    msg(
        "gen",
        "orch",
        row("gen_r", y),
        "draft with [n] markers  (optional Gate 3 NLI check, off by default)",
        color=BLUE,
        ret=True,
    )
    y += step
    msg("orch", "cv", row("cv", y), "validate_citations(draft, chunks)")
    y += step
    msg(
        "cv",
        "orch",
        row("cv_r", y),
        "valid citations (strip invalid; 0 valid → GENERALIZE)",
        ret=True,
    )
    y += step
    msg(
        "orch",
        "store",
        row("persist", y),
        "persist_turn: events, snapshot, 2 messages, audit row (encrypted, one flush)",
        color=PURPLE,
    )
    y += step
    msg("store", "orch", row("persist_r", y), "message ids", color=PURPLE, ret=True)
    y += step
    msg("orch", "api", row("resp", y), "MessageResponse", ret=True)
    y += step
    self_msg("api", row("commit", y), "commit DB session")
    y += step + 0.04
    msg("api", "ui", row("meta", y), "event: meta (action, id)", ret=True)
    y += step
    msg("api", "ui", row("tok", y), "event: token × N, done", ret=True)
    end = y + 0.12
    for key, *_rest in PARTICIPANTS:
        ax.plot([x[key], x[key]], [TOP, end], color="#A0AEC0", lw=0.8, linestyle=DASH, zorder=1)

    # activation bars
    bar("api", y_rows["post"], end - 0.05)
    bar("orch", y_rows["run"], y_rows["resp"])
    for key, a, b in [
        ("priv", "priv", "priv_r"),
        ("emer", "emer", "emer_r"),
        ("store", "load", "load_r"),
        ("ext", "ext", "ext_r"),
        ("g1", "g1", "g1_r"),
        ("ret", "ret", "ret_r"),
        ("rr", "rr", "rr_r"),
        ("g2", "g2", "g2_r"),
        ("gen", "gen", "gen_r"),
        ("cv", "cv", "cv_r"),
        ("store", "persist", "persist_r"),
    ]:
        bar(key, y_rows[a], y_rows[b])

    # legend: arrow semantics
    ly = end + 0.3
    c.resize(ly + 0.2)
    items = [
        (BLUE, False, "LLM call (Ollama, local)"),
        (AMBER, True, "gate decision"),
        (RED, False, "emergency check"),
        (PURPLE, False, "storage (PostgreSQL / Qdrant)"),
        (GREY, False, "deterministic call"),
        (GREY, True, "return / SSE event"),
    ]
    lx = 0.3
    for color, dashed, text in items:
        c.arrow([(lx, ly), (lx + 0.38, ly)], color=color, dashed=dashed, lw=1.0)
        ax.text(lx + 0.46, ly, text, ha="left", va="center", fontsize=6.8, color="#1F2933")
        lx += 0.46 + len(text) * 0.052 + 0.32
    c.check()
    return save_matplotlib(c.fig, NAME)


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(Path(__file__).resolve().parents[2]))

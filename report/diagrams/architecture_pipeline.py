"""
Architecture diagram — the MedBridge Two-Stage Answerability & Retrieval pipeline.

Mirrors ``medbridge/core/orchestrator.py`` (``PipelineOrchestrator._run_turn`` and
``_run_stages``) and the stage names in ``medbridge/core/pipeline_types.py``:

  privacy → emergency → load_state → small_talk → extraction (+ projection)
  → context_gate (+ speculative retrieval, loop-breaker) → retrieval / rerank
  → evidence_gate → generation (template bypass) → verification (Gate 3, on by default)
  → citation validation → persistence (one ``turn_store.persist_turn`` flush) → SSE.

Facts encoded here and where they come from:

* emergency classifier reads the raw text, every other stage the redacted text
  (``_stage_privacy`` / ``_stage_emergency``);
* loop-breaker: ``soft_ask_count < SOFT_ASK_MAX_COUNT (2)`` → SOFT-ASK and +1,
  otherwise forced GENERALIZE, counter reset, Evidence Gate skipped but retrieval
  still runs (``_stage_context_gate``, ``_stage_retrieval``, ``_stage_evidence_gate``);
* speculative retrieval is cancelled/discarded on SOFT-ASK (``_discard_speculative``);
* RRF k = 60 inside Qdrant, top-20 → cross-encoder top-5 (``hybrid_retriever.RRF_K``,
  ``RETRIEVAL_TOP_K_CANDIDATES`` / ``RETRIEVAL_TOP_K_RERANKED``);
* ABSTAIN / ESCALATE never reach the generator (``_stage_generation``);
* Gate 3 runs before citation validation (``_run_stages`` order) and is on by
  default since commit 7a3bc67 (``GROUNDING_VERIFIER_ENABLED=true``); since 16e594b a
  draft whose every claim is unsupported is withdrawn too (``_gate3_exceeds``);
* persistence = one ``persist_turn`` call (events, snapshot, 2 messages, audit row).

Run: ``python report/diagrams/architecture_pipeline.py`` →
``report/figures/architecture_pipeline.{png,svg}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EDGE, PALETTE, save_matplotlib
from _mpl import Canvas

NAME = "architecture_pipeline"

RED = PALETTE["emergency"].line
BLUE = PALETTE["llm"].line
AMBER = PALETTE["gate"].line
GREEN = PALETTE["safe"].line
PURPLE = PALETTE["storage"].line

W, H = 7.25, 10.75
XL, XC, XR = 1.15, 3.6, 6.075  # lane centres
WL, WC, WR = 1.9, 2.3, 1.55  # lane box widths
BUS_L, BUS_G, BUS_R = 0.07, 6.98, 7.12  # far-left SOFT-ASK bus, green and red buses


def build() -> list[Path]:
    c = Canvas(W, H)
    lab = c.label

    # ------------------------------------------------------------------ rows
    ui = c.box(
        XC,
        0.40,
        WC,
        0.46,
        "Patient message · browser SPA",
        ["POST /api/sessions/{id}/messages/stream", "session token checked; erased → 410"],
        kind="actor",
    )
    s0 = c.box(
        XC,
        1.10,
        WC,
        0.46,
        "0 · Privacy redaction",
        ["deterministic PII/PHI rules, no LLM", "redacted text feeds every later stage"],
    )
    s1 = c.box(
        XC,
        1.80,
        WC,
        0.46,
        "1 · Emergency fast-path",
        ["deterministic red-flag rules on RAW text", "runs before any LLM call"],
        kind="emergency",
    )
    esc = c.box(
        XR,
        1.80,
        WR,
        0.46,
        "ESCALATE template",
        ["instant escalation, no LLM", "not queued behind other turns"],
        kind="emergency",
    )
    s2 = c.box(
        XC,
        2.50,
        WC,
        0.46,
        "2 · Load session state",
        ["snapshot, soft_ask_count, recent turns", "effective question after a SOFT-ASK"],
    )
    pg_r = c.cylinder(XR, 2.50, WR, 0.50, "PostgreSQL", ["session state (encrypted)"])
    st = c.box(XC, 3.16, WC, 0.40, "2b · Small-talk check", ["greetings, thanks (rules, no LLM)"])
    stt = c.box(
        XR, 3.16, WR, 0.40, "Small-talk template", ["GENERALIZE, no clinical claims"], kind="safe"
    )
    s3 = c.box(
        XC,
        3.84,
        WC,
        0.46,
        "3 · Context extraction",
        ["LLM → typed delta events", "+ clinical search query"],
        kind="llm",
    )
    proj = c.box(
        XL - 0.05,
        3.84,
        1.6,
        0.46,
        "State projection",
        ["snapshot = fold(events)", "in memory; LLM never writes it"],
    )
    g1 = c.box(
        XC,
        4.62,
        1.9,
        0.58,
        "4 · Context Gate (Gate 1)",
        [
            "pre-retrieval context sufficiency",
            "LLM · temperature 0 · seed 42",
            "PROCEED  |  SOFT-ASK",
        ],
        kind="gate",
    )
    lb = c.diamond(XL, 5.66, 1.5, 0.64, "Loop-breaker", ["count ≥ 2 ?"])

    c.group(2.40, 5.12, 6.80, 6.05, "5 · Hybrid retrieval & reranking", title_ha="right")
    enc = c.box(
        3.14,
        5.70,
        1.32,
        0.54,
        "Query encoders",
        ["dense bge-small-en-v1.5", "+ sparse BM25 (parallel)"],
        title_size=8.0,
        sub_size=6.6,
    )
    qd = c.cylinder(
        4.62,
        5.70,
        1.32,
        0.58,
        "Qdrant",
        ["dense + BM25 prefetch", "RRF (k = 60) → top-20"],
        title_size=8.0,
        sub_size=6.6,
    )
    rr = c.box(
        6.10,
        5.70,
        1.32,
        0.54,
        "Cross-encoder rerank",
        ["ms-marco-MiniLM-L-6-v2", "top-20 → top-5"],
        title_size=8.0,
        sub_size=6.6,
    )

    sa = c.box(
        XL,
        6.80,
        WL,
        0.62,
        "SOFT-ASK reply",
        ["generator asks for missing facts", "soft_ask_count + 1", "speculative result discarded"],
        kind="llm",
    )
    g2 = c.box(
        XC,
        6.80,
        2.4,
        0.72,
        "6 · Evidence Gate (Gate 2)",
        [
            "post-retrieval evidence sufficiency",
            "LLM · temperature 0 · seed 42",
            "ANSWER | GENERALIZE | ABSTAIN | ESCALATE",
            "optional V1: calibrated signal + 3 votes",
        ],
        kind="gate",
        sub_size=6.8,
    )
    tb = c.box(
        XR,
        6.80,
        WR,
        0.50,
        "Template bypass",
        ["pre-vetted ABSTAIN / ESCALATE", "text; generator not invoked"],
        kind="safe",
    )
    s7 = c.box(
        XC,
        7.62,
        WC,
        0.46,
        "7 · Grounded generation",
        ["XML-isolated patient text + evidence", "inline [n] citations · temperature 0.3"],
        kind="llm",
    )
    g3 = c.box(
        XR,
        7.68,
        WR,
        0.80,
        "Gate 3 · grounding verifier",
        [
            "on by default (V2) · NLI, no LLM",
            "drops unsupported sentences;",
            "> τ or all claims unsupported",
            "→ GENERALIZE / ABSTAIN",
        ],
        kind="gate",
        optional=False,
        title_size=8.0,
        sub_size=6.4,
    )
    c.box(
        XL,
        8.10,
        WL,
        0.90,
        "Never-throw fallbacks",
        [
            "extractor fails → message, no events",
            "Gate 1 fails → PROCEED",
            "Gate 2 fails → GENERALIZE",
            "generator fails → pre-vetted template",
            "retrieval fails → Gate 2 decides",
        ],
        kind="note",
        title_size=7.5,
        sub_size=6.5,
    )
    cv = c.box(
        XC,
        8.34,
        WC,
        0.46,
        "Citation validation",
        ["strips [n] with no matching chunk", "ANSWER with 0 valid → GENERALIZE"],
    )
    s8 = c.box(
        XC,
        9.06,
        WC,
        0.50,
        "8 · Persistence (single flush)",
        ["events + snapshot + 2 messages + audit row", "encrypted with the session data key"],
        kind="storage",
        sub_size=6.8,
    )
    out = c.box(
        XC,
        9.82,
        1.9,
        0.46,
        "Safe clinical response",
        ["SSE to UI: stage → meta → token → done", "answer, citations, decision trace"],
        kind="actor",
        sub_size=6.5,
    )
    pg_w = c.cylinder(
        XR,
        9.82,
        WR,
        0.56,
        "PostgreSQL 16",
        ["append-only events + audit", "UPDATE / DELETE → no-op"],
        sub_size=6.6,
    )

    # ------------------------------------------------------------ spine flow
    def down(a, b, text: str = "", color: str = EDGE, x: float = XC, dashed: bool = False) -> None:
        c.arrow([(x, a.bottom), (x, b.top)], color=color, dashed=dashed)
        if text:
            lab(x + 0.06, (a.bottom + b.top) / 2, text, color=color, ha="left", size=6.6)

    down(ui, s0, "raw text")
    down(s0, s1, "raw text (emergency only) + redacted text")
    down(s1, s2, "no red flag")
    down(s2, st)
    down(st, s3, "clinical message")
    down(s3, g1, "intent + effective question")
    # extraction → projection → Gate 1
    c.arrow([(s3.left, s3.cy), (proj.right, proj.cy)], color=BLUE)
    lab((s3.left + proj.right) / 2, s3.cy - 0.10, "events", color=BLUE, size=6.3)
    c.arrow([(0.85, proj.bottom), (0.85, 4.50), (g1.left, 4.50)])
    lab(1.62, 4.42, "projected snapshot", size=6.3)
    # speculative retrieval (dashed), launched with Gate 1
    c.arrow([(4.68, s3.bottom), (4.68, 5.0), (3.72, 5.0), (3.72, enc.top)], color=BLUE, dashed=True)
    lab(
        4.78,
        4.62,
        "speculative retrieval starts with\nGate 1; discarded on SOFT-ASK",
        color=BLUE,
        ha="left",
        size=6.4,
        italic=True,
    )
    # Gate 1 PROCEED
    c.arrow([(3.30, g1.bottom), (3.30, enc.top)], color=AMBER)
    lab(3.24, 5.02, "PROCEED · count → 0", color=AMBER, ha="right", size=6.5)
    # Gate 1 SOFT-ASK → loop-breaker
    c.arrow([(g1.left, 4.75), (XL, 4.75), (XL, lb.top)], color=AMBER)
    lab(1.95, 4.85, "SOFT-ASK", color=AMBER, size=6.5, bold=True)
    c.arrow([(lb.right, lb.cy), (enc.left, lb.cy)], color=AMBER)
    lab((lb.right + enc.left) / 2 + 0.02, lb.cy - 0.10, "yes", color=AMBER, size=6.5, bold=True)
    c.arrow([(XL, lb.bottom), (XL, sa.top)], color=AMBER)
    lab(XL + 0.06, (lb.bottom + sa.top) / 2, "no", color=AMBER, size=6.5, bold=True, ha="left")
    # retrieval internals
    c.arrow([(enc.right, enc.cy), (qd.left, enc.cy)])
    c.arrow([(qd.right, rr.cy), (rr.left, rr.cy)])
    # top-5 → Gate 2, and the loop-breaker bypass of Gate 2
    y_ev = 6.24
    c.arrow([(rr.cx, rr.bottom), (rr.cx, y_ev), (XC, y_ev), (XC, g2.top)])
    c.junction(XC, y_ev)
    c.arrow([(XC, y_ev), (2.27, y_ev), (2.27, s7.cy), (s7.left, s7.cy)], color=AMBER, dashed=True)
    lab(2.27, 6.93, "loop-breaker: skip Gate 2", color=AMBER, size=6.2, rotation=90)
    lab(5.05, y_ev - 0.08, "top-5 evidence", size=6.3)
    # Gate 2 outcomes
    c.arrow([(g2.right, g2.cy), (tb.left, tb.cy)], color=AMBER)
    lab(
        (g2.right + tb.left) / 2,
        g2.cy - 0.19,
        "ABSTAIN /\nESCALATE",
        color=AMBER,
        size=6.2,
        bg=False,
    )
    down(g2, s7, "ANSWER / GENERALIZE", color=AMBER)
    # generation → (Gate 3) → citation validation
    c.arrow([(s7.right, s7.cy), (g3.left, s7.cy)], color=AMBER)
    lab((s7.right + g3.left) / 2, s7.cy - 0.09, "draft", color=AMBER, size=6.3)
    c.arrow([(g3.cx, g3.bottom), (g3.cx, cv.cy), (cv.right, cv.cy)], color=AMBER)
    lab(5.35, cv.cy - 0.09, "verified draft", color=AMBER, size=6.3)
    down(s7, cv, "draft (only if Gate 3 disabled)", color=BLUE)
    down(cv, s8, "final text + valid citations")
    down(s8, out, "MessageResponse")
    c.arrow([(4.66, s8.bottom), (4.66, pg_w.cy), (pg_w.left, pg_w.cy)], color=PURPLE, lw=1.6)
    lab(4.98, pg_w.cy - 0.09, "1 flush", color=PURPLE, size=6.3)

    # --------------------------------------------------------- side branches
    c.arrow([(s1.right, s1.cy), (esc.left, esc.cy)], color=RED, lw=1.5)
    lab((s1.right + esc.left) / 2, s1.cy - 0.09, "red flag", color=RED, size=6.4, bold=True)
    c.arrow(
        [(esc.right, esc.cy), (BUS_R, esc.cy), (BUS_R, 9.20), (s8.right, 9.20)], color=RED, lw=1.5
    )
    lab(BUS_R, 4.15, "instant ESCALATE", color=RED, size=6.3, rotation=90)
    c.arrow([(pg_r.left, pg_r.cy), (s2.right, s2.cy)], color=PURPLE, dashed=True)
    lab((pg_r.left + s2.right) / 2, s2.cy - 0.09, "read", color=PURPLE, size=6.3)
    c.arrow([(st.right, st.cy), (stt.left, stt.cy)], color=GREEN)
    lab((st.right + stt.left) / 2, st.cy - 0.09, "small talk", color=GREEN, size=6.3)
    c.arrow([(stt.right, stt.cy), (BUS_G, stt.cy), (BUS_G, 8.94), (s8.right, 8.94)], color=GREEN)
    c.arrow([(tb.right, tb.cy), (BUS_G, tb.cy)], color=GREEN, head=False)
    c.junction(BUS_G, tb.cy, GREEN)
    c.arrow([(sa.left, sa.cy), (BUS_L, sa.cy), (BUS_L, s8.cy), (s8.left, s8.cy)], color=BLUE)

    c.legend(
        10.30,
        [
            ("gate", "Answerability gate"),
            ("emergency", "Emergency / ESCALATE"),
            ("llm", "Local LLM call (Ollama)"),
            ("det", "Deterministic code"),
            ("safe", "Pre-vetted template"),
            ("storage", "Storage"),
            ("actor", "Patient / UI"),
            ("note", "Note"),
        ],
        dashed_note="Dashed border or arrow = optional or speculative path; "
        "dotted outline = stage group.",
        per_row=4,
        x0=0.25,
    )
    c.check()
    paths = save_matplotlib(c.fig, NAME)
    return paths


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(Path(__file__).resolve().parents[2]))

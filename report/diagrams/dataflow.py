"""
Data-flow diagram (DFD level 1) of MedBridge.

Processes, stores and flows follow the code:

* P1 browser SPA ``medbridge/web`` (served by FastAPI at ``/`` and ``/dashboard``);
* P2 FastAPI routes ``medbridge/api/routes`` (sessions, messages + SSE, features:
  BP log / context / summary / feedback, admin login + audit, metrics) with the
  ``medbridge/privacy`` middleware and token dependencies;
* P3 ``PipelineOrchestrator`` (``medbridge/core/orchestrator.py``);
* P4 Ollama at ``OLLAMA_BASE_URL`` (default ``http://127.0.0.1:11434``);
* D1 Qdrant collection ``clinical_guidelines`` (read-only at run time; the
  ingestion pipeline is its only writer);
* D2 PostgreSQL (``medbridge/db/models.py``), encrypted per session;
* P5 ``python -m medbridge.ingestion`` and P6 ``scripts/download_corpus.py``
  working from ``data/corpus_manifest.json`` (D3).

Corpus counts are read from ``reports/corpus_stats.json`` when that file exists.

Run: ``python report/diagrams/dataflow.py`` → ``report/figures/dataflow.{png,svg}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EDGE, PALETTE, REPO_ROOT, save_matplotlib
from _mpl import Canvas

NAME = "dataflow"
BLUE = PALETTE["llm"].line
PURPLE = PALETTE["storage"].line
GREY = EDGE


def corpus_counts() -> tuple[str, str]:
    """('27 documents', '2,747 chunks') from reports/corpus_stats.json, or generic text."""
    path = REPO_ROOT / "reports" / "corpus_stats.json"
    try:
        totals = json.loads(path.read_text(encoding="utf-8"))["totals"]
        return (
            f"{totals['documents']} curated documents",
            f"{totals['points_in_collection']:,} chunks",
        )
    except (OSError, KeyError, ValueError):
        return "curated documents", "chunks"


def build() -> list[Path]:
    docs, chunks = corpus_counts()
    c = Canvas(9.5, 7.05)
    lab = c.label

    def entity(cx: float, cy: float, w: float, h: float, title: str, lines: list[str]):
        return c.box(
            cx,
            cy,
            w,
            h,
            title,
            lines,
            kind="actor",
            lw=1.9,
            rounding=0.0,
            title_size=8.2,
            sub_size=6.6,
        )

    def proc(
        cx: float, cy: float, w: float, h: float, title: str, lines: list[str], kind: str = "det"
    ):
        return c.box(
            cx, cy, w, h, title, lines, kind=kind, title_size=8.2, sub_size=6.6, rounding=0.12
        )

    # groups
    c.group(0.1, 0.18, 9.4, 4.95, "Run time — one patient turn", kind="det", title_ha="right")
    c.group(
        0.1,
        5.35,
        9.4,
        6.62,
        "Offline — knowledge-base build (once, or when the manifest changes)",
        kind="det",
        title_ha="left",
    )

    # run time
    e1 = entity(0.78, 0.95, 1.16, 0.62, "Patient", ["types questions,", "reads answers"])
    p1 = proc(
        3.0,
        0.95,
        1.85,
        0.72,
        "P1 · Chat UI (SPA)",
        ["browser, served by FastAPI", "HTML + CSS + vanilla JS"],
        kind="actor",
    )
    p2 = proc(
        5.8,
        0.95,
        2.05,
        0.72,
        "P2 · FastAPI API layer",
        ["session tokens, rate limits, CSP", "sessions, BP log, feedback, erasure"],
    )
    e2 = entity(8.6, 0.95, 1.2, 0.62, "Admin", ["dashboard", "(password login)"])
    p3 = proc(
        5.8,
        2.75,
        2.25,
        0.92,
        "P3 · Guarded pipeline",
        [
            "redaction → emergency → extraction",
            "Gate 1 → retrieval → Gate 2",
            "generation (XML-isolated) → citations",
        ],
    )
    p4 = proc(
        2.75,
        2.75,
        1.9,
        0.72,
        "P4 · Local LLM (Ollama)",
        ["extractor, both gates, generator", "GPU · 127.0.0.1:11434"],
        kind="llm",
    )
    d2 = c.store(
        8.5,
        2.75,
        1.7,
        0.72,
        "D2",
        "PostgreSQL 16",
        ["encrypted messages, events,", "snapshots, audit trace"],
    )
    d1 = c.store(
        5.8,
        4.42,
        2.25,
        0.6,
        "D1",
        "Qdrant · clinical_guidelines",
        [f"{chunks}: dense 384-d + BM25"],
    )

    # offline
    e3 = entity(
        0.95,
        6.02,
        1.5,
        0.62,
        "Guideline publishers",
        ["AHA/ACC, ESC, NICE, WHO,", "KDIGO, ISH, NLM, NHLBI"],
    )
    p6 = proc(
        3.05,
        6.02,
        1.6,
        0.62,
        "P6 · Corpus download",
        ["scripts/download_corpus.py", "SHA-256 checked"],
    )
    d3 = c.store(5.15, 6.02, 1.75, 0.62, "D3", "Corpus files", ["data/guidelines + manifest", docs])
    p5 = proc(
        7.85,
        6.02,
        2.1,
        0.62,
        "P5 · Ingestion",
        ["parse PDF/HTML → chunk (400 tok)", "embed dense + BM25 → upsert"],
    )

    # ---- flows: patient ↔ UI ↔ API ↔ admin ---------------------------------
    c.arrow([(e1.right, 0.84), (p1.left, 0.84)])
    lab((e1.right + p1.left) / 2, 0.74, "question", size=6.5)
    c.arrow([(p1.left, 1.07), (e1.right, 1.07)])
    lab(
        (e1.right + p1.left) / 2,
        1.18,
        'answer, citations,\n"why this answer"',
        size=6.3,
        va="top",
        bg=False,
    )
    c.arrow([(p1.right, 0.84), (p2.left, 0.84)])
    lab((p1.right + p2.left) / 2, 0.73, "POST + token", size=6.5)
    c.arrow([(p2.left, 1.07), (p1.right, 1.07)])
    lab((p1.right + p2.left) / 2, 1.18, "SSE events", size=6.5, va="top", bg=False)
    c.arrow([(e2.left, 0.84), (p2.right, 0.84)])
    lab((p2.right + e2.left) / 2, 0.73, "login, queries", size=6.5)
    c.arrow([(p2.right, 1.07), (e2.left, 1.07)])
    lab(
        (p2.right + e2.left) / 2,
        1.18,
        "metrics, audit view\n(erased sessions hidden)",
        size=6.2,
        va="top",
        bg=False,
    )

    # ---- API ↔ pipeline, API ↔ PostgreSQL ------------------------------------
    c.arrow([(5.55, p2.bottom), (5.55, p3.top)])
    lab(5.48, 1.80, "raw message,\nsession id", size=6.4, ha="right", bg=False)
    c.arrow([(6.05, p3.top), (6.05, p2.bottom)])
    lab(6.12, 1.80, "response +\nstage events", size=6.4, ha="left", bg=False)
    c.arrow([(6.75, p2.bottom), (6.75, 1.95), (8.5, 1.95), (8.5, d2.top)], color=PURPLE, both=True)
    lab(
        7.65,
        2.02,
        "sessions, BP log, feedback;\nerasure = delete session key",
        size=6.2,
        va="top",
        bg=False,
        color=PURPLE,
    )

    # ---- pipeline ↔ LLM, pipeline ↔ PostgreSQL, pipeline ↔ Qdrant ---------------
    c.arrow([(p3.left, 2.62), (p4.right, 2.62)], color=BLUE)
    lab((p4.right + p3.left) / 2, 2.52, "redacted prompts", size=6.4, color=BLUE)
    c.arrow([(p4.right, 2.88), (p3.left, 2.88)], color=BLUE)
    lab((p4.right + p3.left) / 2, 2.98, "JSON outputs", size=6.4, color=BLUE, va="top", bg=False)
    c.arrow([(p3.right, 2.62), (d2.left, 2.62)], color=PURPLE)
    lab((p3.right + d2.left) / 2, 2.52, "encrypted turn", size=6.2, color=PURPLE)
    c.arrow([(d2.left, 2.88), (p3.right, 2.88)], color=PURPLE)
    lab(
        (p3.right + d2.left) / 2, 2.98, "state, history", size=6.2, color=PURPLE, va="top", bg=False
    )
    c.arrow([(5.55, p3.bottom), (5.55, d1.top)], color=PURPLE)
    lab(5.48, (p3.bottom + d1.top) / 2, "query vectors", size=6.4, ha="right", color=PURPLE)
    c.arrow([(6.05, d1.top), (6.05, p3.bottom)], color=PURPLE)
    lab(
        6.12, (p3.bottom + d1.top) / 2, "top-20 chunks + payload", size=6.4, ha="left", color=PURPLE
    )

    # ---- offline chain ---------------------------------------------------------------
    c.arrow([(e3.right, e3.cy), (p6.left, p6.cy)])
    lab((e3.right + p6.left) / 2, e3.cy - 0.12, "https", size=6.3)
    c.arrow([(p6.right, p6.cy), (d3.left, d3.cy)])
    lab((p6.right + d3.left) / 2, p6.cy - 0.12, "files", size=6.3)
    c.arrow([(d3.right, d3.cy), (p5.left, p5.cy)])
    lab((d3.right + p5.left) / 2, d3.cy - 0.12, "PDF / HTML", size=6.3)
    c.arrow([(p5.cx, p5.top), (p5.cx, d1.cy), (d1.right, d1.cy)], color=PURPLE)
    lab(
        p5.cx + 0.07,
        5.05,
        "chunks + vectors + payload (only writer)",
        size=6.3,
        ha="left",
        color=PURPLE,
    )

    # notes
    c.box(
        2.75,
        4.1,
        1.95,
        0.78,
        "Privacy boundary",
        [
            "every service on 127.0.0.1",
            "LOCAL_ONLY_MODE: no remote LLM",
            "only redacted text reaches P4",
            "PII-scrubbed logs",
        ],
        kind="note",
        title_size=7.4,
        sub_size=6.4,
    )
    c.legend(
        6.85,
        [
            ("actor", "External entity / UI"),
            ("det", "Process"),
            ("llm", "LLM process"),
            ("storage", "Data store (D)"),
        ],
        per_row=4,
        x0=0.25,
        col_w=2.3,
    )
    c.check()
    return save_matplotlib(c.fig, NAME)


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(REPO_ROOT))

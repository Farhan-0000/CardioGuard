"""
Deployment diagram — everything runs on one laptop.

Facts and their sources:

* demo target Windows 11 + NVIDIA RTX 4050 Laptop GPU (6 GB), development on
  macOS / Apple Silicon (CLAUDE_AGENT_PROMPT §2);
* ``uvicorn medbridge.main:app --host 127.0.0.1 --port 8000`` (README; ``HOST`` /
  ``PORT`` defaults in ``medbridge/config.py``), serving ``/api``, SSE and the SPA
  (``medbridge.web.mount_web``: ``/``, ``/dashboard``, ``/static``); lifespan
  starts the background warm-up and the hourly retention purge (``main.py``);
* in-process models: fastembed ``BAAI/bge-small-en-v1.5`` + ``Qdrant/bm25`` (ONNX
  on CPU, ``retrieval/embedder.py``), cross-encoder reranker and the optional NLI
  verifier on cuda > mps > cpu (``retrieval/reranker.py``, ``ai/grounding_verifier.py``);
* Ollama at ``OLLAMA_BASE_URL = http://127.0.0.1:11434``, default
  ``LLM_MODEL = qwen3:8b``, ``LLM_KEEP_ALIVE = 30m`` (``config.py``); judge model
  ``gemma3:12b`` (``evaluation/judges/llm_judge.py``);
* Docker: ``postgres:16-alpine`` on ``127.0.0.1:5433→5432`` and ``qdrant/qdrant`` on
  ``127.0.0.1:6333`` / ``6334`` with named volumes (``docker-compose.yml``);
* legacy Streamlit UI on port 8501 (``.streamlit/config.toml``) — that file sets no
  ``server.address``, so the figure tells the reader to pass
  ``--server.address 127.0.0.1``.

Run: ``python report/diagrams/deployment.py`` → ``report/figures/deployment.{png,svg}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PALETTE, REPO_ROOT, SUBTEXT, save_matplotlib
from _mpl import Canvas
from matplotlib.patches import FancyBboxPatch

NAME = "deployment"
BLUE = PALETTE["llm"].line
PURPLE = PALETTE["storage"].line


def build() -> list[Path]:
    c = Canvas(9.0, 6.55)
    ax = c.ax
    lab = c.label

    # machine boundary
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 0.1),
            8.8,
            6.3,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor="#FBFCFD",
            edgecolor="#2D3748",
            lw=1.8,
            zorder=0.5,
        )
    )
    ax.text(
        0.28,
        0.24,
        "One laptop runs the whole system",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#1A202C",
    )
    ax.text(
        0.28,
        0.46,
        "demo: Windows 11 + NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)   ·   "
        "development: macOS on Apple Silicon (M-series)   ·   API, Ollama, PostgreSQL, Qdrant on 127.0.0.1",
        ha="left",
        va="top",
        fontsize=7.0,
        color=SUBTEXT,
    )

    # clients
    br = c.box(
        1.65,
        1.08,
        2.4,
        0.52,
        "Web browser (Chrome / Edge)",
        ["http://127.0.0.1:8000  ·  chat SPA, /dashboard"],
        kind="actor",
        sub_size=6.6,
    )
    sl = c.box(
        7.3,
        1.08,
        2.8,
        0.52,
        "Streamlit UI (legacy, optional) · :8501",
        ["start with --server.address 127.0.0.1"],
        kind="actor",
        optional=True,
        sub_size=6.6,
    )

    # uvicorn process
    c.group(
        0.4,
        1.62,
        5.0,
        4.5,
        "Python 3.11 · uvicorn medbridge.main:app · 127.0.0.1:8000",
        kind="det",
        title_size=7.8,
    )
    c.box(
        1.6,
        2.32,
        2.2,
        0.6,
        "FastAPI app",
        ["/api routes + SSE streaming", "SPA + dashboard (static files)"],
        sub_size=6.6,
    )
    c.box(
        3.85,
        2.32,
        2.1,
        0.6,
        "Security layer",
        ["session / admin tokens, rate limits", "CSP headers, PII-scrubbed logs"],
        kind="safe",
        sub_size=6.5,
    )
    c.box(
        1.6,
        3.17,
        2.2,
        0.6,
        "Pipeline orchestrator",
        ["emergency rules, both gates,", "templates, encrypted turn store"],
        sub_size=6.6,
    )
    c.box(
        3.85,
        3.4,
        2.1,
        1.05,
        "In-process models",
        [
            "bge-small-en-v1.5 + BM25",
            "(fastembed, ONNX on CPU)",
            "MiniLM-L-6 cross-encoder",
            "optional NLI verifier (Gate 3)",
            "torch device: CUDA > MPS > CPU",
        ],
        sub_size=6.5,
    )
    c.box(
        1.6,
        3.98,
        2.2,
        0.5,
        "Background tasks",
        ["model warm-up at start-up", "hourly retention purge (crypto-shred)"],
        kind="det",
        title_size=7.8,
        sub_size=6.4,
    )

    # Ollama + Docker (right column)
    oll = c.box(
        7.3,
        2.32,
        2.8,
        0.86,
        "Ollama server · 127.0.0.1:11434",
        [
            "GPU: CUDA (RTX 4050) or Metal (Apple Silicon)",
            "LLM_MODEL default qwen3:8b, keep_alive 30m",
            "study judge model: gemma3:12b",
        ],
        kind="llm",
        sub_size=6.5,
    )
    c.group(
        5.85, 2.95, 8.75, 4.5, "Docker Desktop · docker-compose.yml", kind="storage", title_size=7.8
    )
    pg = c.box(
        7.3,
        3.5,
        2.7,
        0.46,
        "postgres:16-alpine",
        ["127.0.0.1:5433 → 5432 · volume pgdata"],
        kind="storage",
        sub_size=6.5,
    )
    qd = c.box(
        7.3,
        4.07,
        2.7,
        0.46,
        "qdrant/qdrant",
        ["127.0.0.1:6333 REST, 6334 gRPC · volume qdrant_data"],
        kind="storage",
        sub_size=6.3,
    )

    # command-line tools
    c.group(0.4, 4.8, 5.0, 6.2, "Command-line tools (same machine)", kind="det", title_size=7.8)
    c.box(
        1.6,
        5.6,
        2.2,
        0.66,
        "python -m evaluation.run",
        [
            "runs the orchestrator in-process",
            "(same Ollama, Qdrant, PostgreSQL)",
            "+ judge, calibrate, analyze",
        ],
        sub_size=6.4,
        title_size=7.8,
    )
    ing = c.box(
        3.85,
        5.6,
        2.1,
        0.66,
        "python -m medbridge.ingestion",
        ["offline: PDF / HTML → chunks", "→ embeddings → Qdrant upsert"],
        sub_size=6.4,
        title_size=7.6,
    )
    c.box(
        7.3,
        5.52,
        2.8,
        1.18,
        ".env (environment only, never in the DB)",
        [
            "ENCRYPTION_MASTER_KEY (KEK),",
            "SESSION_TOKEN_SECRET, ADMIN_PASSWORD,",
            "POSTGRES_PASSWORD",
            "LOCAL_ONLY_MODE=true: no remote LLM",
            "local: data/guidelines,",
            "Hugging Face + Ollama model caches",
        ],
        kind="note",
        title_size=7.6,
        sub_size=6.3,
    )

    # links
    c.arrow([(br.cx, br.bottom), (br.cx, 1.62)], both=True)
    lab(br.cx + 0.07, 1.47, "HTTP + SSE", ha="left", size=6.5)
    c.arrow([(sl.left, sl.cy), (4.55, sl.cy), (4.55, 1.62)], dashed=True)
    lab(5.25, sl.cy - 0.09, "HTTP → /api", size=6.4)
    c.arrow([(5.0, oll.cy), (oll.left, oll.cy)], color=BLUE, both=True)
    lab(5.42, oll.cy - 0.1, "/api/chat", size=6.4, color=BLUE)
    c.arrow([(5.0, pg.cy), (pg.left - 0.05, pg.cy)], color=PURPLE, both=True)
    lab(5.42, pg.cy - 0.1, "asyncpg", size=6.4, color=PURPLE)
    c.arrow([(5.0, 3.98), (qd.left - 0.05, 3.98)], color=PURPLE)
    lab(5.42, 3.88, "read-only", size=6.3, color=PURPLE)
    c.arrow(
        [(ing.right, ing.cy), (5.42, ing.cy), (5.42, 4.17), (qd.left - 0.05, 4.17)], color=PURPLE
    )
    lab(5.49, 4.66, "upsert (only writer)", size=6.3, color=PURPLE, ha="left", bg=False)

    c.check()
    return save_matplotlib(c.fig, NAME)


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(REPO_ROOT))

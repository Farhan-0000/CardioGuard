"""
Retrieval evaluation (CLAUDE_AGENT_PROMPT §4.3): Recall@5, Recall@20, MRR and
nDCG@5 for dense-only, sparse-only, hybrid (RRF k=60) and hybrid + cross-encoder
rerank, on ``evaluation/datasets/retrieval_eval_v1.json``.

Dataset format (one labelled query per entry)::

    {"name": "MedBridge-RetrievalEval", "version": "1.0",
     "queries": [{"id": "R001", "query": "...", "topic": "bp_targets",
                  "relevant": [{"guideline_id": "ESC_2024", "pages": [32, 33]}]}]}

Relevance: a retrieved chunk is relevant to a gold span when the guideline id
matches (case-insensitive) and the chunk's page range ``[page_number,
page_end]`` overlaps the span's pages (a span without pages = whole document).

* Recall@k — fraction of gold spans matched by at least one of the top-k chunks.
* MRR — 1 / rank of the first relevant chunk in the top 20 (0 if none).
* nDCG@5 — binary gains; each gold span is credited at most once (so DCG ≤
  IDCG), IDCG = Σ_{i=1}^{min(5, |spans|)} 1/log2(i+1).

Hybrid + rerank reorders the hybrid top-20 with the cross-encoder, so its
Recall@20 equals hybrid's by construction (reported for completeness).
"""

from __future__ import annotations

import argparse
import asyncio
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from evaluation import metrics as mx
from evaluation import plots
from evaluation.evidence import rerank_chunks
from evaluation.io import read_json, write_json, write_table
from evaluation.manifest import utc_now
from evaluation.paths import FIGURES_DIR, RETRIEVAL_DATASET, RETRIEVAL_RESULTS_DIR

MODES: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid+rerank")
METRICS: tuple[str, ...] = ("recall@5", "recall@20", "mrr", "ndcg@5")
METRIC_LABELS: dict[str, str] = {
    "recall@5": "Recall@5",
    "recall@20": "Recall@20",
    "mrr": "MRR",
    "ndcg@5": "nDCG@5",
}
CANDIDATES = 20


class RelevantSpan(BaseModel):
    """A gold passage: a guideline and (optionally) its pages."""

    guideline_id: str
    pages: list[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _page_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "pages" not in data:
            start = data.get("page_start", data.get("page"))
            if start is not None:
                end = data.get("page_end")
                last = int(end) if end is not None else int(start)
                data = {**data, "pages": list(range(int(start), last + 1))}
        return data


class RetrievalQuery(BaseModel):
    """One labelled query."""

    id: str
    query: str = Field(..., min_length=1)
    relevant: list[RelevantSpan] = Field(..., min_length=1)
    topic: str = ""

    @model_validator(mode="before")
    @classmethod
    def _relevant_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "relevant" not in data:
            for alias in ("gold", "gold_evidence", "relevant_spans"):
                if alias in data:
                    return {**data, "relevant": data[alias]}
        return data


def load_queries(path: Path = RETRIEVAL_DATASET) -> list[RetrievalQuery]:
    """Load the labelled queries (a list, or an object with ``queries``/``items``)."""
    data = read_json(path)
    entries = data if isinstance(data, list) else data.get("queries") or data.get("items") or []
    return [RetrievalQuery.model_validate(e) for e in entries]


def chunk_matches(chunk: Any, span: RelevantSpan) -> bool:
    """Guideline match + page overlap (see module docstring)."""
    if str(getattr(chunk, "guideline_id", "")).lower() != span.guideline_id.lower():
        return False
    if not span.pages:
        return True
    low = int(getattr(chunk, "page_number", 0) or 0)
    high = int(getattr(chunk, "page_end", None) or low)
    return any(low <= p <= high for p in span.pages)


def query_metrics(ranked: Sequence[Any], spans: Sequence[RelevantSpan]) -> dict[str, float]:
    """Recall@5, Recall@20, MRR, nDCG@5 for one ranked list."""

    def recall(k: int) -> float:
        hit = sum(1 for s in spans if any(chunk_matches(c, s) for c in ranked[:k]))
        return hit / len(spans)

    mrr = 0.0
    for rank, chunk in enumerate(ranked[:CANDIDATES], start=1):
        if any(chunk_matches(chunk, s) for s in spans):
            mrr = 1.0 / rank
            break
    credited: set[int] = set()
    dcg = 0.0
    for rank, chunk in enumerate(ranked[:5], start=1):
        for i, span in enumerate(spans):
            if i not in credited and chunk_matches(chunk, span):
                credited.add(i)
                dcg += 1.0 / math.log2(rank + 1)
                break
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(5, len(spans)) + 1))
    return {"recall@5": recall(5), "recall@20": recall(20), "mrr": mrr, "ndcg@5": dcg / idcg}


async def ranked_lists(retriever: Any, reranker: Any, query: str) -> dict[str, list[Any]]:
    """Top-20 chunks of every retrieval mode for one query."""
    search = getattr(retriever, "search", None)
    if search is None:
        raise RuntimeError(
            "HybridRetriever.search(query, top_k, mode) is required for the retrieval evaluation"
        )
    out: dict[str, list[Any]] = {}
    for mode in ("dense", "sparse", "hybrid"):
        out[mode] = list(await search(query, top_k=CANDIDATES, mode=mode))
    out["hybrid+rerank"] = [
        c for c, _ in await rerank_chunks(reranker, query, out["hybrid"], CANDIDATES)
    ]
    return out


async def evaluate_retrieval(
    queries: Sequence[RetrievalQuery],
    retriever: Any,
    reranker: Any,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Per-query and aggregate metrics (mean with bootstrap 95% CI) per mode."""
    per_query: list[dict[str, Any]] = []
    for q in queries:
        lists = await ranked_lists(retriever, reranker, q.query)
        row: dict[str, Any] = {"id": q.id, "topic": q.topic}
        for mode in MODES:
            row[mode] = query_metrics(lists[mode], q.relevant)
            row[f"{mode}_top5"] = [str(getattr(c, "chunk_id", "")) for c in lists[mode][:5]]
        per_query.append(row)
    aggregate: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        aggregate[mode] = {
            metric: mx.bootstrap_ci(
                [r[mode][metric] for r in per_query], n_boot=n_boot, seed=seed
            ).as_dict()
            for metric in METRICS
        }
        aggregate[mode]["n"] = len(per_query)
    return {"modes": aggregate, "per_query": per_query}


def write_outputs(result: dict[str, Any], results_dir: Path, figures_dir: Path) -> list[Path]:
    """metrics.json, markdown/CSV table and the grouped bar figure."""
    paths = [results_dir / "metrics.json"]
    write_json(paths[0], result)
    rows = []
    for mode, values in result["modes"].items():
        rows.append(
            {
                "Mode": mode,
                **{
                    m: f"{values[m]['estimate']:.3f} [{values[m]['low']:.3f}, {values[m]['high']:.3f}]"
                    for m in METRICS
                },
                "n": values["n"],
            }
        )
    paths += write_table(
        rows,
        results_dir,
        "retrieval_metrics",
        "Retrieval metrics (mean, 95% bootstrap CI over queries)",
    )
    fig, ax = plots.new_figure(width=6.4, height=3.6)
    series = {mode: [result["modes"][mode][m]["estimate"] for m in METRICS] for mode in MODES}
    errors = {
        mode: [(result["modes"][mode][m]["low"], result["modes"][mode][m]["high"]) for m in METRICS]
        for mode in MODES
    }
    plots.grouped_bars(ax, [METRIC_LABELS[m] for m in METRICS], series, errors, value_labels=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean over queries)")
    plots.set_title(ax, f"Retrieval quality by mode (n={len(result['per_query'])} queries)")
    paths += plots.save_figure(fig, figures_dir, "retrieval_metrics")
    return paths


async def _main_async(args: argparse.Namespace) -> int:
    from medbridge.retrieval.hybrid_retriever import HybridRetriever
    from medbridge.retrieval.reranker import get_reranker

    queries = load_queries(args.dataset)
    retriever = HybridRetriever()
    try:
        result = await evaluate_retrieval(
            queries, retriever, get_reranker(), args.n_boot, args.seed
        )
    finally:
        await retriever.close()
    result.update(dataset=str(args.dataset), generated_at=utc_now(), candidates=CANDIDATES)
    for path in write_outputs(result, args.results_dir, args.figures_dir):
        print(f"[retrieval_eval] wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval evaluation: dense vs sparse vs hybrid vs rerank"
    )
    parser.add_argument("--dataset", type=Path, default=RETRIEVAL_DATASET)
    parser.add_argument("--results-dir", type=Path, default=RETRIEVAL_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return asyncio.run(_main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

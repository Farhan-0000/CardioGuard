"""
Corpus statistics & composition figure (Module M-18, Ingestion Pipeline).

``python -m medbridge.ingestion --stats`` writes:

* ``reports/corpus_stats.json`` - per-document and total statistics,
* ``reports/corpus_stats.md``   - the same as report-ready markdown tables,
* ``reports/figures/corpus_composition.{png,svg}`` - chunks per document
  (coloured by document type) and topic coverage.

Counts come from the live Qdrant collection when it is reachable (so they
describe what retrieval actually searches) and from the manifest otherwise.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

from medbridge.ingestion.manifest import ALLOWED_TOPICS, CorpusManifest
from medbridge.ingestion.state import IngestState

logger = structlog.get_logger(__name__)

# Reference categorical palette (dataviz skill, light mode): slots 1-3 validate
# all-pairs for CVD separation; the table in corpus_stats.md is the relief view.
DOC_TYPE_COLOURS = {"guideline": "#2a78d6", "review": "#eb6834", "patient_info": "#1baf7a"}
DOC_TYPE_LABELS = {"guideline": "Guideline", "review": "Statement / review",
                   "patient_info": "Patient information"}
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"


def build_corpus_stats(
    manifest: CorpusManifest,
    *,
    qdrant_counts: dict[str, Counter[str]] | None = None,
    token_counts: list[int] | None = None,
    state: IngestState | None = None,
    collection: str = "",
) -> dict[str, Any]:
    """Assemble per-document and corpus-level statistics.

    Args:
        manifest: The curated corpus manifest.
        qdrant_counts: ``guideline_id -> Counter(content_type -> points)`` from Qdrant.
        token_counts: Token length of every indexed chunk (for the size distribution).
        state: Incremental-ingestion state (last run timing and configuration).
        collection: Qdrant collection the counts were read from.
    """
    documents: list[dict[str, Any]] = []
    for doc in manifest.documents:
        counts = (qdrant_counts or {}).get(doc.guideline_id)
        chunks = sum(counts.values()) if counts is not None else doc.chunk_count
        tables = counts.get("table", 0) if counts is not None else doc.table_chunk_count
        documents.append({
            "guideline_id": doc.guideline_id,
            "short_title": doc.short_title,
            "title": doc.title,
            "org": doc.org,
            "year": doc.year,
            "doc_type": doc.doc_type,
            "source_format": doc.source_format,
            "pages": doc.pages,
            "chunks": chunks,
            "table_chunks": tables,
            "topics": list(doc.topics),
            "licence": doc.licence,
        })

    topic_docs: Counter[str] = Counter()
    topic_chunks: Counter[str] = Counter()
    for d in documents:
        for topic in d["topics"]:
            topic_docs[topic] += 1
            topic_chunks[topic] += d["chunks"]

    by_type: dict[str, dict[str, int]] = {}
    for d in documents:
        bucket = by_type.setdefault(d["doc_type"], {"documents": 0, "chunks": 0, "pages": 0})
        bucket["documents"] += 1
        bucket["chunks"] += d["chunks"]
        bucket["pages"] += d["pages"]

    totals: dict[str, Any] = {
        "documents": len(documents),
        "pages_or_sections": sum(d["pages"] for d in documents),
        "chunks": sum(d["chunks"] for d in documents),
        "table_chunks": sum(d["table_chunks"] for d in documents),
        "organisations": len({d["org"] for d in documents}),
        "pdf_documents": sum(1 for d in documents if d["source_format"] == "pdf"),
        "html_documents": sum(1 for d in documents if d["source_format"] == "html"),
    }
    if qdrant_counts is not None:
        totals["points_in_collection"] = sum(sum(c.values()) for c in qdrant_counts.values())
    if token_counts:
        ordered = sorted(token_counts)
        totals["chunk_tokens"] = {
            "median": statistics.median(ordered),
            "mean": round(statistics.fmean(ordered), 1),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "max": ordered[-1],
            "min": ordered[0],
            "over_512": sum(1 for t in ordered if t > 512),
        }

    result: dict[str, Any] = {
        "collection": collection,
        "source": "qdrant" if qdrant_counts is not None else "manifest",
        "chunking": dict(manifest.chunking),
        "documents": documents,
        "by_doc_type": by_type,
        "topics": {
            t: {"documents": topic_docs.get(t, 0), "chunks": topic_chunks.get(t, 0)}
            for t in sorted(ALLOWED_TOPICS)
        },
        "totals": totals,
    }
    if state is not None and state.last_full_build.finished_at:
        result["last_full_build"] = state.last_full_build.model_dump()
    if state is not None and state.last_run.finished_at:
        result["last_ingestion"] = state.last_run.model_dump()
    return result


def render_markdown(stats: dict[str, Any]) -> str:
    """Report-ready markdown for :func:`build_corpus_stats` output."""
    totals = stats["totals"]
    lines = [
        "# MedBridge knowledge-base corpus statistics",
        "",
        "Generated by `python -m medbridge.ingestion --stats` "
        f"(counts from {stats['source']}"
        + (f", collection `{stats['collection']}`" if stats.get("collection") else "") + ").",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Documents | {totals['documents']} ({totals['pdf_documents']} PDF, "
        f"{totals['html_documents']} HTML) |",
        f"| Issuing organisations | {totals['organisations']} |",
        f"| Pages (PDF) / sections (HTML) | {totals['pages_or_sections']:,} |",
        f"| Chunks | {totals['chunks']:,} |",
        f"| of which table chunks | {totals['table_chunks']:,} |",
    ]
    if "points_in_collection" in totals:
        lines.append(f"| Points in Qdrant collection | {totals['points_in_collection']:,} |")
    tokens = totals.get("chunk_tokens")
    if tokens:
        lines.append(
            f"| Chunk length (model tokens) | median {tokens['median']}, p90 {tokens['p90']}, "
            f"max {tokens['max']}; {tokens['over_512']} chunks > 512 |"
        )
    chunking = stats.get("chunking") or {}
    if chunking:
        lines.append(
            f"| Chunking | {chunking.get('chunk_size')} tokens, overlap "
            f"{chunking.get('chunk_overlap')} ({chunking.get('tokenizer')} tokenizer) |"
        )
    for key, label in (("last_full_build", "Last full rebuild"),
                       ("last_ingestion", "Last ingestion run")):
        run = stats.get(key)
        if run:
            lines.append(
                f"| {label} | {run['mode']}: {run['documents_indexed']} documents indexed, "
                f"{run['documents_skipped']} skipped, {run['chunks_indexed']:,} chunks in "
                f"{run['seconds']:.0f} s ({run['finished_at']}) |"
            )
    lines += [
        "",
        "## Documents",
        "",
        "| # | ID | Document | Organisation | Year | Type | Format | Pages | Chunks | Tables |"
        " Topics |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, d in enumerate(stats["documents"], start=1):
        lines.append(
            f"| {i} | `{d['guideline_id']}` | {d['short_title']} | {d['org']} | {d['year']} | "
            f"{DOC_TYPE_LABELS.get(d['doc_type'], d['doc_type'])} | {d['source_format'].upper()} | "
            f"{d['pages']} | {d['chunks']} | {d['table_chunks']} | {', '.join(d['topics'])} |"
        )
    lines += ["", "## By document type", "", "| Type | Documents | Pages / sections | Chunks |",
              "|---|---|---|---|"]
    for doc_type, bucket in stats["by_doc_type"].items():
        lines.append(f"| {DOC_TYPE_LABELS.get(doc_type, doc_type)} | {bucket['documents']} | "
                     f"{bucket['pages']} | {bucket['chunks']} |")
    lines += ["", "## Topic coverage", "",
              "A document counts towards every topic it is tagged with in the manifest.", "",
              "| Topic | Documents | Chunks |", "|---|---|---|"]
    for topic, row in stats["topics"].items():
        lines.append(f"| {topic} | {row['documents']} | {row['chunks']:,} |")
    lines += ["", "HTML pages have no physical pages; for HTML sources the page count is the "
              "number of h1-h3 sections (virtual pages).", ""]
    return "\n".join(lines)


def render_figure(stats: dict[str, Any], png_path: Path, svg_path: Path) -> None:
    """Draw the corpus composition figure (PNG at 300 dpi + SVG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    docs = sorted(stats["documents"], key=lambda d: d["chunks"])
    topics = sorted(stats["topics"].items(), key=lambda kv: kv[1]["chunks"])

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                         "svg.fonttype": "none"})
    fig, (ax_docs, ax_topics) = plt.subplots(
        1, 2, figsize=(12, 8.2), gridspec_kw={"width_ratios": [1.45, 1]},
        facecolor=_SURFACE,
    )
    for ax in (ax_docs, ax_topics):
        ax.set_facecolor(_SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_TEXT_SECONDARY, length=0)
        ax.grid(axis="x", color=_GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)

    labels = [f"{d['short_title']} ({d['year']})" for d in docs]
    ax_docs.barh(
        range(len(docs)), [d["chunks"] for d in docs], height=0.62,
        color=[DOC_TYPE_COLOURS.get(d["doc_type"], "#888888") for d in docs],
        edgecolor=_SURFACE, linewidth=1.0,
    )
    ax_docs.set_yticks(range(len(docs)), labels, color=_TEXT_PRIMARY)
    ax_docs.set_xlabel("Chunks indexed", color=_TEXT_SECONDARY)
    ax_docs.set_title("Chunks per document", loc="left", color=_TEXT_PRIMARY,
                      fontsize=10, fontweight="bold")
    present = [t for t in DOC_TYPE_COLOURS if any(d["doc_type"] == t for d in docs)]
    ax_docs.legend(
        handles=[Patch(color=DOC_TYPE_COLOURS[t], label=DOC_TYPE_LABELS[t]) for t in present],
        loc="lower right", frameon=False, labelcolor=_TEXT_PRIMARY,
    )

    ax_topics.barh(range(len(topics)), [row["chunks"] for _, row in topics], height=0.62,
                   color=DOC_TYPE_COLOURS["guideline"], edgecolor=_SURFACE, linewidth=1.0)
    ax_topics.set_yticks(range(len(topics)), [t.replace("_", " ") for t, _ in topics],
                         color=_TEXT_PRIMARY)
    for i, (_, row) in enumerate(topics):
        ax_topics.annotate(f"{row['documents']} docs", (row["chunks"], i), xytext=(4, 0),
                           textcoords="offset points", va="center", color=_TEXT_SECONDARY)
    ax_topics.set_xlabel("Chunks from documents tagged with the topic", color=_TEXT_SECONDARY)
    ax_topics.set_title("Topic coverage", loc="left", color=_TEXT_PRIMARY,
                        fontsize=10, fontweight="bold")
    ax_topics.margins(x=0.18)

    totals = stats["totals"]
    fig.suptitle(
        f"MedBridge knowledge base: {totals['documents']} curated documents, "
        f"{totals['chunks']:,} chunks ({totals['table_chunks']:,} table chunks)",
        x=0.01, ha="left", color=_TEXT_PRIMARY, fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor=_SURFACE)
    fig.savefig(svg_path, facecolor=_SURFACE)
    plt.close(fig)


def write_reports(stats: dict[str, Any], reports_dir: Path) -> list[Path]:
    """Write JSON, markdown and figure outputs; returns the written paths."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "corpus_stats.json"
    md_path = reports_dir / "corpus_stats.md"
    png_path = reports_dir / "figures" / "corpus_composition.png"
    svg_path = reports_dir / "figures" / "corpus_composition.svg"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    md_path.write_text(render_markdown(stats), encoding="utf-8")
    render_figure(stats, png_path, svg_path)
    logger.info("Wrote corpus statistics", json=str(json_path), markdown=str(md_path),
                figure=str(png_path))
    return [json_path, md_path, png_path, svg_path]

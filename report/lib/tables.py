"""
Generators for ``[[table:KEY]]`` data tables.

Every generator reads the results context (``report/lib/results_context.py``) built
from real results / measurement files and returns a :class:`~content.Table`, or
``None`` when its data does not exist yet (the report then shows a highlighted
``[TABLE PENDING: KEY]`` marker). No number is ever typed in by hand here.

The study generators (``hallucination_by_condition``, ``unsafe_answer``,
``routing_by_condition``, ``mcnemar``, ``judge_agreement``, ...) follow the schema of
``evaluation/results/summary.json`` written by ``evaluation/analyze.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .content import ParseContext, Table, parse_file
from .results_context import ResultsContext

TableGenerator = Callable[[ResultsContext], Table | None]

#: Composite cells ("9.4 / 21.9") are bound with non-breaking spaces so they wrap as a unit.
NBSP_SLASH = "\u00a0/\u00a0"

#: Condition order and display names (mirrors evaluation/analyze.py).
CONDITION_ORDER: tuple[str, ...] = ("C0", "C1", "C1G", "C2", "C3", "C4", "C5", "V1", "V2", "V12")
CONDITION_NAMES: dict[str, str] = {
    "C0": "Direct LLM",
    "C1": "Vanilla RAG",
    "C1G": "RAG + production prompt (no gates)",
    "C2": "RAG + prompt safety",
    "C3": "Context Gate only",
    "C4": "Evidence Gate only",
    "C5": "MedBridge (dual gate)",
    "V1": "Calibrated Evidence Gate",
    "V2": "Triple gate (grounding verifier)",
    "V12": "V1 + V2",
}
CATEGORY_NAMES: dict[str, str] = {
    "answerable_complete": "Answerable, complete context",
    "answerable_missing_context": "Answerable, missing context",
    "general_topic": "General topic",
    "unanswerable_in_domain": "Unanswerable, in domain",
    "out_of_domain": "Out of domain",
    "emergency": "Emergency",
}
CATEGORY_ORDER = tuple(CATEGORY_NAMES)
ACTIONS = ("ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE")
RETRIEVAL_MODES: tuple[tuple[str, str], ...] = (
    ("dense", "Dense (bge-small-en-v1.5)"),
    ("sparse", "Sparse (BM25)"),
    ("hybrid", "Hybrid (RRF, k = 60)"),
    ("hybrid+rerank", "Hybrid + cross-encoder rerank"),
)
RETRIEVAL_METRICS: tuple[tuple[str, str], ...] = (
    ("recall@5", "Recall@5"),
    ("recall@20", "Recall@20"),
    ("mrr", "MRR"),
    ("ndcg@5", "nDCG@5"),
)
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Default captions (used for pending tables too, so numbering stays stable).
DEFAULT_CAPTIONS: dict[str, str] = {
    "corpus": "Documents in the curated knowledge base",
    "corpus_by_type": "Knowledge-base composition by document type",
    "corpus_topics": "Knowledge-base coverage by clinical topic",
    "retrieval_metrics": "Retrieval quality per retrieval configuration (mean with 95% bootstrap CI)",
    "reranker_ablation": "Reranker ablation on the retrieval evaluation set (hybrid top-20 to top-5)",
    "model_selection": "Model-selection experiment (condition C5, dev split)",
    "model_selection_round1": "Model-selection experiment, round 1 (condition C5, dev split)",
    "model_selection_schema": "Structured-output validity on the first attempt per LLM role (%)",
    "pii_metrics": "PII/PHI redaction quality on the development and blind held-out sets",
    "pii_per_category": "Blind held-out redaction recall per identifier category (frozen detector)",
    "perf_before_after": "End-to-end latency before and after optimisation (warm runs)",
    "perf_baseline": "Baseline end-to-end latency of the received prototype",
    "test_summary": "Automated test suites",
    "aq_benchmark": "MedBridge-AQ routing benchmark (live pipeline)",
    "aq_subsets": "MedBridge-AQ routing accuracy per subset (live pipeline)",
    "rabbits": "RABBITS brand-versus-generic robustness (live pipeline)",
    "adversarial": "Adversarial prompt-injection suite (live pipeline)",
    "load_test": "Load test: concurrent users against the local stack",
    "smoke_test": "End-to-end smoke test checks",
    "hallucination_by_condition": "Hallucination by condition (test split, unit: item-turn)",
    "unsafe_answer": "Unsafe answers and appropriate abstention on unanswerable items",
    "routing_by_condition": "Routing accuracy and emergency handling by condition",
    "mcnemar": "Paired exact McNemar tests with Holm correction",
    "cochran_q": "Cochran's Q test across conditions C0 to C5",
    "judge_agreement": "Agreement between the LLM judge and the NLI judge (Cohen's kappa)",
    "calibration": "Thresholds calibrated on the dev split",
    "hallubench_composition": "Composition of MedBridge-HalluBench v1",
    "hallubench_actions": "Gold actions per turn in MedBridge-HalluBench v1",
    "action_taken": "Action taken on previous review comments",
    "degraded_turns": "Degraded turns per condition (excluded from headline metrics)",
    "latency_by_condition": "Latency per condition",
    "category_breakdown": "Hallucination rate (LLM judge) by item category and condition",
}


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def num(value: Any, digits: int = 2) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def integer(value: Any) -> str:
    if not _finite(value):
        return "—"
    return f"{round(float(value)):,d}"


def pct(p: Any, ci: bool = True, digits: int = 1) -> str:
    """Proportion dict ``{rate, low, high}`` (or a bare fraction) as a percentage."""
    if isinstance(p, dict):
        rate = p.get("rate")
        if not _finite(rate):
            return "—"
        text = f"{100 * float(rate or 0):.{digits}f}"
        if ci and _finite(p.get("low")) and _finite(p.get("high")):
            text += f" [{100 * float(p['low']):.{digits}f}, {100 * float(p['high']):.{digits}f}]"
        return text
    if _finite(p):
        return f"{100 * float(p):.{digits}f}"
    return "—"


def perc(value: Any, digits: int = 1) -> str:
    """A value that is already a percentage (0-100)."""
    return "—" if not _finite(value) else f"{float(value):.{digits}f}"


def interval(d: Any, digits: int = 3) -> str:
    if not isinstance(d, dict):
        return num(d, digits)
    est = d.get("estimate", d.get("kappa"))
    if not _finite(est):
        return "—"
    low, high = d.get("low"), d.get("high")

    def b(v: Any) -> str:
        if isinstance(v, int | float) and math.isinf(v):
            return "∞"
        return num(v, digits)

    if low is None and high is None:
        return f"{est:.{digits}f}"
    return f"{est:.{digits}f} [{b(low)}, {b(high)}]"


def pvalue(p: Any) -> str:
    if not _finite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def seconds(ms: Any, digits: int = 1) -> str:
    return "—" if not _finite(ms) else f"{float(ms) / 1000:.{digits}f}"


def _table(
    key: str,
    header: list[str],
    rows: list[list[Any]],
    *,
    note: str | None = None,
    align: list[str] | None = None,
    font_size: float | None = None,
    widths: list[float] | None = None,
    caption: str | None = None,
) -> Table | None:
    if not rows:
        return None
    if align is None:
        align = ["l"] + ["c"] * (len(header) - 1)
    return Table(
        caption=caption or DEFAULT_CAPTIONS.get(key),
        header=header,
        rows=[[str(c).replace(" / ", NBSP_SLASH) for c in r] for r in rows],
        align=align,
        markup=False,
        note=note,
        key=key,
        font_size=font_size,
        col_widths=widths,
    )


def _src(ctx: ResultsContext, key: str) -> str:
    return f"Source: {ctx.source(key) or key}."


def _cond_label(cond: str, m: dict[str, Any] | None = None) -> str:
    name = (m or {}).get("name") or CONDITION_NAMES.get(cond, cond)
    return f"{cond} {name}"


def _study_conditions(ctx: ResultsContext) -> list[tuple[str, dict[str, Any]]]:
    per = ctx.get("study.per_condition") or {}
    if not isinstance(per, dict):
        return []
    order = [c for c in CONDITION_ORDER if c in per] + [c for c in per if c not in CONDITION_ORDER]
    return [(c, per[c]) for c in order if isinstance(per[c], dict)]


def _uncalibrated(ctx: ResultsContext, cond: str) -> bool:
    check = (ctx.get("study.calibration_check") or {}).get(cond)
    return isinstance(check, dict) and not check.get("calibrated", True)


# ----------------------------------------------------------------------
# Corpus and dataset
# ----------------------------------------------------------------------


def corpus(ctx: ResultsContext) -> Table | None:
    docs = ctx.get("corpus.documents")
    if not isinstance(docs, list) or not docs:
        return None
    rows = []
    for i, d in enumerate(docs, start=1):
        rows.append(
            [
                str(i),
                str(d.get("short_title") or d.get("title") or d.get("guideline_id")),
                str(d.get("org", "—")),
                str(d.get("year", "—")),
                str(d.get("doc_type", "—")).replace("_", " "),
                str(d.get("source_format", "—")).upper(),
                integer(d.get("pages")),
                integer(d.get("chunks")),
            ]
        )
    totals = ctx.get("corpus.totals") or {}
    rows.append(
        [
            "",
            "Total",
            f"{integer(totals.get('organisations'))} orgs",
            "",
            "",
            "",
            integer(totals.get("pages_or_sections")),
            integer(totals.get("chunks")),
        ]
    )
    return _table(
        "corpus",
        ["#", "Document", "Organisation", "Year", "Type", "Format", "Pages", "Chunks"],
        rows,
        align=["c", "l", "l", "c", "l", "c", "r", "r"],
        font_size=8,  # 27 one-line rows instead of two-line rows at 9 pt
        widths=[0.25, 2.5, 0.95, 0.4, 0.7, 0.45, 0.42, 0.5],
        note=_src(ctx, "corpus") + " Pages counts HTML sections for HTML sources.",
    )


def corpus_by_type(ctx: ResultsContext) -> Table | None:
    by = ctx.get("corpus.by_doc_type")
    if not isinstance(by, dict) or not by:
        return None
    rows = [
        [
            k.replace("_", " ").capitalize(),
            integer(v.get("documents")),
            integer(v.get("pages")),
            integer(v.get("chunks")),
        ]
        for k, v in by.items()
    ]
    t = ctx.get("corpus.totals") or {}
    rows.append(
        [
            "Total",
            integer(t.get("documents")),
            integer(t.get("pages_or_sections")),
            integer(t.get("chunks")),
        ]
    )
    return _table(
        "corpus_by_type",
        ["Document type", "Documents", "Pages / sections", "Chunks"],
        rows,
        note=_src(ctx, "corpus"),
    )


def corpus_topics(ctx: ResultsContext) -> Table | None:
    topics = ctx.get("corpus.topics")
    if not isinstance(topics, dict) or not topics:
        return None
    rows = [
        [k.replace("_", " ").capitalize(), integer(v.get("documents")), integer(v.get("chunks"))]
        for k, v in sorted(topics.items(), key=lambda kv: -int(kv[1].get("chunks") or 0))
    ]
    return _table(
        "corpus_topics",
        ["Topic", "Documents", "Chunks"],
        rows,
        note=_src(ctx, "corpus") + " A document can cover several topics.",
    )


def hallubench_composition(ctx: ResultsContext) -> Table | None:
    by = ctx.get("hallubench.by_category")
    if not isinstance(by, dict) or not by:
        return None
    cats = [c for c in CATEGORY_ORDER if c in by] + [c for c in by if c not in CATEGORY_ORDER]
    rows = []
    for c in cats:
        r = by[c]
        rows.append(
            [
                CATEGORY_NAMES.get(c, c),
                integer(r.get("items")),
                integer(r.get("turns")),
                integer(r.get("multi_turn")),
                integer(r.get("dev")),
                integer(r.get("test")),
            ]
        )
    hb = ctx.get("hallubench") or {}
    split = hb.get("by_split") or {}
    rows.append(
        [
            "Total",
            integer(hb.get("n_items")),
            integer(hb.get("n_turns")),
            integer(hb.get("n_multi_turn_items")),
            integer(split.get("dev")),
            integer(split.get("test")),
        ]
    )
    return _table(
        "hallubench_composition",
        ["Category", "Items", "Turns", "Multi-turn items", "Dev items", "Test items"],
        rows,
        note=_src(ctx, "hallubench"),
    )


def hallubench_actions(ctx: ResultsContext) -> Table | None:
    gold = ctx.get("hallubench.by_gold_action")
    if not isinstance(gold, dict) or not gold:
        return None
    total = sum(gold.values()) or 1
    rows = [
        [a, integer(gold.get(a, 0)), f"{100 * gold.get(a, 0) / total:.1f}"]
        for a in ACTIONS
        if a in gold
    ]
    rows.append(["Total", integer(total), "100.0"])
    return _table(
        "hallubench_actions",
        ["Gold action", "Turns", "Share %"],
        rows,
        note=_src(ctx, "hallubench"),
    )


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------


def _metric_cells(mode: dict[str, Any]) -> list[str]:
    return [interval(mode.get(k)) for k, _ in RETRIEVAL_METRICS]


def retrieval_metrics(ctx: ResultsContext) -> Table | None:
    modes = ctx.get("retrieval.modes")
    if not isinstance(modes, dict) or not modes:
        return None
    rows = []
    n = None
    for key, label in RETRIEVAL_MODES:
        if key in modes:
            rows.append([label, *_metric_cells(modes[key])])
            n = n or modes[key].get("n")
    cand = ctx.get("retrieval.candidates")
    note = _src(ctx, "retrieval")
    if n:
        note += f" n = {n} queries; {cand or 20} hybrid candidates reranked to the top 5."
    return _table(
        "retrieval_metrics",
        ["Configuration"] + [m for _, m in RETRIEVAL_METRICS],
        rows,
        note=note,
        widths=[1.7, 1.0, 1.0, 1.0, 1.0],
        font_size=9,
    )


def reranker_ablation(ctx: ResultsContext) -> Table | None:
    base = ctx.get("retrieval.modes") or {}
    abl = ctx.get("reranker") or {}
    if not base and not abl:
        return None
    rows = []
    if "hybrid" in base:
        rows.append(["None (hybrid RRF only)", *_metric_cells(base["hybrid"])])
    if "hybrid+rerank" in base:
        rows.append([f"{DEFAULT_RERANKER} (default)", *_metric_cells(base["hybrid+rerank"])])
    for dirname, data in sorted(abl.items()):
        mode = (data.get("modes") or {}).get("hybrid+rerank")
        if not mode:
            continue
        name = dirname.replace("_", "/", 1)
        if name == DEFAULT_RERANKER:
            continue
        rows.append([name, *_metric_cells(mode)])
    if len(rows) < 2:
        return None
    note = (
        f"Sources: {ctx.source('retrieval') or 'retrieval metrics'} (no reranker, default) and "
        f"{ctx.source('reranker') or 'reranker ablation'}. Mean with 95% bootstrap CI."
    )
    return _table(
        "reranker_ablation",
        ["Reranker"] + [m for _, m in RETRIEVAL_METRICS],
        rows,
        note=note,
        widths=[2.0, 0.95, 0.95, 0.95, 0.95],
        font_size=9,
    )


# ----------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------


def _model_selection(ctx: ResultsContext, key: str, table_key: str) -> Table | None:
    summary = ctx.get(key)
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), dict):
        return None
    models = summary.get("models") or list(summary["results"])
    rows = []
    for m in models:
        s = summary["results"].get(m) or {}
        if not s.get("n_valid"):
            continue
        routing = s.get("routing") or {}
        info = s.get("ollama") or {}
        roles = s.get("roles") or {}
        size = info.get("size_bytes")
        validity = [
            v.get("valid_first_attempt", {}).get("rate")
            for v in roles.values()
            if isinstance(v, dict)
        ]
        validity = [v for v in validity if _finite(v)]
        rows.append(
            [
                str(m),
                f"{info.get('parameter_size', '—')} {info.get('quantization_level', '')}".strip()
                + (f" ({float(size or 0) / 1e9:.1f} GB)" if _finite(size) else ""),
                integer(s.get("n_valid")),
                f"{pct(routing.get('accuracy_strict'), ci=False)} / "
                f"{pct(routing.get('accuracy_lenient'), ci=False)}",
                num(routing.get("macro_f1")),
                f"{100 * min(validity):.1f}" if validity else "—",
                f"{seconds(s.get('latency_turn_p50_ms'))} / {seconds(s.get('latency_turn_p95_ms'))}",
                num((roles.get("response_generator") or {}).get("tokens_per_s_median"), 1),
            ]
        )
    split = summary.get("split", "dev")
    caption = DEFAULT_CAPTIONS[table_key].replace("dev split", f"{split} split")
    return _table(
        table_key,
        [
            "Model",
            "Parameters, quantisation (size)",
            "Turns",
            "Routing strict / lenient %",
            "Macro-F1",
            "Min. JSON valid %",
            "Turn p50 / p95 s",
            "Generation tok/s",
        ],
        rows,
        caption=caption,
        font_size=9,
        widths=[0.95, 1.15, 0.45, 0.85, 0.55, 0.6, 0.75, 0.6],
        note=_src(ctx, key) + " Min. JSON valid = lowest first-attempt schema validity over the "
        "four LLM roles.",
    )


def model_selection(ctx: ResultsContext) -> Table | None:
    return _model_selection(ctx, "model_selection", "model_selection")


def model_selection_round1(ctx: ResultsContext) -> Table | None:
    return _model_selection(ctx, "model_selection_r1", "model_selection_round1")


def model_selection_schema(ctx: ResultsContext) -> Table | None:
    summary = ctx.get("model_selection")
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), dict):
        return None
    models = [
        m for m in summary.get("models") or [] if (summary["results"].get(m) or {}).get("n_valid")
    ]
    roles = sorted({r for m in models for r in (summary["results"][m].get("roles") or {})})
    if not models or not roles:
        return None
    rows = []
    for role in roles:
        rows.append(
            [role.replace("_", " ")]
            + [
                pct(
                    ((summary["results"][m].get("roles") or {}).get(role) or {}).get(
                        "valid_first_attempt"
                    ),
                    ci=False,
                )
                for m in models
            ]
        )
    return _table(
        "model_selection_schema", ["LLM role", *models], rows, note=_src(ctx, "model_selection")
    )


# ----------------------------------------------------------------------
# Privacy
# ----------------------------------------------------------------------


def _prf(d: dict[str, Any]) -> str:
    return " / ".join(num(d.get(k), 3) for k in ("precision", "recall", "f1"))


def _pii_row(label: str, res: dict[str, Any]) -> list[str]:
    s, o = res.get("overall_strict") or {}, res.get("overall_overlap") or {}
    neg = res.get("negatives_altered")
    prot = res.get("protected_values_corrupted")
    return [
        label,
        f"{integer(res.get('n_examples'))} / {integer(s.get('support'))}",
        _prf(s),
        _prf(o),
        integer(len(neg)) if isinstance(neg, list) else integer(neg),
        integer(len(prot)) if isinstance(prot, list) else integer(prot),
    ]


def pii_metrics(ctx: ResultsContext) -> Table | None:
    datasets = ctx.get("pii.datasets")
    if not isinstance(datasets, dict):
        return None
    rows = []
    dev = datasets.get("development") or {}
    for split in ("all", "dev", "heldout"):
        res = (dev.get("results") or {}).get(split)
        if res:
            label = "Development set (in-sample)" if split == "all" else f"  Development / {split}"
            rows.append(_pii_row(label, res))
    blind = datasets.get("blind_v2") or {}
    res = (blind.get("results") or {}).get("all")
    if res:
        rows.append(_pii_row("Blind held-out v2 (frozen detector)", res))
    post = ((blind.get("post_hoc") or {}).get("results") or {}).get("all")
    if post:
        rows.append(_pii_row("Blind v2 after post-hoc fixes (optimistic)", post))
    return _table(
        "pii_metrics",
        [
            "Dataset",
            "Examples / gold spans",
            "P / R / F1 strict",
            "P / R / F1 overlap",
            "PII-free messages altered",
            "Clinical values corrupted",
        ],
        rows,
        font_size=9,
        widths=[1.75, 0.75, 1.2, 1.2, 0.75, 0.75],
        note=_src(ctx, "pii") + " Strict = exact span and category; overlap = any overlap with a "
        "gold span of the same category. Post-hoc numbers are optimistic, not an independent "
        "estimate.",
    )


def pii_per_category(ctx: ResultsContext) -> Table | None:
    res = ctx.get("pii.datasets.blind_v2.results.all")
    dev = ctx.get("pii.datasets.development.results.all")
    if not isinstance(res, dict):
        return None
    cats = res.get("per_category_strict") or {}
    over = res.get("per_category_overlap") or {}
    dev_cats = (dev or {}).get("per_category_strict") or {}
    rows = []
    for cat in sorted(cats):
        c, o = cats[cat], over.get(cat) or {}
        rows.append(
            [
                cat,
                integer(c.get("support")),
                num(c.get("precision"), 3),
                num(c.get("recall"), 3),
                num(o.get("recall"), 3),
                num((dev_cats.get(cat) or {}).get("recall"), 3),
            ]
        )
    return _table(
        "pii_per_category",
        ["Category", "Gold spans", "P strict", "R strict", "R overlap", "R strict (dev set)"],
        rows,
        note=_src(ctx, "pii"),
    )


# ----------------------------------------------------------------------
# Performance, tests, benchmarks
# ----------------------------------------------------------------------

SCENARIO_NAMES: dict[str, str] = {
    "S1": "Emergency fast path",
    "S2": "Out of domain",
    "S3": "Clarification gate",
    "S4": "Clinical guidance (2 turns)",
    "S5": "Lifestyle",
}


def _by_turn(probe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, stats in (probe.get("warm_summary") or {}).items():
        turn, _, action = str(key).partition(":")
        row = out.setdefault(turn, {"actions": [], "p50": [], "p95": [], "n": 0, "cold": None})
        row["actions"].append(action)
        row["p50"].append(float(stats.get("p50", math.nan)))
        row["p95"].append(float(stats.get("p95", math.nan)))
        row["n"] += int(stats.get("n", 0))
    for row in out.values():
        vals = sorted(row["p50"])
        row["p50"] = vals[len(vals) // 2] if vals else math.nan
        row["p95"] = max(row["p95"]) if row["p95"] else math.nan
    for cold in probe.get("cold") or []:
        turn = f"{cold.get('scenario')}-t{cold.get('turn')}"
        if turn in out and _finite(cold.get("latency_s")):
            out[turn]["cold"] = float(cold["latency_s"])
    return out


def perf_before_after(ctx: ResultsContext) -> Table | None:
    before, after = ctx.get("perf_latest.before"), ctx.get("perf_latest.after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    b, a = _by_turn(before), _by_turn(after)
    rows = []
    for turn in sorted(set(b) | set(a)):
        rb, ra = b.get(turn), a.get(turn)
        speed = (
            (rb["p50"] / ra["p50"]) if rb and ra and _finite(ra["p50"]) and ra["p50"] > 0 else None
        )
        act_b = "/".join(rb["actions"]) if rb else "—"
        act_a = "/".join(ra["actions"]) if ra else "—"
        rows.append(
            [
                f"{turn} {SCENARIO_NAMES.get(turn.split('-')[0], '')}".strip(),
                act_b if act_b == act_a else f"{act_b} → {act_a}",
                f"{num(rb['cold']) if rb else '—'} / {num(ra['cold']) if ra else '—'}",
                f"{num(rb['p50']) if rb else '—'} / {num(ra['p50']) if ra else '—'}",
                f"{num(rb['p95']) if rb else '—'} / {num(ra['p95']) if ra else '—'}",
                f"{speed:.1f}×" if speed else "—",
            ]
        )
    src = ctx.get("perf_latest_sources") or {}
    note = (
        f"Sources: {src.get('before', 'before probe')} and {src.get('after', 'after probe')}. "
        "Seconds per turn, before / after; speed-up = before p50 / after p50."
    )
    return _table(
        "perf_before_after",
        [
            "Scenario turn",
            "Action (before → after)",
            "Cold s",
            "Warm p50 s",
            "Warm p95 s",
            "Speed-up (p50)",
        ],
        rows,
        font_size=9,
        note=note,
        widths=[1.6, 1.25, 0.75, 0.8, 0.8, 0.6],
    )


def perf_baseline(ctx: ResultsContext) -> Table | None:
    before = ctx.get("perf_latest.before")
    if not isinstance(before, dict):
        return None
    b = _by_turn(before)
    rows = [
        [
            turn,
            SCENARIO_NAMES.get(turn.split("-")[0], ""),
            "/".join(r["actions"]),
            num(r["cold"]),
            num(r["p50"]),
            num(r["p95"]),
            integer(r["n"]),
        ]
        for turn, r in sorted(b.items())
    ]
    src = (ctx.get("perf_latest_sources") or {}).get("before", "baseline probe")
    return _table(
        "perf_baseline",
        ["Turn", "Scenario", "Action", "Cold s", "Warm p50 s", "Warm p95 s", "Warm runs"],
        rows,
        note=f"Source: {src}.",
    )


def test_summary(ctx: ResultsContext) -> Table | None:
    data = ctx.get("tests")
    if not isinstance(data, dict):
        return None
    suites = data.get("suites")
    if isinstance(suites, dict):
        suites = [{"name": k, **(v if isinstance(v, dict) else {})} for k, v in suites.items()]
    if not isinstance(suites, list) or not suites:
        return None

    def g(s: dict[str, Any], *names: str) -> Any:
        for n in names:
            if s.get(n) is not None:
                return s[n]
        return None

    rows = []
    for s in suites:
        if not isinstance(s, dict):
            continue
        dur = g(s, "duration_s", "duration", "time_s", "seconds")
        rows.append(
            [
                str(g(s, "name", "suite") or "—"),
                str(g(s, "scope", "description", "path") or ""),
                integer(g(s, "total", "tests", "collected")),
                integer(g(s, "passed")),
                integer(g(s, "failed", "failures")),
                integer(g(s, "skipped", "deselected")),
                num(dur, 1) if _finite(dur) else "—",
            ]
        )
    totals = data.get("totals") or data.get("total")
    if isinstance(totals, dict):
        rows.append(
            [
                "Total",
                "",
                integer(g(totals, "total", "tests")),
                integer(totals.get("passed")),
                integer(g(totals, "failed", "failures")),
                integer(totals.get("skipped")),
                "",
            ]
        )
    note = _src(ctx, "tests")
    cov = data.get("coverage")
    if isinstance(cov, dict) and _finite(cov.get("percent", cov.get("total"))):
        note += f" Line coverage: {float(cov.get('percent', cov.get('total')) or 0):.1f}%."
    elif _finite(cov):
        note += f" Line coverage: {float(cov or 0):.1f}%."
    has_scope = any(r[1] for r in rows)
    header = ["Suite", "Scope", "Tests", "Passed", "Failed", "Skipped", "Time s"]
    if not has_scope:
        rows = [[r[0], *r[2:]] for r in rows]
        header = [header[0], *header[2:]]
    return _table("test_summary", header, rows, note=note)


def aq_benchmark(ctx: ResultsContext) -> Table | None:
    m = ctx.get("aq.metrics")
    if not isinstance(m, dict):
        return None
    meta = ctx.get("aq.metadata") or {}
    rows = [
        ["Vignettes evaluated", integer(m.get("total_vignettes"))],
        ["Routing accuracy (%)", perc(m.get("overall_routing_accuracy"))],
        [
            "Emergency detection recall (%)",
            f"{perc(m.get('emergency_detection_recall'))} ({integer(m.get('emergency_detected_count'))}"
            f"/{integer(m.get('emergency_count'))})",
        ],
        [
            "Out-of-domain questions answered",
            f"{integer(m.get('out_of_domain_answer_count'))} of {integer(m.get('out_of_domain_count'))}",
        ],
        ["Citation presence on expected answers (%)", perc(m.get("citation_presence_rate"))],
        ["Citation precision (%)", perc(m.get("overall_citation_precision"))],
        ["Loop-breaker forced GENERALIZE (%)", perc(m.get("loop_breaker_rate"))],
        [
            "Latency p50 / p95 (s)",
            f"{seconds(m.get('latency_p50_ms'))} / {seconds(m.get('latency_p95_ms'))}",
        ],
        ["Errors", integer(m.get("errors"))],
    ]
    model = meta.get("generator_model") or meta.get("gates_model")
    note = _src(ctx, "aq") + (f" Model: {model}." if model else "")
    if meta.get("limit") not in (None, "none", "None"):
        note += f" Partial run (limit {meta.get('limit')})."
    return _table(
        "aq_benchmark", ["Metric", "Value"], rows, note=note, align=["l", "c"], widths=[3.6, 2.1]
    )


def aq_subsets(ctx: ResultsContext) -> Table | None:
    subsets = ctx.get("aq.metrics.subsets")
    if not isinstance(subsets, dict) or not subsets:
        return None
    rows = [
        [
            name.replace("_", " "),
            integer(s.get("total_vignettes")),
            integer(s.get("correct_routings")),
            perc(s.get("routing_accuracy")),
            perc(s.get("avg_citation_precision")),
            seconds(s.get("avg_latency_ms")),
            integer(s.get("errors")),
        ]
        for name, s in subsets.items()
        if isinstance(s, dict)
    ]
    return _table(
        "aq_subsets",
        [
            "Subset",
            "Vignettes",
            "Correct",
            "Accuracy %",
            "Citation precision %",
            "Mean latency s",
            "Errors",
        ],
        rows,
        note=_src(ctx, "aq"),
    )


def rabbits(ctx: ResultsContext) -> Table | None:
    r = ctx.get("rabbits")
    if not isinstance(r, dict):
        return None
    rows = [
        [
            "Brand/generic pairs (Indian brands)",
            f"{integer(r.get('pairs'))} ({integer(r.get('indian_pairs'))})",
        ],
        ["Paired items", integer(r.get("items"))],
        ["Same routing action, brand vs generic (%)", perc(r.get("action_consistency_pct"))],
        ["Same extracted medication (%)", perc(r.get("extraction_consistency_pct"))],
        ["Brand normalised to generic (%)", perc(r.get("brand_normalised_pct"))],
        ["Citation Jaccard (mean)", num(r.get("citation_jaccard_mean"), 3)],
        ["Guideline Jaccard (mean)", num(r.get("guideline_jaccard_mean"), 3)],
        ["Answer length within 20% (%)", perc(r.get("length_ratio_within_20pct"))],
        ["Errors", integer(r.get("errors"))],
    ]
    for origin, s in (r.get("by_origin") or {}).items():
        if isinstance(s, dict):
            rows.append(
                [f"Action consistency, {origin} brands (%)", perc(s.get("action_consistency_pct"))]
            )
    return _table(
        "rabbits",
        ["Metric", "Value"],
        rows,
        note=_src(ctx, "rabbits"),
        align=["l", "c"],
        widths=[3.6, 2.1],
    )


def adversarial(ctx: ResultsContext) -> Table | None:
    results = ctx.get("adversarial.results")
    counts = ctx.get("adversarial.counts") or {}
    if not isinstance(results, list) or not results:
        return None
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if isinstance(r, dict):
            by_cat.setdefault(str(r.get("category", "other")), []).append(r)
    rows = []
    for cat, items in sorted(by_cat.items()):
        scores = [i.get("score") or {} for i in items]
        rows.append(
            [
                cat.replace("_", " "),
                integer(len(items)),
                integer(sum(1 for i in items if i.get("passed"))),
                integer(sum(1 for s in scores if s.get("leaked_system_prompt"))),
                integer(sum(1 for s in scores if s.get("fabricated_prescription"))),
                integer(sum(1 for s in scores if s.get("unsafe_output"))),
                integer(sum(1 for i in items if i.get("error"))),
            ]
        )
    rows.append(
        [
            "Total",
            integer(counts.get("payloads", len(results))),
            integer(counts.get("passed")),
            integer(counts.get("leaked_system_prompt")),
            integer(counts.get("fabricated_prescription")),
            integer(counts.get("unsafe_output")),
            integer(counts.get("errors")),
        ]
    )
    note = _src(ctx, "adversarial")
    if counts.get("emergency_payloads"):
        note += (
            f" Emergency payloads escalated: {integer(counts.get('emergency_escalated'))} of "
            f"{integer(counts.get('emergency_payloads'))}."
        )
    return _table(
        "adversarial",
        [
            "Category",
            "Payloads",
            "Passed",
            "Prompt leaked",
            "Fabricated dose",
            "Unsafe output",
            "Errors",
        ],
        rows,
        note=note,
    )


def load_test(ctx: ResultsContext) -> Table | None:
    levels = ctx.get("load.levels")
    if not isinstance(levels, list) or not levels:
        return None
    rows = [
        [
            integer(lv.get("users")),
            integer(lv.get("requests")),
            integer(lv.get("ok")),
            integer(lv.get("errors")),
            integer(lv.get("rate_limited")),
            pct(lv.get("error_rate"), ci=False),
            num(lv.get("throughput_rps"), 2),
            seconds(lv.get("p50_ms"), 2),
            seconds(lv.get("p95_ms"), 2),
            seconds(lv.get("max_ms"), 2),
        ]
        for lv in levels
        if isinstance(lv, dict)
    ]
    return _table(
        "load_test",
        [
            "Users",
            "Requests",
            "OK",
            "Errors",
            "Rate-limited",
            "Error %",
            "Throughput req/s",
            "p50 s",
            "p95 s",
            "Max s",
        ],
        rows,
        align=["c"] * 10,
        font_size=9,
        note=_src(ctx, "load") + " A single local Ollama server serialises LLM calls.",
    )


def smoke_test(ctx: ResultsContext) -> Table | None:
    checks = ctx.get("smoke.checks")
    if not isinstance(checks, list) or not checks:
        return None
    rows = [
        [
            str(c.get("name", "")).replace("_", " "),
            str(c.get("status", "")),
            str(c.get("expected", "")),
            str(c.get("observed", ""))[:60],
            num(c.get("latency_ms"), 0),
        ]
        for c in checks
        if isinstance(c, dict)
    ]
    return _table(
        "smoke_test",
        ["Check", "Status", "Expected", "Observed", "ms"],
        rows,
        font_size=9,
        note=_src(ctx, "smoke"),
        widths=[1.3, 0.55, 1.6, 1.8, 0.5],
    )


# ----------------------------------------------------------------------
# Hallucination study (evaluation/results/summary.json from evaluation/analyze.py)
# ----------------------------------------------------------------------


def _study_note(ctx: ResultsContext, extra: str = "") -> str:
    split = ctx.get("study.split", "test")
    note = f"{_src(ctx, 'study')} Split: {split}; 95% Wilson CIs; degraded turns excluded."
    flagged = [c for c, _ in _study_conditions(ctx) if _uncalibrated(ctx, c)]
    if flagged:
        note += f" UNCALIBRATED: {', '.join(flagged)}."
    return note + (f" {extra}" if extra else "")


def hallucination_by_condition(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        llm, nli = m.get("llm") or {}, m.get("nli") or {}
        rows.append(
            [
                _cond_label(c, m) + (" (UNCALIBRATED)" if _uncalibrated(ctx, c) else ""),
                integer(m.get("n_valid")),
                pct(llm.get("hallucination")),
                pct(nli.get("hallucination")),
                interval(llm.get("unsupported_claim_ratio")),
                pct(m.get("coverage_answerable"), ci=False),
                pct(m.get("over_abstention"), ci=False),
            ]
        )
    return _table(
        "hallucination_by_condition",
        [
            "Condition",
            "Turns",
            "Hallucination, LLM judge %",
            "Hallucination, NLI judge %",
            "Unsupported-claim ratio (LLM)",
            "Coverage %",
            "Over-abstention %",
        ],
        rows,
        font_size=8.5,
        note=_study_note(ctx),
        widths=[1.55, 0.45, 1.0, 1.0, 0.95, 0.55, 0.65],
    )


def unsafe_answer(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        llm, nli = m.get("llm") or {}, m.get("nli") or {}
        u = llm.get("unsafe_answer") or {}
        rows.append(
            [
                _cond_label(c, m),
                integer(u.get("n")),
                pct(u),
                pct(nli.get("unsafe_answer")),
                pct(m.get("appropriate_abstention")),
            ]
        )
    return _table(
        "unsafe_answer",
        [
            "Condition",
            "Unanswerable turns",
            "Unsafe answer, LLM judge %",
            "Unsafe answer, NLI judge %",
            "Appropriate abstention %",
        ],
        rows,
        font_size=9,
        note=_study_note(ctx),
        widths=[1.65, 0.75, 1.15, 1.15, 1.05],
    )


def routing_by_condition(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        routing = m.get("routing_system") if m.get("gated") else m.get("routing_behaviour")
        routing = routing or {}
        em = m.get("emergency") or {}
        rows.append(
            [
                _cond_label(c, m),
                "system" if m.get("gated") else "behaviour",
                pct(routing.get("accuracy_strict"), ci=False),
                pct(routing.get("accuracy_lenient"), ci=False),
                num(routing.get("macro_f1")),
                pct(
                    em.get("recall_system") if m.get("gated") else em.get("recall_behaviour"),
                    ci=False,
                ),
                pct(
                    em.get("false_escalation_system")
                    if m.get("gated")
                    else em.get("false_escalation_behaviour"),
                    ci=False,
                ),
            ]
        )
    return _table(
        "routing_by_condition",
        [
            "Condition",
            "Routing source",
            "Strict %",
            "Lenient %",
            "Macro-F1",
            "Emergency recall %",
            "False escalation %",
        ],
        rows,
        font_size=9,
        note=_study_note(ctx, "Ungated conditions are scored on the judge's behaviour label."),
        widths=[1.6, 0.75, 0.6, 0.6, 0.6, 0.75, 0.8],
    )


_OUTCOME_NAMES = {
    "hallucination": "Hallucination",
    "unsafe": "Unsafe answer",
    "hallucination_unfiltered": "Hallucination (no filter)",
}


def mcnemar(ctx: ResultsContext) -> Table | None:
    tests = ctx.get("study.tests.mcnemar")
    if not isinstance(tests, list) or not tests:
        return None
    rows = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        rows.append(
            [
                f"{t.get('treatment')} vs {t.get('control')}",
                _OUTCOME_NAMES.get(str(t.get("outcome")), str(t.get("outcome"))),
                str(t.get("judge", "")).upper(),
                integer(t.get("n_pairs")),
                f"{integer(t.get('b_treatment_only'))} / {integer(t.get('c_control_only'))}",
                pvalue(t.get("pvalue")),
                pvalue(t.get("pvalue_holm")),
                interval(t.get("risk_difference")),
                interval(t.get("odds_ratio"), 2),
            ]
        )
    return _table(
        "mcnemar",
        [
            "Comparison",
            "Outcome",
            "Judge",
            "Pairs",
            "b / c",
            "p",
            "p (Holm)",
            "Risk difference [95% CI]",
            "Odds ratio [95% CI]",
        ],
        rows,
        font_size=8,
        widths=[0.75, 0.85, 0.42, 0.42, 0.5, 0.48, 0.5, 1.0, 0.85],
        note=_study_note(
            ctx,
            "b = events only under the first condition, c = events "
            "only under the second; odds ratio < 1 favours the first.",
        ),
    )


def cochran_q(ctx: ResultsContext) -> Table | None:
    rows_in = ctx.get("study.tests.cochran_q")
    if not isinstance(rows_in, list) or not rows_in:
        return None
    rows = [
        [
            _OUTCOME_NAMES.get(str(r.get("outcome")), str(r.get("outcome"))),
            str(r.get("judge", "")).upper(),
            ", ".join(r.get("conditions") or []),
            integer(r.get("n")),
            num(r.get("q"), 2),
            integer(r.get("df")),
            pvalue(r.get("pvalue")),
        ]
        for r in rows_in
        if isinstance(r, dict)
    ]
    return _table(
        "cochran_q",
        ["Outcome", "Judge", "Conditions", "n", "Q", "df", "p"],
        rows,
        note=_study_note(ctx),
    )


def judge_agreement(ctx: ResultsContext) -> Table | None:
    a = ctx.get("study.judge_agreement")
    if not isinstance(a, dict):
        return None
    rows = [
        ["Claim level, 3 classes", integer(a.get("n_claims")), interval(a.get("kappa_3class"))],
        [
            "Claim level, supported vs not",
            integer(a.get("n_claims")),
            interval(a.get("kappa_binary")),
        ],
        [
            "Response level, hallucinated",
            integer(a.get("n_responses")),
            interval(a.get("kappa_response_hallucination")),
        ],
    ]
    human = ctx.get("human_agreement")
    if isinstance(human, dict):
        for k, v in human.items():
            if isinstance(v, dict) and ("kappa" in v or "estimate" in v):
                rows.append(
                    [f"Human vs judge: {k.replace('_', ' ')}", integer(v.get("n")), interval(v)]
                )
    return _table(
        "judge_agreement",
        ["Level", "n", "Cohen's κ [95% CI]"],
        rows,
        note=_study_note(ctx),
        widths=[2.6, 0.8, 2.0],
    )


def calibration(ctx: ResultsContext) -> Table | None:
    cal = ctx.get("calibration")
    if not isinstance(cal, dict):
        return None
    rows: list[list[str]] = []
    nli = cal.get("nli")
    if isinstance(nli, dict):
        rows.append(
            [
                "NLI judge",
                "θ (entailment)",
                num(nli.get("theta")),
                f"binary κ = {num(nli.get('kappa_binary'), 3)}",
            ]
        )
        rows.append(
            [
                "NLI judge",
                "θc (contradiction)",
                num(nli.get("theta_c")),
                f"3-class κ = {num(nli.get('kappa_3class'), 3)}; "
                f"{integer(nli.get('n_claims'))} claims",
            ]
        )
    for block, label in (("v1", "V1 calibrated Evidence Gate"), ("v2", "V2 grounding verifier")):
        b = cal.get(block)
        if not isinstance(b, dict):
            continue
        res = b.get("dev_result") or {}
        detail = []
        if _finite(res.get("coverage")):
            detail.append(f"dev coverage {pct(res.get('coverage'), ci=False)}%")
        if _finite(res.get("false_answer_rate")):
            detail.append(f"false-answer {pct(res.get('false_answer_rate'), ci=False)}%")
        if _finite(res.get("hallucination_rate")):
            detail.append(f"dev hallucination {pct(res.get('hallucination_rate'), ci=False)}%")
        if _finite(b.get("kappa_sentence")):
            detail.append(f"sentence κ = {num(b.get('kappa_sentence'), 3)}")
        for name, value in (b.get("thresholds") or {}).items():
            rows.append(
                [
                    label,
                    str(name),
                    num(value, 3) if _finite(value) else str(value),
                    "; ".join(detail),
                ]
            )
            detail = []
    return _table(
        "calibration",
        ["Component", "Parameter", "Value", "Dev-split result"],
        rows,
        note=_src(ctx, "calibration"),
        widths=[1.5, 2.0, 0.6, 1.65],
        font_size=9,
        align=["l", "l", "c", "l"],
    )


def degraded_turns(ctx: ResultsContext) -> Table | None:
    per = ctx.get("study.degraded.per_condition")
    if not isinstance(per, dict) or not per:
        return None
    order = [c for c in CONDITION_ORDER if c in per] + [c for c in per if c not in CONDITION_ORDER]
    rows = [
        [
            c,
            integer(per[c].get("n_turns")),
            integer(per[c].get("n_degraded")),
            pct(per[c].get("rate"), ci=False),
            integer(per[c].get("n_errors")),
            "; ".join(f"{k} ({v})" for k, v in (per[c].get("by_reason") or {}).items()) or "—",
        ]
        for c in order
    ]
    return _table(
        "degraded_turns",
        ["Condition", "Turns", "Degraded", "Rate %", "Errors", "Reasons"],
        rows,
        font_size=9,
        note=_study_note(ctx),
        widths=[0.7, 0.5, 0.65, 0.55, 0.5, 2.85],
        align=["l", "c", "c", "c", "c", "l"],
    )


def latency_by_condition(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        lat = m.get("latency") or {}
        raw, imp = lat.get("raw") or {}, lat.get("imputed") or {}
        rows.append(
            [
                _cond_label(c, m),
                integer(raw.get("n")),
                seconds(raw.get("p50_ms"), 2),
                seconds(raw.get("p95_ms"), 2),
                seconds(imp.get("p50_ms"), 2),
                seconds(imp.get("p95_ms"), 2),
            ]
        )
    return _table(
        "latency_by_condition",
        ["Condition", "n", "Raw p50 s", "Raw p95 s", "Imputed p50 s", "Imputed p95 s"],
        rows,
        note=_study_note(
            ctx,
            "Imputed = measured latency plus the median uncached "
            "latency of each memoised LLM call.",
        ),
    )


def category_breakdown(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    cats = [c for c in CATEGORY_ORDER]
    rows = []
    for c, m in conds:
        by = (m.get("llm") or {}).get("hallucination_by_category") or {}
        rows.append(
            [c]
            + [
                pct(by.get(cat), ci=False)
                + (f" (n={integer(by[cat].get('n'))})" if isinstance(by.get(cat), dict) else "")
                for cat in cats
            ]
        )
    return _table(
        "category_breakdown",
        ["Condition", *[CATEGORY_NAMES[c] for c in cats]],
        rows,
        font_size=8,
        note=_study_note(ctx),
    )


# ----------------------------------------------------------------------
# Static content
# ----------------------------------------------------------------------


def action_taken(ctx: ResultsContext) -> Table | None:
    """First pipe table of ``report/content/action_taken.md`` (written by the content stream)."""
    content_dir = ctx.data.get("_content_dir")
    root = (
        Path(content_dir) if content_dir else (ctx.repo_root or Path.cwd()) / "report" / "content"
    )
    path = root / "action_taken.md"
    if not path.is_file():
        return None
    blocks = parse_file(path, ParseContext(content_dir=path.parent))
    for block in blocks:
        if isinstance(block, Table):
            block.key = "action_taken"
            block.caption = block.caption or DEFAULT_CAPTIONS["action_taken"]
            block.font_size = block.font_size or 9.5
            return block
    return None


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

GENERATORS: dict[str, TableGenerator] = {
    "corpus": corpus,
    "corpus_by_type": corpus_by_type,
    "corpus_topics": corpus_topics,
    "retrieval_metrics": retrieval_metrics,
    "reranker_ablation": reranker_ablation,
    "model_selection": model_selection,
    "model_selection_round1": model_selection_round1,
    "model_selection_schema": model_selection_schema,
    "pii_metrics": pii_metrics,
    "pii_per_category": pii_per_category,
    "perf_before_after": perf_before_after,
    "perf_baseline": perf_baseline,
    "test_summary": test_summary,
    "aq_benchmark": aq_benchmark,
    "aq_subsets": aq_subsets,
    "rabbits": rabbits,
    "adversarial": adversarial,
    "load_test": load_test,
    "smoke_test": smoke_test,
    "hallucination_by_condition": hallucination_by_condition,
    "unsafe_answer": unsafe_answer,
    "routing_by_condition": routing_by_condition,
    "mcnemar": mcnemar,
    "cochran_q": cochran_q,
    "judge_agreement": judge_agreement,
    "calibration": calibration,
    "degraded_turns": degraded_turns,
    "latency_by_condition": latency_by_condition,
    "category_breakdown": category_breakdown,
    "hallubench_composition": hallubench_composition,
    "hallubench_actions": hallubench_actions,
    "action_taken": action_taken,
}


def generate(key: str, ctx: ResultsContext, errors: list[str] | None = None) -> Table | None:
    """Run a generator; unknown keys and generator errors both yield ``None`` (pending)."""
    fn = GENERATORS.get(key)
    if fn is None:
        if errors is not None:
            errors.append(f"unknown table key '{key}'")
        return None
    try:
        return fn(ctx)
    except Exception as exc:  # a malformed results file must never break the whole report
        if errors is not None:
            errors.append(f"table '{key}': {type(exc).__name__}: {exc}")
        return None

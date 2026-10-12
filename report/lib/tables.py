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
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .content import ParseContext, Table, parse_file
from .results_context import ResultsContext, format_p
from .study_figures import nli_coverage

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
    "rabbits_before_after": "Live RABBITS before and after the brand-name robustness fix",
    "adversarial_before_after": (
        "Live adversarial suite before and after the brand-name fix "
        "(after: Gate 3 on, before 16e594b)"
    ),
    "adversarial": "Adversarial prompt-injection suite (live pipeline)",
    "load_test": "Load test: concurrent users against the local stack",
    "smoke_test": "End-to-end smoke test checks",
    "hallucination_by_condition": "Hallucination by condition (test split, unit: item-turn)",
    "unsafe_answer": "Unsafe answers and appropriate abstention on unanswerable items",
    "routing_by_condition": "Routing accuracy and emergency handling by condition",
    "mcnemar": "Paired exact McNemar tests with Holm correction",
    "cochran_q": "Cochran's Q test across conditions C0 to C5",
    "judge_agreement": "Agreement between the two judges and between the AI annotator and each judge "
    "(Cohen's kappa)",
    "calibration": "Thresholds calibrated on the dev split",
    "hallubench_composition": "Composition of MedBridge-HalluBench v1",
    "hallubench_actions": "Gold actions per turn in MedBridge-HalluBench v1",
    "action_taken": "Action taken on previous review comments",
    "degraded_turns": "Degraded turns per condition (excluded from headline metrics)",
    "latency_by_condition": "Latency per condition",
    "category_breakdown": "Hallucination rate (LLM judge) by item category and condition",
    "claim_profile": "Claims per response and hallucination among responses that contain claims "
    "(LLM judge)",
    "category_safety": "Unsafe answers, appropriate abstention and coverage by item category "
    "(LLM judge)",
    "category_tests": "Paired McNemar tests within item categories (LLM judge, turn level)",
    "mcnemar_sensitivity": "Holm-adjusted McNemar p-values under every analysis view",
    "sensitivity_rates": "Hallucination rate (LLM judge) under the sensitivity analyses",
    "routing_per_class": "Per-class routing quality of the gated pipeline (system actions)",
    "judge_coverage": "Provenance, judge coverage and failed judgments per condition (test split)",
    "judge_confusion": "Claim labels of the LLM judge (rows) against the NLI judge (columns)",
    "study_runs": "Study and judge runs (manifests)",
    "study_models": "Models, versions, hardware and seeds of the hallucination study",
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


def pvalue(p: Any, compact: bool = False) -> str:
    """Exact p-value: 3 decimals, 4 below 0.01, scientific notation below 0.001."""
    if not _finite(p):
        return "—"
    text = format_p(p)
    return text.replace(" × ", "×") if compact else text


def points(d: Any, digits: int = 1) -> str:
    """Risk difference ``{estimate, low, high}`` (fractions) in percentage points."""
    if not isinstance(d, dict) or not _finite(d.get("estimate")):
        return "—"
    text = f"{100 * float(d['estimate']):.{digits}f}"
    if _finite(d.get("low")) and _finite(d.get("high")):
        text += f" [{100 * float(d['low']):.{digits}f}, {100 * float(d['high']):.{digits}f}]"
    return text


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


#: Short display names for the study tables (full names in Table 3.6 and CONDITION_NAMES).
CONDITION_SHORT: dict[str, str] = {
    "C0": "Direct LLM",
    "C1": "Vanilla RAG",
    "C1G": "RAG, prod. prompt",
    "C2": "RAG, safety prompt",
    "C3": "Context Gate",
    "C4": "Evidence Gate",
    "C5": "MedBridge",
    "V1": "Calibrated Gate 2",
    "V2": "Gate 3 verifier",
    "V12": "V1 + V2",
}


def _cond_label(cond: str, m: dict[str, Any] | None = None) -> str:
    name = CONDITION_SHORT.get(cond) or (m or {}).get("name") or CONDITION_NAMES.get(cond, cond)
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


def nli_status(m: dict[str, Any] | None) -> tuple[str, int]:
    """Coverage of the NLI judge for one condition (see :func:`study_figures.nli_coverage`)."""
    return nli_coverage(m)


def _nli_cell(m: dict[str, Any], key: str) -> str:
    status, n = nli_status(m)
    if status == "partial":
        return f"partial (n = {n})†"
    if status == "none":
        return "—"
    return pct((m.get("nli") or {}).get(key))


def _nli_note(conds: list[tuple[str, dict[str, Any]]]) -> str:
    complete = [c for c, m in conds if nli_status(m)[0] == "complete"]
    partial = [(c, nli_status(m)[1]) for c, m in conds if nli_status(m)[0] == "partial"]
    none = [c for c, m in conds if nli_status(m)[0] == "none"]
    if not partial and not none:
        return ""
    parts = []
    if complete:
        parts.append(f"NLI judge complete for {', '.join(complete)}")
    if partial:
        parts.append(
            "† partial subset ("
            + ", ".join(f"{c}: n = {n}" for c, n in partial)
            + "), used only for judge agreement and not a condition result"
        )
    if none:
        parts.append(f"— = not scored by the NLI judge ({', '.join(none)})")
    return "; ".join(parts) + "."


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
            f"{integer(ctx.get('corpus_orgs.split', totals.get('organisations')))} orgs",
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
    "S4": "Clinical guidance",
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


def _probe_model(label: str) -> str:
    """``after_qwen3.5-4b`` -> ``qwen3.5:4b`` (probe files are named after the model tag)."""
    tag = re.sub(r"^(before|after)(_warmup)?_?", "", label)
    return re.sub(r"-(\d+(?:\.\d+)?b)$", r":\1", tag) if tag else label


def perf_before_after(ctx: ResultsContext) -> Table | None:
    before, after = ctx.get("perf_latest.before"), ctx.get("perf_latest.after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    src = ctx.get("perf_latest_sources") or {}
    b, a = _by_turn(before), _by_turn(after)
    # other after-optimisation probes (another model, same scenarios), warm-up runs excluded
    extra = {
        label: _by_turn(probe)
        for label, probe in sorted((ctx.get("perf_latest") or {}).items())
        if label.startswith("after_")
        and "warmup" not in label
        and src.get(label) != src.get("after")
        and isinstance(probe, dict)
        and probe.get("warm_summary")
    }
    rows = []
    for turn in sorted(set(b) | set(a)):
        rb, ra = b.get(turn), a.get(turn)
        speed = (
            (rb["p50"] / ra["p50"]) if rb and ra and _finite(ra["p50"]) and ra["p50"] > 0 else None
        )
        act_b = "/".join(rb["actions"]) if rb else "—"
        act_a = "/".join(ra["actions"]) if ra else "—"
        row = [
            f"{turn} {SCENARIO_NAMES.get(turn.split('-')[0], '')}".strip(),
            act_b if act_b == act_a else f"{act_b} → {act_a}",
            f"{num(rb['cold']) if rb else '—'} / {num(ra['cold']) if ra else '—'}",
            f"{num(rb['p50']) if rb else '—'} / {num(ra['p50']) if ra else '—'}",
            f"{num(rb['p95']) if rb else '—'} / {num(ra['p95']) if ra else '—'}",
            f"{speed:.1f}×" if speed else "—",
        ]
        for probe in extra.values():
            re_ = probe.get(turn)
            row.append(num(re_["p50"]) if re_ else "—")
        rows.append(row)
    model = _probe_model(Path(str(src.get("before", "before"))).stem)
    note = (
        f"Sources: {src.get('before', 'before probe')} (received prototype) and "
        f"{src.get('after', 'after probe')} (post-study code, Gate 3 on, before 16e594b), both with "
        f"{model}. Seconds per turn, "
        "before / after; one cold pass, then three warm repeats; speed-up = before p50 / "
        "after p50."
    )
    header = [
        "Scenario turn",
        "Action",
        "Cold s",
        "Warm p50 s",
        "Warm p95 s",
        "Speed-up (p50)",
    ]
    widths = [2.0, 0.95, 0.9, 0.9, 0.9, 0.6]
    for label in extra:
        header.append(f"Warm p50 s, {_probe_model(label)}")
        widths.append(0.7)
        note += (
            f" Last column: {src.get(label, label)} (deployed model, post-study code before "
            "16e594b)."
        )
    return _table(
        "perf_before_after",
        header,
        rows,
        font_size=8,
        note=note,
        widths=widths,
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
        [
            "Vignettes at the loop-breaker ending in GENERALIZE",
            f"{integer(m.get('loop_breaker_generalize_count'))} of "
            f"{integer(m.get('loop_breaker_count'))}",
        ],
        [
            "Latency p50 / p95 (s)",
            f"{seconds(m.get('latency_p50_ms'))} / {seconds(m.get('latency_p95_ms'))}",
        ],
        ["Errors", integer(m.get("errors"))],
    ]
    model = meta.get("generator_model") or meta.get("gates_model")
    note = _src(ctx, "aq") + (f" Model: {model}." if model else "")
    if meta.get("git_commit"):
        note += f" Run at {_commit_note(meta)}."
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


def _commit_note(meta: dict[str, Any]) -> str:
    commit = str(meta.get("git_commit") or "?").replace("+dirty", " (uncommitted changes)")
    gate3 = "on" if meta.get("grounding_verifier") else "off"
    return f"commit {commit}, Gate 3 {gate3}"


def rabbits_before_after(ctx: ResultsContext) -> Table | None:
    before, after = ctx.get("rabbits_before"), ctx.get("rabbits")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    ib = ctx.get("rabbits_counts.before") or {}
    ia = ctx.get("rabbits_counts.after") or {}

    def tpl(r: dict[str, Any], group: str, name: str) -> Any:
        return ((r.get(group) or {}).get(name) or {}).get("action_consistency_pct")

    def pc(label: str, get: Callable[[dict[str, Any]], Any]) -> list[str]:
        return [label, perc(get(before)), perc(get(after))]

    rows = [
        pc("Same routing action, brand vs generic (%)", lambda r: r.get("action_consistency_pct")),
        pc("  general template (%)", lambda r: tpl(r, "by_template", "general")),
        pc("  personal template (%)", lambda r: tpl(r, "by_template", "personal")),
        pc("  Indian brands (%)", lambda r: tpl(r, "by_origin", "indian")),
        pc("  international brands (%)", lambda r: tpl(r, "by_origin", "international")),
        pc(
            "Same generic stored in the snapshot (%)", lambda r: r.get("extraction_consistency_pct")
        ),
        pc("Brand stored as exactly its generic (%)", lambda r: r.get("brand_normalised_pct")),
        ["Citation Jaccard, chunks / guidelines"]
        + [
            f"{num(r.get('citation_jaccard_mean'), 2)} / {num(r.get('guideline_jaccard_mean'), 2)}"
            for r in (before, after)
        ],
        ["Brand / generic messages abstained"]
        + [
            f"{integer(i.get('brand_abstain'))} / {integer(i.get('generic_abstain'))} of {integer(i.get('items'))}"
            for i in (ib, ia)
        ],
        ["Action flips (brand variant more cautious)"]
        + [
            f"{integer(i.get('flips'))} ({integer(i.get('brand_more_cautious'))})" for i in (ib, ia)
        ],
        ["Pairs (Indian); errors"]
        + [
            f"{integer(r.get('pairs'))} ({integer(r.get('indian_pairs'))}); {integer(r.get('errors'))}"
            for r in (before, after)
        ],
    ]
    note = (
        f"Sources: {ctx.source('rabbits_before')} (before: "
        f"{_commit_note(before.get('metadata') or {})}) and {ctx.source('rabbits')} (after: "
        f"{_commit_note(after.get('metadata') or {})}); live, qwen3.5:4b, same 60 brands."
    )
    return _table(
        "rabbits_before_after",
        ["Metric", "Before fix", "After fix"],
        rows,
        note=note,
        align=["l", "c", "c"],
        widths=[3.0, 1.35, 1.35],
    )


def adversarial_before_after(ctx: ResultsContext) -> Table | None:
    before, after = ctx.get("adversarial_before"), ctx.get("adversarial")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    cb, ca = before.get("counts") or {}, after.get("counts") or {}

    def actions(r: dict[str, Any]) -> dict[str, str]:
        return {
            str(x.get("payload_id")): str(x.get("action"))
            for x in r.get("results") or []
            if isinstance(x, dict)
        }

    ab, aa = actions(before), actions(after)
    changed = sorted(k for k in ab if k in aa and ab[k] != aa[k])

    def dist(acts: dict[str, str]) -> str:
        order = ["ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE"]
        counts = {a: sum(1 for v in acts.values() if v == a) for a in order}
        return ", ".join(f"{a} {n}" for a, n in counts.items() if n)

    rows = [
        [
            "Payloads passing every check",
            *[f"{integer(c.get('passed'))} of {integer(c.get('payloads'))}" for c in (cb, ca)],
        ],
        ["System-prompt leakage", *[integer(c.get("leaked_system_prompt")) for c in (cb, ca)]],
        ["Fabricated prescription", *[integer(c.get("fabricated_prescription")) for c in (cb, ca)]],
        ["Unsafe output", *[integer(c.get("unsafe_output")) for c in (cb, ca)]],
        [
            "Emergency payloads escalated",
            *[
                f"{integer(c.get('emergency_escalated'))} of {integer(c.get('emergency_payloads'))}"
                for c in (cb, ca)
            ],
        ],
        ["Final actions", dist(ab), dist(aa)],
        [
            "Payloads whose action changed",
            "",
            "; ".join(f"{k}: {ab[k]} → {aa[k]}" for k in changed) or "none",
        ],
    ]
    note = (
        f"Sources: {ctx.source('adversarial_before')} (before: "
        f"{_commit_note(before.get('metadata') or {})}; identical to "
        "reports/adversarial_live_report.json at commit 2053216) and "
        f"{ctx.source('adversarial')} (after: {_commit_note(after.get('metadata') or {})})."
    )
    return _table(
        "adversarial_before_after",
        ["Check", "Before", "After"],
        rows,
        note=note,
        align=["l", "c", "c"],
        widths=[1.9, 1.9, 1.9],
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
        widths=[0.5, 0.7, 0.4, 0.55, 0.8, 0.55, 0.9, 0.5, 0.55, 0.55],
        note=_src(ctx, "load") + " A single local Ollama server serialises LLM calls.",
    )


#: Requests of the smoke checks that send no message text (see ``scripts/smoke_test.py``).
SMOKE_REQUESTS: dict[str, str] = {
    "health": "GET /health",
    "create_session": "POST /api/sessions",
    "feedback": "POST …/feedback (rating +1)",
    "context": "GET …/context",
    "summary": "GET …/summary?format=json",
    "auth_missing_token": "GET …/history without token",
    "delete_session": "DELETE /api/sessions/{id}",
    "after_delete_gone": "GET …/history after erasure",
}


def smoke_test(ctx: ResultsContext) -> Table | None:
    checks = ctx.get("smoke.checks")
    if not isinstance(checks, list) or not checks:
        return None
    inputs = ctx.get("smoke_inputs") or {}
    rows = []
    for n, c in enumerate((c for c in checks if isinstance(c, dict)), start=1):
        name = str(c.get("name", ""))
        request = SMOKE_REQUESTS.get(name) or inputs.get(name) or ""
        if name == "sse_stream" and request:
            request = f"stream: {request}"
        rows.append(
            [
                f"ST-{n:02d}",
                name.replace("_", " "),
                request,
                str(c.get("expected", "")),
                re.sub(r"[{}']", "", str(c.get("observed", ""))).replace(": ", " ")[:64],
                str(c.get("status", "")),
            ]
        )
    note = _src(ctx, "smoke")
    if inputs:
        note += f" Message texts from {ctx.source('smoke_inputs')}."
    return _table(
        "smoke_test",
        ["ID", "Check", "Input (request)", "Expected", "Actual", "Status"],
        rows,
        font_size=8,
        note=note,
        widths=[0.45, 0.95, 2.0, 0.9, 1.15, 0.45],
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


def _nb(text: str) -> str:
    """Bind a rate with its CI (``72.8 [66.8, 78.1]``) so it never wraps inside a table cell;
    other text (``partial (n = 76)†``) is left free to wrap."""
    if re.match(r"^-?[\d.]+ \[-?[\d.]+, -?[\d.]+\]$", text):
        return text.replace(" ", "\u00a0")
    return text


def _turns_note(conds: list[tuple[str, dict[str, Any]]], key: str = "n_valid") -> str:
    """``Turns: 239 per condition (238 for C0, V1, V12).`` from the data."""
    ns = [m.get(key) for _c, m in conds if _finite(m.get(key))]
    if not ns:
        return ""
    modal = max(set(ns), key=ns.count)
    other = [f"{integer(m.get(key))} for {c}" for c, m in conds if m.get(key) != modal]
    return f"Turns: {integer(modal)} per condition" + (f" ({', '.join(other)})." if other else ".")


def hallucination_by_condition(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        llm = m.get("llm") or {}
        ratio = llm.get("unsupported_claim_ratio") or {}
        rows.append(
            [
                _cond_label(c, m) + (" (UNCALIBRATED)" if _uncalibrated(ctx, c) else ""),
                _nb(pct(llm.get("hallucination"))),
                _nb(_nli_cell(m, "hallucination")),
                pct(llm.get("hallucination_given_claims"), ci=False),
                num(ratio.get("estimate") if isinstance(ratio, dict) else ratio, 3),
                f"{pct(m.get('coverage_answerable'), ci=False)} / "
                f"{pct(m.get('over_abstention'), ci=False)}",
                pct(cit, ci=False) if (cit := _citation_precision(m)) else "—",
            ]
        )
    failed = [
        f"{c} {integer(m.get('n_valid'))} valid, {integer((m.get('llm') or {}).get('n_judged'))} "
        "judged"
        for c, m in conds
        if _finite((m.get("llm") or {}).get("n_judged"))
        and _finite(m.get("n_valid"))
        and (m.get("llm") or {}).get("n_judged") < m.get("n_valid")
    ]
    extra = " ".join(
        x
        for x in (
            _turns_note(conds),
            f"LLM-judge failures: {'; '.join(failed)}." if failed else "",
            _nli_note(conds),
            "With claims = hallucination among responses with at least one claim. Claim ratio = "
            "(UNSUPPORTED + CONTRADICTED) / all claims. Coverage / over-abstention on answerable "
            "turns. Citations = supported citation precision (markers whose cited passage backs a "
            "SUPPORTED claim of the sentence).",
        )
        if x
    )
    return _table(
        "hallucination_by_condition",
        [
            "Condition",
            "Hallucination, LLM judge %",
            "Hallucination, NLI judge %",
            "With claims %",
            "Claim ratio",
            "Coverage / over-abstention %",
            "Citations %",
        ],
        rows,
        font_size=7.5,
        note=_study_note(ctx, extra),
        widths=[1.3, 1.15, 1.15, 0.5, 0.5, 0.85, 0.55],
    )


def _citation_precision(m: dict[str, Any]) -> dict[str, Any] | None:
    cit = (m.get("citations") or {}).get("supported_precision")
    return cit if isinstance(cit, dict) and cit.get("n") else None


def unsafe_answer(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        llm = m.get("llm") or {}
        u = llm.get("unsafe_answer") or {}
        nli_unsafe = (m.get("nli") or {}).get("unsafe_answer") or {}
        status, _n_nli = nli_status(m)
        if status == "partial" and not nli_unsafe.get("n"):
            nli_cell = "partial, none in subset†"
        else:
            nli_cell = _nli_cell(m, "unsafe_answer")
        rows.append(
            [
                _cond_label(c, m),
                integer(u.get("n")),
                _nb(pct(u)),
                _nb(nli_cell),
                _nb(pct(m.get("appropriate_abstention"))),
            ]
        )
    return _table(
        "unsafe_answer",
        [
            "Condition",
            "n",
            "Unsafe, LLM judge %",
            "Unsafe, NLI judge %",
            "Appropriate abstention %",
        ],
        rows,
        font_size=8,
        note=_study_note(
            ctx,
            "n = unanswerable turns (unanswerable in domain and out of domain). "
            + _nli_note(conds),
        ),
        widths=[1.4, 0.3, 1.2, 1.2, 1.2],
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
                "system" if m.get("gated") else "judge",
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
            "Source",
            "Strict %",
            "Lenient %",
            "Macro-F1",
            "Emergency recall %",
            "False escalation %",
        ],
        rows,
        font_size=8,
        note=_study_note(
            ctx,
            "Source: system = pipeline action (gated conditions); judge = LLM judge's behaviour "
            "label mapped to an action (ungated conditions), so the two groups are not directly "
            "comparable. C0 has no emergency fast path; its recall here is behavioural.",
        ),
        widths=[1.5, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8],
    )


_OUTCOME_NAMES = {
    "hallucination": "Hallucination",
    "unsafe": "Unsafe",
    "hallucination_unfiltered": "Hallucination (no filter)",
}


_FAMILY_NOTES = {
    "c5_vs_baselines": "C5 vs C0–C4",
    "variants_vs_c5": "V1, V2, V12 vs C5",
    "c5_vs_reference": "C5 vs C1G",
}


def _family_note(tests: list[dict[str, Any]]) -> str:
    sizes: dict[str, set[str]] = {}
    for t in tests:
        fam = str(t.get("family", ""))
        sizes.setdefault(fam, set()).add(f"{t.get('treatment')}/{t.get('control')}")
    parts = [
        f"{_FAMILY_NOTES.get(f, f)} ({len(s)} test{'s' if len(s) != 1 else ''})"
        for f, s in sizes.items()
        if f
    ]
    return (
        "Holm correction within each family and outcome: " + ", ".join(parts) + "." if parts else ""
    )


def mcnemar(ctx: ResultsContext) -> Table | None:
    tests = ctx.get("study.tests.mcnemar")
    if not isinstance(tests, list) or not tests:
        return None
    tests = [t for t in tests if isinstance(t, dict)]
    judges = {str(t.get("judge", "")).upper() for t in tests}
    show_judge = len(judges) > 1
    short = {"hallucination": "Halluc.", "unsafe": "Unsafe"}
    rows = []
    for t in tests:
        rrr = t.get("rrr")
        row = [
            f"{t.get('treatment')} vs {t.get('control')}",
            short.get(str(t.get("outcome")), str(t.get("outcome"))),
        ]
        if show_judge:
            row.append(str(t.get("judge", "")).upper())
        row += [
            integer(t.get("n_pairs")),
            f"{integer(t.get('b_treatment_only'))} / {integer(t.get('c_control_only'))}",
            _nb(points(t.get("risk_difference"))),
            _nb(interval(t.get("odds_ratio"), 3)),
            f"{100 * float(rrr):.1f}" if _finite(rrr) else "—",
            pvalue(t.get("pvalue_holm"), compact=True),
        ]
        rows.append(row)
    header = ["Comparison", "Outcome"] + (["Judge"] if show_judge else [])
    header += ["n", "b / c", "Risk difference, pp [95% CI]", "Odds ratio [95% CI]", "RRR %"]
    header += ["p (Holm)"]
    widths = [0.75, 0.7] + ([0.42] if show_judge else [])
    widths += [0.3, 0.45, 1.25, 1.25, 0.4, 0.75]
    if show_judge:
        judge_text = ""
    elif judges == {"LLM"}:
        model = _llm_judge_model(ctx)
        unscored = [c for c, m in _study_conditions(ctx) if nli_status(m)[0] == "none"]
        judge_text = f"All tests use the LLM judge{f' ({model})' if model else ''}."
        if unscored:
            judge_text += (
                f" The NLI judge did not score {', '.join(unscored)} on this split, so no NLI "
                "test exists."
            )
    else:
        judge_text = f"Judge: {', '.join(sorted(judges))}."
    return _table(
        "mcnemar",
        header,
        rows,
        font_size=8,
        widths=widths,
        note=_study_note(
            ctx,
            "Exact two-sided McNemar test on the same item-turns (Halluc. = hallucination); "
            "n = pairs; b = events only under the first condition, c = only under the second; "
            "risk difference = first minus second in percentage points (Newcombe paired CI); "
            "conditional odds ratio < 1 favours the first (Haldane +0.5 when b or c = 0); RRR = "
            "relative risk reduction; unadjusted p-values in Appendix A. "
            f"{_family_note(tests)} {judge_text}".strip(),
        ),
    )


def _llm_judge_model(ctx: ResultsContext) -> str | None:
    """Model of the most recent LLM-judge manifest (``judgments/manifest_judge_*.json``)."""
    manifests = ctx.get("judge_manifests") or {}
    for _stem, m in sorted(manifests.items(), reverse=True):
        if isinstance(m, dict) and m.get("judge") == "llm" and m.get("judge_model"):
            return str(m["judge_model"])
    return None


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

    def obs(k: Any) -> str:
        return pct((k or {}).get("observed_agreement"), ci=False) if isinstance(k, dict) else "—"

    rows = [
        [
            "Claim level, 3 classes",
            integer(a.get("n_claims")),
            interval(a.get("kappa_3class")),
            obs(a.get("kappa_3class")),
        ],
        [
            "Claim level, supported vs not",
            integer(a.get("n_claims")),
            interval(a.get("kappa_binary")),
            obs(a.get("kappa_binary")),
        ],
        [
            "Response level, hallucinated",
            integer(a.get("n_responses")),
            interval(a.get("kappa_response_hallucination")),
            obs(a.get("kappa_response_hallucination")),
        ],
    ]
    human = ctx.get("human_agreement")
    if isinstance(human, dict):
        for k, v in human.items():
            if isinstance(v, dict) and ("kappa" in v or "estimate" in v):
                rows.append(
                    [
                        f"Human vs judge: {k.replace('_', ' ')}",
                        integer(v.get("n")),
                        interval(v),
                        obs(v),
                    ]
                )
    ai = ctx.get("ai_annotation")
    if isinstance(ai, dict):
        for label, block, kappa in (
            ("AI annotator vs LLM judge, 3 classes", ai.get("ai_vs_llm"), "kappa_3class"),
            ("AI annotator vs NLI judge, 3 classes", ai.get("ai_vs_nli"), "kappa_3class"),
        ):
            if isinstance(block, dict) and block.get("n"):
                rows.append(
                    [
                        label,
                        integer(block.get("n")),
                        interval(block.get(kappa)),
                        obs(block.get(kappa)),
                    ]
                )
        resp = ai.get("response_level")
        if isinstance(resp, dict) and resp.get("n"):
            rows.append(
                [
                    "AI annotator vs LLM judge, response level",
                    integer(resp.get("n")),
                    interval(resp.get("kappa")),
                    obs(resp.get("kappa")),
                ]
            )
    conds = _study_conditions(ctx)
    paired = [
        f"{c} {integer(n)}" + (" (partial)" if status == "partial" else "")
        for c, m in conds
        for status, n in [nli_status(m)]
        if status != "none"
    ]
    extra = f"Paired responses (both judges): {', '.join(paired)}." if paired else ""
    if isinstance(ai, dict):
        extra += (
            " AI annotator: Claude Opus 5.5, an AI model and not a human, on the "
            f"{integer(ai.get('n_claims'))} claims of the annotation sample (C5 and C1; NLI labels "
            "exist for C1 only); evaluation/results/ai_annotation_agreement.json."
        )
    if not isinstance(human, dict):
        extra += " No human annotation has been scored."
    return _table(
        "judge_agreement",
        ["Level", "n", "Cohen's κ [95% CI]", "Observed agreement %"],
        rows,
        note=_study_note(ctx, extra.strip()),
        widths=[2.2, 0.7, 1.7, 1.1],
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
        base = b.get("c5_dev_baseline") or {}

        def vs(key: str, label_text: str, res: dict[str, Any] = res, base: dict = base) -> str:
            text = f"{label_text} {pct(res.get(key), ci=False)}%"
            if _finite(base.get(key)):
                text += f" (C5 {pct(base.get(key), ci=False)}%)"
            return text

        detail = []
        if _finite(res.get("false_answer_rate")):
            detail.append(vs("false_answer_rate", "false ANSWER"))
        if _finite(res.get("hallucination_rate")):
            detail.append(vs("hallucination_rate", "hallucination"))
        if _finite(res.get("coverage")):
            detail.append(vs("coverage", "coverage"))
        if _finite(b.get("kappa_sentence")):
            detail.append(f"sentence κ = {num(b.get('kappa_sentence'), 3)}")
        if _finite(res.get("n_turns")):
            detail.append(f"{integer(res.get('n_turns'))} C5 dev turns")
        disabled = b.get("disabled_value")
        for name, value in (b.get("thresholds") or {}).items():
            off = (_finite(disabled) and value == disabled) or (
                name == "EVIDENCE_MIN_ENTITY_COVERAGE" and _finite(value) and value <= 0
            )
            rows.append(
                [
                    label,
                    _PARAM_NAMES.get(str(name), str(name)),
                    (f"{float(value):g}" if _finite(value) else str(value))
                    + (" (off)" if off else ""),
                    "; ".join(detail),
                ]
            )
            detail = []
    return _table(
        "calibration",
        ["Component", "Parameter", "Value", "Dev-split result"],
        rows,
        note=_src(ctx, "calibration") + " All thresholds were fitted on the development split "
        "only; (off) = the signal can never fire at this value.",
        widths=[1.35, 1.75, 0.75, 1.9],
        font_size=8.5,
        align=["l", "l", "c", "l"],
    )


_PARAM_NAMES = {
    "EVIDENCE_MIN_RERANK_ANSWER": "r_answer: min. rerank score for ANSWER",
    "EVIDENCE_MIN_RERANK_ANY": "r_any: min. rerank score for any content",
    "EVIDENCE_MIN_ENTITY_COVERAGE": "c_min: min. entity coverage for ANSWER",
    "GROUNDING_ENTAILMENT_THRESHOLD": "θ: entailment threshold per claim",
    "GROUNDING_MAX_UNSUPPORTED_FRACTION": "τ: max. unsupported fraction kept",
}


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
            "Imputed = measured latency plus the median uncached latency of each memoised LLM "
            "call. Measured with parallel study workers on a GPU shared with other evaluation "
            "jobs: relative indications only, not user-facing latency.",
        ),
    )


def category_breakdown(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    cats = [c for c in CATEGORY_ORDER]
    by_cond = {c: (m.get("llm") or {}).get("hallucination_by_category") or {} for c, m in conds}
    modal: dict[str, Any] = {}
    for cat in cats:
        ns = [by[cat].get("n") for by in by_cond.values() if isinstance(by.get(cat), dict)]
        modal[cat] = max(set(ns), key=ns.count) if ns else None
    rows = []
    for c, _m in conds:
        by = by_cond[c]
        cells = []
        for cat in cats:
            p = by.get(cat)
            text = pct(p, ci=False)
            if isinstance(p, dict) and p.get("n") != modal[cat]:
                text += f" (n = {integer(p.get('n'))})"
            cells.append(text)
        rows.append([c, *cells])
    header = ["Condition"] + [
        CATEGORY_NAMES[c] + (f" (n = {integer(modal[c])})" if modal[c] is not None else "")
        for c in cats
    ]
    return _table(
        "category_breakdown",
        header,
        rows,
        font_size=8,
        note=_study_note(ctx, "Turns per category in the header; a cell shows n when it differs."),
    )


def claim_profile(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    if not conds:
        return None
    rows = []
    for c, m in conds:
        llm = m.get("llm") or {}
        given = llm.get("hallucination_given_claims") or {}
        labels = llm.get("claim_labels") or {}
        cit = (m.get("citations") or {}).get("supported_precision") or {}
        rows.append(
            [
                _cond_label(c, m),
                f"{integer(given.get('n'))} / {integer(llm.get('n_judged'))}",
                pct(given),
                num(llm.get("claims_per_response"), 2),
                " / ".join(
                    integer(labels.get(k, 0)) for k in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")
                ),
                _nb(interval(llm.get("unsupported_claim_ratio"))),
                pct(cit, ci=False) if cit.get("n") else "—",
            ]
        )
    return _table(
        "claim_profile",
        [
            "Condition",
            "Responses with claims / judged",
            "Hallucination among responses with claims %",
            "Claims per response",
            "Claims S / U / C",
            "Claim ratio [95% CI]",
            "Supported citation precision %",
        ],
        rows,
        font_size=8,
        widths=[1.3, 0.75, 1.15, 0.55, 1.0, 1.2, 0.7],
        note=_study_note(
            ctx,
            "Template responses (ABSTAIN, ESCALATE) and replies without clinical claims have zero "
            "claims and count as not hallucinated. S / U / C = SUPPORTED / UNSUPPORTED / "
            "CONTRADICTED claims (at most 25 per response); claim ratio = (U + C) / all claims "
            "with a 95% bootstrap CI over responses.",
        ),
    )


def _cat_metric(ctx: ResultsContext, cat: str, cond: str, key: str) -> Any:
    return ctx.get(f"study.by_category.{cat}.per_condition.{cond}.{key}")


def category_safety(ctx: ResultsContext) -> Table | None:
    by = ctx.get("study.by_category")
    conds = _study_conditions(ctx)
    if not isinstance(by, dict) or not conds:
        return None
    rows = []
    for c, _m in conds:
        rows.append(
            [
                c,
                pct(_cat_metric(ctx, "unanswerable_in_domain", c, "unsafe_llm"), ci=False),
                pct(
                    _cat_metric(ctx, "unanswerable_in_domain", c, "appropriate_abstention"),
                    ci=False,
                ),
                pct(_cat_metric(ctx, "out_of_domain", c, "unsafe_llm"), ci=False),
                pct(_cat_metric(ctx, "out_of_domain", c, "appropriate_abstention"), ci=False),
                pct(_cat_metric(ctx, "answerable_complete", c, "coverage_answerable"), ci=False),
                pct(
                    _cat_metric(ctx, "answerable_missing_context", c, "coverage_answerable"),
                    ci=False,
                ),
                pct(_cat_metric(ctx, "general_topic", c, "coverage_answerable"), ci=False),
            ]
        )

    def n_of(cat: str, key: str) -> str:
        p = _cat_metric(ctx, cat, "C5", key) or {}
        return f" (n = {integer(p.get('n'))})" if isinstance(p, dict) and p.get("n") else ""

    return _table(
        "category_safety",
        [
            "Condition",
            "Unanswerable in domain: unsafe %" + n_of("unanswerable_in_domain", "unsafe_llm"),
            "Unanswerable in domain: appropriate abstention %",
            "Out of domain: unsafe %" + n_of("out_of_domain", "unsafe_llm"),
            "Out of domain: appropriate abstention %",
            "Coverage, answerable complete %" + n_of("answerable_complete", "coverage_answerable"),
            "Coverage, missing context %"
            + n_of("answerable_missing_context", "coverage_answerable"),
            "Coverage, general topic %" + n_of("general_topic", "coverage_answerable"),
        ],
        rows,
        font_size=8,
        widths=[0.6, 0.85, 0.85, 0.75, 0.75, 0.8, 0.8, 0.75],
        note=_study_note(ctx, "n = turns of C5 (a few conditions have fewer valid turns)."),
    )


_CATEGORY_COMPARISONS = (("C5", "C1"), ("C5", "C1G"), ("V2", "C5"))


def category_tests(ctx: ResultsContext) -> Table | None:
    by = ctx.get("study.by_category")
    if not isinstance(by, dict) or not by:
        return None
    rows = []
    for cat in [c for c in CATEGORY_ORDER if c in by]:
        tests = [t for t in (by[cat].get("tests") or []) if isinstance(t, dict)]
        for outcome in ("hallucination", "unsafe"):
            for treat, control in _CATEGORY_COMPARISONS:
                t = next(
                    (
                        x
                        for x in tests
                        if x.get("treatment") == treat
                        and x.get("control") == control
                        and x.get("outcome") == outcome
                    ),
                    None,
                )
                if t is None:
                    continue
                rows.append(
                    [
                        CATEGORY_NAMES.get(cat, cat),
                        f"{treat} vs {control}",
                        _OUTCOME_NAMES.get(outcome, outcome),
                        integer(t.get("n_pairs")),
                        f"{integer(t.get('b_treatment_only'))} / "
                        f"{integer(t.get('c_control_only'))}",
                        points(t.get("risk_difference")),
                        pvalue(t.get("pvalue_holm"), compact=True),
                    ]
                )
    return _table(
        "category_tests",
        [
            "Category",
            "Comparison",
            "Outcome",
            "Pairs",
            "b / c",
            "Risk difference, pp [95% CI]",
            "p (Holm)",
        ],
        rows,
        font_size=8,
        widths=[1.45, 0.8, 0.85, 0.45, 0.5, 1.2, 0.7],
        note=_study_note(
            ctx,
            "Holm correction within each category, family and outcome (the full families also "
            "contain the comparisons not shown).",
        ),
    )


def _match_test(tests: Any, ref: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tests, list):
        return None
    outcome = str(ref.get("outcome"))
    for t in tests:
        if not isinstance(t, dict):
            continue
        t_outcome = str(t.get("outcome")).replace("_unfiltered", "")
        if (
            t.get("treatment") == ref.get("treatment")
            and t.get("control") == ref.get("control")
            and t_outcome == outcome
            and t.get("judge") == ref.get("judge")
        ):
            return t
    return None


def mcnemar_sensitivity(ctx: ResultsContext) -> Table | None:
    base = ctx.get("study.tests.mcnemar")
    if not isinstance(base, list) or not base:
        return None
    views = (
        ("Item level", ctx.get("study.tests.sensitivity_item_level")),
        ("Without prompt_overlap", ctx.get("study.sensitivity.without_prompt_overlap.tests")),
        ("Filter disabled", ctx.get("study.sensitivity.filter_disabled.tests")),
        ("Degraded included", ctx.get("study.sensitivity.including_degraded.tests")),
    )
    views = tuple(v for v in views if isinstance(v[1], list) and v[1])
    rows = []
    for t in base:
        if not isinstance(t, dict):
            continue
        row = [
            f"{t.get('treatment')} vs {t.get('control')}",
            _OUTCOME_NAMES.get(str(t.get("outcome")), str(t.get("outcome"))),
            pvalue(t.get("pvalue"), compact=True),
            pvalue(t.get("pvalue_holm"), compact=True),
        ]
        for _label, tests in views:
            m = _match_test(tests, t)
            row.append(pvalue(m.get("pvalue_holm"), compact=True) if m else "—")
        rows.append(row)
    removed = ctx.get("study.sensitivity.without_prompt_overlap.n_items_removed")
    return _table(
        "mcnemar_sensitivity",
        ["Comparison", "Outcome", "Headline p", "Headline p (Holm)"]
        + [f"{label} p (Holm)" for label, _ in views],
        rows,
        font_size=7.5,
        widths=[0.72, 0.8] + [0.68] * (2 + len(views)),
        note=_study_note(
            ctx,
            "Headline p = unadjusted exact McNemar p. Item level: one outcome per item (items "
            "complete in both conditions; computed for "
            "the C5 vs C0–C4 family only). Without prompt_overlap: "
            + (f"{integer(removed)} " if _finite(removed) else "")
            + "items close to the original prompt examples removed. Filter disabled: generic "
            "referral and self-reference claims counted (hallucination only). Degraded "
            "included: degraded turns kept. — = not computed for this view.",
        ),
    )


def sensitivity_rates(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    sens = ctx.get("study.sensitivity")
    if not conds or not isinstance(sens, dict):
        return None

    def view(name: str, c: str, key: str) -> Any:
        return ctx.get(f"study.sensitivity.{name}.per_condition.{c}.{key}")

    rows = []
    for c, m in conds:
        rows.append(
            [
                _cond_label(c, m),
                pct((m.get("llm") or {}).get("hallucination")),
                pct(view("including_degraded", c, "hallucination_llm")),
                pct(view("without_prompt_overlap", c, "hallucination_llm")),
                pct(view("filter_disabled", c, "hallucination_unfiltered")),
            ]
        )
    return _table(
        "sensitivity_rates",
        [
            "Condition",
            "Headline %",
            "Degraded included %",
            "Without prompt_overlap %",
            "Filter disabled %",
        ],
        rows,
        font_size=8.5,
        widths=[1.6, 1.0, 1.0, 1.0, 1.0],
        note=_study_note(ctx, "Rates with 95% Wilson CIs."),
    )


def routing_per_class(ctx: ResultsContext) -> Table | None:
    conds = [
        (c, m) for c, m in _study_conditions(ctx) if m.get("gated") and m.get("routing_system")
    ]
    main = next((m for c, m in conds if c == "C5"), None)
    if not conds or main is None:
        return None
    per_main = main["routing_system"].get("per_class") or {}
    others = [(c, m) for c, m in conds if c != "C5"]
    rows = []
    for cls in [a for a in ACTIONS if a in per_main]:
        r = per_main[cls]
        rows.append(
            [
                cls,
                integer(r.get("support")),
                integer(r.get("predicted")),
                num(r.get("precision")),
                num(r.get("recall")),
                num(r.get("f1")),
            ]
            + [
                num(((m["routing_system"].get("per_class") or {}).get(cls) or {}).get("f1"))
                for _c, m in others
            ]
        )
    rows.append(
        ["Macro-F1", "", "", "", "", num(main["routing_system"].get("macro_f1"))]
        + [num(m["routing_system"].get("macro_f1")) for _c, m in others]
    )
    never = []
    for c, m in conds:
        pc = m["routing_system"].get("per_class") or {}
        missing = [
            cls
            for cls in ACTIONS
            if (pc.get(cls) or {}).get("support") and not (pc.get(cls) or {}).get("predicted")
        ]
        if missing:
            never.append(f"{c} never predicted {', '.join(missing)}")
    return _table(
        "routing_per_class",
        ["Gold action", "Gold turns", "C5 predicted", "C5 precision", "C5 recall", "C5 F1"]
        + [f"{c} F1" for c, _m in others],
        rows,
        font_size=8,
        note=_study_note(
            ctx,
            "Pipeline actions of the gated conditions against the gold action."
            + (f" F1 = 0 where a class was never predicted ({'; '.join(never)})." if never else ""),
        ),
    )


def judge_coverage(ctx: ResultsContext) -> Table | None:
    conds = _study_conditions(ctx)
    jstats = ctx.get("study_judgments") or {}
    if not conds:
        return None
    llm_stats = jstats.get("llm") or {}
    nli_stats = jstats.get("nli") or {}
    manifests = ctx.get("study_manifests") or {}
    records = (ctx.get("study_records") or {}).get("by_condition") or {}

    def provenance(cond: str) -> str:
        by_commit: dict[str, int] = {}
        for rid, n in ((records.get(cond) or {}).get("final_by_run") or {}).items():
            commit = str(((manifests.get(rid) or {}).get("git") or {}).get("commit") or "?")[:7]
            by_commit[commit] = by_commit.get(commit, 0) + int(n)
        return ", ".join(f"{k} {integer(v)}" for k, v in by_commit.items()) or "—"

    rows = []
    failures: list[str] = []
    for c, m in conds:
        llm = m.get("llm") or {}
        js = llm_stats.get(c) or {}
        kinds = js.get("templates_by_kind") or {}
        status, n_nli = nli_status(m)
        ns = nli_stats.get(c) or {}
        if status == "partial":
            cats = ", ".join(
                f"{CATEGORY_NAMES.get(k, k).lower()} {v}"
                for k, v in sorted((ns.get("by_category") or {}).items())
            )
            nli_text = f"partial: {integer(n_nli)}" + (f" ({cats})" if cats else "")
        elif status == "complete":
            nli_text = f"complete: {integer(n_nli)}"
        else:
            dev = (ns.get("records_by_split") or {}).get("dev")
            nli_text = "not scored" + (f" (dev only: {integer(dev)})" if dev else "")
        for e in js.get("error_items") or []:
            failures.append(
                f"{c} {e.get('item_id')} turn {e.get('turn')} ({e.get('category')}/"
                f"{e.get('subcategory')}: {e.get('error')})"
            )
        rows.append(
            [
                c,
                provenance(c),
                integer(m.get("n_valid")),
                integer(llm.get("n_judged")),
                integer(js.get("errors")) if js else "—",
                f"{integer(kinds.get('abstain_template', 0))} / "
                f"{integer(kinds.get('escalate_template', 0))}"
                if js
                else "—",
                nli_text,
            ]
        )
    note = _study_note(
        ctx,
        "Code = commit of the run that wrote each analysed test turn (all working trees "
        "dirty; see the run manifests). Templates are counted over the latest judgment of every "
        "test turn. " + (f"Failed LLM judgments: {'; '.join(failures)}." if failures else ""),
    )
    if ctx.source("study_judgments"):
        note += f" Counts: {ctx.source('study_judgments')} and {ctx.source('study_records')}."
    return _table(
        "judge_coverage",
        [
            "Condition",
            "Code (analysed turns)",
            "Valid turns",
            "LLM judged",
            "LLM failures",
            "Templates ABSTAIN / ESCALATE",
            "NLI judge (test responses)",
        ],
        rows,
        font_size=8,
        widths=[0.55, 1.05, 0.5, 0.5, 0.5, 0.75, 1.9],
        align=["l", "l", "c", "c", "c", "c", "l"],
        note=note,
    )


def judge_confusion(ctx: ResultsContext) -> Table | None:
    a = ctx.get("study.judge_agreement")
    if not isinstance(a, dict) or not isinstance(a.get("confusion_llm_rows_nli_cols"), list):
        return None
    labels = [str(x) for x in a.get("labels") or ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]]
    matrix = a["confusion_llm_rows_nli_cols"]
    rows = []
    for lab, row in zip(labels, matrix, strict=False):
        rows.append([f"LLM {lab.lower()}", *[integer(v) for v in row], integer(sum(row))])
    totals = [sum(r[j] for r in matrix) for j in range(len(labels))]
    rows.append(["Total", *[integer(v) for v in totals], integer(sum(totals))])
    return _table(
        "judge_confusion",
        ["", *[f"NLI {lab.lower()}" for lab in labels], "Total"],
        rows,
        font_size=9,
        note=_study_note(ctx, f"{integer(a.get('n_claims'))} claims of the paired responses."),
    )


def _short_ts(ts: Any) -> str:
    """``2026-10-10T06:28:57+00:00`` -> ``10-10 06:28`` (the year is given in the note)."""
    return str(ts)[5:16].replace("T", " ") if ts else "—"


def _cond_list(conds: Any) -> str:
    """Condition list with the run C0..C5 collapsed to ``C0–C5``."""
    if not isinstance(conds, list):
        return "—"
    items = [str(c) for c in conds]
    base = ["C0", "C1", "C2", "C3", "C4", "C5"]
    if all(c in items for c in base):
        rest = [c for c in items if c not in base]
        return ", ".join(["C0–C5", *rest])
    return ", ".join(items)


def study_runs(ctx: ResultsContext) -> Table | None:
    runs = ctx.get("study_manifests") or {}
    judges = ctx.get("judge_manifests") or {}
    if not runs and not judges:
        return None
    rec = ctx.get("study_records") or {}
    by_run = rec.get("by_run") or {}
    final: dict[str, int] = {}
    for stats in (rec.get("by_condition") or {}).values():
        for rid, n in (stats.get("final_by_run") or {}).items():
            final[rid] = final.get(rid, 0) + int(n)
    rows = []
    for rid, m in sorted(runs.items()):
        git = m.get("git") or {}
        span = by_run.get(rid) or {}
        end = m.get("finished_at")
        end_text = _short_ts(end) if end else f"{_short_ts(span.get('last'))} (last record)"
        rows.append(
            [
                rid.rsplit("-", 1)[-1],
                f"study ({m.get('split')})",
                _cond_list(m.get("conditions")),
                f"{str(git.get('commit', ''))[:7]}{' (dirty)' if git.get('dirty') else ''}",
                integer(m.get("workers")),
                _short_ts(m.get("started_at")),
                end_text,
                str(m.get("status", "—")),
                integer(final.get(rid, 0)) if m.get("split") == rec.get("split") else "—",
            ]
        )
    for stem, m in sorted(judges.items()):
        git = m.get("git") or {}
        rows.append(
            [
                stem.removeprefix("manifest_judge_").split("T")[-1],
                f"{str(m.get('judge')).upper()} judge ({m.get('split')})",
                _cond_list(m.get("conditions")),
                f"{str(git.get('commit', ''))[:7]}{' (dirty)' if git.get('dirty') else ''}",
                integer(m.get("workers")),
                _short_ts(m.get("started_at")),
                _short_ts(m.get("finished_at")) if m.get("finished_at") else "not recorded",
                "finished" if m.get("finished_at") else "interrupted",
                "—",
            ]
        )
    note = (
        f"Sources: {ctx.source('study_manifests') or 'run manifests'}, "
        f"{ctx.source('judge_manifests') or 'judge manifests'} and "
        f"{ctx.source('study_records') or 'run records'}. Run = suffix of the run id (study) "
        "or time stamp of the judge manifest; models in the table above. Times in UTC "
        "(month-day, 2026). Final test records = "
        "item-turns of the analysed (latest complete) attempts written by the run. A run that "
        "was interrupted and resumed keeps status 'running' in its manifest; its span ends at "
        "its last record. The NLI judge's early-concluded test-split run left no manifest."
    )
    return _table(
        "study_runs",
        [
            "Run",
            "Kind (split)",
            "Conditions",
            "Commit",
            "Workers",
            "Started",
            "Finished",
            "Status",
            "Final test records",
        ],
        rows,
        font_size=7.5,
        widths=[0.6, 0.85, 1.15, 0.75, 0.45, 0.65, 0.8, 0.6, 0.55],
        note=note,
    )


def study_models(ctx: ResultsContext) -> Table | None:
    runs = ctx.get("study_manifests") or {}
    judges = ctx.get("judge_manifests") or {}
    if not runs and not judges:
        return None
    test_runs = [m for _r, m in sorted(runs.items()) if m.get("split") == "test"] or list(
        runs.values()
    )

    def model_row(info: dict[str, Any]) -> str:
        size = info.get("size_bytes")
        parts = [
            str(info.get("parameter_size") or ""),
            str(info.get("quantization_level") or ""),
            f"{float(size) / 1e9:.1f} GB" if _finite(size) else "",
            f"digest {str(info.get('digest'))[:12]}…" if info.get("digest") else "",
        ]
        return ", ".join(p for p in parts if p)

    rows: list[list[str]] = []
    app: dict[str, dict[str, Any]] = {}
    for m in test_runs:
        for name, info in (m.get("ollama") or {}).items():
            app.setdefault(name, info)
    roles = {}
    for m in test_runs:
        for role, name in (m.get("models") or {}).items():
            roles.setdefault(str(name), set()).add(role)
    for name, info in app.items():
        rr = " and ".join(sorted(roles.get(name, {"application"})))
        rows.append([f"Application model ({rr})", f"{name}: {model_row(info)}"])
    llm_judge = next(
        (m for _s, m in sorted(judges.items(), reverse=True) if m.get("judge") == "llm"), None
    )
    if llm_judge:
        for name, info in (llm_judge.get("ollama") or {}).items():
            cfg = llm_judge.get("llm_config") or {}
            extra = (
                f"; seed {cfg.get('seed')}, {cfg.get('claims_per_call')} claims per call, "
                f"at most {cfg.get('max_claims')} claims per response"
                if cfg
                else ""
            )
            rows.append(["LLM judge", f"{name}: {model_row(info)}{extra}"])
    nli = next(
        (m for _s, m in sorted(judges.items(), reverse=True) if m.get("judge") == "nli"), None
    )
    th = ctx.get("study.nli_thresholds") or {}
    if nli and nli.get("nli_model"):
        rows.append(
            [
                "NLI judge",
                f"{nli['nli_model']}; θ = {num(th.get('theta'))}, θc = {num(th.get('theta_c'))} "
                f"({th.get('source', 'calibration')})",
            ]
        )
    verifier = ctx.get("calibration.v2.verifier_model")
    if verifier:
        rows.append(["Grounding verifier (V2)", str(verifier)])
    seeds = next((m.get("seeds") for m in test_runs if m.get("seeds")), None)
    boot = ctx.get("study.bootstrap") or {}
    if seeds or boot:
        parts = []
        if seeds:
            parts.append(
                f"LLM seed {seeds.get('llm_seed')}, gate temperature "
                f"{seeds.get('temperature_gates')}, generator temperature "
                f"{seeds.get('temperature_generator')}"
            )
        if boot:
            parts.append(
                f"bootstrap {integer(boot.get('n_boot'))} resamples, seed {boot.get('seed')}"
            )
        rows.append(["Seeds", "; ".join(parts)])
    ds = next((m.get("dataset") for m in test_runs if m.get("dataset")), None)
    if ds:
        rows.append(
            [
                "Dataset",
                f"{Path(str(ds.get('path', ''))).name}, SHA-256 {str(ds.get('sha256'))[:12]}…, "
                f"{integer(ds.get('n_items_selected'))} test items",
            ]
        )
    hw = next((m.get("hardware") for m in test_runs if m.get("hardware")), None)
    if hw:
        pk = next((m.get("packages") for m in test_runs if m.get("packages")), None) or {}
        rows.append(
            [
                "Hardware and software",
                f"{hw.get('cpu')}, {hw.get('cpu_count')} CPU cores, {num(hw.get('ram_gb'), 0)} GB "
                f"RAM, {hw.get('platform')}; Python {hw.get('python')}"
                + (f", torch {pk.get('torch')}" if pk.get("torch") else ""),
            ]
        )
    commits = sorted(
        {
            f"{str((m.get('git') or {}).get('commit', ''))[:7]}"
            + (" (dirty)" if (m.get("git") or {}).get("dirty") else "")
            for m in test_runs
        }
    )
    if commits:
        rows.append(["Study code (test-split runs)", ", ".join(commits)])
    return _table(
        "study_models",
        ["Item", "Value"],
        rows,
        font_size=8.5,
        widths=[1.6, 4.1],
        align=["l", "l"],
        note=f"Sources: {ctx.source('study_manifests') or 'run manifests'}, "
        f"{ctx.source('judge_manifests') or 'judge manifests'}, "
        f"{ctx.source('calibration') or 'calibration'} and {ctx.source('study') or 'summary'}.",
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
    # included_names=set(): read the table even if a content file also includes the page
    blocks = parse_file(path, ParseContext(content_dir=path.parent, included_names=set()))
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
    "rabbits_before_after": rabbits_before_after,
    "adversarial": adversarial,
    "adversarial_before_after": adversarial_before_after,
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
    "claim_profile": claim_profile,
    "category_safety": category_safety,
    "category_tests": category_tests,
    "mcnemar_sensitivity": mcnemar_sensitivity,
    "sensitivity_rates": sensitivity_rates,
    "routing_per_class": routing_per_class,
    "judge_coverage": judge_coverage,
    "judge_confusion": judge_confusion,
    "study_runs": study_runs,
    "study_models": study_models,
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

#!/usr/bin/env python3
"""
Render result numbers into the documentation from the files the experiments write.

Nothing in the rendered blocks is typed by hand. Every number is read from a
result or measurement file, and the file is named next to the number. A file
that does not exist yet, or a run with no valid data, renders as *pending*.
The script uses the standard library only, so it runs before the evaluation
extras are installed, and it is idempotent: re-running it on unchanged inputs
produces byte-identical files (no timestamps are added).

Marker blocks it fills (HTML comments, invisible on GitHub):

* ``RESULTS:START`` / ``RESULTS:END`` in README.md: the results summary tables
  (hallucination study, retrieval, model selection, PII redaction, corpus,
  operations, latency, tests). Benchmark reports count only when produced in
  live mode; while the current model-selection run has no valid turns, the
  earlier round is shown with an explicit "superseded" label.
* ``SCREENSHOTS:START`` / ``SCREENSHOTS:END`` in README.md: the UI screenshot
  gallery. Images under ``report/screenshots/`` are embedded once they exist.
* ``RESULT:<spec>`` ... ``/RESULT`` (single line, any of the managed files):
  one inline value. Spec grammar::

      rate:<COND>:<dotted.path>        proportion in summary.json per_condition[COND]
                                       e.g. rate:C5:llm.hallucination, rate:C5:emergency.recall_system
      interval:<COND>:<dotted.path>    estimate with CI, e.g. interval:C5:llm.unsupported_claim_ratio
      routing:<COND>:<strict|lenient|macro_f1>
      latency:<COND>                   cache-imputed p50 / p95 per turn
      degraded:<COND>                  degraded (excluded) turns
      test:<T>:<C>:<outcome>:<judge>   paired exact McNemar row (outcome: hallucination|unsafe;
                                       judge: llm|nli), e.g. test:C5:C1:hallucination:llm
      cochran:<outcome>:<judge>        Cochran's Q across C0-C5
      kappa:<3class|binary|response>   LLM-judge vs NLI-judge agreement
      human:<status|human_human|A1_vs_llm|A2_vs_llm|A1_vs_nli|A2_vs_nli|consensus_vs_llm|...>
      split                            split / item / turn counts of the study
      models                           application and judge models of the study
      retrieval:<mode>:<metric>        mode: dense|sparse|hybrid|hybrid+rerank;
                                       metric: recall@5|recall@20|mrr|ndcg@5
      reranker                         reranker-ablation one-liner (MRR per reranker)
      model_selection                  model-selection one-liner
      pii:<development|blind_v2|blind_v2_posthoc>:<strict|overlap|clinical>
      corpus                           corpus size one-liner
      smoke                            live smoke-test result
      aq | rabbits | adversarial | load   live benchmark one-liners (pending unless mode == live)
      perf                             before/after latency one-liner
      tests                            test-suite one-liner

* ``EVIDENCE-CHECK:START`` / ``EVIDENCE-CHECK:END`` in docs/review4_5_checklist.md:
  every repo path written in backticks in the checklist is checked, and the
  paths that do not exist yet are listed.
* ``ADL:<regex>`` ... ``/ADL`` (single line, CHANGELOG.md): the ADL ids whose
  heading in docs/architecture_decision_log.md matches the regex
  (case-insensitive). Unmatched markers render as "ADL entry pending".

Usage::

    python scripts/render_docs_results.py           # rewrite the managed files in place
    python scripts/render_docs_results.py --check   # exit 1 if a file would change, write nothing
    python scripts/render_docs_results.py --root <repo checkout> --files README.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
MANAGED_FILES: tuple[str, ...] = (
    "README.md",
    "docs/DEMO_SCRIPT.md",
    "docs/review4_5_checklist.md",
    "CHANGELOG.md",
)

# Input locations (repo-relative). Kept in one place so the docs can name them.
P_SUMMARY = "evaluation/results/summary.json"
P_CALIBRATION = "evaluation/results/calibration.json"
P_HUMAN = "evaluation/results/human_agreement.json"
P_JUDGE_MANIFESTS = "evaluation/results/judgments/manifest_judge_*.json"
P_TABLES = "evaluation/results/tables/"
P_ERRORS = "evaluation/results/error_analysis.md"
P_RETRIEVAL = "evaluation/results/retrieval/metrics.json"
P_RERANKER_DIR = "evaluation/results/retrieval/reranker_ablation"
P_MODEL_SELECTION = "evaluation/results/model_selection/summary.json"
P_MODEL_SELECTION_EARLIER = "evaluation/results/model_selection_round1_prefix/summary.json"
P_PII = "reports/privacy_redaction_metrics.json"
P_CORPUS = "reports/corpus_stats.json"
P_SMOKE = "reports/smoke_test_latest.json"
P_AQ = "reports/medbridge_aq_benchmark_report.json"
P_RABBITS = "reports/rabbits_benchmark_report.json"
P_ADVERSARIAL = "reports/adversarial_live_report.json"
P_LOAD = "reports/performance/load_test.json"
P_LATENCY_TABLE = "reports/performance/latency_table.csv"
P_PERF_DIR = "reports/performance"
P_JUNIT = ("reports/tests/*.xml", "reports/junit*.xml")
P_COVERAGE = ("reports/tests/coverage.json", "reports/coverage.json")
P_TEST_SUMMARY = ("reports/tests/test_summary.json", "reports/test_summary.json")
P_BASELINE_PYTEST = "reports/baseline/pytest_baseline.txt"
P_ADL = "docs/architecture_decision_log.md"
FIGURES_DIR = "reports/figures"
SCREENSHOTS_DIR = "report/screenshots"

CONDITION_ORDER: tuple[str, ...] = ("C0", "C1", "C1G", "C2", "C3", "C4", "C5", "V1", "V2", "V12")
CONDITION_NAMES: dict[str, str] = {
    "C0": "Direct LLM",
    "C1": "Vanilla RAG",
    "C1G": "RAG + production prompt (no gates)",
    "C2": "RAG + prompt-only safety",
    "C3": "Context Gate only",
    "C4": "Evidence Gate only",
    "C5": "MedBridge (dual gate)",
    "V1": "C5 + calibrated Evidence Gate",
    "V2": "C5 + Gate 3 grounding verifier",
    "V12": "C5 + V1 + V2",
}
GATED: frozenset[str] = frozenset({"C3", "C4", "C5", "V1", "V2", "V12"})
FAMILY_ORDER: tuple[str, ...] = ("c5_vs_baselines", "c5_vs_reference", "variants_vs_c5")
FAMILY_NAMES: dict[str, str] = {
    "c5_vs_baselines": "C5 vs baselines/ablations",
    "c5_vs_reference": "C5 vs C1G reference",
    "variants_vs_c5": "Variants vs C5",
}
RETRIEVAL_MODES: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid+rerank")
RETRIEVAL_METRICS: tuple[str, ...] = ("recall@5", "recall@20", "mrr", "ndcg@5")
RESULT_FIGURES: tuple[tuple[str, str], ...] = (
    ("hallucination_rate_by_condition", "Hallucination rate per condition (95% Wilson CI)"),
    ("unsafe_answer_rate_unanswerable", "Unsafe-answer rate on unanswerable items"),
    ("ablation_waterfall", "Ablation waterfall"),
    ("risk_coverage", "Risk-coverage"),
    ("confusion_matrix_C5", "Routing confusion matrix, C5"),
    ("claim_support_stacked", "Claim support per condition"),
    ("judge_agreement", "LLM judge vs NLI judge"),
    ("latency_by_condition", "Latency per condition"),
    ("latency_before_after", "Latency before vs after optimisation"),
)
SCREENSHOTS: tuple[tuple[str, str], ...] = (
    ("welcome.png", "Welcome screen with suggested prompts"),
    ("answer_citations.png", "ANSWER with inline citations and the source drawer"),
    ("soft_ask.png", "SOFT-ASK: the Context Gate asks for missing facts"),
    ("escalate.png", "ESCALATE: deterministic emergency card"),
    ("abstain.png", "ABSTAIN: out-of-scope question, template reply"),
    ("why_panel.png", '"Why this answer?" panel: Gate 1, evidence, Gate 2, latency'),
    ("bp_trend.png", "BP log and trend chart with ACC/AHA bands"),
    ("dashboard.png", "Admin / evaluation dashboard"),
    ("dark_mode.png", "Dark mode"),
)

PENDING = "*pending*"


# ----------------------------------------------------------------------
# Context: loaded inputs
# ----------------------------------------------------------------------


@dataclass
class Context:
    """All inputs, loaded once. Every attribute may be None (file missing or unreadable)."""

    root: Path
    warnings: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def json(self, rel: str) -> Any:
        """Parsed JSON of a repo-relative file, or None (missing / invalid, with a warning)."""
        if rel in self._cache:
            return self._cache[rel]
        value: Any = None
        path = self.path(rel)
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.warnings.append(f"could not read {rel}: {exc}")
        self._cache[rel] = value
        return value

    def glob(self, pattern: str) -> list[str]:
        """Repo-relative paths matching a glob, sorted (deterministic output)."""
        hits = glob.glob(str(self.root / pattern))
        return sorted(Path(h).relative_to(self.root).as_posix() for h in hits)

    def first_existing(self, candidates: Iterable[str]) -> str | None:
        for rel in candidates:
            if self.path(rel).is_file():
                return rel
        return None


# ----------------------------------------------------------------------
# Formatting helpers (None / NaN safe)
# ----------------------------------------------------------------------


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def dig(obj: Any, path: str | Sequence[str]) -> Any:
    """``obj[a][b][c]`` for ``"a.b.c"``; None when any step is missing."""
    keys = path.split(".") if isinstance(path, str) else list(path)
    cur = obj
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and key.isdigit() and int(key) < len(cur):
            cur = cur[int(key)]
        else:
            return None
    return cur


def as_dict(value: Any) -> dict[str, Any]:
    """``value`` when it is a dict, else an empty dict (malformed inputs never crash)."""
    return value if isinstance(value, dict) else {}


def pct(value: Any, digits: int = 1) -> str:
    f = _num(value)
    return "—" if f is None else f"{100 * f:.{digits}f}%"


def fixed(value: Any, digits: int = 3) -> str:
    f = _num(value)
    return "—" if f is None else f"{f:.{digits}f}"


def is_proportion(d: Any) -> bool:
    return isinstance(d, dict) and "rate" in d and "n" in d


def prop_cell(d: Any) -> str:
    """Table cell for a Wilson proportion: ``12.3% [8.1, 17.9] (12/98)``."""
    if not is_proportion(d) or not d.get("n"):
        return "—"
    rate, low, high = _num(d.get("rate")), _num(d.get("low")), _num(d.get("high"))
    if rate is None:
        return "—"
    ci = f" [{100 * low:.1f}, {100 * high:.1f}]" if low is not None and high is not None else ""
    return f"{100 * rate:.1f}%{ci} ({d.get('k', '?')}/{d.get('n')})"


def prop_text(d: Any, unit: str = "") -> str:
    """Inline text for a proportion: ``12.3% (95% CI 8.1-17.9%; 12/98 turns)``."""
    if not is_proportion(d) or not d.get("n"):
        return ""
    rate, low, high = _num(d.get("rate")), _num(d.get("low")), _num(d.get("high"))
    if rate is None:
        return ""
    unit_text = f" {unit}" if unit else ""
    ci = (
        f"95% CI {100 * low:.1f}–{100 * high:.1f}%; "
        if low is not None and high is not None
        else ""
    )
    return f"{100 * rate:.1f}% ({ci}{d.get('k', '?')}/{d.get('n')}{unit_text})"


def interval_cell(d: Any, digits: int = 3, as_pct: bool = False) -> str:
    if not isinstance(d, dict):
        return "—"
    est, low, high = _num(d.get("estimate")), _num(d.get("low")), _num(d.get("high"))
    if est is None:
        return "—"
    if as_pct:
        ci = f" [{100 * low:.1f}, {100 * high:.1f}]" if low is not None and high is not None else ""
        return f"{100 * est:.1f}%{ci}"
    ci = f" [{low:.{digits}f}, {high:.{digits}f}]" if low is not None and high is not None else ""
    return f"{est:.{digits}f}{ci}"


def pvalue(value: Any) -> str:
    f = _num(value)
    if f is None:
        return "—"
    return "< 0.001" if f < 0.001 else f"{f:.3f}"


def p_eq(value: Any) -> str:
    """``p = 0.031`` / ``p < 0.001`` for running text."""
    text = pvalue(value)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def rd_cell(d: Any) -> str:
    """Risk difference in percentage points with CI."""
    if not isinstance(d, dict):
        return "—"
    est, low, high = _num(d.get("estimate")), _num(d.get("low")), _num(d.get("high"))
    if est is None:
        return "—"
    ci = f" [{100 * low:+.1f}, {100 * high:+.1f}]" if low is not None and high is not None else ""
    return f"{100 * est:+.1f} pp{ci}"


def or_cell(d: Any) -> str:
    if not isinstance(d, dict):
        return "—"
    est, low, high = _num(d.get("estimate")), _num(d.get("low")), _num(d.get("high"))
    if est is None:
        return "—"
    ci = f" [{low:.2f}, {high:.2f}]" if low is not None and high is not None else ""
    tag = " (Haldane)" if d.get("haldane") else ""
    return f"{est:.2f}{ci}{tag}"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: str | None = None) -> str:
    """GitHub markdown table. ``align``: one char per column, ``l`` or ``r``."""

    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    align = align or "l" * len(headers)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def source(*paths: str) -> str:
    return "Source: " + ", ".join(f"`{p}`" for p in paths if p) + "."


def pending_note(ctx: Context, what: str, how: str) -> str:
    ctx.pending.append(what)
    return f"{PENDING} — {what}. {how}"


def date_of(value: Any) -> str:
    text = str(value or "")
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else (text or "unknown date")


# ----------------------------------------------------------------------
# Hallucination study (evaluation/results/summary.json)
# ----------------------------------------------------------------------


def summary_conditions(summary: dict[str, Any]) -> list[str]:
    per = summary.get("per_condition") or {}
    ordered = [c for c in CONDITION_ORDER if c in per]
    return ordered + sorted(c for c in per if c not in ordered)


def condition_label(summary: dict[str, Any], cond: str) -> str:
    name = dig(summary, f"per_condition.{cond}.name") or CONDITION_NAMES.get(cond, cond)
    check = dig(summary, f"calibration_check.{cond}")
    flag = " (UNCALIBRATED)" if isinstance(check, dict) and check.get("calibrated") is False else ""
    return f"{cond} {name}{flag}"


def routing_of(m: dict[str, Any], cond: str) -> tuple[dict[str, Any] | None, str]:
    """System routing for gated conditions, judge-behaviour routing otherwise."""
    if cond in GATED and isinstance(m.get("routing_system"), dict):
        return m["routing_system"], ""
    if isinstance(m.get("routing_behaviour"), dict):
        return m["routing_behaviour"], "†"
    return None, ""


def study_models(ctx: Context, summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    app: set[str] = set()
    for run in summary.get("runs") or []:
        models = run.get("models") if isinstance(run, dict) else None
        if isinstance(models, dict):
            app.update(f"{role}: {name}" for role, name in models.items() if name)
    judges: set[str] = set()
    for rel in ctx.glob(P_JUDGE_MANIFESTS):
        manifest = ctx.json(rel)
        if isinstance(manifest, dict):
            for key in ("judge_model", "nli_model"):
                if manifest.get(key):
                    judges.add(f"{key.replace('_model', '')}: {manifest[key]}")
    return sorted(app), sorted(judges)


def find_test(
    summary: dict[str, Any], treatment: str, control: str, outcome: str, judge: str
) -> dict[str, Any] | None:
    for row in dig(summary, "tests.mcnemar") or []:
        if (
            isinstance(row, dict)
            and row.get("treatment") == treatment
            and row.get("control") == control
            and row.get("outcome") == outcome
            and row.get("judge") == judge
            and row.get("unit", "turn") == "turn"
        ):
            return row
    return None


def test_rows(rows: Iterable[dict[str, Any]]) -> list[list[str]]:
    out = []
    for r in rows:
        out.append(
            [
                FAMILY_NAMES.get(str(r.get("family")), str(r.get("family", "—"))),
                f"{r.get('treatment')} vs {r.get('control')}",
                str(r.get("outcome", "—")),
                str(r.get("n_pairs", "—")),
                pct(r.get("p_treatment")),
                pct(r.get("p_control")),
                rd_cell(r.get("risk_difference")),
                or_cell(r.get("odds_ratio")),
                pvalue(r.get("pvalue")),
                pvalue(r.get("pvalue_holm")),
            ]
        )
    return out


def sorted_tests(summary: dict[str, Any], judge: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in dig(summary, "tests.mcnemar") or []
        if isinstance(r, dict) and r.get("judge") == judge and r.get("unit", "turn") == "turn"
    ]

    def key(r: dict[str, Any]) -> tuple[int, str, int, int]:
        fam = str(r.get("family"))
        fam_i = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        ctrl = str(r.get("control"))
        treat = str(r.get("treatment"))
        order = CONDITION_ORDER.index(ctrl) if ctrl in CONDITION_ORDER else 99
        torder = CONDITION_ORDER.index(treat) if treat in CONDITION_ORDER else 99
        return (fam_i, str(r.get("outcome")), torder, order)

    return sorted(rows, key=key)


TEST_HEADERS = (
    "Family",
    "Comparison",
    "Outcome",
    "Pairs",
    "Treatment",
    "Control",
    "Risk difference [95% CI]",
    "Conditional OR [95% CI]",
    "Exact McNemar p",
    "Holm p",
)


def section_hallucination(ctx: Context) -> list[str]:
    out = ["### Hallucination study (MedBridge-HalluBench v1, conditions C0–C5, V1, V2)", ""]
    summary = ctx.json(P_SUMMARY)
    if not isinstance(summary, dict) or not isinstance(summary.get("per_condition"), dict):
        out.append(
            pending_note(
                ctx,
                "the hallucination study has not produced "
                f"`{P_SUMMARY}` yet, so no hallucination rates, unsafe-answer rates or p-values "
                "are reported",
                "Reproduce with `scripts/run_study.sh` or the steps in "
                "[`evaluation/README.md`](evaluation/README.md), then re-run this script.",
            )
        )
        return [*out, ""]

    app, judges = study_models(ctx, summary)
    nli = as_dict(summary.get("nli_thresholds"))
    degraded = as_dict(summary.get("degraded"))
    provenance = [
        f"Split: **{summary.get('split', '?')}** (dev split used only for calibration); "
        f"unit: {summary.get('unit', 'item-turn')}; degraded turns excluded from headline numbers.",
        f"Application model(s): {', '.join(app) if app else 'not recorded'}; "
        f"judges: {', '.join(judges) if judges else 'see judgment manifests'}; "
        f"NLI thresholds: {nli.get('source', 'unknown')}.",
    ]
    if degraded.get("exceeded"):
        provenance.append(
            "**Warning:** the degraded-turn limit was exceeded for "
            f"{', '.join(map(str, degraded.get('exceeded') or []))}"
            + (" (reported with --allow-degraded)." if degraded.get("override") else ".")
        )
    out += ["- " + p for p in provenance] + [""]

    headers = (
        "Condition",
        "Valid turns",
        "Hallucination (LLM judge)",
        "Hallucination (NLI judge)",
        "Unsupported-claim ratio (LLM)",
        "Unsafe answers, unanswerable (LLM)",
        "Coverage, answerable",
        "Routing accuracy, lenient",
        "Emergency recall",
    )
    rows = []
    for cond in summary_conditions(summary):
        m = summary["per_condition"][cond]
        if not isinstance(m, dict):
            continue
        routing, mark = routing_of(m, cond)
        lenient = prop_cell(routing.get("accuracy_lenient")) if routing else "—"
        rows.append(
            [
                condition_label(summary, cond),
                f"{m.get('n_valid', '—')}/{m.get('n_turns', '—')}",
                prop_cell(dig(m, "llm.hallucination")),
                prop_cell(dig(m, "nli.hallucination")),
                interval_cell(dig(m, "llm.unsupported_claim_ratio"), as_pct=True),
                prop_cell(dig(m, "llm.unsafe_answer")),
                prop_cell(m.get("coverage_answerable")),
                (lenient + mark) if lenient != "—" else "—",
                # C0 has no fast-path by definition, so only its behaviour can escalate
                prop_cell(dig(m, "emergency.recall_behaviour"))
                if cond == "C0"
                else prop_cell(dig(m, "emergency.recall_system")),
            ]
        )
    out += [
        table(headers, rows, "lrrrrrrrr"),
        "",
        "Cells: rate [95% Wilson CI] (events/turns); unsupported-claim ratio with a 10k cluster-"
        "bootstrap CI. † routing from the LLM judge's behaviour label (conditions without gates); "
        "C0 emergency recall is also from the behaviour label, since C0 has no fast-path. "
        "Definitions of every metric are stored in `summary.json` → `definitions`.",
        "",
    ]

    llm_tests = sorted_tests(summary, "llm")
    if llm_tests:
        out += [
            "Paired tests (LLM judge; exact McNemar on item-turn pairs, Holm correction within "
            "each family and outcome; risk difference = treatment − control):",
            "",
            table(TEST_HEADERS, test_rows(llm_tests), "lllrrrrrrr"),
            "",
        ]
    nli_tests = sorted_tests(summary, "nli")
    if nli_tests:
        out += [
            "<details><summary>Paired tests with the NLI judge</summary>",
            "",
            table(TEST_HEADERS, test_rows(nli_tests), "lllrrrrrrr"),
            "",
            "</details>",
            "",
        ]
    cochran = [r for r in dig(summary, "tests.cochran_q") or [] if isinstance(r, dict)]
    if cochran:
        out.append(
            "Cochran's Q across "
            + ", ".join(map(str, cochran[0].get("conditions") or []))
            + ": "
            + "; ".join(
                f"{r.get('outcome')} ({r.get('judge')}) Q = {fixed(r.get('q'), 2)}, "
                f"df = {r.get('df')}, {p_eq(r.get('pvalue'))}, n = {r.get('n')}"
                for r in cochran
            )
            + "."
        )
        out.append("")
    agreement = summary.get("judge_agreement")
    if isinstance(agreement, dict):
        out.append(
            "Judge agreement (Cohen's κ, LLM judge vs NLI judge): "
            f"{kappa_text(agreement.get('kappa_3class'))} three-class and "
            f"{kappa_text(agreement.get('kappa_binary'))} binary over "
            f"{agreement.get('n_claims', '?')} claims; "
            f"{kappa_text(agreement.get('kappa_response_hallucination'))} on the response-level "
            f"hallucination label over {agreement.get('n_responses', '?')} responses."
        )
        out.append("")
    out.append(human_sentence(ctx))
    out.append("")
    per_deg = degraded.get("per_condition") if isinstance(degraded, dict) else None
    deg_parts = [
        f"{c} {d.get('n_degraded', 0)}/{d.get('n_turns', 0)}"
        for c, d in sorted(
            (per_deg or {}).items(),
            key=lambda kv: CONDITION_ORDER.index(kv[0]) if kv[0] in CONDITION_ORDER else 99,
        )
        if isinstance(d, dict)
    ]
    if deg_parts:
        out += ["Degraded (excluded) turns: " + ", ".join(deg_parts) + ".", ""]
    out += figure_gallery(ctx)
    out.append(source(P_SUMMARY) + f" Full tables: `{P_TABLES}`; error analysis: `{P_ERRORS}`.")
    out.append("")
    return out


def kappa_text(k: Any) -> str:
    if not isinstance(k, dict) or _num(k.get("kappa")) is None:
        return "—"
    low, high = _num(k.get("low")), _num(k.get("high"))
    ci = f" [{low:.2f}, {high:.2f}]" if low is not None and high is not None else ""
    return f"κ = {float(k['kappa']):.2f}{ci}"


def human_sentence(ctx: Context) -> str:
    human = ctx.json(P_HUMAN)
    if not isinstance(human, dict):
        return (
            "Human annotation (50-item sheet): *pending* — "
            f"`{P_HUMAN}` not generated yet (`python -m evaluation.human_annotation --score`)."
        )
    status = human.get("status", "unknown")
    if status == "not_filled":
        return (
            "Human annotation (50-item sheet): *pending* — the sheet has not been labelled yet "
            f"(`{P_HUMAN}` status `not_filled`)."
        )
    parts = []
    for key in ("human_human", "A1_vs_llm", "A2_vs_llm", "consensus_vs_llm", "consensus_vs_nli"):
        block = human.get(key)
        if isinstance(block, dict) and block.get("n"):
            parts.append(
                f"{key.replace('_', ' ')} {kappa_text(block.get('kappa_3class'))} (n = {block['n']})"
            )
    joined = "; ".join(parts) if parts else "no agreement blocks"
    return f"Human annotation (status `{status}`): {joined}. {source(P_HUMAN)}"


def figure_gallery(ctx: Context) -> list[str]:
    present = [
        (name, caption)
        for name, caption in RESULT_FIGURES
        if ctx.exists(f"{FIGURES_DIR}/{name}.png")
    ]
    if not present:
        return []
    out = ["<details><summary>Result figures</summary>", ""]
    for name, caption in present:
        out += [f"**{caption}**", "", f"![{caption}]({FIGURES_DIR}/{name}.png)", ""]
    out += ["</details>", ""]
    return out


# ----------------------------------------------------------------------
# Retrieval, reranker ablation, model selection
# ----------------------------------------------------------------------


def section_retrieval(ctx: Context) -> list[str]:
    out = ["### Retrieval evaluation (70 labelled queries, 27-document corpus)", ""]
    data = ctx.json(P_RETRIEVAL)
    modes = data.get("modes") if isinstance(data, dict) else None
    if not isinstance(modes, dict):
        out.append(
            pending_note(
                ctx,
                f"`{P_RETRIEVAL}` not generated yet",
                "Run `python -m evaluation.retrieval_eval`.",
            )
        )
        return [*out, ""]
    rows = []
    for mode in [m for m in RETRIEVAL_MODES if m in modes] + sorted(
        m for m in modes if m not in RETRIEVAL_MODES
    ):
        block = modes[mode]
        rows.append(
            [mode]
            + [interval_cell(block.get(metric)) for metric in RETRIEVAL_METRICS]
            + [str(block.get("n", "—"))]
        )
    out += [
        table(("Mode", *RETRIEVAL_METRICS, "Queries"), rows, "lrrrrr"),
        "",
        "Mean with 95% bootstrap CI over queries. " + source(P_RETRIEVAL),
        "",
    ]
    reranker = reranker_rows(ctx)
    if reranker:
        rows2, overlap = reranker
        out += [
            "Reranker ablation (top-20 hybrid candidates → top-5):",
            "",
            table(("Reranker", *RETRIEVAL_METRICS), rows2, "lrrrr"),
            "",
        ]
        if overlap is not None:
            out.append(
                "Computed check: the 95% CIs of every reranker "
                + ("overlap" if overlap else "do **not** all overlap")
                + " with RRF fusion alone on every metric. "
                + source(f"{P_RERANKER_DIR}/*/metrics.json", P_RETRIEVAL)
            )
            out.append("")
    return out


def reranker_rows(ctx: Context) -> tuple[list[list[str]], bool | None] | None:
    base = ctx.json(P_RETRIEVAL)
    variants = ctx.glob(f"{P_RERANKER_DIR}/*/metrics.json")
    if not isinstance(base, dict) or not variants:
        return None
    modes = base.get("modes") or {}
    entries: list[tuple[str, dict[str, Any]]] = []
    if "hybrid" in modes:
        entries.append(("none (hybrid RRF k=60 only)", modes["hybrid"]))
    if "hybrid+rerank" in modes:
        default = (base.get("candidates") or {}) if isinstance(base.get("candidates"), dict) else {}
        label = default.get("reranker") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        entries.append((f"{label} (default)", modes["hybrid+rerank"]))
    for rel in variants:
        data = ctx.json(rel)
        block = dig(data, "modes.hybrid+rerank")
        if isinstance(block, dict):
            name = Path(rel).parent.name.replace("_", "/", 1)
            entries.append((name, block))
    rows = [[label] + [interval_cell(b.get(m)) for m in RETRIEVAL_METRICS] for label, b in entries]
    overlap: bool | None = None
    if entries and entries[0][0].startswith("none"):
        ref = entries[0][1]
        overlap = True
        for _, block in entries[1:]:
            for metric in RETRIEVAL_METRICS:
                a, b = ref.get(metric) or {}, block.get(metric) or {}
                lo_a, hi_a, lo_b, hi_b = (
                    _num(a.get("low")),
                    _num(a.get("high")),
                    _num(b.get("low")),
                    _num(b.get("high")),
                )
                if None in (lo_a, hi_a, lo_b, hi_b):
                    continue
                if hi_a < lo_b or hi_b < lo_a:  # type: ignore[operator]
                    overlap = False
    return rows, overlap


def model_selection_valid(ms: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(ms, dict):
        return []
    results = ms.get("results") or {}
    order = ms.get("models") or list(results)
    return [
        (name, results[name])
        for name in order
        if isinstance(results.get(name), dict) and (results[name].get("n_valid") or 0) > 0
    ]


def model_selection_table(ms: dict[str, Any], valid: list[tuple[str, dict[str, Any]]]) -> str:
    rows = []
    for name, r in valid:
        ollama = r.get("ollama") or {}
        routing = r.get("routing") or {}
        roles = r.get("roles") or {}
        json_rates = [
            _num(dig(role, "valid_first_attempt.rate"))
            for role in roles.values()
            if isinstance(role, dict)
        ]
        json_rates_f = [x for x in json_rates if x is not None]
        size = _num(ollama.get("size_bytes"))
        p50, p95 = _num(r.get("latency_turn_p50_ms")), _num(r.get("latency_turn_p95_ms"))
        rows.append(
            [
                name,
                str(ollama.get("parameter_size", "—")),
                str(ollama.get("quantization_level", "—")),
                f"{size / 1e9:.1f}" if size else "—",
                f"{r.get('n_valid')}/{r.get('n_turns')}",
                prop_cell(routing.get("accuracy_strict")),
                prop_cell(routing.get("accuracy_lenient")),
                fixed(routing.get("macro_f1"), 2),
                pct(min(json_rates_f)) if json_rates_f else "—",
                fixed(p50 / 1000, 1) if p50 is not None else "—",
                fixed(p95 / 1000, 1) if p95 is not None else "—",
            ]
        )
    return table(
        (
            "Model",
            "Params",
            "Quant",
            "Size GB",
            "Valid turns",
            "Routing strict",
            "Routing lenient",
            "Macro-F1",
            "Valid JSON 1st try (worst role)",
            "Turn p50 s",
            "Turn p95 s",
        ),
        rows,
        "lllrrrrrrrr",
    )


def run_commits(ctx: Context, results_dir: str) -> list[str]:
    """Short git commits recorded in the run manifests of a model-selection directory."""
    commits: set[str] = set()
    for rel in ctx.glob(f"{results_dir}/runs/*/manifest_*.json"):
        commit = dig(ctx.json(rel), "git.commit")
        if isinstance(commit, str) and commit:
            commits.add(commit[:7])
    return sorted(commits)


def earlier_model_selection(ctx: Context) -> tuple[str, dict[str, Any], list[str]] | None:
    """The earlier, superseded round (shown only while the current run has no valid data)."""
    ms = ctx.json(P_MODEL_SELECTION_EARLIER)
    if not model_selection_valid(ms):
        return None
    return (
        P_MODEL_SELECTION_EARLIER,
        ms,
        run_commits(ctx, Path(P_MODEL_SELECTION_EARLIER).parent.as_posix()),
    )


def section_model_selection(ctx: Context) -> list[str]:
    out = ["### Model selection (application model, condition C5, dev split)", ""]
    ms = ctx.json(P_MODEL_SELECTION)
    valid = model_selection_valid(ms)
    if not valid:
        detail = ""
        if isinstance(ms, dict):
            errors = {
                name: (r or {}).get("n_errors") for name, r in (ms.get("results") or {}).items()
            }
            detail = (
                f" (latest run {date_of(ms.get('generated_at'))}: no valid turns; errors per model "
                + ", ".join(f"{k} {v}" for k, v in errors.items())
                + ")"
            )
        out += [
            pending_note(
                ctx,
                f"current model-selection results are not available{detail}",
                "Re-run `python -m evaluation.model_selection` (see evaluation/README.md).",
            ),
            "",
        ]
        earlier = earlier_model_selection(ctx)
        if earlier:
            rel, ems, commits = earlier
            out += [
                "Earlier round, kept for provenance of the study's model choice and superseded "
                "once the re-run above finishes (recorded at commit "
                f"{', '.join(commits) or 'unknown'}, before the later prompt revisions):",
                "",
                model_selection_table(ems, model_selection_valid(ems)),
                "",
                f"Run of {date_of(ems.get('generated_at'))}, split `{ems.get('split', '?')}`. "
                + source(rel),
                "",
            ]
        return out
    missing = [name for name in (ms.get("models") or []) if name not in {n for n, _ in valid}]
    out += [model_selection_table(ms, valid), ""]
    if missing:
        out.append(f"No valid turns yet for: {', '.join(missing)} (pending).")
        out.append("")
    out.append(
        f"Run of {date_of(ms.get('generated_at'))}, split `{ms.get('split', '?')}`. "
        + source(P_MODEL_SELECTION)
    )
    out.append("")
    return out


# ----------------------------------------------------------------------
# Privacy, corpus, operations, latency, tests
# ----------------------------------------------------------------------


def pii_block(pii: Any, which: str) -> tuple[str, dict[str, Any]] | None:
    """(label, results.all) for development | blind_v2 | blind_v2_posthoc."""
    if not isinstance(pii, dict):
        return None
    if which == "development":
        block = dig(pii, "datasets.development.results.all")
        label = "Development set (written with the detector; in-sample)"
    elif which == "blind_v2":
        block = dig(pii, "datasets.blind_v2.results.all")
        label = "Blind held-out v2 (frozen detector; primary estimate)"
    elif which == "blind_v2_posthoc":
        block = dig(pii, "datasets.blind_v2.post_hoc.results.all")
        label = "Blind v2 after post-hoc fixes (optimistic, not independent)"
    else:
        return None
    return (label, block) if isinstance(block, dict) else None


def prf(d: Any) -> str:
    if not isinstance(d, dict):
        return "—"
    return f"{fixed(d.get('precision'))} / {fixed(d.get('recall'))} / {fixed(d.get('f1'))}"


def section_privacy(ctx: Context) -> list[str]:
    out = ["### PII/PHI redaction (span level, synthetic test sets)", ""]
    pii = ctx.json(P_PII)
    blocks = [
        b for b in (pii_block(pii, w) for w in ("development", "blind_v2", "blind_v2_posthoc")) if b
    ]
    if not blocks:
        out.append(
            pending_note(ctx, f"`{P_PII}` not generated yet", "Run `python scripts/pii_eval.py`.")
        )
        return [*out, ""]
    rows = []
    for label, r in blocks:
        rows.append(
            [
                label,
                f"{r.get('n_examples', '—')} ({r.get('n_positive_examples', '—')}/{r.get('n_negative_examples', '—')})",
                str(dig(r, "overall_strict.support") or "—"),
                prf(r.get("overall_strict")),
                prf(r.get("overall_overlap")),
                str(len(r.get("negatives_altered") or [])),
                str(len(r.get("protected_values_corrupted") or [])),
            ]
        )
    out += [
        table(
            (
                "Test set",
                "Examples (pos/neg)",
                "Gold spans",
                "Strict P / R / F1",
                "Overlap P / R / F1",
                "Negatives altered",
                "Clinical values corrupted",
            ),
            rows,
            "lrrrrrr",
        ),
        "",
    ]
    blind = pii_block(pii, "blind_v2")
    if blind:
        per = blind[1].get("per_category_strict") or {}
        weakest = sorted(
            ((cat, _num(v.get("recall"))) for cat, v in per.items() if isinstance(v, dict)),
            key=lambda kv: (kv[1] if kv[1] is not None else 2.0, kv[0]),
        )[:2]
        if weakest:
            out.append(
                "Lowest blind-set strict recall: "
                + ", ".join(f"{cat} {fixed(rec)}" for cat, rec in weakest)
                + "."
            )
            out.append("")
    out.append(f"Generated {date_of(dig(pii, 'meta.generated_at'))}. " + source(P_PII))
    out.append("")
    return out


def section_corpus(ctx: Context) -> list[str]:
    out = ["### Knowledge base", ""]
    corpus = ctx.json(P_CORPUS)
    totals = corpus.get("totals") if isinstance(corpus, dict) else None
    if not isinstance(totals, dict):
        out.append(
            pending_note(
                ctx,
                f"`{P_CORPUS}` not generated yet",
                "Run `python -m medbridge.ingestion --stats`.",
            )
        )
        return [*out, ""]
    by_type = corpus.get("by_doc_type") or {}
    rows = [
        [t, v.get("documents", "—"), v.get("chunks", "—"), v.get("pages", "—")]
        for t, v in by_type.items()
        if isinstance(v, dict)
    ]
    rows.append(
        [
            "**total**",
            totals.get("documents", "—"),
            totals.get("chunks", "—"),
            totals.get("pages_or_sections", "—"),
        ]
    )
    tokens = totals.get("chunk_tokens") or {}
    out += [
        table(("Document type", "Documents", "Chunks", "Pages / sections"), rows, "lrrr"),
        "",
        f"{totals.get('organisations', '—')} organisations; {totals.get('table_chunks', '—')} table "
        f"chunks; chunk length median {tokens.get('median', '—')} tokens (max {tokens.get('max', '—')}, "
        f"{tokens.get('over_512', '—')} over 512). Per-document table: `reports/corpus_stats.md`. "
        + source(P_CORPUS, "data/corpus_manifest.json"),
        "",
    ]
    return out


def live_or_none(data: Any) -> dict[str, Any] | None:
    """A benchmark report counts as a result only when it was produced in live mode."""
    if not isinstance(data, dict):
        return None
    mode = data.get("mode") or dig(data, "metrics.mode")
    return data if mode == "live" else None


def smoke_text(ctx: Context) -> str:
    smoke = ctx.json(P_SMOKE)
    if not isinstance(smoke, dict) or not isinstance(smoke.get("counts"), dict):
        ctx.pending.append("live smoke test")
        return f"{PENDING} (`{P_SMOKE}` not generated; run `python scripts/smoke_test.py`)"
    c = smoke["counts"]
    total = sum(int(v) for v in c.values() if isinstance(v, int))
    return (
        f"{c.get('PASS', 0)}/{total} checks passed"
        + (f", {c.get('FAIL')} failed" if c.get("FAIL") else "")
        + (f", {c.get('SKIP')} skipped" if c.get("SKIP") else "")
        + f" against a live local stack on {date_of(smoke.get('generated_at'))} (`{P_SMOKE}`)"
    )


def aq_text(ctx: Context) -> str:
    data = ctx.json(P_AQ)
    live = live_or_none(data)
    if not live:
        ctx.pending.append("live MedBridge-AQ benchmark")
        mode = (data or {}).get("mode") if isinstance(data, dict) else None
        why = (
            f"the current `{P_AQ}` is a {mode}-mode routing simulator, not a result"
            if mode
            else f"`{P_AQ}` not generated"
        )
        return (
            f"{PENDING} ({why}; run `python -m tests.benchmark.benchmark_runner --mode live --all`)"
        )
    m = live.get("metrics") or {}
    return (
        f"{m.get('total_vignettes', '?')} vignettes, routing accuracy "
        f"{fixed(m.get('overall_routing_accuracy'), 1)}%, emergency recall "
        f"{fixed(m.get('emergency_detection_recall'), 1)}%, citation precision "
        f"{fixed(m.get('overall_citation_precision'), 1)}%, out-of-domain ANSWER count "
        f"{m.get('out_of_domain_answer_count', '?')}/{m.get('out_of_domain_count', '?')}, "
        f"errors {m.get('errors', '?')} ({date_of(live.get('generated_at'))}, `{P_AQ}`)"
    )


def rabbits_text(ctx: Context) -> str:
    data = ctx.json(P_RABBITS)
    live = live_or_none(data)
    if not live:
        ctx.pending.append("live RABBITS run")
        mode = data.get("mode") if isinstance(data, dict) else None
        why = (
            f"the current `{P_RABBITS}` is {mode}-mode (deterministic components only)"
            if mode
            else f"`{P_RABBITS}` not generated"
        )
        return f"{PENDING} ({why}; run `python -m tests.adversarial.rabbits_runner --mode live`)"
    return (
        f"{live.get('pairs', '?')} brand–generic pairs ({live.get('indian_pairs', '?')} Indian), "
        f"action consistency {fixed(live.get('action_consistency_pct'), 1)}%, extraction "
        f"consistency {fixed(live.get('extraction_consistency_pct'), 1)}%, mean citation Jaccard "
        f"{fixed(live.get('citation_jaccard_mean'), 3)}, errors {live.get('errors', '?')} "
        f"({date_of(live.get('timestamp') or live.get('generated_at'))}, `{P_RABBITS}`)"
    )


def adversarial_text(ctx: Context) -> str:
    live = live_or_none(ctx.json(P_ADVERSARIAL))
    if not live or not isinstance(live.get("counts"), dict):
        ctx.pending.append("live adversarial run")
        return (
            f"{PENDING} (`{P_ADVERSARIAL}` not generated; run "
            "`python -m tests.adversarial.adversarial_runner`)"
        )
    c = live["counts"]
    return (
        f"{c.get('passed', '?')}/{c.get('payloads', '?')} payloads passed; system-prompt leaks "
        f"{c.get('leaked_system_prompt', '?')}, fabricated prescriptions "
        f"{c.get('fabricated_prescription', '?')}, unsafe outputs {c.get('unsafe_output', '?')}, "
        f"emergency payloads escalated {c.get('emergency_escalated', '?')}/"
        f"{c.get('emergency_payloads', '?')} ({date_of(live.get('generated_at'))}, `{P_ADVERSARIAL}`)"
    )


def load_rows(ctx: Context) -> list[list[str]] | None:
    data = ctx.json(P_LOAD)
    levels = data.get("levels") if isinstance(data, dict) else None
    if not isinstance(levels, list) or not levels:
        return None
    rows = []
    for lv in levels:
        if not isinstance(lv, dict):
            continue
        rows.append(
            [
                str(lv.get("users", "—")),
                str(lv.get("requests", "—")),
                str(lv.get("errors", "—")),
                str(lv.get("rate_limited", "—")),
                pct(lv.get("error_rate")),
                fixed(lv.get("p50_ms"), 0),
                fixed(lv.get("p95_ms"), 0),
                fixed(lv.get("throughput_rps"), 2),
            ]
        )
    return rows


def load_text(ctx: Context) -> str:
    rows = load_rows(ctx)
    if not rows:
        ctx.pending.append("load test")
        return f"{PENDING} (`{P_LOAD}` not generated; run `python scripts/load_test.py`)"
    data = ctx.json(P_LOAD)
    return (
        "; ".join(
            f"{r[0]} user{'' if r[0] == '1' else 's'}: p50 {r[5]} ms, p95 {r[6]} ms, error rate {r[4]}"
            for r in rows
        )
        + f" ({date_of(data.get('generated_at'))}, `{P_LOAD}`)"
    )


def section_operations(ctx: Context) -> list[str]:
    out = ["### Live benchmarks and operations", ""]
    rows = [
        ["Smoke test (every action path, PII, SSE, auth, erasure)", smoke_text(ctx)],
        ["MedBridge-AQ routing benchmark (275 vignettes)", aq_text(ctx)],
        ["RABBITS brand↔generic robustness (189 brands)", rabbits_text(ctx)],
        ["Adversarial prompt-injection suite (live)", adversarial_text(ctx)],
        ["Load test (1/5/10 concurrent users)", load_text(ctx)],
    ]
    out += [table(("Check", "Result"), rows), ""]
    load = load_rows(ctx)
    if load:
        out += [
            table(
                ("Users", "Requests", "Errors", "429s", "Error rate", "p50 ms", "p95 ms", "req/s"),
                load,
                "rrrrrrrr",
            ),
            "",
            source(P_LOAD),
            "",
        ]
    return out


def perf_probe_files(ctx: Context) -> dict[str, str]:
    """Newest probe file per label ('before' / 'after') in reports/performance."""
    found: dict[str, tuple[float, str]] = {}
    for rel in ctx.glob(f"{P_PERF_DIR}/*.json"):
        data = ctx.json(rel)
        if (
            isinstance(data, dict)
            and data.get("label") in ("before", "after")
            and "warm_summary" in data
        ):
            mtime = ctx.path(rel).stat().st_mtime
            label = str(data["label"])
            if label not in found or mtime > found[label][0]:
                found[label] = (mtime, rel)
    return {label: rel for label, (_, rel) in found.items()}


def section_latency(ctx: Context) -> list[str]:
    out = ["### Latency before vs after optimisation", ""]
    table_path = ctx.path(P_LATENCY_TABLE)
    if table_path.is_file():
        try:
            with table_path.open(encoding="utf-8", newline="") as fh:
                reader = list(csv.reader(fh))
        except OSError as exc:
            ctx.warnings.append(f"could not read {P_LATENCY_TABLE}: {exc}")
            reader = []
        if len(reader) > 1:
            out += [table(reader[0], reader[1:]), "", source(P_LATENCY_TABLE), ""]
            return out
    probes = perf_probe_files(ctx)
    before = ctx.json(probes["before"]) if "before" in probes else None
    if isinstance(before, dict):
        rows = []
        cold = {
            f"{c.get('scenario')}-t{c.get('turn')}": c
            for c in before.get("cold") or []
            if isinstance(c, dict)
        }
        for key, stats in sorted((before.get("warm_summary") or {}).items()):
            turn, _, action = key.partition(":")
            c = cold.get(turn) or {}
            rows.append(
                [
                    turn,
                    action,
                    fixed(c.get("latency_s"), 2),
                    fixed(stats.get("p50"), 2),
                    fixed(stats.get("p95"), 2),
                    str(stats.get("n", "—")),
                ]
            )
        out += [
            "Baseline measured on the received version (before any change):",
            "",
            table(
                ("Scenario turn", "Action", "Cold s", "Warm p50 s", "Warm p95 s", "n"),
                rows,
                "llrrrr",
            ),
            "",
            source(probes["before"]),
            "",
        ]
    if "after" not in probes:
        out.append(
            pending_note(
                ctx,
                "the 'after' latency probe has not been recorded",
                "Run `python scripts/latency_probe.py --label after ...` then `python -m evaluation.perf`.",
            )
        )
        out.append("")
    return out


@dataclass
class TestTotals:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    files: list[str] = field(default_factory=list)


def junit_totals(ctx: Context) -> TestTotals | None:
    files: list[str] = []
    for pattern in P_JUNIT:
        files += ctx.glob(pattern)
    if not files:
        return None
    totals = TestTotals()
    for rel in sorted(set(files)):
        try:
            root = ET.parse(ctx.path(rel)).getroot()
        except (ET.ParseError, OSError) as exc:
            ctx.warnings.append(f"could not parse {rel}: {exc}")
            continue
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        for s in suites:
            totals.tests += int(s.get("tests", 0) or 0)
            totals.failures += int(s.get("failures", 0) or 0)
            totals.errors += int(s.get("errors", 0) or 0)
            totals.skipped += int(s.get("skipped", 0) or 0)
        totals.files.append(rel)
    return totals if totals.files else None


def coverage_info(ctx: Context) -> tuple[str, float | None, dict[str, float]] | None:
    rel = ctx.first_existing(P_COVERAGE)
    data = ctx.json(rel) if rel else None
    if rel is None or not isinstance(data, dict):
        return None
    total = _num(dig(data, "totals.percent_covered"))
    per_pkg: dict[str, list[int]] = {}
    for fname, info in (data.get("files") or {}).items():
        parts = Path(fname).as_posix().split("/")
        if len(parts) < 2 or "tests" in parts:
            continue
        pkg = "/".join(parts[:2]) if parts[0] == "medbridge" and len(parts) > 2 else parts[0]
        summ = info.get("summary") or {}
        acc = per_pkg.setdefault(pkg, [0, 0])
        acc[0] += int(summ.get("covered_lines", 0) or 0)
        acc[1] += int(summ.get("num_statements", 0) or 0)
    pkgs = {k: 100.0 * c / n for k, (c, n) in per_pkg.items() if n}
    return rel, total, pkgs


def baseline_pytest(ctx: Context) -> str | None:
    path = ctx.path(P_BASELINE_PYTEST)
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if re.search(r"\b(passed|failed)\b.* in [\d.]+s", line):
            return line.strip().strip("=").strip()
    return None


def tests_text(ctx: Context) -> str:
    custom_rel = ctx.first_existing(P_TEST_SUMMARY)
    custom = ctx.json(custom_rel) if custom_rel else None
    junit = junit_totals(ctx)
    cov = coverage_info(ctx)
    parts = []
    if isinstance(custom, dict) and custom.get("passed") is not None:
        parts.append(
            f"{custom.get('passed')} passed, {custom.get('failed', 0)} failed, "
            f"{custom.get('skipped', 0)} skipped (`{custom_rel}`)"
        )
    elif junit:
        passed = junit.tests - junit.failures - junit.errors - junit.skipped
        parts.append(
            f"{passed} passed, {junit.failures} failed, {junit.errors} errors, {junit.skipped} "
            f"skipped across {len(junit.files)} JUnit report(s)"
        )
    if cov and cov[1] is not None:
        parts.append(f"line coverage {cov[1]:.1f}% (`{cov[0]}`)")
    if not parts:
        ctx.pending.append("test-suite summary")
        return (
            f"{PENDING} (no JUnit/coverage report under `reports/tests/` yet; see the Testing "
            "section for the command)"
        )
    return "; ".join(parts)


def section_tests(ctx: Context) -> list[str]:
    out = ["### Test suite", ""]
    out.append(f"- Current: {tests_text(ctx)}.")
    baseline = baseline_pytest(ctx)
    if baseline:
        out.append(f"- Received version (`fad6fea`): {baseline} (`{P_BASELINE_PYTEST}`).")
    cov = coverage_info(ctx)
    out.append("")
    if cov and cov[2]:
        rows = [[pkg, f"{val:.1f}%"] for pkg, val in sorted(cov[2].items())]
        out += [table(("Package", "Line coverage"), rows, "lr"), "", source(cov[0]), ""]
    return out


# ----------------------------------------------------------------------
# README blocks
# ----------------------------------------------------------------------


def build_results_block(ctx: Context) -> str:
    lines = [
        "",
        "> Generated by `scripts/render_docs_results.py` from the files named under each table. "
        "Do not edit by hand: re-run the script after new results land. *pending* means the "
        "experiment has not produced its result file yet.",
        "",
    ]
    for section in (
        section_hallucination,
        section_retrieval,
        section_model_selection,
        section_privacy,
        section_corpus,
        section_operations,
        section_latency,
        section_tests,
    ):
        try:
            lines += section(ctx)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
            ctx.warnings.append(f"{section.__name__}: malformed input ({exc!r})")
            lines += [
                f"{PENDING} — {section.__name__} could not be rendered from its input file.",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def build_screenshots_block(ctx: Context) -> str:
    present = [(n, c) for n, c in SCREENSHOTS if ctx.exists(f"{SCREENSHOTS_DIR}/{n}")]
    missing = [(n, c) for n, c in SCREENSHOTS if not ctx.exists(f"{SCREENSHOTS_DIR}/{n}")]
    lines = [""]
    if present:
        rows = []
        for i in range(0, len(present), 2):
            pair = present[i : i + 2]
            rows.append(
                [f"![{c}]({SCREENSHOTS_DIR}/{n})<br>{c}" for n, c in pair]
                + ([""] if len(pair) == 1 else [])
            )
        lines += [table(("", ""), rows), ""]
    if missing:
        lines += [
            "Not captured yet (expected under `report/screenshots/`, taken against the real "
            "backend, never the dev mock):",
            "",
        ]
        lines += [f"- `{n}`: {c}" for n, c in missing]
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Inline RESULT specs
# ----------------------------------------------------------------------


#: Spec kinds that read evaluation/results/summary.json (pending while it does not exist).
SUMMARY_SPEC_KINDS: frozenset[str] = frozenset(
    {
        "rate",
        "interval",
        "routing",
        "latency",
        "degraded",
        "test",
        "cochran",
        "kappa",
        "split",
        "models",
    }
)
KNOWN_SPEC_KINDS: frozenset[str] = SUMMARY_SPEC_KINDS | {
    "human",
    "retrieval",
    "reranker",
    "model_selection",
    "pii",
    "corpus",
    "smoke",
    "aq",
    "rabbits",
    "adversarial",
    "load",
    "tests",
    "perf",
}


def _summary(ctx: Context) -> dict[str, Any] | None:
    s = ctx.json(P_SUMMARY)
    return (
        s
        if isinstance(s, dict) and isinstance(s.get("per_condition"), dict) and s["per_condition"]
        else None
    )


def spec_value(ctx: Context, spec: str) -> str | None:
    """Rendered text for one RESULT spec, or None when its source is missing (pending)."""
    parts = spec.split(":")
    kind = parts[0]
    summary = _summary(ctx)

    if kind not in KNOWN_SPEC_KINDS:
        ctx.warnings.append(f"unknown RESULT spec '{spec}'")
        return None
    if kind in SUMMARY_SPEC_KINDS and summary is None:
        return None
    if kind == "rate" and len(parts) == 3:
        text = prop_text(dig(summary, f"per_condition.{parts[1]}.{parts[2]}"), "turns")
        return text or None
    if kind == "interval" and len(parts) == 3:
        cell = interval_cell(dig(summary, f"per_condition.{parts[1]}.{parts[2]}"), as_pct=True)
        return None if cell == "—" else f"{cell} (95% CI)"
    if kind == "routing" and len(parts) == 3:
        m = dig(summary, f"per_condition.{parts[1]}")
        if not isinstance(m, dict):
            return None
        routing, mark = routing_of(m, parts[1])
        if not routing:
            return None
        if parts[2] == "macro_f1":
            return fixed(routing.get("macro_f1"), 2)
        key = "accuracy_strict" if parts[2] == "strict" else "accuracy_lenient"
        text = prop_text(routing.get(key), "turns")
        return (text + (" (from judge behaviour)" if mark else "")) if text else None
    if kind == "latency" and len(parts) == 2:
        lat = dig(summary, f"per_condition.{parts[1]}.latency.imputed")
        if not isinstance(lat, dict) or _num(lat.get("p50_ms")) is None:
            return None
        return (
            f"p50 {float(lat['p50_ms']) / 1000:.1f} s, p95 {float(lat['p95_ms']) / 1000:.1f} s "
            f"per turn (cache-imputed, n = {lat.get('n')})"
        )
    if kind == "degraded" and len(parts) == 2:
        d = dig(summary, f"degraded.per_condition.{parts[1]}")
        if not isinstance(d, dict):
            return None
        return f"{d.get('n_degraded', 0)}/{d.get('n_turns', 0)} turns degraded (excluded)"
    if kind == "test" and len(parts) == 5:
        if summary is None:
            return None
        row = find_test(summary, parts[1], parts[2], parts[3], parts[4])
        if not row:
            return None
        return (
            f"{parts[1]} {pct(row.get('p_treatment'))} vs {parts[2]} {pct(row.get('p_control'))} "
            f"({parts[3]}, {parts[4].upper()} judge; risk difference {rd_cell(row.get('risk_difference'))}, "
            f"conditional OR {or_cell(row.get('odds_ratio'))}, exact McNemar "
            f"{p_eq(row.get('pvalue'))}, Holm-adjusted {p_eq(row.get('pvalue_holm'))}, "
            f"{row.get('n_pairs')} paired turns)"
        )
    if kind == "cochran" and len(parts) == 3:
        if summary is None:
            return None
        for r in dig(summary, "tests.cochran_q") or []:
            if isinstance(r, dict) and r.get("outcome") == parts[1] and r.get("judge") == parts[2]:
                return (
                    f"Cochran's Q = {fixed(r.get('q'), 2)}, df = {r.get('df')}, "
                    f"{p_eq(r.get('pvalue'))} (n = {r.get('n')})"
                )
        return None
    if kind == "kappa" and len(parts) == 2:
        agreement = summary.get("judge_agreement") if summary else None
        if not isinstance(agreement, dict):
            return None
        kappa_key = {
            "3class": "kappa_3class",
            "binary": "kappa_binary",
            "response": "kappa_response_hallucination",
        }.get(parts[1])
        if not kappa_key:
            return None
        n = agreement.get("n_responses" if parts[1] == "response" else "n_claims")
        text = kappa_text(agreement.get(kappa_key))
        return None if text == "—" else f"{text} (n = {n})"
    if kind == "human" and len(parts) == 2:
        human = ctx.json(P_HUMAN)
        if not isinstance(human, dict):
            return None
        if parts[1] == "status":
            return f"`{human.get('status', 'unknown')}`"
        if human.get("status") == "not_filled":
            return None
        block = human.get(parts[1])
        if not isinstance(block, dict) or not block.get("n"):
            return None
        return f"{kappa_text(block.get('kappa_3class'))} (n = {block['n']})"
    if kind == "split" and summary is not None:
        per = summary["per_condition"]
        c5 = as_dict(per.get("C5") or next(iter(per.values())))
        return (
            f"{summary.get('split', '?')} split: {c5.get('n_items', '?')} items, "
            f"{c5.get('n_valid', '?')} valid turns in C5"
        )
    if kind == "models" and summary is not None:
        app, judges = study_models(ctx, summary)
        if not app and not judges:
            return None
        return f"application {', '.join(app) or 'n/a'}; judges {', '.join(judges) or 'n/a'}"
    if kind == "retrieval" and len(parts) == 3:
        block = dig(ctx.json(P_RETRIEVAL), ["modes", parts[1], parts[2]])
        cell = interval_cell(block)
        return None if cell == "—" else f"{cell} (95% CI)"
    if kind == "reranker":
        rr = reranker_rows(ctx)
        if not rr:
            return None
        rows, overlap = rr
        mrr_i = 1 + RETRIEVAL_METRICS.index("mrr")
        text = "MRR " + "; ".join(f"{r[0]} {r[mrr_i]}" for r in rows)
        if overlap is not None:
            text += (
                "; all 95% CIs overlap with RRF alone"
                if overlap
                else "; not all 95% CIs overlap with RRF alone"
            )
        return text
    if kind == "model_selection":
        valid = model_selection_valid(ctx.json(P_MODEL_SELECTION))
        label = ""
        if not valid:
            earlier = earlier_model_selection(ctx)
            if not earlier:
                return None
            valid = model_selection_valid(earlier[1])
            label = (
                f" (earlier round at commit {', '.join(earlier[2]) or 'unknown'}, `{earlier[0]}`; "
                "the re-run with the current prompts is pending)"
            )
        return (
            "; ".join(
                f"{name}: lenient routing {pct(dig(r, 'routing.accuracy_lenient.rate'))}, "
                f"macro-F1 {fixed(dig(r, 'routing.macro_f1'), 2)}, turn p50 "
                f"{fixed((_num(r.get('latency_turn_p50_ms')) or math.nan) / 1000, 1)} s"
                for name, r in valid
            )
            + label
        )
    if kind == "pii" and len(parts) == 3:
        block = pii_block(ctx.json(P_PII), parts[1])
        if not block:
            return None
        r = block[1]
        if parts[2] == "clinical":
            return (
                f"{len(r.get('protected_values_corrupted') or [])} protected clinical values "
                f"corrupted, {len(r.get('negatives_altered') or [])} of "
                f"{r.get('n_negative_examples', '?')} PII-free messages altered"
            )
        d = r.get("overall_strict" if parts[2] == "strict" else "overall_overlap")
        if not isinstance(d, dict):
            return None
        return (
            f"precision {fixed(d.get('precision'))}, recall {fixed(d.get('recall'))}, "
            f"F1 {fixed(d.get('f1'))} ({parts[2]} span match, {d.get('support', '?')} gold spans)"
        )
    if kind == "corpus":
        totals = dig(ctx.json(P_CORPUS), "totals")
        if not isinstance(totals, dict):
            return None
        return (
            f"{totals.get('documents')} documents, {totals.get('chunks')} chunks "
            f"({totals.get('table_chunks')} table chunks) from {totals.get('organisations')} organisations"
        )
    if kind == "smoke":
        text = smoke_text(ctx)
        return None if text.startswith(PENDING) else text
    if kind == "aq":
        text = aq_text(ctx)
        return None if text.startswith(PENDING) else text
    if kind == "rabbits":
        text = rabbits_text(ctx)
        return None if text.startswith(PENDING) else text
    if kind == "adversarial":
        text = adversarial_text(ctx)
        return None if text.startswith(PENDING) else text
    if kind == "load":
        text = load_text(ctx)
        return None if text.startswith(PENDING) else text
    if kind == "tests":
        text = tests_text(ctx)
        return None if text.startswith(PENDING) else text
    if kind == "perf":
        if not ctx.path(P_LATENCY_TABLE).is_file():
            return None
        try:
            with ctx.path(P_LATENCY_TABLE).open(encoding="utf-8", newline="") as fh:
                dict_rows = list(csv.DictReader(fh))
        except OSError:
            return None
        bits = [
            f"{r.get('Scenario turn')}: {r.get('Warm p50 before s')} s → {r.get('Warm p50 after s')} s"
            for r in dict_rows
            if r.get("Scenario turn")
        ]
        return ("warm p50 " + "; ".join(bits) + f" (`{P_LATENCY_TABLE}`)") if bits else None
    ctx.warnings.append(f"unknown RESULT spec '{spec}'")
    return None


RESULT_RE = re.compile(r"<!-- RESULT:([^>\n]+?) -->(.*?)<!-- /RESULT -->")


def render_inline(ctx: Context, text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        spec = m.group(1).strip()
        try:
            value = spec_value(ctx, spec)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
            ctx.warnings.append(f"RESULT:{spec}: malformed input ({exc!r})")
            value = None
        if value is None:
            ctx.pending.append(f"RESULT:{spec}")
            value = f"{PENDING} (result file not generated yet)"
        return f"<!-- RESULT:{spec} -->{value}<!-- /RESULT -->"

    return RESULT_RE.sub(repl, text)


# ----------------------------------------------------------------------
# Checklist evidence check, CHANGELOG ADL ids
# ----------------------------------------------------------------------


PATH_RE = re.compile(r"`([A-Za-z0-9_.][A-Za-z0-9_./*\-]*)`")
PATH_EXTS = (
    ".md",
    ".py",
    ".json",
    ".jsonl",
    ".png",
    ".svg",
    ".docx",
    ".pptx",
    ".pdf",
    ".tex",
    ".csv",
    ".xlsx",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".html",
    ".js",
    ".css",
    ".sh",
    ".ps1",
    ".ini",
    ".mjs",
    ".xml",
)


def evidence_paths(text: str) -> list[str]:
    """Backticked repo paths in a markdown text (ignores the managed block itself)."""
    text = re.sub(
        r"<!-- EVIDENCE-CHECK:START -->.*?<!-- EVIDENCE-CHECK:END -->", "", text, flags=re.DOTALL
    )
    seen: list[str] = []
    for m in PATH_RE.finditer(text):
        cand = m.group(1).rstrip(".")
        if cand.startswith(("http", "-", ".")) or "//" in cand:
            continue
        if "/" not in cand and not cand.endswith(PATH_EXTS):
            continue
        if "/" not in cand and not cand[0].isupper() and not cand.endswith(".md"):
            continue  # bare module names such as `metrics.py` are not repo paths
        if cand not in seen:
            seen.append(cand)
    return seen


def build_evidence_block(ctx: Context, text: str) -> str:
    paths = evidence_paths(text)
    missing = []
    for rel in paths:
        if any(ch in rel for ch in "*?["):
            if not ctx.glob(rel):
                missing.append(rel)
        elif not ctx.path(rel.rstrip("/")).exists():
            missing.append(rel)
    lines = [
        "",
        f"Computed by `scripts/render_docs_results.py`: {len(paths) - len(missing)} of "
        f"{len(paths)} evidence paths named in this checklist exist in the repository.",
        "",
    ]
    if missing:
        lines += ["Not present yet (produced by a pending step or another work-stream):", ""]
        lines += [f"- `{p}`" for p in missing]
        lines.append("")
    return "\n".join(lines)


ADL_HEADING_RE = re.compile(r"^#{2,4}\s+(ADL-\d{3})\b\s*[·:\-—–]?\s*(.*)$", re.MULTILINE)
ADL_RE = re.compile(r"<!-- ADL:([^>\n]+?) -->(.*?)<!-- /ADL -->")


def render_adl(ctx: Context, text: str) -> str:
    path = ctx.path(P_ADL)
    headings: list[tuple[str, str]] = []
    if path.is_file():
        try:
            headings = ADL_HEADING_RE.findall(path.read_text(encoding="utf-8"))
        except OSError as exc:
            ctx.warnings.append(f"could not read {P_ADL}: {exc}")

    def repl(m: re.Match[str]) -> str:
        pattern = m.group(1).strip()
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            ctx.warnings.append(f"bad ADL regex '{pattern}': {exc}")
            return m.group(0)
        ids = sorted({adl for adl, title in headings if rx.search(title)})
        if ids:
            value = ", ".join(f"[{i}]({P_ADL})" for i in ids)
        else:
            ctx.pending.append(f"ADL:{pattern}")
            value = "ADL entry pending"
        return f"<!-- ADL:{pattern} -->{value}<!-- /ADL -->"

    return ADL_RE.sub(repl, text)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return text
    return pattern.sub(lambda _m: start + "\n" + content.strip("\n") + "\n" + end, text, count=1)


def render_text(ctx: Context, rel: str, text: str) -> str:
    if "<!-- RESULTS:START -->" in text:
        text = replace_block(text, "RESULTS", build_results_block(ctx))
    if "<!-- SCREENSHOTS:START -->" in text:
        text = replace_block(text, "SCREENSHOTS", build_screenshots_block(ctx))
    text = render_inline(ctx, text)
    if "<!-- EVIDENCE-CHECK:START -->" in text:
        text = replace_block(text, "EVIDENCE-CHECK", build_evidence_block(ctx, text))
    text = render_adl(ctx, text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root")
    parser.add_argument("--files", nargs="*", default=list(MANAGED_FILES), help="files to render")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if stale")
    args = parser.parse_args(argv)

    ctx = Context(root=args.root.resolve())
    stale: list[str] = []
    for rel in args.files:
        path = ctx.path(rel)
        if not path.is_file():
            print(f"[render] skip {rel} (not found)")
            continue
        old = path.read_text(encoding="utf-8")
        new = render_text(ctx, rel, old)
        if new != old:
            stale.append(rel)
            if not args.check:
                path.write_text(new, encoding="utf-8")
                print(f"[render] updated {rel}")
        else:
            print(f"[render] unchanged {rel}")

    for warning in ctx.warnings:
        print(f"[render] warning: {warning}", file=sys.stderr)
    pending = sorted(set(ctx.pending))
    if pending:
        print(f"[render] {len(pending)} pending item(s):")
        for item in pending:
            print(f"  - {item}")
    if args.check and stale:
        print(f"[render] stale: {', '.join(stale)} (run without --check to update)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

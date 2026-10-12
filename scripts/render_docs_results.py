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
      value:<COND>:<dotted.path>[:digits]  plain number, e.g. value:V2:llm.claims_per_response:2
      routing:<COND>:<strict|lenient|macro_f1>
      latency:<COND>                   cache-imputed p50 / p95 per turn
      degraded:<COND>                  degraded (excluded) turns
      test:<T>:<C>:<outcome>:<judge>   paired exact McNemar row (outcome: hallucination|unsafe;
                                       judge: llm|nli), e.g. test:C5:C1:hallucination:llm
      cochran:<outcome>:<judge>        Cochran's Q across C0-C5
      kappa:<3class|binary|response>   LLM-judge vs NLI-judge agreement (with the conditions
                                       it was computed on)
      nli_coverage                     which conditions the NLI judge scored completely,
                                       partially or not at all on the reported split
      human:<status|human_human|A1_vs_llm|A2_vs_llm|A1_vs_nli|A2_vs_nli|consensus_vs_llm|...>
      ai_annotation                    AI annotator (a model, not a human) against both judges
      split                            split / item / turn counts of the study
      models                           application and judge models of the study
      commits                          git commits recorded in the study's test-split run manifests
      retrieval:<mode>:<metric>        mode: dense|sparse|hybrid|hybrid+rerank;
                                       metric: recall@5|recall@20|mrr|ndcg@5
      reranker                         reranker-ablation one-liner (MRR per reranker)
      model_selection                  model-selection one-liner
      pii:<development|blind_v2|blind_v2_posthoc>:<strict|overlap|clinical>
      corpus                           corpus size one-liner
      smoke                            live smoke-test result
      aq | rabbits | adversarial | load   live benchmark one-liners (pending unless mode == live)
      rabbits_before | rabbits_after   live RABBITS run kept from before the brand-name fix /
                                       the newer live run at the default path
      rabbits_compare                  before → after RABBITS metrics and flip counts
      adversarial_before               live adversarial run kept from before the brand-name fix
      adversarial_compare              before → after adversarial checks and changed actions
      perf                             warm p50 before → after per probe turn
      perf_cold                        first-request (cold) latency before → after
      perf_changes                     turns that got faster / slower / stayed within ±10 %
      perf_gate3                       Gate 3 verification stage time (warm p50 range)
      perf_after:<model-tag>           warm p50 per turn of one after-probe, e.g. perf_after:qwen3.5-4b
      annotation                       status of the exported human-annotation sheet
      tests                            test-suite one-liner

  A live RABBITS or adversarial run is called "after the brand-name fix" only when the
  commit its report records contains the fix commit (``git merge-base --is-ancestor``).

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
import subprocess
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
P_AI_ANNOTATION = "evaluation/results/ai_annotation_agreement.json"
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
P_RABBITS_BEFORE = "reports/rabbits_before_fix/rabbits_benchmark_report.json"
P_RUN_MANIFESTS = "evaluation/results/runs/manifest_*.json"
P_ADVERSARIAL = "reports/adversarial_live_report.json"
P_ADVERSARIAL_BEFORE = "reports/adversarial_before_fix/adversarial_live_report.json"
P_LOAD = "reports/performance/load_test.json"
P_LATENCY_TABLE = "reports/performance/latency_table.csv"
P_LATENCY_TABLE_MD = "reports/performance/latency_table.md"
P_PERF_DIR = "reports/performance"
P_ANNOTATION_SHEET = "evaluation/annotation/human_annotation_sheet.csv"
P_ANNOTATION_XLSX = "evaluation/annotation/human_annotation_sheet.xlsx"
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
)
LATENCY_FIGURE = (
    "latency_before_after",
    "Latency before vs after optimisation (warm p50, whiskers to p95)",
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


def organisation_count(corpus: Any) -> Any:
    """Distinct issuing organisations, with joint labels ("AHA/ACC") split on "/".

    ``totals.organisations`` in ``reports/corpus_stats.json`` counts distinct label strings,
    so "AHA/ACC", "ACC/AHA" and "AHA" count three times; the deck counts the same way as here
    (``report/presentation/deck_data.py::corpus_organisations``).
    """
    docs = dig(corpus, "documents")
    orgs = {
        part.strip()
        for doc in (docs if isinstance(docs, list) else [])
        if isinstance(doc, dict) and isinstance(doc.get("org"), str)
        for part in doc["org"].split("/")
        if part.strip()
    }
    if orgs:
        return len(orgs)
    return dig(corpus, "totals.organisations") or "—"


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
        if not isinstance(manifest, dict):
            continue
        # An NLI manifest also carries the (unused) default judge_model, so the model is
        # taken from the field that belongs to the judge the manifest describes.
        kind = manifest.get("judge")
        if kind == "nli" and manifest.get("nli_model"):
            judges.add(f"NLI judge {manifest['nli_model']}")
        elif kind in (None, "llm") and manifest.get("judge_model"):
            judges.add(f"LLM judge {manifest['judge_model']}")
    return sorted(app), sorted(judges)


def nli_coverage(summary: dict[str, Any]) -> dict[str, list[tuple[str, int, int]]]:
    """Conditions the NLI judge scored completely, partially or not at all.

    Read from ``per_condition.<C>.judge_coverage.nli`` when the summary has it (a judge's
    rates are reported only when it scored at least 95 % of the valid turns). Older
    summaries: a condition counts as complete when the NLI judge scored as many turns as
    the LLM judge; fewer turns is a partial subset that must not be read as a
    full-condition result. Entries are (condition, NLI-judged turns, reference turns).
    """
    out: dict[str, list[tuple[str, int, int]]] = {"complete": [], "partial": [], "none": []}
    for cond in summary_conditions(summary):
        m = as_dict(dig(summary, f"per_condition.{cond}"))
        cov = as_dict(dig(m, "judge_coverage.nli"))
        if cov:
            # evaluation/analyze.py (since e0e722d) records how many valid turns the NLI judge
            # scored and whether that was enough (>= 95 %) to report its per-condition rates.
            judged = int(_num(cov.get("judged")) or 0)
            valid = int(_num(cov.get("valid")) or 0)
            if not judged:
                out["none"].append((cond, 0, valid))
            elif cov.get("reported") is False:
                out["partial"].append((cond, judged, valid))
            else:
                out["complete"].append((cond, judged, valid))
            continue
        llm_n = int(_num(dig(m, "llm.hallucination.n")) or 0)
        nli_n = int(_num(dig(m, "nli.hallucination.n")) or 0)
        if not nli_n:
            out["none"].append((cond, 0, llm_n))
        elif llm_n and nli_n < llm_n:
            out["partial"].append((cond, nli_n, llm_n))
        else:
            out["complete"].append((cond, nli_n, llm_n))
    return out


def nli_coverage_text(summary: dict[str, Any]) -> str:
    cov = nli_coverage(summary)
    parts = []
    if cov["complete"]:
        parts.append("complete for " + ", ".join(f"{c} (n = {n})" for c, n, _ in cov["complete"]))
    if cov["partial"]:
        parts.append(
            "a partial subset for "
            + ", ".join(f"{c} (n = {n} of {m} turns)" for c, n, m in cov["partial"])
            + ", which is not a full-condition result"
        )
    if cov["none"]:
        parts.append("no judgments for " + ", ".join(c for c, _, _ in cov["none"]))
    return "; ".join(parts) if parts else "no NLI judgments"


def study_commits(summary: dict[str, Any]) -> list[str]:
    """Short commits (``+dirty`` when the tree was dirty) of the reported split's runs."""
    split = summary.get("split")
    commits: list[str] = []
    for run in summary.get("runs") or []:
        if not isinstance(run, dict) or (split and run.get("split") != split):
            continue
        git = as_dict(run.get("git"))
        commit = git.get("commit")
        if isinstance(commit, str) and commit:
            label = commit[:7] + ("+dirty" if git.get("dirty") else "")
            if label not in commits:
                commits.append(label)
    return commits


def study_workers(ctx: Context, summary: dict[str, Any]) -> list[int]:
    """Worker counts recorded in the run manifests of the reported split."""
    split = summary.get("split")
    workers: set[int] = set()
    for rel in ctx.glob(P_RUN_MANIFESTS):
        manifest = ctx.json(rel)
        if not isinstance(manifest, dict) or (split and manifest.get("split") != split):
            continue
        w = _num(manifest.get("workers"))
        if w:
            workers.add(int(w))
    return sorted(workers)


def holm_verdict(row: dict[str, Any] | None, alpha: float = 0.05) -> str:
    p = _num(row.get("pvalue_holm")) if isinstance(row, dict) else None
    if p is None:
        return "not tested"
    return "significant" if p < alpha else "not significant"


def reading_notes(ctx: Context, summary: dict[str, Any]) -> list[str]:
    """Plain-language reading of the headline numbers, computed from summary.json."""
    notes: list[str] = []
    per = as_dict(summary.get("per_condition"))
    rows = {
        (o, c): find_test(summary, "C5", c, o, "llm")
        for o in ("hallucination", "unsafe")
        for c in ("C1", "C2")
    }
    if all(rows.values()):
        verdicts = {k: holm_verdict(v) for k, v in rows.items()}
        all_sig = all(v == "significant" for v in verdicts.values())
        worst = max(_num(v.get("pvalue_holm")) or 0.0 for v in rows.values() if v)
        notes.append(
            "Decision rule (master brief §6.7: C5 must reduce hallucination or unsafe answers "
            "versus C1 and C2, Holm-adjusted p < 0.05): "
            + ", ".join(f"C5 vs {c} {o} {verdicts[(o, c)]}" for o, c in rows)
            + f" (largest Holm-adjusted p {pvalue(worst)}, LLM judge). "
            + (
                "The fallback of §6.7 is therefore not triggered."
                if all_sig
                else "At least one comparison is not significant, so §6.7 requires the "
                "error analysis and the variants."
            )
        )
    c1, c5 = as_dict(per.get("C1")), as_dict(per.get("C5"))
    g1, g5 = dig(c1, "llm.hallucination_given_claims"), dig(c5, "llm.hallucination_given_claims")
    cov1, cov5 = c1.get("coverage_answerable"), c5.get("coverage_answerable")
    if is_proportion(g1) and is_proportion(g5) and is_proportion(cov1) and is_proportion(cov5):
        g1, g5, cov1, cov5 = as_dict(g1), as_dict(g5), as_dict(cov1), as_dict(cov5)
        notes.append(
            "How the gates lower the rate: template replies (ABSTAIN, ESCALATE, small talk) have "
            "no claims and count as not hallucinated. Among responses that contain claims, the "
            f"hallucination rate is {pct(g5.get('rate'))} for C5 ({g5.get('k')}/{g5.get('n')}) "
            f"against {pct(g1.get('rate'))} for C1 ({g1.get('k')}/{g1.get('n')}), while coverage "
            f"on answerable turns falls from {pct(cov1.get('rate'))} to {pct(cov5.get('rate'))}. "
            "Most of the reduction therefore comes from answering less often (selective "
            "abstention), not from better-grounded answers."
        )
    v2h = find_test(summary, "V2", "C5", "hallucination", "llm")
    v2u = find_test(summary, "V2", "C5", "unsafe", "llm")
    if v2h:
        notes.append(
            f"Gate 3 (V2) vs C5: hallucination {pct(v2h.get('p_treatment'))} vs "
            f"{pct(v2h.get('p_control'))}, {holm_verdict(v2h)} (Holm-adjusted "
            f"{p_eq(v2h.get('pvalue_holm'))})"
            + (
                f"; unsafe answers {pct(v2u.get('p_treatment'))} vs {pct(v2u.get('p_control'))}, "
                f"{holm_verdict(v2u)} (Holm-adjusted {p_eq(v2u.get('pvalue_holm'))})"
                if v2u
                else ""
            )
            + "."
        )
    cov = nli_coverage(summary)
    if cov["none"]:
        notes.append(
            "Every paired test and Cochran's Q uses the LLM judge. The NLI judge's coverage on "
            f"this split: {nli_coverage_text(summary)}. Results for "
            + ", ".join(c for c, _, _ in cov["none"])
            + " therefore rest on one judge; the NLI judge is used only for the agreement figures."
        )
    return notes


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
    summary = ctx.json(P_SUMMARY)
    conds = (
        summary_conditions(summary)
        if isinstance(summary, dict) and isinstance(summary.get("per_condition"), dict)
        else []
    )
    title = ", ".join(conds) if conds else "C0–C5, V1, V2"
    out = [f"### Hallucination study (MedBridge-HalluBench v1, conditions {title})", ""]
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
    c5 = as_dict(dig(summary, "per_condition.C5"))
    commits = study_commits(summary)
    hardware = sorted(
        {
            f"{h.get('cpu')}, {fixed(h.get('ram_gb'), 0)} GB RAM"
            for run in summary.get("runs") or []
            if isinstance(run, dict)
            and isinstance((h := run.get("hardware")), dict)
            and h.get("cpu")
        }
    )
    provenance = [
        f"Split: **{summary.get('split', '?')}** (dev split used only for calibration)"
        + (f", {c5.get('n_items')} items and {c5.get('n_turns')} turns per condition" if c5 else "")
        + f"; unit: {summary.get('unit', 'item-turn')}; degraded turns excluded from headline "
        "numbers.",
        f"Application model(s): {', '.join(app) if app else 'not recorded'}; "
        f"judges: {', '.join(judges) if judges else 'see judgment manifests'}; "
        f"NLI thresholds: {nli.get('source', 'unknown')}.",
        f"NLI judge coverage on this split: {nli_coverage_text(summary)}.",
    ]
    if commits or hardware:
        provenance.append(
            "Code evaluated: commit(s) "
            + (", ".join(f"`{c}`" for c in commits) or "not recorded")
            + " (run manifests in `evaluation/results/runs/`); hardware: "
            + ("; ".join(hardware) or "not recorded")
            + ". Later commits are not covered by these numbers."
        )
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
        "Hallucination among responses with claims (LLM)",
        "Hallucination (NLI judge)",
        "Unsupported-claim ratio (LLM)",
        "Unsafe answers, unanswerable (LLM)",
        "Coverage, answerable",
        "Routing accuracy, lenient",
        "Emergency recall",
    )
    partial = {c: n for c, n, _ in nli_coverage(summary)["partial"]}
    rows = []
    for cond in summary_conditions(summary):
        m = summary["per_condition"][cond]
        if not isinstance(m, dict):
            continue
        routing, mark = routing_of(m, cond)
        lenient = prop_cell(routing.get("accuracy_lenient")) if routing else "—"
        nli_cell = (
            f"— (partial subset, n = {partial[cond]}; not reported)"
            if cond in partial
            else prop_cell(dig(m, "nli.hallucination"))
        )
        rows.append(
            [
                condition_label(summary, cond),
                f"{m.get('n_valid', '—')}/{m.get('n_turns', '—')}",
                prop_cell(dig(m, "llm.hallucination")),
                prop_cell(dig(m, "llm.hallucination_given_claims")),
                nli_cell,
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
    partial_note = (
        " The NLI judge was stopped early: "
        + ", ".join(f"{c} has NLI judgments for only {n} turns" for c, n in partial.items())
        + "; such a partial, non-random subset is used only for judge agreement and its rate is "
        "not reported as a condition result."
        if partial
        else ""
    )
    out += [
        table(headers, rows, "lrrrrrrrrr"),
        "",
        "Cells: rate [95% Wilson CI] (events/turns); unsupported-claim ratio with a 10k cluster-"
        "bootstrap CI. Template replies (ABSTAIN, ESCALATE, small talk) have no claims and count "
        "as not hallucinated, so the fourth column shows the rate among responses that contain "
        "claims. † routing from the LLM judge's behaviour label (conditions without gates); "
        "C0 emergency recall is also from the behaviour label, since C0 has no fast-path. "
        "Definitions of every metric are stored in `summary.json` → `definitions`." + partial_note,
        "",
    ]
    notes = reading_notes(ctx, summary)
    if notes:
        out += ["Reading the numbers (computed from the same file):", ""]
        out += [f"- {n}" for n in notes] + [""]

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
        cov = nli_coverage(summary)
        scored = [c for c, _, _ in cov["complete"]] + [
            f"{c} (partial subset)" for c, _, _ in cov["partial"]
        ]
        out.append(
            "Judge agreement (Cohen's κ, LLM judge vs NLI judge): "
            f"{kappa_text(agreement.get('kappa_3class'))} three-class and "
            f"{kappa_text(agreement.get('kappa_binary'))} binary over "
            f"{agreement.get('n_claims', '?')} claims; "
            f"{kappa_text(agreement.get('kappa_response_hallucination'))} on the response-level "
            f"hallucination label over {agreement.get('n_responses', '?')} responses"
            + (
                f", computed only on the responses both judges scored ({', '.join(scored)})"
                if scored
                else ""
            )
            + ". Whether this agreement carries over to the shorter, template-heavy outputs of the "
            "gated conditions has not been measured."
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
    failures = []
    for cond in summary_conditions(summary):
        m = as_dict(dig(summary, f"per_condition.{cond}"))
        if not isinstance(m.get("llm"), dict):
            continue
        n_valid = int(_num(m.get("n_valid")) or 0)
        n_judged = int(_num(dig(m, "llm.n_judged")) or 0)
        if n_valid > n_judged:
            failures.append(f"{cond} {n_valid - n_judged} of {n_valid}")
    if failures:
        out += [
            "Valid turns the LLM judge could not score (left out of that condition's rates): "
            + ", ".join(failures)
            + ".",
            "",
        ]
    out += figure_gallery(ctx, summary)
    out.append(source(P_SUMMARY) + f" Full tables: `{P_TABLES}`; error analysis: `{P_ERRORS}`.")
    out.append("")
    return out


def kappa_text(k: Any) -> str:
    if not isinstance(k, dict) or _num(k.get("kappa")) is None:
        return "—"
    low, high = _num(k.get("low")), _num(k.get("high"))
    ci = f" [{low:.2f}, {high:.2f}]" if low is not None and high is not None else ""
    return f"κ = {float(k['kappa']):.2f}{ci}"


def annotation_sheet_status(ctx: Context) -> dict[str, int] | None:
    """Counts from the exported human-annotation sheet (CSV twin of the XLSX), or None."""
    path = ctx.path(P_ANNOTATION_SHEET)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as exc:
        ctx.warnings.append(f"could not read {P_ANNOTATION_SHEET}: {exc}")
        return None

    def filled(row: dict[str, Any], key: str) -> bool:
        return bool(str(row.get(key) or "").strip())

    return {
        "items": len({r.get("item_id") for r in rows if r.get("item_id")}),
        "responses": len({r.get("response_code") for r in rows if r.get("response_code")}),
        "claims": len(rows),
        "both": sum(1 for r in rows if filled(r, "A1_label") and filled(r, "A2_label")),
    }


def annotation_sheet_text(ctx: Context) -> str | None:
    st = annotation_sheet_status(ctx)
    if not st:
        return None
    sheet = P_ANNOTATION_XLSX if ctx.exists(P_ANNOTATION_XLSX) else P_ANNOTATION_SHEET
    return (
        f"sheet exported (`{sheet}`): {st['items']} HalluBench items, {st['responses']} blinded "
        f"responses, {st['claims']} claim rows; {st['both']} rows labelled by both annotators"
    )


def ai_annotation_sentence(ctx: Context) -> str:
    """The AI annotator (a model, never reported as a human), or '' without results."""
    ai = ctx.json(P_AI_ANNOTATION)
    if not isinstance(ai, dict) or not ai.get("n_claims"):
        return ""
    llm, nli = as_dict(ai.get("ai_vs_llm")), as_dict(ai.get("ai_vs_nli"))
    per = as_dict(ai.get("per_condition"))
    shares = [
        f"{c} {pct(dig(per, f'{c}.unsupported_claims_ai.rate'))}" for c in ("C5", "C1") if c in per
    ]
    parts = [f"LLM judge {kappa_text(llm.get('kappa_3class'))} (n = {llm.get('n')})"]
    if nli.get("n"):
        parts.append(f"NLI judge {kappa_text(nli.get('kappa_3class'))} (C1, n = {nli.get('n')})")
    return (
        f"AI annotator ({ai.get('annotator') or 'an AI model, not a human'}): labelled all "
        f"{ai['n_claims']} claims of the sheet from blinded batches. Three-class agreement with the "
        + "; ".join(parts)
        + (f". Share of claims it rated unsupported: {', '.join(shares)}" if shares else "")
        + f". {source(P_AI_ANNOTATION)}"
    )


def human_sentence(ctx: Context) -> str:
    ai = ai_annotation_sentence(ctx)
    ai = f" {ai}" if ai else ""
    human = ctx.json(P_HUMAN)
    if not isinstance(human, dict) or human.get("status") == "not_filled":
        sheet = annotation_sheet_text(ctx)
        if sheet:
            return (
                f"Human annotation: {sheet}. No human labels were collected; "
                f"`python -m evaluation.human_annotation --score` computes human agreement "
                f"(`{P_HUMAN}`) if the A1/A2 columns are filled later.{ai}"
            )
        return f"Human annotation (50-item sheet): no human labels collected.{ai}"
    status = human.get("status", "unknown")
    parts = []
    for key in ("human_human", "A1_vs_llm", "A2_vs_llm", "consensus_vs_llm", "consensus_vs_nli"):
        block = human.get(key)
        if isinstance(block, dict) and block.get("n"):
            parts.append(
                f"{key.replace('_', ' ')} {kappa_text(block.get('kappa_3class'))} (n = {block['n']})"
            )
    joined = "; ".join(parts) if parts else "no agreement blocks"
    return f"Human annotation (status `{status}`): {joined}. {source(P_HUMAN)}{ai}"


#: Figures that draw NLI-judge bars or panels next to the LLM judge.
NLI_FIGURES: frozenset[str] = frozenset(
    {"hallucination_rate_by_condition", "claim_support_stacked"}
)


def figure_note(ctx: Context, name: str, summary: dict[str, Any] | None) -> str:
    """Reading note under a figure, derived from the data the figure was drawn from."""
    if summary is None:
        return ""
    if name in NLI_FIGURES:
        cov = nli_coverage(summary)
        if cov["partial"] or cov["none"]:
            shown = ", ".join(c for c, _, _ in cov["complete"]) or "no condition"
            blank = ", ".join(c for c, _, _ in cov["partial"] + cov["none"])
            return (
                f"NLI judge: {nli_coverage_text(summary)}. No partial-subset NLI bar is drawn: "
                f"its rates are shown only for the conditions it scored at >= 95% of turns "
                f"({shown}); {blank} are shown as n/a or left empty."
            )
    if name == "latency_by_condition":
        workers = study_workers(ctx, summary)
        return (
            "Cache-imputed study latencies, measured while the study ran with parallel workers"
            + (f" (up to {max(workers)} per run process)" if workers else "")
            + " on one shared local GPU. They compare conditions under the same load and are not "
            "the latency a single user sees."
        )
    return ""


def figure_gallery(ctx: Context, summary: dict[str, Any] | None = None) -> list[str]:
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
        note = figure_note(ctx, name, summary)
        if note:
            out += [note, ""]
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


def _record_run_ids(ctx: Context, rel: str) -> set[str]:
    """``run_id`` values of the records in a JSONL run file (empty when unreadable)."""
    ids: set[str] = set()
    try:
        with ctx.path(rel).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    run_id = json.loads(line).get("run_id")
                except (ValueError, AttributeError):
                    continue
                if isinstance(run_id, str):
                    ids.add(run_id)
    except OSError:
        return set()
    return ids


def run_commits(ctx: Context, results_dir: str) -> list[str]:
    """Short git commits of the runs that produced a model-selection directory's records.

    Only manifests whose run id occurs in the kept ``*.jsonl`` records count, so the
    manifests of earlier, superseded attempts left in the same folder are ignored. When
    no record names a run id, every manifest in the folder is used.
    """
    commits: set[str] = set()
    for model_dir in ctx.glob(f"{results_dir}/runs/*"):
        run_ids: set[str] = set()
        for rel in ctx.glob(f"{model_dir}/*.jsonl"):
            run_ids |= _record_run_ids(ctx, rel)
        for rel in ctx.glob(f"{model_dir}/manifest_*.json"):
            manifest = ctx.json(rel)
            if run_ids and dig(manifest, "run_id") not in run_ids:
                stem = Path(rel).stem.removeprefix("manifest_")
                if stem not in run_ids:
                    continue
            commit = dig(manifest, "git.commit")
            if isinstance(commit, str) and commit:
                commits.add(commit[:7] + ("+dirty" if dig(manifest, "git.dirty") else ""))
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
    commits = run_commits(ctx, Path(P_MODEL_SELECTION).parent.as_posix())
    decision = Path(P_MODEL_SELECTION).parent.as_posix() + "/DECISION.md"
    out.append(
        f"Run of {date_of(ms.get('generated_at'))}, split `{ms.get('split', '?')}`"
        + (f", run manifests at commit(s) {', '.join(commits)}" if commits else "")
        + ". "
        + source(P_MODEL_SELECTION)
        + (f" Decision and paired tests: `{decision}`." if ctx.exists(decision) else "")
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
        f"{organisation_count(ctx.json(P_CORPUS))} organisations (joint labels split); "
        f"{totals.get('table_chunks', '—')} table "
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
        f"errors {m.get('errors', '?')} ({date_of(live.get('generated_at'))}"
        f"{commit_of(live)}, `{P_AQ}`)"
    )


def commit_of(report: dict[str, Any]) -> str:
    """``, commit abc1234+dirty`` from a benchmark report's metadata (empty when absent)."""
    commit = dig(report, "metadata.git_commit")
    return f", commit `{commit}`" if isinstance(commit, str) and commit else ""


def gate3_of(report: dict[str, Any]) -> str:
    """``, Gate 3 on`` / ``, Gate 3 off`` from a benchmark report's metadata (empty when absent)."""
    flag = dig(report, "metadata.grounding_verifier")
    if not isinstance(flag, bool):
        return ""
    return ", Gate 3 on" if flag else ", Gate 3 off"


def gate3_flag(report: dict[str, Any]) -> str:
    """``on`` / ``off`` / ``—`` for a table cell."""
    flag = dig(report, "metadata.grounding_verifier")
    return "—" if not isinstance(flag, bool) else ("on" if flag else "off")


def run_label(report: dict[str, Any], rel: str) -> str:
    """``2026-10-11, commit `abc1234+dirty`, Gate 3 off, `path` `` for one live report."""
    return (
        f"{date_of(report.get('timestamp') or report.get('generated_at'))}{commit_of(report)}"
        f"{gate3_of(report)}, `{rel}`"
    )


#: Commit of the brand-name robustness fix (ADL-059). A live run counts as "after the fix"
#: only when the commit its report records contains this commit (checked with git).
BRAND_FIX_COMMIT = "02d9774"


def commit_contains(ctx: Context, commit: Any, ancestor: str) -> bool | None:
    """True/False when ``commit`` (``abc1234`` or ``abc1234+dirty``) contains ``ancestor``.

    None when it cannot be decided (no commit recorded, git unavailable, unknown commit).
    The answer depends only on the repository history, so the rendered text is stable.
    """
    if not isinstance(commit, str) or not commit:
        return None
    sha = commit.split("+", 1)[0].strip()
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return None
    key = f"git-contains:{sha}:{ancestor}"
    if key in ctx._cache:
        return ctx._cache[key]
    result: bool | None = None
    try:
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, sha],
            cwd=ctx.root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        result = True if proc.returncode == 0 else False if proc.returncode == 1 else None
    except (OSError, subprocess.SubprocessError):
        result = None
    ctx._cache[key] = result
    return result


def fix_status(ctx: Context, report: dict[str, Any]) -> str:
    """``after`` / ``before`` / ``unknown`` relative to the brand-name fix, from the report's commit."""
    contains = commit_contains(ctx, dig(report, "metadata.git_commit"), BRAND_FIX_COMMIT)
    return "unknown" if contains is None else ("after" if contains else "before")


def _pct_value(value: Any) -> str:
    """A value that is already a percentage (``48.33`` → ``48.3%``)."""
    f = _num(value)
    return "—" if f is None else f"{f:.1f}%"


def rabbits_line(report: dict[str, Any], rel: str) -> str:
    return (
        f"{report.get('pairs', '?')} brand–generic pairs ({report.get('indian_pairs', '?')} Indian), "
        f"action consistency {fixed(report.get('action_consistency_pct'), 1)}%, extraction "
        f"consistency {fixed(report.get('extraction_consistency_pct'), 1)}%, brand normalised to "
        f"its generic {fixed(report.get('brand_normalised_pct'), 1)}%, mean citation Jaccard "
        f"{fixed(report.get('citation_jaccard_mean'), 3)}, errors {report.get('errors', '?')} "
        f"({run_label(report, rel)})"
    )


def rabbits_runs(ctx: Context) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(live run kept from before the brand-name fix, newer live run at the default path)."""
    before = live_or_none(ctx.json(P_RABBITS_BEFORE))
    after = live_or_none(ctx.json(P_RABBITS))
    if after and before and after.get("timestamp") == before.get("timestamp"):
        after = None  # the default path still holds a copy of the pre-fix run
    return before, after


def rabbits_before_text(ctx: Context) -> str | None:
    before, _ = rabbits_runs(ctx)
    return rabbits_line(before, P_RABBITS_BEFORE) if before else None


def rabbits_flips(report: dict[str, Any]) -> dict[str, int] | None:
    """Action flips counted from ``items_detail`` (brand and generic message routed differently)."""
    items = [i for i in report.get("items_detail") or [] if isinstance(i, dict)]
    if not items:
        return None
    answered = {"ANSWER", "GENERALIZE"}
    flips = [i for i in items if i.get("action_match") is False]
    brand = sum(
        1
        for i in flips
        if i.get("brand_action") == "ABSTAIN" and i.get("generic_action") in answered
    )
    generic = sum(
        1
        for i in flips
        if i.get("generic_action") == "ABSTAIN" and i.get("brand_action") in answered
    )
    return {
        "flips": len(flips),
        "brand_declined": brand,
        "generic_declined": generic,
        "other": len(flips) - brand - generic,
    }


RABBITS_PCT_ROWS: tuple[tuple[str, str], ...] = (
    ("Action consistency, all items", "action_consistency_pct"),
    ("Action consistency, general template", "by_template.general.action_consistency_pct"),
    ("Action consistency, personal template", "by_template.personal.action_consistency_pct"),
    ("Action consistency, Indian brands", "by_origin.indian.action_consistency_pct"),
    ("Action consistency, international brands", "by_origin.international.action_consistency_pct"),
    ("Stored-generic consistency (personal template)", "extraction_consistency_pct"),
    ("Brand stored as its generic (personal template)", "brand_normalised_pct"),
)
RABBITS_NUM_ROWS: tuple[tuple[str, str], ...] = (
    ("Citation-set Jaccard, chunks", "citation_jaccard_mean"),
    ("Citation-set Jaccard, guidelines", "guideline_jaccard_mean"),
)


def rabbits_compare_rows(before: dict[str, Any], after: dict[str, Any]) -> list[list[str]]:
    rows = [
        [label, _pct_value(dig(before, path)), _pct_value(dig(after, path))]
        for label, path in RABBITS_PCT_ROWS
    ]
    rows += [
        [label, fixed(dig(before, path)), fixed(dig(after, path))]
        for label, path in RABBITS_NUM_ROWS
    ]
    fb, fa = rabbits_flips(before) or {}, rabbits_flips(after) or {}
    for label, key in (
        ("Action flips (of the items above)", "flips"),
        ("– brand declined (ABSTAIN) while the generic was answered", "brand_declined"),
        ("– generic declined while the brand was answered", "generic_declined"),
        ("– other differences (e.g. ANSWER vs GENERALIZE)", "other"),
    ):
        rows.append([label, str(fb.get(key, "—")), str(fa.get(key, "—"))])

    def seconds(report: dict[str, Any]) -> str:
        ms = _num(report.get("latency_p50_ms"))
        return "—" if ms is None else f"{ms / 1000:.1f} s"

    rows += [
        ["Median turn latency (client)", seconds(before), seconds(after)],
        [
            "Pipeline turns / harness errors",
            f"{before.get('turns_run', '—')} / {before.get('errors', '—')}",
            f"{after.get('turns_run', '—')} / {after.get('errors', '—')}",
        ],
        [
            "Code (commit recorded by the run)",
            f"`{dig(before, 'metadata.git_commit') or '—'}`",
            f"`{dig(after, 'metadata.git_commit') or '—'}`",
        ],
        ["Gate 3 during the run", gate3_flag(before), gate3_flag(after)],
        [
            "Run date",
            date_of(before.get("timestamp")),
            date_of(after.get("timestamp")),
        ],
    ]
    return rows


def rabbits_compare_text(ctx: Context) -> str | None:
    before, after = rabbits_runs(ctx)
    if not (before and after):
        return None
    fb, fa = rabbits_flips(before) or {}, rabbits_flips(after) or {}
    status = fix_status(ctx, after)
    after_word = {
        "after": "after the brand-name fix",
        "before": "latest run (still before the fix)",
    }.get(status, "latest run")
    return (
        f"live RABBITS on the same {after.get('pairs', '?')} brand–generic pairs "
        f"({after.get('indian_pairs', '?')} Indian): action consistency "
        f"{_pct_value(before.get('action_consistency_pct'))} → "
        f"{_pct_value(after.get('action_consistency_pct'))}, Indian brands "
        f"{_pct_value(dig(before, 'by_origin.indian.action_consistency_pct'))} → "
        f"{_pct_value(dig(after, 'by_origin.indian.action_consistency_pct'))}, stored-generic "
        f"consistency {_pct_value(before.get('extraction_consistency_pct'))} → "
        f"{_pct_value(after.get('extraction_consistency_pct'))}, brand stored as its generic "
        f"{_pct_value(before.get('brand_normalised_pct'))} → "
        f"{_pct_value(after.get('brand_normalised_pct'))}, action flips {fb.get('flips', '?')} → "
        f"{fa.get('flips', '?')} (brand declined while the generic was answered: "
        f"{fb.get('brand_declined', '?')} → {fa.get('brand_declined', '?')}); before the fix: "
        f"{run_label(before, P_RABBITS_BEFORE)}; {after_word}: {run_label(after, P_RABBITS)}"
    )


def rabbits_text(ctx: Context) -> str:
    """Latest live RABBITS run, plus the run kept from before the brand-name fix (ADL-059).

    The run in ``reports/rabbits_before_fix/`` is always labelled as the pre-fix run. The run
    at the default path is labelled "after the fix" only when the commit it records contains
    the fix commit (``git merge-base --is-ancestor``); otherwise only by its commit and date.
    """
    data = ctx.json(P_RABBITS)
    before, after = rabbits_runs(ctx)
    before_part = rabbits_line(before, P_RABBITS_BEFORE) if before else ""
    if not after:
        ctx.pending.append("live RABBITS run")
        mode = data.get("mode") if isinstance(data, dict) else None
        why = (
            f"the current `{P_RABBITS}` is {mode}-mode (deterministic components only)"
            if mode and mode != "live"
            else f"no newer live run in `{P_RABBITS}`"
        )
        pending = (
            f"{PENDING} for the live re-run ({why}; run "
            "`python -m tests.adversarial.rabbits_runner --mode live`)"
        )
        return (
            f"Before the brand-name fix: {before_part}. Latest code: {pending}"
            if before
            else pending
        )
    label = {"after": "After the brand-name fix", "before": "Latest live run (before the fix)"}.get(
        fix_status(ctx, after), "Latest live run"
    )
    latest = f"{label}: {rabbits_line(after, P_RABBITS)}"
    return f"{latest}. Before the fix: {before_part}" if before else latest


def adversarial_line(report: dict[str, Any], rel: str) -> str:
    c = as_dict(report.get("counts"))
    return (
        f"{c.get('passed', '?')}/{c.get('payloads', '?')} payloads passed; system-prompt leaks "
        f"{c.get('leaked_system_prompt', '?')}, fabricated prescriptions "
        f"{c.get('fabricated_prescription', '?')}, unsafe outputs {c.get('unsafe_output', '?')}, "
        f"emergency payloads escalated {c.get('emergency_escalated', '?')}/"
        f"{c.get('emergency_payloads', '?')} ({run_label(report, rel)})"
    )


def adversarial_runs(ctx: Context) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(live run kept from before the brand-name fix, newer live run at the default path)."""
    before = live_or_none(ctx.json(P_ADVERSARIAL_BEFORE))
    after = live_or_none(ctx.json(P_ADVERSARIAL))
    if before and after and before.get("generated_at") == after.get("generated_at"):
        after = None
    if after is not None and not isinstance(after.get("counts"), dict):
        after = None
    if before is not None and not isinstance(before.get("counts"), dict):
        before = None
    return before, after


def adversarial_changes(
    before: dict[str, Any], after: dict[str, Any]
) -> list[tuple[str, str, str, str]]:
    """Payloads whose final action differs: (payload, action before, action after, passed after)."""
    old = {
        str(r.get("payload_id")): str(r.get("action"))
        for r in before.get("results") or []
        if isinstance(r, dict)
    }
    out = []
    for r in after.get("results") or []:
        if not isinstance(r, dict):
            continue
        pid, action = str(r.get("payload_id")), str(r.get("action"))
        if pid in old and old[pid] != action:
            out.append((pid, old[pid], action, "yes" if r.get("passed") else "no"))
    return out


def adversarial_compare_text(ctx: Context) -> str | None:
    before, after = adversarial_runs(ctx)
    if not (before and after):
        return None
    cb, ca = as_dict(before.get("counts")), as_dict(after.get("counts"))
    keys = ("leaked_system_prompt", "fabricated_prescription", "unsafe_output")
    same_checks = all(cb.get(k) == ca.get(k) for k in keys) and (
        cb.get("emergency_escalated"),
        cb.get("emergency_payloads"),
    ) == (ca.get("emergency_escalated"), ca.get("emergency_payloads"))
    checks = (
        f"system-prompt leaks {ca.get('leaked_system_prompt')}, fabricated prescriptions "
        f"{ca.get('fabricated_prescription')}, unsafe outputs {ca.get('unsafe_output')} and "
        f"emergency payloads escalated {ca.get('emergency_escalated')}/"
        f"{ca.get('emergency_payloads')} in both runs"
        if same_checks
        else "check counts before and after differ, see the table"
    )
    changes = adversarial_changes(before, after)
    changed = (
        f"; final action changed on {len(changes)} payload{'s' if len(changes) != 1 else ''} ("
        + ", ".join(f"{p} {a} → {b}" for p, a, b, _ in changes)
        + ")"
        if changes
        else "; no payload changed its final action"
    )
    after_word = {"after": "after the fix", "before": "latest run"}.get(
        fix_status(ctx, after), "latest run"
    )
    return (
        f"{cb.get('passed', '?')}/{cb.get('payloads', '?')} payloads passed before the brand-name "
        f"fix and {ca.get('passed', '?')}/{ca.get('payloads', '?')} {after_word}; {checks}{changed} "
        f"(before: {run_label(before, P_ADVERSARIAL_BEFORE)}; {after_word}: "
        f"{run_label(after, P_ADVERSARIAL)})"
    )


def adversarial_text(ctx: Context) -> str:
    before, after = adversarial_runs(ctx)
    latest = after or before
    if not latest:
        ctx.pending.append("live adversarial run")
        return (
            f"{PENDING} (`{P_ADVERSARIAL}` not generated; run "
            "`python -m tests.adversarial.adversarial_runner`)"
        )
    if after and before:
        label = {"after": "After the brand-name fix", "before": "Latest live run"}.get(
            fix_status(ctx, after), "Latest live run"
        )
        return (
            f"{label}: {adversarial_line(after, P_ADVERSARIAL)}. Before the fix: "
            f"{adversarial_line(before, P_ADVERSARIAL_BEFORE)}"
        )
    rel = P_ADVERSARIAL if after else P_ADVERSARIAL_BEFORE
    return adversarial_line(latest, rel)


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


LOAD_KINDS: tuple[str, ...] = (
    "emergency",
    "small_talk",
    "general",
    "out_of_domain",
    "personal",
    "clarification",
)


def load_kind_rows(ctx: Context) -> list[list[str]] | None:
    """p50 latency (s) per query kind and concurrency level, from ``by_kind``."""
    data = ctx.json(P_LOAD)
    levels = data.get("levels") if isinstance(data, dict) else None
    if not isinstance(levels, list) or not levels:
        return None
    rows = []
    for lv in levels:
        if not isinstance(lv, dict):
            continue
        by_kind = as_dict(lv.get("by_kind"))
        cells = []
        for kind in LOAD_KINDS:
            ms = _num(as_dict(by_kind.get(kind)).get("p50_ms"))
            cells.append("—" if ms is None else f"{ms / 1000:.2f}")
        rows.append([str(lv.get("users", "—")), *cells])
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
        ["RABBITS brand↔generic robustness (live)", rabbits_text(ctx)],
        ["Adversarial prompt-injection suite (live)", adversarial_text(ctx)],
        ["Load test (1/5/10 concurrent users)", load_text(ctx)],
    ]
    out += [table(("Check", "Result"), rows), ""]
    aq = live_or_none(ctx.json(P_AQ))
    if aq:
        status = fix_status(ctx, aq)
        if status == "before":
            out += [
                "The live MedBridge-AQ run above was recorded on code from before the brand-name "
                f"fix (commit `{dig(aq, 'metadata.git_commit')}`, which does not contain `{BRAND_FIX_COMMIT}`); "
                "it has not been re-run since.",
                "",
            ]

    before, after = rabbits_runs(ctx)
    if before and after:
        out += [
            f"**RABBITS before vs {'after the brand-name fix' if fix_status(ctx, after) == 'after' else 'latest run'}** "
            f"(same {after.get('pairs', '?')} brand → generic pairs and the same two message templates; "
            "a brand and its generic should get the same action and the same stored medicine):",
            "",
            table(("Metric", "Before", "After"), rabbits_compare_rows(before, after), "lrr"),
            "",
            "Flip counts are computed from `items_detail` in each report. "
            + source(P_RABBITS_BEFORE, P_RABBITS),
            "",
        ]

    abefore, aafter = adversarial_runs(ctx)
    if abefore and aafter:
        cb, ca = as_dict(abefore.get("counts")), as_dict(aafter.get("counts"))
        arows = [
            [
                "Payloads passed",
                f"{cb.get('passed')}/{cb.get('payloads')}",
                f"{ca.get('passed')}/{ca.get('payloads')}",
            ],
            [
                "System-prompt leaks",
                str(cb.get("leaked_system_prompt")),
                str(ca.get("leaked_system_prompt")),
            ],
            [
                "Fabricated prescriptions",
                str(cb.get("fabricated_prescription")),
                str(ca.get("fabricated_prescription")),
            ],
            ["Unsafe outputs", str(cb.get("unsafe_output")), str(ca.get("unsafe_output"))],
            [
                "Emergency payloads escalated",
                f"{cb.get('emergency_escalated')}/{cb.get('emergency_payloads')}",
                f"{ca.get('emergency_escalated')}/{ca.get('emergency_payloads')}",
            ],
            ["Harness errors", str(cb.get("errors")), str(ca.get("errors"))],
            [
                "Code (commit recorded by the run)",
                f"`{dig(abefore, 'metadata.git_commit') or '—'}`",
                f"`{dig(aafter, 'metadata.git_commit') or '—'}`",
            ],
            ["Gate 3 during the run", gate3_flag(abefore), gate3_flag(aafter)],
            ["Run date", date_of(abefore.get("generated_at")), date_of(aafter.get("generated_at"))],
        ]
        changes = adversarial_changes(abefore, aafter)
        change_text = (
            "Payloads whose final action changed: "
            + "; ".join(f"{p} {a} → {b} (passed: {ok})" for p, a, b, ok in changes)
            + "."
            if changes
            else "No payload changed its final action."
        )
        out += [
            "**Adversarial suite before vs after the brand-name fix:**",
            "",
            table(("Check", "Before", "After"), arows, "lrr"),
            "",
            change_text + " " + source(P_ADVERSARIAL_BEFORE, P_ADVERSARIAL),
            "",
        ]

    load = load_rows(ctx)
    if load:
        out += [
            "**Load test** (burst: every simulated user sends its requests at once):",
            "",
            table(
                ("Users", "Requests", "Errors", "429s", "Error rate", "p50 ms", "p95 ms", "req/s"),
                load,
                "rrrrrrrr",
            ),
            "",
        ]
        kinds = load_kind_rows(ctx)
        if kinds:
            out += [
                "p50 latency per query kind, seconds (— = that kind was not sent at this level):",
                "",
                table(("Users", *LOAD_KINDS), kinds, "r" * (1 + len(LOAD_KINDS))),
                "",
            ]
        notes = [n for n in as_dict(ctx.json(P_LOAD)).get("notes") or [] if isinstance(n, str)]
        if notes:
            out += [f"Note recorded with the run: {notes[0]}", ""]
        out += [source(P_LOAD), ""]
    return out


# ----------------------------------------------------------------------
# Latency before vs after (scripts/latency_probe.py files, evaluation.perf table)
# ----------------------------------------------------------------------


def probe_files(ctx: Context) -> dict[str, str]:
    """``label`` → repo-relative path of every latency-probe file (newest file per label)."""
    found: dict[str, tuple[float, str]] = {}
    for rel in ctx.glob(f"{P_PERF_DIR}/*.json"):
        data = ctx.json(rel)
        if isinstance(data, dict) and isinstance(data.get("label"), str) and "warm_summary" in data:
            mtime = ctx.path(rel).stat().st_mtime
            label = data["label"]
            if label not in found or mtime > found[label][0]:
                found[label] = (mtime, rel)
    return {label: rel for label, (_, rel) in sorted(found.items())}


def latency_pair(ctx: Context) -> tuple[str | None, str | None]:
    """The before/after probe files the latency table compares (named in its Markdown header)."""
    md = ctx.path(P_LATENCY_TABLE_MD)
    if md.is_file():
        try:
            head = md.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            head = ""
        m = re.search(r"\(([\w.\-]+\.json)\s*→\s*([\w.\-]+\.json)", head)
        if m:
            b, a = f"{P_PERF_DIR}/{m.group(1)}", f"{P_PERF_DIR}/{m.group(2)}"
            if ctx.exists(b) and ctx.exists(a):
                return b, a
    probes = probe_files(ctx)
    return probes.get("before"), probes.get("after")


def model_of(rel: str) -> str:
    """``reports/performance/after_qwen3.5-4b.json`` → ``qwen3.5:4b`` (tag written by run_post_study.sh)."""
    stem = Path(rel).stem
    for prefix in ("after_warmup_", "after_", "before_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    head, sep, tail = stem.rpartition("-")
    return f"{head}:{tail}" if sep and head else stem


def scenario_names(ctx: Context) -> dict[str, str]:
    """``S5-t1`` → ``Lifestyle`` from the latency table written by evaluation.perf."""
    path = ctx.path(P_LATENCY_TABLE)
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return {
                r["Scenario turn"]: r.get("Scenario", "")
                for r in csv.DictReader(fh)
                if r.get("Scenario turn")
            }
    except (OSError, KeyError):
        return {}


def warm_by_turn(probe: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    """``S4-t2`` → (action, warm stats). A turn with several actions keeps the first one."""
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for key, stats in sorted(as_dict(probe.get("warm_summary")).items()):
        turn, _, action = key.partition(":")
        if isinstance(stats, dict) and turn not in out:
            out[turn] = (action, stats)
    return out


def cold_by_turn(probe: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """``S4-t2`` → (action, seconds) of the first (cold) pass."""
    out: dict[str, tuple[str, float]] = {}
    for c in probe.get("cold") or []:
        if not isinstance(c, dict):
            continue
        sec = _num(c.get("latency_s"))
        if sec is not None:
            out.setdefault(f"{c.get('scenario')}-t{c.get('turn')}", (str(c.get("action")), sec))
    return out


def latency_changes(
    ctx: Context, before: dict[str, Any], after: dict[str, Any]
) -> dict[str, list[str]]:
    """Turns that got faster, slower or stayed the same (warm p50; ±10 % counts as the same)."""
    names = scenario_names(ctx)
    b, a = warm_by_turn(before), warm_by_turn(after)
    out: dict[str, list[str]] = {"faster": [], "slower": [], "same": [], "no_llm": []}
    for turn in sorted(set(b) & set(a)):
        (ab, sb), (aa, sa) = b[turn], a[turn]
        pb, pa = _num(sb.get("p50")), _num(sa.get("p50"))
        if pb is None or pa is None:
            continue
        action = aa if aa == ab else f"{ab} → {aa}"
        name = names.get(turn)
        desc = f"{turn} ({name + ', ' if name else ''}{action}) {pb:.2f} s → {pa:.2f} s"
        if max(pb, pa) < 0.1:
            out["no_llm"].append(desc)
        elif pa > 0 and pb / pa >= 1.1:
            out["faster"].append(desc)
        elif pa > 0 and pb / pa <= 1 / 1.1:
            out["slower"].append(desc)
        else:
            out["same"].append(desc)
    return out


def cold_changes(ctx: Context, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    names = scenario_names(ctx)
    b, a = cold_by_turn(before), cold_by_turn(after)
    out = []
    for turn in sorted(set(b) & set(a)):
        (ab, sb), (aa, sa) = b[turn], a[turn]
        action = aa if aa == ab else f"{ab} → {aa}"
        name = names.get(turn)
        out.append(f"{turn} ({name + ', ' if name else ''}{action}) {sb:.2f} s → {sa:.2f} s")
    return out


def after_probe_files(ctx: Context) -> list[str]:
    """Probe files of the final code with warm repeats (labels ``after_<model>``)."""
    return [
        rel
        for label, rel in probe_files(ctx).items()
        if label.startswith("after_") and not label.startswith("after_warmup_")
    ]


def stage_seconds(stages: dict[str, Any], *names: str) -> str:
    vals = [_num(stages.get(n)) for n in names]
    present = [v for v in vals if v is not None]
    return f"{sum(present) / 1000:.2f}" if present else "—"


def gate3_stage_range(ctx: Context) -> tuple[float, float, list[str]] | None:
    """Min and max warm p50 of the Gate 3 (verification) stage, seconds, over all after files."""
    values: list[float] = []
    files: list[str] = []
    for rel in after_probe_files(ctx):
        probe = as_dict(ctx.json(rel))
        found = False
        for stats in as_dict(probe.get("warm_summary")).values():
            ms = _num(as_dict(as_dict(stats).get("server_stage_ms_p50")).get("verification"))
            if ms is not None:
                values.append(ms / 1000)
                found = True
        if found:
            files.append(rel)
    return (min(values), max(values), files) if values else None


def gate3_stage_text(ctx: Context) -> str | None:
    rng = gate3_stage_range(ctx)
    if not rng:
        return None
    low, high, files = rng
    span = f"{low:.2f} s" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f}–{high:.2f} s"
    return (
        f"the Gate 3 verification stage takes {span} per ANSWER turn (server-side warm p50; "
        + ", ".join(f"`{f}`" for f in files)
        + ")"
    )


def section_latency(ctx: Context) -> list[str]:
    out = ["### Latency before vs after optimisation", ""]
    before_rel, after_rel = latency_pair(ctx)
    before = as_dict(ctx.json(before_rel)) if before_rel else {}
    after = as_dict(ctx.json(after_rel)) if after_rel else {}
    table_path = ctx.path(P_LATENCY_TABLE)
    rendered_table = False
    if table_path.is_file():
        try:
            with table_path.open(encoding="utf-8", newline="") as fh:
                reader = list(csv.reader(fh))
        except OSError as exc:
            ctx.warnings.append(f"could not read {P_LATENCY_TABLE}: {exc}")
            reader = []
        if len(reader) > 1:
            if before_rel and after_rel:
                same_model = model_of(before_rel) == model_of(after_rel)
                n_warm = {str(s.get("n")) for _, s in warm_by_turn(after).values()}
                gate3_on = any(
                    "verification" in as_dict(stats.get("server_stage_ms_p50"))
                    for _, stats in warm_by_turn(after).values()
                )
                out += [
                    f"Before: `{before_rel}` (the received code). After: `{after_rel}` (the code "
                    "after the optimisations, measured before commit `16e594b`"
                    + (", with Gate 3 running on its ANSWER turns" if gate3_on else "")
                    + "). "
                    + (
                        f"Both used `{model_of(after_rel)}`, so the comparison is like for like. "
                        if same_model
                        else f"Models differ (`{model_of(before_rel)}` vs `{model_of(after_rel)}`). "
                    )
                    + "Cold = the first request after a server start with the model not yet "
                    "loaded; warm = repeated requests"
                    + (f" (n = {n_warm.pop()} per turn)" if len(n_warm) == 1 else "")
                    + ".",
                    "",
                ]
            out += [
                table(reader[0], reader[1:]),
                "",
                source(P_LATENCY_TABLE, *(p for p in (before_rel, after_rel) if p)),
                "",
            ]
            rendered_table = True
    if not rendered_table:
        out.append(
            pending_note(
                ctx,
                "the before/after latency table has not been written",
                "Run `python scripts/latency_probe.py --label after_<model> ...` then "
                "`python -m evaluation.perf --before ... --after ...`.",
            )
        )
        out.append("")
        return out

    if before and after:
        ch = latency_changes(ctx, before, after)
        notes = []
        if ch["faster"]:
            notes.append("Faster (warm p50): " + "; ".join(ch["faster"]) + ".")
        if ch["slower"]:
            notes.append("Slower (warm p50): " + "; ".join(ch["slower"]) + ".")
        if ch["same"]:
            notes.append("Within ±10 %: " + "; ".join(ch["same"]) + ".")
        if ch["no_llm"]:
            notes.append(
                "No LLM call, below 0.1 s in both versions: " + "; ".join(ch["no_llm"]) + "."
            )
        cold = cold_changes(ctx, before, after)
        if cold:
            notes.append("First request (cold): " + "; ".join(cold) + ".")
        g3 = gate3_stage_text(ctx)
        if g3:
            notes.append(f"Gate 3 is new in the after version: {g3[0].lower()}{g3[1:]}.")
        abstain = [
            (turn, stats)
            for turn, (action, stats) in warm_by_turn(after).items()
            if action == "ABSTAIN"
        ]
        for turn, stats in abstain:
            g2 = _num(as_dict(stats.get("server_stage_ms_p50")).get("evidence_gate"))
            if g2 is not None:
                notes.append(
                    f"On the ABSTAIN turn {turn}, Gate 2 alone takes {g2 / 1000:.2f} s (warm p50, "
                    f"`{after_rel}`). The before file records no per-stage timings, so which stage "
                    "grew cannot be measured from these files."
                )
        if notes:
            out += ["Reading the table (computed from the two probe files):", ""]
            out += [f"- {n}" for n in notes]
            out.append("")

    name, caption = LATENCY_FIGURE
    if ctx.exists(f"{FIGURES_DIR}/{name}.png"):
        out += [f"![{caption}]({FIGURES_DIR}/{name}.png)", ""]

    after_files = after_probe_files(ctx)
    if after_files:
        out += [
            "Where the time goes in the post-study code with Gate 3 on (the probes ran before "
            "commit `16e594b`, ADL-061; server-side warm p50 per stage, seconds; "
            "retrieval is served from the in-process cache on repeated questions, reranking is not):",
            "",
        ]
        rows = []
        for rel in after_files:
            probe = as_dict(ctx.json(rel))
            for turn, (action, stats) in warm_by_turn(probe).items():
                st = as_dict(stats.get("server_stage_ms_p50"))
                total = _num(stats.get("server_total_ms_p50"))
                rows.append(
                    [
                        model_of(rel),
                        turn,
                        action,
                        stage_seconds(st, "extraction"),
                        stage_seconds(st, "context_gate"),
                        stage_seconds(st, "retrieval", "rerank"),
                        stage_seconds(st, "evidence_gate"),
                        stage_seconds(st, "generation"),
                        stage_seconds(st, "verification"),
                        "—" if total is None else f"{total / 1000:.2f}",
                    ]
                )
        out += [
            table(
                (
                    "Model",
                    "Turn",
                    "Action",
                    "Extractor",
                    "Gate 1",
                    "Retrieval + rerank",
                    "Gate 2",
                    "Generator",
                    "Gate 3",
                    "Server total",
                ),
                rows,
                "lllrrrrrrr",
            ),
            "",
            "Stage p50s are taken separately over the repeats, so they need not add up to the "
            "total p50. " + source(*after_files),
            "",
        ]

    probes = probe_files(ctx)
    wrows = []
    wfiles: list[str] = []
    for rel in after_files:
        label = as_dict(ctx.json(rel)).get("label")
        tag = str(label)[len("after_") :] if isinstance(label, str) else ""
        wrel = probes.get(f"after_warmup_{tag}")
        if not wrel:
            continue
        off, on = as_dict(ctx.json(rel)), as_dict(ctx.json(wrel))
        c_off, c_on, warm = cold_by_turn(off), cold_by_turn(on), warm_by_turn(off)
        for turn in sorted(set(c_off) | set(c_on)):
            action = (c_off.get(turn) or c_on.get(turn) or ("—", 0.0))[0]
            wrows.append(
                [
                    model_of(rel),
                    turn,
                    action,
                    f"{c_off[turn][1]:.2f}" if turn in c_off else "—",
                    f"{c_on[turn][1]:.2f}" if turn in c_on else "—",
                    fixed(warm[turn][1].get("p50"), 2) if turn in warm else "—",
                ]
            )
        wfiles += [rel, wrel]
    if wrows:
        out += [
            "First request after a server start, without and with the start-up warm-up "
            "(`WARMUP_ON_STARTUP`), seconds:",
            "",
            table(
                ("Model", "Turn", "Action", "Warm-up off (cold)", "Warm-up on", "Warm p50"),
                wrows,
                "lllrrr",
            ),
            "",
            source(*wfiles),
            "",
        ]
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
    # scripts/test_summary.py writes {"totals": {"passed", "collected", "skipped", ...}};
    # a flat {"passed", "failed", "skipped"} file is accepted too.
    totals = as_dict(custom.get("totals")) if isinstance(custom, dict) else {}
    flat = custom if isinstance(custom, dict) and custom.get("passed") is not None else {}
    counts = totals if totals.get("passed") is not None else flat
    if counts:
        failed = counts.get("failed", 0) or 0
        errors = counts.get("errors", 0) or 0
        exit_code = custom.get("exit_code") if isinstance(custom, dict) else None
        parts.append(
            f"{counts.get('passed')} passed, {failed} failed"
            + (f", {errors} errors" if errors else "")
            + f", {counts.get('skipped', 0) or 0} skipped"
            + (f" of {counts['collected']} collected" if counts.get("collected") else "")
            + (f", exit code {exit_code}" if exit_code is not None else "")
            + f" (run of {date_of(as_dict(custom).get('generated_at'))}, `{custom_rel}`)"
        )
        overall = _num(dig(custom, "coverage.overall_percent"))
        if overall is not None and not (cov and cov[1] is not None):
            parts.append(f"line coverage {overall:.1f}% (`{custom_rel}`)")
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
        "value",
        "routing",
        "latency",
        "degraded",
        "test",
        "cochran",
        "kappa",
        "split",
        "models",
        "nli_coverage",
        "commits",
    }
)
KNOWN_SPEC_KINDS: frozenset[str] = SUMMARY_SPEC_KINDS | {
    "rabbits_before",
    "human",
    "ai_annotation",
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
    "rabbits_after",
    "rabbits_compare",
    "adversarial_before",
    "adversarial_compare",
    "perf_cold",
    "perf_changes",
    "perf_gate3",
    "perf_after",
    "annotation",
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
    if kind == "value" and len(parts) in (3, 4):
        digits = int(parts[3]) if len(parts) == 4 and parts[3].isdigit() else 2
        text = fixed(dig(summary, f"per_condition.{parts[1]}.{parts[2]}"), digits)
        return None if text == "—" else text
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
            f"per turn (cache-imputed, n = {lat.get('n')}; measured with parallel study workers "
            "on one shared local GPU, so a relative figure, not the latency a single user sees)"
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
        unit = "responses" if parts[1] == "response" else "claims"
        text = kappa_text(agreement.get(kappa_key))
        cov = nli_coverage(summary) if summary else {"complete": [], "partial": []}
        scored = [c for c, _, _ in cov["complete"]] + [
            f"{c} partial subset" for c, _, _ in cov["partial"]
        ]
        scope = f"; conditions both judges scored: {', '.join(scored)}" if scored else ""
        return None if text == "—" else f"{text} (n = {n} {unit}{scope})"
    if kind == "nli_coverage" and summary is not None:
        return nli_coverage_text(summary)
    if kind == "commits" and summary is not None:
        commits = study_commits(summary)
        return ", ".join(f"`{c}`" for c in commits) if commits else None
    if kind == "rabbits_before":
        return rabbits_before_text(ctx)
    if kind == "ai_annotation":
        return ai_annotation_sentence(ctx) or None
    if kind == "human" and len(parts) == 2:
        human = ctx.json(P_HUMAN)
        if not isinstance(human, dict):
            if parts[1] == "status":
                sheet = annotation_sheet_text(ctx)
                return f"{sheet}; agreement not yet scored" if sheet else None
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
            f"({totals.get('table_chunks')} table chunks) from "
            f"{organisation_count(ctx.json(P_CORPUS))} organisations (joint labels split)"
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
    if kind == "rabbits_after":
        _, after = rabbits_runs(ctx)
        return rabbits_line(after, P_RABBITS) if after else None
    if kind == "rabbits_compare":
        return rabbits_compare_text(ctx)
    if kind == "adversarial_before":
        before, _ = adversarial_runs(ctx)
        return adversarial_line(before, P_ADVERSARIAL_BEFORE) if before else None
    if kind == "adversarial_compare":
        return adversarial_compare_text(ctx)
    if kind == "annotation":
        return annotation_sheet_text(ctx)
    if kind in ("perf", "perf_cold", "perf_changes"):
        before_rel, after_rel = latency_pair(ctx)
        if not (before_rel and after_rel and ctx.path(P_LATENCY_TABLE).is_file()):
            return None
        before, after = as_dict(ctx.json(before_rel)), as_dict(ctx.json(after_rel))
        files = f"`{before_rel}` → `{after_rel}`"
        if kind == "perf_cold":
            bits = cold_changes(ctx, before, after)
            return ("first request (cold) " + "; ".join(bits) + f" ({files})") if bits else None
        ch = latency_changes(ctx, before, after)
        if kind == "perf":
            bits = ch["no_llm"] + ch["same"] + ch["faster"] + ch["slower"]
            bits.sort()
            return ("warm p50 " + "; ".join(bits) + f" ({files})") if bits else None
        parts_out = []
        if ch["faster"]:
            parts_out.append("faster: " + "; ".join(ch["faster"]))
        if ch["slower"]:
            parts_out.append("slower: " + "; ".join(ch["slower"]))
        if ch["same"]:
            parts_out.append("within ±10 %: " + "; ".join(ch["same"]))
        if not parts_out:
            return None
        return (
            "warm p50 "
            + "; ".join(parts_out)
            + f" ({files}, same model: "
            + ("yes" if model_of(before_rel) == model_of(after_rel) else "no")
            + ")"
        )
    if kind == "perf_gate3":
        return gate3_stage_text(ctx)
    if kind == "perf_after" and len(parts) == 2:
        rel = probe_files(ctx).get(f"after_{parts[1]}")
        if not rel:
            return None
        names = scenario_names(ctx)
        bits = [
            f"{turn} ({names[turn] + ', ' if names.get(turn) else ''}{action}) "
            f"{fixed(stats.get('p50'), 2)} s"
            for turn, (action, stats) in warm_by_turn(as_dict(ctx.json(rel))).items()
        ]
        return (
            (f"warm p50 with `{model_of(rel)}`: " + "; ".join(bits) + f" (`{rel}`)")
            if bits
            else None
        )
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

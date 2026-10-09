"""
Calibration on the DEV split only (the test split is never used for tuning).

1. **NLI thresholds** — θ maximises Cohen's κ between the NLI judge's binary
   decision (entail ≥ θ) and the LLM judge's SUPPORTED-vs-not label on dev
   claims; θ_c (given θ) then maximises the 3-class κ. Ties prefer the value
   closest to 0.5.

2. **V1 calibrated Evidence Gate thresholds** — ``EVIDENCE_MIN_RERANK_ANSWER``,
   ``EVIDENCE_MIN_RERANK_ANY`` and ``EVIDENCE_MIN_ENTITY_COVERAGE`` are chosen
   by grid search on the C5 dev traces. For every turn where Gate 2 actually
   decided (``explanation.gate2.source == "llm"``) the V1 rule is replayed on
   the logged Gate-2 action and evidence features::

       ESCALATE stays ESCALATE
       max_rerank < MIN_RERANK_ANY                                  → ABSTAIN
       action == ANSWER and (max_rerank < MIN_RERANK_ANSWER
                             or entity_coverage < MIN_ENTITY_COVERAGE) → GENERALIZE
       otherwise the Gate-2 action

   This replays exactly the override logic of
   ``medbridge.ai.evidence_calibration.calibrated_evidence_decision`` (applied
   there to the majority of the permutation votes; C5 traces carry a single
   Gate-2 vote, so the vote component itself is not simulated).
   Objective: minimise the false-ANSWER rate on unanswerable dev turns (the
   action-level proxy for unsafe answers — content-level safety of a re-routed
   turn cannot be known without regenerating it) subject to coverage on
   answerable dev turns ≥ C5 coverage − 0.05; ties → higher coverage → higher
   strict routing accuracy → the least intrusive thresholds. Features are taken
   from the trace when Gate 2 logged them (``gate2.features``), else computed
   with the production ``evidence_features`` on the logged question and full
   evidence texts (without the session snapshot, which is not logged), else
   recomputed locally from the logged scores and texts.
   ``-100`` encodes "threshold disabled" (cross-encoder logits are ≫ −100).

Writes ``evaluation/results/calibration.json`` and prints the ``.env`` lines.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import metrics as mx
from evaluation.datasets.schema import ANSWERABLE_CATEGORIES, UNANSWERABLE_CATEGORIES
from evaluation.io import load_all_judgments, load_run_records, read_json, write_json
from evaluation.judges.common import CLAIM_LABELS, SUPPORTED
from evaluation.judges.nli_judge import nli_label
from evaluation.manifest import utc_now
from evaluation.outcomes import build_outcomes, filter_split, max_rerank_score
from evaluation.paths import CALIBRATION_PATH, JUDGMENTS_DIR, RUNS_DIR

DISABLED = -100.0
COVERAGE_SLACK = 0.05
ANSWER_LIKE = frozenset({"ANSWER", "GENERALIZE"})

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{3,}")
_STOPWORDS = frozenset(
    "what which when where with that this have about should could would there their they them "
    "your yours from into does doing been being much many some more most very just like also "
    "really still okay know need want take taking tell said says thing things good best right "
    "after before because while than then over under only even ever every other another".split()
)


# ----------------------------------------------------------------------
# NLI thresholds
# ----------------------------------------------------------------------


def nli_claim_pairs(outcomes: dict[str, list[Any]]) -> list[tuple[str, float, float]]:
    """(LLM label, NLI entailment, NLI contradiction) for every claim scored by both judges."""
    pairs: list[tuple[str, float, float]] = []
    for rows in outcomes.values():
        for o in rows:
            llm, nli = o.claims.get("llm"), o.claims.get("nli")
            if not llm or not nli:
                continue
            by_index = {c.get("index"): c for c in nli}
            for claim in llm:
                other = by_index.get(claim.get("index"))
                if other is not None and other.get("entailment") is not None:
                    pairs.append(
                        (
                            str(claim["label"]),
                            float(other["entailment"]),
                            float(other.get("contradiction", 0.0)),
                        )
                    )
    return pairs


def calibrate_nli(
    pairs: Sequence[tuple[str, float, float]], grid: Sequence[float] | None = None
) -> dict[str, Any] | None:
    """Choose θ (binary κ) then θ_c (3-class κ) on dev claims."""
    if not pairs:
        return None
    grid = list(grid) if grid is not None else [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    gold_binary = [str(label == SUPPORTED) for label, _, _ in pairs]
    gold_3 = [label for label, _, _ in pairs]

    def kappa_binary(theta: float) -> float:
        pred = [str(e >= theta) for _, e, _ in pairs]
        value = mx.cohen_kappa(gold_binary, pred, labels=["True", "False"]).kappa
        return value if math.isfinite(value) else -1.0

    theta = max(grid, key=lambda t: (kappa_binary(t), -abs(t - 0.5)))

    def kappa_3(theta_c: float) -> float:
        pred = [nli_label(e, c, theta, theta_c) for _, e, c in pairs]
        value = mx.cohen_kappa(gold_3, pred, labels=CLAIM_LABELS).kappa
        return value if math.isfinite(value) else -1.0

    theta_c = max(grid, key=lambda t: (kappa_3(t), -abs(t - 0.5)))
    return {
        "theta": theta,
        "theta_c": theta_c,
        "kappa_binary": kappa_binary(theta),
        "kappa_3class": kappa_3(theta_c),
        "n_claims": len(pairs),
        "grid": grid,
        "objective": "theta: max binary Cohen's kappa vs LLM judge; theta_c: max 3-class kappa given theta",
    }


# ----------------------------------------------------------------------
# V1 thresholds
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """V1 gate thresholds (``DISABLED`` switches a threshold off)."""

    min_rerank_answer: float
    min_rerank_any: float
    min_entity_coverage: float

    def env_lines(self) -> list[str]:
        return [
            f"EVIDENCE_MIN_RERANK_ANSWER={self.min_rerank_answer:g}",
            f"EVIDENCE_MIN_RERANK_ANY={self.min_rerank_any:g}",
            f"EVIDENCE_MIN_ENTITY_COVERAGE={self.min_entity_coverage:g}",
        ]


NO_OP = Thresholds(DISABLED, DISABLED, 0.0)


@dataclass(frozen=True)
class GateTrace:
    """One C5 dev turn with what Gate 2 saw and decided."""

    item_id: str
    turn_index: int
    category: str
    gold_action: str
    acceptable: tuple[str, ...]
    final_action: str
    gate2_action: str | None
    gate2_ran: bool
    max_rerank: float
    entity_coverage: float


def query_terms(text: str) -> set[str]:
    """Content terms of a query (lower-case, ≥ 4 letters, stop-words removed)."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def local_features(
    query: str, evidence_texts: Sequence[str], rerank_scores: Sequence[float]
) -> dict[str, float]:
    """``max_rerank``, ``mean_rerank`` and ``entity_coverage`` recomputed from logged traces."""
    terms = query_terms(query)
    corpus = " ".join(evidence_texts).lower()
    coverage = sum(1 for t in terms if t in corpus) / len(terms) if terms else 0.0
    return {
        "max_rerank": max(rerank_scores) if rerank_scores else DISABLED,
        "mean_rerank": float(np.mean(rerank_scores)) if rerank_scores else DISABLED,
        "entity_coverage": coverage,
    }


def record_features(record: dict[str, Any]) -> dict[str, float]:
    """Gate features of a run record: logged → production ``evidence_features`` → local."""
    explanation = record.get("explanation") or {}
    logged = (explanation.get("gate2") or {}).get("features") or {}
    if "max_rerank" in logged and "entity_coverage" in logged:
        return {k: float(v) for k, v in logged.items() if isinstance(v, int | float)}
    evidence = record.get("evidence_texts") or []
    question = str(record.get("user_text", ""))
    try:
        from medbridge.ai.evidence_calibration import evidence_features
        from medbridge.retrieval.hybrid_retriever import RetrievedChunk
    except ImportError:
        scores = [float(e["score_rerank"]) for e in evidence if e.get("score_rerank") is not None]
        return local_features(question, [str(e.get("text", "")) for e in evidence], scores)
    chunks = [
        RetrievedChunk(
            chunk_id=str(e.get("chunk_id")),
            chunk_text=str(e.get("text", "")),
            guideline_id=str(e.get("guideline_id", "")),
            section_title=str(e.get("section", "")),
            page_number=int(e.get("page") or 0),
            score_rrf=e.get("score_rrf"),
            score_rerank=e.get("score_rerank"),
        )
        for e in evidence
    ]
    return {k: float(v) for k, v in evidence_features(question, chunks, {}).items()}


def build_traces(
    records: Sequence[dict[str, Any]],
    features: Callable[[dict[str, Any]], dict[str, float]] = record_features,
) -> list[GateTrace]:
    """Gate traces from C5 run records (errored turns skipped)."""
    traces: list[GateTrace] = []
    for rec in records:
        if rec.get("error") or rec.get("action") is None:
            continue
        gate2 = (rec.get("explanation") or {}).get("gate2") or {}
        feats = features(rec) if gate2.get("source") == "llm" else {}
        rerank = max_rerank_score(rec)
        traces.append(
            GateTrace(
                item_id=str(rec["item_id"]),
                turn_index=int(rec["turn_index"]),
                category=str(rec.get("category", "")),
                gold_action=str(rec.get("gold_action", "")),
                acceptable=tuple(rec.get("acceptable_actions") or [rec.get("gold_action", "")]),
                final_action=str(rec["action"]),
                gate2_action=gate2.get("action"),
                gate2_ran=gate2.get("source") == "llm" and gate2.get("action") is not None,
                max_rerank=float(
                    feats.get("max_rerank", rerank if rerank is not None else DISABLED)
                ),
                entity_coverage=float(feats.get("entity_coverage", 0.0)),
            )
        )
    return traces


def threshold_rule(action: str, max_rerank: float, entity_coverage: float, thr: Thresholds) -> str:
    """The V1 decision rule (see module docstring)."""
    if action == "ESCALATE":
        return action
    if max_rerank < thr.min_rerank_any:
        return "ABSTAIN"
    if action == "ANSWER" and (
        max_rerank < thr.min_rerank_answer or entity_coverage < thr.min_entity_coverage
    ):
        return "GENERALIZE"
    return action


Rule = Callable[[str, float, float, Thresholds], str]


def simulated_action(trace: GateTrace, thr: Thresholds, rule: Rule = threshold_rule) -> str:
    """Final action had V1 used ``thr`` (downstream downgrades of an ANSWER are preserved)."""
    if not trace.gate2_ran or trace.gate2_action is None:
        return trace.final_action
    sim = rule(trace.gate2_action, trace.max_rerank, trace.entity_coverage, thr)
    if sim == trace.gate2_action:
        return trace.final_action
    return sim


def evaluate(
    traces: Sequence[GateTrace], thr: Thresholds, rule: Rule = threshold_rule
) -> dict[str, float]:
    """False-ANSWER rate (unanswerable), coverage (answerable) and strict routing accuracy."""
    unanswerable = answerable = false_answers = covered = correct = 0
    for t in traces:
        action = simulated_action(t, thr, rule)
        correct += int(action == t.gold_action)
        if t.category in UNANSWERABLE_CATEGORIES and "ANSWER" not in t.acceptable:
            unanswerable += 1
            false_answers += int(action == "ANSWER")
        if t.category in ANSWERABLE_CATEGORIES and t.gold_action in ANSWER_LIKE:
            answerable += 1
            covered += int(action in ANSWER_LIKE)
    return {
        "false_answer_rate": false_answers / unanswerable if unanswerable else math.nan,
        "coverage": covered / answerable if answerable else math.nan,
        "routing_strict": correct / len(traces) if traces else math.nan,
        "n_unanswerable": unanswerable,
        "n_answerable": answerable,
        "n_turns": len(traces),
    }


def threshold_grid(traces: Sequence[GateTrace], n_quantiles: int = 21) -> dict[str, list[float]]:
    """Candidate values from the observed max-rerank distribution (plus 'disabled')."""
    observed = [t.max_rerank for t in traces if t.gate2_ran and t.max_rerank > DISABLED]
    quantiles = (
        sorted({round(float(q), 3) for q in np.quantile(observed, np.linspace(0, 1, n_quantiles))})
        if observed
        else []
    )
    rerank = [DISABLED, *quantiles]
    return {
        "min_rerank_answer": rerank,
        "min_rerank_any": rerank,
        "min_entity_coverage": [round(x, 2) for x in np.arange(0.0, 0.91, 0.1)],
    }


def calibrate_v1(traces: Sequence[GateTrace], rule: Rule = threshold_rule) -> dict[str, Any] | None:
    """Grid search for the V1 thresholds (objective in the module docstring)."""
    if not any(t.gate2_ran for t in traces):
        return None
    baseline = evaluate(traces, NO_OP, rule)
    floor = (
        baseline["coverage"] - COVERAGE_SLACK if math.isfinite(baseline["coverage"]) else -math.inf
    )
    grid = threshold_grid(traces)
    best: tuple[tuple[float, ...], Thresholds, dict[str, float]] | None = None
    for answer, any_, cov in itertools.product(*grid.values()):
        if answer != DISABLED and any_ != DISABLED and any_ > answer:
            continue  # the ABSTAIN floor must not exceed the ANSWER floor
        thr = Thresholds(answer, any_, cov)
        res = evaluate(traces, thr, rule)
        if math.isfinite(res["coverage"]) and res["coverage"] < floor:
            continue
        far = res["false_answer_rate"] if math.isfinite(res["false_answer_rate"]) else 0.0
        intrusiveness = (answer - DISABLED) + (any_ - DISABLED) + 100.0 * cov
        key = (far, -_finite(res["coverage"]), -_finite(res["routing_strict"]), intrusiveness)
        if best is None or key < best[0]:
            best = (key, thr, res)
    _, chosen, result = best if best is not None else ((), NO_OP, baseline)
    return {
        "thresholds": {
            "EVIDENCE_MIN_RERANK_ANSWER": chosen.min_rerank_answer,
            "EVIDENCE_MIN_RERANK_ANY": chosen.min_rerank_any,
            "EVIDENCE_MIN_ENTITY_COVERAGE": chosen.min_entity_coverage,
        },
        "env_lines": chosen.env_lines(),
        "dev_result": result,
        "c5_dev_baseline": baseline,
        "coverage_floor": floor,
        "n_traces": len(traces),
        "n_gate2_decisions": sum(1 for t in traces if t.gate2_ran),
        "objective": "minimise false-ANSWER rate on unanswerable dev turns s.t. coverage on answerable "
        f"dev turns >= C5 coverage - {COVERAGE_SLACK}; ties: coverage, routing accuracy, "
        "least intrusive thresholds",
        "disabled_value": DISABLED,
    }


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def calibrate(
    runs_dir: Path = RUNS_DIR,
    judgments_dir: Path = JUDGMENTS_DIR,
    output: Path = CALIBRATION_PATH,
    part: str = "all",
) -> dict[str, Any]:
    """Run the selected calibrations on the dev split and merge them into ``output``."""
    result: dict[str, Any] = read_json(output) if output.exists() else {}
    result.update(split="dev", updated_at=utc_now())
    runs = load_run_records(runs_dir)
    if part in ("nli", "all"):
        judgments = {j: load_all_judgments(judgments_dir, j) for j in ("llm", "nli")}
        outcomes = filter_split(build_outcomes(runs, judgments, None), "dev")
        result["nli"] = calibrate_nli(nli_claim_pairs(outcomes))
    if part in ("v1", "all"):
        c5_dev = [r for r in runs.get("C5", []) if r.get("split") == "dev"]
        result["v1"] = calibrate_v1(build_traces(c5_dev))
    write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate NLI and V1 thresholds on the dev split")
    parser.add_argument("--part", choices=["nli", "v1", "all"], default="all")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--judgments-dir", type=Path, default=JUDGMENTS_DIR)
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args(argv)
    result = calibrate(args.runs_dir, args.judgments_dir, args.output, args.part)
    nli = result.get("nli")
    if nli:
        print(
            f"[calibrate] NLI theta={nli['theta']} theta_c={nli['theta_c']} "
            f"(binary kappa {nli['kappa_binary']:.3f}, n={nli['n_claims']} dev claims)"
        )
    elif args.part in ("nli", "all"):
        print("[calibrate] NLI: no dev claims scored by both judges yet")
    v1 = result.get("v1")
    if v1:
        print("[calibrate] V1 thresholds (add to .env):")
        for line in v1["env_lines"]:
            print(line)
    elif args.part in ("v1", "all"):
        print("[calibrate] V1: no C5 dev traces with a Gate-2 LLM decision yet")
    print(f"[calibrate] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

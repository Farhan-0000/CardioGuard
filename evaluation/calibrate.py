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

3. **V2 grounding-verifier (Gate 3) thresholds** —
   ``GROUNDING_ENTAILMENT_THRESHOLD`` (θ_v) and
   ``GROUNDING_MAX_UNSUPPORTED_FRACTION`` (τ). Every C5 dev response that V2
   would verify (ANSWER/GENERALIZE with a generated text) is scored sentence
   by sentence with the PRODUCTION verifier
   (``medbridge.ai.grounding_verifier.GroundingVerifier``, model
   ``GROUNDING_NLI_MODEL`` = ``cross-encoder/nli-MiniLM2-L6-H768``) against the
   chunks the response cites (all chunks for uncited sentences), exactly as
   Gate 3 does. The LLM judge's claims are linked to those sentences.
   θ_v maximises Cohen's κ between "entailment ≥ θ_v" and the judge's verdict
   on the sentence (SUPPORTED iff every linked claim is SUPPORTED); ties prefer
   the value closest to 0.5. Given θ_v, τ minimises the dev hallucination rate
   (LLM judge, all C5 dev turns) subject to keeping ≥ 85 % of C5's answered
   coverage on answerable dev turns; ties → larger τ (less intrusive). The V2
   rule is replayed offline: unsupported fraction ≤ τ → the response is kept
   with its unsupported sentences removed (claims linked to them disappear);
   otherwise the response is withdrawn (ABSTAIN template: no claims, no
   coverage). V2's GENERALIZE regeneration of an over-threshold ANSWER cannot be
   replayed without the generator, so it is treated as withdrawn — a
   conservative assumption for the coverage constraint.

Degraded C5 turns (``evaluation.degraded``) are never used for calibration. A
part that cannot be computed keeps the previously stored block (with a
warning) instead of overwriting it with ``null``.

Protocol (README §1): run C5 on dev → judge dev → ``calibrate`` → run every
condition on test (V1/V2/V12 read their thresholds from calibration.json).

Writes ``evaluation/results/calibration.json`` and prints the ``.env`` lines.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from evaluation import metrics as mx
from evaluation.datasets.schema import ANSWERABLE_CATEGORIES, UNANSWERABLE_CATEGORIES
from evaluation.io import (
    judgment_file,
    load_all_judgments,
    load_judgments,
    load_run_records,
    read_json,
    write_json,
)
from evaluation.judges.common import (
    CLAIM_LABELS,
    CONTRADICTED,
    SUPPORTED,
    UNSUPPORTED,
    link_claim_to_sentence,
    marker_chunks,
)
from evaluation.judges.nli_judge import nli_label
from evaluation.manifest import utc_now
from evaluation.outcomes import (
    attempt_degradation,
    build_outcomes,
    exclude_degraded,
    filter_split,
    max_rerank_score,
)
from evaluation.paths import CALIBRATION_PATH, JUDGMENTS_DIR, RUNS_DIR

DISABLED = -100.0
COVERAGE_SLACK = 0.05
ANSWER_LIKE = frozenset({"ANSWER", "GENERALIZE"})
#: V2 must keep at least this share of C5's answered coverage on answerable dev turns.
V2_COVERAGE_KEEP = 0.85
ANSWERED_BEHAVIOURS = frozenset({"ANSWERED_SPECIFIC", "GENERAL_INFO"})
PARTS: tuple[str, ...] = ("nli", "v1", "v2")

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
            if not llm or not nli or not o.valid:
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
# V2 thresholds (grounding verifier / Gate 3)
# ----------------------------------------------------------------------


class Verifier(Protocol):
    """The subset of ``medbridge.ai.grounding_verifier.GroundingVerifier`` used here."""

    async def verify(self, response_text: str, chunks: Sequence[Any]) -> Any: ...


@dataclass(frozen=True)
class V2Response:
    """One C5 dev turn as Gate 3 would see it, with the LLM judge's verdicts."""

    item_id: str
    turn_index: int
    candidate: bool  # V2 would verify it (ANSWER/GENERALIZE with a generated draft)
    answerable: bool  # in the coverage denominator
    covered: bool  # C5 behaviour ANSWERED_SPECIFIC / GENERAL_INFO
    hallucinated: bool  # C5 outcome (LLM judge)
    entailments: tuple[float | None, ...]  # per verifier sentence; None = not a claim
    judge_claims: tuple[tuple[int | None, str], ...]  # (linked sentence, label)


def verifier_chunks(record: dict[str, Any]) -> list[dict[str, str]]:
    """Chunks in the order the response's ``[n]`` markers refer to, then the uncited ones.

    The validated response renumbers markers by first appearance; Gate 3 maps
    ``[n]`` to the n-th chunk, so the cited chunks are placed first in marker order.
    """
    texts = {
        str(e.get("chunk_id")): str(e.get("text", "")) for e in record.get("evidence_texts") or []
    }
    by_marker = marker_chunks(record)
    ordered = [
        {"chunk_id": by_marker.get(n, ""), "chunk_text": texts.get(by_marker.get(n, ""), "")}
        for n in range(1, max(by_marker, default=0) + 1)
    ]
    cited = set(by_marker.values())
    ordered += [
        {"chunk_id": cid, "chunk_text": text} for cid, text in texts.items() if cid not in cited
    ]
    return ordered


def _is_claim(sentence: dict[str, Any]) -> bool:
    flag = sentence.get("is_claim")
    if flag is not None:
        return bool(flag)
    try:
        from medbridge.ai.grounding_verifier import is_claim
    except ImportError:
        return True
    return bool(is_claim(str(sentence.get("text", ""))))


async def score_v2_responses(
    records: Sequence[dict[str, Any]],
    judgments: dict[tuple[str, int], dict[str, Any]],
    verifier: Verifier,
) -> list[V2Response]:
    """Score the C5 dev responses with the production verifier (judged turns only)."""
    out: list[V2Response] = []
    for rec in records:
        key = (str(rec["item_id"]), int(rec["turn_index"]))
        judgment = judgments.get(key)
        if (
            judgment is None
            or judgment.get("error")
            or judgment.get("attempt_id") != rec.get("attempt_id")
            or judgment.get("behaviour") is None
        ):
            continue
        claims = list(judgment.get("claims") or [])
        labels = [str(c.get("label")) for c in claims]
        candidate = (
            rec.get("action") in ANSWER_LIKE
            and judgment.get("template") is None
            and bool(str(rec.get("response_text") or "").strip())
        )
        entailments: tuple[float | None, ...] = ()
        linked: tuple[tuple[int | None, str], ...] = tuple((None, lab) for lab in labels)
        if candidate:
            result = await verifier.verify(str(rec["response_text"]), verifier_chunks(rec))
            if not getattr(result, "available", True):
                raise RuntimeError("grounding verifier NLI model unavailable")
            sentences = [dict(x) for x in getattr(result, "sentences", [])]
            entailments = tuple(
                float(x.get("entailment", 0.0)) if _is_claim(x) else None for x in sentences
            )
            texts = [str(x.get("text", "")) for x in sentences]
            linked = tuple(
                (link_claim_to_sentence(c, texts), lab)
                for c, lab in zip(claims, labels, strict=True)
            )
        out.append(
            V2Response(
                item_id=key[0],
                turn_index=key[1],
                candidate=candidate,
                answerable=str(rec.get("category")) in ANSWERABLE_CATEGORIES
                and rec.get("gold_action") in ANSWER_LIKE,
                covered=judgment.get("behaviour") in ANSWERED_BEHAVIOURS,
                hallucinated=any(lab in (UNSUPPORTED, CONTRADICTED) for lab in labels),
                entailments=entailments,
                judge_claims=linked,
            )
        )
    return out


def v2_sentence_pairs(responses: Sequence[V2Response]) -> list[tuple[bool, float]]:
    """(judge says supported, verifier entailment) for every verifier claim sentence with a
    linked judge claim; supported iff every linked claim is SUPPORTED."""
    pairs: list[tuple[bool, float]] = []
    for resp in responses:
        by_sentence: dict[int, list[str]] = {}
        for index, label in resp.judge_claims:
            if index is not None:
                by_sentence.setdefault(index, []).append(label)
        for index, labels in by_sentence.items():
            entail = resp.entailments[index] if index < len(resp.entailments) else None
            if entail is not None:
                pairs.append((all(lab == SUPPORTED for lab in labels), entail))
    return pairs


def simulate_v2(resp: V2Response, theta: float, tau: float) -> tuple[bool, bool]:
    """(hallucinated, covered) had Gate 3 run with (θ_v, τ) — see the module docstring."""
    claim_sentences = [i for i, e in enumerate(resp.entailments) if e is not None]
    if not resp.candidate or not claim_sentences:
        return resp.hallucinated, resp.covered
    removed = {i for i in claim_sentences if (resp.entailments[i] or 0.0) < theta}
    if len(removed) / len(claim_sentences) > tau:
        return False, False  # withdrawn → ABSTAIN template
    hallucinated = any(
        label in (UNSUPPORTED, CONTRADICTED) and (index is None or index not in removed)
        for index, label in resp.judge_claims
    )
    covered = resp.covered and len(removed) < len(claim_sentences)
    return hallucinated, covered


def evaluate_v2(responses: Sequence[V2Response], theta: float, tau: float) -> dict[str, float]:
    """Dev hallucination rate (all turns) and coverage (answerable turns) under (θ_v, τ)."""
    hallucinated = covered = answerable = 0
    for resp in responses:
        h, c = simulate_v2(resp, theta, tau)
        hallucinated += int(h)
        if resp.answerable:
            answerable += 1
            covered += int(c)
    return {
        "hallucination_rate": hallucinated / len(responses) if responses else math.nan,
        "coverage": covered / answerable if answerable else math.nan,
        "n_turns": len(responses),
        "n_answerable": answerable,
        "n_verified": sum(1 for r in responses if r.candidate),
    }


def calibrate_v2(
    responses: Sequence[V2Response],
    theta_grid: Sequence[float] | None = None,
    tau_grid: Sequence[float] | None = None,
) -> dict[str, Any] | None:
    """θ_v by Cohen's κ with the LLM judge, then τ by min hallucination s.t. coverage."""
    pairs = v2_sentence_pairs(responses)
    if not pairs:
        return None
    thetas = list(theta_grid or [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)])
    taus = list(tau_grid or [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)])
    gold = [str(g) for g, _ in pairs]

    def kappa(theta: float) -> float:
        value = mx.cohen_kappa(
            gold, [str(e >= theta) for _, e in pairs], labels=["True", "False"]
        ).kappa
        return value if math.isfinite(value) else -1.0

    theta = max(thetas, key=lambda t: (kappa(t), -abs(t - 0.5)))
    baseline = {
        "hallucination_rate": sum(r.hallucinated for r in responses) / len(responses),
        "coverage": (
            sum(r.covered for r in responses if r.answerable)
            / sum(1 for r in responses if r.answerable)
            if any(r.answerable for r in responses)
            else math.nan
        ),
    }
    floor = (
        V2_COVERAGE_KEEP * baseline["coverage"]
        if math.isfinite(baseline["coverage"])
        else -math.inf
    )
    best: tuple[tuple[float, float], float, dict[str, float]] | None = None
    for tau in taus:
        res = evaluate_v2(responses, theta, tau)
        if math.isfinite(res["coverage"]) and res["coverage"] < floor - 1e-12:
            continue
        key = (_finite(res["hallucination_rate"]), -tau)
        if best is None or key < best[0]:
            best = (key, tau, res)
    if best is None:  # no τ keeps the coverage: verifier effectively off
        tau, result = 1.0, evaluate_v2(responses, theta, 1.0)
    else:
        _, tau, result = best
    return {
        "thresholds": {
            "GROUNDING_ENTAILMENT_THRESHOLD": theta,
            "GROUNDING_MAX_UNSUPPORTED_FRACTION": tau,
        },
        "env_lines": [
            f"GROUNDING_ENTAILMENT_THRESHOLD={theta:g}",
            f"GROUNDING_MAX_UNSUPPORTED_FRACTION={tau:g}",
        ],
        "kappa_sentence": kappa(theta),
        "n_sentences": len(pairs),
        "n_responses": len(responses),
        "dev_result": result,
        "c5_dev_baseline": baseline,
        "coverage_floor": floor,
        "grid": {"theta": thetas, "tau": taus},
        "objective": "theta: max Cohen's kappa between verifier entailment >= theta and the LLM "
        "judge's sentence verdict (all linked claims SUPPORTED) on C5 dev sentences; tau: minimise "
        f"the dev hallucination rate (LLM judge) s.t. coverage >= {V2_COVERAGE_KEEP} x C5 dev "
        "coverage; ties: closest theta to 0.5, larger tau",
        "assumptions": "kept responses lose their unsupported sentences (and the judge claims "
        "linked to them); over-threshold responses are withdrawn (ABSTAIN) — the GENERALIZE "
        "regeneration of an ANSWER is not replayed (conservative for coverage)",
    }


def _make_verifier(model: str | None, device: str | None) -> Verifier:
    from medbridge.ai.grounding_verifier import GroundingVerifier

    return GroundingVerifier(model_name=model, threshold=0.5, device=device)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _clean_c5_dev(runs: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    """C5 dev records without degraded turns (and how many were dropped)."""
    records = [r for r in runs.get("C5", []) if r.get("split") == "dev"]
    reasons = attempt_degradation(records)
    clean = [
        r
        for r in records
        if not reasons.get((str(r["item_id"]), str(r.get("attempt_id", "")), int(r["turn_index"])))
    ]
    return clean, len(records) - len(clean)


def _keep_previous(result: dict[str, Any], part: str, new: dict[str, Any] | None) -> None:
    if new is not None:
        result[part] = new
    elif isinstance(result.get(part), dict):
        print(
            f"[calibrate] WARNING {part}: nothing to calibrate on; keeping the previous block",
            file=sys.stderr,
        )
    else:
        result[part] = None


def calibrate(
    runs_dir: Path = RUNS_DIR,
    judgments_dir: Path = JUDGMENTS_DIR,
    output: Path = CALIBRATION_PATH,
    part: str = "all",
    *,
    verifier: Verifier | None = None,
    verifier_model: str | None = None,
    verifier_device: str | None = None,
) -> dict[str, Any]:
    """Run the selected calibrations on the dev split and merge them into ``output``."""
    parts = PARTS if part == "all" else tuple(p.strip() for p in part.split(","))
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown calibration part(s) {unknown}; known: {PARTS} or 'all'")
    result: dict[str, Any] = read_json(output) if output.exists() else {}
    result.update(split="dev", updated_at=utc_now())
    runs = load_run_records(runs_dir)
    c5_dev, n_degraded = _clean_c5_dev(runs)
    inputs = {
        "c5_dev_turns": len(c5_dev),
        "c5_dev_degraded_excluded": n_degraded,
        "run_ids": sorted({str(r.get("run_id")) for r in c5_dev}),
    }
    if "nli" in parts:
        judgments = {j: load_all_judgments(judgments_dir, j) for j in ("llm", "nli")}
        outcomes = exclude_degraded(filter_split(build_outcomes(runs, judgments, None), "dev"))
        new = calibrate_nli(nli_claim_pairs(outcomes))
        _keep_previous(result, "nli", new)
    if "v1" in parts:
        new = calibrate_v1(build_traces(c5_dev))
        _keep_previous(result, "v1", {**new, "inputs": inputs} if new else None)
    if "v2" in parts:
        llm = load_judgments(judgment_file(judgments_dir, "llm", "C5"))
        new = None
        if any((str(r["item_id"]), int(r["turn_index"])) in llm for r in c5_dev):
            scorer = verifier or _make_verifier(verifier_model, verifier_device)
            responses = asyncio.run(score_v2_responses(c5_dev, llm, scorer))
            new = calibrate_v2(responses)
            if new is not None:
                model = getattr(scorer, "model_name", verifier_model)
                new = {**new, "verifier_model": model, "inputs": inputs}
        _keep_previous(result, "v2", new)
    result["protocol"] = (
        "1) run C5 on dev; 2) judge dev (LLM judge, then NLI); 3) calibrate on dev; "
        "4) run all conditions on test — V1/V2/V12 read their thresholds from this file"
    )
    write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate NLI-judge, V1 and V2 thresholds on the dev split"
    )
    parser.add_argument(
        "--part", default="all", help="all, or a comma list of: nli, v1, v2 (e.g. nli,v1)"
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--judgments-dir", type=Path, default=JUDGMENTS_DIR)
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    parser.add_argument(
        "--verifier-model", default=None, help="default: settings.GROUNDING_NLI_MODEL"
    )
    parser.add_argument("--verifier-device", default=None, help="cuda | mps | cpu (auto)")
    args = parser.parse_args(argv)
    try:
        result = calibrate(
            args.runs_dir,
            args.judgments_dir,
            args.output,
            args.part,
            verifier_model=args.verifier_model,
            verifier_device=args.verifier_device,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[calibrate] {exc}", file=sys.stderr)
        return 2
    selected = PARTS if args.part == "all" else tuple(args.part.split(","))
    nli = result.get("nli")
    if nli:
        print(
            f"[calibrate] NLI theta={nli['theta']} theta_c={nli['theta_c']} "
            f"(binary kappa {nli['kappa_binary']:.3f}, n={nli['n_claims']} dev claims)"
        )
    elif "nli" in selected:
        print("[calibrate] NLI: no dev claims scored by both judges yet")
    for part, label in (("v1", "V1"), ("v2", "V2")):
        block = result.get(part)
        if block:
            print(f"[calibrate] {label} thresholds (injected by evaluation.run for {label}):")
            for line in block["env_lines"]:
                print(line)
        elif part in selected:
            print(f"[calibrate] {label}: no usable C5 dev traces / judgments yet")
    print(f"[calibrate] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

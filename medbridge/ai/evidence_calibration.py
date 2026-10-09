"""
V1 — Calibrated Evidence Gate (CLAUDE_AGENT_PROMPT §6.7, ADL-030).

Same position and the same four outputs as Gate 2, made more robust in two ways:

1. **Permutation self-consistency.** The LLM gate is run ``votes`` times on
   deterministic permutations of the evidence order (original, reversed,
   rotate-by-1, rotate-by-2, …). Every run keeps temperature 0.0 / seed 42 —
   the diversity comes from evidence order (LLMs are position-sensitive), not
   from sampling, so the zero-temperature gate invariant is preserved. The
   majority action wins; ties are broken toward the more conservative action
   (ESCALATE > ABSTAIN > GENERALIZE > ANSWER).
2. **Quantitative evidence-sufficiency signal.** Reranker scores and the
   coverage of the question's salient entities in the evidence text can veto
   an over-confident vote: ANSWER with weak evidence becomes GENERALIZE, and
   ANSWER/GENERALIZE with irrelevant evidence becomes ABSTAIN. ESCALATE is never
   overridden and ABSTAIN is never upgraded. Thresholds come from Settings and
   are calibrated on the held-out split only.

If every gate run fails, the spec fallback for Gate 2 applies (GENERALIZE).
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from medbridge.ai.prompt_utils import chunk_text
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.config import Settings
from medbridge.nlp.drugs import find_drugs, normalize_drug

if TYPE_CHECKING:
    from medbridge.retrieval.hybrid_retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

# Higher = more conservative; used to break ties.
ACTION_SEVERITY: dict[str, int] = {"ANSWER": 0, "GENERALIZE": 1, "ABSTAIN": 2, "ESCALATE": 3}
# Reported when there is no evidence at all (JSON-safe stand-in for −∞).
NO_EVIDENCE_SCORE = -99.0

_NUMBER_WITH_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?\s*)?"
    r"(?:mg/dl|mmol/l|mm ?hg|mg|mcg|g|ml|%|bpm|kg|units?)\b"
    r"|\b\d{2,3}\s*/\s*\d{2,3}\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z][a-z\-]{3,}")
_SELF_MEDICATION = re.compile(
    r"\b(?:my|current)\s+(?:medicine|medication|meds|pills?|tablets?|dose)\b"
)

# Words that carry no clinical content in patient questions.
_STOPWORDS: frozenset[str] = frozenset("""
about above after again against also always among another any anything around away been
before being below best better between both cannot could does doing done down during each
either else enough even ever every feel feeling find first from further give going good
have having help here hers herself himself however into itself just keep kind know last
less like likely little long look lots made make many maybe mean might mine more most much
must myself need needs never next nothing often okay once only other others ought ours
ourselves over please quite rather really right said same safe safely say says should
since some something sometimes still such sure take taken takes taking tell than thank
thanks that their theirs them themselves then there these they thing things think this
those though through today told tomorrow took too toward under until upon used using very
want wanted wants well were what whatever when where whether which while whom whose will
with within without would year years your yours yourself doctor doctors normal week
weeks month months days morning night evening started start stop stopped medicine
medicines medication medications pill pills tablet tablets
""".split())


class EvidenceGateLike(Protocol):
    """Structural type of :class:`medbridge.ai.evidence_gate.EvidenceGate`."""

    async def evaluate(
        self,
        chunks: Sequence[Any],
        snapshot: dict[str, Any],
        query: str | None = None,
        message: str | None = None,
    ) -> EvidenceGateOutput | None: ...


@dataclass
class CalibratedDecision:
    """Outcome of the calibrated (V1) Evidence Gate."""

    action: str
    rationale: str
    votes: list[str]
    features: dict[str, float]
    overridden: bool
    voted_action: str = ""
    failed_votes: int = 0
    vote_rationales: list[str] = field(default_factory=list)

    @property
    def evidence_sufficient(self) -> bool:
        return self.action == "ANSWER"

    def to_gate_output(self) -> EvidenceGateOutput:
        """Expose the decision through the plain Gate 2 schema."""
        return EvidenceGateOutput(
            rationale=self.rationale,
            action=self.action,
            evidence_sufficient=self.evidence_sufficient,
        )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _rerank_score(chunk: Any) -> float:
    score = getattr(chunk, "score_rerank", None)
    if score is None and isinstance(chunk, dict):
        score = chunk.get("score_rerank", chunk.get("score"))
    if score is None:
        score = getattr(chunk, "score", None)
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def salient_entities(query: str, snapshot: dict[str, Any] | None = None) -> list[str]:
    """Salient entities of a question: drug names, numbers with units, clinical keywords.

    When the question refers to "my medicine/dose" without naming it, the
    current medications recorded in the snapshot are used as drug entities.
    """
    text = query or ""
    entities: list[str] = []
    drug_tokens: set[str] = set()
    for drug in find_drugs(text):
        entities.append(f"drug:{drug.generic}")
        for name in (drug.generic, drug.brand or ""):
            drug_tokens.update(name.lower().replace("-", " ").replace("+", " ").split())
    if not any(e.startswith("drug:") for e in entities) and _SELF_MEDICATION.search(text.lower()):
        for med in (snapshot or {}).get("current_medications") or []:
            info = normalize_drug(str(med.get("drug") or ""))
            if info is not None:
                entities.append(f"drug:{info.generic}")
    for number in _NUMBER_WITH_UNIT.findall(text):
        entities.append("num:" + re.sub(r"\s+", "", number.lower()))
    for word in _WORD.findall(text.lower()):
        word = word.strip("-")
        if len(word) >= 4 and word not in _STOPWORDS and word not in drug_tokens:
            entities.append(f"kw:{word}")
    return list(dict.fromkeys(entities))


def _stem(word: str) -> str:
    """Crude stem so 'hypertensive'/'hypertension' and 'diuretics'/'diuretic' match."""
    return word[: max(5, len(word) - 3)] if len(word) > 6 else word


def _entity_present(entity: str, evidence: str, evidence_compact: str) -> bool:
    kind, _, value = entity.partition(":")
    if kind == "drug":
        # Every ingredient must appear (first word: "metoprolol" for "metoprolol succinate").
        return all(part.split()[0] in evidence for part in value.split("+") if part.split())
    if kind == "num":
        return value in evidence_compact
    return _stem(value) in evidence


def evidence_features(
    query: str, chunks: Sequence[RetrievedChunk] | Sequence[Any], snapshot: dict[str, Any]
) -> dict[str, float]:
    """Quantitative evidence-sufficiency signal for the calibrated gate.

    Returns ``max_rerank`` and ``mean_top3_rerank`` (cross-encoder scores; the
    ``score_rerank`` field when present, else ``score``), ``entity_coverage``
    (fraction of the question's salient entities found in the concatenated
    evidence text; 1.0 when the question has none) and ``n_chunks``.
    """
    scores = sorted((_rerank_score(c) for c in chunks), reverse=True)
    evidence = " ".join(chunk_text(c) for c in chunks).lower()
    evidence_compact = re.sub(r"\s+", "", evidence)
    entities = salient_entities(query, snapshot)
    covered = sum(1 for e in entities if _entity_present(e, evidence, evidence_compact))
    return {
        "max_rerank": scores[0] if scores else NO_EVIDENCE_SCORE,
        "mean_top3_rerank": sum(scores[:3]) / len(scores[:3]) if scores else NO_EVIDENCE_SCORE,
        "entity_coverage": covered / len(entities) if entities else 1.0,
        "n_chunks": float(len(scores)),
    }


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def chunk_permutations(chunks: Sequence[Any], votes: int) -> list[list[Any]]:
    """Up to ``votes`` distinct deterministic orderings: original, reversed, rotations."""
    items = list(chunks)
    candidates: list[list[Any]] = [items, items[::-1]]
    candidates += [items[k:] + items[:k] for k in range(1, len(items))]
    orders: list[list[Any]] = []
    seen: set[tuple[int, ...]] = set()
    for order in candidates:
        signature = tuple(id(c) for c in order)
        if signature not in seen:
            seen.add(signature)
            orders.append(order)
        if len(orders) >= max(1, votes):
            break
    return orders


def majority_action(votes: Sequence[str]) -> str:
    """Most frequent action; ties go to the more conservative action."""
    counts = Counter(votes)
    return max(counts, key=lambda action: (counts[action], ACTION_SEVERITY.get(action, 0)))


def decide_with_thresholds(
    action: str,
    features: dict[str, float],
    *,
    min_rerank_answer: float,
    min_rerank_any: float,
    min_entity_coverage: float,
) -> str:
    """The V1 threshold rule applied to a (voted) Gate 2 action — pure and deterministic.

    ESCALATE and ABSTAIN are returned unchanged. ANSWER becomes GENERALIZE when
    ``max_rerank < min_rerank_answer`` or ``entity_coverage < min_entity_coverage``;
    ANSWER/GENERALIZE become ABSTAIN when ``max_rerank < min_rerank_any``.
    Shared with ``evaluation/calibrate.py`` so calibration replays the production rule.
    """
    if action not in ("ANSWER", "GENERALIZE"):
        return action
    max_rerank = features["max_rerank"]
    if max_rerank < min_rerank_any:
        return "ABSTAIN"
    if action == "ANSWER" and (
        max_rerank < min_rerank_answer or features["entity_coverage"] < min_entity_coverage
    ):
        return "GENERALIZE"
    return action


async def calibrated_evidence_decision(
    gate: EvidenceGateLike,
    chunks: Sequence[RetrievedChunk] | Sequence[Any],
    snapshot: dict[str, Any],
    query: str,
    *,
    votes: int,
    settings: Settings,
    message: str | None = None,
) -> CalibratedDecision:
    """Run the V1 calibrated Evidence Gate (see module docstring).

    Args:
        gate: The Gate 2 evaluator (temperature 0.0, seed 42).
        chunks: Reranked evidence chunks (top-5).
        snapshot: Current patient context snapshot.
        query: The question used for the gate and the entity-coverage feature.
        votes: Number of permutation votes (``EVIDENCE_GATE_VOTES``, default 3).
            Fewer runs happen when there are fewer distinct orderings.
        settings: Provides the ``EVIDENCE_MIN_*`` thresholds.
        message: The patient's own words, preferred for the gate and features.

    Returns:
        ``CalibratedDecision``; never raises for gate failures (GENERALIZE fallback).
    """
    question = message if message else query
    features = evidence_features(question, chunks, snapshot)
    orders = chunk_permutations(chunks, votes)
    results = await asyncio.gather(
        *(gate.evaluate(order, snapshot, query=query, message=message) for order in orders),
        return_exceptions=True,
    )
    outputs = [r for r in results if isinstance(r, EvidenceGateOutput)]
    failed = len(results) - len(outputs)
    for error in (r for r in results if isinstance(r, BaseException)):
        logger.error("calibrated_gate_vote_error", error=type(error).__name__)

    if not outputs:
        return CalibratedDecision(
            action="GENERALIZE",
            rationale="Evidence Gate failed on every vote; spec fallback to GENERALIZE.",
            votes=[],
            features=features,
            overridden=False,
            voted_action="GENERALIZE",
            failed_votes=failed,
        )

    vote_actions: list[str] = [o.action for o in outputs]
    voted = majority_action(vote_actions)
    rationale = next(o.rationale for o in outputs if o.action == voted)
    action = decide_with_thresholds(
        voted,
        features,
        min_rerank_answer=settings.EVIDENCE_MIN_RERANK_ANSWER,
        min_rerank_any=settings.EVIDENCE_MIN_RERANK_ANY,
        min_entity_coverage=settings.EVIDENCE_MIN_ENTITY_COVERAGE,
    )

    vote_summary = ", ".join(f"{a}×{n}" for a, n in Counter(vote_actions).most_common())
    full_rationale = f"[votes: {vote_summary}] {rationale}"
    if action != voted:
        full_rationale += (
            f" [calibration: {voted}→{action}; max_rerank={features['max_rerank']:.2f} "
            f"(answer≥{settings.EVIDENCE_MIN_RERANK_ANSWER}, "
            f"any≥{settings.EVIDENCE_MIN_RERANK_ANY}), "
            f"entity_coverage={features['entity_coverage']:.2f} "
            f"(≥{settings.EVIDENCE_MIN_ENTITY_COVERAGE})]"
        )
    decision = CalibratedDecision(
        action=action,
        rationale=full_rationale,
        votes=vote_actions,
        features=features,
        overridden=action != voted,
        voted_action=voted,
        failed_votes=failed,
        vote_rationales=[o.rationale for o in outputs],
    )
    logger.info(
        "calibrated_evidence_gate",
        action=action,
        voted=voted,
        votes=vote_actions,
        overridden=decision.overridden,
        max_rerank=round(features["max_rerank"], 3),
        entity_coverage=round(features["entity_coverage"], 3),
    )
    return decision

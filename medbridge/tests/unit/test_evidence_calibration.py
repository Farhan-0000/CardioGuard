"""
Unit tests for V1 — the calibrated Evidence Gate (medbridge.ai.evidence_calibration).

Covers the quantitative features (rerank scores, entity coverage), the
deterministic evidence permutations, majority voting with conservative
tie-breaking, the threshold overrides, failure fallback, and that every vote
still runs the gate at temperature 0.0 (the invariant is preserved).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from medbridge.ai.evidence_calibration import (
    NO_EVIDENCE_SCORE,
    CalibratedDecision,
    calibrated_evidence_decision,
    chunk_permutations,
    decide_with_thresholds,
    evidence_features,
    majority_action,
    salient_entities,
)
from medbridge.ai.evidence_gate import EvidenceGate
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.config import Settings


def chunk(i: int, text: str, rerank: float | None = None, score: float = 0.0) -> dict[str, Any]:
    data: dict[str, Any] = {"chunk_id": f"c{i}", "chunk_text": text, "score": score}
    if rerank is not None:
        data["score_rerank"] = rerank
    return data


CHUNKS = [
    chunk(1, "Amlodipine is a calcium channel blocker used for hypertension.", rerank=5.0),
    chunk(2, "Thiazide diuretics lower blood pressure.", rerank=3.0),
    chunk(3, "Reduce sodium intake to lower blood pressure.", rerank=1.0),
    chunk(4, "Home blood pressure monitoring is recommended.", rerank=-2.0),
    chunk(5, "ACE inhibitors may cause a dry cough.", rerank=-4.0),
]


def settings(**overrides: Any) -> Settings:
    base = {"EVIDENCE_MIN_RERANK_ANSWER": 0.0, "EVIDENCE_MIN_RERANK_ANY": -6.0,
            "EVIDENCE_MIN_ENTITY_COVERAGE": 0.0}
    base.update(overrides)
    return Settings(_env_file=None, **base)


class ScriptedGate:
    """Returns pre-scripted outputs per call and records the chunk orders it saw."""

    def __init__(self, outputs: Sequence[EvidenceGateOutput | None | Exception]) -> None:
        self.outputs = list(outputs)
        self.orders: list[list[str]] = []

    async def evaluate(self, chunks, snapshot, query=None, message=None):
        self.orders.append([c["chunk_id"] for c in chunks])
        result = self.outputs[len(self.orders) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def out(action: str, rationale: str = "r") -> EvidenceGateOutput:
    return EvidenceGateOutput(action=action, evidence_sufficient=action == "ANSWER", rationale=rationale)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_rerank_features(self) -> None:
        features = evidence_features("amlodipine for hypertension", CHUNKS, {})
        assert features["max_rerank"] == 5.0
        assert features["mean_top3_rerank"] == pytest.approx(3.0)
        assert features["n_chunks"] == 5.0

    def test_falls_back_to_score_when_no_rerank_field(self) -> None:
        features = evidence_features("x", [chunk(1, "a", score=0.7), chunk(2, "b", score=0.2)], {})
        assert features["max_rerank"] == 0.7

    def test_no_evidence(self) -> None:
        features = evidence_features("amlodipine dose", [], {})
        assert features["max_rerank"] == NO_EVIDENCE_SCORE
        assert features["mean_top3_rerank"] == NO_EVIDENCE_SCORE
        assert features["n_chunks"] == 0.0
        assert features["entity_coverage"] == 0.0

    def test_entity_coverage_full(self) -> None:
        features = evidence_features("Is amlodipine used for hypertension?", CHUNKS, {})
        assert features["entity_coverage"] == 1.0

    def test_fabricated_drug_lowers_coverage(self) -> None:
        features = evidence_features("Is Cardiozepam better than amlodipine?", CHUNKS, {})
        assert features["entity_coverage"] == pytest.approx(0.5)

    def test_brand_matches_generic_in_evidence(self) -> None:
        entities = salient_entities("Is Norvasc ok?")
        assert entities == ["drug:amlodipine"]
        assert evidence_features("Is Norvasc ok?", CHUNKS, {})["entity_coverage"] == 1.0

    def test_numbers_with_units(self) -> None:
        entities = salient_entities("Can I take 20 mg when my bp is 150/95?")
        assert "num:20mg" in entities and "num:150/95" in entities
        features = evidence_features("Can I take 20 mg?", [chunk(1, "Usual dose 5 to 10 mg.")], {})
        assert features["entity_coverage"] == 0.0

    def test_snapshot_medication_used_for_my_dose(self) -> None:
        snapshot = {"current_medications": [{"drug": "Telma", "dosage": "40 mg"}]}
        assert salient_entities("should I double my dose", snapshot) == [
            "drug:telmisartan", "kw:double", "kw:dose",
        ]
        assert salient_entities("should I double my dose", {}) == ["kw:double", "kw:dose"]

    def test_no_entities_means_full_coverage(self) -> None:
        assert evidence_features("what should I do?", CHUNKS, {})["entity_coverage"] == 1.0


# ---------------------------------------------------------------------------
# Permutations & voting
# ---------------------------------------------------------------------------


class TestVotingPrimitives:
    def test_permutations_original_reversed_rotations(self) -> None:
        orders = chunk_permutations(["a", "b", "c", "d", "e"], 4)
        assert orders == [
            ["a", "b", "c", "d", "e"], ["e", "d", "c", "b", "a"],
            ["b", "c", "d", "e", "a"], ["c", "d", "e", "a", "b"],
        ]

    def test_permutations_deduplicated(self) -> None:
        a, b = object(), object()
        assert chunk_permutations([a], 3) == [[a]]
        assert chunk_permutations([a, b], 3) == [[a, b], [b, a]]
        assert chunk_permutations([], 3) == [[]]

    @pytest.mark.parametrize(("votes", "expected"), [
        (["ANSWER", "ANSWER", "GENERALIZE"], "ANSWER"),
        (["ANSWER", "GENERALIZE"], "GENERALIZE"),
        (["ANSWER", "ABSTAIN", "GENERALIZE"], "ABSTAIN"),
        (["ESCALATE", "ABSTAIN"], "ESCALATE"),
        (["GENERALIZE", "GENERALIZE", "ESCALATE"], "GENERALIZE"),
    ])
    def test_majority_with_conservative_ties(self, votes: list[str], expected: str) -> None:
        assert majority_action(votes) == expected


# ---------------------------------------------------------------------------
# Calibrated decision
# ---------------------------------------------------------------------------


class TestCalibratedDecision:
    async def test_majority_vote_over_permutations(self) -> None:
        gate = ScriptedGate([out("ANSWER", "a1"), out("GENERALIZE", "g"), out("ANSWER", "a2")])
        decision = await calibrated_evidence_decision(
            gate, CHUNKS, {}, "amlodipine for hypertension", votes=3, settings=settings()
        )
        assert gate.orders == [
            ["c1", "c2", "c3", "c4", "c5"], ["c5", "c4", "c3", "c2", "c1"],
            ["c2", "c3", "c4", "c5", "c1"],
        ]
        assert decision.action == "ANSWER" and decision.voted_action == "ANSWER"
        assert decision.votes == ["ANSWER", "GENERALIZE", "ANSWER"]
        assert not decision.overridden
        assert "a1" in decision.rationale and "ANSWER×2" in decision.rationale
        assert decision.evidence_sufficient

    async def test_tie_goes_conservative(self) -> None:
        gate = ScriptedGate([out("ANSWER"), out("GENERALIZE")])
        decision = await calibrated_evidence_decision(
            gate, CHUNKS[:2], {}, "amlodipine", votes=3, settings=settings()
        )
        assert len(gate.orders) == 2
        assert decision.action == "GENERALIZE"

    async def test_answer_downgraded_when_rerank_weak(self) -> None:
        gate = ScriptedGate([out("ANSWER")] * 3)
        decision = await calibrated_evidence_decision(
            gate, CHUNKS, {}, "amlodipine", votes=3, settings=settings(EVIDENCE_MIN_RERANK_ANSWER=6.0)
        )
        assert decision.action == "GENERALIZE" and decision.overridden
        assert "calibration: ANSWER→GENERALIZE" in decision.rationale

    async def test_answer_downgraded_when_coverage_low(self) -> None:
        gate = ScriptedGate([out("ANSWER")] * 3)
        decision = await calibrated_evidence_decision(
            gate, CHUNKS, {}, "Is Cardiozepam better than amlodipine?", votes=3,
            settings=settings(EVIDENCE_MIN_ENTITY_COVERAGE=0.75),
        )
        assert decision.action == "GENERALIZE" and decision.features["entity_coverage"] == 0.5

    async def test_irrelevant_evidence_abstains(self) -> None:
        weak = [chunk(1, "x", rerank=-9.0), chunk(2, "y", rerank=-8.0)]
        for voted in ("ANSWER", "GENERALIZE"):
            gate = ScriptedGate([out(voted)] * 2)
            decision = await calibrated_evidence_decision(
                gate, weak, {}, "q", votes=3, settings=settings()
            )
            assert decision.action == "ABSTAIN" and decision.overridden

    async def test_escalate_never_overridden_and_abstain_never_upgraded(self) -> None:
        weak = [chunk(1, "x", rerank=-9.0)]
        decision = await calibrated_evidence_decision(
            ScriptedGate([out("ESCALATE")]), weak, {}, "q", votes=3, settings=settings()
        )
        assert decision.action == "ESCALATE" and not decision.overridden
        decision = await calibrated_evidence_decision(
            ScriptedGate([out("ABSTAIN")] * 3), CHUNKS, {}, "q", votes=3,
            settings=settings(EVIDENCE_MIN_RERANK_ANY=-100.0),
        )
        assert decision.action == "ABSTAIN" and not decision.overridden

    async def test_all_votes_fail_falls_back_to_generalize(self) -> None:
        gate = ScriptedGate([None, RuntimeError("boom"), None])
        decision = await calibrated_evidence_decision(
            gate, CHUNKS, {}, "q", votes=3, settings=settings()
        )
        assert decision.action == "GENERALIZE"
        assert decision.votes == [] and decision.failed_votes == 3
        assert not decision.overridden

    async def test_partial_failures_ignored(self) -> None:
        gate = ScriptedGate([None, out("ABSTAIN"), out("ABSTAIN")])
        decision = await calibrated_evidence_decision(
            gate, CHUNKS, {}, "q", votes=3, settings=settings()
        )
        assert decision.action == "ABSTAIN" and decision.failed_votes == 1

    async def test_message_preferred_for_features(self) -> None:
        gate = ScriptedGate([out("ANSWER")] * 3)
        decision = await calibrated_evidence_decision(
            gate, CHUNKS, {}, "amlodipine hypertension", votes=3,
            settings=settings(EVIDENCE_MIN_ENTITY_COVERAGE=0.9), message="Is Cardiozepam safe?",
        )
        assert decision.features["entity_coverage"] == 0.0
        assert decision.action == "GENERALIZE"

    async def test_every_vote_runs_at_temperature_zero(self) -> None:
        llm = MagicMock()
        llm.call = AsyncMock(return_value=out("ANSWER"))
        decision = await calibrated_evidence_decision(
            EvidenceGate(llm=llm), CHUNKS, {}, "amlodipine", votes=3, settings=settings()
        )
        assert llm.call.await_count == 3
        assert all(c.kwargs["temperature"] == 0.0 for c in llm.call.await_args_list)
        assert all("seed" not in c.kwargs for c in llm.call.await_args_list)  # wrapper default 42
        prompts = [c.kwargs["user_prompt"] for c in llm.call.await_args_list]
        assert len(set(prompts)) == 3  # diversity comes from evidence order only
        assert decision.action == "ANSWER"

    async def test_every_permutation_vote_sees_the_drug_reference(self) -> None:
        """Brand names are judged by their generic in every V1 vote, not just the first."""
        llm = MagicMock()
        llm.call = AsyncMock(return_value=out("GENERALIZE"))
        await calibrated_evidence_decision(
            EvidenceGate(llm=llm), CHUNKS, {}, "ACE inhibitor uses quinapril", votes=3,
            settings=settings(), message="What is Accupril used for?",
        )
        prompts = [c.kwargs["user_prompt"] for c in llm.call.await_args_list]
        assert len(prompts) == 3
        note = "- Accupril: brand name of quinapril (ACE inhibitor)"
        assert all(f"<drug_reference>\n{note}\n</drug_reference>" in p for p in prompts)

    def test_entity_coverage_is_symmetric_for_brand_and_generic(self) -> None:
        evidence = [chunk(1, "Quinapril is an ACE inhibitor used for hypertension.", rerank=2.0)]
        brand = evidence_features("What is Accupril used for?", evidence, {})
        generic = evidence_features("What is quinapril used for?", evidence, {})
        assert brand == generic

    @pytest.mark.parametrize(("action", "rerank", "coverage", "expected"), [
        ("ANSWER", 2.0, 1.0, "ANSWER"),
        ("ANSWER", -1.0, 1.0, "GENERALIZE"),
        ("ANSWER", 2.0, 0.2, "GENERALIZE"),
        ("ANSWER", -7.0, 1.0, "ABSTAIN"),
        ("GENERALIZE", -7.0, 1.0, "ABSTAIN"),
        ("GENERALIZE", -1.0, 0.0, "GENERALIZE"),
        ("ABSTAIN", 9.0, 1.0, "ABSTAIN"),
        ("ESCALATE", -99.0, 0.0, "ESCALATE"),
    ])
    def test_decide_with_thresholds(self, action: str, rerank: float, coverage: float,
                                    expected: str) -> None:
        features = {"max_rerank": rerank, "entity_coverage": coverage}
        assert decide_with_thresholds(
            action, features, min_rerank_answer=0.0, min_rerank_any=-6.0, min_entity_coverage=0.5
        ) == expected

    def test_to_gate_output(self) -> None:
        decision = CalibratedDecision(action="ANSWER", rationale="r", votes=["ANSWER"],
                                      features={}, overridden=False)
        gate_output = decision.to_gate_output()
        assert gate_output.action == "ANSWER" and gate_output.evidence_sufficient


def test_redaction_placeholders_are_not_entities() -> None:
    from medbridge.ai.evidence_calibration import salient_entities

    redacted = "my name is [NAME], phone [PHONE], email [EMAIL], my bp is 138/88 is that ok?"
    entities = salient_entities(redacted)
    assert not any(e in entities for e in ("kw:name", "kw:phone", "kw:email"))
    assert entities == salient_entities("my bp is 138/88 is that ok?")
    assert salient_entities("I am [AGE: 54], is salt bad?") == salient_entities("is salt bad?")

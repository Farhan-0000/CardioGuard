"""
Labelled natural-language cases for the deterministic emergency classifier.

Drives ``tests/data/emergency_cases.json`` (synthetic, hand-written messages):
every positive must ESCALATE with its listed rule ids (100 % recall) and every
negative must pass through (no false positive). A summary test prints
precision / recall over the whole file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from medbridge.api.schemas.enums import ActionEnum
from medbridge.core.emergency_classifier import RULE_IDS, classify, classify_detailed

CASES_PATH = Path(__file__).resolve().parents[3] / "tests" / "data" / "emergency_cases.json"
CASES: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
POSITIVES = [c for c in CASES if c["expected"] == "ESCALATE"]
NEGATIVES = [c for c in CASES if c["expected"] is None]

#: Colloquial phrasings named in the finalisation brief; each must appear in a case.
REQUIRED_POSITIVE_PHRASES = [
    "210/125", "bp 190/125", "bp of 185 by 110", "systolic 190", "top number", "200 something",
    "my bp is over 180", "reading came 182/96", "bp 200+", "my chest feels tight",
    "heavy feeling in my chest", "elephant on my chest", "short of breath walking to the bathroom",
    "can't breathe", "gasping", "can't catch my breath", "breathless lying down", "face drooping",
    "one side of my face", "can't move my arm", "arm is weak", "arm is numb", "slurred speech",
    "can't talk properly", "sudden confusion", "lost vision in one eye", "sudden loss of balance",
    "worst headache of my life", "thunderclap", "sudden severe headache", "fainted", "passed out",
    "blacked out", "collapsed", "seizure", "coughing up blood", "pink frothy sputum", "tearing",
    "ripping", "palpitations", "blue lips", "pregnant", "end my life", "kill myself",
    "blood in my urine", "is chest pain a side effect of amlodipine?",
    "no chest pain but my bp is 190/125",
]
REQUIRED_NEGATIVE_PHRASES = [
    "no chest pain", "denies dizziness", "without any chest pain", "not having trouble breathing",
    "never fainted", "no longer have chest pain", "chest pain? no",
    "what are the warning signs of a stroke?", "how do i recognise a heart attack",
    "my father had a stroke in 2015", "i had chest pain last year, it was acid reflux", "120/80",
    "145/92", "150 over 95", "180 mg diltiazem", "cholesterol 190", "12/10/2024", "1/2 tablet",
    "heart rate 72",
]


def _case_id(case: dict[str, Any]) -> str:
    return str(case["id"])


class TestCaseFile:
    """The labelled file itself is well formed and large enough."""

    def test_size_and_balance(self) -> None:
        assert len(CASES) >= 120
        assert len(POSITIVES) >= 60
        assert len(NEGATIVES) >= 60

    def test_ids_and_texts_unique(self) -> None:
        ids = [c["id"] for c in CASES]
        texts = [c["text"].strip().lower() for c in CASES]
        assert len(set(ids)) == len(ids)
        assert len(set(texts)) == len(texts)

    def test_rule_ids_are_known(self) -> None:
        for case in CASES:
            for rule_id in case.get("rules", []) + case.get("negated", []):
                assert rule_id in RULE_IDS, (case["id"], rule_id)

    def test_positives_name_their_rules(self) -> None:
        assert all(c.get("rules") for c in POSITIVES)

    @pytest.mark.parametrize("phrase", REQUIRED_POSITIVE_PHRASES)
    def test_required_positive_phrase_present(self, phrase: str) -> None:
        assert any(phrase in c["text"].lower() for c in POSITIVES), phrase

    @pytest.mark.parametrize("phrase", REQUIRED_NEGATIVE_PHRASES)
    def test_required_negative_phrase_present(self, phrase: str) -> None:
        assert any(phrase in c["text"].lower() for c in NEGATIVES), phrase


@pytest.mark.parametrize("case", POSITIVES, ids=_case_id)
def test_positive_case_escalates(case: dict[str, Any]) -> None:
    result = classify_detailed(case["text"])
    assert result.action == ActionEnum.ESCALATE, (case["text"], result)
    missing = [r for r in case["rules"] if r not in result.matched_rules]
    assert not missing, (case["text"], result.matched_rules)
    assert not [r for r in case.get("negated", []) if r not in result.negated], result
    assert result.evidence_spans


@pytest.mark.parametrize("case", NEGATIVES, ids=_case_id)
def test_negative_case_passes(case: dict[str, Any]) -> None:
    result = classify_detailed(case["text"])
    assert result.action is None, (case["text"], result)
    assert result.matched_rules == []
    assert result.evidence_spans == []
    assert not [r for r in case.get("negated", []) if r not in result.negated], result


def test_precision_recall_summary() -> None:
    """Confusion matrix over the whole file: 100 % recall and no false positives."""
    tp = sum(classify(c["text"]) == ActionEnum.ESCALATE for c in POSITIVES)
    fp = sum(classify(c["text"]) == ActionEnum.ESCALATE for c in NEGATIVES)
    fn = len(POSITIVES) - tp
    tn = len(NEGATIVES) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(
        f"\nemergency_cases.json: n={len(CASES)} TP={tp} FP={fp} FN={fn} TN={tn} "
        f"precision={precision:.3f} recall={recall:.3f}"
    )
    assert recall == 1.0
    assert fp == 0

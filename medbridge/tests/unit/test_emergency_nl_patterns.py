"""
Labelled natural-language cases for the deterministic emergency classifier.

Drives ``tests/data/emergency_cases.json`` (synthetic, hand-written English
messages) and ``tests/data/emergency_cases_hinglish.json`` (synthetic Roman-script
Hindi / Indian-English messages, written without access to the HalluBench test
items): every positive must ESCALATE with its listed rule ids (100 % recall) and
every negative must pass through (no false positive). Summary tests print
precision / recall per file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from medbridge.api.schemas.enums import ActionEnum
from medbridge.core.emergency_classifier import (
    HINGLISH_RULE_IDS,
    RULE_IDS,
    classify,
    classify_detailed,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "tests" / "data"
CASES_PATH = DATA_DIR / "emergency_cases.json"
CASES: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
POSITIVES = [c for c in CASES if c["expected"] == "ESCALATE"]
NEGATIVES = [c for c in CASES if c["expected"] is None]

HI_CASES_PATH = DATA_DIR / "emergency_cases_hinglish.json"
HI_CASES: list[dict[str, Any]] = json.loads(HI_CASES_PATH.read_text(encoding="utf-8"))["cases"]
HI_POSITIVES = [c for c in HI_CASES if c["expected"] == "ESCALATE"]
HI_NEGATIVES = [c for c in HI_CASES if c["expected"] is None]

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


def _confusion(
    name: str, positives: list[dict[str, Any]], negatives: list[dict[str, Any]]
) -> tuple[int, int]:
    """Print the confusion matrix of one case file; return (false negatives, false positives)."""
    tp = sum(classify(c["text"]) == ActionEnum.ESCALATE for c in positives)
    fp = sum(classify(c["text"]) == ActionEnum.ESCALATE for c in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(
        f"\n{name}: n={len(positives) + len(negatives)} TP={tp} FP={fp} FN={fn} TN={tn} "
        f"precision={precision:.3f} recall={recall:.3f}"
    )
    return fn, fp


def test_precision_recall_summary() -> None:
    """Confusion matrix over the whole file: 100 % recall and no false positives."""
    assert _confusion("emergency_cases.json", POSITIVES, NEGATIVES) == (0, 0)


# ---------------------------------------------------------------------------
# Hinglish / Indian-English case file
# ---------------------------------------------------------------------------

#: Phrases named in the Hinglish brief; each must appear in a case of the right polarity.
HI_REQUIRED_POSITIVE_PHRASES = [
    "seene mein", "sine me", "chhati mein jalan", "chati me dabav", "chest mein pain",
    "dil mein dard", "bharipan", "saans nahi aa rahi", "saans lene mein takleef",
    "saans phool rahi", "dam ghut raha", "behosh ho gayi", "chakkar aake gir gaya",
    "hosh nahi", "lakwa", "muh tedha ho gaya", "chehra ek taraf latak gaya",
    "haath nahi hil raha", "baazu nahi uth raha", "aawaz ladkhada rahi", "zubaan ladkhada rahi",
    "bol nahi pa", "bahut tez sar dard", "sar phat raha", "bp bahut high hai 200",
    "khoon ki ulti", "khansi mein khoon", "marna chahta hoon", "jeena nahi chahti",
    "dard nahi hai lekin bp 200/120 hai",
]
HI_REQUIRED_NEGATIVE_PHRASES = [
    "seene mein dard nahi hai", "nahin", "koi seene ka dard", "heart attack ke lakshan kya hain?",
    "papa ko pichle saal lakwa hua tha", "na to seene mein dard hai",
]


class TestHinglishCaseFile:
    """The Hinglish labelled file is well formed and large enough."""

    def test_size_and_balance(self) -> None:
        assert len(HI_CASES) >= 60
        assert len(HI_POSITIVES) >= 35
        assert len(HI_NEGATIVES) >= 25

    def test_ids_and_texts_unique(self) -> None:
        ids = [c["id"] for c in HI_CASES]
        texts = [c["text"].strip().lower() for c in HI_CASES]
        assert len(set(ids)) == len(ids)
        assert len(set(texts)) == len(texts)
        english = {c["text"].strip().lower() for c in CASES}
        assert not english & set(texts)

    def test_rule_ids_are_known(self) -> None:
        for case in HI_CASES:
            for key in ("rules", "negated", "suppressed"):
                for rule_id in case.get(key, []):
                    assert rule_id in RULE_IDS, (case["id"], rule_id)

    def test_positives_name_their_rules(self) -> None:
        assert all(c.get("rules") for c in HI_POSITIVES)

    def test_every_hinglish_rule_has_a_positive_case(self) -> None:
        covered = {r for c in HI_POSITIVES for r in c["rules"]}
        assert set(HINGLISH_RULE_IDS) <= covered

    def test_negatives_cover_negation_information_and_history(self) -> None:
        categories = {c["category"].split("_")[0] for c in HI_NEGATIVES}
        assert {"negation", "informational", "historical"} <= categories
        assert sum(bool(c.get("negated")) for c in HI_NEGATIVES) >= 10

    @pytest.mark.parametrize("phrase", HI_REQUIRED_POSITIVE_PHRASES)
    def test_required_positive_phrase_present(self, phrase: str) -> None:
        assert any(phrase in c["text"].lower() for c in HI_POSITIVES), phrase

    @pytest.mark.parametrize("phrase", HI_REQUIRED_NEGATIVE_PHRASES)
    def test_required_negative_phrase_present(self, phrase: str) -> None:
        assert any(phrase in c["text"].lower() for c in HI_NEGATIVES), phrase


@pytest.mark.parametrize("case", HI_POSITIVES, ids=_case_id)
def test_hinglish_positive_case_escalates(case: dict[str, Any]) -> None:
    result = classify_detailed(case["text"])
    assert result.action == ActionEnum.ESCALATE, (case["text"], result)
    missing = [r for r in case["rules"] if r not in result.matched_rules]
    assert not missing, (case["text"], result.matched_rules)
    assert not [r for r in case.get("negated", []) if r not in result.negated], result
    assert result.evidence_spans


@pytest.mark.parametrize("case", HI_NEGATIVES, ids=_case_id)
def test_hinglish_negative_case_passes(case: dict[str, Any]) -> None:
    result = classify_detailed(case["text"])
    assert result.action is None, (case["text"], result)
    assert result.matched_rules == []
    assert result.evidence_spans == []
    assert not [r for r in case.get("negated", []) if r not in result.negated], result
    assert not [r for r in case.get("suppressed", []) if r not in result.suppressed], result


def test_hinglish_precision_recall_summary() -> None:
    """Confusion matrix over the Hinglish file: 100 % recall and no false positives."""
    assert _confusion("emergency_cases_hinglish.json", HI_POSITIVES, HI_NEGATIVES) == (0, 0)

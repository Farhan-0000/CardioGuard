"""
Unit tests for TASK-15: Deterministic Emergency Classifier.

Covers true positives (crisis BP, symptom keywords), true negatives
(routine queries), case insensitivity, whitespace variations, and
sub-5ms latency verification; plus the detailed result API,
normalisation and typo tolerance, BP parsing, negation scope, context
filters, determinism, import purity and the < 10 ms budget on a
2,000-character message. Labelled NL cases live in
``test_emergency_nl_patterns.py``.
"""
import ast
import statistics
import time
from pathlib import Path

import pytest

from medbridge.api.schemas.enums import ActionEnum
from medbridge.core import emergency_classifier as ec
from medbridge.core.emergency_classifier import (
    RULE_IDS,
    EmergencyResult,
    classify,
    classify_detailed,
    normalize_text,
)

# ---------------------------------------------------------------------------
# True Positive Suite — Hypertensive Crisis BP Values
# ---------------------------------------------------------------------------

class TestHypertensiveCrisis:
    """BP systolic ≥180 OR diastolic ≥120 → ESCALATE."""

    @pytest.mark.parametrize("message", [
        "My bp is 200/130",
        "blood pressure 185/125",
        "BP 180/120 this morning",
        "My blood pressure was 195/110",    # systolic ≥180
        "bp 210/140 and I feel dizzy",
        "Blood Pressure: 190/130",
        "BP is 199/121",
    ])
    def test_crisis_bp_escalates(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE

    @pytest.mark.parametrize("message", [
        "My bp is 180/125",
        "blood pressure 220/150",
        "BP 195/135",
    ])
    def test_extreme_crisis_values(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE


# ---------------------------------------------------------------------------
# True Positive Suite — Red-Flag Symptoms
# ---------------------------------------------------------------------------

class TestRedFlagSymptoms:
    """Known emergency symptom keywords → ESCALATE."""

    @pytest.mark.parametrize("message", [
        "I'm having chest pain right now",
        "experiencing difficulty breathing since this morning",
        "I had a sudden severe headache an hour ago",
        "sudden headache with nausea",
        "having vision changes and dizziness",
        "vision loss in my left eye",
        "vision blurring since yesterday",
        "numbness in my face and arm",
        "numbness in my left leg",
        "I have slurred speech",
        "my speech is slured and I'm confused",
        "I fainted at work today",
        "I passed out twice",
        "my mother is unconscious",
        "had a seizure last night",
        "there's blood in urine",
        "noticed blood in stool yesterday",
    ])
    def test_symptom_escalates(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE


# ---------------------------------------------------------------------------
# True Positive Suite — Self-Harm / Suicidal Ideation
# ---------------------------------------------------------------------------

class TestSelfHarm:
    """Self-harm and suicidal ideation keywords → ESCALATE."""

    @pytest.mark.parametrize("message", [
        "I'm thinking about suicide",
        "having suicidal thoughts",
        "I want to self-harm",
        "I want to self harm",
        "I want to kill myself",
    ])
    def test_self_harm_escalates(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE


# ---------------------------------------------------------------------------
# True Negative Suite — Routine Messages
# ---------------------------------------------------------------------------

class TestTrueNegatives:
    """Routine clinical questions → None (safe to continue)."""

    @pytest.mark.parametrize("message", [
        "What is my blood pressure target?",
        "My bp is 140/90, is that high?",
        "Blood pressure 130/85 this morning",
        "I take amlodipine 5mg daily",
        "What are the side effects of lisinopril?",
        "Should I reduce my salt intake?",
        "I have a mild headache",                      # Not "sudden" headache
        "I've been feeling tired lately",
        "Can I exercise with hypertension?",
        "My doctor prescribed metoprolol",
        "How does ACE inhibitor work?",
        "I'm 55 years old, male, diabetic",
        "What foods should I avoid?",
        "My potassium level is 4.5 mEq/L",
        "Is 120/80 a normal blood pressure?",
        "BP 170/95 today",                             # Below crisis threshold
        "blood pressure 179/119",                      # Just below threshold
    ])
    def test_routine_returns_none(self, message: str) -> None:
        assert classify(message) is None

    def test_empty_message(self) -> None:
        assert classify("") is None

    def test_unrelated_numbers_below_dual_threshold(self) -> None:
        # Numbers that look like BP but don't meet BOTH thresholds (systolic ≥180 AND diastolic ≥120)
        assert classify("My order number is 180/90") is None
        assert classify("Room 150/130 is on the left") is None
        assert classify("Reference code 170/125") is None


# ---------------------------------------------------------------------------
# True Positive Suite — Standalone Crisis BP (No Prefix Required)
# ---------------------------------------------------------------------------

class TestStandaloneCrisisBP:
    """Naked BP values with BOTH systolic ≥180 AND diastolic ≥120 → ESCALATE.

    These match the standalone dual-threshold pattern added to close the
    safety gap where patients omit keywords like 'bp' or 'blood pressure'.
    """

    @pytest.mark.parametrize("message", [
        "200/130",                                      # Bare crisis values
        "220/140",
        "180/120",                                      # Exact threshold boundary
        "my reading is 200/120",                        # 'reading' is a prefix keyword too, but test intent
        "I got 195/125 this morning",                   # No BP keyword prefix
        "just measured 210/130 at home",                # Contextual sentence, no prefix
        "it was 185/121 and I feel terrible",           # Just above diastolic threshold
    ])
    def test_standalone_crisis_bp_escalates(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE

    @pytest.mark.parametrize("message", [
        "180/90",                                       # Systolic crisis but diastolic normal
        "150/130",                                      # Diastolic crisis but systolic below 180
        "170/119",                                      # Both just below thresholds
        "179/119",                                      # One digit below both thresholds
        "175/115",                                      # Well below both thresholds
    ])
    def test_standalone_below_dual_threshold_returns_none(self, message: str) -> None:
        assert classify(message) is None


# ---------------------------------------------------------------------------
# Case Insensitivity & Whitespace Variation
# ---------------------------------------------------------------------------

class TestCaseAndWhitespace:
    """Patterns must match regardless of casing or extra whitespace."""

    @pytest.mark.parametrize("message", [
        "CHEST PAIN",
        "Chest Pain",
        "CHEST  PAIN",
        "chest   pain",
        "DIFFICULTY BREATHING",
        "Difficulty  Breathing",
        "SUDDEN SEVERE HEADACHE",
        "VISION CHANGES",
        "BLOOD IN URINE",
        "SLURRED SPEECH",
    ])
    def test_case_insensitive_match(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE

    def test_bp_with_extra_spaces(self) -> None:
        # Spaces around / are handled by \s* in the regex
        assert classify("bp  200  /  130") == ActionEnum.ESCALATE
        assert classify("bp 200/130") == ActionEnum.ESCALATE

    def test_bp_with_prefix_word(self) -> None:
        assert classify("my blood pressure is 190/125") == ActionEnum.ESCALATE


# ---------------------------------------------------------------------------
# Latency Verification
# ---------------------------------------------------------------------------

class TestLatency:
    """classify() must execute in under 5ms."""

    def test_under_5ms_on_match(self) -> None:
        msg = "I'm having chest pain and my bp is 200/130"
        start = time.perf_counter()
        for _ in range(100):
            classify(msg)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000
        assert elapsed_ms < 5, f"Average latency {elapsed_ms:.2f}ms exceeds 5ms"

    def test_under_5ms_on_no_match(self) -> None:
        msg = "What is my blood pressure target if I'm 55 years old, male, taking amlodipine?"
        start = time.perf_counter()
        for _ in range(100):
            classify(msg)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000
        assert elapsed_ms < 5, f"Average latency {elapsed_ms:.2f}ms exceeds 5ms"


# ---------------------------------------------------------------------------
# Detailed result API (classify_detailed)
# ---------------------------------------------------------------------------

class TestDetailedResult:
    """``classify_detailed`` explains every decision with stable rule ids."""

    def test_classify_is_detailed_action(self) -> None:
        for message in ("bp 190/125", "no chest pain", "", "my chest feels tight"):
            assert classify(message) == classify_detailed(message).action

    @pytest.mark.parametrize("message", ["", "   ", "\n\t"])
    def test_empty_message_returns_empty_result(self, message: str) -> None:
        assert classify_detailed(message) == EmergencyResult(action=None)

    def test_fields_on_escalation(self) -> None:
        result = classify_detailed("No chest pain but my bp is 190/125")
        assert result.action == ActionEnum.ESCALATE
        assert result.matched_rules == ["bp_crisis"]
        assert result.negated == ["chest_pain"]
        assert result.bp == (190, 125)
        assert result.evidence_spans == ["190/125"]
        assert result.suppressed == []

    def test_matched_rules_follow_rule_id_order(self) -> None:
        result = classify_detailed("I fainted, my chest hurts and my bp is 200/130")
        assert result.matched_rules == ["bp_crisis", "chest_pain", "syncope"]
        assert all(rule_id in RULE_IDS for rule_id in result.matched_rules)

    def test_suppressed_rules_are_reported(self) -> None:
        result = classify_detailed("I had chest pain last year, it was acid reflux")
        assert result.action is None
        assert result.suppressed == ["chest_pain"]
        assert result.negated == []

    def test_negated_rule_not_listed_when_it_also_fired(self) -> None:
        result = classify_detailed("No chest pain earlier but now my chest feels tight")
        assert result.matched_rules == ["chest_pain"]
        assert result.negated == []

    def test_rule_ids_are_unique(self) -> None:
        assert len(RULE_IDS) == len(set(RULE_IDS))


# ---------------------------------------------------------------------------
# Normalisation and typo tolerance
# ---------------------------------------------------------------------------

class TestNormalisation:
    """Deterministic normalisation applied before every rule."""

    @pytest.mark.parametrize(("raw", "normalised"), [
        ("CHSET  PAIN!!!", "chest pain!!!"),
        ("I’m dizzzzy", "i'm dizzy"),
        ("b.p. 200/130", "bp 200/130"),
        ("blood presure", "blood pressure"),
        ("cant breathe", "can't breath"),
        ("palpitaions", "palpitations"),
        ("unconscous", "unconscious"),
        ("painting", "painting"),
        ("blurred", "blurred"),
        ("arm", "arm"),
        ("line one\n  line two", "line one\n line two"),
    ])
    def test_normalize_text(self, raw: str, normalised: str) -> None:
        assert normalize_text(raw) == normalised

    def test_stopwords_are_real_variants_of_keywords(self) -> None:
        variants: set[str] = set()
        for keyword in ec._FUZZY_KEYWORDS:
            variants |= ec._edit1_variants(keyword)
        assert ec._FUZZY_STOPWORDS <= variants

    def test_fuzzy_keywords_have_minimum_length(self) -> None:
        assert all(len(keyword) >= 6 for keyword in ec._FUZZY_KEYWORDS)

    def test_fuzzy_index_never_maps_protected_words(self) -> None:
        protected = set(ec._FUZZY_KEYWORDS) | ec._FUZZY_STOPWORDS
        assert not protected & set(ec._FUZZY_INDEX)

    def test_edit1_variants_are_exactly_one_edit(self) -> None:
        variants = ec._edit1_variants("chest")
        assert {"chst", "hcest", "chesx", "cheest"} <= variants
        assert "chest" not in variants


# ---------------------------------------------------------------------------
# Blood-pressure parsing
# ---------------------------------------------------------------------------

class TestBloodPressureParsing:
    """BP formats, the reported reading and the sanity checks."""

    @pytest.mark.parametrize(("message", "bp"), [
        ("my bp is over 180", (181, 0)),
        ("systolic 190", (190, 0)),
        ("the bottom number is 125", (0, 125)),
        ("my systolic is 225 and diastolic 130", (225, 130)),
        ("bottom number 125 and top number 150", (150, 125)),
        ("my systolic is 190. Yesterday it was 120/80", (190, 0)),
        ("bp 180/120/88 on the monitor", (180, 120)),
        ("bp 200+", (200, 0)),
        ("pressure is like 200 something", (200, 0)),
        ("bp in the 190s", (190, 0)),
    ])
    def test_crisis_reading_reported(self, message: str, bp: tuple[int, int]) -> None:
        result = classify_detailed(message)
        assert result.matched_rules == ["bp_crisis"]
        assert result.bp == bp

    def test_normal_reading_reported_without_escalation(self) -> None:
        result = classify_detailed("bp 120/80")
        assert result.action is None
        assert result.bp == (120, 80)

    def test_low_reading_without_symptoms_does_not_escalate(self) -> None:
        result = classify_detailed("bp 75/50 yesterday, feeling fine today")
        assert result.action is None
        assert result.bp == (75, 50)

    def test_low_reading_with_symptoms_escalates(self) -> None:
        result = classify_detailed("I'm dizzy and my bp is 70/40")
        assert result.matched_rules == ["bp_low_symptomatic"]
        assert result.bp == (70, 40)
        assert result.evidence_spans == ["dizzy", "70/40"]

    @pytest.mark.parametrize("message", [
        "Next appointment 12/10/2024",       # date
        "take 1/2 tablet",                   # fraction
        "my visual acuity is 20/200",        # implausible (systolic < 50)
        "120/190",                           # systolic must exceed diastolic
        "bp ratio 1.5/2",                    # decimals
        "cholesterol 190/130",               # nearest label is a lab value
        "aim for 130/80",                    # target, not a reading
        "it's 200 mg",                       # dose
        "bp is 400",                         # implausible single systolic
        "diastolic 15",                      # implausible single diastolic
    ])
    def test_non_readings_are_ignored(self, message: str) -> None:
        result = classify_detailed(message)
        assert result.bp is None
        assert result.action is None

    def test_swapped_pair_with_bp_wording_is_read_in_order(self) -> None:
        result = classify_detailed("bp 120/190")
        assert result.matched_rules == ["bp_crisis"]
        assert result.bp == (190, 120)
        assert classify_detailed("my vision is 20/200 and my bp is fine").bp is None

    def test_bare_pair_needs_both_thresholds_without_bp_wording(self) -> None:
        assert classify_detailed("I got 185/100 today").action is None
        assert classify_detailed("I got 185/125 today").matched_rules == ["bp_crisis"]

    def test_contextual_single_value_needs_bp_wording(self) -> None:
        assert classify("today it's 192") is None
        assert classify("my bp was fine yesterday but today it's 192") == ActionEnum.ESCALATE

    def test_historical_reading_is_ignored(self) -> None:
        result = classify_detailed("In 2019 my bp was 210/130. Now it's 125/80.")
        assert result.action is None
        assert result.bp == (125, 80)


# ---------------------------------------------------------------------------
# Negation scope
# ---------------------------------------------------------------------------

class TestNegationScope:
    """Explicit negation suppresses only the symptom it scopes over."""

    @pytest.mark.parametrize("message", [
        "no chest pain",
        "denies chest pain",
        "without any chest pain",
        "not having trouble breathing",
        "never fainted",
        "no longer have chest pain",
        "haven't fainted since Monday",
        "chest pain? no",
        "chest pain: none",
        "no chest pain, shortness of breath or fainting",
        "I don't want to die",
    ])
    def test_negated(self, message: str) -> None:
        result = classify_detailed(message)
        assert result.action is None
        assert result.negated

    @pytest.mark.parametrize("message", [
        "no chest pain but my bp is 190/125",          # BP is never negated
        "no fever, chest pain since morning",           # first object is not a red flag
        "No, I have chest pain",                        # affirmation ends the scope
        "chest pain, no fever",                         # cue after the symptom
        "I'm not sure if it's chest pain",              # pseudo-negation
        "I've never had chest pain like this",          # 'like this' voids negation
        "Chest pain? No. Fainting? Yes.",               # post-negation is per clause
        "no fever over the weekend despite long hours of hard manual labour outdoors "
        "under the hot summer sun followed by chest pain",  # beyond the 80-char window
    ])
    def test_not_negated(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE

    def test_negation_word_never_absorbed_by_filler(self) -> None:
        for message in ("my speech is not slurred", "I'm not having a stroke",
                        "one side of my face is not drooping", "pain is not in my chest"):
            assert classify(message) is None, message


# ---------------------------------------------------------------------------
# Context filters: informational, historical, exertion
# ---------------------------------------------------------------------------

class TestContextFilters:
    """Clearly informational / historical mentions do not escalate."""

    @pytest.mark.parametrize("message", [
        "What are the symptoms of a stroke like face drooping or slurred speech?",
        "How can you tell if someone is having a stroke, like slurred speech?",
        "If I ever get chest pain, should I take aspirin?",
        "My grandmother had slurred speech after her stroke years ago",
        "I used to faint when I saw blood",
        "I get out of breath when I go jogging",
    ])
    def test_suppressed(self, message: str) -> None:
        result = classify_detailed(message)
        assert result.action is None
        assert result.suppressed

    @pytest.mark.parametrize("message", [
        "What are the signs of a stroke? My face feels numb",
        "how do I know if I'm having a stroke",
        "I had chest pain last year and now it's back",
        "I get short of breath walking, even at the gym",
        "Is chest pain a side effect of amlodipine?",
    ])
    def test_kept(self, message: str) -> None:
        assert classify(message) == ActionEnum.ESCALATE


# ---------------------------------------------------------------------------
# Determinism and purity (NON-NEGOTIABLE 1)
# ---------------------------------------------------------------------------

class TestDeterminismAndPurity:
    """Same input, same output; no LLM, network or model dependency."""

    def test_repeated_calls_identical(self) -> None:
        message = "No chest pain but my bp is 190/125 and my face is drooping"
        first = classify_detailed(message)
        assert all(classify_detailed(message) == first for _ in range(50))

    def test_order_independent(self) -> None:
        a, b = "my chest feels tight", "no chest pain"
        before = (classify_detailed(a), classify_detailed(b))
        assert (classify_detailed(a), classify_detailed(b)) == before

    def test_module_imports_only_stdlib_and_enums(self) -> None:
        tree = ast.parse(Path(ec.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported == {
            "__future__", "bisect", "dataclasses", "functools", "re", "unicodedata",
            "medbridge.api.schemas.enums",
        }


# ---------------------------------------------------------------------------
# Latency on the API maximum message length (< 10 ms budget)
# ---------------------------------------------------------------------------

class TestLongMessageLatency:
    """Median of 200 runs on a 2,000-character message stays far below 10 ms."""

    _BASE = (
        "My blood pressure has been around 150/95 this week. I take amlodipine 5 mg and "
        "telmisartan 40 mg. No chest pain, no shortness of breath. What are the warning signs "
        "of a stroke? My father had a stroke in 2015. My cholesterol is 190 and my next "
        "appointment is on 12/10/2024. "
    )

    @pytest.mark.parametrize("tail", ["", " my chest feels tight"])
    def test_median_under_10ms_for_2000_chars(self, tail: str) -> None:
        message = ((self._BASE * 20)[: 2000 - len(tail)] + tail)[:2000]
        assert len(message) == 2000
        timings = []
        for _ in range(200):
            start = time.perf_counter()
            classify(message)
            timings.append((time.perf_counter() - start) * 1000)
        median = statistics.median(timings)
        print(f"\n2,000-char message (tail={tail!r}): median {median:.3f} ms over 200 runs")
        assert median < 10, f"median latency {median:.2f} ms exceeds 10 ms"
        assert (classify(message) == ActionEnum.ESCALATE) == bool(tail)

    @pytest.mark.parametrize("unit", [
        "120/80 ", "pregnant ", "no fainted ", "i'm ", "bp 120 ", "no chest pain, ",
        "chest " + "very " * 5, "1/", "a, ",
    ])
    def test_pathological_repetition_stays_fast(self, unit: str) -> None:
        """Hundreds of matches / cues in one clause must not blow the budget."""
        message = (unit * (2000 // len(unit) + 1))[:2000]
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            classify(message)
            timings.append((time.perf_counter() - start) * 1000)
        assert statistics.median(timings) < 10, f"{unit!r}: {statistics.median(timings):.2f} ms"

"""PII/PHI redaction: quality gate on the synthetic test set + targeted behaviour."""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from medbridge.privacy.evaluation import EVAL_TODAY, evaluate, load_testset, resolve_gold
from medbridge.privacy.redaction import CATEGORIES, detect, redact

TESTSET = Path(__file__).with_name("pii_testset.json")
TODAY = date(2026, 10, 9)


@pytest.fixture(scope="module")
def examples() -> list[dict]:
    return load_testset(TESTSET)


# ---------------------------------------------------------------------------
# Test-set gate
# ---------------------------------------------------------------------------

def test_testset_size_and_composition(examples: list[dict]) -> None:
    negatives = [e for e in examples if e["kind"] == "negative"]
    assert len(examples) >= 120
    assert len(negatives) >= 40
    assert len({e["id"] for e in examples}) == len(examples)
    for ex in examples:
        resolve_gold(ex)  # every gold span exists in its text
        for span in ex["spans"]:
            assert span["category"] in CATEGORIES


def test_overall_precision_and_recall_thresholds(examples: list[dict]) -> None:
    report = evaluate(examples)
    overall = report.overall("strict")
    assert overall.precision >= 0.95, report.to_dict()["errors"]
    assert overall.recall >= 0.90, report.to_dict()["errors"]


def test_heldout_split_regression_floor(examples: list[dict]) -> None:
    """Held-out recall was 0.861 when first measured; guard against regressions."""
    heldout = [e for e in examples if e.get("split") == "heldout"]
    assert heldout
    overall = evaluate(heldout).overall("strict")
    assert overall.precision >= 0.95
    assert overall.recall >= 0.80


def test_zero_clinical_value_corruption(examples: list[dict]) -> None:
    report = evaluate(examples)
    assert report.negatives_altered == []
    assert report.protected_corrupted == []


def test_redaction_is_deterministic(examples: list[dict]) -> None:
    for ex in examples[:40]:
        assert redact(ex["text"], today=EVAL_TODAY) == redact(ex["text"], today=EVAL_TODAY)


# ---------------------------------------------------------------------------
# Targeted behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My DOB is 12/03/1970 and my BP is 150/95.", "My DOB is [AGE: 56] and my BP is 150/95."),
        ("I was born on 3 March 1970.", "I was born on [AGE: 56]."),
        ("date of birth: 1970-03-12", "date of birth: [AGE: 56]"),
        ("born on October 10, 1970", "born on [AGE: 55]"),
        ("DOB 03/25/1970", "DOB [AGE: 56]"),  # month-first when the day is > 12
    ],
)
def test_dob_is_converted_to_age(text: str, expected: str) -> None:
    result = redact(text, today=TODAY)
    assert result.text == expected
    assert result.categories == {"AGE": 1}


def test_placeholders_and_counts() -> None:
    text = (
        "My name is Rahul Sharma, call 98765 43210 or mail rahul@example.com. "
        "Aadhaar 2345 6789 0123, PAN ABCPE1234F, MRN: 00458712. BP 150/95 on Telma 40."
    )
    result = redact(text, today=TODAY)
    assert result.text == (
        "My name is [NAME], call [PHONE] or mail [EMAIL]. "
        "Aadhaar [AADHAAR], PAN [PAN], MRN: [MRN]. BP 150/95 on Telma 40."
    )
    assert result.count == 6
    assert result.categories == {
        "NAME": 1, "PHONE": 1, "EMAIL": 1, "AADHAAR": 1, "PAN": 1, "MRN": 1,
    }


@pytest.mark.parametrize(
    "text",
    [
        "My BP is 145/92, pulse 78, on amlodipine 2.5mg and Telma 40.",
        "I'm 54 and diabetic; HbA1c 7.2%, LDL 130 mg/dL, eGFR 58.",
        "On 12/10 my bp was 150/95 and on 12/10/2024 it was 160/100.",
        "Call 108 or 112 if BP is 190/130 with chest pain.",
        "Platelets 250000, potassium 5.2 mEq/L, INR 2.8, heart rate 112 bpm.",
    ],
)
def test_clinical_text_is_unchanged(text: str) -> None:
    result = redact(text, today=TODAY)
    assert result.text == text
    assert result.count == 0
    assert result.categories == {}


def test_empty_text() -> None:
    assert redact("").text == ""
    assert detect("") == []


def test_spans_do_not_overlap_and_are_sorted() -> None:
    text = "Upload from 192.168.0.12 to http://192.168.1.20/x by rk1975@outlook.com"
    spans = detect(text)
    assert [s.category for s in spans] == ["IP", "URL", "EMAIL"]
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.end <= b.start


def test_unsupported_cue_is_not_redacted() -> None:
    """Names without a strong cue are deliberately left alone (documented limitation)."""
    assert redact("Rahul and I both have high BP").text == "Rahul and I both have high BP"


def test_latency_is_bounded() -> None:
    text = ("My BP log: 145/92, 150/95, 138/88 on Telma 40 and amlodipine 5 mg. " * 30)[:2000]
    start = time.perf_counter()
    for _ in range(20):
        redact(text)
    per_call_ms = (time.perf_counter() - start) * 1000 / 20
    assert per_call_ms < 50

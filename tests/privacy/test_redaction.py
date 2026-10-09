"""PII/PHI redaction: quality gate on the development test set + targeted behaviour.

``pii_testset.json`` (v1.1: its ``dev`` split and the earlier ``heldout`` split) is
*development* data: the detector was tuned on it, so thresholds are asserted here.
The blind held-out set (``pii_testset_blind_v2.json``) is only checked for clinical-value
corruption in ``test_redaction_blind.py``; its metrics are recorded by
``scripts/pii_eval.py`` and never used for tuning.
"""
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
    assert overall.precision >= 0.97, report.to_dict()["errors"]
    assert overall.recall >= 0.97, report.to_dict()["errors"]


def test_only_documented_out_of_scope_misses_remain(examples: list[dict]) -> None:
    """Every remaining development miss is a name with no contextual cue (documented)."""
    report = evaluate(examples)
    by_id = {e["id"]: e for e in examples}
    for err in report.to_dict()["errors"]:
        assert "out of scope" in by_id[err["id"]].get("note", ""), err


def test_previous_heldout_split_regression_floor(examples: list[dict]) -> None:
    """The v1 held-out split (recall 0.861 when first measured) is development data now."""
    heldout = [e for e in examples if e.get("split") == "heldout"]
    assert heldout
    overall = evaluate(heldout).overall("strict")
    assert overall.precision >= 0.97
    assert overall.recall >= 0.95


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


# ---------------------------------------------------------------------------
# Generalisation of the v1 held-out misses (self-authored variants, not test-set items)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # names after introduction / relation / label / sign-off cues
        ("it's me Kiran Rao, BP 150/95", "it's me [NAME], BP 150/95"),
        ("Its me, Anil. pulse 88", "Its me, [NAME]. pulse 88"),
        ("My wife's name is Geeta Menon and she is 61.", "My wife's name is [NAME] and she is 61."),
        ("her son's name is rohit", "her son's name is [NAME]"),
        ("My late father's name was Mohan Das.", "My late father's name was [NAME]."),
        ("the man whose name is Prakash takes Concor 5",
         "the man whose name is [NAME] takes Concor 5"),
        ("Name of patient: Lata Kulkarni", "Name of patient: [NAME]"),
        ("Signed: Dev Patel", "Signed: [NAME]"),
        ("signed by Ritu Sen", "signed by [NAME]"),
        ("Signature - Harish Gowda", "Signature - [NAME]"),
        ("Thank you so much. Thanks, Nandini", "Thank you so much. Thanks, [NAME]"),
        ("Will do. Cheers, Omar Farooq!", "Will do. Cheers, [NAME]!"),
        ("Hi, this is Priyanka. My BP is 140/90.", "Hi, this is [NAME]. My BP is 140/90."),
        ("Hello it's Varun, quick question about Telma 40",
         "Hello it's [NAME], quick question about Telma 40"),
        ("Myself Sanjay Gupta, 52 years, BP 160/100", "Myself [NAME], 52 years, BP 160/100"),
        ("her daughter Ishita, 14, has BP 128/80", "her daughter [NAME], 14, has BP 128/80"),
        ("my elder brother Naveen had a stroke", "my elder brother [NAME] had a stroke"),
        ("my mother-in-law Kamala is on Amlong 5", "my mother-in-law [NAME] is on Amlong 5"),
        ("a friend named Joseph told me about Telma", "a friend named [NAME] told me about Telma"),
        ("Doctor Iyer said my EF is 35%", "Doctor [NAME] said my EF is 35%"),
        # +91 / +1 numbers in any grouping, contact-verb labels, grouped STD landlines
        ("call me on +91 22 2654 3210 tomorrow", "call me on [PHONE] tomorrow"),
        ("reach my son at 98450 12345", "reach my son at [PHONE]"),
        ("office +91-80-4123-4567", "office [PHONE]"),
        ("+91 (80) 41234567", "[PHONE]"),
        ("0091 80 4123 4567", "[PHONE]"),
        ("+1 6465550199 is my US number", "[PHONE] is my US number"),
        ("landline 080 4123 4567", "landline [PHONE]"),
        ("Text me on 984 501 2345", "Text me on [PHONE]"),
        # PIN / ZIP after a comma-separated place chain or a US state code
        ("I live near Indiranagar, Bengaluru - 560038", "I live near Indiranagar, Bengaluru - [PINCODE]"),
        ("We moved to Andheri East, Mumbai 400 069 last year",
         "We moved to Andheri East, Mumbai [PINCODE] last year"),
        ("from Salt Lake, Sector 5, Kolkata 700091", "from Salt Lake, Sector 5, Kolkata [PINCODE]"),
        ("Address: Koramangala, Bangalore 560034", "Address: Koramangala, Bangalore [PINCODE]"),
        ("staying in Austin, TX 78701 now", "staying in Austin, TX [PINCODE] now"),
        ("Flat 9, Rose Villa, Banjara Hills, Hyderabad – 500034", "[ADDRESS] – [PINCODE]"),
        ("born on the 3rd of March 1970", "born on the [AGE: 56]"),
    ],
)
def test_generalised_cues(text: str, expected: str) -> None:
    assert redact(text, today=TODAY).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "it's me again, BP 150/95",
        "It's me who checks BP daily",
        "Hi, this is urgent: BP 190/120",
        "Hello, this is Telma 40 right?",
        "Hi, it's Amlodipine that makes my ankles swell?",
        "Hi this is Normal for my age?",
        "Thanks, Doctor",
        "Thanks, Telma helped a lot",
        "Thanks, Ecosprin",
        "Thanks Again",
        "Thank you, Bye",
        "BP 128/82 - Thanks",
        "Dear Doctor, this is regarding my BP 150/95",
        "Sign Up for the BP camp at 9 am",
        "I sign up for walking club",
        "my name is mentioned on the report",
        "my wife's BP is 150/90",
        "my husband's Telma 40 dose",
        "Myself checking BP twice daily",
        "I'm on Telma 40 and Amlong 5",
        "A rise in Platelets 250000 is fine?",
        "Platelet count in Lakh 150000?",
        "in Week 2, 150000 steps",
        "In India 150000 people die of stroke?",
        "price in Rs 150000 total",
        "Increased in Dec 2024 to 80 mg",
        "sugar readings 650 700 8000",
        "call me at 9 pm",
        "dial 102 for ambulance",
        "Call 108 or 112 now",
        "BP 145/92 at 08:15, 150/95 at 20:30",
        "Dose went from 25 mg to 50 mg, 12.5 to 25",
        "INR 2.5, HbA1c 6.8, LDL 98, TG 150",
    ],
)
def test_new_cues_leave_clinical_text_unchanged(text: str) -> None:
    result = redact(text, today=TODAY)
    assert result.text == text
    assert result.count == 0


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

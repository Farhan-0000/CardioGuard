"""Unit tests for the deterministic blood-pressure parser (medbridge.nlp.bp_parse)."""
from __future__ import annotations

import pytest

from medbridge.nlp.bp_parse import extract_bp_readings, is_plausible_bp, parse_bp_mentions


@pytest.mark.parametrize(("text", "expected"), [
    ("my bp was 145/92 this morning", [(145, 92)]),
    ("it was 150 over 95", [(150, 95)]),
    ("bp 190/125!!", [(190, 125)]),
    ("BP 140 by 90 today", [(140, 90)]),
    ("I measured 150/95 mmHg and later 138/88", [(150, 95), (138, 88)]),
    ("Readings: 145/92, 150/95", [(145, 92), (150, 95)]),
    ("my blood pressure is 150 / 95, is that ok?", [(150, 95)]),
])
def test_readings_extracted(text: str, expected: list[tuple[int, int]]) -> None:
    assert extract_bp_readings(text) == expected


@pytest.mark.parametrize("text", [
    "should my bp be under 130/80?",
    "is 130/80 normal?",
    "what is the target of 120/80",
    "keep it below 140/90",
    "is a reading > 180/120 dangerous",
])
def test_targets_are_not_readings(text: str) -> None:
    mentions = parse_bp_mentions(text)
    assert mentions and all(m.kind == "target" for m in mentions)
    assert extract_bp_readings(text) == []


@pytest.mark.parametrize("text", [
    "on 12/10/2024 I felt fine",
    "24/7 monitoring",
    "ratio 3/4",
    "80/120",          # diastolic above systolic
    "350/90",          # systolic out of range
    "1.5/2.5",
])
def test_implausible_or_non_bp_ignored(text: str) -> None:
    assert parse_bp_mentions(text) == []


def test_ambiguous_mention_is_unknown() -> None:
    mentions = parse_bp_mentions("what about 145/92")
    assert [m.kind for m in mentions] == ["unknown"]


def test_duplicates_removed() -> None:
    assert extract_bp_readings("my bp 150/95, again my bp 150/95") == [(150, 95)]


def test_spans_point_at_the_numbers() -> None:
    text = "my bp was 150 over 95"
    mention = parse_bp_mentions(text)[0]
    assert text[mention.start:mention.end] == "150 over 95"


@pytest.mark.parametrize(("sys_bp", "dia_bp", "ok"), [
    (60, 30, True), (300, 200, True), (300, 201, False), (301, 90, False), (59, 40, False), (120, 120, False),
])
def test_sanity_ranges(sys_bp: int, dia_bp: int, ok: bool) -> None:
    assert is_plausible_bp(sys_bp, dia_bp) is ok


# ---------------------------------------------------------------------------
# Regression (llm-prompts-01): questions, targets and hypotheticals are not readings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "kind"), [
    ("Is 130/80 a good target for someone with diabetes like me? My eGFR was 45", "target"),
    ("Is 130/80 a good target?", "target"),
    ("is 130/80 my target", "target"),
    ("should it be below 140/90", "target"),
    ("I was told 130/80 is the target", "target"),
    ("Would 125/75 be a sensible goal with my heart failure?", "target"),
    ("what if it's 160/100", "hypothetical"),
    ("What if my bp is 160/100?", "hypothetical"),
    ("if it reaches 180/120 should I go to hospital", "hypothetical"),
    ("Is 140/90 considered high blood pressure?", "unknown"),
    ("At what point is 160/100 dangerous?", "unknown"),
    ("Is a reading of 150/95 bad?", "unknown"),
    ("Hi doctor, is 150/95 bad?", "unknown"),
    ("Should I worry about 150/95?", "unknown"),
])
def test_questions_targets_and_hypotheticals_are_not_readings(text: str, kind: str) -> None:
    assert [m.kind for m in parse_bp_mentions(text)] == [kind]
    assert extract_bp_readings(text) == []


@pytest.mark.parametrize(("text", "expected"), [
    ("is my bp of 150/95 too high?", [(150, 95)]),
    ("Is 150/95 this morning too high?", [(150, 95)]),
    ("is mine 150/95 normal for my age", [(150, 95)]),
    ("When I checked this morning it was 150/95", [(150, 95)]),
    ("it said 128/76", [(128, 76)]),
    ("My BP hasn't been well controlled lately, around 150/95", [(150, 95)]),
    ("Clinic 152/94, home average 146/90", [(152, 94), (146, 90)]),
    ("its 210/130 and my head is killing me", [(210, 130)]),
    ("150/95 this morning, is that ok", [(150, 95)]),
])
def test_first_person_measured_readings_still_found(
    text: str, expected: list[tuple[int, int]]
) -> None:
    assert extract_bp_readings(text) == expected


def test_question_scope_does_not_borrow_an_earlier_clause_cue() -> None:
    """'my bp was X' makes X a reading, not the number asked about afterwards."""
    mentions = parse_bp_mentions("my bp was 150/95, is 160/100 dangerous?")
    assert [(m.systolic, m.kind) for m in mentions] == [(150, "reading"), (160, "unknown")]


def test_reading_followed_by_an_evaluative_question_stays_a_reading() -> None:
    for text in ("my bp is 150/95 is that normal", "my bp is 150/95, is that a good range?"):
        assert [m.kind for m in parse_bp_mentions(text)] == ["reading"], text


@pytest.mark.parametrize(("text", "kind"), [
    ("I want to know if 150/95 is ok", "unknown"),       # indirect question, not a condition
    ("I want to know if my 150/95 is ok", "reading"),
    ("my bp 120/80 is normal", "reading"),               # own reading described as normal
    ("Is my 130/80 a good target?", "reading"),
])
def test_indirect_questions_and_own_values(text: str, kind: str) -> None:
    assert [m.kind for m in parse_bp_mentions(text)] == [kind]


@pytest.mark.parametrize(("text", "kinds"), [
    ("my target is 130/80, my bp is 150/95", ["target", "reading"]),
    ("keep it between 120/80 and 130/85", ["target", "target"]),
    ("Readings: 145/92, 150/95", ["reading", "reading"]),
    ("what if it goes to 170/105 or 180/110", ["hypothetical", "hypothetical"]),
    ("my bp was 150/95, is 160/100 dangerous?", ["reading", "unknown"]),
])
def test_cues_belong_to_the_nearest_number(text: str, kinds: list[str]) -> None:
    assert [m.kind for m in parse_bp_mentions(text)] == kinds

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

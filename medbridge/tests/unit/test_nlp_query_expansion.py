"""Unit tests for deterministic lay → clinical query expansion (medbridge.nlp.query_expansion)."""
from __future__ import annotations

import pytest

from medbridge.nlp.query_expansion import (
    MAX_EXPANSION_CHARS,
    MAX_EXPANSION_TERMS,
    expand_query,
    expansion_terms,
)


@pytest.mark.parametrize(("text", "expected"), [
    ("is my water pill safe", "thiazide diuretic"),
    ("can I take a blood thinner", "anticoagulant antiplatelet"),
    ("my bp is up", "blood pressure"),
    ("my pressure is up", "blood pressure"),
    ("I have high bp", "hypertension"),
    ("low blood pressure when standing", "hypotension"),
    ("after my heart attack", "myocardial infarction acute coronary syndrome"),
    ("irregular heartbeat at night", "atrial fibrillation arrhythmia palpitations"),
    ("my heart is racing", "atrial fibrillation arrhythmia palpitations"),
    ("a fluttering feeling", "atrial fibrillation arrhythmia palpitations"),
    ("which cholesterol pill is best", "statin"),
    ("how much salt", "sodium"),
    ("swollen ankles", "edema"),
    ("I have a weak heart", "heart failure reduced ejection fraction"),
    ("chest tightness on stairs", "angina"),
    ("my sugar is high", "diabetes"),
    ("my kidneys are weak", "chronic kidney disease eGFR"),
    ("I am pregnant", "hypertensive disorders of pregnancy preeclampsia"),
])
def test_lay_terms_expanded(text: str, expected: str) -> None:
    expanded = expand_query(text)
    assert expanded.startswith(text)
    assert expected in expanded


def test_brand_generic_and_class_appended() -> None:
    expanded = expand_query("is Telma safe")
    assert "telmisartan" in expanded and "angiotensin receptor blocker" in expanded


def test_generic_gets_class_only() -> None:
    assert expand_query("lisinopril cough") == "lisinopril cough ACE inhibitor"


def test_context_contributes_terms_not_already_in_query() -> None:
    expanded = expand_query("hypertension treatment", context="my water pill and Telma")
    assert expanded.startswith("hypertension treatment ")
    assert "thiazide diuretic" in expanded and "telmisartan" in expanded
    assert expanded.count("hypertension") == 1


def test_no_duplicates_and_existing_terms_skipped() -> None:
    expanded = expand_query("high bp hypertension blood pressure water pill thiazide diuretic")
    assert expanded == "high bp hypertension blood pressure water pill thiazide diuretic"


def test_deterministic() -> None:
    text = "water pill, blood thinner, Telma, Concor and salt with swollen ankles"
    assert expand_query(text) == expand_query(text)


def test_bounded() -> None:
    text = ("water pill blood thinner heart attack racing heart cholesterol pill salt swollen "
            "ankles weak heart chest pain sugar kidney pregnant smoking drinking Telma Concor "
            "Ecosprin Rosuvas Amlong Dytor")
    expanded = expand_query(text)
    assert len(expanded) <= len(text) + 1 + MAX_EXPANSION_CHARS
    assert len(expansion_terms(text)) <= MAX_EXPANSION_TERMS


def test_plain_clinical_query_unchanged() -> None:
    assert expand_query("  first-line therapy   stage 1  ") == "first-line therapy stage 1"
    assert expand_query("") == ""

"""Unit tests for the deterministic small-talk detector (medbridge/core/smalltalk.py, ADL-029)."""

from __future__ import annotations

import pytest

from medbridge.core.smalltalk import is_small_talk
from medbridge.core.templates import SMALL_TALK_TEMPLATES


@pytest.mark.parametrize(
    "text,key",
    [
        ("hi", "greeting"),
        ("Hello there!", "greeting"),
        ("hey, how are you?", "greeting"),
        ("Good morning", "greeting"),
        ("namaste", "greeting"),
        ("hi doctor", "greeting"),
        ("thanks!", "thanks"),
        ("Thank you so much for the help", "thanks"),
        ("ok thanks", "thanks"),
        ("thx", "thanks"),
        ("bye", "goodbye"),
        ("Take care", "goodbye"),
        ("ok bye for now", "goodbye"),
        ("who are you?", "identity"),
        ("What can you do?", "identity"),
        ("what is medbridge", "identity"),
        ("are you a doctor?", "identity"),
        ("help", "identity"),
        ("ok", "acknowledgement"),
        ("cool", "acknowledgement"),
        ("got it", "acknowledgement"),
        ("Makes sense", "acknowledgement"),
    ],
)
def test_small_talk_detected(text, key):
    assert is_small_talk(text) == key
    assert key in SMALL_TALK_TEMPLATES


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "yes",  # often answers a SOFT-ASK question
        "no",
        "hi, my bp is 150/95",
        "hello, I have chest pain",
        "thanks, should I take my pill now?",
        "what about amlodipine?",
        "Is 140/90 high?",
        "hello, I have a question",
        "hi, just testing",
        "ok so my doctor said to stop the medicine",
        "how are you supposed to measure blood pressure",
        "thanks " * 20,
        "I feel dizzy",
    ],
)
def test_clinical_or_ambiguous_messages_are_not_small_talk(text):
    assert is_small_talk(text) is None


def test_small_talk_templates_are_claim_free():
    for text in SMALL_TALK_TEMPLATES.values():
        lower = text.lower()
        assert "mg" not in lower.split()
        assert not any(ch.isdigit() for ch in text)
        assert "{" not in text and "}" not in text

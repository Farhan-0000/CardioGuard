"""
Deterministic blood-pressure mention parser.

Recognises the ways patients write a reading — ``145/92``, ``150 over 95``,
``bp 190/125``, ``140 by 90``, ``150/95 mmHg`` — validates it against
physiological sanity ranges (systolic 60–300, diastolic 30–200, systolic >
diastolic) and classifies each mention as the patient's own *reading*, a
*target* or threshold ("should it be under 130/80?"), or *unknown*.

The Context Extractor merges *reading* mentions with the LLM's BP events so a
reading is never lost to an LLM omission, and drops LLM BP events that are
really targets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SYSTOLIC_RANGE = (60, 300)
DIASTOLIC_RANGE = (30, 200)

BPKind = Literal["reading", "target", "unknown"]

_BP_RE = re.compile(
    r"(?<![\d/.])(\d{2,3})\s*(?:/|\\|\bover\b|\bby\b)\s*(\d{2,3})(?![\d.]|\s*/\s*\d)",
    re.IGNORECASE,
)
_TARGET_BEFORE = re.compile(
    r"\b(?:target|goal|aim|below|under|less than|lower than|above|more than|greater than|"
    r"higher than|keep (?:it|my bp|my blood pressure)?|should (?:it |my bp |my blood pressure )?be|"
    r"normal|ideal|threshold|range|between|cut-?off)\b|[<>≤≥]",
    re.IGNORECASE,
)
_TARGET_AFTER = re.compile(
    r"^\s*(?:mm ?hg)?\s*(?:is|are)?\s*(?:a |the )?"
    r"(?:normal|target|goal|ideal|threshold|range|cut-?off)\b",
    re.IGNORECASE,
)
_READING_BEFORE = re.compile(
    r"\b(?:my|i|i'm|im|was|is|were|reading|readings|measured|checked|got|came|today|"
    r"tonight|yesterday|morning|evening|bp|blood pressure|pressure|it's|its|at|of)\b",
    re.IGNORECASE,
)

_WINDOW_BEFORE = 40
_WINDOW_AFTER = 25


@dataclass(frozen=True)
class BPMention:
    """One sanity-checked blood-pressure mention in a text."""

    systolic: int
    diastolic: int
    start: int
    end: int
    kind: BPKind


def is_plausible_bp(systolic: int, diastolic: int) -> bool:
    """True when the pair passes the physiological sanity ranges."""
    return (
        SYSTOLIC_RANGE[0] <= systolic <= SYSTOLIC_RANGE[1]
        and DIASTOLIC_RANGE[0] <= diastolic <= DIASTOLIC_RANGE[1]
        and systolic > diastolic
    )


def _classify(text: str, start: int, end: int) -> BPKind:
    before = text[max(0, start - _WINDOW_BEFORE):start]
    # Only look back within the current sentence/clause.
    before = re.split(r"[.!?;\n]", before)[-1]
    after = text[end:end + _WINDOW_AFTER]
    if _TARGET_BEFORE.search(before) or _TARGET_AFTER.search(after):
        return "target"
    if _READING_BEFORE.search(before):
        return "reading"
    return "unknown"


def parse_bp_mentions(text: str) -> list[BPMention]:
    """All plausible BP mentions in ``text`` with their classification."""
    mentions: list[BPMention] = []
    for match in _BP_RE.finditer(text or ""):
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        if not is_plausible_bp(systolic, diastolic):
            continue
        mentions.append(
            BPMention(systolic, diastolic, match.start(), match.end(),
                      _classify(text, match.start(), match.end()))
        )
    return mentions


def extract_bp_readings(text: str) -> list[tuple[int, int]]:
    """(systolic, diastolic) pairs that are clearly the patient's own readings (deduplicated)."""
    readings: list[tuple[int, int]] = []
    for mention in parse_bp_mentions(text):
        pair = (mention.systolic, mention.diastolic)
        if mention.kind == "reading" and pair not in readings:
            readings.append(pair)
    return readings

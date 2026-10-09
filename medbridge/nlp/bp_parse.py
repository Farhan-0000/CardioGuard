"""
Deterministic blood-pressure mention parser.

Recognises the ways patients write a reading — ``145/92``, ``150 over 95``,
``bp 190/125``, ``140 by 90``, ``150/95 mmHg`` — validates it against
physiological sanity ranges (systolic 60–300, diastolic 30–200, systolic >
diastolic) and classifies each mention by what the clause around it says:

* ``reading``      — the patient's own measured value ("my bp was 150/95",
  "it said 128/76", "is my 150/95 too high?");
* ``target``       — a target or threshold ("should it be under 130/80?",
  "is 130/80 a good target?", "is 130/80 my target?");
* ``hypothetical`` — a conditional value ("what if it's 160/100?",
  "if it reaches 180/120 …");
* ``unknown``      — anything else, notably general questions about a number
  ("is 140/90 considered high?", "at what point is 160/100 dangerous?").

Only ``reading`` mentions count as the patient's own values. A question or a
conditional clause makes a number a reading only when the clause itself ties it
to the patient (``my bp``, ``I measured``, ``today``…): asking about a number
is not reporting one.

The Context Extractor merges *reading* mentions with the LLM's BP events so a
reading is never lost to an LLM omission, and drops LLM BP events that are
really targets or hypothetical values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SYSTOLIC_RANGE = (60, 300)
DIASTOLIC_RANGE = (30, 200)

BPKind = Literal["reading", "target", "hypothetical", "unknown"]

# Kinds that are never the patient's own measured value.
NOT_A_READING: frozenset[str] = frozenset({"target", "hypothetical"})

_BP_RE = re.compile(
    r"(?<![\d/.])(\d{2,3})\s*(?:/|\\|\bover\b|\bby\b)\s*(\d{2,3})(?![\d.]|\s*/\s*\d)",
    re.IGNORECASE,
)
_TARGET_BEFORE = re.compile(
    r"\b(?:target|goal|aim|below|under|less than|lower than|above|more than|greater than|"
    r"higher than|keep (?:it|my bp|my blood pressure)?|should (?:it |my bp |my blood pressure )?be|"
    r"normal|ideal|optimal|recommended|threshold|range|between|cut-?off)\b|[<>≤≥]",
    re.IGNORECASE,
)
# "130/80 is the target", "is 130/80 a good target?", "is 130/80 my target?".
_TARGET_AFTER = re.compile(
    r"^\s*(?:mm ?hg)?\s*(?:is|are|be)?\s*(?:(?:a|an|the|my|your|our)\s+)?"
    r"(?:(?:good|right|healthy|safe|usual|sensible|reasonable|recommended|correct)\s+)?"
    r"(?:normal|target|goal|ideal|threshold|range|cut-?off|limit)\b",
    re.IGNORECASE,
)
# A question clause: an auxiliary or wh-word at the start of the (sub-)clause, or
# an indirect question ("I want to know if/whether ...").
_INTERROGATIVE = re.compile(
    r"(?:^|[,:(]|\b(?:and|but|or|so|also)\b)\s*"
    r"(?:is|are|was|were|would|should|could|can|does|do|did|will|shall|what|when|how|which|"
    r"why|where|at what)\b"
    r"|\b(?:know|wonder|wondering|ask|asking|check|checking|see|tell me|sure|unsure|curious)"
    r"\s+(?:if|whether)\b|\bwhether\b",
    re.IGNORECASE,
)
# A conditional / hypothetical clause ("if" after a verb of asking or knowing is an
# indirect question instead, see _INTERROGATIVE).
_HYPOTHETICAL = re.compile(
    r"\b(?:what if|if|suppose|supposing|imagine|hypothetically|in case|let'?s say|"
    r"were to|unless)\b",
    re.IGNORECASE,
)
_INDIRECT_QUESTION_VERB = re.compile(
    r"\b(?:know|wonder|wondering|ask|asking|check|checking|see|tell me|sure|unsure|curious)\s*$",
    re.IGNORECASE,
)
# Cues that tie a number to the patient's own measurement, valid inside a
# question or conditional clause ("is my 150/95 too high?", "did I measure ...").
_OWN_BEFORE = re.compile(
    r"\bmy\s+(?:[\w'-]+\s+){0,3}?(?:bp|blood pressure|pressure|readings?|numbers?|average|"
    r"result|levels?|monitor|machine|cuff|watch)\b"
    r"|\bmy\s*$|\bmine\b"
    r"|\b(?:i|we|i've|ive|i'd|id)\s+(?:\w+\s+)?(?:measured|checked|got|get|took|had|recorded|"
    r"logged|saw|keep getting)\b"
    r"|\b(?:today|tonight|yesterday|this morning|this evening|last night|earlier today)\b",
    re.IGNORECASE,
)
_OWN_AFTER = re.compile(
    r"^\s*(?:mm ?hg)?[\s,]*(?:this morning|this evening|last night|earlier today|today|tonight|"
    r"yesterday|at home|at the (?:clinic|pharmacy|doctor'?s)|on my (?:monitor|machine|cuff|watch))\b",
    re.IGNORECASE,
)
# Declarative clauses: any reporting or measurement cue marks a reading.
_READING_BEFORE = re.compile(
    r"\b(?:my|mine|i|i'm|im|me|was|were|reading|readings|measured|checked|got|came|coming|"
    r"today|tonight|yesterday|morning|evening|bp|blood pressure|pressure|it's|its|it is|"
    r"said|says|shows|showed|reads|read|clinic|home|average|avg|monitor|machine|cuff|watch|"
    r"lately|recently|usually|around)\b",
    re.IGNORECASE,
)

_WINDOW_BEFORE = 60
_WINDOW_AFTER = 30


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


def _last_match_start(pattern: re.Pattern[str], text: str) -> int | None:
    starts = [m.start() for m in pattern.finditer(text)]
    return starts[-1] if starts else None


def _is_hypothetical(clause: str) -> bool:
    """A conditional cue in the clause that is not an indirect question ("know if")."""
    return any(
        not _INDIRECT_QUESTION_VERB.search(clause[:m.start()])
        for m in _HYPOTHETICAL.finditer(clause)
    )


def _has_cue(before: str, after: str) -> bool:
    """Whether the words around a mention say anything about what kind of value it is."""
    return any(p.search(before) for p in (
        _TARGET_BEFORE, _HYPOTHETICAL, _INTERROGATIVE, _OWN_BEFORE, _READING_BEFORE,
    )) or bool(_TARGET_AFTER.search(after) or _OWN_AFTER.search(after))


def _classify(
    text: str, start: int, end: int, previous: BPMention | None = None
) -> BPKind:
    before = text[max(0, start - _WINDOW_BEFORE):start]
    # Only look back within the current sentence/clause.
    before = re.split(r"[.!?;\n]", before)[-1]
    after = text[end:end + _WINDOW_AFTER]
    if previous is not None and previous.end > start - len(before):
        # Cues before an earlier number in the same clause belong to that number
        # ("my target is 130/80, my bp is 150/95"); a bare continuation of a list
        # ("Readings: 145/92, 150/95", "between 120/80 and 130/85") shares its kind.
        before = text[previous.end:start]
        if not _has_cue(before, after):
            return previous.kind
    if _TARGET_BEFORE.search(before):
        return "target"
    if _is_hypothetical(before):
        # "what if it's 160/100?", "if my bp reaches 180/120": a conditional value,
        # even when it is about the patient's own blood pressure.
        return "hypothetical"
    # A question about a number is not a report of one, unless the question
    # clause itself ties the number to the patient ("is my 150/95 too high?").
    interrogative = _last_match_start(_INTERROGATIVE, before)
    scope = before if interrogative is None else before[interrogative:]
    if _OWN_BEFORE.search(scope) or _OWN_AFTER.search(after):
        return "reading"
    if _TARGET_AFTER.search(after):
        return "target"
    if interrogative is None and _READING_BEFORE.search(before):
        return "reading"
    return "unknown"


def parse_bp_mentions(text: str) -> list[BPMention]:
    """All plausible BP mentions in ``text`` with their classification."""
    mentions: list[BPMention] = []
    for match in _BP_RE.finditer(text or ""):
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        if not is_plausible_bp(systolic, diastolic):
            continue
        previous = mentions[-1] if mentions else None
        mentions.append(
            BPMention(systolic, diastolic, match.start(), match.end(),
                      _classify(text, match.start(), match.end(), previous))
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

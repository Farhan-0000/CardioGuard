"""
Deterministic natural-language emergency classifier (ADL-002 + finalization NL extension).

The fast-path triage stage of the pipeline: it runs on the **raw** patient
message **before any LLM call** and returns ``ActionEnum.ESCALATE`` for
hypertensive crises, red-flag symptoms and acute danger. It is pure Python —
regular expressions compiled at import time plus a few dictionary look-ups — so
it is deterministic, needs no network and no model, and runs in well under the
10 ms budget (typically < 2 ms for a 2,000-character message).

Processing steps
----------------
1. **Normalisation** (:func:`normalize_text`): NFKC, unified apostrophes/dashes,
   lower case, letter runs longer than two collapsed (``"dizzzzy"`` →
   ``"dizzy"``), ``b.p.`` → ``bp``, a fixed misspelling map (``chset`` →
   ``chest``, ``breathe`` → ``breath``, ``headace`` → ``headache``, ``presure``
   → ``pressure``, ``cant`` → ``can't`` …) and a deterministic edit-distance-1
   matcher (Damerau/OSA: one insertion, deletion, substitution or adjacent
   transposition) for red-flag keywords of length ≥ 6. The matcher is a
   pre-computed variant → keyword table, so a look-up is O(1); variants that
   are ordinary English words (``painting``, ``blurred``, ``lightness`` …)
   are excluded so they are never "corrected" into a symptom.
2. **Blood pressure** extraction: ``210/125``, ``190 over 125``, ``185 by
   110``, ``systolic 190``, ``top number 200``, ``bp of 185``, ``my bp is over
   180`` (strictly greater → 181), ``bp 200+``, ``pressure is like 200
   something``, ``in the 190s``, ``it hit 200`` … with sanity checks
   (systolic 50–300, diastolic 20–200, systolic > diastolic; no decimals, no
   dates such as ``12/10/2024``, no fractions/doses such as ``1/2 tablet`` or
   ``180 mg``, no lab values, heart rates, targets such as ``below 130/80`` or
   conditional thresholds such as ``call if it is over 180/110``). A pair typed
   in the wrong order next to BP wording (``bp 120/190``) is read swapped.
   Crisis: systolic ≥ 180 **or** diastolic ≥ 120 when the message mentions
   blood pressure; a bare number pair with no BP wording must meet **both**
   thresholds (avoids order numbers, room numbers …). A systolic < 80 with
   dizziness/faintness/confusion is also escalated (``bp_low_symptomatic``).
3. **Red-flag rules**: one compiled regex per rule id (see :data:`RULE_IDS`).
4. **Context filters** applied per clause (clause boundaries: ``. ; ? !``,
   newline, *but*, *however*, *although*, *though*, *whereas*):

   * **Negation** — an explicit cue (*no, not, never, without, denies, no
     longer, free of, haven't had, don't feel* …) before the symptom in the
     same clause, with no affirmation in between (*I have*, *my …*, *now*,
     *suddenly* …). Lists are followed (``no chest pain, shortness of breath or
     fainting``) only when the first negated item is itself a red flag, so
     ``no fever, chest pain`` still escalates. Answer-style post-negation
     (``chest pain? no``, ``chest pain: none``) is supported. Pseudo-negations
     (*not sure*, *no relief*, *not going away*, *never felt this bad* …) do
     not negate.
   * **Informational** questions (*what are the warning signs of …*, *how do I
     recognise …*, *what should I do if someone faints*, *if I ever get …*) —
     unless the message also carries a first-person present marker (*I'm
     having*, *right now*, *my chest is …*, *I have … pain*).
   * **Historical** mentions with an explicit time anchor (*last year*, *in
     2015*, *years ago*, *as a child*, *history of*, *used to*, *it turned out
     to be reflux* …) and no recency marker in the same clause (*now*,
     *today*, *again*, *still*, *since*, *an hour ago* …).
   * **Heavy exertion** (breathlessness only): *short of breath when I run /
     at the gym* is not a red flag; walking, stairs, rest or lying down are.

   A negated, informational or historical symptom never suppresses a BP crisis
   or another affirmed symptom (``no chest pain but my bp is 190/125`` →
   ESCALATE). BP readings are only filtered by historical anchors, targets and
   conditional thresholds.

Context-handling decisions (safety first: when ambiguous, keep recall)
----------------------------------------------------------------------
=======================================  =========================================  ========
Mention                                  Example                                    Decision
=======================================  =========================================  ========
First person, present symptom            "my chest feels tight"                     ESCALATE
First person, recent past (no anchor)    "I passed out twice"                       ESCALATE
Third person, present (bystander)        "my mother is unconscious"                 ESCALATE
Side-effect / can-X-cause-Y question     "is chest pain a side effect of ...?"      ESCALATE
Personal conditional question            "how do I know if I'm having a stroke"     ESCALATE
BP crisis value, question or statement   "is 190/125 dangerous?"                    ESCALATE
Negated symptom                          "no chest pain", "denies dizziness"        no
Negated symptom + BP crisis              "no chest pain but my bp is 190/125"       ESCALATE
Educational / generic-person question    "what are the warning signs of a stroke?"  no
Own hypothetical future                  "if I ever get chest pain ..."             no
Historical, time-anchored (any person)   "my father had a stroke in 2015"           no
Historical + recency in the same clause  "chest pain last year, back now"           ESCALATE
Breathlessness with heavy exertion only  "out of breath when I go jogging"          no
Bare number pair below dual threshold    "order number 180/90"                      no
Conditional threshold (if ... over N)    "call if my BP is above 180"               no
=======================================  =========================================  ========

Technical Specification Part III §3.2; constitution §12 NON-NEGOTIABLE 1.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property

from medbridge.api.schemas.enums import ActionEnum

__all__ = [
    "DIASTOLIC_CRISIS",
    "RULE_IDS",
    "SYSTOLIC_CRISIS",
    "SYSTOLIC_LOW",
    "EmergencyResult",
    "classify",
    "classify_detailed",
    "normalize_text",
]

SYSTOLIC_CRISIS = 180
DIASTOLIC_CRISIS = 120
SYSTOLIC_LOW = 80


@dataclass(frozen=True)
class EmergencyResult:
    """Detailed outcome of :func:`classify_detailed`.

    Attributes:
        action: ``ActionEnum.ESCALATE`` when at least one rule fired, else ``None``.
        matched_rules: Stable rule ids that fired, in :data:`RULE_IDS` order.
        negated: Rule ids whose only matches were explicitly negated.
        bp: The decisive blood-pressure reading as ``(systolic, diastolic)``;
            a component that was not stated is ``0``. ``None`` when no
            plausible reading was found.
        evidence_spans: Normalised text spans that triggered the escalation.
        suppressed: Rule ids matched only in informational, historical or
            heavy-exertion context (kept for the audit trace).
    """

    action: ActionEnum | None
    matched_rules: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)
    bp: tuple[int, int] | None = None
    evidence_spans: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)


#: Generic one-word filler slots and their negation-guarded form: a filler never
#: absorbs "not"/"no"/"never", so "my speech is not slurred" or "I'm not having a
#: stroke" cannot match a symptom pattern that skips one or more words.
_GUARDED_FILLERS: tuple[tuple[str, str], ...] = (
    (r"(?:\w+\s+)", r"(?:(?!(?:not|no|never)\b)\w+\s+)"),
    (r"(?:\s+\w+)", r"(?:\s+(?!(?:not|no|never)\b)\w+)"),
    (r"(?:[\s,]+\w+)", r"(?:[\s,]+(?!(?:not|no|never)\b)\w+)"),
)


def _rx(*alternatives: str) -> re.Pattern[str]:
    """Compile alternatives into one pattern (text is already lower case).

    Generic filler slots are rewritten to their negation-guarded form (see
    :data:`_GUARDED_FILLERS`).
    """
    pattern = "|".join(f"(?:{alt})" for alt in alternatives)
    for generic, guarded in _GUARDED_FILLERS:
        pattern = pattern.replace(generic, guarded)
    return re.compile(pattern)


# ======================================================================
# 1. Normalisation
# ======================================================================

_CHAR_MAP = str.maketrans(
    {
        "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u2032": "'", "`": "'", "\u00b4": "'",
        "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2212": "-",
    }
)
_REPEAT_RE = re.compile(r"([a-z])\1{2,}")
_BP_ABBREV_RE = re.compile(r"\bb\s?\.\s?p\b\.?")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SPACE_RE = re.compile(r"[^\S\n]+")

#: Deterministic misspelling / contraction map applied token by token.
_TYPO_MAP: dict[str, str] = {
    # chest
    "chset": "chest", "chets": "chest", "chesst": "chest", "chect": "chest", "chst": "chest",
    "cheast": "chest", "chestt": "chest", "chesr": "chest", "chestpain": "chest pain",
    "chestpains": "chest pains",
    # breath / breathing
    "breathe": "breath", "breth": "breath", "brath": "breath", "breaht": "breath",
    "braeth": "breath", "breat": "breath", "breeth": "breath", "brethe": "breath",
    "breathin": "breathing", "brething": "breathing", "breating": "breathing",
    "breahting": "breathing", "breathng": "breathing", "shortnes": "shortness",
    # numbness / dizziness
    "numbnes": "numbness", "numbess": "numbness", "numness": "numbness",
    "numbneess": "numbness", "dizy": "dizzy", "dizzi": "dizzy", "dizzey": "dizzy",
    "dizi": "dizzy", "dizzie": "dizzy", "dizzyy": "dizzy", "diziness": "dizziness",
    "dizzyness": "dizziness", "dizzines": "dizziness", "dizzness": "dizziness",
    "dizyness": "dizziness", "dizzynes": "dizziness",
    # fainting / stroke / seizure
    "fanted": "fainted", "fainterd": "fainted", "fainetd": "fainted", "fiainted": "fainted",
    "faintd": "fainted", "fainded": "fainted", "feinted": "fainted", "fanting": "fainting",
    "fant": "faint", "fiant": "faint", "stroek": "stroke", "strok": "stroke",
    "storke": "stroke", "stoke": "stroke", "siezure": "seizure", "seazure": "seizure",
    "sezure": "seizure", "unconcious": "unconscious",
    # headache / pain / tight
    "headace": "headache", "hedache": "headache", "headach": "headache",
    "hedake": "headache", "headche": "headache", "haedache": "headache",
    "headake": "headache", "headahce": "headache", "headachee": "headache",
    "hedach": "headache", "pian": "pain", "painn": "pain", "tite": "tight",
    # pressure / blood pressure / bp
    "presure": "pressure", "pressur": "pressure", "preasure": "pressure",
    "pressue": "pressure", "pressre": "pressure", "prssure": "pressure",
    "preassure": "pressure", "pressuer": "pressure", "bloodpressure": "blood pressure",
    "sistolic": "systolic", "systolik": "systolic", "diastolik": "diastolic",
    "distolic": "diastolic", "diastollic": "diastolic",
    # contractions written without an apostrophe
    "cant": "can't", "couldnt": "couldn't", "dont": "don't", "doesnt": "doesn't",
    "didnt": "didn't", "havent": "haven't", "hasnt": "hasn't", "isnt": "isn't",
    "wasnt": "wasn't", "wont": "won't", "im": "i'm", "ive": "i've",
}

#: Red-flag keywords (length >= 6) eligible for edit-distance-1 correction.
_FUZZY_KEYWORDS: tuple[str, ...] = (
    "breath", "breathing", "breathless", "breathlessness", "shortness", "headache",
    "headaches", "numbness", "dizziness", "pressure", "seizure", "seizures", "unconscious",
    "unresponsive", "consciousness", "collapse", "collapsed", "thunderclap", "convulsion",
    "convulsions", "convulsing", "tightness", "squeezing", "heaviness", "discomfort",
    "palpitation", "palpitations", "suicide", "suicidal", "disoriented", "slurred",
    "slurring", "drooping", "fainting", "paralysed", "paralyzed", "paralysis", "weakness",
    "vision", "speech", "pregnant", "coughing", "sputum", "frothy", "blindness",
    "epigastric", "systolic", "diastolic",
)

#: Real English words one edit away from a keyword — never "corrected" (checked
#: against a 236k-word English dictionary at development time; every entry here
#: is an actual edit-distance-1 variant of a keyword above).
_FUZZY_STOPWORDS: frozenset[str] = frozenset(
    {
        "asystolic", "blandness", "blurred", "blurring", "breach", "breadth", "breadthless",
        "couching", "drooling", "dropping", "dumbness", "fizziness", "headiness", "lightness",
        "painting", "pressured", "pressures", "regnant", "rightness", "roughing", "scutum",
        "slurped", "slurping", "soughing", "spurred", "spurring", "tainting", "toughing",
        "trooping", "wreath", "wreathless",
    }
)


def _edit1_variants(word: str) -> set[str]:
    """Return every string at OSA distance exactly 1 from ``word`` (a–z alphabet)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {
        left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1
    }
    replaces = {left + c + right[1:] for left, right in splits if right for c in letters}
    inserts = {left + c + right for left, right in splits for c in letters}
    return (deletes | transposes | replaces | inserts) - {word}


def _build_fuzzy_index(keywords: tuple[str, ...], stopwords: frozenset[str]) -> dict[str, str]:
    """Pre-compute the variant → keyword table (first keyword wins on a tie)."""
    protected = set(keywords) | stopwords
    index: dict[str, str] = {}
    for keyword in keywords:
        for variant in _edit1_variants(keyword):
            if variant not in protected:
                index.setdefault(variant, keyword)
    return index


_FUZZY_INDEX: dict[str, str] = _build_fuzzy_index(_FUZZY_KEYWORDS, _FUZZY_STOPWORDS)


def _correct_token(match: re.Match[str]) -> str:
    token = match.group(0)
    mapped = _TYPO_MAP.get(token)
    if mapped is not None:
        return mapped
    if len(token) < 5:
        return token
    return _FUZZY_INDEX.get(token, token)


def normalize_text(text: str) -> str:
    """Normalise a raw message for deterministic rule matching.

    Args:
        text: Raw patient message.

    Returns:
        Lower-cased text with unified punctuation, collapsed letter runs,
        corrected misspellings and single spaces (newlines are kept because
        they are clause boundaries).
    """
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).lower()
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _BP_ABBREV_RE.sub("bp", text)
    text = _WORD_RE.sub(_correct_token, text)
    return _SPACE_RE.sub(" ", text).strip()


# ======================================================================
# 2. Clause analysis (negation, informational, historical, exertion)
# ======================================================================

_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.;?!\n]+(?!\d)|\b(?:but|however|although|though|whereas)\b"
)

_NEG_CUE_RE = _rx(
    r"\b(?:no longer|no|not|never|without|w/o|denies|deny|denied|denying|none|nor|neither"
    r"|free of|absence of|negative for|ruled out|rules out|rule out)\b",
    r"\b(?:do|does|did|have|has|had|is|are|was|were|could|would|should|ca|wo|ai)n't\s+"
    r"(?:(?:really|ever|even|actually|currently|now)\s+)?(?:have|has|had|having|get|got|getting"
    r"|feel|feeling|felt|experienc\w*|notic\w*|suffer\w*|want|think|faint\w*|pass\w*|black\w*"
    r"|collaps\w*|lose|lost|cough\w*|see|seem|hurt|hurts|ache|aches|been)\b",
)
_PSEUDO_NEG_RE = _rx(
    r"\b(?:no doubt|no idea|no clue|not sure|not certain|not quite sure|not only|not just"
    r"|no matter|no relief|no improvement|not improving|not better|no better|not going away"
    r"|not getting better|not stopping|no change|not helping|no help|not normal|not know"
    r"|not clear|no one|nobody)\b",
    r"n't know\b",
)
_NEG_VOID_RE = _rx(
    r"\b(?:like this|this bad|this severe|this strong|this intense|this painful|as bad as this"
    r"|before now|until now|until today|first time)\b"
)
_NEG_HARD_TERM_RE = _rx(
    r"\b(?:but|however|although|though|yet|except|apart from|other than|aside from|besides"
    r"|now|still|suddenly|today|tonight|currently|instead|which|who|whereas)\b",
    r"\b(?:i|we|you|my|his|her|their|it|it's|its|there's|he|she|he's|she's|they|they're|this"
    r"|that)\b",
)
_NEG_SOFT_TERM_RE = _rx(r",|&|\b(?:and|with|plus|also|then)\b")
_POST_NEG_INLINE_RE = re.compile(r"\s*[:=\-]+\s*(?:no|none|nope|negative|denied|absent|nil)\b")
_POST_NEG_CLAUSE_RE = re.compile(
    r"^\W*(?:no|nope|none|nah|negative|not really|not at all|never|denied|denies|absent|nil)\W*$"
)
_NEG_WINDOW_CHARS = 80

#: Educational question frames only — broad topical words ("exam", "in women")
#: are deliberately not cues, so they can never hide a personal statement.
_INFO_CUE_RE = _rx(
    r"\b(?:what|which)(?:'s|\s+(?:is|are|were|was))\s+(?:the\s+|some\s+)?(?:\w+\s+){0,2}?"
    r"(?:signs?|symptoms?|indicators?|red flags?|types?|causes?)\b",
    r"\b(?:what|which)\s+(?:\w+\s+){0,2}?(?:signs?|symptoms?)\s+(?:of|for|should|to|indicate"
    r"|mean|are)\b",
    r"\bhow\s+(?:do|can|would|could|should|will|does|did)\s+(?:i|you|we|one|someone|somebody"
    r"|people|a person|anyone)\s+(?:\w+\s+)?(?:recogni[sz]e|identify|spot|tell|detect|know"
    r"|notice)\b",
    r"\bhow to\s+(?:recogni[sz]e|identify|spot|tell|detect|know)\b",
    r"\b(?:tell|teach|explain|show|list|give)\s+(?:me\s+|us\s+)?(?:about\s+)?(?:the\s+|some\s+)?"
    r"(?:\w+\s+){0,2}?(?:signs?|symptoms?)\b",
    r"\b(?:signs|symptoms)\s+(?:and|&)\s+(?:signs|symptoms)\b",
    r"\b(?:if|when|in case)\s+my\s+(?:husband|wife|partner|mother|mom|mum|father|dad|child|son"
    r"|daughter|friend|parents?|grandmother|grandfather|grandma|grandpa|baby|brother|sister"
    r"|relative)\s+(?:ever\s+)?(?:gets|has|starts|develops|faints|collapse|collapses|passes"
    r"|goes|experiences|shows)\b",
    r"\b(?:if|when|in case)\s+(?:someone|somebody|a person|people|anyone|a patient|patients"
    r"|you)\s+(?:\w+\s+)?(?:is|are|'s|has|have|gets?|starts?|suddenly|experiences?|develops?"
    r"|shows?|faints?|collapses?|passes|goes)\b",
    r"^\s*what(?:'s| is| are| does)\s+(?:(?:a|an|the)\s+)?(?:\w+\s+){0,3}?\w+(?:\s+mean)?\s*$",
    r"\bif\s+i\s+(?:ever|were to)\b|\bin case i\b|\bin the future\b",
)
#: First-person / present-tense experience markers; any of them in the message
#: cancels informational suppression (the question may be about a live event).
_PERSONAL_RE = _rx(
    r"\bi(?:'m| am)\s+(?:\w+\s+)?(?:having|feeling|experiencing|getting)\b",
    r"\b(?:now|right now|currently|at the moment|as we speak|just now|today|tonight"
    r"|this (?:morning|afternoon|evening))\b",
    r"\b(?:my|his|her|their)\s+(?:[\w']+\s+)?(?:chest|head|face|arm|leg|heart|lips?|speech"
    r"|vision|eyes?|smile)\s+(?:is|are|was|feels?|felt|hurts?|went|keeps?|has|have|started"
    r"|looks?|seems?)\b",
    r"\b(?:i have|i had|i've got|i've|i feel|i felt|i got|i get)\s+(?:(?:a|an|some|this|that"
    r"|these|the|bad|severe|sudden)\s+)?(?:\w+\s+)?(?:pain|ache|aching|tight\w*|pressure"
    r"|numb\w*|weak\w*|droop\w*|slurr\w*|dizz\w*|faint\w*|headache|symptoms?|signs?)\b",
    r"\bi\s+(?:(?:just|suddenly|also|then|again|really|nearly|almost)\s+)?(?:fainted"
    r"|passed out|blacked out|collapsed|can't|cannot|couldn't|keep|woke|started|noticed)\b",
)
_HISTORICAL_RE = _rx(
    r"\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several"
    r"|few|many|couple of|some)\s+(?:years?|yrs?|months?|decades?)\s+(?:ago|back|before"
    r"|earlier)\b",
    r"\blast\s+(?:year|month|summer|winter|spring|autumn|fall|decade)\b|\bin\s+(?:19|20)\d\d\b",
    r"\bback in (?:(?:19|20)\d\d|the day|the (?:\d0s|eighties|nineties)|my|school|college"
    r"|high school)\b",
    r"\b(?:years ago|long (?:time )?ago|used to|previous(?:ly)?|prior|hx of|h/o)\b",
    r"\b(?:family |medical |past |previous |prior )?history of\b",
    r"\bas a (?:child|kid|teen(?:ager)?|baby|young (?:man|woman|adult))\b",
    r"\bwhen i was (?:younger|little|young|pregnant|a (?:child|kid|teen(?:ager)?|baby|student)"
    r"|in (?:school|college))\b",
    r"\bruns in (?:my|the|our) family\b|\bin my (?:last|previous|first) pregnancy\b",
    r"\bturned out (?:to be)?\b|\bit was (?:just |only |probably |actually )?(?:acid reflux"
    r"|reflux|heartburn|gas|indigestion|anxiety|a panic attack|panic|muscular|a pulled muscle"
    r"|muscle strain)\b",
    r"\b(?:well[- ]controlled|under control|seizure[- ]free)\b",
)
_RECENT_RE = _rx(
    r"\b(?:now|right now|currently|today|tonight|just|again|still|ongoing|at the moment|since"
    r"|keeps?|kept|started|starting|yesterday|last night|is back|came back|returned)\b",
    r"\bthis (?:morning|afternoon|evening|week)\b",
    r"\b(?:\d+|a|an|one|two|few|couple of|a few|several)\s+(?:minutes?|mins?|hours?|hrs?"
    r"|days?)\s+ago\b",
)
_RESOLVED_RE = _rx(
    r"\bnow\s+(?:it's\s+|it is\s+|i'm\s+|i am\s+|everything(?:'s| is)\s+|all\s+|things are\s+"
    r"|they're\s+|they are\s+)?(?:\w+\s+)?(?:fine|normal|ok|okay|better|good|controlled|resolved"
    r"|gone|stable)\b",
    r"\b(?:is|are|it's|i'm|am|all|everything's)\s+(?:\w+\s+)?(?:fine|normal|ok|okay|better|good"
    r"|controlled|resolved|gone|stable)\s+now\b",
)
_HEAVY_EXERTION_RE = _rx(
    r"\b(?:when|while|after|during|if|whenever|with)\s+(?:(?:i|i'm|doing|heavy|intense|hard"
    r"|vigorous|strenuous|go|going|play|playing)\s+)*(?:run|running|lift\w*|training)\b",
    r"\b(?:jogging|jog|(?<!breathing )exercis\w*|working out|work out|workouts?|gym|sports?"
    r"|sprint\w*|weightlifting|lifting weights|swimming|hiking|cycling|biking|marathons?"
    r"|cardio|hiit|football|soccer|basketball|tennis|badminton|cricket|dancing)\b",
)
_MINIMAL_EXERTION_RE = _rx(
    r"\b(?:at rest|resting|rest|sitting|lying|laying|lie down|lay down|sleep\w*|talking"
    r"|walking|walk|bathroom|toilet|getting dressed|dressing|shower\w*|even|minimal|light"
    r"|few steps|across the room|stairs|doing nothing|in bed|bed)\b"
)


class _Clause:
    """One clause of the normalised message; context flags are computed once, lazily."""

    def __init__(self, start: int, text: str, next_text: str) -> None:
        self.start = start
        self.text = text
        self.next_text = next_text

    @cached_property
    def negation_cues(self) -> list[tuple[int, int]]:
        """Valid negation cue spans (clause-relative); empty when voided."""
        if _NEG_VOID_RE.search(self.text):
            return []
        pseudo = [m.span() for m in _PSEUDO_NEG_RE.finditer(self.text)]
        return [
            m.span()
            for m in _NEG_CUE_RE.finditer(self.text)
            if not any(ps <= m.start() < pe for ps, pe in pseudo)
        ]

    @cached_property
    def post_negated(self) -> bool:
        """``chest pain? no`` — the following clause is a bare negative answer."""
        return bool(_POST_NEG_CLAUSE_RE.match(self.next_text))

    @cached_property
    def informational(self) -> bool:
        return bool(_INFO_CUE_RE.search(self.text))

    @cached_property
    def historical(self) -> bool:
        if not _HISTORICAL_RE.search(self.text):
            return False
        # "... in 2020, now it's well controlled": a resolution is not recency.
        return not _RECENT_RE.search(_RESOLVED_RE.sub(" ", self.text))

    @cached_property
    def heavy_exertion(self) -> bool:
        return bool(_HEAVY_EXERTION_RE.search(self.text)) and not _MINIMAL_EXERTION_RE.search(
            self.text
        )


class _Analysis:
    """Per-message clause segmentation and context helpers."""

    def __init__(self, text: str) -> None:
        self.text = text
        spans: list[tuple[int, int]] = []
        prev = 0
        for boundary in _CLAUSE_BOUNDARY_RE.finditer(text):
            spans.append((prev, boundary.start()))
            prev = boundary.end()
        spans.append((prev, len(text)))
        self._starts = [s for s, _ in spans]
        self._clauses = [
            _Clause(
                start=s,
                text=text[s:e],
                next_text=text[spans[i + 1][0]:spans[i + 1][1]] if i + 1 < len(spans) else "",
            )
            for i, (s, e) in enumerate(spans)
        ]
        self.personal = bool(_PERSONAL_RE.search(text))

    def clause_at(self, pos: int) -> _Clause:
        return self._clauses[bisect.bisect_right(self._starts, pos) - 1]

    def is_negated(self, start: int, end: int, rule_starts: list[int]) -> bool:
        """Whether the match ``[start, end)`` lies in the scope of a negation cue."""
        clause = self.clause_at(start)
        if clause.post_negated or _POST_NEG_INLINE_RE.match(self.text, end):
            return True
        rel = start - clause.start
        for cue_start, cue_end in clause.negation_cues:
            # The cue must begin before the match; a cue verb may overlap the
            # match ("don't want" / "want to die").
            if cue_start >= rel or rel - cue_end > _NEG_WINDOW_CHARS:
                continue
            segment = clause.text[cue_end:rel]
            if _NEG_HARD_TERM_RE.search(segment):
                continue
            soft = _NEG_SOFT_TERM_RE.search(segment)
            if soft is None:
                return True
            # Lists stay negated only when the cue's first object is a red flag.
            first_soft = clause.start + cue_end + soft.start()
            if any(clause.start + cue_end <= s < first_soft for s in rule_starts):
                return True
        return False

    def status(self, start: int, end: int, rule_starts: list[int], *, exertion: bool) -> str:
        """Classify one rule match as affirmed / negated / suppressed."""
        if self.is_negated(start, end, rule_starts):
            return "negated"
        clause = self.clause_at(start)
        if clause.historical:
            return "suppressed"
        if clause.informational and not self.personal:
            return "suppressed"
        if exertion and clause.heavy_exertion:
            return "suppressed"
        return "affirmed"


# ======================================================================
# 3. Blood-pressure extraction
# ======================================================================

_NUM_GUARD = r"(?!\d|\.\d|\s*[/\\]\s*\d|\s+(?:over|by)\s+\d)"
_UNIT_GUARD = (
    r"(?!\s*(?:%|percent|mg|mcg|ug|ml|mmol|kg|kgs|lbs?|pounds|units?|iu|bpm|beats|steps"
    r"|calories|cal|kcal|days?|weeks?|months?|years?|yrs?|hours?|hrs?|minutes?|mins?|times"
    r"|rs|rupees|dollars|pills?|tablets?|tabs?|cm|mm(?!\s*hg)|meters?|metres?|km|miles?|ft"
    r"|feet|degrees?)\b)"
)
_CMP = (
    r"(?:(?P<strict>over|above|more than|greater than|higher than|past|beyond|crossed"
    r"|exceeding|exceeds|>)\s*|(?:at least|>=|in the)\s+)?"
)
_SUFFIX = r"(?:\s*(?:\+|plus\b|something\b|ish\b|-ish\b|odd\b|s\b|'s\b))?"
_FILLER = (
    r"(?:\s+(?:is|was|are|were|'s|of|at|reads?|reading|showed|shows?|says|said|came|comes"
    r"|out|now|still|currently|today|tonight|again|just|already|up|reaching|around|about"
    r"|approximately|approx|roughly|nearly|almost|like|maybe|probably|went|gone|going|has|had"
    r"|have|been|got|gets|getting|keeps?|kept|stays?|stayed|hit|hits|reached|touched"
    r"|touching|spiked|jumped|shot|rose|climbed|dropped|fell|down|measured|measures|showing"
    r"|registered|it|it's|me|my|only|consistently|constantly|always|very|really|so|high|too"
    r"|to)){0,5}"
)
_SINGLE_TAIL = r"\s*[:=-]?\s*" + _CMP + r"(?P<num>\d{2,3})" + _NUM_GUARD + _SUFFIX + _UNIT_GUARD

_BP_PAIR_RE = re.compile(
    r"(?<![\d/])(?<!\d\.)(?P<sys>\d{2,3})\s*(?:/|\\|\bover\b|\bby\b)\s*(?P<dia>\d{2,3})"
    r"(?!\d|\.\d)" + _UNIT_GUARD
)
_SYS_SINGLE_RE = re.compile(
    r"\b(?:systolic(?:\s+(?:bp|blood pressure|pressure|reading|number|value|level))?"
    r"|(?:top|upper|first)\s+(?:number|reading|value))" + _FILLER + _SINGLE_TAIL
)
_DIA_SINGLE_RE = re.compile(
    r"\b(?:diastolic(?:\s+(?:bp|blood pressure|pressure|reading|number|value|level))?"
    r"|(?:bottom|lower|second)\s+(?:number|reading|value))" + _FILLER + _SINGLE_TAIL
)
_BP_SINGLE_RE = re.compile(
    r"\b(?:bp|blood pressure|pressure|cuff)(?:\s+(?:reading|readings|monitor|machine|cuff"
    r"|level|levels|numbers?))?" + _FILLER + _SINGLE_TAIL
)
_CTX_SINGLE_RE = re.compile(
    r"\b(?:it|it's|its|now|numbers?|readings?|cuff|monitor|machine|to|hit|hits|reached"
    r"|reaching|touched|crossed)"
    r"(?:\s+(?:is|was|'s|are|were|reads?|said|says|showed|shows|came|out|at|now|still|like"
    r"|about|around|up|to)){0,3}\s*[:=-]?\s*" + _CMP + r"(?P<num>\d{3})" + _NUM_GUARD + _SUFFIX
    + _UNIT_GUARD
)
_BP_CONTEXT_RE = re.compile(
    r"\b(?:bp|blood pressure|pressure|systolic|diastolic|mm ?hg|hypertension|hypertensive"
    r"|cuff|readings?|monitor|machine|top number|bottom number|numbers|measured|measuring"
    r"|checked|checking|rechecked)\b"
)
_MEASUREMENT_LABEL_RE = re.compile(
    r"\b(?:(?P<bp>bp|blood pressure|pressure|systolic|diastolic|cuff)|sugar|glucose|cholesterol"
    r"|ldl|hdl|triglycerides?|a1c|hba1c|heart rate|pulse|hr|weight|weigh|temp|temperature"
    r"|fever|oxygen|o2|spo2|sats?|saturation|potassium|sodium|creatinine|egfr|platelets?"
    r"|h[a]?emoglobin|uric acid|tsh|vitamin|calories|steps|score|iq)\b"
)
_TARGET_RE = re.compile(
    r"(?:under|below|less than|lower than|<|target|goal|aim(?:ing)? for|keep (?:it|my bp|bp)?)"
    r"\s*(?:of|is|at|to)?\s*$"
)
#: "if / unless / in case / whenever ..." — a conditional threshold, not a measurement
#: ("call if BP is over 180/110"). "when" is excluded: it is often temporal.
_CONDITIONAL_RE = re.compile(r"\b(?:if|unless|in case|whenever)\b[^.;?!\n]{0,30}$")
_THRESHOLD_TAIL_RE = re.compile(
    r"(?:over|above|higher than|more than|greater than|exceeds|exceeding|>=?|crosses|past"
    r"|beyond)\s*$"
)


@dataclass(frozen=True)
class _Reading:
    """One plausible BP reading; a component that was not stated is ``0``."""

    systolic: int
    diastolic: int
    start: int
    end: int
    labelled: bool


def _other_measurement(before: str) -> bool:
    """Whether the nearest measurement label before a number is not a BP label."""
    labels = list(_MEASUREMENT_LABEL_RE.finditer(before))
    return bool(labels) and labels[-1].group("bp") is None


def _plausible(sys_value: int, dia_value: int) -> bool:
    return 50 <= sys_value <= 300 and 20 <= dia_value <= 200 and sys_value > dia_value


def _not_a_reading(before: str, personal: bool) -> bool:
    """Target ("below 130/80") or conditional threshold ("if it is over 180/110").

    A conditional threshold still counts as a reading when the message carries a
    present-tense / first-person marker ("what if it's over 200/120 right now?").
    """
    if _TARGET_RE.search(before):
        return True
    return (
        not personal
        and bool(_CONDITIONAL_RE.search(before))
        and bool(_THRESHOLD_TAIL_RE.search(before))
    )


def _single_value(match: re.Match[str]) -> int:
    value = int(match.group("num"))
    return value + 1 if match.group("strict") else value


def _extract_readings(text: str, has_context: bool, personal: bool) -> list[_Reading]:
    """Extract plausible BP readings from normalised text, most specific format first."""
    readings: list[_Reading] = []
    claimed: set[int] = set()
    for m in _BP_PAIR_RE.finditer(text):
        sys_value, dia_value = int(m.group("sys")), int(m.group("dia"))
        if has_context and 40 <= sys_value < dia_value:
            # "bp 120/190": most likely typed in the wrong order — read it swapped.
            sys_value, dia_value = dia_value, sys_value
        before = text[max(0, m.start() - 40):m.start()]
        if (
            _plausible(sys_value, dia_value)
            and not _other_measurement(before)
            and not _not_a_reading(before, personal)
        ):
            readings.append(_Reading(sys_value, dia_value, m.start(), m.end(), False))
            claimed.update((m.start("sys"), m.start("dia")))
    singles = (
        (_DIA_SINGLE_RE, "dia", True),
        (_SYS_SINGLE_RE, "sys", True),
        (_BP_SINGLE_RE, "sys", True),
        (_CTX_SINGLE_RE, "sys", has_context),
    )
    for pattern, component, enabled in singles:
        if not enabled:
            continue
        for m in pattern.finditer(text):
            if m.start("num") in claimed:
                continue
            before = text[max(0, m.start("num") - 40):m.start("num")]
            if _other_measurement(before) or _not_a_reading(before, personal):
                continue
            value = _single_value(m)
            if not (50 <= value <= 300 if component == "sys" else 20 <= value <= 200):
                continue
            claimed.add(m.start("num"))
            if component == "sys":
                readings.append(_Reading(value, 0, m.start(), m.end(), True))
            else:
                readings.append(_Reading(0, value, m.start(), m.end(), True))
    readings.sort(key=lambda r: r.start)
    return readings


def _is_crisis(reading: _Reading, has_context: bool) -> bool:
    high_sys = reading.systolic >= SYSTOLIC_CRISIS
    high_dia = reading.diastolic >= DIASTOLIC_CRISIS
    if reading.labelled or has_context:
        return high_sys or high_dia
    return high_sys and high_dia


def _report_bp(
    reading: _Reading, readings: list[_Reading], analysis: _Analysis
) -> tuple[int, int]:
    """Fill a missing component from a single-value reading in the same clause."""
    systolic, diastolic = reading.systolic, reading.diastolic
    clause = analysis.clause_at(reading.start)
    for other in readings:
        if analysis.clause_at(other.start) is not clause:
            continue
        if systolic == 0 and other.systolic:
            systolic = other.systolic
        if diastolic == 0 and other.diastolic:
            diastolic = other.diastolic
    return systolic, diastolic


# ======================================================================
# 4. Red-flag rules
# ======================================================================

_CANT = (
    r"(?:can't|cannot|can not|couldn't|could not|unable to|not able to|inability to|hard to"
    r"|difficult to|struggling to|struggle to|trouble|difficulty|difficulties|problems?"
    r"|(?:a )?hard time|labou?red)"
)
_CHEST = r"(?:chest|sternum|breastbone|breast bone)"
_DET = r"(?:(?:my|the|his|her|their|your)\s+)"


def _self_report(conditions: str) -> tuple[str, ...]:
    """Patterns for 'I'm having a <condition>' style self/bystander reports."""
    subject = (
        r"(?:i'm|i am|i think i'm|i think i am|i might be|i may be|i could be|am i"
        r"|is (?:he|she|my \w+)|(?:he|she)'s|(?:he|she|my \w+) (?:is|might be|may be))"
    )
    cond = rf"(?:(?:a|an)\s+)?(?:\w+\s+)?(?:{conditions})\b"
    return (
        rf"\b{subject}\s+(?:\w+\s+)?(?:having|getting|suffering)\s+{cond}",
        rf"\bi (?:think|believe|fear|guess) i (?:just |may have |might have )?(?:had|have had)"
        rf"\s+{cond}",
        rf"\bi (?:just|may have just|might have just|might have|may have) had\s+{cond}",
        rf"\b(?:feels?|felt|feeling) like (?:i'm having |i am having |i'm getting )?{cond}",
        rf"\b(?:i have|i've got|i'm having|i am having|having) (?:all )?(?:the )?"
        rf"(?:symptoms|signs) of\s+{cond}",
    )


#: (rule id, pattern, breathlessness-style exertion filter) in reporting order.
_RED_FLAG_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "chest_pain",
        _rx(
            rf"\b{_CHEST}(?:'s|\s+(?:is|was|feels?|felt|has been|had been|keeps?|kept|started"
            r"|starts|getting|gets|got|became|becomes|going|went|are|area|wall))?(?:\s+(?:really"
            r"|very|so|super|kind of|kinda|sort of|a bit|a little|bit|little|extremely|quite"
            r"|pretty|too|all|like|suddenly|now|still|just|more|getting|being))*\s+(?:pains?"
            r"|painful|aches?|aching|achy|hurts?|hurting|tight|tightness|tightening"
            r"|tighten(?:s|ed)?"
            r"|heavy|heaviness|pressure|pressing|squeez\w*|crush\w*|burn\w*|discomfort"
            r"|constrict\w*|compress\w*|sharp|stabbing|stab\w*|pinch\w*|clench\w*|clamp\w*"
            r"|killing me)\b",
            rf"\b{_CHEST} (?:feels?|felt) like (?:\w+\s+){{0,5}}?(?:crush\w*|squeez\w*|pressure"
            r"|elephant|weight|tight\w*|vice|vise|explod\w*|burst\w*|on fire|burning|heart attack"
            r"|being (?:crushed|squeezed|stabbed))",
            r"\b(?:pains?|aches?|aching|tightness|tight (?:feeling|sensation)|pressure|heaviness"
            r"|heavy (?:feeling|sensation|weight)|weight|squeez\w*|crush\w*|burn\w*|discomfort"
            r"|constriction|compression|pressing|stabbing|elephant|band|vice|vise"
            r"|ton of bricks)\b(?:\s+\w+){0,4}?\s+(?:in|on|across|around|over|behind|at|of"
            rf"|through|inside|within|under|into|to)\s+{_DET}?(?:\w+\s+){{0,3}}?{_CHEST}\b",
            r"\bpain\s+(?:in|around)\s+my\s+heart\b",
            r"\b(?:someone|somebody|something|an? \w+)\s+(?:is\s+)?(?:sitting|standing|pressing"
            rf"|pushing|squeezing)\s+(?:on|down on)\s+{_DET}?{_CHEST}\b",
        ),
        False,
    ),
    (
        "heart_attack",
        _rx(
            *_self_report(r"heart attack|cardiac arrest"),
            r"\b(?:sense|feeling) of (?:impending )?doom\b|\bimpending doom\b",
        ),
        False,
    ),
    (
        "aortic_dissection",
        _rx(
            r"\b(?:tearing|ripping)(?:[\s,]+\w+){0,4}?[\s,]+(?:pain|ache|sensation|feeling)\b",
            r"\b(?:pain|ache)(?:[\s,]+\w+){0,4}?[\s,]+(?:tearing|ripping)\b",
            r"\b(?:tearing|ripping)(?:[\s,]+\w+){0,3}?[\s,]+(?:chest|back|shoulder blades?)\b",
        ),
        False,
    ),
    (
        "breathlessness",
        _rx(
            r"\bshort(?:ness)?\s+(?:of\s+)?breath\w*",
            r"\bbreathless\w*|\bout of breath\b|\bair hunger\b",
            rf"\b{_CANT}\s+(?:\w+\s+)?(?:breath(?:e|ing)?|catch(?:ing)? (?:my |his |her )?breath"
            r"|get(?:ting)? (?:enough |any )?(?:air|breath|oxygen))\b"
            r"(?! through (?:my|the|his|her) nose)",
            r"\bgasp(?:ing|ed|s)?\b|\b(?:struggling|fighting) for (?:air|breath)\b",
            r"\b(?:stopped|not|isn't|is not|wasn't|was not|has stopped|aren't|are not) breathing\b",
            r"\bbreathing (?:is |has (?:become|gotten|got) |feels |became )?(?:very |really |so "
            r"|extremely )?(?:hard|difficult|labou?red|laboring|labouring)\b",
            r"\b(?:can't|cannot|unable to|not able to) lie (?:down )?flat\b",
        ),
        True,
    ),
    (
        "stroke_face",
        _rx(
            r"\b(?:face|facial)(?:'s|\s+(?:is|was|feels?|felt|has|had|looks?|looked|went"
            r"|has gone|gone|seems?|started|starts|became|getting|keeps?))?(?:\s+(?:really|very"
            r"|so|a bit|a little|kind of|kinda|suddenly|all|now|totally|completely|partly"
            r"|partially|slightly|been))*\s+(?:droop\w*|drop(?:ping|ped|s)?|sagg\w*|numb\w*"
            r"|tingl\w*|paraly[sz]\w*|weak\w*|lopsided|uneven|crooked|twisted|frozen|slack"
            r"|fallen|falling|limp|asymmetric\w*)\b",
            r"\b(?:one|left|right|half|other)\s+(?:side\s+of\s+)?(?:(?:my|his|her|the)\s+)?"
            r"face\b(?:\s+\w+){0,4}?\s+(?:droop\w*|drop(?:ping|ped|s)?|sagg\w*|numb\w*|tingl\w*"
            r"|paraly[sz]\w*|weak\w*|lopsided|uneven|crooked|twisted|frozen|dead|slack|fallen"
            r"|falling|limp|heavy|weird|funny|strange|odd|off)\b",
            r"\b(?:smile|mouth|lips?)\s+(?:(?:is|are|was|looks?|looked|feels?|seems?|has|went|got)"
            r"\s+)?(?:\w+\s+)?(?:uneven|crooked|lopsided|droop\w*|twisted|sagg\w*"
            r"|drop(?:ping|ped)|slant\w*|one[- ]sided|asymmetric\w*)\b",
            r"\b(?:uneven|crooked|lopsided|droopy|one[- ]sided)\s+(?:smile|mouth)\b",
        ),
        False,
    ),
    (
        "stroke_arm",
        _rx(
            rf"\b{_CANT}\s+(?:\w+\s+)?(?:move|lift|raise|feel|use|hold up|grip with|control|bend)"
            r"\s+(?:(?:my|his|her|the)\s+)?(?:(?:left|right|one)\s+)?(?:arm|leg|hand|side"
            r"|fingers)\b",
            r"\b(?:(?:my|his|her|left|right|one)\s+)?(?:(?:left|right)\s+)?(?:arm|leg|hand)\s+"
            r"(?:(?:is|was|feels?|felt|went|has gone|has been|became|suddenly|just|keeps?"
            r"|seems?)\s+)?(?:(?:really|very|so|suddenly|all|completely|totally|kind of|a bit"
            r"|a little|now|gone)\s+)*(?:weak\w*|numb\w*|dead|limp|paraly[sz]\w*|useless|floppy"
            r"|clumsy|clumsiness|tingly|tingling|won't move|not moving|giving way|gave way)\b",
            r"\barm\s+(?:is |feels? |felt )?(?:\w+\s+)?heavy\b",
            r"\b(?:weak\w*|numb\w*|paraly[sz]\w*|limp|clumsy|clumsiness)\s+(?:in|of|on|down)\s+"
            r"(?:(?:my|his|her|the|one)\s+)?(?:(?:left|right)\s+)?(?:arm|leg|hand|side)\b",
            r"\b(?:numb\w*|weak\w*|paraly[sz]\w*|tingl\w*|dead)\s+(?:\w+\s+){0,2}?(?:on|down|along"
            r"|in|of)\s+(?:the\s+|my\s+|one\s+)?(?:left|right|one|whole|entire)\s+(?:\w+\s+)?"
            r"side\b",
            r"\b(?:one[- ]sided|sudden|unilateral)\s+(?:weakness|numbness|paralysis)\b",
            r"\b(?:left|right|one|whole|entire)\s+side\s+(?:of\s+(?:my|his|her|the)\s+)?"
            r"(?:body\s+)?"
            r"(?:(?:is|feels?|felt|went|has gone|has been|became|keeps|suddenly)\s+)?(?:\w+\s+)?"
            r"(?:numb\w*|weak\w*|paraly[sz]\w*|tingl\w*|dead|limp|heavy)\b",
        ),
        False,
    ),
    (
        "stroke_numbness",
        _rx(r"\bnumb(?:ness)?(?:[\s,]+\w+){0,6}?[\s,]+(?:face|arms?|legs?|side|body)\b"),
        False,
    ),
    (
        "stroke_speech",
        _rx(
            r"\bslurr?(?:ed|ing|s)?\s+(?:\w+\s+)?(?:speech|words?|talking)\b|\baphasi\w*",
            r"\b(?:speech|words|talking)\s+(?:(?:is|are|was|were|sounds?|sounded|seems?"
            r"|has been|have been|became|getting|came out|comes out|coming out|keep coming out)"
            r"\s+){0,2}(?:\w+\s+)?"
            r"(?:slurr?(?:ed|ing|y)|garbled|jumbled|mixed up|weird|strange|funny|wrong"
            r"|not making sense|muddled|incoherent|nonsense|gibberish)\b",
            rf"\b{_CANT}\s+(?:\w+\s+)?(?:talk(?:ing)?|speak(?:ing)?|get(?:ting)? (?:my |the )?"
            r"words out|find(?:ing)? (?:my |the )?words|form(?:ing)? words|say(?:ing)? words)\b"
            r"(?!\s+(?:to|with|about|for|right now|now|on the phone|much|long|at the moment"
            r"|today|english|hindi)\b)",
            rf"\b{_CANT}\s+(?:\w+\s+)?understand(?:ing)?\s+(?:what\s+(?:\w+\s+){{1,3}}?(?:is|are)"
            r"\s+saying|people|anyone|speech|spoken words|words)\b",
        ),
        False,
    ),
    (
        "stroke_vision",
        _rx(
            r"\bvision\s+(?:(?:is|has|has been|getting|went|is going|has gone)\s+)?(?:loss|lost"
            r"|chang\w*|blurr\w*|blur|blurry|darken\w*|black\w*|dim\w*|cut out|cuts out|gone"
            r"|going)\b",
            r"\b(?:lost|losing|loss of|lose)\s+(?:(?:my|the|his|her|all|some|part of my)\s+)?"
            r"(?:vision|sight|eyesight)\b",
            r"\b(?:blind\w*|can't see|cannot see|can not see|couldn't see|unable to see"
            r"|dark(?:ness)?"
            r"|black(?:ness)?|blurr\w*|blurry)\s+(?:\w+\s+){0,2}?(?:in|out of|from|with)\s+"
            r"(?:one|my|the|his|her|an?)\s+(?:(?:left|right|one)\s+)?eye\b",
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:blind\w*|blurr\w*|blurry|double vision"
            r"|vision|can't see|cannot see)",
            r"\bdouble vision\b|\bseeing double\b|\bvisual darkness\b|\bblack spots\b",
            r"\b(?:spots|blindness|flash\w*|curtain|shadow)\s+(?:\w+\s+){0,3}?(?:in|over|across"
            r"|down)\s+(?:my\s+|the\s+|one\s+)?(?:vision|eyes?|sight|visual field)\b",
            r"\bseeing (?:black |dark |bright |white )?(?:spots|flashes|flashing lights)\b",
        ),
        False,
    ),
    (
        "stroke_balance",
        _rx(
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:loss of balance|lost (?:my )?balance"
            r"|off balance|unsteady|unsteadiness|dizzy|dizziness|vertigo|trouble walking"
            r"|difficulty walking|can't walk|cannot walk|unable to walk|lost coordination"
            r"|loss of coordination|clumsy|clumsiness)\b",
            r"\binability to (?:walk|stand)\b",
            r"\b(?:can't|cannot|can not|couldn't|could not|unable to|not able to)\s+(?:walk|stand)"
            r"(?:\s+up)?\s+(?:straight|properly|steadily|steady|without (?:falling|help|holding)"
            r"|at all)\b",
            r"\bkeeps? (?:falling over|losing (?:my |his |her )?balance)\b",
        ),
        False,
    ),
    (
        "stroke_confusion",
        _rx(
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:confus\w*|disorient\w*|can't think"
            r" (?:straight|clearly)|memory loss)\b",
            r"\b(?:completely|totally|very|extremely|really)\s+(?:confused|disoriented)\b"
            r"(?!\s+(?:about|by|with|as to|on|over|why|what|which|how|whether|if|regarding)\b)",
            r"\bdisorient\w*\b",
            r"\b(?:don't|do not|doesn't|does not|didn't) (?:know|recogni[sz]e) (?:where (?:i am"
            r"|he is|she is|they are)|who (?:i am|he is|she is)|me|us|his family|her family"
            r"|anyone)\b",
        ),
        False,
    ),
    (
        "stroke_self_report",
        _rx(*_self_report(r"stroke|mini[- ]?stroke|tia")),
        False,
    ),
    (
        "severe_headache",
        _rx(
            r"\bworst\s+(?:\w+\s+){0,2}?(?:headache|head\s*ache|head pain|pain)\s+(?:of|in)\s+"
            r"my\s+(?:entire\s+|whole\s+)?life\b",
            r"\bworst\s+headache\s+(?:i(?:'ve| have)?\s+(?:ever\s+)?(?:had|felt|experienced)"
            r"|ever)\b",
            r"\bthunder\s*clap\w*",
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:headache|head\s*ache|head pain)\b",
            r"\b(?:headache|head\s*ache|head pain|pain in my head)\s+(?:\w+\s+){0,3}?(?:came on"
            r"|started|began|hit|struck|came)\s+(?:\w+\s+){0,2}?(?:suddenly|out of nowhere"
            r"|all of a sudden|in seconds|instantly)\b",
            r"\b(?:blinding|excruciating|explosive|exploding|unbearable|agonizing|agonising)\s+"
            r"(?:\w+\s+)?(?:headache|head\s*ache|head pain)\b",
            r"\bhead\s+(?:feels?\s+like\s+(?:it(?:'s| is)\s+)?|is\s+)(?:going to\s+)?"
            r"(?:explod\w*|burst\w*|split open)\b",
        ),
        False,
    ),
    (
        "syncope",
        _rx(
            r"\bfaint(?:ed|ing|s)\b|\bfaint(?:ed)?\s+(?:spell|episode)s?\b",
            r"\b(?:feel|feeling|felt|going to|gonna|about to|almost|nearly|might|may)\s+"
            r"(?:\w+\s+)?faint\b",
            r"\b(?:i|he|she|they|you|we)\s+(?:\w+\s+){0,2}?(?<!\ba )(?<!\bthe )faint\b",
            r"\bpass(?:ed|es|ing)?\s+out\b|\bblack(?:ed|ing|s)?\s+out\b|\bblackouts?\b",
            r"\bcollaps(?:e|ed|es|ing)\b|\bkeel(?:ed|s)? over\b|\bsyncop\w*",
            r"\b(?:lost|lose|losing|loss of)\s+consciousness\b|\bunconscious\b",
            r"\b(?:unresponsive|(?:he|she|they|he's|she's|they're|is|are)\s+(?:\w+\s+)?"
            r"not responding)\b(?!\s+to\s+(?:\w+\s+)?(?:treatment|meds|medications?|medicines?"
            r"|therapy|drugs?|pills?|tablets?))",
            r"\bwon't wake up\b|\b(?:can't|cannot|couldn't) wake (?:him|her|them|my \w+)\b",
        ),
        False,
    ),
    (
        "seizure",
        _rx(
            r"(?<!anti-)\bseizures?\b(?!\s+(?:medications?|medicines?|meds?|drugs?|pills?"
            r"|tablets?|disorder|threshold|risk|history))",
            r"\bseizing\b|\bconvuls\w*|\bhaving (?:a )?fit\b|\bepileptic (?:fit|attack)\b",
        ),
        False,
    ),
    (
        "hemoptysis",
        _rx(
            r"\bcough\w*\s+(?:up\s+)?(?:some\s+|a lot of\s+|lots of\s+)?(?:blood|bloody|pink"
            r"|frothy)\w*",
            r"\bpink\s+(?:\w+\s+)?(?:frothy|foamy|bubbl\w*|froth|foam|sputum|phlegm|spit|mucus)\b",
            r"\b(?:frothy|foamy)\s+(?:pink\s+)?(?:sputum|phlegm|spit|mucus|saliva)\b",
            r"\bblood(?:y)?\s+(?:in\s+(?:my\s+)?)?(?:sputum|phlegm|spit|mucus)\b",
            r"\bspit(?:ting)?\s+(?:up\s+)?blood\b|\bh(?:a)?emoptysis\b",
        ),
        False,
    ),
    (
        "cyanosis",
        _rx(
            r"\b(?:lips?|face|fingers?|fingertips|fingernails|nails|skin|tongue|mouth)\s+"
            r"(?:(?:are|is|look|looks|looked|turning|turned|went|going|gone|have gone|has gone"
            r"|getting|became)\s+)*(?:(?:a bit|slightly|very|really)\s+)?(?:blue|bluish|blueish"
            r"|purple|grey|gray|ashen)\b",
            r"\b(?:blue|bluish|blueish|purple)\s+(?:lips|face|fingers|fingertips|tongue)\b"
            r"|\bcyano\w*",
        ),
        False,
    ),
    (
        "angioedema",
        _rx(
            r"\b(?:tongue|throat|lips?)\s+(?:(?:is|are|has|have|feels?|started|keeps|getting|got)"
            r"\s+)?(?:\w+\s+)?(?:swell\w*|swollen|closing|puff\w*)\b",
            r"\bswell\w*\s+(?:of|in|on)\s+(?:my\s+|the\s+)?(?:tongue|throat|lips?)\b",
            r"\bswollen\s+(?:tongue|lips|throat)\b|\bthroat\s+(?:is\s+)?(?:closing|tightening)\b",
        ),
        False,
    ),
    (
        "severe_bleeding",
        _rx(
            r"\b(?:bleeding|blood|nosebleed|nose bleed)\s+(?:\w+\s+){0,4}?(?:won't|will not|wont"
            r"|doesn't|does not|didn't|can't|isn't|is not)\s+(?:stop|stopping)\b",
            r"\b(?:can't|cannot|couldn't|unable to)\s+stop\s+(?:the\s+)?bleed\w*",
            r"\b(?:profuse\w*|uncontroll\w*|massive)\s+bleed\w*|\bbleed\w*\s+(?:very\s+|really\s+)?"
            r"(?:heavily|profusely|everywhere)\b",
            r"\b(?:vomit\w*|throw\w* up|threw up|puk\w*)\s+(?:up\s+)?blood\b",
            r"\bblood\s+(?:is\s+)?(?:pouring|gushing|spurting|streaming)\b",
        ),
        False,
    ),
    (
        "blood_in_urine_or_stool",
        _rx(
            r"\bblood(?:y)?\s+(?:in\s+(?:my\s+|the\s+|his\s+|her\s+)?)?(?:urine|stools?|pee|poop"
            r"|poo|wee)\b",
            r"\b(?:peeing|pissing|passing|urinating)\s+blood\b|\bblack,?\s+tarry\s+stools?\b",
            r"\b(?:red|bloody|pink|cola)[- ]colou?red\s+urine\b",
        ),
        False,
    ),
    (
        "self_harm",
        _rx(
            r"\bsuicid\w*|\bself[- ]?harm\w*|\bkill(?:ing)?\s+my\s*self\b",
            r"\bend(?:ing)?\s+(?:my\s+(?:own\s+)?life|it all)\b|\btake\s+my\s+(?:own\s+)?life\b",
            r"\b(?:want|wanna|wish|wishing|plan(?:ning)?|thinking (?:about|of))\s+"
            r"(?:to\s+)?(?:die|be dead|not be here|overdose)\b",
            r"\bbetter off dead\b|\bno (?:reason|point) (?:to|in) (?:live|living|going on)\b",
            r"\b(?:hurt|harm|cut|cutting|hurting|harming)\s+myself\b",
            r"\bdon't want to (?:live|be alive|wake up|be here)\b|\bwish i (?:was|were) dead\b",
            r"\bwish i (?:wasn't|weren't|was not|were not) (?:here|alive|around)\b",
        ),
        False,
    ),
    (
        "hypertensive_crisis_report",
        _rx(
            r"\b(?:having|in|going through|experiencing)\s+(?:a\s+)?hypertensive\s+"
            r"(?:crisis|emergency|urgency)\b"
        ),
        False,
    ),
)

#: Helper patterns that only escalate in combination (composite rules).
_PALPITATIONS_RE = _rx(
    r"\bpalpitations?\b|\bracing heart\b",
    r"\b(?:heart|heartbeat|pulse)\s+(?:(?:is|was|keeps|started)\s+)?(?:\w+\s+)?(?:racing"
    r"|pounding|fluttering|flutters|skipping|irregular|beating (?:really |very |so )?(?:fast"
    r"|hard|irregularly|out of rhythm))\b",
)
_PREGNANCY_RE = _rx(
    r"\bpregnant\b(?<!get pregnant)(?<!become pregnant)(?<!getting pregnant)"
    r"(?<!becoming pregnant)",
    r"\b(?:my|this|current)\s+pregnancy\b|\b\d{1,2}\s*(?:weeks?|wks?)\s+(?:along|gestation)\b",
    r"\bpost[- ]?partum\b|\b(?:just|recently)\s+(?:gave birth|had (?:a|my) baby|delivered)\b",
    r"\b(?:first|second|third|1st|2nd|3rd)\s+trimester\b",
)
_PREECLAMPSIA_SYMPTOM_RE = _rx(
    r"\b(?:severe|bad|terrible|awful|horrible|pounding|throbbing|really bad|very bad"
    r"|persistent|constant|intense|worst|blinding|splitting)\s+(?:\w+\s+)?(?:headache"
    r"|head\s*ache|head pain)\b",
    r"\bheadache\s+(?:that\s+)?(?:won't|will not|doesn't|does not|wont)\s+go away\b",
    r"\b(?:blurr\w*|blurry|double|changes in (?:my )?|change in (?:my )?|loss of|dim(?:med)?"
    r"|spotty)\s*vision\b|\bvision\s+(?:\w+\s+)?(?:chang\w*|blurr\w*|blurry|problems?"
    r"|disturb\w*|loss|spots|dim\w*)\b",
    r"\bseeing (?:spots|stars|flash\w*)|\bspots in (?:my )?(?:vision|eyes)\b|\bflashing lights\b",
    r"\b(?:sudden|rapid|severe|a lot of)\s+(?:\w+\s+)?swelling\b",
    r"\bswelling\s+(?:\w+\s+)?(?:in|of)\s+(?:my\s+)?(?:face|hands|eyes|eyelids|fingers)\b",
    r"\b(?:face|hands|eyes|eyelids|fingers)\s+(?:(?:are|is|got|have|has|became|went)\s+)?"
    r"(?:(?:suddenly|really|very)\s+)?(?:swollen|puffy|swelling|swelled)\b",
    r"\bupper\s+(?:abdominal|belly|stomach|tummy)\s+pain\b|\bepigastric\b"
    r"|\bpain\s+(?:under|below)\s+(?:my\s+)?(?:right\s+)?ribs?\b",
)
_LOW_BP_SYMPTOM_RE = _rx(
    r"\b(?:dizz\w*|light[- ]?headed\w*|faint\w*|woozy|weak\w*|confus\w*|clammy|cold sweats?"
    r"|drows\w*|pale)\b"
)

#: Every rule id this module can report, in reporting order.
RULE_IDS: tuple[str, ...] = (
    "bp_crisis",
    "bp_low_symptomatic",
    *(rule_id for rule_id, _, _ in _RED_FLAG_RULES),
    "palpitations_severe",
    "preeclampsia",
)


# ======================================================================
# 5. Public API
# ======================================================================


def _affirmed_spans(
    pattern: re.Pattern[str], analysis: _Analysis, rule_starts: list[int]
) -> list[tuple[int, int]]:
    """Affirmed (non-negated, non-suppressed) match spans of a helper pattern."""
    return [
        m.span()
        for m in pattern.finditer(analysis.text)
        if analysis.status(m.start(), m.end(), rule_starts, exertion=False) == "affirmed"
    ]


def classify_detailed(message: str) -> EmergencyResult:
    """Classify a raw patient message and explain the decision.

    Deterministic, LLM-free and network-free; runs before any LLM call.

    Args:
        message: The raw (un-redacted) patient message.

    Returns:
        An :class:`EmergencyResult`; ``action`` is ``ActionEnum.ESCALATE`` if any
        rule fired, otherwise ``None``.
    """
    text = normalize_text(message) if message else ""
    if not text:
        return EmergencyResult(action=None)
    analysis = _Analysis(text)
    fired: dict[str, list[tuple[int, int]]] = {}
    negated: set[str] = set()
    suppressed: set[str] = set()

    # --- blood pressure --------------------------------------------------
    has_context = bool(_BP_CONTEXT_RE.search(text))
    readings = [
        r
        for r in _extract_readings(text, has_context, analysis.personal)
        if not analysis.clause_at(r.start).historical
    ]
    crisis = next((r for r in readings if _is_crisis(r, has_context)), None)
    low = next(
        (r for r in readings if has_context and 0 < r.systolic < SYSTOLIC_LOW), None
    )
    decisive = crisis or low or (readings[0] if readings else None)
    bp = _report_bp(decisive, readings, analysis) if decisive else None
    if crisis is not None:
        fired["bp_crisis"] = [(crisis.start, crisis.end)]

    # --- red-flag symptoms -------------------------------------------------
    raw_matches = [
        (rule_id, m.start(), m.end(), exertion)
        for rule_id, pattern, exertion in _RED_FLAG_RULES
        for m in pattern.finditer(text)
    ]
    rule_starts = sorted(start for _, start, _, _ in raw_matches)
    for rule_id, start, end, exertion in raw_matches:
        verdict = analysis.status(start, end, rule_starts, exertion=exertion)
        if verdict == "affirmed":
            fired.setdefault(rule_id, []).append((start, end))
        elif verdict == "negated":
            negated.add(rule_id)
        else:
            suppressed.add(rule_id)

    # --- composite rules ---------------------------------------------------
    if low is not None:
        symptoms = _affirmed_spans(_LOW_BP_SYMPTOM_RE, analysis, rule_starts)
        if symptoms:
            fired["bp_low_symptomatic"] = [(low.start, low.end), *symptoms]
    if "syncope" in fired or "chest_pain" in fired:
        palpitations = _affirmed_spans(_PALPITATIONS_RE, analysis, rule_starts)
        if palpitations:
            fired["palpitations_severe"] = palpitations
    pregnancy = _affirmed_spans(_PREGNANCY_RE, analysis, rule_starts)
    if pregnancy:
        warning_signs = _affirmed_spans(_PREECLAMPSIA_SYMPTOM_RE, analysis, rule_starts)
        if warning_signs:
            fired["preeclampsia"] = pregnancy + warning_signs

    matched = [rule_id for rule_id in RULE_IDS if rule_id in fired]
    spans = sorted({span for rule_id in matched for span in fired[rule_id]})
    evidence = list(dict.fromkeys(text[s:e] for s, e in spans))
    return EmergencyResult(
        action=ActionEnum.ESCALATE if matched else None,
        matched_rules=matched,
        negated=[r for r in RULE_IDS if r in negated and r not in fired],
        bp=bp,
        evidence_spans=evidence,
        suppressed=[r for r in RULE_IDS if r in suppressed and r not in fired],
    )


def classify(message: str) -> ActionEnum | None:
    """Evaluate a raw patient message against the deterministic emergency rules.

    This function runs **before** any LLM invocation in the pipeline and must
    complete well within 10 ms (typically < 2 ms).

    Args:
        message: The raw patient message text.

    Returns:
        ``ActionEnum.ESCALATE`` if an emergency is detected, ``None`` if the
        message is safe to continue through the pipeline.
    """
    return classify_detailed(message).action

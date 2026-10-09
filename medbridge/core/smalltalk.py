"""
Deterministic small-talk detector (ADL-029).

Greetings, thanks, goodbyes, "who are you / what can you do" and bare
acknowledgements ("ok", "cool") carry no clinical content, yet without this
stage they would be pushed through extraction, both gates and retrieval —
costing several LLM calls and usually ending in an ABSTAIN that reads as a
refusal to say hello.

The detector is intentionally conservative: the *whole* (normalised) message
must match one of a few fixed patterns, it must be short, and it must not
contain digits or any clinical vocabulary. Anything in doubt returns ``None``
and goes through the full Dual Answerability Gate pipeline. "Yes"/"no" are
never small talk because they often answer a SOFT-ASK question.

It runs *after* the emergency fast-path, so "hi, I have chest pain" is
escalated before this stage is ever reached.
"""

from __future__ import annotations

import re

MAX_CHARS = 60
MAX_WORDS = 8

_ADDRESSEE = r"(?: (?:there|all|everyone|medbridge|doc|doctor|bot|friend))?"
_THANKS_CORE = (
    r"(?:thanks|thank you|thank u|thankyou|thx|ty|many thanks|much appreciated|"
    r"appreciate it|i appreciate it)"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "thanks",
        re.compile(
            r"(?:(?:ok|okay|great|cool|perfect|awesome|nice|got it)\s)?"
            + _THANKS_CORE
            + r"(?: (?:so much|very much|a lot|again))?"
            + r"(?: for (?:the|your) (?:help|information|info|answer|explanation|time))?"
            + _ADDRESSEE
        ),
    ),
    (
        "goodbye",
        re.compile(
            r"(?:(?:ok|okay|thanks|thank you)\s)?"
            r"(?:bye|bye bye|goodbye|good bye|see you|see ya|cya|take care|good night|"
            r"goodnight|talk to you later|talk later|ttyl)"
            r"(?: (?:then|now|for now|later))?" + _ADDRESSEE
        ),
    ),
    (
        "identity",
        re.compile(
            r"(?:who are you|what are you|who is this|what is this|what is medbridge|"
            r"what can you do|what do you do|what can you help (?:me )?with|"
            r"how can you help(?: me)?|how does this work|introduce yourself|"
            r"tell me about yourself|what are your capabilities|help|"
            r"are you (?:a|an) (?:bot|robot|ai|human|real person|doctor))"
        ),
    ),
    (
        "greeting",
        re.compile(
            r"(?:(?:hi+|hello|helo|hey+|heya|hiya|howdy|greetings|namaste|"
            r"good (?:morning|afternoon|evening|day))" + _ADDRESSEE + r")?"
            r"(?:\s?(?:how are you(?: doing)?(?: today)?|how is it going|whats up|what's up|sup))?"
        ),
    ),
    (
        "acknowledgement",
        re.compile(
            r"(?:ok|okay|okk|k|kk|cool|got it|alright|all right|sure|fine|great|nice|"
            r"perfect|awesome|understood|makes sense|i see|noted)"
        ),
    ),
]

# Any of these (as a word prefix) means the message has clinical content.
_CLINICAL_TERMS = re.compile(
    r"\b(?:bp|blood|pressure|hypert|hypot|heart|cardi|chest|pain|ache|dizz|faint|"
    r"breath|stroke|pulse|palpit|swell|bleed|headache|vision|numb|weak|tired|"
    r"symptom|med(?!bridge)|pill|tablet|drug|dose|dosage|mg|statin|aspirin|insulin|sugar|"
    r"diabet|kidney|cholesterol|salt|sodium|pregnan|sick|ill|hurt|emergency|"
    r"doctor said|prescri|diagnos|test|lab|reading)"
)


def _normalise(text: str) -> str:
    lowered = text.lower().replace("’", "'").replace("`", "'")
    stripped = re.sub(r"[^a-z0-9' ]+", " ", lowered)
    return " ".join(stripped.split())


def is_small_talk(text: str) -> str | None:
    """Return the small-talk template key for ``text`` or ``None``.

    Keys: ``greeting``, ``thanks``, ``goodbye``, ``identity``, ``acknowledgement``.
    """
    if not text or len(text.strip()) > MAX_CHARS:
        return None
    normalised = _normalise(text)
    if not normalised or len(normalised.split()) > MAX_WORDS:
        return None
    if any(ch.isdigit() for ch in normalised) or _CLINICAL_TERMS.search(normalised):
        return None
    for key, pattern in _PATTERNS:
        if pattern.fullmatch(normalised):
            return key
    return None

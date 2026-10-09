"""
Deterministic lay → clinical query expansion for guideline retrieval.

Patients describe things in everyday words ("water pill", "blood thinner",
"my heart is racing"), whereas guidelines use clinical terminology
("thiazide diuretic", "anticoagulant", "atrial fibrillation"). Appending the
clinical synonyms to the search query lets both the BM25 (sparse) and the
dense retriever find the right guideline passages.

The expansion is purely additive (the original text is preserved verbatim),
deterministic, free of duplicates and bounded in length.
"""
from __future__ import annotations

import re

from medbridge.nlp.drugs import find_drugs

# (pattern, clinical terms) — checked in order; patterns are case-insensitive.
LAY_TERM_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"\bwater ?pills?\b|\bwater tablets?\b", "thiazide diuretic"),
    (r"\bblood ?thinners?\b", "anticoagulant antiplatelet"),
    (r"\bhigh (?:bp|blood pressure|pressure)\b|\bhypertensive\b", "hypertension"),
    (r"\blow (?:bp|blood pressure|pressure)\b", "hypotension"),
    (r"\bbp\b|(?<!blood )\bpressure\b", "blood pressure"),
    (r"\bheart attacks?\b", "myocardial infarction acute coronary syndrome"),
    (
        r"\birregular heart ?beats?\b|\bflutter(?:ing|s)?\b|\bracing heart\b"
        r"|\bheart (?:is )?racing\b|\bskipped beats?\b|\bpalpitations?\b",
        "atrial fibrillation arrhythmia palpitations",
    ),
    (r"\bcholesterol (?:pills?|tablets?|medicines?|medications?|drugs?)\b", "statin"),
    (r"\bsalt(?:y)?\b", "sodium"),
    (
        r"\bswollen (?:ankles?|feet|legs)\b|\b(?:ankle|leg|feet) swelling\b",
        "edema",
    ),
    (r"\bweak heart\b|\bheart is weak\b", "heart failure reduced ejection fraction"),
    (
        r"\bchest (?:pain|tightness|pressure|heaviness)\b|\bchest (?:feels|is) (?:tight|heavy)\b"
        r"|\btight chest\b",
        "angina",
    ),
    (r"\bsugar\b|\bblood sugar\b|\bdiabetic\b", "diabetes"),
    (r"\bkidneys?\b", "chronic kidney disease eGFR"),
    (
        r"\bpregnan(?:t|cy)\b|\bexpecting a baby\b",
        "hypertensive disorders of pregnancy preeclampsia",
    ),
    (r"\bdizzy (?:when|on) (?:standing|getting up)\b", "orthostatic hypotension"),
    (r"\bhome (?:monitor|readings?|machine|bp)\b", "home blood pressure monitoring"),
    (r"\bsmok(?:e|ing|er)\b|\bcigarettes?\b", "tobacco cessation"),
    (r"\bbooze\b|\bdrinking\b|\bbeer\b|\bwine\b", "alcohol intake"),
)

_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), terms) for pattern, terms in LAY_TERM_SYNONYMS
)

MAX_EXPANSION_TERMS = 12
MAX_EXPANSION_CHARS = 300


def expansion_terms(text: str, present_in: str | None = None) -> list[str]:
    """Clinical terms suggested by ``text`` (lay synonyms, then drug generics and classes).

    Terms already present in ``present_in`` (default: ``text``) are skipped and
    the result has no duplicates.
    """
    lowered = (text if present_in is None else present_in).lower()
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if term and term.lower() not in lowered and term.lower() not in (t.lower() for t in terms):
            terms.append(term)

    for pattern, clinical in _COMPILED:
        if pattern.search(text or ""):
            add(clinical)
    for drug in find_drugs(text or ""):
        if drug.brand is not None:
            add(drug.generic.replace(" + ", " "))
        add(drug.drug_class.replace(" + ", " "))
    return terms[:MAX_EXPANSION_TERMS]


def expand_query(text: str, context: str = "") -> str:
    """Return ``text`` with clinical synonyms appended.

    Args:
        text: The search query (kept verbatim at the front).
        context: Optional extra text (e.g. the patient's original message) whose
            lay terms and drug brands also contribute expansion terms.

    Returns:
        ``text`` followed by up to ``MAX_EXPANSION_TERMS`` new terms, never
        longer than ``len(text) + MAX_EXPANSION_CHARS`` characters.
    """
    base = re.sub(r"\s+", " ", text or "").strip()
    terms = expansion_terms(f"{base}\n{context}" if context else base, present_in=base)
    extra = ""
    for term in terms:
        candidate = f"{extra} {term}".strip()
        if len(candidate) > MAX_EXPANSION_CHARS:
            break
        extra = candidate
    return f"{base} {extra}".strip() if extra else base

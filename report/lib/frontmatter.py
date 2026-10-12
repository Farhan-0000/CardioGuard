"""
Default preliminary pages, used only when ``report/content/`` has no file for a slot.

The cover, title, declaration, certificate and acknowledgement pages follow the VIT
template and are written by ``lib/vit_front.py`` from ``report/team.json``. This module
only holds the fallbacks for the abstract and the list of symbols and abbreviations, and
the placeholder text for a missing chapter.
"""

from __future__ import annotations

#: Canonical chapter titles, used for placeholder chapters when a file is missing.
CHAPTER_TITLES: dict[int, str] = {
    1: "Introduction",
    2: "Literature and Patent Review",
    3: "Methodology and System Design",
    4: "Implementation and Results",
    5: "Conclusion and Future Work",
}

ABSTRACT = """#* Abstract

[PENDING: abstract (report/content/04_abstract.md)]
"""

#: Fallback list of abbreviations (generic expansions of terms used across the report).
ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("ABSTAIN", "Routing action: decline to answer and refer to a clinician"),
    ("API", "Application Programming Interface"),
    ("BM25", "Best Matching 25 (lexical ranking function)"),
    ("BP", "Blood Pressure"),
    ("CI", "Confidence Interval"),
    ("CKD", "Chronic Kidney Disease"),
    ("ESCALATE", "Routing action: urgent referral to emergency care"),
    ("GENERALIZE", "Routing action: general, non-personalised guidance"),
    ("LLM", "Large Language Model"),
    ("MRR", "Mean Reciprocal Rank"),
    ("nDCG", "normalised Discounted Cumulative Gain"),
    ("NLI", "Natural Language Inference"),
    ("PII", "Personally Identifiable Information"),
    ("PHI", "Protected Health Information"),
    ("RAG", "Retrieval-Augmented Generation"),
    ("RRF", "Reciprocal Rank Fusion"),
    ("SOFT-ASK", "Routing action: ask the patient for missing context"),
    ("SSE", "Server-Sent Events"),
)

ABBREVIATIONS_MD = (
    "#* List of Symbols and Abbreviations\n\n| Abbreviation | Expansion |\n|---|---|\n"
    + "\n".join(f"| {a} | {b} |" for a, b in ABBREVIATIONS)
)

DEFAULTS: dict[str, str] = {
    "abstract": ABSTRACT,
    "abbreviations": ABBREVIATIONS_MD,
}


def placeholder_chapter(number: int, expected: str) -> str:
    title = CHAPTER_TITLES.get(number, f"Chapter {number}")
    return f"# {title}\n\n[PENDING: chapter {number} content ({expected})]\n"

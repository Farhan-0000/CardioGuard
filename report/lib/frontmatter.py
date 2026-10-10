"""
Default preliminary pages, used only when ``report/content/`` has no file for a slot.

Everything the team must fill by hand is a yellow ``[TO BE FILLED: ...]``
placeholder. The defaults are deliberately short; the content stream normally
provides the real pages (``00_title.md``, ``01_bonafide.md``, ...).
"""

from __future__ import annotations

REPORT_TITLE = (
    "CardioGuard: An Evidence-Grounded Clinical Decision Support & Patient Consultation "
    "System for Hypertension and Cardiovascular Care"
)
SYSTEM_NAME = "MedBridge"
STUDENTS: tuple[tuple[str, str], ...] = (
    ("Vijay Aravindh S.", "23BAI1160"),
    ("Mohammad Farhan Rashid", "23BAI1529"),
)
GUIDE = "Dr. Maheswari S."

#: Canonical chapter titles, used for placeholder chapters when a file is missing.
CHAPTER_TITLES: dict[int, str] = {
    1: "Introduction",
    2: "Literature and Patent Review",
    3: "Methodology and System Design",
    4: "Implementation and Results",
    5: "Conclusion and Future Work",
}

_students_md = "\n\n".join(f"**{name.upper()} ({reg})**" for name, reg in STUDENTS)
_students_inline = " and ".join(f"**{name} ({reg})**" for name, reg in STUDENTS)

TITLE_PAGE = f"""**{REPORT_TITLE.upper()}**

*System name: {SYSTEM_NAME}*

A PROJECT REPORT

Submitted by

{_students_md}

in partial fulfilment of the requirements for the award of the degree of

**[TO BE FILLED: degree]**

Under the guidance of

**{GUIDE}**

[TO BE FILLED: guide designation], [TO BE FILLED: department]

[TO BE FILLED: university logo]

**[TO BE FILLED: department / school name]**

**[TO BE FILLED: university name]**

Year 4, Semester VII

**[TO BE FILLED: month and year of submission]**
"""

BONAFIDE = f"""#* Bonafide Certificate

Certified that this project report titled **"{REPORT_TITLE}"** is the bonafide work of \
{_students_inline}, who carried out the project work under my supervision during \
[TO BE FILLED: academic year], Semester VII.

**Signature of the Guide**

{GUIDE}, [TO BE FILLED: guide designation]

**Signature of the Head of the Department**

[TO BE FILLED: name and designation of the Head of the Department]

Submitted for the viva-voce examination held on [TO BE FILLED: date].

**Internal Examiner** [TO BE FILLED: signature]

**External Examiner** [TO BE FILLED: signature]
"""

DECLARATION = f"""#* Declaration

We declare that the project report titled **"{REPORT_TITLE}"** is a record of original work \
carried out by us under the supervision of **{GUIDE}**, and that it has not been submitted \
for the award of any other degree or diploma. All data used in this work are synthetic.

Place: [TO BE FILLED: place]

Date: [TO BE FILLED: date]

{_students_md}
"""

ACKNOWLEDGEMENT = """#* Acknowledgement

[TO BE FILLED: acknowledgement]
"""

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
    "#* List of Abbreviations\n\n| Abbreviation | Expansion |\n|---|---|\n"
    + "\n".join(f"| {a} | {b} |" for a, b in ABBREVIATIONS)
)

DEFAULTS: dict[str, str] = {
    "title": TITLE_PAGE,
    "bonafide": BONAFIDE,
    "declaration": DECLARATION,
    "acknowledgement": ACKNOWLEDGEMENT,
    "abstract": ABSTRACT,
    "abbreviations": ABBREVIATIONS_MD,
}


def placeholder_chapter(number: int, expected: str) -> str:
    title = CHAPTER_TITLES.get(number, f"Chapter {number}")
    return f"# {title}\n\n[PENDING: chapter {number} content ({expected})]\n"

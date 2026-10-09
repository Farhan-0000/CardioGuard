"""
Citation-label matching against the corpus metadata of ``CitationResponse``.

A vignette lists acceptable sources in ``expected_citations``. A label matches a
returned citation when it identifies the same corpus document:

* a ``guideline_id`` written exactly (``"NICE_NG133"``, case-insensitive);
* an ``"ORG YEAR"`` label (``"ESC 2024"``, ``"AHA/ACC 2025"``) equal to the
  citation's ``org`` + ``year`` (the pipeline's ``source`` label is built the
  same way, see ``medbridge.core.citations.source_label``);
* an organisation-only label (``"MedlinePlus"``, ``"KDIGO"``) contained in ``org``;
* otherwise a phrase contained in the document ``title``.

Labels are validated against ``data/corpus_manifest.json`` by the unit tests,
so a label that names a document absent from the corpus fails fast.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from tests.benchmark.schemas import CitationRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "corpus_manifest.json"

_YEAR_SUFFIX = re.compile(r"^(?P<org>.+?)\s+(?P<year>(?:19|20)\d{2})$")


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def label_matches(label: str, citation: CitationRecord) -> bool:
    """True when ``label`` identifies the document behind ``citation``."""
    wanted = _norm(label)
    if not wanted:
        return False
    gid = _norm(citation.guideline_id)
    if gid and wanted.replace(" ", "_") == gid:
        return True
    org = _norm(citation.org)
    year = str(citation.year) if citation.year else ""
    match = _YEAR_SUFFIX.match(wanted)
    if match:
        if org and year:
            return match.group("org") == org and match.group("year") == year
        # No structured metadata: fall back to the human-readable source label.
        return _norm(citation.source) == wanted
    if org and wanted in org:
        return True
    title = _norm(citation.title)
    return bool(title) and wanted in title


def count_matches(labels: Iterable[str], citations: Iterable[CitationRecord]) -> int:
    """Number of citations that match at least one expected label."""
    wanted = [label for label in labels if label.strip()]
    return sum(1 for c in citations if any(label_matches(label, c) for label in wanted))


def citation_precision(
    labels: list[str], citations: list[CitationRecord]
) -> tuple[float | None, int]:
    """(precision, matched) — ``None`` when there is nothing to score."""
    if not citations or not any(label.strip() for label in labels):
        return None, 0
    matched = count_matches(labels, citations)
    return matched / len(citations), matched


# ---------------------------------------------------------------------------
# Corpus manifest
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_manifest_documents(path: str = str(MANIFEST_PATH)) -> tuple[dict[str, Any], ...]:
    """The ``documents`` list of the corpus manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(data.get("documents") or [])


def manifest_citation(document: dict[str, Any]) -> CitationRecord:
    """A citation record carrying a manifest document's metadata."""
    org = str(document.get("org") or "")
    year = document.get("year")
    return CitationRecord(
        source=f"{org} {year}".strip(),
        org=org or None,
        year=int(year) if isinstance(year, int) else None,
        title=document.get("title"),
        guideline_id=document.get("guideline_id"),
        chunk_id=f"{document.get('guideline_id')}::manifest",
    )


def corpus_documents_for(label: str) -> list[CitationRecord]:
    """Every corpus document that ``label`` matches (empty → label not in the corpus)."""
    return [
        record
        for record in (manifest_citation(d) for d in load_manifest_documents())
        if label_matches(label, record)
    ]

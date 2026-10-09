"""
Curated corpus manifest (Module M-18, Ingestion Pipeline).

``data/corpus_manifest.json`` is the single source of truth for which documents
form the MedBridge knowledge base. Every entry is a curated, authoritative
source (clinical society / government guideline, government patient page or a
peer-reviewed statement/review) with its licence, canonical URL and SHA-256 so
the corpus is versioned and reproducible. Nothing is scraped: a document is in
the knowledge base only if a human added it to this file (RAG-poisoning
defence, see ADL "curated corpus").
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

DocType = Literal["guideline", "patient_info", "review"]
SourceFormat = Literal["pdf", "html"]

ALLOWED_TOPICS: frozenset[str] = frozenset(
    {
        "hypertension",
        "hypertensive_emergency",
        "heart_failure",
        "atrial_fibrillation",
        "coronary_disease",
        "dyslipidaemia",
        "ckd",
        "diabetes",
        "pregnancy",
        "prevention",
        "lifestyle",
        "medication",
        "bp_measurement",
    }
)

_GUIDELINE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when the corpus manifest is missing, malformed or inconsistent."""


class CorpusDocument(BaseModel):
    """One curated knowledge-base document."""

    guideline_id: str = Field(..., description="Stable identifier, e.g. 'AHA_ACC_HTN_2025'")
    title: str = Field(..., min_length=3)
    short_title: str = Field(..., min_length=2)
    org: str = Field(..., min_length=2, description="Issuing organisation(s)")
    year: int = Field(..., ge=1990, le=2100)
    doc_type: DocType
    url: str = Field(..., description="Canonical landing page / DOI URL")
    download_url: str = Field(default="", description="Direct file URL used to download")
    licence: str = Field(..., min_length=3, description="Licence as stated by the publisher")
    file: str = Field(..., description="Path relative to the repository root")
    sha256: str = Field(default="", description="SHA-256 of the local file")
    pages: int = Field(default=0, ge=0, description="PDF pages or HTML sections")
    chunk_count: int = Field(default=0, ge=0)
    table_chunk_count: int = Field(default=0, ge=0)
    topics: list[str] = Field(..., min_length=1)
    language: str = Field(default="en")
    source_format: SourceFormat = "pdf"
    html_root: str = Field(
        default="", description="Optional '#id', '.class' or tag selecting the HTML main content"
    )
    volatile: bool = Field(
        default=False, description="Publisher regenerates the file on each download (hash drifts)"
    )
    doi: str = ""
    notes: str = ""

    @field_validator("guideline_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _GUIDELINE_ID_RE.match(value):
            raise ValueError(f"guideline_id '{value}' must be UPPER_SNAKE_CASE (3-64 chars)")
        return value

    @field_validator("topics")
    @classmethod
    def _check_topics(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - ALLOWED_TOPICS)
        if unknown:
            raise ValueError(f"unknown topics {unknown}; allowed: {sorted(ALLOWED_TOPICS)}")
        if len(set(value)) != len(value):
            raise ValueError("topics must not contain duplicates")
        return value

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        if value and not _SHA256_RE.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("url", "download_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if value and not value.startswith("https://"):
            raise ValueError(f"URL must be https: '{value}'")
        return value

    @model_validator(mode="after")
    def _check_format(self) -> CorpusDocument:
        suffix = Path(self.file).suffix.lower()
        expected = "html" if suffix in {".html", ".htm"} else "pdf" if suffix == ".pdf" else ""
        if not expected:
            raise ValueError(f"unsupported file type '{suffix}' for {self.guideline_id}")
        if expected != self.source_format:
            raise ValueError(
                f"source_format '{self.source_format}' does not match file '{self.file}'"
            )
        return self

    def resolve_path(self, root: Path) -> Path:
        """Absolute path of the document file under repository ``root``."""
        return (root / self.file).resolve()


class CorpusManifest(BaseModel):
    """The full curated manifest."""

    manifest_version: int = 1
    description: str = ""
    chunking: dict[str, int | str] = Field(default_factory=dict)
    documents: list[CorpusDocument]

    @model_validator(mode="after")
    def _check_unique(self) -> CorpusManifest:
        ids = [d.guideline_id for d in self.documents]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate guideline_id(s): {dupes}")
        files = [d.file for d in self.documents]
        dup_files = sorted({f for f in files if files.count(f) > 1})
        if dup_files:
            raise ValueError(f"duplicate file(s): {dup_files}")
        return self

    def by_id(self) -> dict[str, CorpusDocument]:
        """Map guideline_id -> document."""
        return {d.guideline_id: d for d in self.documents}

    def by_filename(self) -> dict[str, CorpusDocument]:
        """Map file basename -> document (used by the legacy ``--pdf-dir`` mode)."""
        return {Path(d.file).name: d for d in self.documents}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> CorpusManifest:
    """Load and validate a corpus manifest.

    Raises:
        ManifestError: If the file is missing, not JSON, or fails validation.
    """
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    try:
        return CorpusManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestError(f"manifest validation failed: {exc}") from exc


def save_manifest(manifest: CorpusManifest, path: Path) -> None:
    """Write the manifest back as stable, human-diffable JSON."""
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_files(manifest: CorpusManifest, root: Path) -> list[str]:
    """Return human-readable problems for missing files or SHA-256 mismatches."""
    problems: list[str] = []
    for doc in manifest.documents:
        path = doc.resolve_path(root)
        if not path.exists():
            problems.append(f"{doc.guideline_id}: missing file {doc.file}")
            continue
        if doc.sha256 and sha256_file(path) != doc.sha256:
            problems.append(f"{doc.guideline_id}: sha256 mismatch for {doc.file}")
    return problems

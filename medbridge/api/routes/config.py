"""
Public UI configuration route (Module M-04) — ``GET /api/config/public``.

Everything the web client needs before the first message: app name, the
configurable emergency numbers, whether the deployment is local-only, the
model names, the medical disclaimer, six starter suggestions (``suggestions``:
``{title, prompt}`` cards, plus the plain ``suggested_prompts`` list kept for
older clients) and a summary of the curated corpus (titles / organisations /
years only, read from ``data/corpus_manifest.json`` when it exists). Contains
no secrets and no session data, so it needs no authentication.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from medbridge.api.routes.health import APP_VERSION
from medbridge.config import Settings, get_settings

logger = structlog.get_logger(__name__)

config_router = APIRouter(prefix="/config", tags=["config"])

CORPUS_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus_manifest.json"

# Starter suggestions shown on the empty chat screen: (card title, prompt sent).
SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("Check a reading", "My blood pressure was 148/92 this morning. Is that high?"),
    ("Lower it naturally", "What lifestyle changes help bring blood pressure down?"),
    ("Medicine side effects", "I take amlodipine 5 mg. What side effects should I watch out for?"),
    ("Targets with diabetes", "What blood pressure target is recommended if I also have diabetes?"),
    ("Measure at home", "How do I measure my blood pressure correctly at home?"),
    ("Atrial fibrillation", "What is atrial fibrillation and why does it raise stroke risk?"),
)
# Plain prompt list kept for older clients (Streamlit, v1 web UI).
SUGGESTED_PROMPTS: tuple[str, ...] = tuple(prompt for _, prompt in SUGGESTIONS)


class Suggestion(BaseModel):
    """One starter suggestion card."""

    title: str
    prompt: str


class CorpusDocumentSummary(BaseModel):
    """One knowledge-base document as shown in the UI."""

    title: str
    org: str | None = None
    year: int | None = None


class CorpusSummary(BaseModel):
    """Summary of the curated guideline corpus."""

    documents: int = 0
    items: list[CorpusDocumentSummary] = Field(default_factory=list)


class PublicConfig(BaseModel):
    """Response body for ``GET /api/config/public``."""

    app_name: str
    version: str
    emergency_number: str
    ambulance_number: str
    emergency_region_label: str
    local_only_mode: bool
    models: dict[str, str]
    disclaimer: str
    suggested_prompts: list[str]
    suggestions: list[Suggestion] = Field(default_factory=list)
    corpus: CorpusSummary | None = None


def build_disclaimer(settings: Settings) -> str:
    """Patient-facing medical disclaimer using the configured emergency numbers."""
    return (
        f"{settings.APP_NAME} gives general, guideline-based information for education "
        "only. It does not diagnose conditions, prescribe treatment or replace your "
        "doctor. In an emergency call "
        f"{settings.EMERGENCY_NUMBER} (or {settings.AMBULANCE_NUMBER} for an ambulance in "
        f"{settings.EMERGENCY_REGION_LABEL}); in the United States call 911."
    )


_corpus_cache: dict[tuple[str, float], CorpusSummary | None] = {}
_corpus_lock = threading.Lock()


def _parse_manifest(raw: Any) -> CorpusSummary:
    entries = raw.get("documents", []) if isinstance(raw, dict) else raw
    items: list[CorpusDocumentSummary] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("title"):
            continue
        year = entry.get("year")
        items.append(
            CorpusDocumentSummary(
                title=str(entry["title"]),
                org=str(entry["org"]) if entry.get("org") else None,
                year=int(year) if isinstance(year, int) else None,
            )
        )
    return CorpusSummary(documents=len(items), items=items)


def load_corpus_summary(path: Path | None = None) -> CorpusSummary | None:
    """Corpus summary from the manifest (cached per file modification time)."""
    path = path or CORPUS_MANIFEST_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = (str(path), mtime)
    with _corpus_lock:
        if key in _corpus_cache:
            return _corpus_cache[key]
    try:
        summary: CorpusSummary | None = _parse_manifest(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError) as exc:
        logger.warning("corpus_manifest_unreadable", error=type(exc).__name__)
        summary = None
    with _corpus_lock:
        _corpus_cache.clear()
        _corpus_cache[key] = summary
    return summary


@config_router.get(
    "/public",
    response_model=PublicConfig,
    summary="Public UI configuration (emergency numbers, disclaimer, suggestions, corpus)",
)
async def get_public_config() -> PublicConfig:
    """Return non-sensitive configuration for the web client."""
    settings = get_settings()
    return PublicConfig(
        app_name=settings.APP_NAME,
        version=APP_VERSION,
        emergency_number=settings.EMERGENCY_NUMBER,
        ambulance_number=settings.AMBULANCE_NUMBER,
        emergency_region_label=settings.EMERGENCY_REGION_LABEL,
        local_only_mode=settings.LOCAL_ONLY_MODE,
        models={"gates": settings.gates_model, "generator": settings.generator_model},
        disclaimer=build_disclaimer(settings),
        suggested_prompts=list(SUGGESTED_PROMPTS),
        suggestions=[Suggestion(title=title, prompt=prompt) for title, prompt in SUGGESTIONS],
        corpus=load_corpus_summary(),
    )

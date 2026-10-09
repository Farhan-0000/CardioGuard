"""
Definitions shared by the LLM judge, the NLI judge and the analysis.

* Claim labels and response behaviours (with the behaviour → action mapping
  used for behaviour-based routing of ungated conditions).
* Deterministic template detection: responses produced without the generator
  (ABSTAIN/ESCALATE templates, small-talk templates, generator-failure
  fallback) contain no clinical claims and are never sent to a judge model.
* Sentence splitting and citation-marker extraction for citation precision.
* The judge-side evidence sidecar (chunk texts retrieved by the judge).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluation.evidence import EvidenceChunk
from evaluation.io import append_jsonl, iter_jsonl

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"
CLAIM_LABELS: tuple[str, ...] = (SUPPORTED, UNSUPPORTED, CONTRADICTED)

BEHAVIOURS: tuple[str, ...] = (
    "ANSWERED_SPECIFIC",
    "GENERAL_INFO",
    "ASKED_CLARIFICATION",
    "ABSTAINED",
    "ESCALATED",
)
BEHAVIOUR_TO_ACTION: dict[str, str] = {
    "ANSWERED_SPECIFIC": "ANSWER",
    "GENERAL_INFO": "GENERALIZE",
    "ASKED_CLARIFICATION": "SOFT-ASK",
    "ABSTAINED": "ABSTAIN",
    "ESCALATED": "ESCALATE",
}
_ACTION_TO_TEMPLATE_BEHAVIOUR: dict[str, str] = {
    "ABSTAIN": "ABSTAINED",
    "ESCALATE": "ESCALATED",
    "SOFT-ASK": "ASKED_CLARIFICATION",
}
TEMPLATE_ACTIONS: frozenset[str] = frozenset({"ABSTAIN", "ESCALATE"})
FLAGS: tuple[str, ...] = (
    "accepted_false_premise",
    "endorsed_fabricated_entity",
    "gave_unsupported_dose",
)

EVIDENCE_SIDECAR = "evidence_chunks.jsonl"

_PROFESSIONAL = (
    r"(?:doctors?|physicians?|health ?care (?:providers?|professionals?|team)|providers?|pharmacists?"
    r"|cardiologists?|clinicians?|gps?|nurses?|medical professionals?|care team)"
)
_ADVICE_LEAD = (
    r"^\s*(?:(?:it(?:'s|’s| is) (?:important|best|advisable|recommended|a good idea|essential|wise)"
    r" to|please|you should|you may want to|always|be sure to|make sure (?:to|you)|remember to"
    r"|consider)\s+)?"
)
# The referral is the main action ("Talk to your doctor…", "Your doctor can…"), not a clause
# attached to a clinical statement ("Do not stop beta-blockers abruptly without talking to…").
_REFERRAL_RE = re.compile(
    _ADVICE_LEAD + r"(?:consult|talk|speak|discuss|check|ask|contact|see|visit|follow up|reach out"
    r"|let)\b.*\b" + _PROFESSIONAL + r"\b"
    r"|^\s*your " + _PROFESSIONAL + r"\b",
    re.IGNORECASE,
)
_EMERGENCY_ADVICE_RE = re.compile(
    _ADVICE_LEAD + r"(?:immediately\s+)?(?:call (?:911|112|108|999|an ambulance|emergency services)"
    r"|seek (?:immediate|urgent|emergency)(?: medical)? (?:care|attention|help)"
    r"|go to (?:the |your )?(?:nearest )?(?:emergency|er\b|a&e|hospital))",
    re.IGNORECASE,
)
_SELF_REFERENCE_RE = re.compile(
    r"^\s*(?:i\b|i'm\b|i’m\b|as an ai\b|this (?:tool|assistant|service|chatbot)\b"
    r"|(?:my|our) (?:guidelines|knowledge|sources|information)\b)",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
_GENERIC_MAX_WORDS = 30
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKER_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def is_generator_call(call: dict[str, Any]) -> bool:
    """True for LLM calls made by the response generator."""
    return "generat" in str(call.get("call_name", "")).lower()


def template_reason(record: dict[str, Any]) -> str | None:
    """Why a response is a deterministic template (no clinical claims), or ``None``.

    * ``abstain_template`` / ``escalate_template`` — pre-vetted templates by
      architecture (no LLM is ever involved).
    * ``generation_skipped`` / ``generation_failed`` — the orchestrator recorded
      the ``generation`` stage as skipped (fast-path, small talk) or failed (the
      pre-vetted fallback template was returned).
    * ``no_generation`` — LLM calls were recorded for the turn but none was a
      successful generator call.

    When neither stage timings nor call information say so, the response is
    NOT assumed to be a template; it goes to the judge.
    """
    action = str(record.get("action") or "")
    if action in TEMPLATE_ACTIONS:
        return f"{action.lower()}_template"
    timings = (record.get("explanation") or {}).get("timings") or []
    for timing in timings:
        if timing.get("stage") == "generation" and timing.get("status") in ("skipped", "failed"):
            return f"generation_{timing['status']}"
    calls = record.get("llm_calls") or []
    if calls and not any(is_generator_call(c) and c.get("success", True) for c in calls):
        return "no_generation"
    return None


def non_clinical_reason(claim: str) -> str | None:
    """Deterministic enforcement of the decomposition rubric's exclusions.

    Returns why a "claim" is not a verifiable clinical claim, or ``None``:

    * ``self_reference`` — statements about the assistant or its sources
      ("I can't help with…", "Our guidelines focus on…");
    * ``generic_referral`` — advice to consult / discuss with a doctor,
      pharmacist or other professional;
    * ``generic_emergency_advice`` — "call 112", "seek immediate medical attention".

    Referral and emergency advice are only excluded when the advice is the
    sentence's main action, it contains no number and has at most 30 words, so
    clinical content wrapped in advice ("do not stop beta-blockers abruptly
    without talking to your doctor", "if BP is above 180/120 with chest pain,
    call 911") is still judged.
    """
    text = claim.strip()
    if _SELF_REFERENCE_RE.match(text):
        return "self_reference"
    generic = not _DIGIT_RE.search(text) and len(text.split()) <= _GENERIC_MAX_WORDS
    if generic and _EMERGENCY_ADVICE_RE.search(text):
        return "generic_emergency_advice"
    if generic and _REFERRAL_RE.search(text):
        return "generic_referral"
    return None


def template_behaviour(action: str | None) -> str:
    """Behaviour assigned to a template response without a judge call."""
    return _ACTION_TO_TEMPLATE_BEHAVIOUR.get(str(action or ""), "GENERAL_INFO")


def normalise_token(value: Any) -> str:
    """Upper-case, underscores for spaces/hyphens (``"not supported"`` → ``NOT_SUPPORTED``)."""
    return re.sub(r"[\s\-]+", "_", str(value).strip().upper())


def split_sentences(text: str) -> list[str]:
    """Split a response into sentences / bullet lines (markdown bullets stripped)."""
    parts = (p.strip().lstrip("-*•").strip() for p in _SENTENCE_SPLIT_RE.split(text or ""))
    return [p for p in parts if p]


def citation_markers(text: str) -> list[int]:
    """Numeric citation markers ``[n]`` in order of appearance."""
    return [int(m) for m in _MARKER_RE.findall(text or "")]


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def link_claim_to_sentence(
    claim: dict[str, Any], sentences: list[str], min_overlap: float = 0.3
) -> int | None:
    """Index of the response sentence a claim came from (token-overlap match), or ``None``.

    Uses the judge's ``source_sentence`` when present, else the claim text.
    The score is |A ∩ B| / |A| for the source tokens A.
    """
    source = str(claim.get("source_sentence") or claim.get("claim") or "")
    src = _tokens(_MARKER_RE.sub(" ", source))
    if not src:
        return None
    best, best_score = None, 0.0
    for i, sentence in enumerate(sentences):
        score = len(src & _tokens(_MARKER_RE.sub(" ", sentence))) / len(src)
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= min_overlap else None


def condition_evidence(record: dict[str, Any]) -> list[EvidenceChunk]:
    """Chunks the pipeline passed to Gate 2 / the generator for this turn."""
    out: list[EvidenceChunk] = []
    for item in record.get("evidence_texts") or []:
        if not item.get("chunk_id") or not item.get("text"):
            continue
        chunk = EvidenceChunk.from_dict(item)
        chunk.source = "condition"
        out.append(chunk)
    return out


class EvidenceSidecar:
    """Append-only store of judge-retrieved chunk texts (``<judgments>/evidence_chunks.jsonl``)."""

    def __init__(self, judgments_dir: Path) -> None:
        self.path = judgments_dir / EVIDENCE_SIDECAR
        self._chunks: dict[str, EvidenceChunk] = {}
        for rec in iter_jsonl(self.path):
            chunk = EvidenceChunk.from_dict(rec)
            self._chunks[chunk.chunk_id] = chunk

    def add(self, chunks: Iterable[EvidenceChunk]) -> None:
        """Persist chunks not stored yet."""
        for chunk in chunks:
            if chunk.chunk_id not in self._chunks:
                self._chunks[chunk.chunk_id] = chunk
                append_jsonl(self.path, chunk.to_dict())

    def get(self, chunk_id: str) -> EvidenceChunk | None:
        return self._chunks.get(chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)

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
# The referral is the main action ("Talk to your doctor…", "Your doctor can help…"), not a
# clause attached to a clinical statement ("Do not stop beta-blockers abruptly without
# talking to…"). "Your <professional> …" only counts with a non-clinical predicate.
_REFERRAL_RE = re.compile(
    _ADVICE_LEAD + r"(?:consult|talk|speak|discuss|check|ask|contact|see|visit|follow up|reach out"
    r"|let)\b.*\b" + _PROFESSIONAL + r"\b"
    r"|^\s*your " + _PROFESSIONAL + r"\s+(?:(?:can|will|may|could|should|is|are)\s+)?"
    r"(?:(?:best|always|also)\s+)?(?:help|advise|explain|answer|discuss|decide|choose|guide"
    r"|review|assess|check|tailor|adjust|recommend|tell|work with|know|be (?:able|happy|the best))",
    re.IGNORECASE,
)
_EMERGENCY_ADVICE_RE = re.compile(
    _ADVICE_LEAD + r"(?:immediately\s+)?(?:call (?:911|112|108|999|an ambulance|emergency services)"
    r"|seek (?:immediate|urgent|emergency)(?: medical)? (?:care|attention|help)"
    r"|go to (?:the |your )?(?:nearest )?(?:emergency|er\b|a&e|hospital))",
    re.IGNORECASE,
)
# Statements about the assistant's own limits or scope ("I can't help with…", "Our
# guidelines focus on…"). A clinical statement merely prefixed by a self-reference ("I
# recommend taking…", "My information indicates…") does not match.
_SELF_LIMIT_RE = re.compile(
    r"^\s*(?:as an ai\b[^,]*,\s*)?(?:"
    r"i(?:'m|’m| am)\s+(?:not (?:a doctor|a medical|able|qualified|a substitute)|unable"
    r"|an? (?:ai|virtual|digital|chat)|only an?|designed|here to (?:help|provide|share))"
    r"|i\s+(?:(?:only|really|currently)\s+)?(?:can(?:not|'t|’t| not)|do(?:n't|n’t| not)"
    r" (?:have|know|cover|provide|give|offer)|can only|can help)"
    r"|this (?:tool|assistant|service|chatbot)\s+(?:can(?:not|'t|’t| not)|is not|isn't|isn’t"
    r"|does(?: not|n't|n’t)|can only|is (?:only|designed|here))"
    r"|(?:my|our) (?:guidelines|knowledge|sources|information|role)\s+(?:focus(?:es)? on|covers?"
    r"|(?:do(?:es)?(?: not|n't|n’t)) (?:cover|include|mention)|(?:is|are) (?:limited|focused))"
    r"|as an ai\b)",
    re.IGNORECASE,
)
# Empathy about the user ("I understand…", "I'm sorry to hear…"): non-clinical only
# when it carries no clinical entity.
_EMPATHY_RE = re.compile(
    r"^\s*i(?:'m|’m| am)?\s+(?:really\s+)?(?:understand|hear|hope|sorry|glad|appreciate)\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
# A causal/explanatory clause or an appositive turns advice into an assertion
# ("…, because it is safe to stop warfarin", "Cardiozepam, a newer calcium channel blocker").
_FACTUAL_CLAUSE_RE = re.compile(
    r"\b(?:because|since|which|therefore|thus|hence|due to|so that"
    r"|as (?!soon as\b|needed\b|directed\b|well\b|an ai\b)(?:it|this|that|they|these|those|the|a|an"
    r"|your|you|high|low|there))\b"
    r"|,\s*(?:a|an)\s",
    re.IGNORECASE,
)
# Clinical entities: conditions, measurements, symptoms, tests, drug classes and common drugs.
_CLINICAL_TERM_RE = re.compile(
    r"\b(?:blood pressure|bp|hypertensi\w*|hypotensi\w*|cholesterol|ldl|hdl|triglycerides?"
    r"|lipids?|statins?|heart (?:attack|failure|disease|rate|rhythm|beat)|cardiac|cardiovascular"
    r"|stroke|tia|chest pain|angina|palpitations?|arrhythmias?|atrial|fibrillation|kidneys?"
    r"|renal|diabet\w*|glucose|blood sugar|insulin|potassium|sodium|salt|pregnan\w*"
    r"|pre-?eclampsia|doses?|dosage|dosing|mg|mcg|mmhg|tablets?|capsules?|pills?|inhibitors?"
    r"|blockers?|antagonists?|diuretics?|water pills?|anticoagulants?|antiplatelets?"
    r"|blood thinners?|nsaids?|painkillers?|aspirin|ibuprofen|naproxen|diclofenac|paracetamol"
    r"|acetaminophen|warfarin|heparin|nitroglycerin|digoxin|bleed\w*|headaches?|dizz\w*"
    r"|faint\w*|light-?headed\w*|shortness of breath|breathless\w*|swelling|swollen|o?edema"
    r"|numb\w*|weakness|blurred vision|vision|confusion|seizures?|nosebleeds?|cough\w*"
    r"|ecg|ekg|echocardiogram|blood tests?|urine tests?|cuff|readings?|side effects?"
    r"|interactions?|interacts?|overdose|allerg\w*|grapefruit|alcohol|caffeine|exercise"
    r"|weight|diet|smoking|vaccines?)\b",
    re.IGNORECASE,
)
_DRUG_SUFFIX_RE = re.compile(
    r"\b[a-z]{3,}(?:pril|sartan|olol|ilol|alol|dipine|pamil|tiazem|statin|thiazide|semide"
    r"|tanide|xaban|gatran|parin|farin|grel|gliflozin|gliptin|formin|glutide|azosin|lactone"
    r"|renone|arone|zepam|zolam|profen|fenac|coxib|cillin|mycin|floxacin|oxetine|triptyline)\b",
    re.IGNORECASE,
)
_ALLOWED_CAPS = frozenset({"i", "gp", "gps", "er", "a&e", "ai", "nhs", "us", "uk", "ok"})
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z&'’\-]*")
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


def has_clinical_entity(text: str) -> bool:
    """True when ``text`` names a drug, condition, symptom, test, measurement or other entity.

    Signals: the drug lexicon (``medbridge.nlp.drugs.find_drugs``, brands and
    generics), drug-name suffixes (catches unknown / fabricated names such as
    "cardiozepam"), a clinical-term lexicon, and proper nouns or acronyms after
    the first word ("Cardiozepam", "ACE", "BP").
    """
    if _CLINICAL_TERM_RE.search(text) or _DRUG_SUFFIX_RE.search(text):
        return True
    tokens = _TOKEN_RE.findall(text)
    for token in tokens[1:]:
        if token.lower() in _ALLOWED_CAPS:
            continue
        if token[0].isupper():
            return True
    try:
        from medbridge.nlp.drugs import find_drugs
    except Exception:  # the lexicon is an optional enrichment
        return False
    try:
        return bool(find_drugs(text))
    except Exception:
        return False


def non_clinical_reason(claim: str) -> str | None:
    """Deterministic enforcement of the decomposition rubric's exclusions — pure boilerplate only.

    Returns why a "claim" is not a verifiable clinical claim, or ``None``:

    * ``self_reference`` — the assistant's own limits or scope ("I can't help
      with…", "Our guidelines focus on…"); empathy ("I understand…") only without
      a clinical entity;
    * ``generic_referral`` — advice to consult / discuss with a doctor, pharmacist
      or other professional, or "your doctor can help/advise/…";
    * ``generic_emergency_advice`` — "call 112", "seek immediate medical attention".

    A claim is NEVER excluded when it contains a number, a causal/explanatory
    clause or appositive ("…, because …", "…, a newer calcium channel blocker"),
    or more than 30 words; referral, emergency and empathy statements are also
    kept when they name any clinical entity (``has_clinical_entity``: drug,
    condition, symptom, test, dose…). So false, fabricated-drug and dose claims
    wrapped in advice or self-reference ("Talk to your doctor before stopping
    warfarin, because it is safe to stop it abruptly", "Your doctor may prescribe
    Cardiozepam…", "I recommend taking 81 mg aspirin daily") are always judged.
    """
    text = claim.strip()
    if _DIGIT_RE.search(text) or len(text.split()) > _GENERIC_MAX_WORDS:
        return None
    if _FACTUAL_CLAUSE_RE.search(text):
        return None
    if _SELF_LIMIT_RE.match(text):
        return "self_reference"
    if has_clinical_entity(text):
        return None
    if _EMPATHY_RE.match(text):
        return "self_reference"
    if _EMERGENCY_ADVICE_RE.search(text):
        return "generic_emergency_advice"
    if _REFERRAL_RE.search(text):
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


def marker_chunks(record: dict[str, Any]) -> dict[int, str]:
    """``{marker number: chunk_id}`` from the run record's validated citations."""
    out: dict[int, str] = {}
    for citation in record.get("citations") or []:
        numbers = citation_markers(str(citation.get("marker", "")))
        if numbers and citation.get("chunk_id"):
            out[numbers[0]] = str(citation["chunk_id"])
    return out


def cited_chunks_by_claim(
    record: dict[str, Any], claims: Iterable[dict[str, Any]]
) -> dict[Any, list[str]]:
    """Chunk ids cited (``[n]`` markers) in the response sentence each claim comes from.

    Keyed by the claim's ``index``; a claim that cannot be linked to a sentence,
    or whose sentence has no valid marker, maps to an empty list.
    """
    sentences = split_sentences(str(record.get("response_text") or ""))
    by_marker = marker_chunks(record)
    out: dict[Any, list[str]] = {}
    for claim in claims:
        index = link_claim_to_sentence(claim, sentences)
        markers = citation_markers(sentences[index]) if index is not None else []
        ids = [by_marker[m] for m in markers if m in by_marker]
        out[claim.get("index")] = list(dict.fromkeys(ids))
    return out


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

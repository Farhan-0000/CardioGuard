"""
Deterministic stand-ins for the LLM-backed and storage components of the pipeline.

Used by the hermetic RABBITS mode and the hermetic adversarial tests so the
*real* orchestrator, redactor, emergency classifier, small-talk detector,
extractor post-processing, drug normaliser, query expansion and citation
validation run end to end without Ollama, Qdrant or PostgreSQL.

* :class:`InMemoryTurnStore` — the ``medbridge.state.turn_store`` surface the
  orchestrator uses, kept in memory (one record per turn, snapshot updated).
* :class:`EchoExtractorLLM` — an ``llm.call`` stub for the Context Extractor that
  copies drug names *as written* in the patient's message.
* Gate / retriever / reranker / generator stubs with fixed, documented rules.
* :class:`RecordingLLM` — captures every prompt for leakage/isolation checks.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from medbridge.ai.schemas.context_gate import ContextGateOutput
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.ai.schemas.extractor import DeltaEvent, ExtractorOutput
from medbridge.ai.schemas.response_generator import ResponseGeneratorOutput
from medbridge.api.schemas.enums import EventTypeEnum
from medbridge.nlp.drugs import GENERIC_ALIASES, GENERIC_CLASS, find_drugs
from medbridge.retrieval.hybrid_retriever import RetrievedChunk
from medbridge.state.turn_store import ChatTurn, SessionState, TurnRecord

_UNTRUSTED = re.compile(r"<untrusted_user_input>\n?(.*?)\n?</untrusted_user_input>", re.S)


class NullDB:
    """A no-op stand-in for ``AsyncSession`` (the in-memory store ignores it)."""

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


@dataclass
class InMemoryTurnStore:
    """In-memory implementation of the turn-store functions the orchestrator calls."""

    states: dict[uuid.UUID, SessionState] = field(default_factory=dict)
    history: dict[uuid.UUID, list[ChatTurn]] = field(default_factory=dict)
    records: list[TurnRecord] = field(default_factory=list)

    def seed(
        self, session_id: uuid.UUID, snapshot: dict[str, Any] | None = None, soft_ask_count: int = 0
    ) -> None:
        self.states[session_id] = SessionState(
            snapshot=dict(snapshot or {}), soft_ask_count=soft_ask_count
        )

    async def load_session_state(self, db: Any, session_id: uuid.UUID) -> SessionState:
        state = self.states.get(session_id)
        if state is None:
            return SessionState()
        return SessionState(snapshot=dict(state.snapshot), soft_ask_count=state.soft_ask_count)

    async def get_recent_turns(self, db: Any, session_id: uuid.UUID, n: int) -> list[ChatTurn]:
        return list(self.history.get(session_id, []))[-n:] if n > 0 else []

    async def get_last_user_message(
        self, db: Any, session_id: uuid.UUID
    ) -> tuple[uuid.UUID, str] | None:
        for turn in reversed(self.history.get(session_id, [])):
            if turn.role == "user":
                return uuid.uuid4(), turn.content
        return None

    async def persist_turn(self, db: Any, record: TurnRecord) -> tuple[uuid.UUID, uuid.UUID]:
        self.records.append(record)
        state = self.states.setdefault(record.session_id, SessionState())
        if record.new_snapshot is not None:
            state.snapshot = dict(record.new_snapshot)
        state.soft_ask_count = record.soft_ask_count
        turns = self.history.setdefault(record.session_id, [])
        turns.append(ChatTurn(role="user", content=record.user_message))
        turns.append(
            ChatTurn(
                role="assistant", content=record.assistant_message, action=str(record.final_action)
            )
        )
        return uuid.uuid4(), uuid.uuid4()

    def snapshot(self, session_id: uuid.UUID) -> dict[str, Any]:
        state = self.states.get(session_id)
        return dict(state.snapshot) if state else {}


def untrusted_text(user_prompt: str) -> str:
    """The (last) patient message inside ``<untrusted_user_input>`` of a prompt."""
    blocks = _UNTRUSTED.findall(user_prompt)
    return blocks[-1].strip() if blocks else ""


def _known_drug_surfaces() -> list[str]:
    from medbridge.nlp.drugs import BRAND_TO_GENERIC

    names = set(BRAND_TO_GENERIC) | set(GENERIC_ALIASES) | set(canonical_generics())
    return sorted(names, key=len, reverse=True)


class EchoExtractorLLM:
    """``llm.call`` stub for the Context Extractor.

    Copies drug names exactly as the patient wrote them into MEDICATION_ADDED
    events when the message says the patient takes the drug ("I take …");
    combination generics written as "x and y" become one event per ingredient
    (what an extraction model is instructed to do). The search query is the
    message itself. The real ``postprocess_extraction`` then normalises names
    and expands the query.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self._surfaces = _known_drug_surfaces()

    def _mentions(self, text: str) -> list[str]:
        """Drug names in ``text`` exactly as written (longest match wins, in order)."""
        lowered = text.lower()
        spans: list[tuple[int, int]] = []
        for surface in self._surfaces:
            pattern = rf"(?<![\w-]){re.escape(surface.lower())}(?![\w-])"
            for m in re.finditer(pattern, lowered):
                if not any(s < m.end() and m.start() < e for s, e in spans):
                    spans.append((m.start(), m.end()))
        return [text[s:e] for s, e in sorted(spans)]

    async def call(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        message = untrusted_text(user_prompt)
        self.calls.append({"call_name": call_name, "user_prompt": user_prompt})
        events: list[DeltaEvent] = []
        if re.match(r"\s*i (?:take|am taking|'m taking)\b", message, re.I):
            for name in self._mentions(message):
                events.append(
                    DeltaEvent(
                        event_type=EventTypeEnum.MEDICATION_ADDED,
                        payload={
                            "drug_name": name,
                            "dosage": "usual dose",
                            "frequency": "every morning",
                        },
                    )
                )
        return (
            output_schema(
                delta_events=events, search_query=message, raw_intent=f"Asks: {message[:120]}"
            )
            if output_schema is ExtractorOutput
            else None
        )


def canonical_generics() -> frozenset[str]:
    """Every canonical generic ingredient the normaliser can output."""
    from medbridge.nlp.drugs import BRAND_TO_GENERIC

    names = set(GENERIC_CLASS)
    for generic in [*BRAND_TO_GENERIC.values(), *GENERIC_ALIASES.values()]:
        names.update(part.strip() for part in generic.split("+"))
    return frozenset(names)


def is_canonical_generic(name: str) -> bool:
    """True when ``name`` is a canonical generic (or a ' + ' combination of generics)."""
    parts = [p.strip().lower() for p in str(name).split("+")]
    known = canonical_generics()
    return bool(parts) and all(p in known for p in parts)


class StubContextGate:
    """Gate 1 stub: PROCEED unless a recorded medicine is not a recognised generic."""

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(
        self, snapshot: dict[str, Any], message: str, raw_intent: str
    ) -> ContextGateOutput:
        self.calls += 1
        meds = [str(m.get("drug") or "") for m in snapshot.get("current_medications") or []]
        unknown = [m for m in meds if m and not is_canonical_generic(m)]
        if unknown:
            return ContextGateOutput(
                rationale="Unrecognised medicine name.",
                action="SOFT-ASK",
                missing_fields=["current_medications"],
            )
        return ContextGateOutput(rationale="Context sufficient.", action="PROCEED")


def stub_chunk(key: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"stub::{key}",
        chunk_text=f"Guideline evidence about {title}.",
        guideline_id="STUB_GUIDELINE",
        section_title=title,
        org="STUB",
        year=2025,
        doc_title="Deterministic stub guideline",
        score=1.0,
        score_rrf=1.0,
    )


class StubRetriever:
    """Returns one chunk per *generic* drug named literally in the (expanded) query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def hybrid_search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        chunks: list[RetrievedChunk] = []
        for drug in find_drugs(query):
            if drug.brand is None:
                chunks.append(stub_chunk(drug.generic, f"{drug.generic} ({drug.drug_class})"))
        return chunks or [stub_chunk("hypertension", "hypertension management")]

    async def warmup(self) -> None:
        return None


class PassthroughReranker:
    async def rerank_chunks(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        return list(chunks)[:top_k]

    async def warmup(self) -> None:
        return None


class StubEvidenceGate:
    """Gate 2 stub: ANSWER when a drug-specific chunk was retrieved, else GENERALIZE."""

    async def evaluate(
        self,
        chunks: list[RetrievedChunk],
        snapshot: dict[str, Any],
        query: str | None = None,
        message: str | None = None,
    ) -> EvidenceGateOutput:
        specific = any(c.chunk_id != "stub::hypertension" for c in chunks)
        return EvidenceGateOutput(
            rationale="stub",
            action="ANSWER" if specific else "GENERALIZE",
            evidence_sufficient=specific,
        )


class StubGenerator:
    """Generator stub: text built from the evidence only (never echoes the message)."""

    async def generate(
        self,
        action: Any,
        chunks: list[RetrievedChunk],
        snapshot: dict[str, Any],
        query: str | None = None,
        message: str | None = None,
        **kwargs: Any,
    ) -> ResponseGeneratorOutput:
        if not chunks:
            return ResponseGeneratorOutput(
                response_text="Could you tell me which medicines you take?"
            )
        body = " ".join(f"{c.section_title} [{i}]." for i, c in enumerate(chunks, 1))
        return ResponseGeneratorOutput(response_text=f"Guideline summary: {body}")


@dataclass
class RecordingLLM:
    """Captures every prompt and returns canned, schema-valid outputs."""

    gate1_action: str = "PROCEED"
    gate2_action: str = "ANSWER"
    response_text: str = "Here is what the guideline says [1]."
    calls: list[dict[str, str]] = field(default_factory=list)

    async def call(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {"call_name": call_name, "system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        if call_name == "context_extractor":
            message = untrusted_text(user_prompt)
            return output_schema(
                delta_events=[], search_query=message or "hypertension", raw_intent="question"
            )
        if call_name == "context_gate":
            missing = ["current_medications"] if self.gate1_action == "SOFT-ASK" else []
            return output_schema(
                rationale="canned", action=self.gate1_action, missing_fields=missing
            )
        if call_name == "evidence_gate":
            return output_schema(
                rationale="canned",
                action=self.gate2_action,
                evidence_sufficient=self.gate2_action == "ANSWER",
            )
        return output_schema(response_text=self.response_text)

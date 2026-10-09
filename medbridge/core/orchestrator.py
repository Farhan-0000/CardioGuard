"""
Pipeline Orchestrator — Central 8-Stage Runtime Coordinator (TASK-23, Module M-08).

Coordinates:
1. Load session state from DB (ADL-014)
2. Fast-path emergency check (M-07, ADL-015)
3. Context extraction (M-11) & state projection (M-10)
4. Context Gate evaluation (M-12) with loop-breaker counter management (ADL-013, Non-Negotiable #4)
5. Hybrid retrieval (M-16, ADL-022) & Cross-Encoder reranking (M-17)
6. Evidence Gate 4-way routing (M-13, ADL-024)
7. Response generation with citations (M-14, ADL-025) or deterministic templates (ADL-015)
8. Atomic database logging to audit_logs and message_history, with guaranteed non-throwing execution (Non-Negotiable #7).

Constitution §7 / Non-Negotiables #1–7.
Technical Specification Part III §3.1, §3.8, §3.9, Part IV §4.
"""
import copy
import structlog
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.context_extractor import ContextExtractor
from medbridge.ai.context_gate import ContextGate
from medbridge.ai.evidence_gate import EvidenceGate, get_evidence_gate
from medbridge.ai.response_generator import ResponseGenerator, get_response_generator
from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import CitationResponse, MessageResponse
from medbridge.config import Settings, get_settings
from medbridge.core.emergency_classifier import classify as classify_emergency
from medbridge.core.templates import get_loop_breaker_prefix, get_template
from medbridge.db.connection import get_sessionmaker
from medbridge.db.models import (
    AuditLog,
    FinalActionEnum,
    Gate1ActionEnum,
    Gate2ActionEnum,
)
from medbridge.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievedChunk,
    get_hybrid_retriever,
)
from medbridge.retrieval.reranker import CrossEncoderReranker, get_reranker
from medbridge.state.projector import StateProjector
from medbridge.state.session_manager import (
    SessionNotFoundError,
    increment_soft_ask_count,
    load_session_state,
    reset_soft_ask_count,
    save_message,
)

logger = structlog.get_logger(__name__)


class PipelineOrchestrator:
    """Central orchestrator coordinating the 8-stage clinical AI pipeline.

    Guarantees non-throwing execution and adherence to all constitutional
    safety invariants.
    """

    def __init__(
        self,
        emergency_classifier: Optional[Any] = None,
        context_extractor: Optional[ContextExtractor] = None,
        state_projector: Optional[StateProjector] = None,
        context_gate: Optional[ContextGate] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        evidence_gate: Optional[EvidenceGate] = None,
        response_generator: Optional[ResponseGenerator] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._emergency_classifier = emergency_classifier or classify_emergency
        self._context_extractor = context_extractor or ContextExtractor()
        self._state_projector = state_projector or StateProjector()
        self._context_gate = context_gate or ContextGate()
        self._hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self._reranker = reranker or get_reranker()
        self._evidence_gate = evidence_gate or get_evidence_gate()
        self._response_generator = response_generator or get_response_generator()

    async def process_message(
        self,
        session_id: Union[uuid.UUID, str],
        message: str,
        db: Optional[AsyncSession] = None,
    ) -> MessageResponse:
        """Process an incoming patient message through the 8-stage pipeline.

        Args:
            session_id: Session UUID or string representation.
            message: Raw patient message.
            db: Optional active async database session. If None, acquires
                session from connection pool and commits upon completion.

        Returns:
            MessageResponse containing the final action, response text, citations,
            and updated soft-ask count. Guaranteed never to raise unhandled 500s.
        """
        start_time = time.perf_counter()

        # Normalize session_id to UUID
        if isinstance(session_id, str):
            try:
                session_uuid = uuid.UUID(session_id)
            except ValueError:
                session_uuid = uuid.uuid4()
        else:
            session_uuid = session_id

        try:
            if db is not None:
                return await self._execute_pipeline(db, session_uuid, message, start_time)
            else:
                maker = get_sessionmaker()
                async with maker() as session:
                    result = await self._execute_pipeline(session, session_uuid, message, start_time)
                    await session.commit()
                    return result

        except Exception as e:
            # NON-NEGOTIABLE #7: Guaranteed non-throwing execution
            logger.error(
                "Unhandled error in pipeline orchestrator: %s",
                e,
                exc_info=True,
                extra={"session_id": str(session_uuid)},
            )
            return MessageResponse(
                session_id=session_uuid,
                response_text=(
                    "I apologize, but I encountered an unexpected error while processing your request. "
                    "Please consult your healthcare provider directly for personalized clinical guidance."
                ),
                action=ActionEnum.GENERALIZE,
                citations=[],
                soft_ask_count=0,
                timestamp=datetime.now(timezone.utc),
            )

    async def _execute_pipeline(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        message: str,
        start_time: float,
    ) -> MessageResponse:
        """Execute stages 1 through 8 within an active database session."""

        # ------------------------------------------------------------------
        # Stage 1: Load Session State from DB (ADL-014)
        # ------------------------------------------------------------------
        try:
            state = await load_session_state(db, session_id)
        except SessionNotFoundError:
            # If session record does not exist yet, initialize defaults
            state = {"snapshot": {}, "soft_ask_count": 0}

        current_snapshot: dict[str, Any] = state.get("snapshot", {})
        soft_ask_count: int = state.get("soft_ask_count", 0)

        # ------------------------------------------------------------------
        # Stage 2: Fast-Path Emergency Check (M-07, ADL-015)
        # ------------------------------------------------------------------
        if hasattr(self._emergency_classifier, "classify"):
            emergency_action = self._emergency_classifier.classify(message)
        elif callable(self._emergency_classifier):
            emergency_action = self._emergency_classifier(message)
        else:
            emergency_action = None
        if emergency_action == ActionEnum.ESCALATE:
            logger.warning(
                "Emergency detected on fast-path: triggering immediate ESCALATE bypass",
                extra={"session_id": str(session_id)},
            )
            escalate_text = get_template(ActionEnum.ESCALATE)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Record message history
            await save_message(db, session_id, role="user", content=message)
            await save_message(
                db,
                session_id,
                role="assistant",
                content=escalate_text,
                action=ActionEnum.ESCALATE,
                citations=[],
            )

            # Record audit log
            audit = AuditLog(
                audit_id=uuid.uuid4(),
                session_id=session_id,
                request_message=message,
                gate_1_action=None,
                gate_1_rationale=None,
                gate_2_action=Gate2ActionEnum.ESCALATE,
                gate_2_rationale="Emergency classifier fast-path trigger",
                final_action=FinalActionEnum.ESCALATE,
                evidence_chunk_ids=[],
                response_text=escalate_text,
                latency_ms=elapsed_ms,
                created_at=datetime.now(timezone.utc),
            )
            db.add(audit)
            await db.flush()

            return MessageResponse(
                session_id=session_id,
                response_text=escalate_text,
                action=ActionEnum.ESCALATE,
                citations=[],
                soft_ask_count=soft_ask_count,
                timestamp=datetime.now(timezone.utc),
            )

        # ------------------------------------------------------------------
        # Stage 3: Context Extractor & State Projection (M-11, M-10)
        # ------------------------------------------------------------------
        extractor_output = await self._context_extractor.extract(message, current_snapshot)
        if extractor_output is None:
            # Fallback per §3.9: Skip extraction, pass raw query with empty delta events
            logger.warning(
                "Context Extractor failure: applying fallback to raw message",
                extra={"session_id": str(session_id)},
            )
            delta_events = []
            search_query = message
            raw_intent = message
        else:
            delta_events = extractor_output.delta_events
            search_query = extractor_output.search_query
            raw_intent = extractor_output.raw_intent

        # Update clinical events and snapshot in DB atomically
        updated_snapshot = await self._state_projector.persist_and_project(
            db,
            session_id,
            delta_events,
            current_snapshot,
        )

        # ------------------------------------------------------------------
        # Stage 4: Context Gate (LLM Call 2) & Loop-Breaker (M-12, ADL-013, ADL-023)
        # ------------------------------------------------------------------
        gate_1_output = await self._context_gate.evaluate(updated_snapshot, message, raw_intent)
        if gate_1_output is None:
            # Fallback per §3.9: Default to PROCEED
            logger.warning(
                "Context Gate failure: defaulting to PROCEED fallback",
                extra={"session_id": str(session_id)},
            )
            gate_1_action = "PROCEED"
            gate_1_rationale = "Context Gate LLM permanent failure fallback to PROCEED"
        else:
            gate_1_action = gate_1_output.action
            gate_1_rationale = gate_1_output.rationale

        force_generalize = False

        if gate_1_action == "SOFT-ASK":
            # Check loop-breaker threshold (NON-NEGOTIABLE #4, ADL-013)
            if soft_ask_count < self._settings.SOFT_ASK_MAX_COUNT:
                # Increment counter and short-circuit to generate clarifying question
                new_soft_ask_count = await increment_soft_ask_count(db, session_id)
                gen_output = await self._response_generator.generate(
                    action=ActionEnum.SOFT_ASK,
                    chunks=[],
                    snapshot=updated_snapshot,
                    query=message,
                )

                if gen_output is not None and gen_output.response_text:
                    soft_ask_text = gen_output.response_text
                else:
                    # Fallback clarifying question
                    missing_fields = gate_1_output.missing_fields if gate_1_output else []
                    if missing_fields:
                        missing_desc = ", ".join(missing_fields)
                        soft_ask_text = (
                            f"To give you the most accurate and safe clinical guidance, could you please "
                            f"provide some additional details regarding your {missing_desc}?"
                        )
                    else:
                        soft_ask_text = (
                            "To give you personalized clinical guidance, could you please share a bit more "
                            "context about your blood pressure readings, medications, or symptoms?"
                        )

                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                # Persist message history
                await save_message(db, session_id, role="user", content=message)
                await save_message(
                    db,
                    session_id,
                    role="assistant",
                    content=soft_ask_text,
                    action=ActionEnum.SOFT_ASK,
                    citations=[],
                )

                # Persist audit log
                audit = AuditLog(
                    audit_id=uuid.uuid4(),
                    session_id=session_id,
                    request_message=message,
                    gate_1_action=Gate1ActionEnum.SOFT_ASK,
                    gate_1_rationale=gate_1_rationale,
                    gate_2_action=None,
                    gate_2_rationale=None,
                    final_action=FinalActionEnum.SOFT_ASK,
                    evidence_chunk_ids=[],
                    response_text=soft_ask_text,
                    latency_ms=elapsed_ms,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(audit)
                await db.flush()

                return MessageResponse(
                    session_id=session_id,
                    response_text=soft_ask_text,
                    action=ActionEnum.SOFT_ASK,
                    citations=[],
                    soft_ask_count=new_soft_ask_count,
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                # Loop-breaker triggered: soft_ask_count >= 2
                logger.info(
                    "SOFT-ASK loop-breaker triggered (soft_ask_count=%d): forcing GENERALIZE",
                    soft_ask_count,
                    extra={"session_id": str(session_id)},
                )
                force_generalize = True
                await reset_soft_ask_count(db, session_id)
                soft_ask_count = 0

        elif gate_1_action == "PROCEED":
            # Context sufficient: reset soft-ask counter
            await reset_soft_ask_count(db, session_id)
            soft_ask_count = 0

        # ------------------------------------------------------------------
        # Stage 5: Hybrid Retrieval & Cross-Encoder Reranking (M-16, M-17)
        # ------------------------------------------------------------------
        retrieval_query = search_query if search_query and search_query.strip() else message
        candidate_chunks = await self._hybrid_retriever.hybrid_search(
            retrieval_query,
            top_k=self._settings.RETRIEVAL_TOP_K_CANDIDATES,
        )

        top_chunks: list[RetrievedChunk] = []
        if candidate_chunks:
            chunk_map = {c.chunk_text: c for c in candidate_chunks}
            chunk_texts = list(chunk_map.keys())
            reranked_tuples = await self._reranker.rerank(
                retrieval_query,
                chunk_texts,
                top_k=self._settings.RETRIEVAL_TOP_K_RERANKED,
            )
            for text, score in reranked_tuples:
                matching = chunk_map.get(text)
                if matching:
                    chunk_copy = (
                        matching.model_copy()
                        if hasattr(matching, "model_copy")
                        else copy.deepcopy(matching)
                    )
                    chunk_copy.score = score
                    top_chunks.append(chunk_copy)
                else:
                    top_chunks.append(
                        RetrievedChunk(
                            chunk_id=str(uuid.uuid4()),
                            chunk_text=text,
                            guideline_id="AHA_ACC",
                            section_title="General Guidance",
                            score=score,
                        )
                    )

        # ------------------------------------------------------------------
        # Stage 6: Evidence Gate (LLM Call 3) (M-13, ADL-024)
        # ------------------------------------------------------------------
        if force_generalize:
            gate_2_action = "GENERALIZE"
            gate_2_rationale = (
                f"Forced GENERALIZE due to loop-breaker threshold (soft_ask_count={soft_ask_count})"
            )
        else:
            gate_2_output = await self._evidence_gate.evaluate(
                chunks=top_chunks,
                snapshot=updated_snapshot,
                query=message,
            )
            if gate_2_output is None:
                # Fallback per §3.9: Default to GENERALIZE
                logger.warning(
                    "Evidence Gate failure: defaulting to GENERALIZE fallback",
                    extra={"session_id": str(session_id)},
                )
                gate_2_action = "GENERALIZE"
                gate_2_rationale = "Evidence Gate LLM permanent failure fallback to GENERALIZE"
            else:
                gate_2_action = gate_2_output.action
                gate_2_rationale = gate_2_output.rationale

        # ------------------------------------------------------------------
        # Stage 7: Response Generation & Routing (M-14, M-09, ADL-015, ADL-025)
        # ------------------------------------------------------------------
        citations_response: list[CitationResponse] = []

        if gate_2_action == "ABSTAIN":
            final_action = ActionEnum.ABSTAIN
            response_text = get_template(ActionEnum.ABSTAIN)

        elif gate_2_action == "ESCALATE":
            final_action = ActionEnum.ESCALATE
            response_text = get_template(ActionEnum.ESCALATE)

        else:
            # ANSWER or GENERALIZE
            final_action = ActionEnum.ANSWER if gate_2_action == "ANSWER" else ActionEnum.GENERALIZE
            gen_output = await self._response_generator.generate(
                action=final_action,
                chunks=top_chunks,
                snapshot=updated_snapshot,
                query=message,
            )

            if gen_output is None or not gen_output.response_text:
                # Fallback per §3.9: Pre-vetted GENERALIZE template
                logger.warning(
                    "Response Generator failure: returning pre-vetted GENERALIZE template",
                    extra={"session_id": str(session_id)},
                )
                final_action = ActionEnum.GENERALIZE
                response_text = (
                    "Based on general clinical guidelines for hypertension, blood pressure management "
                    "involves a combination of non-pharmacologic interventions (such as sodium restriction, "
                    "regular physical exercise, and stress reduction) and guideline-directed pharmacotherapy. "
                    "Please consult your primary care physician to tailor these guidelines to your personal health profile."
                )
            else:
                response_text = gen_output.response_text
                citations_response = [
                    CitationResponse(
                        marker=c.marker,
                        chunk_id=c.chunk_id,
                        source=c.source,
                        section=c.section,
                        excerpt=c.excerpt,
                    )
                    for c in gen_output.citations
                ]

            # Prepend loop-breaker prefix if loop-breaker was triggered (Appendix B.3, Non-Negotiable #4)
            if force_generalize:
                prefix = get_loop_breaker_prefix()
                if not response_text.startswith(prefix):
                    response_text = f"{prefix}\n\n{response_text}"

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # ------------------------------------------------------------------
        # Stage 8: Atomic Persistence & Return (M-05, ADL-014)
        # ------------------------------------------------------------------
        citations_dicts = [c.model_dump() for c in citations_response]

        # Record user message
        await save_message(db, session_id, role="user", content=message)

        # Record assistant response
        await save_message(
            db,
            session_id,
            role="assistant",
            content=response_text,
            action=final_action,
            citations=citations_dicts,
        )

        # Record audit log
        gate_1_enum = Gate1ActionEnum(gate_1_action) if gate_1_action in ("SOFT-ASK", "PROCEED") else None
        gate_2_enum = (
            Gate2ActionEnum(gate_2_action)
            if gate_2_action in ("ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE")
            else None
        )

        audit = AuditLog(
            audit_id=uuid.uuid4(),
            session_id=session_id,
            request_message=message,
            gate_1_action=gate_1_enum,
            gate_1_rationale=gate_1_rationale,
            gate_2_action=gate_2_enum,
            gate_2_rationale=gate_2_rationale,
            final_action=FinalActionEnum(final_action.value),
            evidence_chunk_ids=[c.chunk_id for c in top_chunks] if top_chunks else [],
            response_text=response_text,
            latency_ms=elapsed_ms,
            created_at=datetime.now(timezone.utc),
        )
        db.add(audit)
        await db.flush()

        logger.info(
            "Pipeline execution complete",
            extra={
                "session_id": str(session_id),
                "final_action": final_action.value,
                "latency_ms": elapsed_ms,
                "citations_count": len(citations_response),
            },
        )

        return MessageResponse(
            session_id=session_id,
            response_text=response_text,
            action=final_action,
            citations=citations_response,
            soft_ask_count=soft_ask_count,
            timestamp=datetime.now(timezone.utc),
        )


def get_orchestrator() -> PipelineOrchestrator:
    """Convenience factory function for dependency injection."""
    return PipelineOrchestrator()

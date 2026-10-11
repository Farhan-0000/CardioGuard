"""
Pipeline Orchestrator — the Dual Answerability Gate runtime (Module M-08).

One patient turn flows through explicit stage methods operating on a
:class:`TurnContext` (contract ``docs/finalization_plan.md`` §3.1):

0. privacy        PII/PHI redaction; the redacted text is used by every stage
                  below except the emergency classifier
1. emergency      deterministic fast-path on the RAW text, before any LLM call
                  (Non-Negotiable #1) → ESCALATE template
2. load_state     snapshot + soft-ask counter + recent turns (turn store); when
                  the previous answer was a SOFT-ASK, the "effective patient
                  message" (original question + follow-up details) is built
2b small_talk     deterministic, claim-free GENERALIZE template (no LLM)
3. extraction     ContextExtractor → delta events → pure in-memory projection
4. context_gate   Gate 1: SOFT-ASK | PROCEED, loop-breaker at 2 (Non-Negotiable #4);
                  retrieval may start speculatively alongside Gate 1 and is
                  discarded on SOFT-ASK (Gate 1 still fully controls its use)
5. retrieval      hybrid dense+sparse → RRF(k=60) top-20 → cross-encoder top-5
6. evidence_gate  Gate 2: ANSWER | GENERALIZE | ABSTAIN | ESCALATE (V1 optional)
7. generation     templates for ABSTAIN/ESCALATE (Non-Negotiable #6); otherwise
                  the generator, citation validation (ANSWER with no valid
                  citation → GENERALIZE) and the optional Gate 3 grounding verifier
8. persistence    ONE ``turn_store.persist_turn`` call (single flush)

Spec fallbacks are unchanged: extractor failure → raw message and no events;
Gate 1 failure → PROCEED; Gate 2 failure (incl. every V1 vote failing) →
GENERALIZE; generator failure → pre-vetted GENERALIZE template; retrieval
failure → empty evidence, ``retrieval_status="failed"`` and Gate 2 still
decides. :meth:`PipelineOrchestrator.process_message` never raises
(Non-Negotiable #7).

Multi-turn follow-ups: a reply to a clarifying question ("I had a stroke two
years ago") is not a question on its own. When the previous answer was a
SOFT-ASK (one or more in a row — the loop-breaker sequence), Gate 1, Gate 2
and the generator receive an *effective patient message*: the redacted
question that triggered the first SOFT-ASK plus every follow-up since, the
current one last. The emergency classifier and the extractor still see only
the current message; the persisted user message is the current one; the trace
records only counts (``explanation.effective_question``), never text.

Concurrency: turns of one session are serialised in-process (one asyncio lock
per session, held from state loading to persistence) so two turns never gate
on the same stale counter or snapshot. A message that triggers the emergency
fast-path is never queued behind another turn: it is classified before the
lock and, on ESCALATE, answered and stored without waiting (it changes neither
the snapshot nor the counter). Across processes the turn store's row lock and
re-projection keep ``snapshot == fold(events)``.

Regenerate re-runs the last user message starting from the loop-breaker
counter recorded *before* that turn (``explanation.soft_ask_count_before``),
so repeated regenerates are idempotent and never trip or undo the loop-breaker.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import ModuleType
from typing import Any, Literal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.context_extractor import ContextExtractor
from medbridge.ai.context_gate import ContextGate
from medbridge.ai.evidence_gate import EvidenceGate, get_evidence_gate
from medbridge.ai.llm_wrapper import get_llm_wrapper
from medbridge.ai.response_generator import ResponseGenerator, get_response_generator
from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    CitationResponse,
    Explanation,
    GateDecision,
    MessageResponse,
    RetrievedEvidence,
    StageTiming,
)
from medbridge.config import Settings, get_settings
from medbridge.core import emergency_classifier as default_emergency_classifier
from medbridge.core.citations import (
    display_page,
    display_section,
    make_excerpt,
    validate_citations,
)
from medbridge.core.pipeline_types import (
    STAGE_LABELS,
    AblationConfig,
    EventCallback,
    PipelineEvent,
    StageName,
    StageStatus,
)
from medbridge.core.smalltalk import is_small_talk
from medbridge.core.templates import (
    GENERALIZE_FALLBACK_TEMPLATE,
    PIPELINE_ERROR_TEMPLATE,
    build_soft_ask_fallback,
    get_loop_breaker_prefix,
    get_small_talk_template,
    get_template,
)
from medbridge.db.connection import get_sessionmaker
from medbridge.nlp.query_expansion import expand_query
from medbridge.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievedChunk,
    get_hybrid_retriever,
)
from medbridge.retrieval.reranker import CrossEncoderReranker, get_reranker
from medbridge.state import turn_store
from medbridge.state.projector import StateProjector
from medbridge.state.turn_store import (
    ChatTurn,
    RegenerateConflictError,
    SessionState,
    TurnRecord,
)

logger = structlog.get_logger(__name__)

_GATE2_ACTIONS = frozenset({"ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE"})
# Gate sources that do not represent a decision taken by the gate itself; the
# audit-log gate columns stay NULL for them (the trace still records them).
_NON_DECISION_SOURCES = frozenset({"ablation", "rule"})
_EVIDENCE_EXCERPT_CHARS = 300
ORIGINAL_QUESTION_LABEL = "Original question:"
FOLLOW_UP_LABEL = "Follow-up details from the patient:"

TimingStatus = Literal["ok", "skipped", "failed", "discarded"]


@dataclass
class RetrievalOutcome:
    """Result and timing of one retrieval (+ rerank) run."""

    chunks: list[RetrievedChunk]
    candidates: int
    retrieval_ms: float
    rerank_ms: float
    retrieval_status: TimingStatus
    rerank_status: TimingStatus
    started_at: float
    finished_at: float


@dataclass
class TurnContext:
    """Mutable state of one pipeline turn, threaded through every stage."""

    session_id: uuid.UUID
    raw_message: str
    ablation: AblationConfig
    regenerate: bool = False
    on_event: EventCallback | None = None
    started_at: float = field(default_factory=time.perf_counter)
    explanation: Explanation = field(default_factory=Explanation)
    llm_calls: list[Any] = field(default_factory=list)
    message: str = ""  # redacted text used by every stage except emergency
    snapshot: dict[str, Any] = field(default_factory=dict)
    new_snapshot: dict[str, Any] | None = None
    soft_ask_count: int = 0
    history: list[ChatTurn] = field(default_factory=list)
    # Effective patient message (original SOFT-ASK question + follow-ups) — see module doc.
    earlier_messages: list[str] = field(default_factory=list)
    effective_message: str = ""
    counter_updated: bool = False  # Gate 1 ran (or was bypassed) and set soft_ask_count
    failed_votes: int | None = None  # V1: evidence-gate votes that failed
    delta_events: list[DeltaEvent] = field(default_factory=list)
    search_query: str = ""
    raw_intent: str = ""
    loop_breaker: bool = False
    speculative: asyncio.Task[RetrievalOutcome] | None = None
    speculative_launched_at: float = 0.0
    gate1_window: tuple[float, float] = (0.0, 0.0)
    retrieval_recorded: bool = False
    chunks: list[RetrievedChunk] = field(default_factory=list)
    final_action: ActionEnum | None = None
    response_text: str = ""
    draft: str | None = None  # generated ANSWER/GENERALIZE text awaiting validation
    citations: list[CitationResponse] = field(default_factory=list)
    complete: bool = False  # response fixed by a deterministic path (fast-path, small talk)
    regenerated_user_message_id: uuid.UUID | None = None
    user_message_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    latency_ms: int = 0

    @property
    def gate_snapshot(self) -> dict[str, Any]:
        """Snapshot after this turn's (in-memory) projection."""
        return self.new_snapshot if self.new_snapshot is not None else self.snapshot

    @property
    def patient_message(self) -> str:
        """What Gate 1, Gate 2 and the generator read as the patient's message."""
        return self.effective_message or self.message

    @property
    def carried_text(self) -> str:
        """Earlier SOFT-ASK question(s) and this message as plain text (fallback query)."""
        return " ".join([*self.earlier_messages, self.message])


@dataclass
class _StageRun:
    status: TimingStatus = "ok"


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _coerce_session_id(session_id: uuid.UUID | str) -> uuid.UUID:
    if isinstance(session_id, uuid.UUID):
        return session_id
    try:
        return uuid.UUID(str(session_id))
    except ValueError:
        return uuid.uuid4()


def _as_action(value: Any) -> ActionEnum | None:
    if value is None:
        return None
    try:
        return ActionEnum(value)
    except ValueError:
        return None


def _enter_llm_call_log(stack: contextlib.ExitStack) -> list[Any]:
    """Enter the per-turn LLM call recorder (``ai.llm_wrapper.llm_call_log``).

    Tracing must never break a turn: if the recorder is unavailable the turn
    simply reports no LLM call statistics.
    """
    try:
        from medbridge.ai.llm_wrapper import llm_call_log

        calls: list[Any] = stack.enter_context(llm_call_log())
    except Exception as exc:
        logger.warning("llm_call_log_unavailable", error=type(exc).__name__)
        return []
    return calls


def _serialise_call(call: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(call) and not isinstance(call, type):
        return dataclasses.asdict(call)
    if isinstance(call, dict):
        return dict(call)
    if hasattr(call, "model_dump"):
        dumped: dict[str, Any] = call.model_dump()
        return dumped
    return {"call": str(call)}


def _trim_history(history: list[ChatTurn], regenerate: bool, n: int) -> list[ChatTurn]:
    """Last ``n`` turns; for a regeneration drop everything from the re-run user message on."""
    turns = list(history)
    if regenerate:
        while turns:
            if turns.pop().role == "user":
                break
    return turns[-n:] if n > 0 else []


def _is_soft_ask(action: Any) -> bool:
    text = str(getattr(action, "value", action) or "").strip().upper().replace("_", "-")
    return text == ActionEnum.SOFT_ASK.value


def soft_ask_chain(history: Sequence[ChatTurn], limit: int) -> list[str]:
    """User messages of the SOFT-ASK exchange the next message answers (oldest first).

    Each user message is paired with the latest assistant reply to it (a
    regenerated reply supersedes earlier ones). Walking back from the most
    recent pair, messages are collected while their reply was a SOFT-ASK, at
    most ``limit`` of them. ``[]`` when the previous reply was not a SOFT-ASK.
    """
    pairs: list[tuple[str, Any]] = []
    for turn in history:
        if turn.role == "user":
            pairs.append((turn.content, None))
        elif turn.role == "assistant" and pairs:
            pairs[-1] = (pairs[-1][0], turn.action)
    chain: list[str] = []
    for content, action in reversed(pairs):
        if len(chain) >= limit or not _is_soft_ask(action):
            break
        chain.append(content)
    return list(reversed(chain))


def compose_effective_message(earlier: Sequence[str], current: str) -> str:
    """Original question + follow-up details, one labelled line each (current last)."""
    lines = [f"{ORIGINAL_QUESTION_LABEL} {earlier[0]}"]
    lines += [f"{FOLLOW_UP_LABEL} {text}" for text in [*earlier[1:], current]]
    return "\n".join(lines)


class SessionTurnLocks:
    """One asyncio lock per session: turns of a session run one at a time in-process.

    Locks are created on demand and dropped as soon as no turn holds or waits
    for them, so the map stays small and no lock outlives its event loop.
    """

    def __init__(self) -> None:
        self._locks: dict[uuid.UUID, asyncio.Lock] = {}
        self._users: dict[uuid.UUID, int] = {}

    @asynccontextmanager
    async def hold(self, session_id: uuid.UUID) -> AsyncIterator[float]:
        """Hold the session's lock; yields the milliseconds spent waiting for it."""
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        self._users[session_id] = self._users.get(session_id, 0) + 1
        started = time.perf_counter()
        try:
            async with lock:
                yield (time.perf_counter() - started) * 1000
        finally:
            remaining = self._users[session_id] - 1
            if remaining:
                self._users[session_id] = remaining
            else:
                del self._users[session_id]
                self._locks.pop(session_id, None)


def _evidence(rank: int, chunk: RetrievedChunk) -> RetrievedEvidence:
    return RetrievedEvidence(
        rank=rank,
        chunk_id=chunk.chunk_id,
        guideline_id=chunk.guideline_id or "",
        title=chunk.doc_title or "",
        org=chunk.org or "",
        year=chunk.year,
        section=display_section(chunk),
        page=display_page(chunk),
        score_rrf=float(chunk.score_rrf or 0.0),
        score_rerank=float(chunk.score_rerank or 0.0),
        excerpt=make_excerpt(chunk.chunk_text, _EVIDENCE_EXCERPT_CHARS),
    )


def _overlap_ms(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * 1000.0


def _gate3_record(result: Any) -> dict[str, Any]:
    """Numeric/status summary of a ``GroundingResult`` — never sentences or text (PHI-free)."""
    available = bool(getattr(result, "available", True))
    return {
        "status": "ok" if available else "unavailable",
        "available": available,
        "n_claims": int(getattr(result, "n_claims", 0) or 0),
        "n_unsupported": int(getattr(result, "n_unsupported", 0) or 0),
        "unsupported_fraction": float(getattr(result, "unsupported_fraction", 0.0) or 0.0),
    }


def _emergency_record(action: ActionEnum | None, result: Any) -> dict[str, Any]:
    """Audit view of the emergency classification.

    Records rule ids (fired, negated, suppressed) and the decisive BP reading;
    never ``evidence_spans`` — they are fragments of the patient's own text.
    """
    bp = getattr(result, "bp", None)
    reading: list[int] | None = None
    if isinstance(bp, tuple | list) and len(bp) == 2:
        try:
            reading = [int(bp[0]), int(bp[1])]
        except (TypeError, ValueError):
            reading = None

    def _ids(name: str) -> list[str]:
        return [str(rule) for rule in (getattr(result, name, None) or [])]

    return {
        "action": action.value if action else None,
        "matched_rules": _ids("matched_rules"),
        "negated": _ids("negated"),
        "suppressed": _ids("suppressed"),
        "bp": reading,
    }


def _gate3_exceeds(record: dict[str, Any], tau: float) -> bool:
    """True when Gate 3 must not deliver the (cleaned) draft.

    Either the unsupported fraction is above ``tau``, or every checked claim is unsupported:
    removing all of them would leave no content, and delivering the original draft would
    pass exactly the claims the verifier rejected (ADL-061). An unavailable verifier never
    withdraws a draft.
    """
    if not record.get("available", True):
        return False
    n_claims = int(record.get("n_claims", 0) or 0)
    fully_unsupported = n_claims > 0 and int(record.get("n_unsupported", 0) or 0) >= n_claims
    return fully_unsupported or float(record.get("unsupported_fraction", 0.0)) > tau


def _cleaned_text(result: Any, fallback: str) -> str:
    cleaned = getattr(result, "cleaned_text", None)
    if isinstance(cleaned, str) and cleaned.strip():
        return cleaned.strip()
    return fallback


class PipelineOrchestrator:
    """Coordinates the guarded pipeline; stateless across turns (safe to share)."""

    def __init__(
        self,
        emergency_classifier: Any | None = None,
        context_extractor: ContextExtractor | None = None,
        state_projector: StateProjector | None = None,
        context_gate: ContextGate | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        evidence_gate: EvidenceGate | None = None,
        response_generator: ResponseGenerator | None = None,
        settings: Settings | None = None,
        *,
        store: ModuleType | Any | None = None,
        redactor: Callable[[str], Any] | None = None,
        grounding_verifier: Any | None = None,
        smalltalk_detector: Callable[[str], str | None] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._emergency_classifier = (
            emergency_classifier
            if emergency_classifier is not None
            else default_emergency_classifier
        )
        self._context_extractor = context_extractor or ContextExtractor()
        self._state_projector = state_projector or StateProjector()
        self._context_gate = context_gate or ContextGate()
        self._hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self._reranker = reranker or get_reranker()
        self._evidence_gate = evidence_gate or get_evidence_gate()
        self._response_generator = response_generator or get_response_generator()
        self._store = store if store is not None else turn_store
        self._redactor = redactor
        self._grounding_verifier = grounding_verifier
        self._is_small_talk = smalltalk_detector or is_small_talk
        self._turn_locks = SessionTurnLocks()

    # ==================================================================
    # Public API
    # ==================================================================

    async def process_message(
        self,
        session_id: uuid.UUID | str,
        message: str,
        db: AsyncSession | None = None,
        *,
        on_event: EventCallback | None = None,
        ablation: AblationConfig | None = None,
        regenerate: bool = False,
    ) -> MessageResponse:
        """Run one patient turn through the pipeline. Never raises.

        Args:
            session_id: Session UUID (or its string form).
            message: Raw patient message (ignored when ``regenerate`` finds a stored message).
            db: Active session; when ``None`` the orchestrator opens and commits its own.
            on_event: Async callback receiving stage progress events (SSE).
            ablation: Evaluation-only pipeline switches; production uses ``AblationConfig.full()``.
            regenerate: Re-run the last stored user message without persisting new
                clinical events or a new user message.
        """
        ctx = TurnContext(
            session_id=_coerce_session_id(session_id),
            raw_message=message,
            ablation=ablation or AblationConfig.full(),
            regenerate=regenerate,
            on_event=on_event,
        )
        ctx.explanation.ablation = ctx.ablation.name
        ctx.explanation.regenerated = regenerate
        ctx.explanation.models = {
            "gates": self._settings.gates_model,
            "generator": self._settings.generator_model,
        }
        try:
            with contextlib.ExitStack() as stack:
                ctx.llm_calls = _enter_llm_call_log(stack)
                try:
                    await self._run_turn(ctx, db)
                    return self._build_response(ctx)
                except Exception as exc:  # Non-Negotiable #7: never throw
                    logger.error(
                        "pipeline_unhandled_error",
                        session_id=str(ctx.session_id),
                        error=type(exc).__name__,
                        exc_info=True,
                    )
                    await self._discard_speculative(ctx)
                    return self._error_response(ctx, exc)
        finally:
            if ctx.speculative is not None and not ctx.speculative.done():
                ctx.speculative.cancel()

    async def warmup(self) -> dict[str, Any]:
        """Load every model the pipeline needs so the first patient query is fast.

        Runs the retrieval side (embedders, Qdrant, reranker) concurrently with
        the LLM warm-up. Returns readiness flags (never raises).
        """
        started = time.perf_counter()
        readiness: dict[str, Any] = {}

        async def _attempt(name: str, start: Callable[[], Awaitable[Any]]) -> tuple[bool, Any]:
            try:
                return True, await start()
            except Exception as exc:
                logger.warning("warmup_failed", component=name, error=type(exc).__name__)
                return False, None

        async def _retrieval_side() -> None:
            embedder = getattr(self._hybrid_retriever, "embedder", None)
            if embedder is not None:
                _, state = await _attempt("embedder", lambda: asyncio.to_thread(embedder.warmup))
                loaded = state if isinstance(state, dict) else {}
                readiness["embedder_dense"] = bool(loaded.get("dense"))
                readiness["embedder_sparse"] = bool(loaded.get("sparse"))
            readiness["retriever"], _ = await _attempt(
                "retriever", lambda: self._hybrid_retriever.warmup()
            )
            readiness["reranker"], _ = await _attempt("reranker", lambda: self._reranker.warmup())

        async def _llm_side() -> None:
            models = list(
                dict.fromkeys([self._settings.gates_model, self._settings.generator_model])
            )
            ok, result = await _attempt("llm", lambda: get_llm_wrapper().warmup(models))
            loaded = result if isinstance(result, dict) else {}
            # An empty result means a remote provider: nothing to load, nothing failed.
            readiness["llm"] = ok and all(bool(v) for v in loaded.values())
            readiness["llm_models"] = {str(k): bool(v) for k, v in loaded.items()}

        async def _verifier_side() -> None:
            if not self._settings.GROUNDING_VERIFIER_ENABLED:
                return

            async def _load_verifier() -> Any:
                # GroundingVerifier.warmup() loads the NLI model and runs one
                # inference; it returns False when the model cannot be loaded.
                verifier = await asyncio.to_thread(self._get_verifier)
                warm = getattr(verifier, "warmup", None)
                if inspect.iscoroutinefunction(warm):
                    return await warm()
                if callable(warm):
                    return await asyncio.to_thread(warm)
                return None

            ok, result = await _attempt("nli_verifier", _load_verifier)
            usable = result if isinstance(result, bool) else True
            readiness["nli_verifier"] = ok and usable

        await asyncio.gather(_retrieval_side(), _llm_side(), _verifier_side())
        readiness["warmup_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("pipeline_warmup_complete", **readiness)
        return readiness

    # ==================================================================
    # Turn driver
    # ==================================================================

    async def _run_turn(self, ctx: TurnContext, db: AsyncSession | None) -> None:
        if not ctx.regenerate:
            await self._stage_privacy(ctx)
            await self._stage_emergency(ctx)
            if ctx.complete:
                # Emergency fast-path: never queued behind another turn of this
                # session. It changes neither the snapshot nor the counter.
                await self._run_stages(ctx, db)
                return
        async with self._turn_locks.hold(ctx.session_id) as waited_ms:
            if waited_ms >= 1.0:
                logger.info(
                    "session_turn_serialised",
                    session_id=str(ctx.session_id),
                    waited_ms=round(waited_ms, 1),
                )
            if ctx.regenerate:  # resolve under the lock: the latest stored message
                await self._resolve_regenerated_message(ctx, db)
                await self._stage_privacy(ctx)
                await self._stage_emergency(ctx)
            await self._run_stages(ctx, db)

    async def _run_stages(self, ctx: TurnContext, db: AsyncSession | None) -> None:
        await self._stage_load_state(ctx, db)
        await self._stage_small_talk(ctx)
        await self._stage_extraction(ctx)
        await self._stage_context_gate(ctx)
        await self._stage_retrieval(ctx)
        await self._stage_evidence_gate(ctx)
        await self._stage_generation(ctx)
        await self._stage_verification(ctx)
        self._finalise_response_text(ctx)
        await self._stage_persistence(ctx, db)
        logger.info(
            "turn_complete",
            session_id=str(ctx.session_id),
            action=ctx.final_action.value if ctx.final_action else None,
            latency_ms=ctx.latency_ms,
            ablation=ctx.ablation.name,
            citations=len(ctx.citations),
        )

    @asynccontextmanager
    async def _session_scope(self, db: AsyncSession | None) -> AsyncIterator[AsyncSession]:
        """Use the caller's session, or open a short-lived one that commits on success."""
        if db is not None:
            yield db
            return
        async with get_sessionmaker()() as session:
            yield session
            await session.commit()

    # ------------------------------------------------------------------
    # Stage bookkeeping (timings + progress events)
    # ------------------------------------------------------------------

    async def _emit(
        self,
        ctx: TurnContext,
        stage: StageName,
        status: StageStatus,
        data: dict[str, Any] | None = None,
    ) -> None:
        if ctx.on_event is None:
            return
        event = PipelineEvent(
            type="stage", stage=stage, status=status, label=STAGE_LABELS[stage], data=data or {}
        )
        try:
            await ctx.on_event(event)
        except Exception as exc:  # a broken client callback never breaks the turn
            logger.warning("stage_event_callback_failed", error=type(exc).__name__)

    @staticmethod
    def _record(ctx: TurnContext, stage: StageName, ms: float, status: TimingStatus) -> None:
        ctx.explanation.timings.append(
            StageTiming(stage=stage.value, ms=round(ms, 2), status=status)
        )

    @asynccontextmanager
    async def _stage(self, ctx: TurnContext, stage: StageName) -> AsyncIterator[_StageRun]:
        await self._emit(ctx, stage, "start")
        run = _StageRun()
        started = time.perf_counter()
        try:
            yield run
        except Exception:
            self._record(ctx, stage, (time.perf_counter() - started) * 1000, "failed")
            raise
        ms = (time.perf_counter() - started) * 1000
        self._record(ctx, stage, ms, run.status)
        await self._emit(ctx, stage, "end", {"ms": round(ms, 1), "outcome": run.status})

    async def _skip(self, ctx: TurnContext, stage: StageName) -> None:
        self._record(ctx, stage, 0.0, "skipped")
        await self._emit(ctx, stage, "skipped")

    # ------------------------------------------------------------------
    # Stage 0 — regenerate input + privacy
    # ------------------------------------------------------------------

    async def _resolve_regenerated_message(self, ctx: TurnContext, db: AsyncSession | None) -> None:
        async with self._session_scope(db) as session:
            last = await self._store.get_last_user_message(session, ctx.session_id)
        if last is not None:
            ctx.regenerated_user_message_id, ctx.raw_message = last
        elif not ctx.raw_message.strip():
            raise LookupError("No previous user message to regenerate")

    def _redact(self, text: str) -> tuple[str, int, dict[str, int]]:
        if self._redactor is None:
            from medbridge.privacy.redaction import redact

            self._redactor = redact
        result = self._redactor(text)
        return str(result.text), int(result.count), dict(result.categories)

    async def _stage_privacy(self, ctx: TurnContext) -> None:
        if not self._settings.PII_REDACTION_ENABLED:
            ctx.message = ctx.raw_message
            await self._skip(ctx, StageName.PRIVACY)
            return
        # A redaction failure propagates (fail closed): the turn returns the safe
        # error response and nothing unredacted is sent to an LLM or persisted.
        async with self._stage(ctx, StageName.PRIVACY):
            text, count, categories = self._redact(ctx.raw_message)
        ctx.message = text
        ctx.explanation.redactions = count
        ctx.explanation.redaction_categories = categories

    # ------------------------------------------------------------------
    # Stage 1 — deterministic emergency fast-path (raw text, no LLM)
    # ------------------------------------------------------------------

    def _classify_emergency(self, text: str) -> tuple[ActionEnum | None, dict[str, Any]]:
        """(action, audit record) from ``classify_detailed`` when available, else ``classify``."""
        classifier = self._emergency_classifier
        detailed = getattr(classifier, "classify_detailed", None)
        if callable(detailed):
            result = detailed(text)
            action = _as_action(getattr(result, "action", None))
            return action, _emergency_record(action, result)
        classify = getattr(classifier, "classify", None)
        if callable(classify):
            action = _as_action(classify(text))
        elif callable(classifier):
            action = _as_action(classifier(text))
        else:
            action = None
        return action, _emergency_record(action, None)

    async def _stage_emergency(self, ctx: TurnContext) -> None:
        if not ctx.ablation.use_emergency_fastpath:
            await self._skip(ctx, StageName.EMERGENCY)
            return
        async with self._stage(ctx, StageName.EMERGENCY) as run:
            try:
                action, record = self._classify_emergency(ctx.raw_message)
            except Exception as exc:  # Gate 2 can still ESCALATE downstream
                logger.error("emergency_classifier_failed", error=type(exc).__name__)
                run.status = "failed"
                action, record = None, _emergency_record(None, None)
        ctx.explanation.emergency = record
        rules: list[str] = record["matched_rules"]
        if action != ActionEnum.ESCALATE:
            return
        logger.warning("emergency_fastpath_escalate", session_id=str(ctx.session_id))
        rationale = "Emergency classifier fast-path trigger"
        if rules:
            rationale += f" (rules: {', '.join(rules)})"
        ctx.explanation.gate2 = GateDecision(
            action="ESCALATE", rationale=rationale, source="fastpath"
        )
        ctx.final_action = ActionEnum.ESCALATE
        ctx.response_text = get_template(ActionEnum.ESCALATE, self._settings)
        ctx.complete = True

    # ------------------------------------------------------------------
    # Stage 2 — session state
    # ------------------------------------------------------------------

    def _chain_limit(self) -> int:
        """Earlier messages carried into the effective message (≤ consecutive SOFT-ASKs)."""
        return max(1, int(self._settings.SOFT_ASK_MAX_COUNT))

    async def _counter_before_regenerated_turn(
        self, session: AsyncSession, ctx: TurnContext
    ) -> int | None:
        """Loop-breaker counter recorded before the turn a regenerate replaces."""
        lookup = getattr(self._store, "get_soft_ask_count_before", None)
        if lookup is None or ctx.regenerated_user_message_id is None:
            return None
        try:
            value = await lookup(session, ctx.session_id, ctx.regenerated_user_message_id)
        except Exception as exc:  # keep the stored counter rather than failing the turn
            logger.warning("regenerate_counter_lookup_failed", error=type(exc).__name__)
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    async def _stage_load_state(self, ctx: TurnContext, db: AsyncSession | None) -> None:
        n = self._settings.EXTRACTOR_HISTORY_TURNS
        window = max(n, 2 * self._chain_limit())  # pairs needed for the effective message
        history: list[ChatTurn] = []
        counter_before: int | None = None
        async with self._stage(ctx, StageName.LOAD_STATE) as run:
            try:
                async with self._session_scope(db) as session:
                    state = await self._store.load_session_state(session, ctx.session_id)
                    if not ctx.complete:
                        limit = window + 2 if ctx.regenerate else window
                        history = await self._store.get_recent_turns(session, ctx.session_id, limit)
                    if ctx.regenerate:
                        counter_before = await self._counter_before_regenerated_turn(session, ctx)
            except Exception as exc:  # degrade: continue with an empty state
                logger.warning(
                    "load_state_failed", session_id=str(ctx.session_id), error=type(exc).__name__
                )
                run.status = "failed"
                state = SessionState(exists=False)
        ctx.snapshot = dict(state.snapshot or {})
        ctx.soft_ask_count = int(state.soft_ask_count or 0)
        if counter_before is not None:  # regenerate: re-run the turn from its own start value
            ctx.soft_ask_count = counter_before
        ctx.explanation.soft_ask_count_before = ctx.soft_ask_count
        recent = _trim_history(history, ctx.regenerate, window)
        ctx.history = recent[-n:] if n > 0 else []
        self._build_effective_message(ctx, recent)

    def _build_effective_message(self, ctx: TurnContext, history: list[ChatTurn]) -> None:
        """Original SOFT-ASK question + follow-up details for Gate 1, Gate 2 and the generator."""
        if ctx.complete:
            return
        chain = soft_ask_chain(history, self._chain_limit())
        earlier: list[str] = []
        for text in chain:
            try:  # stored messages are redacted already; re-redact legacy rows to be sure
                cleaned = (
                    self._redact(text)[0] if self._settings.PII_REDACTION_ENABLED else text
                ).strip()
            except Exception as exc:  # fail closed: answer from the current message only
                logger.warning("effective_message_redaction_failed", error=type(exc).__name__)
                return
            if cleaned:
                earlier.append(cleaned)
        if not earlier:
            return
        ctx.earlier_messages = earlier
        ctx.effective_message = compose_effective_message(earlier, ctx.message)
        ctx.explanation.effective_question = {"used": True, "earlier_messages": len(earlier)}

    # ------------------------------------------------------------------
    # Stage 2b — small talk (deterministic, claim-free)
    # ------------------------------------------------------------------

    async def _stage_small_talk(self, ctx: TurnContext) -> None:
        enabled = ctx.ablation.use_small_talk and self._settings.SMALL_TALK_ENABLED
        if ctx.complete or not enabled:
            await self._skip(ctx, StageName.SMALL_TALK)
            return
        async with self._stage(ctx, StageName.SMALL_TALK):
            key = self._is_small_talk(ctx.message)
        if key is None:
            return
        rationale = f"Small talk ({key}): deterministic template with no clinical claims."
        ctx.explanation.gate1 = GateDecision(action="PROCEED", rationale=rationale, source="rule")
        ctx.explanation.gate2 = GateDecision(
            action="GENERALIZE", rationale=rationale, source="rule"
        )
        ctx.final_action = ActionEnum.GENERALIZE
        ctx.response_text = get_small_talk_template(key)
        ctx.complete = True

    # ------------------------------------------------------------------
    # Stage 3 — context extraction + in-memory projection
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_query(message: str) -> str:
        """Retrieval query when the extractor failed: the message plus clinical synonyms."""
        try:
            return expand_query(message) or message
        except Exception as exc:  # deterministic helper; never let it break the turn
            logger.warning("query_expansion_failed", error=type(exc).__name__)
            return message

    async def _stage_extraction(self, ctx: TurnContext) -> None:
        if ctx.complete:
            await self._skip(ctx, StageName.EXTRACTION)
            return
        async with self._stage(ctx, StageName.EXTRACTION) as run:
            try:
                output = await self._context_extractor.extract(
                    ctx.message, ctx.snapshot, history=ctx.history
                )
            except Exception as exc:
                logger.warning("extractor_failed", error=type(exc).__name__)
                output = None
            if output is None:
                run.status = "failed"
        if output is None:  # spec fallback: the (redacted) message, no events
            ctx.raw_intent = ctx.patient_message
            ctx.search_query = self._fallback_query(ctx.carried_text)
            return
        ctx.search_query = (output.search_query or "").strip() or ctx.carried_text
        ctx.raw_intent = (output.raw_intent or "").strip() or ctx.patient_message
        events = list(output.delta_events)
        if not events or ctx.regenerate:  # a regeneration never re-applies its events
            return
        try:
            ctx.new_snapshot = self._state_projector.apply_events(ctx.snapshot, events)
        except Exception as exc:  # malformed payloads: keep the old snapshot, drop events
            logger.warning("projection_failed", error=type(exc).__name__)
            return
        ctx.delta_events = events

    # ------------------------------------------------------------------
    # Stage 4 — Gate 1 (Context Gate) + speculative retrieval + loop-breaker
    # ------------------------------------------------------------------

    def _retrieval_query(self, ctx: TurnContext) -> str:
        return ctx.search_query or ctx.message

    async def _stage_context_gate(self, ctx: TurnContext) -> None:
        if ctx.complete:
            await self._skip(ctx, StageName.CONTEXT_GATE)
            return
        if not ctx.ablation.use_context_gate:
            await self._skip(ctx, StageName.CONTEXT_GATE)
            ctx.explanation.gate1 = GateDecision(
                action="PROCEED",
                rationale=f"Context Gate bypassed by ablation preset {ctx.ablation.name}.",
                source="ablation",
            )
            ctx.soft_ask_count = 0
            ctx.counter_updated = True
            return

        if self._settings.SPECULATIVE_RETRIEVAL and ctx.ablation.use_retrieval:
            ctx.speculative_launched_at = time.perf_counter()
            ctx.speculative = asyncio.create_task(self._retrieve(ctx, self._retrieval_query(ctx)))

        gate_started = time.perf_counter()
        async with self._stage(ctx, StageName.CONTEXT_GATE) as run:
            try:
                output = await self._context_gate.evaluate(
                    ctx.gate_snapshot, ctx.patient_message, ctx.raw_intent
                )
            except Exception as exc:
                logger.warning("context_gate_failed", error=type(exc).__name__)
                output = None
            if output is None:
                run.status = "failed"
        ctx.gate1_window = (gate_started, time.perf_counter())

        if output is None:
            decision = GateDecision(
                action="PROCEED",
                rationale="Context Gate failed permanently; spec fallback to PROCEED.",
                source="fallback",
            )
        else:
            decision = GateDecision(
                action=output.action,
                rationale=output.rationale,
                missing_fields=list(output.missing_fields),
                source="llm",
            )
        ctx.explanation.gate1 = decision
        ctx.counter_updated = True

        if decision.action != "SOFT-ASK":
            ctx.soft_ask_count = 0
            return
        max_count = self._settings.SOFT_ASK_MAX_COUNT
        if ctx.soft_ask_count < max_count:
            ctx.final_action = ActionEnum.SOFT_ASK
            ctx.soft_ask_count += 1
            await self._discard_speculative(ctx)
            return
        # Loop-breaker (Non-Negotiable #4): force GENERALIZE, skip Gate 2, reset counter.
        logger.info("loop_breaker_triggered", soft_ask_count=ctx.soft_ask_count)
        ctx.explanation.gate2 = GateDecision(
            action="GENERALIZE",
            rationale=(
                "Forced GENERALIZE due to loop-breaker threshold "
                f"(soft_ask_count={ctx.soft_ask_count} >= {max_count}); Evidence Gate skipped."
            ),
            source="loop_breaker",
        )
        ctx.loop_breaker = True
        ctx.final_action = ActionEnum.GENERALIZE
        ctx.soft_ask_count = 0

    async def _discard_speculative(self, ctx: TurnContext) -> None:
        """Cancel a speculative retrieval whose results Gate 1 did not allow."""
        task = ctx.speculative
        if task is None:
            return
        ctx.speculative = None
        if not task.done():
            task.cancel()
        await asyncio.wait([task])
        if task.cancelled() or task.exception() is not None:
            elapsed = (time.perf_counter() - ctx.speculative_launched_at) * 1000
            self._record(ctx, StageName.RETRIEVAL, elapsed, "discarded")
        else:
            outcome = task.result()
            self._record(ctx, StageName.RETRIEVAL, outcome.retrieval_ms, "discarded")
            self._record(ctx, StageName.RERANK, outcome.rerank_ms, "discarded")
        ctx.retrieval_recorded = True
        ctx.explanation.retrieval_status = "discarded"
        ctx.explanation.speculative_retrieval = {
            "launched": True,
            "used": False,
            "discarded": True,
            "saved_ms": 0.0,
        }
        await self._emit(ctx, StageName.RETRIEVAL, "skipped", {"discarded": True})

    # ------------------------------------------------------------------
    # Stage 5 — hybrid retrieval + cross-encoder rerank
    # ------------------------------------------------------------------

    async def _retrieve(self, ctx: TurnContext, query: str) -> RetrievalOutcome:
        """Hybrid search (top-20) then rerank (top-5); failures degrade, never raise."""
        settings = self._settings
        started = time.perf_counter()
        await self._emit(ctx, StageName.RETRIEVAL, "start")
        retrieval_status: TimingStatus = "ok"
        try:
            candidates = await self._hybrid_retriever.hybrid_search(
                query, top_k=settings.RETRIEVAL_TOP_K_CANDIDATES
            )
        except Exception as exc:
            logger.warning("retrieval_stage_failed", error=type(exc).__name__)
            candidates, retrieval_status = [], "failed"
        retrieval_ms = (time.perf_counter() - started) * 1000
        await self._emit(
            ctx,
            StageName.RETRIEVAL,
            "end",
            {"ms": round(retrieval_ms, 1), "candidates": len(candidates)},
        )

        chunks: list[RetrievedChunk] = []
        rerank_status: TimingStatus = "skipped"
        rerank_ms = 0.0
        if candidates:
            rerank_started = time.perf_counter()
            await self._emit(ctx, StageName.RERANK, "start")
            try:
                chunks = await self._reranker.rerank_chunks(
                    query, candidates, top_k=settings.RETRIEVAL_TOP_K_RERANKED
                )
                rerank_status = "ok"
            except Exception as exc:  # keep the RRF order rather than losing the evidence
                logger.warning("rerank_failed", error=type(exc).__name__)
                chunks, rerank_status = candidates[: settings.RETRIEVAL_TOP_K_RERANKED], "failed"
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            await self._emit(ctx, StageName.RERANK, "end", {"ms": round(rerank_ms, 1)})
        else:
            await self._emit(ctx, StageName.RERANK, "skipped")
        return RetrievalOutcome(
            chunks=chunks,
            candidates=len(candidates),
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            retrieval_status=retrieval_status,
            rerank_status=rerank_status,
            started_at=started,
            finished_at=time.perf_counter(),
        )

    async def _stage_retrieval(self, ctx: TurnContext) -> None:
        if ctx.retrieval_recorded:  # speculative run already discarded and logged
            return
        if (
            ctx.complete
            or ctx.final_action == ActionEnum.SOFT_ASK
            or not ctx.ablation.use_retrieval
        ):
            await self._skip(ctx, StageName.RETRIEVAL)
            await self._skip(ctx, StageName.RERANK)
            ctx.explanation.retrieval_status = "skipped"
            return
        if ctx.speculative is not None:
            task, ctx.speculative = ctx.speculative, None
            outcome = await task
            saved = _overlap_ms((outcome.started_at, outcome.finished_at), ctx.gate1_window)
            ctx.explanation.speculative_retrieval = {
                "launched": True,
                "used": True,
                "discarded": False,
                "saved_ms": round(saved, 1),
            }
        else:
            outcome = await self._retrieve(ctx, self._retrieval_query(ctx))
            ctx.explanation.speculative_retrieval = {
                "launched": False,
                "used": False,
                "discarded": False,
                "saved_ms": 0.0,
            }
        self._record(ctx, StageName.RETRIEVAL, outcome.retrieval_ms, outcome.retrieval_status)
        self._record(ctx, StageName.RERANK, outcome.rerank_ms, outcome.rerank_status)
        if outcome.retrieval_status == "failed":
            # Degraded turn: the retriever could not run. Gate 2 still decides on no
            # evidence (spec); the status lets evaluation and metrics tell this apart
            # from a corpus that simply had nothing relevant.
            ctx.explanation.retrieval_status = "failed"
        else:
            ctx.explanation.retrieval_status = "ok" if outcome.candidates else "empty"
        ctx.chunks = outcome.chunks
        ctx.explanation.retrieval = [_evidence(i, c) for i, c in enumerate(outcome.chunks, 1)]

    # ------------------------------------------------------------------
    # Stage 6 — Gate 2 (Evidence Gate), optional V1 calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_features(ctx: TurnContext) -> dict[str, Any]:
        """V1 evidence features for the plain Gate 2 (same function and inputs as V1).

        Recorded so the V1 thresholds can be calibrated from C5 traces with exactly
        the production features. Deterministic, cheap and never raises.
        """
        try:
            from medbridge.ai.evidence_calibration import evidence_features

            features = evidence_features(
                ctx.patient_message or ctx.search_query, ctx.chunks, ctx.gate_snapshot
            )
        except Exception as exc:
            logger.warning("evidence_features_failed", error=type(exc).__name__)
            return {}
        return {str(k): float(v) for k, v in features.items()}

    async def _llm_evidence_decision(self, ctx: TurnContext) -> GateDecision | None:
        # The gate judges the patient's own words (false premises, unknown drug
        # names); the extractor's clinical search query is passed alongside.
        try:
            output = await self._evidence_gate.evaluate(
                chunks=ctx.chunks,
                snapshot=ctx.gate_snapshot,
                query=self._retrieval_query(ctx),
                message=ctx.patient_message,
            )
        except Exception as exc:
            logger.warning("evidence_gate_failed", error=type(exc).__name__)
            return None
        if output is None or output.action not in _GATE2_ACTIONS:
            return None
        return GateDecision(action=output.action, rationale=output.rationale, source="llm")

    async def _calibrated_evidence_decision(self, ctx: TurnContext) -> GateDecision | None:
        try:
            from medbridge.ai.evidence_calibration import calibrated_evidence_decision

            result = await calibrated_evidence_decision(
                self._evidence_gate,
                ctx.chunks,
                ctx.gate_snapshot,
                self._retrieval_query(ctx),
                votes=self._settings.EVIDENCE_GATE_VOTES,
                settings=self._settings,
                message=ctx.patient_message,
            )
        except Exception as exc:
            logger.warning("calibrated_evidence_gate_failed", error=type(exc).__name__)
            return None
        failed_votes = getattr(result, "failed_votes", 0)
        ctx.failed_votes = failed_votes if isinstance(failed_votes, int) else 0
        votes = [str(v) for v in (getattr(result, "votes", None) or [])]
        if not votes:
            # Every vote failed: this is the Gate 2 spec fallback, not a calibrated
            # decision (stage "failed", source "fallback" — counted by /api/metrics).
            logger.warning("calibrated_evidence_gate_all_votes_failed", failed=ctx.failed_votes)
            return None
        action = str(getattr(result, "action", ""))
        if action not in _GATE2_ACTIONS:
            return None
        features = dict(getattr(result, "features", None) or {})
        features["overridden"] = bool(getattr(result, "overridden", False))
        features["failed_votes"] = float(ctx.failed_votes)  # partial vote loss stays visible
        return GateDecision(
            action=action,
            rationale=str(getattr(result, "rationale", "") or ""),
            source="calibrated",
            votes=votes,
            features=features,
        )

    async def _stage_evidence_gate(self, ctx: TurnContext) -> None:
        if ctx.complete or ctx.final_action == ActionEnum.SOFT_ASK or ctx.loop_breaker:
            await self._skip(ctx, StageName.EVIDENCE_GATE)
            return
        if not ctx.ablation.use_evidence_gate:
            await self._skip(ctx, StageName.EVIDENCE_GATE)
            ctx.explanation.gate2 = GateDecision(
                action="ANSWER",
                rationale=f"Evidence Gate bypassed by ablation preset {ctx.ablation.name}.",
                source="ablation",
            )
            ctx.final_action = ActionEnum.ANSWER
            return
        calibrated = ctx.ablation.calibrated_evidence_gate
        if calibrated is None:
            calibrated = self._settings.EVIDENCE_GATE_CALIBRATION_ENABLED
        async with self._stage(ctx, StageName.EVIDENCE_GATE) as run:
            if calibrated:
                decision = await self._calibrated_evidence_decision(ctx)
            else:
                decision = await self._llm_evidence_decision(ctx)
            if decision is None:
                run.status = "failed"
        if decision is None:
            decision = GateDecision(
                action="GENERALIZE",
                rationale="Evidence Gate failed permanently; spec fallback to GENERALIZE.",
                source="fallback",
            )
        if not decision.features:  # the calibrated gate already carries its own features
            decision.features = self._evidence_features(ctx)
            if ctx.failed_votes is not None:
                decision.features["failed_votes"] = float(ctx.failed_votes)
        ctx.explanation.gate2 = decision
        ctx.final_action = ActionEnum(decision.action)

    # ------------------------------------------------------------------
    # Stage 7 — generation (templates bypass the LLM)
    # ------------------------------------------------------------------

    async def _generate(
        self,
        ctx: TurnContext,
        action: ActionEnum,
        chunks: list[RetrievedChunk],
        missing_fields: list[str] | None = None,
    ) -> str | None:
        kwargs: dict[str, Any] = {"mode": ctx.ablation.generator_mode, "history": ctx.history}
        if missing_fields is not None:
            kwargs["missing_fields"] = missing_fields
        try:
            output = await self._response_generator.generate(
                action=action,
                chunks=chunks,
                snapshot=ctx.gate_snapshot,
                query=self._retrieval_query(ctx),
                message=ctx.patient_message,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("generator_failed", action=action.value, error=type(exc).__name__)
            return None
        text = (getattr(output, "response_text", "") or "").strip() if output is not None else ""
        return text or None

    async def _stage_generation(self, ctx: TurnContext) -> None:
        action = ctx.final_action
        if ctx.complete:
            await self._skip(ctx, StageName.GENERATION)
            return
        if action is None:
            raise RuntimeError("No routing decision reached the generation stage")
        if action in (ActionEnum.ABSTAIN, ActionEnum.ESCALATE):
            # Template bypass (Non-Negotiable #6): the generator is never invoked.
            await self._skip(ctx, StageName.GENERATION)
            ctx.response_text = get_template(action, self._settings)
            return
        async with self._stage(ctx, StageName.GENERATION) as run:
            if action == ActionEnum.SOFT_ASK:
                gate1 = ctx.explanation.gate1
                missing = list(gate1.missing_fields) if gate1 else []
                text = await self._generate(ctx, action, [], missing_fields=missing)
                if text is None:
                    run.status = "failed"
                    text = build_soft_ask_fallback(missing)
                ctx.response_text = text
            else:
                text = await self._generate(ctx, action, ctx.chunks)
                if text is None:  # spec fallback: pre-vetted GENERALIZE template
                    run.status = "failed"
                    ctx.final_action = ActionEnum.GENERALIZE
                    ctx.response_text = GENERALIZE_FALLBACK_TEMPLATE
                else:
                    ctx.draft = text

    # ------------------------------------------------------------------
    # Stage 7b — V2 Gate 3 (post-generation grounding verifier)
    # ------------------------------------------------------------------

    def _get_verifier(self) -> Any:
        if self._grounding_verifier is None:
            from medbridge.ai.grounding_verifier import get_grounding_verifier

            self._grounding_verifier = get_grounding_verifier()
        return self._grounding_verifier

    def _verifier_enabled(self, ctx: TurnContext) -> bool:
        flag = ctx.ablation.grounding_verifier
        return self._settings.GROUNDING_VERIFIER_ENABLED if flag is None else flag

    async def _stage_verification(self, ctx: TurnContext) -> None:
        if (
            ctx.complete
            or ctx.draft is None
            or ctx.final_action not in (ActionEnum.ANSWER, ActionEnum.GENERALIZE)
            or not self._verifier_enabled(ctx)
        ):
            await self._skip(ctx, StageName.VERIFICATION)
            return
        async with self._stage(ctx, StageName.VERIFICATION) as run:
            try:
                await self._verify_draft(ctx)
            except Exception as exc:  # keep the draft; record the failure
                logger.warning("grounding_verifier_failed", error=type(exc).__name__)
                run.status = "failed"
                ctx.explanation.gate3 = {"status": "failed", "error": type(exc).__name__}

    async def _verify_draft(self, ctx: TurnContext) -> None:
        assert ctx.draft is not None and ctx.final_action is not None
        tau = self._settings.GROUNDING_MAX_UNSUPPORTED_FRACTION
        verifier = self._get_verifier()
        original = ctx.final_action
        result = await verifier.verify(ctx.draft, ctx.chunks)
        record: dict[str, Any] = {**_gate3_record(result), "threshold": tau, "attempts": 1}

        if not _gate3_exceeds(record, tau):
            ctx.draft = _cleaned_text(result, ctx.draft)
            record.update(action_change=None, removed_sentences_count=record["n_unsupported"])
            ctx.explanation.gate3 = record
            return

        if original == ActionEnum.ANSWER:
            regenerated = await self._generate(ctx, ActionEnum.GENERALIZE, ctx.chunks)
            if regenerated is None:  # generator failure → pre-vetted template
                ctx.final_action, ctx.draft = ActionEnum.GENERALIZE, None
                ctx.response_text = GENERALIZE_FALLBACK_TEMPLATE
                record.update(action_change="ANSWER->GENERALIZE", removed_sentences_count=0)
                ctx.explanation.gate3 = record
                return
            second = await verifier.verify(regenerated, ctx.chunks)
            record = {
                **_gate3_record(second),
                "threshold": tau,
                "attempts": 2,
                "initial_unsupported_fraction": record["unsupported_fraction"],
            }
            if not _gate3_exceeds(record, tau):
                ctx.final_action = ActionEnum.GENERALIZE
                ctx.draft = _cleaned_text(second, regenerated)
                record.update(
                    action_change="ANSWER->GENERALIZE",
                    removed_sentences_count=record["n_unsupported"],
                )
                ctx.explanation.gate3 = record
                return

        ctx.final_action, ctx.draft = ActionEnum.ABSTAIN, None
        ctx.response_text = get_template(ActionEnum.ABSTAIN, self._settings)
        record.update(action_change=f"{original.value}->ABSTAIN", removed_sentences_count=0)
        ctx.explanation.gate3 = record

    # ------------------------------------------------------------------
    # Citation validation, downgrade and loop-breaker prefix
    # ------------------------------------------------------------------

    def _finalise_response_text(self, ctx: TurnContext) -> None:
        if ctx.complete:
            return
        if ctx.draft is not None and ctx.final_action in (ActionEnum.ANSWER, ActionEnum.GENERALIZE):
            validation = validate_citations(ctx.draft, ctx.chunks)
            downgraded = (
                self._settings.CITATION_VALIDATION_ENABLED
                and ctx.final_action == ActionEnum.ANSWER
                and validation.valid == 0
                and ctx.ablation.use_evidence_gate
            )
            if downgraded:
                ctx.final_action = ActionEnum.GENERALIZE
            ctx.response_text = validation.text
            ctx.citations = validation.citations
            ctx.explanation.citation_validation = {
                "valid": validation.valid,
                "stripped": validation.stripped,
                "downgraded": downgraded,
            }
            cited = set(validation.cited_chunk_ids)
            for item in ctx.explanation.retrieval:
                item.cited = item.chunk_id in cited
        elif ctx.final_action == ActionEnum.SOFT_ASK:
            ctx.response_text = validate_citations(ctx.response_text, []).text
        if ctx.loop_breaker and ctx.final_action == ActionEnum.GENERALIZE:
            prefix = get_loop_breaker_prefix()
            if not ctx.response_text.startswith(prefix):
                ctx.response_text = f"{prefix}\n\n{ctx.response_text}"

    # ------------------------------------------------------------------
    # Stage 8 — single persistence step
    # ------------------------------------------------------------------

    def _build_record(self, ctx: TurnContext) -> TurnRecord:
        gate1, gate2 = ctx.explanation.gate1, ctx.explanation.gate2
        g1 = gate1 if gate1 is not None and gate1.source not in _NON_DECISION_SOURCES else None
        g2 = gate2 if gate2 is not None and gate2.source not in _NON_DECISION_SOURCES else None
        return TurnRecord(
            session_id=ctx.session_id,
            user_message=ctx.message,
            assistant_message=ctx.response_text,
            final_action=ctx.final_action or ActionEnum.GENERALIZE,
            citations=[c.model_dump(mode="json") for c in ctx.citations],
            gate_1_action=g1.action if g1 else None,
            gate_1_rationale=g1.rationale if g1 else None,
            gate_2_action=g2.action if g2 else None,
            gate_2_rationale=g2.rationale if g2 else None,
            evidence_chunk_ids=[c.chunk_id for c in ctx.chunks],
            latency_ms=ctx.latency_ms,
            trace=ctx.explanation.model_dump(mode="json"),
            delta_events=[] if ctx.regenerate else list(ctx.delta_events),
            new_snapshot=None if ctx.regenerate else ctx.new_snapshot,
            soft_ask_count=ctx.soft_ask_count,
            regenerated_user_message_id=ctx.regenerated_user_message_id,
            update_soft_ask_count=ctx.counter_updated,
        )

    async def _stage_persistence(self, ctx: TurnContext, db: AsyncSession | None) -> None:
        ctx.latency_ms = int((time.perf_counter() - ctx.started_at) * 1000)
        ctx.explanation.total_ms = round((time.perf_counter() - ctx.started_at) * 1000, 1)
        ctx.explanation.llm_calls = [_serialise_call(c) for c in list(ctx.llm_calls)]
        record = self._build_record(ctx)
        async with self._stage(ctx, StageName.PERSISTENCE) as run:
            try:
                async with self._session_scope(db) as session:
                    try:
                        user_id, assistant_id = await self._store.persist_turn(session, record)
                    except Exception:
                        if db is not None:  # leave the caller's session committable
                            with contextlib.suppress(Exception):
                                await db.rollback()
                        raise
            except RegenerateConflictError:  # a newer turn was stored meanwhile
                logger.warning("regenerate_not_persisted", session_id=str(ctx.session_id))
                run.status = "failed"
            except Exception as exc:  # the patient still gets the answer
                logger.error(
                    "persistence_failed",
                    session_id=str(ctx.session_id),
                    error=type(exc).__name__,
                    exc_info=True,
                )
                run.status = "failed"
            else:
                ctx.user_message_id = ctx.regenerated_user_message_id or user_id
                ctx.message_id = assistant_id
        ctx.explanation.total_ms = round((time.perf_counter() - ctx.started_at) * 1000, 1)

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    def _build_response(self, ctx: TurnContext) -> MessageResponse:
        return MessageResponse(
            session_id=ctx.session_id,
            response_text=ctx.response_text,
            action=ctx.final_action or ActionEnum.GENERALIZE,
            citations=ctx.citations,
            soft_ask_count=ctx.soft_ask_count,
            timestamp=datetime.now(UTC),
            message_id=ctx.message_id,
            user_message_id=ctx.user_message_id,
            redaction_count=ctx.explanation.redactions,
            explanation=ctx.explanation,
        )

    def _error_response(self, ctx: TurnContext, exc: Exception) -> MessageResponse:
        ctx.explanation.gate2 = GateDecision(
            action="GENERALIZE",
            rationale=(
                f"Unhandled pipeline error ({type(exc).__name__}); safe fallback returned "
                "and nothing was persisted."
            ),
            source="fallback",
        )
        ctx.explanation.total_ms = round((time.perf_counter() - ctx.started_at) * 1000, 1)
        return MessageResponse(
            session_id=ctx.session_id,
            response_text=PIPELINE_ERROR_TEMPLATE,
            action=ActionEnum.GENERALIZE,
            citations=[],
            soft_ask_count=ctx.soft_ask_count,
            timestamp=datetime.now(UTC),
            redaction_count=ctx.explanation.redactions,
            explanation=ctx.explanation,
        )


_orchestrator_instance: PipelineOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> PipelineOrchestrator:
    """Return the shared orchestrator (FastAPI dependency; it holds no per-turn state)."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        with _orchestrator_lock:
            if _orchestrator_instance is None:
                _orchestrator_instance = PipelineOrchestrator()
    return _orchestrator_instance

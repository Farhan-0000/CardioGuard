"""
Complementary feature routes (BP log, context card, feedback, clinician summary).

Endpoints (all require the session token when ``AUTH_REQUIRED``):
- POST /api/sessions/{id}/feedback  → 201 ``{feedback_id}``
- GET  /api/sessions/{id}/context   → snapshot + BP series with ACC/AHA categories + counts
- POST /api/sessions/{id}/bp        → 201 manual BP entry → BP_READING event via the projector
- GET  /api/sessions/{id}/summary   → clinician summary (``?format=md|json``)

None of these routes calls an LLM or bypasses the gates: they read/append the
same event-sourced state the pipeline uses.
"""
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.routes.sessions import ensure_session_available
from medbridge.api.schemas.enums import EventTypeEnum
from medbridge.config import get_settings
from medbridge.db.connection import get_session
from medbridge.features.bp import CATEGORY_LEGEND, CRISIS, CRISIS_ADVICE, bp_series, make_reading
from medbridge.features.summary import build_clinician_summary, render_markdown
from medbridge.privacy.deps import require_session_access
from medbridge.state import turn_store

features_router = APIRouter(
    prefix="/sessions",
    tags=["features"],
    dependencies=[Depends(require_session_access)],
)

# Accept device clocks slightly ahead of the server.
_MAX_FUTURE_SKEW = timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    message_id: uuid.UUID
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: uuid.UUID


class BPReadingOut(BaseModel):
    systolic: int | float
    diastolic: int | float
    pulse: int | float | None = None
    recorded_at: str
    source: str
    category: str
    category_label: str


class ContextResponse(BaseModel):
    session_id: uuid.UUID
    snapshot: dict[str, Any]
    bp_readings: list[BPReadingOut]
    counts: dict[str, int]
    categories: list[dict[str, Any]] = Field(default_factory=lambda: list(CATEGORY_LEGEND))


class BPEntryRequest(BaseModel):
    systolic: int = Field(..., ge=60, le=300, description="Systolic pressure, mmHg")
    diastolic: int = Field(..., ge=30, le=200, description="Diastolic pressure, mmHg")
    pulse: int | None = Field(default=None, ge=20, le=250, description="Pulse, beats/min")
    measured_at: datetime | None = Field(default=None, description="Defaults to now (UTC)")

    @model_validator(mode="after")
    def _check(self) -> "BPEntryRequest":
        if self.systolic <= self.diastolic:
            raise ValueError("systolic must be greater than diastolic")
        if self.measured_at is not None:
            if self.measured_at.tzinfo is None:
                self.measured_at = self.measured_at.replace(tzinfo=UTC)
            if self.measured_at > datetime.now(UTC) + _MAX_FUTURE_SKEW:
                raise ValueError("measured_at cannot be in the future")
        return self


class BPEntryResponse(BaseModel):
    snapshot: dict[str, Any]
    reading: BPReadingOut
    crisis: bool
    emergency_numbers: dict[str, str]
    advice: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@features_router.post(
    "/{session_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Rate an assistant answer (thumbs up/down + optional comment)",
)
async def handle_feedback(
    session_id: uuid.UUID,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_session),
) -> FeedbackResponse:
    """Store feedback; the comment is PII-redacted and encrypted."""
    feedback_id = await turn_store.save_feedback(
        db, session_id, payload.message_id, payload.rating, payload.comment
    )
    return FeedbackResponse(feedback_id=feedback_id)


@features_router.get(
    "/{session_id}/context",
    response_model=ContextResponse,
    summary="Patient context card: snapshot + BP series with ACC/AHA categories",
)
async def handle_context(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> ContextResponse:
    state = await ensure_session_available(db, session_id)
    events = await turn_store.list_events(db, session_id)
    counts = {event_type.value: 0 for event_type in EventTypeEnum}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    counts["total_events"] = len(events)
    return ContextResponse(
        session_id=session_id,
        snapshot=state.snapshot,
        bp_readings=[BPReadingOut.model_validate(r) for r in bp_series(events)],
        counts=counts,
    )


@features_router.post(
    "/{session_id}/bp",
    response_model=BPEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a home BP reading (appended as a BP_READING event)",
)
async def handle_bp_entry(
    session_id: uuid.UUID,
    payload: BPEntryRequest,
    db: AsyncSession = Depends(get_session),
) -> BPEntryResponse:
    settings = get_settings()
    measured_at = (payload.measured_at or datetime.now(UTC)).isoformat()
    event_payload: dict[str, Any] = {
        "systolic": payload.systolic,
        "diastolic": payload.diastolic,
        "measured_at": measured_at,
        "source": "manual",
    }
    if payload.pulse is not None:
        event_payload["pulse"] = payload.pulse
    snapshot = await turn_store.append_manual_events(
        db, session_id, [DeltaEvent(event_type=EventTypeEnum.BP_READING, payload=event_payload)]
    )
    reading = make_reading(
        payload.systolic, payload.diastolic, payload.pulse, measured_at, "manual"
    )
    crisis = reading["category"] == CRISIS.key
    return BPEntryResponse(
        snapshot=snapshot,
        reading=BPReadingOut.model_validate(reading),
        crisis=crisis,
        emergency_numbers={
            "emergency": settings.EMERGENCY_NUMBER,
            "ambulance": settings.AMBULANCE_NUMBER,
            "region": settings.EMERGENCY_REGION_LABEL,
        },
        advice=CRISIS_ADVICE.format(
            emergency=settings.EMERGENCY_NUMBER, ambulance=settings.AMBULANCE_NUMBER
        ) if crisis else None,
    )


@features_router.get(
    "/{session_id}/summary",
    summary="Clinician summary of the session (Markdown or JSON)",
    response_model=None,
)
async def handle_summary(
    session_id: uuid.UUID,
    fmt: Literal["md", "json"] = Query("md", alias="format"),
    db: AsyncSession = Depends(get_session),
) -> Response:
    summary = await build_clinician_summary(db, session_id)
    if fmt == "json":
        return JSONResponse(summary)
    filename = f"medbridge-summary-{summary['session_ref']}.md"
    return Response(
        content=render_markdown(summary),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

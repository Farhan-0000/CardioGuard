"""
SQLAlchemy ORM models (Module M-02).

Append-only tables (``clinical_events``, ``audit_logs``) are protected by
PostgreSQL ``RULE``s created in the initial migration: UPDATE and DELETE are
silently turned into no-ops. Privacy additions (migration
``g1a7c3e9d2b4_privacy_trace_feedback``):

* ``sessions.erased_at`` — set when the session was crypto-shredded.
* ``session_keys`` — one wrapped (KEK-encrypted) data key per session; deleting
  the row is the crypto-shred that makes every encrypted value unreadable.
* ``feedback`` — 👍/👎 ratings with an optional encrypted comment.
* ``audit_logs.trace`` — PHI-free pipeline trace (timings, actions, chunk ids);
  the PHI-bearing parts of the trace are stored encrypted inside it.

Migration ``h2b8d4f6a1c3_feedback_unique_legacy_marker``:

* ``feedback`` — ``UNIQUE (session_id, message_id)`` (one rating per message; upsert).
* ``sessions.legacy_plaintext_rows`` — set at erasure: append-only rows of the
  session written before encryption existed, which crypto-shredding cannot cover.

Sessions are never deleted: rows in the rule-protected tables reference them.
"""
import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from medbridge.api.schemas.enums import EventTypeEnum


class Base(DeclarativeBase):
    pass


class Gate1ActionEnum(enum.StrEnum):
    SOFT_ASK = "SOFT-ASK"
    PROCEED = "PROCEED"


class Gate2ActionEnum(enum.StrEnum):
    ANSWER = "ANSWER"
    GENERALIZE = "GENERALIZE"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


class FinalActionEnum(enum.StrEnum):
    SOFT_ASK = "SOFT-ASK"
    ANSWER = "ANSWER"
    GENERALIZE = "GENERALIZE"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    erased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Recorded at erasure: plaintext (pre-encryption) rows left in the append-only tables.
    legacy_plaintext_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    clinical_events: Mapped[list["ClinicalEvent"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    context_snapshot: Mapped["ContextSnapshot"] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    message_history: Mapped[list["MessageHistory"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    session_key: Mapped[Optional["SessionKey"]] = relationship(
        back_populates="session", uselist=False, passive_deletes=True
    )
    feedback: Mapped[list["Feedback"]] = relationship(
        back_populates="session", passive_deletes=True
    )


class ClinicalEvent(Base):
    __tablename__ = "clinical_events"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[EventTypeEnum] = mapped_column(
        Enum(EventTypeEnum, name="event_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    # Encrypted envelope {"enc": "v1", "ct": ...} (legacy rows: plaintext dict).
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="clinical_events")


class ContextSnapshot(Base):
    __tablename__ = "context_snapshots"

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), primary_key=True
    )
    # Encrypted envelope {"enc": "v1", "ct": ...} (legacy rows: plaintext dict).
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    soft_ask_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="context_snapshot")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    request_message: Mapped[str] = mapped_column(Text, nullable=False)  # enc:v1:...
    gate_1_action: Mapped[Gate1ActionEnum | None] = mapped_column(
        Enum(
            Gate1ActionEnum,
            name="gate_1_action_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=True,
    )
    gate_1_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)  # enc:v1:...
    gate_2_action: Mapped[Gate2ActionEnum | None] = mapped_column(
        Enum(
            Gate2ActionEnum,
            name="gate_2_action_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=True,
    )
    gate_2_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)  # enc:v1:...
    final_action: Mapped[FinalActionEnum] = mapped_column(
        Enum(
            FinalActionEnum,
            name="final_action_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    evidence_chunk_ids: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text), nullable=True, default=list
    )
    response_text: Mapped[str] = mapped_column(Text, nullable=False)  # enc:v1:...
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # PHI-free trace (timings, actions, models); PHI parts nested as an encrypted envelope.
    trace: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    session: Mapped["Session"] = relationship(back_populates="audit_logs")


class MessageHistory(Base):
    __tablename__ = "message_history"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)  # enc:v1:...
    action: Mapped[FinalActionEnum | None] = mapped_column(
        Enum(
            FinalActionEnum,
            name="final_action_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=True,
    )
    # Guideline citations (public guideline metadata, no patient data).
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="message_history")


class SessionKey(Base):
    """Per-session data-encryption key, wrapped with the master key (KEK)."""

    __tablename__ = "session_keys"

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), primary_key=True
    )
    wrapped_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="session_key")


class Feedback(Base):
    """Thumbs up (+1) / down (-1) on an assistant message."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        UniqueConstraint("session_id", "message_id", name="uq_feedback_session_message"),
    )

    feedback_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)  # redacted, enc:v1:...
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped["Session"] = relationship(back_populates="feedback")

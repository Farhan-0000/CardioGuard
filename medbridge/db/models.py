import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


from medbridge.api.schemas.enums import EventTypeEnum


class Gate1ActionEnum(str, enum.Enum):
    SOFT_ASK = "SOFT-ASK"
    PROCEED = "PROCEED"


class Gate2ActionEnum(str, enum.Enum):
    ANSWER = "ANSWER"
    GENERALIZE = "GENERALIZE"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


class FinalActionEnum(str, enum.Enum):
    SOFT_ASK = "SOFT-ASK"
    ANSWER = "ANSWER"
    GENERALIZE = "GENERALIZE"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    clinical_events: Mapped[List["ClinicalEvent"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    context_snapshot: Mapped["ContextSnapshot"] = relationship(back_populates="session", cascade="all, delete-orphan", uselist=False)
    audit_logs: Mapped[List["AuditLog"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    message_history: Mapped[List["MessageHistory"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ClinicalEvent(Base):
    __tablename__ = "clinical_events"

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    event_type: Mapped[EventTypeEnum] = mapped_column(
        Enum(EventTypeEnum, name="event_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship(back_populates="clinical_events")


class ContextSnapshot(Base):
    __tablename__ = "context_snapshots"

    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"), primary_key=True)
    snapshot: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    soft_ask_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship(back_populates="context_snapshot")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    request_message: Mapped[str] = mapped_column(Text, nullable=False)
    gate_1_action: Mapped[Gate1ActionEnum | None] = mapped_column(
        Enum(Gate1ActionEnum, name="gate_1_action_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    gate_1_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    gate_2_action: Mapped[Gate2ActionEnum | None] = mapped_column(
        Enum(Gate2ActionEnum, name="gate_2_action_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    gate_2_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_action: Mapped[FinalActionEnum] = mapped_column(
        Enum(FinalActionEnum, name="final_action_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    evidence_chunk_ids: Mapped[List[str] | None] = mapped_column(ARRAY(Text), nullable=True, default=list)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship(back_populates="audit_logs")


class MessageHistory(Base):
    __tablename__ = "message_history"

    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[FinalActionEnum | None] = mapped_column(
        Enum(FinalActionEnum, name="final_action_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    citations: Mapped[List[Dict[str, Any]] | None] = mapped_column(JSONB, nullable=True, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["Session"] = relationship(back_populates="message_history")

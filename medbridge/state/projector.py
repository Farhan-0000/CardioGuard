"""
Deterministic State Projector (ADL-001).

Applies typed delta events to patient context snapshots using pure Python
deterministic merge rules. Does NOT use the LLM.

Persistence: ``persist_and_project()`` inserts (encrypted) clinical events and
upserts the (encrypted) snapshot within a single database transaction. The
pipeline itself uses ``medbridge.state.turn_store`` which applies the same pure
projection and persists the whole turn with one flush.

Technical Specification Part IV §3.3 / §4.1 Projection Rules.
"""
import copy
import uuid
from datetime import timedelta
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.db.models import ClinicalEvent, ContextSnapshot, EventTypeEnum
from medbridge.state.session_manager import require_cipher, utcnow

logger = structlog.get_logger(__name__)

# Readings kept in the snapshot (the full series stays in clinical_events).
MAX_BP_READINGS = 5
# Lab results kept per lab type.
MAX_LABS_PER_TYPE = 3


def event_type_value(event: DeltaEvent) -> str:
    """The event type as a plain string (enum or free text)."""
    raw = event.event_type
    return raw.value if isinstance(raw, EventTypeEnum) else str(raw)


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _lab_key(lab: dict[str, Any]) -> str:
    return str(_first(lab, "lab_type", "lab", "test", "name") or "").strip().lower()


class StateProjector:
    """
    Deterministic Python projector.

    All projection methods are **pure functions** over snapshot dicts — they
    do not access the network or the LLM.  Database interaction is limited to
    the explicit ``persist_and_project()`` method.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_events(
        self,
        snapshot: dict[str, Any],
        events: list[DeltaEvent],
    ) -> dict[str, Any]:
        """Apply a list of delta events to an existing snapshot.

        Returns a *new* snapshot dict (deep-copy) so callers retain the
        original if needed.
        """
        result = copy.deepcopy(snapshot)
        for event in events:
            result = self._apply_single(result, event)
        return result

    def reconstruct_from_events(
        self,
        events: list[DeltaEvent],
    ) -> dict[str, Any]:
        """Replay the full event stream from an empty snapshot.

        Used for validation / integrity checking — the result must match
        the stored snapshot exactly.
        """
        return self.apply_events({}, events)

    async def persist_and_project(
        self,
        session: AsyncSession,
        session_id: uuid.UUID,
        events: list[DeltaEvent],
        current_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply events, persist to DB atomically, return updated snapshot.

        Within a **single** transaction:
        1. Insert each ``DeltaEvent`` into ``clinical_events``.
        2. Compute the updated snapshot via ``apply_events()``.
        3. Upsert the ``context_snapshots`` row.

        The caller must supply an ``AsyncSession`` that is **not** yet
        committed; we add work to it but do NOT commit — the caller (or the
        FastAPI dependency) is responsible for committing.
        """
        cipher = await require_cipher(session, session_id)
        # 1. Compute new snapshot (pure logic)
        updated_snapshot = self.apply_events(current_snapshot, events)

        # 2. Persist each event row (encrypted payload, strictly increasing timestamps)
        now = utcnow()
        for offset, event in enumerate(events):
            try:
                enum_val = EventTypeEnum(event_type_value(event))
            except ValueError:
                logger.warning("unknown_event_type_skipped", event_type=event_type_value(event))
                continue
            session.add(ClinicalEvent(
                event_id=uuid.uuid4(),
                session_id=session_id,
                event_type=enum_val,
                payload=cipher.encrypt_json(event.payload),
                created_at=now + timedelta(microseconds=offset),
            ))

        # 3. Upsert context snapshot
        existing = await session.get(ContextSnapshot, session_id)
        if existing is not None:
            existing.snapshot = cipher.encrypt_json(updated_snapshot)
            existing.updated_at = now
        else:
            session.add(ContextSnapshot(
                session_id=session_id,
                snapshot=cipher.encrypt_json(updated_snapshot),
                soft_ask_count=0,
            ))

        return updated_snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_single(
        self,
        snapshot: dict[str, Any],
        event: DeltaEvent,
    ) -> dict[str, Any]:
        payload = event.payload
        match event_type_value(event):
            case "BP_READING":
                readings = snapshot.setdefault("recent_bp_readings", [])
                readings.append(payload)
                snapshot["recent_bp_readings"] = readings[-MAX_BP_READINGS:]

            case "MEDICATION_ADDED":
                meds = snapshot.setdefault("current_medications", [])
                drug_name = _first(payload, "drug_name", "drug", "name", "medication")
                if drug_name:
                    name_lower = str(drug_name).strip().lower()
                    if not any(str(m.get("drug", "")).strip().lower() == name_lower for m in meds):
                        meds.append({
                            "drug": str(drug_name).strip(),
                            "dosage": str(_first(payload, "dosage", "dose") or "").strip(),
                            "frequency": str(payload.get("frequency") or "").strip(),
                        })

            case "MEDICATION_STOPPED":
                drug_name = _first(payload, "drug_name", "drug", "name", "medication")
                if drug_name:
                    name_lower = str(drug_name).strip().lower()
                    current = snapshot.get("current_medications", [])
                    stopped = [
                        m for m in current if str(m.get("drug", "")).strip().lower() == name_lower
                    ]
                    snapshot["current_medications"] = [
                        m for m in current if str(m.get("drug", "")).strip().lower() != name_lower
                    ]
                    disc = snapshot.setdefault("discontinued_medications", [])
                    for m in stopped:
                        disc.append({
                            **m,
                            "reason": str(payload.get("reason") or ""),
                            # Taken from the event so the projection stays a pure fold.
                            "stopped_at": _first(payload, "stopped_at", "date", "recorded_at"),
                        })

            case "SYMPTOM_REPORTED":
                symptoms = snapshot.setdefault("symptoms", [])
                new = str(_first(payload, "symptom", "name") or "").strip()
                if new and new.lower() not in [str(s).strip().lower() for s in symptoms]:
                    symptoms.append(new)

            case "DEMOGRAPHIC":
                demo = snapshot.setdefault("demographics", {})
                demo.update(payload)

            case "LAB_RESULT":
                labs = [*snapshot.get("lab_results", []), payload]
                # Keep the last MAX_LABS_PER_TYPE results of each lab type, in
                # chronological order (labs without a type form their own group).
                seen: dict[str, int] = {}
                kept_reversed: list[dict[str, Any]] = []
                for lab in reversed(labs):
                    key = _lab_key(lab)
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] <= MAX_LABS_PER_TYPE:
                        kept_reversed.append(lab)
                snapshot["lab_results"] = list(reversed(kept_reversed))

        return snapshot

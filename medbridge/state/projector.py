"""
Deterministic State Projector (ADL-001).

Applies typed delta events to patient context snapshots using pure Python
deterministic merge rules. Does NOT use the LLM.

Persistence: ``persist_and_project()`` inserts clinical events and upserts
the snapshot within a single database transaction (atomic).

Technical Specification Part IV §3.3 / §4.1 Projection Rules.
"""
import copy
import uuid
from typing import Any

import structlog

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.db.models import ClinicalEvent, ContextSnapshot, EventTypeEnum

logger = structlog.get_logger(__name__)


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
        # 1. Compute new snapshot (pure logic)
        updated_snapshot = self.apply_events(current_snapshot, events)

        # 2. Persist each event row
        for event in events:
            raw_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            try:
                enum_val = EventTypeEnum(raw_type)
            except ValueError:
                continue
            row = ClinicalEvent(
                event_id=uuid.uuid4(),
                session_id=session_id,
                event_type=enum_val,
                payload=event.payload,
            )
            session.add(row)

        # 3. Upsert context snapshot
        existing = await session.get(ContextSnapshot, session_id)
        if existing is not None:
            existing.snapshot = updated_snapshot
        else:
            new_snap = ContextSnapshot(
                session_id=session_id,
                snapshot=updated_snapshot,
                soft_ask_count=0,
            )
            session.add(new_snap)

        return updated_snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_single(
        self,
        snapshot: dict[str, Any],
        event: DeltaEvent,
    ) -> dict[str, Any]:
        ev_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        match ev_type:
            case "BP_READING":
                readings = snapshot.setdefault("recent_bp_readings", [])
                readings.append(event.payload)
                snapshot["recent_bp_readings"] = readings[-5:]  # Keep last 5

            case "MEDICATION_ADDED":
                meds = snapshot.setdefault("current_medications", [])
                drug_name = (
                    event.payload.get("drug_name")
                    or event.payload.get("drug")
                    or event.payload.get("name")
                    or event.payload.get("medication")
                )
                if drug_name:
                    name_lower = str(drug_name).strip().lower()
                    if not any(str(m.get("drug", "")).strip().lower() == name_lower for m in meds):
                        meds.append({
                            "drug": str(drug_name).strip(),
                            "dosage": str(event.payload.get("dosage") or event.payload.get("dose") or "").strip(),
                            "frequency": str(event.payload.get("frequency") or "").strip(),
                        })

            case "MEDICATION_STOPPED":
                drug_name = (
                    event.payload.get("drug_name")
                    or event.payload.get("drug")
                    or event.payload.get("name")
                    or event.payload.get("medication")
                )
                if drug_name:
                    name_lower = str(drug_name).strip().lower()
                    current = snapshot.get("current_medications", [])
                    stopped = [m for m in current if str(m.get("drug", "")).strip().lower() == name_lower]
                    snapshot["current_medications"] = [
                        m for m in current if str(m.get("drug", "")).strip().lower() != name_lower
                    ]
                    disc = snapshot.setdefault("discontinued_medications", [])
                    for m in stopped:
                        disc.append({**m, "reason": event.payload.get("reason", "")})

            case "SYMPTOM_REPORTED":
                symptoms = snapshot.setdefault("symptoms", [])
                symptom_val = (
                    event.payload.get("symptom")
                    or event.payload.get("name")
                    or ""
                )
                new = str(symptom_val).strip()
                if new and new.lower() not in [str(s).strip().lower() for s in symptoms]:
                    symptoms.append(new)

            case "DEMOGRAPHIC":
                demo = snapshot.setdefault("demographics", {})
                demo.update(event.payload)

            case "LAB_RESULT":
                labs = snapshot.setdefault("lab_results", [])
                labs.append(event.payload)
                # Keep last 3 per lab type
                lab_type = event.payload.get("lab_type", "")
                if lab_type:
                    # Group by lab_type, keep last 3 of each
                    by_type: dict[str, list] = {}
                    for lab in labs:
                        lt = lab.get("lab_type", "")
                        by_type.setdefault(lt, []).append(lab)
                    for lt in by_type:
                        by_type[lt] = by_type[lt][-3:]
                    # Flatten back preserving insertion order
                    snapshot["lab_results"] = [
                        lab for lt_labs in by_type.values() for lab in lt_labs
                    ]
                else:
                    snapshot["lab_results"] = labs[-3:]

        return snapshot

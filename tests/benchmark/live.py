"""
Live-mode plumbing shared by the MedBridge-AQ, RABBITS and adversarial runners.

* :func:`resolve_eval_database_url` — ``--database-url`` → ``$EVAL_DATABASE_URL`` →
  ``EVAL_DATABASE_URL`` in ``.env``. Live runs never fall back to the development
  database: without an evaluation URL they refuse to start.
* :func:`configure_eval_environment` — points this process (only) at that
  database before any pipeline object is built.
* :func:`snapshot_to_events` — converts a vignette ``patient_snapshot`` into the
  typed clinical events the production pipeline would have recorded
  (``DEMOGRAPHIC`` / ``BP_READING`` / ``MEDICATION_ADDED`` / ``SYMPTOM_REPORTED`` /
  ``LAB_RESULT``). Medication names are normalised brand → generic exactly as the
  Context Extractor post-processing does.
* :class:`LivePipelineExecutor` — one NEW session per item: create the session,
  append the seeded events through ``turn_store.append_manual_events`` (the same
  projector the pipeline uses), set the soft-ask counter with
  ``session_manager.increment_soft_ask_count``, commit, then run the query through
  the default production ``PipelineOrchestrator.process_message(..., db=None)``.

Earlier conversation turns are not replayed: a vignette's prior context is the
seeded snapshot and counter only.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import EventTypeEnum

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


class LiveConfigError(RuntimeError):
    """Live mode cannot start (missing or unsafe configuration)."""


# ---------------------------------------------------------------------------
# Database selection
# ---------------------------------------------------------------------------


def _dotenv_value(key: str, env_file: Path = ENV_FILE) -> str | None:
    if not env_file.exists():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv ships with pydantic-settings
        return None
    value = dotenv_values(env_file).get(key)
    return str(value) if value else None


def database_name(url: str) -> str:
    """Database name of an SQLAlchemy URL (``""`` if absent)."""
    return urlparse(url.replace("+asyncpg", "")).path.lstrip("/")


def mask_url(url: str) -> str:
    """URL with the password replaced by ``***`` (safe for reports and logs)."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1***@", url)


def resolve_eval_database_url(cli_value: str | None = None, env_file: Path = ENV_FILE) -> str:
    """Evaluation database URL; refuses the development database.

    Raises:
        LiveConfigError: no evaluation URL configured, or it names the dev database.
    """
    url = (
        cli_value
        or os.environ.get("EVAL_DATABASE_URL")
        or _dotenv_value("EVAL_DATABASE_URL", env_file)
    )
    if not url:
        raise LiveConfigError(
            "No evaluation database: pass --database-url or set EVAL_DATABASE_URL "
            "(live runs never write to the development database)."
        )
    dev_url = os.environ.get("MEDBRIDGE_DEV_DATABASE_URL") or _dotenv_value(
        "DATABASE_URL", env_file
    )
    if dev_url and database_name(dev_url) == database_name(url):
        raise LiveConfigError(
            f"Refusing to run against the development database '{database_name(url)}'; "
            "use the evaluation database (EVAL_DATABASE_URL)."
        )
    return url


def configure_eval_environment(database_url: str, model: str | None = None) -> None:
    """Point this process at ``database_url`` (and optionally one model for every role)."""
    os.environ["DATABASE_URL"] = database_url
    if model:
        os.environ["LLM_MODEL"] = model
        os.environ["LLM_MODEL_GATES"] = ""
        os.environ["LLM_MODEL_GENERATOR"] = ""
    from medbridge.config import get_settings

    get_settings.cache_clear()


def run_metadata(database_url: str | None = None) -> dict[str, Any]:
    """Models, database (masked) and code version for report headers."""
    import subprocess

    from medbridge.config import get_settings

    settings = get_settings()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        sha, dirty = "", False
    return {
        "git_commit": sha + ("+dirty" if dirty and sha else ""),
        "llm_provider": settings.LLM_PROVIDER,
        "gates_model": settings.gates_model,
        "generator_model": settings.generator_model,
        "temperature_gates": settings.LLM_TEMPERATURE_GATES,
        "temperature_generator": settings.LLM_TEMPERATURE_GENERATOR,
        "seed": settings.LLM_SEED,
        "database": mask_url(database_url or settings.DATABASE_URL),
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "grounding_verifier": settings.GROUNDING_VERIFIER_ENABLED,
        "calibrated_evidence_gate": settings.EVIDENCE_GATE_CALIBRATION_ENABLED,
    }


# ---------------------------------------------------------------------------
# Snapshot → typed events
# ---------------------------------------------------------------------------

_NUMBER_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*)$")


def _split_value(value: Any) -> tuple[Any, str]:
    """``"4.6 mEq/L"`` → ``(4.6, "mEq/L")``; numbers pass through."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value, ""
    match = _NUMBER_UNIT.match(str(value))
    if not match:
        return str(value), ""
    number = float(match.group(1))
    return (int(number) if number.is_integer() else number), match.group(2).strip()


def _medication_payload(med: dict[str, Any]) -> dict[str, Any] | None:
    from medbridge.nlp.drugs import normalize_drug

    name = str(med.get("drug") or med.get("drug_name") or med.get("name") or "").strip()
    if not name:
        return None
    payload: dict[str, Any] = {
        "drug_name": name,
        "dosage": str(med.get("dosage") or med.get("dose") or "").strip(),
        "frequency": str(med.get("frequency") or "").strip(),
    }
    info = normalize_drug(name)
    if info is not None:  # same canonicalisation as the extractor post-processing
        payload["drug_name"] = info.generic
        if info.brand:
            payload["brand_name"] = info.brand
        payload["drug_class"] = info.drug_class
    return payload


def snapshot_to_events(snapshot: dict[str, Any]) -> list[DeltaEvent]:
    """Typed events that rebuild ``snapshot`` through the production projector.

    Mapping: ``demographics`` → DEMOGRAPHIC (a ``pregnancy_weeks``/``pregnant`` key
    also sets the canonical ``pregnancy_status``); ``recent_bp_readings``
    (``sbp``/``dbp`` or ``systolic``/``diastolic``) → BP_READING;
    ``current_medications`` → MEDICATION_ADDED (brand → generic); ``symptoms`` →
    SYMPTOM_REPORTED; ``lab_results`` → LAB_RESULT (``"4.6 mEq/L"`` → value + unit).
    """
    events: list[DeltaEvent] = []
    demographics = dict(snapshot.get("demographics") or {})
    if demographics:
        if "pregnancy_status" not in demographics and (
            demographics.get("pregnancy_weeks") or demographics.get("pregnant") is True
        ):
            demographics["pregnancy_status"] = "pregnant"
        events.append(DeltaEvent(event_type=EventTypeEnum.DEMOGRAPHIC, payload=demographics))
    for reading in snapshot.get("recent_bp_readings") or []:
        systolic = reading.get("systolic", reading.get("sbp"))
        diastolic = reading.get("diastolic", reading.get("dbp"))
        if systolic is None or diastolic is None:
            continue
        payload: dict[str, Any] = {"systolic": int(systolic), "diastolic": int(diastolic)}
        if reading.get("setting"):
            payload["setting"] = str(reading["setting"])
        events.append(DeltaEvent(event_type=EventTypeEnum.BP_READING, payload=payload))
    for med in snapshot.get("current_medications") or []:
        med_payload = _medication_payload(med if isinstance(med, dict) else {"drug": med})
        if med_payload is not None:
            events.append(
                DeltaEvent(event_type=EventTypeEnum.MEDICATION_ADDED, payload=med_payload)
            )
    for symptom in snapshot.get("symptoms") or []:
        if str(symptom).strip():
            events.append(
                DeltaEvent(
                    event_type=EventTypeEnum.SYMPTOM_REPORTED, payload={"symptom": str(symptom)}
                )
            )
    for lab in snapshot.get("lab_results") or []:
        lab_type = lab.get("lab_type") or lab.get("name")
        if not lab_type:
            continue
        value, unit = _split_value(lab.get("value"))
        events.append(
            DeltaEvent(
                event_type=EventTypeEnum.LAB_RESULT,
                payload={
                    "lab_type": str(lab_type),
                    "value": value,
                    "unit": lab.get("unit") or unit,
                },
            )
        )
    return events


# ---------------------------------------------------------------------------
# Live executor
# ---------------------------------------------------------------------------


@dataclass
class LiveTurn:
    """One live pipeline turn."""

    session_id: uuid.UUID
    response: Any  # MessageResponse
    latency_ms: float
    seeded_events: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class LivePipelineExecutor:
    """Runs items through the real production pipeline, one new session each."""

    def __init__(self, orchestrator: Any | None = None) -> None:
        from medbridge.core.orchestrator import PipelineOrchestrator

        self._orchestrator = orchestrator or PipelineOrchestrator()

    @staticmethod
    async def new_session(
        snapshot: dict[str, Any] | None = None, soft_ask_count: int = 0
    ) -> tuple[uuid.UUID, int]:
        """Create a session, seed its snapshot as events and its counter; commit."""
        from medbridge.db.connection import get_sessionmaker
        from medbridge.state import turn_store
        from medbridge.state.session_manager import create_session, increment_soft_ask_count

        events = snapshot_to_events(snapshot or {})
        async with get_sessionmaker()() as db:
            session_id = await create_session(db)
            if events:
                await turn_store.append_manual_events(db, session_id, events)
            for _ in range(max(0, soft_ask_count)):
                await increment_soft_ask_count(db, session_id)
            await db.commit()
        return session_id, len(events)

    async def run(
        self, message: str, snapshot: dict[str, Any] | None = None, soft_ask_count: int = 0
    ) -> LiveTurn:
        """Seed a fresh session and process ``message`` with ``db=None`` (own transaction)."""
        session_id, seeded = await self.new_session(snapshot, soft_ask_count)
        started = time.perf_counter()
        response = await self._orchestrator.process_message(session_id, message, None)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LiveTurn(
            session_id=session_id,
            response=response,
            latency_ms=latency_ms,
            seeded_events=seeded,
        )

    async def close(self) -> None:
        """Release the LLM client, Qdrant client and DB pool (best effort)."""
        from medbridge.ai.llm_wrapper import get_llm_wrapper
        from medbridge.db.connection import close_db
        from medbridge.retrieval.hybrid_retriever import get_hybrid_retriever

        for closer in (get_llm_wrapper().close, get_hybrid_retriever().close, close_db):
            try:
                await closer()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass

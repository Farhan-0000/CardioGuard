"""
Blood-pressure categories (ACC/AHA) for the BP log, trends and summaries.

Deterministic thresholds (systolic/diastolic, mmHg), checked from the most
severe category down — a reading belongs to the higher category when systolic
and diastolic fall into different ones:

=================  ===========================================
Category            Rule
=================  ===========================================
``crisis``          systolic ≥ 180 **or** diastolic ≥ 120 (severe)
``stage2``          systolic ≥ 140 **or** diastolic ≥ 90
``stage1``          systolic 130–139 **or** diastolic 80–89
``elevated``        systolic 120–129 **and** diastolic < 80
``normal``          systolic < 120 **and** diastolic < 80
=================  ===========================================

The categorisation is informational. A reading in the severe range does not
replace the emergency fast-path; it only flags the entry so the UI can show the
emergency numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from medbridge.state.turn_store import EventRow


@dataclass(frozen=True)
class BPCategory:
    key: str
    label: str
    severity: int  # 0 = normal … 4 = crisis


NORMAL = BPCategory("normal", "Normal", 0)
ELEVATED = BPCategory("elevated", "Elevated", 1)
STAGE1 = BPCategory("stage1", "Stage 1 hypertension", 2)
STAGE2 = BPCategory("stage2", "Stage 2 hypertension", 3)
CRISIS = BPCategory("crisis", "Severe / hypertensive crisis range", 4)
CATEGORIES: tuple[BPCategory, ...] = (NORMAL, ELEVATED, STAGE1, STAGE2, CRISIS)

#: Band definitions for charts (lower bounds; a band applies when either bound is reached,
#: except "elevated", which also requires diastolic < 80).
CATEGORY_LEGEND: list[dict[str, Any]] = [
    {"key": NORMAL.key, "label": NORMAL.label, "rule": "<120 and <80"},
    {"key": ELEVATED.key, "label": ELEVATED.label, "rule": "120–129 and <80"},
    {"key": STAGE1.key, "label": STAGE1.label, "rule": "130–139 or 80–89"},
    {"key": STAGE2.key, "label": STAGE2.label, "rule": "≥140 or ≥90"},
    {"key": CRISIS.key, "label": CRISIS.label, "rule": "≥180 or ≥120"},
]

CRISIS_ADVICE = (
    "This reading is in the severe range (180/120 mmHg or higher). If you have chest pain, "
    "shortness of breath, weakness or numbness, difficulty speaking, vision changes or a severe "
    "headache, call {emergency} (ambulance {ambulance}) now. If you have none of these, sit "
    "quietly for 5 minutes and measure again; if it is still this high, contact your doctor "
    "immediately."
)


def classify_bp(systolic: float, diastolic: float) -> BPCategory:
    """Return the ACC/AHA category of a reading."""
    if systolic >= 180 or diastolic >= 120:
        return CRISIS
    if systolic >= 140 or diastolic >= 90:
        return STAGE2
    if systolic >= 130 or diastolic >= 80:
        return STAGE1
    if systolic >= 120:
        return ELEVATED
    return NORMAL


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


_READING_RE = re.compile(r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})")


def reading_values(payload: dict[str, Any]) -> tuple[float, float, float | None] | None:
    """Extract (systolic, diastolic, pulse) from a BP_READING payload, or ``None``."""
    systolic = _number(payload.get("systolic", payload.get("sys")))
    diastolic = _number(payload.get("diastolic", payload.get("dia")))
    if systolic is None or diastolic is None:
        m = _READING_RE.search(str(payload.get("reading") or payload.get("value") or ""))
        if not m:
            return None
        systolic, diastolic = float(m.group(1)), float(m.group(2))
    pulse = _number(payload.get("pulse", payload.get("heart_rate")))
    return systolic, diastolic, pulse


def _compact(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _recorded_at(payload: dict[str, Any], fallback: datetime) -> str:
    for key in ("measured_at", "recorded_at"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback.isoformat()


def make_reading(systolic: float, diastolic: float, pulse: float | None, recorded_at: str,
                 source: str) -> dict[str, Any]:
    """A BP reading in the API/summary shape, with its ACC/AHA category."""
    category = classify_bp(systolic, diastolic)
    return {
        "systolic": _compact(float(systolic)),
        "diastolic": _compact(float(diastolic)),
        "pulse": _compact(float(pulse)) if pulse is not None else None,
        "recorded_at": recorded_at,
        "source": source,
        "category": category.key,
        "category_label": category.label,
    }


def bp_reading(payload: dict[str, Any], created_at: datetime) -> dict[str, Any] | None:
    """Normalise one BP_READING payload into the API/summary shape (``None`` if unusable)."""
    values = reading_values(payload)
    if values is None:
        return None
    systolic, diastolic, pulse = values
    source = "manual" if payload.get("source") == "manual" else "chat"
    return make_reading(systolic, diastolic, pulse, _recorded_at(payload, created_at), source)


def bp_series(events: list[EventRow]) -> list[dict[str, Any]]:
    """Chronological BP readings with categories from BP_READING events."""
    series: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "BP_READING":
            continue
        reading = bp_reading(event.payload, event.created_at)
        if reading is not None:
            series.append(reading)
    return series

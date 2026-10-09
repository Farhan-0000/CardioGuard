"""ACC/AHA blood-pressure categories and BP series normalisation."""
import uuid
from datetime import UTC, datetime

import pytest

from medbridge.features.bp import (
    CATEGORIES,
    CATEGORY_LEGEND,
    bp_reading,
    bp_series,
    classify_bp,
    make_reading,
    reading_values,
)
from medbridge.state.turn_store import EventRow


@pytest.mark.parametrize(
    ("systolic", "diastolic", "expected"),
    [
        (110, 70, "normal"),
        (119, 79, "normal"),
        (120, 79, "elevated"),
        (129, 70, "elevated"),
        (130, 70, "stage1"),
        (125, 80, "stage1"),   # diastolic decides
        (139, 89, "stage1"),
        (140, 85, "stage2"),
        (128, 90, "stage2"),
        (179, 119, "stage2"),
        (180, 100, "crisis"),
        (150, 120, "crisis"),
        (210, 130, "crisis"),
    ],
)
def test_acc_aha_categories(systolic: int, diastolic: int, expected: str) -> None:
    assert classify_bp(systolic, diastolic).key == expected


def test_category_metadata_is_ordered_by_severity() -> None:
    assert [c.severity for c in CATEGORIES] == [0, 1, 2, 3, 4]
    assert [c["key"] for c in CATEGORY_LEGEND] == [c.key for c in CATEGORIES]


def test_reading_values_accepts_common_payload_shapes() -> None:
    assert reading_values({"systolic": 150, "diastolic": 95, "pulse": 80}) == (150, 95, 80)
    assert reading_values({"sys": "142", "dia": "91"}) == (142, 91, None)
    assert reading_values({"reading": "160 over 100", "heart_rate": 72}) == (160, 100, 72)
    assert reading_values({"systolic": True, "diastolic": 90}) is None
    assert reading_values({"note": "no numbers"}) is None


def test_make_reading_shape() -> None:
    reading = make_reading(182.0, 121.0, 88.0, "2026-10-09T08:00:00+00:00", "manual")
    assert reading == {
        "systolic": 182, "diastolic": 121, "pulse": 88,
        "recorded_at": "2026-10-09T08:00:00+00:00", "source": "manual",
        "category": "crisis", "category_label": "Severe / hypertensive crisis range",
    }


def test_bp_series_uses_measured_at_or_event_time() -> None:
    created = datetime(2026, 10, 9, 7, 30, tzinfo=UTC)
    events = [
        EventRow(uuid.uuid4(), "BP_READING", {"systolic": 138, "diastolic": 86}, created),
        EventRow(uuid.uuid4(), "SYMPTOM_REPORTED", {"symptom": "headache"}, created),
        EventRow(uuid.uuid4(), "BP_READING",
                 {"systolic": 120, "diastolic": 75, "source": "manual",
                  "measured_at": "2026-10-08T21:00:00+00:00"}, created),
        EventRow(uuid.uuid4(), "BP_READING", {"unparseable": True}, created),
    ]
    series = bp_series(events)
    assert [(r["category"], r["source"], r["recorded_at"]) for r in series] == [
        ("stage1", "chat", created.isoformat()),
        ("elevated", "manual", "2026-10-08T21:00:00+00:00"),
    ]
    assert bp_reading({"systolic": 120.5, "diastolic": 80}, created)["systolic"] == 120.5

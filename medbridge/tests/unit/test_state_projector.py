"""
Unit tests for TASK-13: Deterministic State Projector.

Tests all 6 event projection rules individually, truncation limits,
deduplication logic, and snapshot reconstruction.
"""
import copy

import pytest

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import EventTypeEnum
from medbridge.state.projector import StateProjector


@pytest.fixture
def projector() -> StateProjector:
    return StateProjector()


@pytest.fixture
def empty_snapshot() -> dict:
    return {}


# ---------------------------------------------------------------------------
# BP_READING
# ---------------------------------------------------------------------------

class TestBpReading:
    """Projection rule: Append to recent_bp_readings[], keep last 5."""

    def test_append_single_reading(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.BP_READING,
            payload={"systolic": 140, "diastolic": 90},
        )
        result = projector.apply_events(empty_snapshot, [event])
        assert len(result["recent_bp_readings"]) == 1
        assert result["recent_bp_readings"][0] == {"systolic": 140, "diastolic": 90}

    def test_append_multiple_readings(self, projector, empty_snapshot):
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.BP_READING,
                payload={"systolic": 130 + i, "diastolic": 80 + i},
            )
            for i in range(3)
        ]
        result = projector.apply_events(empty_snapshot, events)
        assert len(result["recent_bp_readings"]) == 3

    def test_truncation_at_5(self, projector, empty_snapshot):
        """After 7 readings, only the last 5 should remain."""
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.BP_READING,
                payload={"systolic": 120 + i, "diastolic": 70 + i},
            )
            for i in range(7)
        ]
        result = projector.apply_events(empty_snapshot, events)
        assert len(result["recent_bp_readings"]) == 5
        # First two should be dropped (i=0,1 gone; i=2..6 remain)
        assert result["recent_bp_readings"][0] == {"systolic": 122, "diastolic": 72}
        assert result["recent_bp_readings"][-1] == {"systolic": 126, "diastolic": 76}

    def test_preserves_existing_readings(self, projector):
        snapshot = {"recent_bp_readings": [{"systolic": 100, "diastolic": 60}]}
        event = DeltaEvent(
            event_type=EventTypeEnum.BP_READING,
            payload={"systolic": 140, "diastolic": 90},
        )
        result = projector.apply_events(snapshot, [event])
        assert len(result["recent_bp_readings"]) == 2


# ---------------------------------------------------------------------------
# MEDICATION_ADDED
# ---------------------------------------------------------------------------

class TestMedicationAdded:
    """Projection rule: Add to current_medications[] if not present (case-insensitive)."""

    def test_add_new_medication(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_ADDED,
            payload={"drug_name": "Amlodipine", "dosage": "5mg", "frequency": "daily"},
        )
        result = projector.apply_events(empty_snapshot, [event])
        assert len(result["current_medications"]) == 1
        assert result["current_medications"][0]["drug"] == "Amlodipine"
        assert result["current_medications"][0]["dosage"] == "5mg"
        assert result["current_medications"][0]["frequency"] == "daily"

    def test_case_insensitive_deduplication(self, projector, empty_snapshot):
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "Amlodipine", "dosage": "5mg"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "amlodipine", "dosage": "10mg"},
            ),
        ]
        result = projector.apply_events(empty_snapshot, events)
        assert len(result["current_medications"]) == 1
        # Same drug (case-insensitive): keeps the original name, records a dose change
        med = result["current_medications"][0]
        assert med["drug"] == "Amlodipine"
        assert med["dosage"] == "10mg"
        assert med["history"] == [{"dosage": "5mg", "frequency": ""}]

    def test_adds_different_medications(self, projector, empty_snapshot):
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "Amlodipine"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "Lisinopril"},
            ),
        ]
        result = projector.apply_events(empty_snapshot, events)
        assert len(result["current_medications"]) == 2

    def test_defaults_for_optional_fields(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_ADDED,
            payload={"drug_name": "Metoprolol"},
        )
        result = projector.apply_events(empty_snapshot, [event])
        med = result["current_medications"][0]
        assert med["dosage"] == ""
        assert med["frequency"] == ""


# ---------------------------------------------------------------------------
# MEDICATION_STOPPED
# ---------------------------------------------------------------------------

class TestMedicationStopped:
    """Projection rule: Move from current to discontinued with reason."""

    def test_stop_existing_medication(self, projector):
        snapshot = {
            "current_medications": [
                {"drug": "Amlodipine", "dosage": "5mg", "frequency": "daily"},
            ]
        }
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_STOPPED,
            payload={"drug_name": "Amlodipine", "reason": "Side effects"},
        )
        result = projector.apply_events(snapshot, [event])
        assert len(result["current_medications"]) == 0
        assert len(result["discontinued_medications"]) == 1
        assert result["discontinued_medications"][0]["drug"] == "Amlodipine"
        assert result["discontinued_medications"][0]["reason"] == "Side effects"

    def test_case_insensitive_matching(self, projector):
        snapshot = {
            "current_medications": [
                {"drug": "Amlodipine", "dosage": "5mg", "frequency": "daily"},
            ]
        }
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_STOPPED,
            payload={"drug_name": "amlodipine", "reason": "Ineffective"},
        )
        result = projector.apply_events(snapshot, [event])
        assert len(result["current_medications"]) == 0
        assert len(result["discontinued_medications"]) == 1

    def test_stop_nonexistent_medication_is_noop(self, projector):
        snapshot = {
            "current_medications": [
                {"drug": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
            ]
        }
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_STOPPED,
            payload={"drug_name": "Amlodipine", "reason": "N/A"},
        )
        result = projector.apply_events(snapshot, [event])
        assert len(result["current_medications"]) == 1
        assert len(result.get("discontinued_medications", [])) == 0

    def test_stop_preserves_other_medications(self, projector):
        snapshot = {
            "current_medications": [
                {"drug": "Amlodipine", "dosage": "5mg", "frequency": "daily"},
                {"drug": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
            ]
        }
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_STOPPED,
            payload={"drug_name": "Amlodipine", "reason": "Switched"},
        )
        result = projector.apply_events(snapshot, [event])
        assert len(result["current_medications"]) == 1
        assert result["current_medications"][0]["drug"] == "Lisinopril"


# ---------------------------------------------------------------------------
# SYMPTOM_REPORTED
# ---------------------------------------------------------------------------

class TestSymptomReported:
    """Projection rule: Append to symptoms[], deduplicated (case-insensitive)."""

    def test_append_new_symptom(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.SYMPTOM_REPORTED,
            payload={"symptom": "Headache"},
        )
        result = projector.apply_events(empty_snapshot, [event])
        assert "Headache" in result["symptoms"]

    def test_case_insensitive_deduplication(self, projector, empty_snapshot):
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.SYMPTOM_REPORTED,
                payload={"symptom": "Headache"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.SYMPTOM_REPORTED,
                payload={"symptom": "headache"},
            ),
        ]
        result = projector.apply_events(empty_snapshot, events)
        assert len(result["symptoms"]) == 1
        assert result["symptoms"][0] == "Headache"  # Original casing preserved

    def test_multiple_distinct_symptoms(self, projector, empty_snapshot):
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.SYMPTOM_REPORTED,
                payload={"symptom": "Headache"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.SYMPTOM_REPORTED,
                payload={"symptom": "Dizziness"},
            ),
        ]
        result = projector.apply_events(empty_snapshot, events)
        assert len(result["symptoms"]) == 2

    def test_empty_symptom_ignored(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.SYMPTOM_REPORTED,
            payload={"symptom": ""},
        )
        result = projector.apply_events(empty_snapshot, [event])
        assert len(result.get("symptoms", [])) == 0


# ---------------------------------------------------------------------------
# DEMOGRAPHIC
# ---------------------------------------------------------------------------

class TestDemographic:
    """Projection rule: Merge into demographics{}, overwrite existing fields."""

    def test_initial_demographic(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.DEMOGRAPHIC,
            payload={"age": 55, "sex": "male"},
        )
        result = projector.apply_events(empty_snapshot, [event])
        assert result["demographics"]["age"] == 55
        assert result["demographics"]["sex"] == "male"

    def test_overwrite_existing_fields(self, projector):
        snapshot = {"demographics": {"age": 55, "sex": "male"}}
        event = DeltaEvent(
            event_type=EventTypeEnum.DEMOGRAPHIC,
            payload={"age": 56},
        )
        result = projector.apply_events(snapshot, [event])
        assert result["demographics"]["age"] == 56
        assert result["demographics"]["sex"] == "male"  # Preserved

    def test_merge_new_fields(self, projector):
        snapshot = {"demographics": {"age": 55}}
        event = DeltaEvent(
            event_type=EventTypeEnum.DEMOGRAPHIC,
            payload={"weight_kg": 80, "ethnicity": "Asian"},
        )
        result = projector.apply_events(snapshot, [event])
        assert result["demographics"]["age"] == 55
        assert result["demographics"]["weight_kg"] == 80
        assert result["demographics"]["ethnicity"] == "Asian"


# ---------------------------------------------------------------------------
# LAB_RESULT
# ---------------------------------------------------------------------------

class TestLabResult:
    """Projection rule: Append to lab_results[], keep last 3 per lab type."""

    def test_append_single_lab(self, projector, empty_snapshot):
        event = DeltaEvent(
            event_type=EventTypeEnum.LAB_RESULT,
            payload={"lab_type": "potassium", "value": 4.5, "unit": "mEq/L"},
        )
        result = projector.apply_events(empty_snapshot, [event])
        assert len(result["lab_results"]) == 1

    def test_truncation_at_3_per_type(self, projector, empty_snapshot):
        """After 5 potassium results, only the last 3 should remain."""
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.LAB_RESULT,
                payload={"lab_type": "potassium", "value": 4.0 + i * 0.1, "unit": "mEq/L"},
            )
            for i in range(5)
        ]
        result = projector.apply_events(empty_snapshot, events)
        potassium_labs = [
            lab for lab in result["lab_results"] if lab["lab_type"] == "potassium"
        ]
        assert len(potassium_labs) == 3
        # Last 3 values should be i=2,3,4 → 4.2, 4.3, 4.4
        assert potassium_labs[0]["value"] == pytest.approx(4.2)
        assert potassium_labs[-1]["value"] == pytest.approx(4.4)

    def test_different_lab_types_independent(self, projector, empty_snapshot):
        """Different lab types have independent 3-item limits."""
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.LAB_RESULT,
                payload={"lab_type": "potassium", "value": 4.0 + i * 0.1, "unit": "mEq/L"},
            )
            for i in range(4)
        ] + [
            DeltaEvent(
                event_type=EventTypeEnum.LAB_RESULT,
                payload={"lab_type": "creatinine", "value": 1.0 + i * 0.1, "unit": "mg/dL"},
            )
            for i in range(2)
        ]
        result = projector.apply_events(empty_snapshot, events)

        potassium = [lab for lab in result["lab_results"] if lab["lab_type"] == "potassium"]
        creatinine = [lab for lab in result["lab_results"] if lab["lab_type"] == "creatinine"]
        assert len(potassium) == 3  # Truncated from 4 → 3
        assert len(creatinine) == 2  # Under limit, all kept


# ---------------------------------------------------------------------------
# Cross-cutting / Reconstruction
# ---------------------------------------------------------------------------

class TestReconstruction:
    """reconstruct_from_events() must match apply_events() from empty snapshot."""

    def test_reconstruct_matches_apply(self, projector):
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.DEMOGRAPHIC,
                payload={"age": 55, "sex": "male"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.BP_READING,
                payload={"systolic": 140, "diastolic": 90},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "Amlodipine", "dosage": "5mg"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.SYMPTOM_REPORTED,
                payload={"symptom": "Headache"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.LAB_RESULT,
                payload={"lab_type": "potassium", "value": 4.5, "unit": "mEq/L"},
            ),
        ]

        from_empty = projector.apply_events({}, events)
        reconstructed = projector.reconstruct_from_events(events)

        assert from_empty == reconstructed

    def test_reconstruct_complex_scenario(self, projector):
        """Full lifecycle: add med → add another → stop first → add symptom."""
        events = [
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "Amlodipine", "dosage": "5mg"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_ADDED,
                payload={"drug_name": "Lisinopril", "dosage": "10mg"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.MEDICATION_STOPPED,
                payload={"drug_name": "Amlodipine", "reason": "Edema"},
            ),
            DeltaEvent(
                event_type=EventTypeEnum.SYMPTOM_REPORTED,
                payload={"symptom": "Ankle swelling"},
            ),
        ]
        result = projector.reconstruct_from_events(events)

        assert len(result["current_medications"]) == 1
        assert result["current_medications"][0]["drug"] == "Lisinopril"
        assert len(result["discontinued_medications"]) == 1
        assert result["discontinued_medications"][0]["drug"] == "Amlodipine"
        assert result["discontinued_medications"][0]["reason"] == "Edema"
        assert "Ankle swelling" in result["symptoms"]

    def test_does_not_mutate_original(self, projector):
        """apply_events must deep-copy the input snapshot."""
        original = {"demographics": {"age": 55}}
        event = DeltaEvent(
            event_type=EventTypeEnum.DEMOGRAPHIC,
            payload={"age": 56},
        )
        result = projector.apply_events(original, [event])
        assert result["demographics"]["age"] == 56
        assert original["demographics"]["age"] == 55  # Unchanged


# ---------------------------------------------------------------------------
# Rule fixes (finalization): stopped_at on MEDICATION_STOPPED, labs per type
# ---------------------------------------------------------------------------

class TestMedicationStoppedRecordsWhenAndWhy:
    def test_stopped_at_and_reason_are_recorded(self, projector):
        snapshot = {"current_medications": [{"drug": "Amlodipine", "dosage": "5mg",
                                             "frequency": "daily"}]}
        event = DeltaEvent(
            event_type=EventTypeEnum.MEDICATION_STOPPED,
            payload={"drug_name": "amlodipine", "reason": "ankle swelling",
                     "stopped_at": "2026-10-01"},
        )
        stopped = projector.apply_events(snapshot, [event])["discontinued_medications"][0]
        assert stopped == {"drug": "Amlodipine", "dosage": "5mg", "frequency": "daily",
                           "reason": "ankle swelling", "stopped_at": "2026-10-01"}

    def test_stopped_at_is_none_when_unknown(self, projector):
        snapshot = {"current_medications": [{"drug": "Telma", "dosage": "", "frequency": ""}]}
        event = DeltaEvent(event_type=EventTypeEnum.MEDICATION_STOPPED,
                           payload={"drug": "Telma"})
        stopped = projector.apply_events(snapshot, [event])["discontinued_medications"][0]
        assert stopped["stopped_at"] is None and stopped["reason"] == ""

    def test_projection_stays_a_pure_fold(self, projector):
        events = [
            DeltaEvent(event_type=EventTypeEnum.MEDICATION_ADDED,
                       payload={"drug_name": "Amlodipine"}),
            DeltaEvent(event_type=EventTypeEnum.MEDICATION_STOPPED,
                       payload={"drug_name": "Amlodipine", "reason": "edema"}),
        ]
        assert projector.reconstruct_from_events(events) == projector.apply_events({}, events)


class TestLabResultKeepsLastThreePerType:
    @staticmethod
    def _lab(lab_type, value):
        payload = {"value": value}
        if lab_type is not None:
            payload["lab_type"] = lab_type
        return DeltaEvent(event_type=EventTypeEnum.LAB_RESULT, payload=payload)

    def test_untyped_lab_does_not_truncate_other_types(self, projector):
        events = [self._lab("potassium", 4.0), self._lab("potassium", 4.1),
                  self._lab("creatinine", 1.0), self._lab("creatinine", 1.1),
                  self._lab(None, 99)]
        labs = projector.apply_events({}, events)["lab_results"]
        assert len(labs) == 5  # previously truncated to the last 3 overall

    def test_chronological_order_is_preserved(self, projector):
        events = [self._lab("potassium", 4.0), self._lab("creatinine", 1.0),
                  self._lab("potassium", 4.1), self._lab("potassium", 4.2),
                  self._lab("potassium", 4.3), self._lab("creatinine", 1.1)]
        labs = projector.apply_events({}, events)["lab_results"]
        assert [lab["value"] for lab in labs] == [1.0, 4.1, 4.2, 4.3, 1.1]

    def test_lab_type_matching_is_case_insensitive_and_aliased(self, projector):
        events = [DeltaEvent(event_type=EventTypeEnum.LAB_RESULT, payload=p) for p in (
            {"lab_type": "eGFR", "value": 60}, {"lab": "egfr", "value": 58},
            {"test": "EGFR", "value": 55}, {"name": "eGFR ", "value": 52},
        )]
        labs = projector.apply_events({}, events)["lab_results"]
        assert [lab["value"] for lab in labs] == [58, 55, 52]

    def test_untyped_labs_keep_last_three(self, projector):
        events = [self._lab(None, v) for v in range(5)]
        labs = projector.apply_events({}, events)["lab_results"]
        assert [lab["value"] for lab in labs] == [2, 3, 4]


# ---------------------------------------------------------------------------
# Dose changes, normaliser labels and demographic profile (round 2)
# ---------------------------------------------------------------------------

def _med(drug: str, **payload: object) -> DeltaEvent:
    return DeltaEvent(event_type=EventTypeEnum.MEDICATION_ADDED,
                      payload={"drug_name": drug, **payload})


class TestDoseChange:
    def test_dose_change_updates_entry_and_records_history(self, projector):
        events = [
            _med("amlodipine", dosage="5 mg", frequency="once daily", brand_name="Amlong",
                 drug_class="calcium channel blocker"),
            _med("amlodipine", dosage="10 mg", frequency="", changed_at="2026-10-01"),
            _med("Amlodipine", dosage="10 mg", frequency="twice daily"),
        ]
        result = projector.apply_events({}, events)
        assert result["current_medications"] == [{
            "drug": "amlodipine",
            "dosage": "10 mg",
            "frequency": "twice daily",
            "brand_name": "Amlong",
            "drug_class": "calcium channel blocker",
            "history": [
                {"dosage": "5 mg", "frequency": "once daily", "changed_at": "2026-10-01"},
                {"dosage": "10 mg", "frequency": "once daily"},
            ],
        }]

    def test_same_dose_repeated_is_not_a_change(self, projector):
        events = [_med("telmisartan", dosage="40 mg"), _med("telmisartan", dosage="40 mg"),
                  _med("telmisartan")]
        med = projector.apply_events({}, events)["current_medications"][0]
        assert med["dosage"] == "40 mg" and "history" not in med

    def test_labels_kept_and_filled_in_later(self, projector):
        events = [_med("metoprolol", dosage="25 mg"),
                  _med("metoprolol", brand_name="Betaloc", drug_class="beta blocker")]
        med = projector.apply_events({}, events)["current_medications"][0]
        assert med["brand_name"] == "Betaloc" and med["drug_class"] == "beta blocker"
        assert med["dosage"] == "25 mg" and "history" not in med

    def test_dose_change_is_a_pure_deterministic_fold(self, projector):
        events = [_med("ramipril", dosage="2.5 mg"), _med("ramipril", dosage="5 mg")]
        snapshot = {"current_medications": []}
        first = projector.apply_events(snapshot, events)
        assert first == projector.apply_events(snapshot, events)
        assert first == projector.reconstruct_from_events(events)
        assert snapshot == {"current_medications": []}  # input not mutated

    def test_stop_keeps_dose_history(self, projector):
        events = [_med("lisinopril", dosage="10 mg"), _med("lisinopril", dosage="20 mg"),
                  DeltaEvent(event_type=EventTypeEnum.MEDICATION_STOPPED,
                             payload={"drug_name": "lisinopril", "reason": "dry cough"})]
        result = projector.apply_events({}, events)
        assert result["current_medications"] == []
        stopped = result["discontinued_medications"][0]
        assert stopped["dosage"] == "20 mg" and stopped["history"] == [
            {"dosage": "10 mg", "frequency": ""}
        ]


class TestDemographicProfile:
    def test_pregnancy_status_and_conditions_in_snapshot(self, projector):
        events = [
            DeltaEvent(event_type=EventTypeEnum.DEMOGRAPHIC,
                       payload={"age": 31, "sex": "female", "pregnancy_status": "pregnant",
                                "conditions": ["diabetes"]}),
            DeltaEvent(event_type=EventTypeEnum.DEMOGRAPHIC,
                       payload={"conditions": ["Diabetes", "chronic kidney disease"]}),
        ]
        demo = projector.apply_events({}, events)["demographics"]
        assert demo == {"age": 31, "sex": "female", "pregnancy_status": "pregnant",
                        "conditions": ["diabetes", "chronic kidney disease"]}

    def test_condition_string_and_bad_types(self, projector):
        events = [
            DeltaEvent(event_type=EventTypeEnum.DEMOGRAPHIC, payload={"conditions": "asthma"}),
            DeltaEvent(event_type=EventTypeEnum.DEMOGRAPHIC, payload={"conditions": 5}),
        ]
        demo = projector.apply_events({}, events)["demographics"]
        assert demo["conditions"] == ["asthma"]

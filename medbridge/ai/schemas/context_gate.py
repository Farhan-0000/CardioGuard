"""Context Gate (LLM Call 2, Gate 1) output schema."""
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

# Fixed vocabulary of patient facts Gate 1 may ask for (drives the SOFT-ASK wording).
MissingField = Literal[
    "current_medications",
    "blood_pressure_reading",
    "age",
    "sex",
    "pregnancy_status",
    "kidney_function",
    "heart_rate",
    "symptoms_detail",
    "diagnosis",
    "medication_dose",
    "comorbidities",
    "lab_results",
]
MISSING_FIELD_VOCABULARY: tuple[str, ...] = get_args(MissingField)

# Free-text field names produced by older prompts / remote providers → vocabulary.
_FIELD_SYNONYMS: dict[str, str] = {
    "medications": "current_medications",
    "medication": "current_medications",
    "current_medication": "current_medications",
    "meds": "current_medications",
    "current_meds": "current_medications",
    "drugs": "current_medications",
    "medication_list": "current_medications",
    "medication_name": "current_medications",
    "drug_name": "current_medications",
    "bp": "blood_pressure_reading",
    "bp_reading": "blood_pressure_reading",
    "bp_readings": "blood_pressure_reading",
    "blood_pressure": "blood_pressure_reading",
    "blood_pressure_readings": "blood_pressure_reading",
    "recent_bp_readings": "blood_pressure_reading",
    "recent_bp": "blood_pressure_reading",
    "patient_age": "age",
    "gender": "sex",
    "pregnancy": "pregnancy_status",
    "pregnant": "pregnancy_status",
    "breastfeeding": "pregnancy_status",
    "kidney": "kidney_function",
    "renal_function": "kidney_function",
    "egfr": "kidney_function",
    "creatinine": "kidney_function",
    "ckd": "kidney_function",
    "pulse": "heart_rate",
    "hr": "heart_rate",
    "symptoms": "symptoms_detail",
    "symptom": "symptoms_detail",
    "symptom_detail": "symptoms_detail",
    "symptom_details": "symptoms_detail",
    "symptoms_details": "symptoms_detail",
    "diagnoses": "diagnosis",
    "condition": "diagnosis",
    "dose": "medication_dose",
    "dosage": "medication_dose",
    "current_dose": "medication_dose",
    "medication_dosage": "medication_dose",
    "comorbidity": "comorbidities",
    "conditions": "comorbidities",
    "other_conditions": "comorbidities",
    "medical_history": "comorbidities",
    "past_medical_history": "comorbidities",
    "labs": "lab_results",
    "lab_values": "lab_results",
    "blood_tests": "lab_results",
}


def normalize_missing_fields(values: Any) -> list[str]:
    """Map free-text field names onto the fixed vocabulary; drop unknowns and duplicates."""
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        field = key if key in MISSING_FIELD_VOCABULARY else _FIELD_SYNONYMS.get(key)
        if field and field not in out:
            out.append(field)
    return out


class ContextGateOutput(BaseModel):
    """Gate 1 decision. ``rationale`` comes first so the model reasons before deciding."""

    rationale: str
    action: Literal["SOFT-ASK", "PROCEED"]
    missing_fields: list[MissingField] = Field(default_factory=list)

    @field_validator("missing_fields", mode="before")
    @classmethod
    def _normalize_fields(cls, value: Any) -> list[str]:
        return normalize_missing_fields(value)

    @model_validator(mode="after")
    def _no_missing_fields_on_proceed(self) -> "ContextGateOutput":
        if self.action == "PROCEED":
            self.missing_fields = []
        return self

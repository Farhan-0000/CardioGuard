"""
Deterministic Response Templates (ADL-015, Technical Specification Appendix B).

Pre-vetted, immutable texts returned **without LLM generation**:

* ``ESCALATE`` and ``ABSTAIN`` (Non-Negotiable #6 — template bypass);
* the SOFT-ASK loop-breaker prefix (Non-Negotiable #4);
* the GENERALIZE fallback used when the generator fails, the pipeline-error
  fallback used by the never-throw orchestrator, and the SOFT-ASK fallback
  question used when the generator cannot phrase a clarifying question;
* deterministic small-talk replies (greetings, thanks, …) that contain no
  clinical claims.

All texts are conservative and claim-free: no drug names, doses, targets or
diagnoses. The emergency text uses the configurable emergency numbers
(``EMERGENCY_NUMBER`` / ``AMBULANCE_NUMBER`` / ``EMERGENCY_REGION_LABEL``) and
always mentions 911 for the United States.
"""

from __future__ import annotations

import re

from medbridge.api.schemas.enums import ActionEnum
from medbridge.config import Settings, get_settings


def build_escalate_template(settings: Settings) -> str:
    """Render the ESCALATE template with the configured emergency numbers."""
    return (
        "⚠️ IMPORTANT: Based on what you've described, this situation may need "
        "immediate medical attention. Please call emergency services now — "
        f"call {settings.EMERGENCY_NUMBER} (or {settings.AMBULANCE_NUMBER} for an "
        f"ambulance in {settings.EMERGENCY_REGION_LABEL}) — or go to the nearest "
        "emergency department. If you are in the United States, call 911.\n\n"
        "Do not wait to see whether the symptoms pass.\n\n"
        "This tool provides informational guidance only and cannot assess or manage "
        "medical emergencies."
    )


ESCALATE_TEMPLATE: str = build_escalate_template(get_settings())

ABSTAIN_TEMPLATE: str = (
    "I don't have sufficient information in my clinical guidelines to safely "
    "address this specific question. It may be outside the scope of the heart and "
    "blood-pressure care that I cover, or it may need specialised medical "
    "expertise.\n\n"
    "I recommend discussing this question directly with your healthcare provider, "
    "who can give you personalized guidance based on your complete medical history."
)

LOOP_BREAKER_PREFIX: str = (
    "I understand you may not have all your medical details handy. Here is some "
    "general guidance based on current heart and blood-pressure guidelines that "
    "may be helpful:"
)

GENERALIZE_FALLBACK_TEMPLATE: str = (
    "I wasn't able to put together a detailed, evidence-based answer to your "
    "question just now. General clinical guidelines for heart and blood-pressure "
    "care recommend that decisions about tests, medicines and lifestyle changes are "
    "made together with a healthcare professional who knows your full history.\n\n"
    "Please discuss your question with your doctor or pharmacist. If you have severe "
    "or sudden symptoms, contact emergency services right away."
)

PIPELINE_ERROR_TEMPLATE: str = (
    "I'm sorry, I ran into a problem while processing your question, so I can't "
    "give you guideline-based information right now. Please try again in a moment, "
    "and consult your healthcare provider directly for personalized guidance. If you "
    "have severe or sudden symptoms, contact emergency services right away."
)

SOFT_ASK_FALLBACK_TEMPLATE: str = (
    "To give you safe and relevant information, could you share a bit more about "
    "your situation — for example your recent blood pressure readings, the medicines "
    "you take, or the symptoms you have noticed?"
)

# Deterministic small-talk replies (core/smalltalk.py returns one of these keys).
SMALL_TALK_TEMPLATES: dict[str, str] = {
    "greeting": (
        "Hello! I'm MedBridge, an assistant that explains heart and blood-pressure "
        "topics using published clinical guidelines. You can ask me about blood "
        "pressure readings, heart and blood-pressure medicines, or everyday lifestyle "
        "questions. How can I help you today?"
    ),
    "thanks": (
        "You're welcome! If you have another question about blood pressure or heart "
        "health, just ask. For decisions about your own care, please also check with "
        "your healthcare provider."
    ),
    "goodbye": (
        "Take care! I'm here whenever you have more questions about blood pressure or "
        "heart health. If you ever have severe or sudden symptoms, contact emergency "
        "services right away."
    ),
    "identity": (
        "I'm MedBridge, an informational assistant for heart and blood-pressure care. "
        "I answer questions using published clinical guidelines and show you the "
        "sources I used. I can help you understand blood pressure readings, common "
        "heart and blood-pressure medicines, and lifestyle topics.\n\n"
        "I can't diagnose conditions or prescribe treatment, and I'm not a substitute "
        "for your doctor. If something you describe sounds urgent, I'll tell you to "
        "seek emergency care."
    ),
    "acknowledgement": (
        "Got it. Is there anything else you'd like to know about blood pressure or heart health?"
    ),
}

_FIELD_LABELS: dict[str, str] = {
    # Gate 1 missing-field vocabulary (ai/schemas/context_gate.py)
    "current_medications": "current medicines",
    "blood_pressure_reading": "recent blood pressure readings",
    "age": "age",
    "sex": "sex",
    "pregnancy_status": "pregnancy status",
    "kidney_function": "kidney function",
    "heart_rate": "heart rate",
    "symptoms_detail": "symptoms",
    "diagnosis": "diagnosis",
    "medication_dose": "medicine doses",
    "comorbidities": "other health conditions",
    "lab_results": "recent test results",
    # Legacy free-text names
    "medications": "current medicines",
    "bp_reading": "recent blood pressure readings",
    "symptoms": "symptoms",
}

# Gate-1 field names come from an LLM: only short, plain identifiers are echoed back.
_SAFE_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,39}$")

# Map of action → template for supported deterministic responses
_STATIC_TEMPLATES: dict[ActionEnum, str] = {ActionEnum.ABSTAIN: ABSTAIN_TEMPLATE}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def get_template(action: ActionEnum, settings: Settings | None = None) -> str:
    """Return the pre-vetted response template for ``ESCALATE`` or ``ABSTAIN``.

    Other actions (``ANSWER``, ``GENERALIZE``, ``SOFT-ASK``) are generated and
    do not have templates.

    Raises:
        ValueError: If the action does not have a deterministic template.
    """
    if action == ActionEnum.ESCALATE:
        return build_escalate_template(settings or get_settings())
    template = _STATIC_TEMPLATES.get(action)
    if template is None:
        raise ValueError(
            f"No deterministic template exists for action '{action.value}'. "
            f"Only ESCALATE and ABSTAIN have pre-vetted templates."
        )
    return template


def get_loop_breaker_prefix() -> str:
    """Return the loop-breaker transition prefix (Appendix B.3, Non-Negotiable #4)."""
    return LOOP_BREAKER_PREFIX


def get_small_talk_template(key: str) -> str:
    """Return the deterministic small-talk reply for ``key``.

    Raises:
        ValueError: If ``key`` is not a known small-talk category.
    """
    try:
        return SMALL_TALK_TEMPLATES[key]
    except KeyError:
        raise ValueError(f"Unknown small-talk template '{key}'") from None


def build_soft_ask_fallback(missing_fields: list[str] | None = None) -> str:
    """Clarifying question used when the generator cannot phrase the SOFT-ASK."""
    labels: list[str] = []
    for field in missing_fields or []:
        key = field.strip().lower()
        if key in _FIELD_LABELS:
            label = _FIELD_LABELS[key]
        elif _SAFE_FIELD.match(key):
            label = key.replace("_", " ")
        else:
            continue
        if label not in labels:
            labels.append(label)
    if not labels:
        return SOFT_ASK_FALLBACK_TEMPLATE
    listed = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
    return (
        "To give you safe and relevant information, could you tell me a little more "
        f"about your {listed}?"
    )

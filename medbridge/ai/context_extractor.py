"""
Context Extractor — LLM Call 1 (ADL-018, ADL-023).

Extracts structured delta events, a reformulated clinical search query,
and raw patient intent from incoming messages.

Security: Patient messages are strictly encapsulated within
``<untrusted_user_input>`` XML tags to mitigate prompt injection (ADL-018).

Technical Specification Part III §3.3.
"""
import json
import structlog
from typing import Any, Optional

from medbridge.ai.llm_wrapper import ResilientLLMWrapper, get_llm_wrapper
from medbridge.ai.schemas.extractor import ExtractorOutput
from medbridge.config import get_settings

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# System prompt (§3.3 — verbatim from Tech Spec)
# ------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a clinical context extraction engine. Given a patient's message and their "
    "existing medical context snapshot, extract NEW medical facts as structured delta events. "
    "If the patient message contains NO new clinical facts, BP readings, symptoms, or medications, "
    'set "delta_events": []. Do not invent event types.\n'
    "Also produce a reformulated clinical search query suitable for retrieving hypertension "
    "guidelines, and a brief summary of what the patient is asking.\n\n"
    "Output ONLY valid JSON matching the schema below. Do not include explanations.\n\n"
    "Schema:\n"
    "{\n"
    '  "delta_events": [\n'
    '    {"event_type": "BP_READING", "payload": {"systolic": 145, "diastolic": 92}},\n'
    '    {"event_type": "MEDICATION_ADDED", "payload": {"drug_name": "Lisinopril", "dosage": "10mg", "frequency": "daily"}},\n'
    '    {"event_type": "MEDICATION_STOPPED", "payload": {"drug_name": "Amlodipine", "reason": "side effects"}},\n'
    '    {"event_type": "SYMPTOM_REPORTED", "payload": {"symptom": "headache"}},\n'
    '    {"event_type": "DEMOGRAPHIC", "payload": {"age": 55, "sex": "male"}},\n'
    '    {"event_type": "LAB_RESULT", "payload": {"lab_type": "eGFR", "value": 60, "unit": "mL/min"}}\n'
    '  ],\n'
    '  "search_query": "clinical search query for guideline retrieval",\n'
    '  "raw_intent": "what the patient wants to know"\n'
    "}"
)


def _build_user_prompt(message: str, snapshot: dict[str, Any]) -> str:
    """Construct the user prompt with XML-isolated patient input (ADL-018).

    The patient's raw message is wrapped in ``<untrusted_user_input>`` tags
    to separate it from system instructions and prevent injection.
    """
    snapshot_json = json.dumps(snapshot, indent=2, default=str)

    return (
        f"Current patient context:\n"
        f"{snapshot_json}\n\n"
        f"New patient message:\n"
        f"<untrusted_user_input>\n"
        f"{message}\n"
        f"</untrusted_user_input>"
    )


class ContextExtractor:
    """LLM Call 1 — Structured fact extraction from patient messages.

    Uses the ``ResilientLLMWrapper`` for retry logic, JSON repair, and
    provider failover.  Returns ``ExtractorOutput`` or ``None`` on
    permanent failure (the orchestrator then skips extraction per §3.9).
    """

    def __init__(self, llm: Optional[ResilientLLMWrapper] = None) -> None:
        self._llm = llm or get_llm_wrapper()
        self._settings = get_settings()

    async def extract(
        self,
        message: str,
        snapshot: dict[str, Any],
    ) -> Optional[ExtractorOutput]:
        """Extract delta events, search query, and intent from a message.

        Args:
            message: The raw patient message.
            snapshot: The current patient context snapshot (dict).

        Returns:
            ``ExtractorOutput`` with ``delta_events``, ``search_query``,
            and ``raw_intent``, or ``None`` on permanent LLM failure.
        """
        user_prompt = _build_user_prompt(message, snapshot)

        result = await self._llm.call(
            call_name="context_extractor",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_schema=ExtractorOutput,
            temperature=self._settings.LLM_TEMPERATURE_GATES,  # 0.0
            max_tokens=self._settings.LLM_MAX_TOKENS_GATES,    # 512
        )

        if result is None:
            logger.error("Context Extractor permanent failure — returning None")

        return result

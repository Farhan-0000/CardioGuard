"""
Context Gate — LLM Call 2 (ADL-023).

Performs binary classification (SOFT-ASK vs PROCEED) based on whether the
accumulated patient context snapshot is clinically sufficient for safe
personalized clinical guidance.

Constitution §7.2 / Non-Negotiable #3: Temperature must be 0.0 with seed 42.
Constitution §7.1 / Non-Negotiable #5: Input encapsulated in <untrusted_user_input> XML tags.

Technical Specification Part III §3.4.
"""
import json
import structlog
from typing import Any, Optional

from medbridge.ai.llm_wrapper import ResilientLLMWrapper, get_llm_wrapper
from medbridge.ai.schemas.context_gate import ContextGateOutput
from medbridge.config import get_settings

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# System prompt (§3.4 — verbatim from Tech Spec)
# ------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a clinical context sufficiency classifier. Given a patient's accumulated "
    "medical context and their question intent, determine whether sufficient information "
    "exists to safely provide clinical guidance.\n\n"
    "If critical context is missing for personalized guidance (for example, patient asks if they should adjust or change their medication dose, but their current medication or blood pressure is unknown), output SOFT-ASK.\n"
    "If the patient is asking a general informational question about hypertension guidelines (such as lifestyle, dietary changes, exercise, blood pressure definitions or targets), or if sufficient context exists, output PROCEED.\n\n"
    "Output ONLY valid JSON. Schema:\n"
    "{\n"
    '  "action": "SOFT-ASK" | "PROCEED",\n'
    '  "missing_fields": ["field1", "field2"],\n'
    '  "rationale": "explanation"\n'
    "}"
)


def _build_user_prompt(snapshot: dict[str, Any], message: str, raw_intent: str) -> str:
    """Construct the user prompt isolating untrusted patient input in XML tags (ADL-018)."""
    snapshot_json = json.dumps(snapshot, indent=2, default=str)

    return (
        f"Current patient context snapshot:\n"
        f"{snapshot_json}\n\n"
        f"Question Intent:\n"
        f"{raw_intent}\n\n"
        f"Patient message:\n"
        f"<untrusted_user_input>\n"
        f"{message}\n"
        f"</untrusted_user_input>"
    )


class ContextGate:
    """LLM Call 2 — Binary context sufficiency classifier.

    Uses ResilientLLMWrapper with temperature 0.0 and seed 42 (NON-NEGOTIABLE #3).
    Returns ContextGateOutput, or None on permanent failure (orchestrator defaults
    to PROCEED fallback per Technical Specification Part III §3.9).
    """

    def __init__(self, llm: Optional[ResilientLLMWrapper] = None) -> None:
        self._llm = llm or get_llm_wrapper()
        self._settings = get_settings()

    async def evaluate(
        self,
        snapshot: dict[str, Any],
        message: str,
        raw_intent: str,
    ) -> Optional[ContextGateOutput]:
        """Evaluate context sufficiency for a clinical question.

        Args:
            snapshot: Current patient context snapshot (dict).
            message: Raw patient message.
            raw_intent: Extracted intent from Context Extractor (LLM 1).

        Returns:
            ContextGateOutput with action ('SOFT-ASK' or 'PROCEED'),
            missing_fields, and rationale, or None on permanent LLM failure.
        """
        user_prompt = _build_user_prompt(snapshot, message, raw_intent)

        result = await self._llm.call(
            call_name="context_gate",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_schema=ContextGateOutput,
            temperature=self._settings.LLM_TEMPERATURE_GATES,  # 0.0
            max_tokens=self._settings.LLM_MAX_TOKENS_GATES,    # 512
        )

        if result is None:
            logger.error("Context Gate permanent failure — returning None")

        return result

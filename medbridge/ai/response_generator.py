"""
Response Generator — LLM Call 4 (ADL-018, ADL-025, TASK-22).

Generates evidence-grounded clinical responses with inline numeric citation markers ([1], [2])
using strict XML-isolated prompt boundaries.

Constitution §7.2 / Non-Negotiable #3: Temperature must be 0.3 for Call 4 fluency.
Constitution §7.1 / Non-Negotiable #5: User input encapsulated in <untrusted_user_input> XML tags.
Technical Specification Part III §3.8, §3.9.
"""
import json
import structlog
from typing import Any, Optional, Union

from medbridge.ai.llm_wrapper import ResilientLLMWrapper, get_llm_wrapper
from medbridge.ai.prompt_utils import format_chunks
from medbridge.ai.schemas.response_generator import (
    GeneratedCitation,
    ResponseGeneratorOutput,
)
from medbridge.api.schemas.enums import ActionEnum
from medbridge.config import get_settings

logger = structlog.get_logger(__name__)

# Actions that invoke LLM Call 4
ALLOWED_ACTIONS: set[str] = {"ANSWER", "GENERALIZE", "SOFT-ASK"}

# Actions that bypass LLM Call 4 and use deterministic templates (ADL-015)
BYPASS_ACTIONS: set[str] = {"ABSTAIN", "ESCALATE"}

SYSTEM_INSTRUCTIONS_TEMPLATE: str = (
    "<system_instructions>\n"
    "You are a clinical communication assistant. Generate a response using\n"
    "ONLY the evidence provided in <clinical_evidence>. Do not assume or\n"
    "infer patient information not present in <patient_context>. Follow\n"
    "the routing action: {action_decision}.\n"
    "If action is SOFT-ASK, ask the patient for clarifying clinical context needed.\n"
    "Include inline citation markers [1], [2], etc. for each clinical claim.\n"
    "Do not provide diagnosis or prescriptions. Use empathetic, plain language.\n\n"
    "Output ONLY valid JSON. Schema:\n"
    "{\n"
    '  "response_text": "your response with [1] [2] markers",\n'
    '  "citations": [{"marker":"[1]","chunk_id":"...","source":"...","section":"...","excerpt":"..."}]\n'
    "}\n"
    "</system_instructions>"
)


def _format_chunks(chunks: list[Any]) -> str:
    """Format up to top-5 retrieved chunks into XML tags (ADL-025)."""
    return format_chunks(chunks)



def _build_system_prompt(action: str) -> str:
    """Build system instructions with the routing action decision (ADL-025)."""
    return SYSTEM_INSTRUCTIONS_TEMPLATE.replace("{action_decision}", action)


def _build_user_prompt(
    chunks: list[Any],
    snapshot: dict[str, Any],
    query: str,
) -> str:
    """Build user prompt with patient context, clinical evidence, and user query."""
    snapshot_json = json.dumps(snapshot, indent=2, default=str)
    evidence_xml = _format_chunks(chunks)

    return (
        f"<patient_context>\n"
        f"{snapshot_json}\n"
        f"</patient_context>\n\n"
        f"<clinical_evidence>\n"
        f"{evidence_xml}\n"
        f"</clinical_evidence>\n\n"
        f"<user_query>\n"
        f"<untrusted_user_input>\n"
        f"{query}\n"
        f"</untrusted_user_input>\n"
        f"</user_query>"
    )


def build_full_prompt(
    action: Union[ActionEnum, str],
    chunks: list[Any],
    snapshot: dict[str, Any],
    query: str,
) -> str:
    """Construct the complete XML-isolated prompt template combining all 4 sections (ADL-025)."""
    action_str = action.value if isinstance(action, ActionEnum) else str(action)
    return f"{_build_system_prompt(action_str)}\n\n{_build_user_prompt(chunks, snapshot, query)}"


class ResponseGenerator:
    """LLM Call 4 — Response Generator with XML prompt isolation and citations (ADL-025).

    Invoked only for ANSWER, GENERALIZE, and SOFT-ASK actions.
    Bypassed for ABSTAIN and ESCALATE (which use deterministic templates, ADL-015).
    Uses temperature 0.3 for natural conversational fluency.
    """

    def __init__(self, llm: Optional[ResilientLLMWrapper] = None) -> None:
        self._llm = llm or get_llm_wrapper()
        self._settings = get_settings()

    async def generate(
        self,
        action: Union[ActionEnum, str],
        chunks: list[Any],
        snapshot: dict[str, Any],
        query: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Optional[ResponseGeneratorOutput]:
        """Generate clinical response with inline citations.

        Args:
            action: Routing action (ANSWER, GENERALIZE, SOFT-ASK).
            chunks: Retrieved / reranked guideline chunks (up to Top-5).
            snapshot: Current patient context snapshot (dict).
            query: Patient clinical query string.
            message: Alias for query if passed as keyword argument.

        Returns:
            ResponseGeneratorOutput (response_text, citations) or None if bypassed/failed.
        """
        if isinstance(action, ActionEnum):
            action_str = action.value
        else:
            action_str = str(action).upper()

        if action_str == "SOFT_ASK":
            action_str = "SOFT-ASK"

        # Bypass check: Only invoked for ANSWER, GENERALIZE, and SOFT-ASK actions (bypassed for ABSTAIN/ESCALATE)
        if action_str in BYPASS_ACTIONS or action_str not in ALLOWED_ACTIONS:
            logger.info(
                "Response Generator bypassed for action '%s' (deterministic template or invalid)",
                action_str,
            )
            return None

        user_query = query if query is not None else (message or "")
        system_prompt = _build_system_prompt(action_str)
        user_prompt = _build_user_prompt(chunks, snapshot, user_query)

        result = await self._llm.call(
            call_name="response_generator",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ResponseGeneratorOutput,
            temperature=self._settings.LLM_TEMPERATURE_GENERATOR,  # 0.3
            max_tokens=self._settings.LLM_MAX_TOKENS_GENERATOR,    # 1024
        )

        if result is None:
            logger.error("Response Generator permanent failure — returning None")

        return result


def get_response_generator() -> ResponseGenerator:
    """Convenience factory function for dependency injection."""
    return ResponseGenerator()

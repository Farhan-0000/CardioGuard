"""
Evidence Gate — LLM Call 3 (ADL-024, TASK-21).

Evaluates whether retrieved Top-5 guideline chunks provide sufficient evidence
to answer the patient's clinical query, classifying into one of four routing
decisions: ANSWER, GENERALIZE, ABSTAIN, or ESCALATE.

Constitution §7.2 / Non-Negotiable #3: Temperature must be 0.0 with seed 42.
Constitution §7.1 / Non-Negotiable #5: Input encapsulated in <untrusted_user_input> XML tags.
Technical Specification Part III §3.7, §3.8, §3.9.
"""
import json
import structlog
from typing import Any, Optional

from medbridge.ai.llm_wrapper import ResilientLLMWrapper, get_llm_wrapper
from medbridge.ai.prompt_utils import format_chunks
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.config import get_settings

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# System prompt (§3.7 — verbatim from Tech Spec & ADL-024)
# ------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a clinical evidence sufficiency evaluator. Given a patient's context,\n"
    "their question, and the top-5 retrieved guideline chunks, determine the\n"
    "appropriate routing action:\n\n"
    "- ANSWER: Evidence directly addresses the patient's specific clinical parameters.\n"
    "- GENERALIZE: Evidence covers the general topic but not the patient's exact scenario.\n"
    "- ABSTAIN: Evidence is irrelevant or query is outside the knowledge base domain.\n"
    "- ESCALATE: Query indicates acute clinical danger regardless of evidence.\n\n"
    "Output ONLY valid JSON. Schema:\n"
    "{\n"
    '  "action": "ANSWER" | "GENERALIZE" | "ABSTAIN" | "ESCALATE",\n'
    '  "evidence_sufficient": true | false,\n'
    '  "rationale": "explanation"\n'
    "}"
)


def _format_chunks(chunks: list[Any]) -> str:
    """Format up to top-5 retrieved chunks into XML tags (ADL-025)."""
    return format_chunks(chunks, empty_placeholder="No clinical evidence available.")


def _build_user_prompt(
    chunks: list[Any],
    snapshot: dict[str, Any],
    query: str,
) -> str:
    """Construct XML-isolated prompt enclosing patient context, evidence, and query (ADL-018, ADL-025, Constitution §7.1)."""
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


class EvidenceGate:
    """LLM Call 3 — Clinical evidence sufficiency & 4-way routing evaluator (ADL-024).

    Evaluates retrieved guideline evidence against patient context and query,
    classifying into ANSWER, GENERALIZE, ABSTAIN, or ESCALATE.

    Uses ResilientLLMWrapper with temperature 0.0 and seed 42 (NON-NEGOTIABLE #3).
    Returns EvidenceGateOutput, or None on permanent failure (orchestrator defaults
    to GENERALIZE fallback per Technical Specification Part III §3.9).
    """

    def __init__(self, llm: Optional[ResilientLLMWrapper] = None) -> None:
        self._llm = llm or get_llm_wrapper()
        self._settings = get_settings()

    async def evaluate(
        self,
        chunks: list[Any],
        snapshot: dict[str, Any],
        query: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Optional[EvidenceGateOutput]:
        """Evaluate evidence sufficiency and determine routing action.

        Args:
            chunks: Retrieved and reranked guideline chunks (list of RetrievedChunk, dict, or str).
            snapshot: Current patient context snapshot (dict).
            query: Patient clinical query string.
            message: Alias for query if passed as keyword argument.

        Returns:
            EvidenceGateOutput with action, evidence_sufficient, and rationale,
            or None on permanent LLM failure.
        """
        user_query = query if query is not None else (message or "")
        user_prompt = _build_user_prompt(chunks, snapshot, user_query)

        result = await self._llm.call(
            call_name="evidence_gate",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_schema=EvidenceGateOutput,
            temperature=self._settings.LLM_TEMPERATURE_GATES,  # 0.0
            max_tokens=self._settings.LLM_MAX_TOKENS_GATES,    # 512
        )

        if result is None:
            logger.error("Evidence Gate permanent failure — returning None")

        return result


def get_evidence_gate() -> EvidenceGate:
    """Convenience factory function for dependency injection."""
    return EvidenceGate()

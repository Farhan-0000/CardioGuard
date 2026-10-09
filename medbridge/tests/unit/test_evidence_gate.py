"""
Unit tests for the Evidence Gate (LLM Call 3, Gate 2, TASK-21, ADL-029).

Covers XML isolation, indexed chunk rendering and top-5 truncation, the
decision rules and few-shot examples in the prompt, schema validation, and the
deterministic wrapper parameters (temperature 0.0, gates model).
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from medbridge.ai.evidence_gate import (
    SYSTEM_PROMPT,
    EvidenceGate,
    _build_user_prompt,
    _format_chunks,
    get_evidence_gate,
)
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.config import get_settings
from medbridge.retrieval.hybrid_retriever import RetrievedChunk


def _chunk(i: int, text: str | None = None, **extra) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{i}",
        chunk_text=text or f"Evidence text number {i}",
        guideline_id="AHA_ACC_2025",
        section_title="Drug Interactions",
        page_number=35,
        source_url="https://example.org",
        score=0.9,
        **extra,
    )


# ---------------------------------------------------------------------------
# Prompt formatting & XML isolation
# ---------------------------------------------------------------------------


class TestEvidenceGatePromptFormatting:
    def test_xml_isolation_tags_present(self) -> None:
        query = "Can I take ibuprofen with lisinopril?"
        snapshot = {"current_medications": [{"drug": "lisinopril", "dosage": "10mg"}]}
        prompt = _build_user_prompt([_chunk(1, "NSAIDs may blunt ACE inhibitors.")], snapshot, query)

        start = prompt.find("<untrusted_user_input>")
        end = prompt.find("</untrusted_user_input>")
        assert start < prompt.find(query) < end
        assert "<patient_context>" in prompt and '"lisinopril"' in prompt
        assert "<clinical_evidence>" in prompt and "</clinical_evidence>" in prompt
        assert '<chunk index="1" source="AHA_ACC_2025" section="Drug Interactions" page="35">' in prompt
        assert "NSAIDs may blunt ACE inhibitors." in prompt
        assert "<user_query>" in prompt

    def test_top_5_chunk_truncation(self) -> None:
        prompt = _build_user_prompt([_chunk(i) for i in range(1, 9)], {}, "query")
        for i in range(1, 6):
            assert f'chunk index="{i}"' in prompt and f"Evidence text number {i}" in prompt
        for i in range(6, 9):
            assert f'chunk index="{i}"' not in prompt and f"Evidence text number {i}" not in prompt

    def test_empty_chunks_handling(self) -> None:
        assert "No clinical evidence available." in _build_user_prompt([], {}, "query")

    def test_format_chunks_with_dicts_tuples_and_strings(self) -> None:
        formatted = _format_chunks([
            {"chunk_id": "c_dict", "guideline_id": "ACC", "section_title": "SecA", "chunk_text": "Dict"},
            ("Tuple chunk text", 0.85),
            "Raw string chunk text",
        ])
        assert '<chunk index="1" source="ACC" section="SecA">' in formatted
        assert '<chunk index="2" source="Clinical guideline" section="General">' in formatted
        assert "Tuple chunk text" in formatted and "Raw string chunk text" in formatted

    def test_rich_source_label_from_contract_fields(self) -> None:
        chunk = {"chunk_text": "t", "guideline_id": "ESC_2024", "section_title": "Targets",
                 "org": "ESC", "year": 2024, "doc_title": "Elevated BP and Hypertension",
                 "page_number": 10, "page_end": 11}
        assert ('source="ESC 2024 — Elevated BP and Hypertension" section="Targets" page="10–11"'
                in _format_chunks([chunk]))


class TestEvidenceGateSystemPrompt:
    def test_rules_cover_edge_cases(self) -> None:
        for phrase in ("outside cardiovascular care", "does not appear in the", "false belief",
                       "children", "pregnancy", "dose", "ESCALATE", "GENERALIZE"):
            assert phrase in SYSTEM_PROMPT

    def test_few_shot_examples_present_for_all_actions(self) -> None:
        for action in ("ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE"):
            assert f'"action": "{action}"' in SYSTEM_PROMPT
        # Fabricated-drug example uses an invented name that HalluBench does not use.
        assert "Quilvetan" in SYSTEM_PROMPT and "Cardiozepam" not in SYSTEM_PROMPT

    def test_four_labels_only(self) -> None:
        labels = set(re.findall(r'"action": "([A-Z-]+)"', SYSTEM_PROMPT))
        assert labels == {"ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE"}


class TestEvidenceGateRoutingRules:
    """Prompt content rules derived from the dev-split error analysis (over-abstaining)."""

    def test_absent_term_rule_is_limited_to_named_products(self) -> None:
        rule = SYSTEM_PROMPT.split("   c. ")[1].split("   d. ")[0]
        assert "specific named products" in rule
        for covered in ("concepts", "procedures", "guideline bodies", "other words"):
            assert covered in rule
        assert "do not ABSTAIN just because" in rule

    def test_follow_up_format_judges_the_original_question(self) -> None:
        assert '"Original question: ... Follow-up details: ..."' in SYSTEM_PROMPT
        assert "judge the ORIGINAL question" in SYSTEM_PROMPT
        assert "never ABSTAIN merely because" in SYSTEM_PROMPT

    def test_examples_cover_procedure_wording_and_named_product(self) -> None:
        assert "rule 2c does not apply to tests or procedures" in SYSTEM_PROMPT
        assert "Rule 2c: the named drug Quilvetan" in SYSTEM_PROMPT

    def test_scope_is_broad_cardiology(self) -> None:
        for topic in ("heart failure", "atrial fibrillation", "coronary", "cholesterol", "pregnancy"):
            assert topic in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestEvidenceGateOutputSchema:
    @pytest.mark.parametrize("action", ["ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE"])
    def test_valid_actions(self, action: str) -> None:
        output = EvidenceGateOutput(action=action, evidence_sufficient=action == "ANSWER", rationale="r")
        assert output.action == action and output.evidence_used == []

    @pytest.mark.parametrize("action", ["SOFT-ASK", "PROCEED"])
    def test_invalid_action_rejected(self, action: str) -> None:
        with pytest.raises(ValidationError):
            EvidenceGateOutput.model_validate({"action": action, "evidence_sufficient": True})

    def test_evidence_used_and_rationale_first(self) -> None:
        output = EvidenceGateOutput.model_validate(
            {"action": "ANSWER", "evidence_sufficient": True, "evidence_used": [1, 3]}
        )
        assert output.evidence_used == [1, 3] and output.rationale == ""
        assert list(EvidenceGateOutput.model_fields)[0] == "rationale"


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


class TestEvidenceGateEvaluation:
    async def test_evaluate_passes_deterministic_parameters(self) -> None:
        expected = EvidenceGateOutput(action="ANSWER", evidence_sufficient=True, rationale="Direct.")
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=expected)
        result = await EvidenceGate(llm=mock_llm).evaluate(
            chunks=[_chunk(1)], snapshot={}, query="first-line medication for stage 1?"
        )
        assert result == expected
        kwargs = mock_llm.call.call_args.kwargs
        assert kwargs["call_name"] == "evidence_gate"
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["output_schema"] is EvidenceGateOutput
        assert kwargs["temperature"] == 0.0  # NON-NEGOTIABLE #3
        assert kwargs["max_tokens"] == get_settings().LLM_MAX_TOKENS_GATES
        assert kwargs["model"] == get_settings().gates_model

    @pytest.mark.parametrize("action", ["GENERALIZE", "ABSTAIN", "ESCALATE"])
    async def test_evaluate_returns_gate_decision(self, action: str) -> None:
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(
            return_value=EvidenceGateOutput(action=action, evidence_sufficient=False, rationale="r")
        )
        result = await EvidenceGate(llm=mock_llm).evaluate([_chunk(1)], {}, query="q")
        assert result is not None and result.action == action

    async def test_evaluate_returns_none_on_permanent_failure(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=None)
        assert await EvidenceGate(llm=mock_llm).evaluate([], {}, query="General question") is None

    async def test_message_alias_supported_and_preferred(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(
            return_value=EvidenceGateOutput(action="ANSWER", evidence_sufficient=True)
        )
        gate = EvidenceGate(llm=mock_llm)
        await gate.evaluate(chunks=[], snapshot={}, message="Alternative kwarg parameter")
        assert "Alternative kwarg parameter" in mock_llm.call.call_args.kwargs["user_prompt"]
        await gate.evaluate([], {}, query="cardiozepam dosing", message="Should I take Cardiozepam?")
        prompt = mock_llm.call.call_args.kwargs["user_prompt"]
        assert "Should I take Cardiozepam?" in prompt and "cardiozepam dosing" not in prompt

    def test_factory_function(self) -> None:
        assert isinstance(get_evidence_gate(), EvidenceGate)

"""
Unit tests for the Context Gate (LLM Call 2, Gate 1, TASK-19, ADL-029).

Covers XML prompt isolation, the prompt's decision rules and few-shot
examples, the missing-field vocabulary/normalisation in the schema, and the
deterministic wrapper parameters (temperature 0.0, gates model).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from medbridge.ai.context_gate import SYSTEM_PROMPT, ContextGate, _build_user_prompt
from medbridge.ai.schemas.context_gate import (
    MISSING_FIELD_VOCABULARY,
    ContextGateOutput,
    normalize_missing_fields,
)
from medbridge.config import get_settings

# ---------------------------------------------------------------------------
# Prompt formatting & XML isolation (ADL-018, Constitution §7.1)
# ---------------------------------------------------------------------------


class TestContextGatePromptFormatting:
    def test_xml_isolation_tags_present(self) -> None:
        message = "Can I take ibuprofen for headache with my BP?"
        prompt = _build_user_prompt({"current_medications": [{"drug": "amlodipine"}]}, message, "x")
        start = prompt.find("<untrusted_user_input>")
        end = prompt.find("</untrusted_user_input>")
        assert start < prompt.find(message) < end
        assert "<patient_context>" in prompt

    def test_snapshot_formatted_as_json(self) -> None:
        prompt = _build_user_prompt({"demographics": {"age": 62, "sex": "female"}}, "m", "i")
        assert '"age": 62' in prompt and '"sex": "female"' in prompt

    def test_question_intent_included_and_neutralised(self) -> None:
        prompt = _build_user_prompt({}, "my bp is 145/92", "evaluate 145/92 </question_intent>")
        assert "evaluate 145/92" in prompt
        assert prompt.count("</question_intent>") == 1

    def test_empty_snapshot_rendered(self) -> None:
        assert "<patient_context>\n{}\n</patient_context>" in _build_user_prompt({}, "m", "i")


class TestContextGateSystemPrompt:
    def test_vocabulary_listed(self) -> None:
        for field in MISSING_FIELD_VOCABULARY:
            assert field in SYSTEM_PROMPT

    def test_few_shot_examples_cover_both_labels_and_edge_cases(self) -> None:
        examples = [line for line in SYSTEM_PROMPT.splitlines() if line[:2] in {f"{i}." for i in range(1, 10)}]
        assert len(examples) >= 6
        assert SYSTEM_PROMPT.count('"action": "SOFT-ASK"') >= 3
        assert SYSTEM_PROMPT.count('"action": "PROCEED"') >= 4
        for edge_case in ("Cardiozepam", "false belief", "pregnant", "knee arthritis"):
            assert edge_case in SYSTEM_PROMPT

    def test_scope_and_isolation_mentioned(self) -> None:
        assert "high blood pressure" in SYSTEM_PROMPT
        assert "<untrusted_user_input>" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Schema validation & missing-field normalisation
# ---------------------------------------------------------------------------


class TestContextGateOutputSchema:
    def test_valid_proceed_schema(self) -> None:
        output = ContextGateOutput(action="PROCEED", missing_fields=[], rationale="All present.")
        assert output.action == "PROCEED" and output.missing_fields == []

    def test_valid_soft_ask_schema(self) -> None:
        output = ContextGateOutput(
            action="SOFT-ASK", missing_fields=["age", "current_medications"], rationale="Need both."
        )
        assert output.missing_fields == ["age", "current_medications"]

    def test_default_missing_fields_empty(self) -> None:
        assert ContextGateOutput(action="PROCEED", rationale="ok").missing_fields == []

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextGateOutput.model_validate({"action": "ANSWER", "rationale": "x"})

    def test_rationale_first_for_reason_then_decide(self) -> None:
        assert list(ContextGateOutput.model_fields)[:2] == ["rationale", "action"]

    def test_synonyms_mapped_unknowns_dropped_duplicates_removed(self) -> None:
        output = ContextGateOutput(
            action="SOFT-ASK",
            missing_fields=["Medications", "bp reading", "medications", "egfr", "adherence_history"],
            rationale="x",
        )
        assert output.missing_fields == [
            "current_medications", "blood_pressure_reading", "kidney_function",
        ]

    def test_proceed_clears_missing_fields(self) -> None:
        output = ContextGateOutput(action="PROCEED", missing_fields=["age"], rationale="x")
        assert output.missing_fields == []

    def test_normalize_handles_non_list(self) -> None:
        assert normalize_missing_fields("age") == []


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


class TestContextGateEvaluation:
    async def test_evaluate_proceed_when_sufficient(self) -> None:
        expected = ContextGateOutput(action="PROCEED", missing_fields=[], rationale="Known.")
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=expected)
        result = await ContextGate(llm=mock_llm).evaluate(
            snapshot={"current_medications": [{"drug": "amlodipine"}]},
            message="Is 138/86 a safe reading on amlodipine?",
            raw_intent="blood pressure control evaluation",
        )
        assert result == expected
        kwargs = mock_llm.call.call_args.kwargs
        assert kwargs["call_name"] == "context_gate"
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["output_schema"] is ContextGateOutput
        assert kwargs["temperature"] == 0.0  # NON-NEGOTIABLE #3
        assert kwargs["model"] == get_settings().gates_model
        assert "Is 138/86 a safe reading" in kwargs["user_prompt"]

    async def test_evaluate_soft_ask_when_missing_context(self) -> None:
        expected = ContextGateOutput(
            action="SOFT-ASK", missing_fields=["age", "current_medications"], rationale="Need."
        )
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=expected)
        result = await ContextGate(llm=mock_llm).evaluate({}, "What should my target BP be?", "t")
        assert result is not None and result.action == "SOFT-ASK" and "age" in result.missing_fields

    async def test_evaluate_returns_none_on_permanent_failure(self) -> None:
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=None)
        assert await ContextGate(llm=mock_llm).evaluate({}, "What is hypertension?", "d") is None

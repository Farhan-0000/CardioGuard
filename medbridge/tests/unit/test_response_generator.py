"""
Unit tests for the Response Generator (LLM Call 4, TASK-22, ADL-029).

Covers XML prompt construction, indexed chunk rendering, the generator modes
(grounded / direct / prompt_safety), SOFT-ASK prompts driven by Gate 1's
missing fields, template bypass for ABSTAIN/ESCALATE, and wrapper parameters
(temperature 0.3, generator model). The LLM is always mocked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from medbridge.ai.response_generator import (
    ALLOWED_ACTIONS,
    BYPASS_ACTIONS,
    MISSING_FIELD_PHRASES,
    PROMPT_SAFETY_RULE,
    ResponseGenerator,
    _build_system_prompt,
    _build_user_prompt,
    _format_chunks,
    build_full_prompt,
    get_response_generator,
)
from medbridge.ai.schemas.response_generator import GeneratedCitation, ResponseGeneratorOutput
from medbridge.api.schemas.enums import ActionEnum
from medbridge.config import get_settings
from medbridge.retrieval.hybrid_retriever import RetrievedChunk
from medbridge.state.turn_store import ChatTurn


def _chunk(i: int, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c_{i}",
        chunk_text=text or f"Clinical text snippet {i}",
        guideline_id="AHA_ACC_2025",
        section_title="First-Line Therapy",
        page_number=18,
        source_url="",
        score=0.9,
    )


def _mock_llm(text: str = "Thiazide diuretics are a first-line option [1].") -> MagicMock:
    llm = MagicMock()
    llm.call = AsyncMock(return_value=ResponseGeneratorOutput(response_text=text))
    return llm


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_all_xml_tags_present_in_full_prompt(self) -> None:
        query = "What is the recommended medication for high blood pressure?"
        prompt = build_full_prompt(
            ActionEnum.ANSWER, [_chunk(1)], {"demographics": {"age": 55}}, query
        )
        assert "<system_instructions>" in prompt and "</system_instructions>" in prompt
        assert "Follow the routing action: ANSWER." in prompt
        assert "<patient_context>" in prompt and '"age": 55' in prompt
        assert "<clinical_evidence>" in prompt
        assert ('<chunk index="1" source="AHA_ACC_2025" section="First-Line Therapy" page="18">'
                in prompt)
        assert "<user_query>" in prompt
        start = prompt.find("<untrusted_user_input>")
        end = prompt.find("</untrusted_user_input>")
        assert start < prompt.find(query) < end

    @pytest.mark.parametrize("action", ["ANSWER", "GENERALIZE"])
    def test_system_prompt_action_interpolation(self, action: str) -> None:
        prompt = _build_system_prompt(action)
        assert f"the routing action: {action}." in prompt
        assert f"{action} means:" in prompt

    def test_grounded_rules_present(self) -> None:
        prompt = _build_system_prompt("ANSWER")
        for rule in ("ONLY facts stated in <clinical_evidence>", "[n]", "Never state numbers",
                     "mistaken belief", "markdown", "never diagnose or prescribe"):
            assert rule in prompt
        assert PROMPT_SAFETY_RULE.strip() not in prompt

    def test_generalize_says_not_personalised(self) -> None:
        assert "not personalised advice" in _build_system_prompt("GENERALIZE")

    def test_top_5_chunk_truncation(self) -> None:
        prompt = _build_user_prompt([_chunk(i) for i in range(1, 8)], {}, "query")
        for i in range(1, 6):
            assert f'chunk index="{i}"' in prompt and f"Clinical text snippet {i}" in prompt
        assert "Clinical text snippet 6" not in prompt and "Clinical text snippet 7" not in prompt

    def test_chunk_text_truncated_to_configured_length(self) -> None:
        limit = get_settings().EVIDENCE_MAX_CHARS_PER_CHUNK
        prompt = _build_user_prompt([_chunk(1, "word " * (limit // 2))], {}, "q")
        body = prompt.split('page="18">\n')[1].split("\n</chunk>")[0]
        assert len(body) <= limit + 2 and body.endswith("…")

    def test_empty_chunks_formatting(self) -> None:
        assert "No guideline evidence was retrieved" in _format_chunks([])

    def test_format_chunks_with_dicts_tuples_and_strings(self) -> None:
        formatted = _format_chunks([
            {"chunk_id": "d", "guideline_id": "ACC", "section_title": "Sec1", "chunk_text": "Dict text"},
            ("Tuple text", 0.88),
            "Raw string text",
        ])
        assert '<chunk index="1" source="ACC" section="Sec1">' in formatted
        assert "Tuple text" in formatted and "Raw string text" in formatted
        assert '<chunk index="3"' in formatted

    def test_attributes_escaped(self) -> None:
        formatted = _format_chunks([{"chunk_text": "t", "guideline_id": 'X" onload="y',
                                     "section_title": "<b>"}])
        assert 'source="X&quot; onload=&quot;y"' in formatted and 'section="&lt;b&gt;"' in formatted

    def test_history_included_and_isolated(self) -> None:
        history = [ChatTurn(role="user", content="I take Telma"), ChatTurn(role="assistant", content="Ok.")]
        prompt = build_full_prompt("ANSWER", [_chunk(1)], {}, "is it safe?", history=history)
        assert '<turn role="patient"><untrusted_user_input>\nI take Telma' in prompt
        assert '<turn role="assistant">Ok.</turn>' in prompt


class TestGeneratorModes:
    def test_direct_mode_has_no_evidence_or_grounding_rules(self) -> None:
        prompt = build_full_prompt("ANSWER", [_chunk(1)], {}, "q", mode="direct")
        assert "<clinical_evidence>" not in prompt
        assert "Clinical text snippet 1" not in prompt
        assert "general medical knowledge" in prompt and "informational only" in prompt
        assert "<untrusted_user_input>" in prompt

    def test_prompt_safety_mode_adds_instruction(self) -> None:
        prompt = build_full_prompt("ANSWER", [_chunk(1)], {}, "q", mode="prompt_safety")
        assert PROMPT_SAFETY_RULE.strip() in prompt
        assert "<clinical_evidence>" in prompt

    def test_soft_ask_lists_missing_fields_and_has_no_evidence(self) -> None:
        prompt = build_full_prompt(
            "SOFT-ASK", [_chunk(1)], {}, "Should I double my dose?",
            missing_fields=["current_medications", "bp"],
        )
        assert "Follow the routing action: SOFT-ASK." in prompt
        assert "<clinical_evidence>" not in prompt and "Clinical text snippet" not in prompt
        assert MISSING_FIELD_PHRASES["current_medications"] in prompt
        assert MISSING_FIELD_PHRASES["blood_pressure_reading"] in prompt
        assert "Do NOT give medical facts" in prompt

    def test_soft_ask_without_fields_uses_default_request(self) -> None:
        prompt = build_full_prompt("SOFT-ASK", [], {}, "q")
        assert "<missing_information>\n- the details needed" in prompt

    def test_every_vocabulary_field_has_a_phrase(self) -> None:
        from medbridge.ai.schemas.context_gate import MISSING_FIELD_VOCABULARY

        assert set(MISSING_FIELD_PHRASES) == set(MISSING_FIELD_VOCABULARY)


# ---------------------------------------------------------------------------
# Bypass & invocation
# ---------------------------------------------------------------------------


class TestActionBypassing:
    @pytest.mark.parametrize("action", ["ABSTAIN", ActionEnum.ABSTAIN, "ESCALATE",
                                        ActionEnum.ESCALATE, "INVALID_UNKNOWN_ACTION"])
    async def test_bypassed_without_llm(self, action) -> None:
        llm = _mock_llm()
        assert await ResponseGenerator(llm=llm).generate(action, [], {}, query="q") is None
        llm.call.assert_not_called()

    def test_action_sets(self) -> None:
        assert ALLOWED_ACTIONS == {"ANSWER", "GENERALIZE", "SOFT-ASK"}
        assert BYPASS_ACTIONS == {"ABSTAIN", "ESCALATE"}

    async def test_unknown_mode_returns_none(self) -> None:
        llm = _mock_llm()
        assert await ResponseGenerator(llm=llm).generate("ANSWER", [], {}, "q", mode="bogus") is None
        llm.call.assert_not_called()


class TestInvocation:
    async def test_invokes_for_answer_with_generator_parameters(self) -> None:
        llm = _mock_llm()
        result = await ResponseGenerator(llm=llm).generate(
            ActionEnum.ANSWER, [_chunk(1)], {"demographics": {"age": 60}}, query="What meds for BP?"
        )
        assert result is not None and "[1]" in result.response_text
        kwargs = llm.call.call_args.kwargs
        settings = get_settings()
        assert kwargs["call_name"] == "response_generator"
        assert kwargs["output_schema"] is ResponseGeneratorOutput
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == settings.LLM_MAX_TOKENS_GENERATOR
        assert kwargs["model"] == settings.generator_model

    @pytest.mark.parametrize("action", ["GENERALIZE", "SOFT-ASK", ActionEnum.SOFT_ASK, "soft_ask"])
    async def test_invokes_for_other_allowed_actions(self, action) -> None:
        llm = _mock_llm("Could you share your current medicines?")
        assert await ResponseGenerator(llm=llm).generate(action, [], {}, query="q") is not None
        llm.call.assert_called_once()

    async def test_message_preferred_over_query(self) -> None:
        llm = _mock_llm()
        await ResponseGenerator(llm=llm).generate(
            "ANSWER", [_chunk(1)], {}, query="search terms", message="my own words"
        )
        prompt = llm.call.call_args.kwargs["user_prompt"]
        assert "my own words" in prompt and "search terms" not in prompt

    async def test_mode_and_missing_fields_forwarded(self) -> None:
        llm = _mock_llm()
        gen = ResponseGenerator(llm=llm)
        await gen.generate("ANSWER", [_chunk(1)], {}, "q", mode="direct")
        assert "<clinical_evidence>" not in llm.call.call_args.kwargs["user_prompt"]
        await gen.generate("SOFT-ASK", [], {}, "q", missing_fields=["age"])
        assert MISSING_FIELD_PHRASES["age"] in llm.call.call_args.kwargs["user_prompt"]

    async def test_permanent_llm_failure_returns_none(self) -> None:
        llm = MagicMock()
        llm.call = AsyncMock(return_value=None)
        assert await ResponseGenerator(llm=llm).generate("ANSWER", [], {}, query="q") is None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_citations_optional(self) -> None:
        output = ResponseGeneratorOutput.model_validate({"response_text": "Hello [1]."})
        assert output.citations == []

    def test_citation_parsing(self) -> None:
        output = ResponseGeneratorOutput.model_validate({
            "response_text": "x [1]",
            "citations": [{"marker": "[1]", "chunk_id": "a", "source": "AHA", "section": "s",
                           "excerpt": "e"}],
        })
        assert output.citations[0].chunk_id == "a"

    def test_citation_optional_fields_fallback(self) -> None:
        citation = GeneratedCitation(marker="[1]")
        assert (citation.chunk_id, citation.source, citation.section, citation.excerpt) == ("",) * 4

    def test_empty_response_rejected(self) -> None:
        with pytest.raises(ValueError):
            ResponseGeneratorOutput(response_text="")


def test_factory_function() -> None:
    assert isinstance(get_response_generator(), ResponseGenerator)


def test_grounded_output_format_shows_citation_markers() -> None:
    """The JSON example the model mirrors must itself carry [n] markers (grounded modes only)."""
    from medbridge.ai.response_generator import _build_system_prompt

    for action, mode in (("ANSWER", "grounded"), ("GENERALIZE", "grounded"),
                         ("ANSWER", "prompt_safety")):
        prompt = _build_system_prompt(action, mode)
        assert 'Eating less salt lowers blood pressure [2].' in prompt, (action, mode)
    for action, mode in (("SOFT-ASK", "grounded"), ("ANSWER", "direct")):
        prompt = _build_system_prompt(action, mode)
        assert "[2]" not in prompt and '{"response_text": "<markdown answer>"}' in prompt

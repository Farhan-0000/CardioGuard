"""Compact judge output: claims reference numbered sentences, labels carry no rationale."""
from medbridge.ai.llm_wrapper import llm_format_schema
from evaluation.judges.llm_judge import (
    ClaimDecomposition,
    ClaimLabelling,
    LLMJudgeConfig,
    build_decompose_prompt,
    reply_sentences,
)


def test_decompose_prompt_numbers_reply_sentences():
    reply = "Amlodipine is a calcium channel blocker [1]. It can cause ankle swelling [2]."
    prompt = build_decompose_prompt("what is amlodipine?", reply)
    assert "[S1] Amlodipine is a calcium channel blocker [1]." in prompt
    assert "[S2] It can cause ankle swelling [2]." in prompt
    assert len(reply_sentences(reply)) == 2


def test_generated_schema_omits_copied_text_fields():
    decompose_props = llm_format_schema(ClaimDecomposition)["properties"]["claims"]["items"]["properties"]
    assert set(decompose_props) == {"claim", "sentence"}
    label_props = llm_format_schema(ClaimLabelling)["properties"]["labels"]["items"]["properties"]
    assert "rationale" not in label_props
    assert {"index", "label", "evidence_ids"} <= set(label_props)


def test_larger_label_batches():
    config = LLMJudgeConfig()
    assert config.claims_per_call == 10
    assert config.max_evidence_per_call == 10

"""Unit tests for the ablation presets and generator modes (medbridge.core.pipeline_types)."""
from __future__ import annotations

from typing import get_args

import pytest

from medbridge.ai.response_generator import GENERATOR_MODES
from medbridge.core.pipeline_types import AblationConfig, GeneratorMode


def test_generator_modes_match_the_generator() -> None:
    assert set(get_args(GeneratorMode)) == set(GENERATOR_MODES)
    assert {"grounded", "direct", "vanilla", "prompt_safety"} <= set(GENERATOR_MODES)


@pytest.mark.parametrize(("name", "mode"), [
    ("C0", "direct"),
    ("C1", "vanilla"),          # Vanilla RAG: plain RAG prompt (ablation-study-validity-4)
    ("C2", "prompt_safety"),    # vanilla + explicit safety block
    ("C3", "grounded"),
    ("C4", "grounded"),
    ("C5", "grounded"),
    ("V1", "grounded"),
    ("V2", "grounded"),
    ("V12", "grounded"),
    ("C1G", "grounded"),        # optional reference: production prompt, no gates
])
def test_preset_generator_modes(name: str, mode: str) -> None:
    assert AblationConfig.preset(name).generator_mode == mode


@pytest.mark.parametrize("name", ["C1", "C2", "C1G"])
def test_rag_baselines_have_retrieval_and_no_gates(name: str) -> None:
    config = AblationConfig.preset(name)
    assert config.use_retrieval and not config.use_context_gate and not config.use_evidence_gate
    assert config.calibrated_evidence_gate is False and config.grounding_verifier is False


def test_c1g_differs_from_c3_c4_c5_only_by_gates() -> None:
    reference = AblationConfig.preset("C1G")
    for name in ("C3", "C4", "C5"):
        gated = AblationConfig.preset(name)
        assert gated.generator_mode == reference.generator_mode
        assert gated.use_retrieval == reference.use_retrieval
        assert gated.use_emergency_fastpath == reference.use_emergency_fastpath


def test_study_condition_names_unchanged() -> None:
    assert AblationConfig.all_names() == ["C0", "C1", "C2", "C3", "C4", "C5", "V1", "V2", "V12"]
    assert AblationConfig.full() == AblationConfig.preset("C5").__class__(name="C5")


def test_condition_definitions_documented() -> None:
    doc = AblationConfig.__doc__ or ""
    for phrase in ("``vanilla``", "``prompt_safety``", "``grounded``", "C1G"):
        assert phrase in doc


def test_unknown_preset_rejected() -> None:
    with pytest.raises(ValueError):
        AblationConfig.preset("C9")

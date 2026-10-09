"""
Unit tests for V2 — the post-generation Grounding Verifier, "Gate 3".

The NLI cross-encoder is always mocked except in one optional ``slow`` test
that runs only when the configured model is already in the local Hugging Face
cache (it is never downloaded by the test suite).
"""
from __future__ import annotations

import re
from typing import Any

import pytest

from medbridge.ai import grounding_verifier as gv
from medbridge.ai.grounding_verifier import (
    GroundingVerifier,
    evidence_windows,
    get_grounding_verifier,
    is_claim,
    parse_markers,
    plain_text,
    split_sentences,
)
from medbridge.core.templates import get_loop_breaker_prefix


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class OverlapNLI:
    """Fake NLI: entailment = fraction of hypothesis words present in the premise."""

    def __init__(self, id2label: dict[int, str] | None = None) -> None:
        self.config = type("Cfg", (), {"id2label": id2label or {
            0: "contradiction", 1: "entailment", 2: "neutral"}})()
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[list[float]]:
        self.calls.append(pairs)
        assert kwargs.get("apply_softmax") is True
        labels = {v: k for k, v in self.config.id2label.items()}
        rows = []
        for premise, hypothesis in pairs:
            hyp = _words(hypothesis)
            entail = len(hyp & _words(premise)) / max(1, len(hyp))
            row = [0.0, 0.0, 0.0]
            row[labels.get("entailment", labels.get("ENTAILMENT", 1))] = entail
            row[labels.get("contradiction", labels.get("CONTRADICTION", 0))] = 0.1
            rows.append(row)
        return rows


CHUNKS = [
    {"chunk_text": "Eating less salt lowers blood pressure. The DASH diet lowers blood pressure."},
    {"chunk_text": "Hypertension usually needs long-term treatment. Stopping medicines can raise "
                   "blood pressure again."},
]


def verifier(model: Any = None, threshold: float = 0.8) -> GroundingVerifier:
    return GroundingVerifier(model_name="fake-nli", threshold=threshold, model=model or OverlapNLI())


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------


class TestSegmentation:
    @pytest.mark.parametrize(("text", "expected"), [
        ("Eat less salt [1]. Walk daily [2].", ["Eat less salt [1].", "Walk daily [2]."]),
        ("Limit salt, e.g. processed food. Walk more.", ["Limit salt, e.g. processed food.", "Walk more."]),
        ("That is, i.e. daily. Fine.", ["That is, i.e. daily.", "Fine."]),
        ("Ask Dr. Rao today. Then rest.", ["Ask Dr. Rao today.", "Then rest."]),
        ("A dose of 2.5 mg is low. Next.", ["A dose of 2.5 mg is low.", "Next."]),
        ("Keep it under control. [1] Next point.", ["Keep it under control. [1]", "Next point."]),
        ("Is it safe? Yes, mostly.", ["Is it safe?", "Yes, mostly."]),
        ("lower case. continuation stays", ["lower case. continuation stays"]),
    ])
    def test_split_sentences(self, text: str, expected: list[str]) -> None:
        assert split_sentences(text) == expected

    def test_markers(self) -> None:
        assert parse_markers("a [1] b [2, 4] c [3-5] d [1]") == [1, 2, 4, 3, 5]
        assert parse_markers("no markers") == []

    def test_plain_text(self) -> None:
        assert plain_text("**Salt** lowers _blood_ pressure [1].") == "Salt lowers blood pressure ."

    @pytest.mark.parametrize(("sentence", "claim"), [
        ("Thiazide diuretics lower blood pressure [1].", True),
        ("Statins lower LDL.", True),
        ("Hello!", False),
        ("Is this safe for me?", False),
        ("Here are some tips:", False),
        ("I understand this can feel worrying.", False),
        ("Please talk to your doctor before changing your dose.", False),
        ("This is general information, not personalised advice.", False),
        ("If you have chest pain, call emergency services right away.", False),
        ("Seek immediate medical attention if symptoms worsen.", False),
        ("Thanks for asking.", False),
    ])
    def test_claim_detection(self, sentence: str, claim: bool) -> None:
        assert is_claim(sentence) is claim

    def test_evidence_windows(self) -> None:
        text = " ".join(f"Sentence number {i} here." for i in range(1, 8))
        windows = evidence_windows(text)
        assert windows[0].startswith("Sentence number 1") and "number 3" in windows[0]
        assert windows[-1].endswith("number 7 here.")
        assert all(len(w) <= gv.WINDOW_MAX_CHARS for w in windows)
        assert evidence_windows("") == []
        assert evidence_windows("One sentence only.") == ["One sentence only."]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerify:
    async def test_supported_claims_kept_unsupported_removed(self) -> None:
        text = ("Eating less salt lowers blood pressure [1]. Garlic pills cure hypertension in a "
                "week [1].")
        result = await verifier().verify(text, CHUNKS)
        assert result.n_claims == 2 and result.n_unsupported == 1
        assert result.unsupported_fraction == 0.5
        assert result.cleaned_text == "Eating less salt lowers blood pressure [1]."
        assert result.unsupported_sentences == ["Garlic pills cure hypertension in a week [1]."]
        first, second = result.sentences
        assert first["supported"] and first["cited"] and first["markers"] == [1]
        assert not second["supported"] and second["entailment"] < 0.8
        assert set(first) == {"text", "markers", "cited", "is_claim", "supported",
                              "entailment", "contradiction"}

    async def test_premise_restricted_to_cited_chunk(self) -> None:
        text = "Eating less salt lowers blood pressure [2]."
        result = await verifier().verify(text, CHUNKS)
        assert result.n_unsupported == 1  # chunk 2 does not support it

    async def test_uncited_claim_checked_against_all_chunks(self) -> None:
        result = await verifier().verify("Stopping medicines can raise blood pressure again.", CHUNKS)
        assert result.n_unsupported == 0 and not result.sentences[0]["cited"]

    async def test_invalid_marker_treated_as_uncited(self) -> None:
        result = await verifier().verify("The DASH diet lowers blood pressure [9].", CHUNKS)
        record = result.sentences[0]
        assert record["markers"] == [9] and not record["cited"] and record["supported"]

    async def test_markdown_structure_preserved(self) -> None:
        text = (
            "I understand your concern.\n\n"
            "**What the guidelines say**\n"
            "- Eating less salt lowers blood pressure [1].\n"
            "- Moon dust lowers cholesterol quickly [1].\n\n"
            "**Other ideas**\n"
            "1. Unicorn tea reverses heart failure [2].\n\n"
            "Please talk to your doctor before changing anything."
        )
        result = await verifier().verify(text, CHUNKS)
        assert result.cleaned_text == (
            "I understand your concern.\n\n"
            "**What the guidelines say**\n"
            "- Eating less salt lowers blood pressure [1].\n\n"
            "Please talk to your doctor before changing anything."
        )
        assert result.n_claims == 3 and result.n_unsupported == 2

    async def test_non_claims_never_removed(self) -> None:
        text = "Hello! Is this about your tablets? Please talk to your doctor about your dose."
        result = await verifier().verify(text, CHUNKS)
        assert result.n_claims == 0 and result.unsupported_fraction == 0.0
        assert result.cleaned_text == text

    async def test_loop_breaker_prefix_kept(self) -> None:
        prefix = get_loop_breaker_prefix()
        text = f"{prefix}\n\nEating less salt lowers blood pressure [1]. Moon dust cures it [1]."
        result = await verifier().verify(text, CHUNKS)
        assert result.cleaned_text.startswith(prefix)
        assert "Moon dust" not in result.cleaned_text
        assert not any(prefix[:30] in s["text"] for s in result.sentences)

    async def test_label_order_read_from_config(self) -> None:
        model = OverlapNLI({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
        result = await verifier(model).verify("Eating less salt lowers blood pressure [1].", CHUNKS)
        assert result.sentences[0]["entailment"] == 1.0
        assert result.sentences[0]["contradiction"] == 0.1

    async def test_unnamed_labels_fall_back_to_default_order(self) -> None:
        model = OverlapNLI({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        v = verifier(model)
        assert v._label_indices(model) == (1, 0)

    async def test_single_batched_inference(self) -> None:
        model = OverlapNLI()
        await verifier(model).verify("Salt lowers blood pressure [1]. DASH lowers it too [1].", CHUNKS)
        assert len(model.calls) == 1

    async def test_no_evidence_means_unsupported(self) -> None:
        result = await verifier().verify("Eating less salt lowers blood pressure.", [])
        assert result.n_unsupported == 1 and result.cleaned_text == ""

    async def test_model_unavailable_returns_text_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        v = GroundingVerifier(model_name="missing/model", threshold=0.5)

        def broken_load() -> None:
            v._load_error = "OSError: not found"
            return None

        monkeypatch.setattr(v, "_load", broken_load)
        text = "Garlic pills cure hypertension [1]."
        result = await v.verify(text, CHUNKS)
        assert result.available is False
        assert result.cleaned_text == text and result.n_unsupported == 0
        assert result.unsupported_fraction == 0.0

    async def test_load_failure_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sentence_transformers

        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError("no network")

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", boom)
        v = GroundingVerifier(model_name="missing/model", device="cpu")
        assert v.loaded is False and v.load_error is None
        assert await v.warmup() is False
        assert v._load_error is not None
        assert v.loaded is False and v.load_error is not None

    async def test_warmup_with_model(self) -> None:
        v = verifier()
        assert await v.warmup() is True
        assert v.loaded is True

    def test_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gv, "_verifier_instance", None)
        assert get_grounding_verifier() is get_grounding_verifier()


def _model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)


@pytest.mark.slow
async def test_real_nli_model_if_cached() -> None:
    from medbridge.config import get_settings

    name = get_settings().GROUNDING_NLI_MODEL
    if not _model_cached(name):
        pytest.skip(f"{name} is not in the local Hugging Face cache")
    v = GroundingVerifier(model_name=name)
    result = await v.verify(
        "Reducing salt intake lowers blood pressure [1]. Eating more salt cures hypertension [1].",
        [{"chunk_text": "Reducing dietary sodium lowers blood pressure in adults with hypertension."}],
    )
    assert result.available
    assert result.sentences[0]["supported"] and not result.sentences[1]["supported"]

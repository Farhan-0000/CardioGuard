"""PII/PHI redaction on the blind held-out set (``pii_testset_blind_v2.json``).

The blind set was written by a separate author who never saw the detector. Its
precision / recall are *recorded* by ``scripts/pii_eval.py`` (see
``reports/privacy_redaction_metrics.md``) and deliberately not asserted here, so the
set can never become a tuning target. The only hard requirement checked is the
safety property: redaction must not corrupt clinical values.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from medbridge.privacy.evaluation import evaluate, resolve_gold
from medbridge.privacy.redaction import CATEGORIES

ROOT = Path(__file__).resolve().parents[2]
BLIND = Path(__file__).with_name("pii_testset_blind_v2.json")
SCRIPT = ROOT / "scripts" / "pii_eval.py"

pytestmark = pytest.mark.skipif(not BLIND.exists(), reason="blind test set not present")


@pytest.fixture(scope="module")
def pii_eval() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pii_eval_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def blind_examples(pii_eval: ModuleType) -> list[dict[str, Any]]:
    _, examples = pii_eval.load_examples(BLIND)
    return examples


def test_blind_set_is_well_formed(blind_examples: list[dict[str, Any]]) -> None:
    assert len({e["id"] for e in blind_examples}) == len(blind_examples)
    assert any(e["kind"] == "negative" for e in blind_examples)
    for ex in blind_examples:
        gold = resolve_gold(ex)
        for span, resolved in zip(ex["spans"], gold, strict=True):
            assert span["category"] in CATEGORIES
            if "start" in span:  # explicit offsets are honoured exactly
                assert (resolved.start, resolved.end) == (span["start"], span["end"])


def test_blind_zero_clinical_value_corruption(blind_examples: list[dict[str, Any]]) -> None:
    """No protected clinical value (BP, lab, dose, drug, age, reading date) is altered."""
    report = evaluate(blind_examples)
    assert report.protected_corrupted == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known false positive found by the single blind run (detector frozen, not fixed "
        "post-hoc): B2N072 'Dr. Google says ...' -> 'Dr. [NAME] says ...'. The title cue "
        "accepts a non-person word; no clinical value is touched. See "
        "reports/privacy_redaction_metrics.md (open issue)."
    ),
)
def test_blind_negatives_unaltered(blind_examples: list[dict[str, Any]]) -> None:
    report = evaluate(blind_examples)
    assert report.negatives_altered == []


def test_offset_normalisation_handles_repeated_substrings(pii_eval: ModuleType) -> None:
    example = {
        "id": "X1", "kind": "positive", "text": "Ravi called Ravi",
        "spans": [{"category": "NAME", "text": "Ravi", "start": 12, "end": 16}],
    }
    normalised = pii_eval.normalise_offsets(example)
    assert normalised["spans"][0]["occurrence"] == 1
    (gold,) = resolve_gold(normalised)
    assert (gold.start, gold.end) == (12, 16)
    with pytest.raises(ValueError):
        pii_eval.normalise_offsets({**example, "spans": [{**example["spans"][0], "start": 0}]})

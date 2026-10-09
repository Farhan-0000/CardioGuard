"""PII/PHI redaction on the blind held-out set (``pii_testset_blind_v2.json``).

The blind set was written by a separate author who never saw the detector. Its
precision / recall are *recorded* by ``scripts/pii_eval.py`` (see
``reports/privacy_redaction_metrics.md``) and deliberately not asserted here, so the
set can never become a tuning target. The hard requirements checked are the safety
properties (no clinical value corrupted, no clinically dense negative altered) and the
reporting protocol: the frozen detector's blind numbers stay the primary result, and
numbers of a detector changed after the blind run are labelled post-hoc.
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


def test_blind_negatives_unaltered(blind_examples: list[dict[str, Any]]) -> None:
    """Regression guard for the 'Dr. <non-name>' false positive (B2N072, fixed post-hoc).

    The frozen detector altered one negative ('Dr. Google says ...'); the blind report
    keeps that frozen result and shows the fix only in its post-hoc section.
    """
    report = evaluate(blind_examples)
    assert report.negatives_altered == []


FROZEN_SHA = "e75b67bc708d05227fb4e9338919d06b5150530aa2f6a877dc252dc90c9c1015"


@pytest.fixture(scope="module")
def frozen_detector(pii_eval: ModuleType) -> ModuleType:
    analysis = pii_eval.POST_HOC_ANALYSIS[pii_eval.sha256_file(BLIND)]
    assert analysis["detector_sha256"] == FROZEN_SHA
    module = pii_eval.load_frozen_detector(FROZEN_SHA, analysis["detector_commit"])
    if module is None:
        pytest.skip("frozen detector source not available (no git history)")
    return module


def test_frozen_detector_source_is_hash_checked(
    pii_eval: ModuleType, frozen_detector: ModuleType
) -> None:
    commit = pii_eval.POST_HOC_ANALYSIS[pii_eval.sha256_file(BLIND)]["detector_commit"]
    assert pii_eval.load_frozen_detector("0" * 64, commit) is None
    # the frozen detector still has the reported boundary bug; the current one does not
    leaky = "Flat 3, Phase 2B, Electronic City, Bangalore 560100"
    assert frozen_detector.redact(leaky).text.startswith("[ADDRESS]B")
    assert pii_eval.current_detector.redact(leaky).text == "[ADDRESS] [PINCODE]"


def test_blind_report_keeps_frozen_numbers_primary(
    pii_eval: ModuleType, frozen_detector: ModuleType
) -> None:
    ds = pii_eval.evaluate_dataset(pii_eval.Dataset("blind_v2", BLIND, "blind_heldout"))
    if pii_eval.sha256_file(pii_eval.DETECTOR) == FROZEN_SHA:
        pytest.skip("the current detector is the frozen one; no post-hoc section")
    assert ds["detector"]["sha256"] == FROZEN_SHA
    assert ds["frozen_check"]["reproduced"], ds["frozen_check"]
    assert ds["post_hoc_analysis"]["detector_matches"]
    post = ds["post_hoc"]
    assert post["detector_sha256"] == pii_eval.sha256_file(pii_eval.DETECTOR)
    assert post["label"] == (
        "Post-hoc fixes (after blind evaluation; optimistic, not an independent estimate)"
    )
    assert post["fixes"]
    assert post["results"]["all"]["protected_values_corrupted"] == []
    md = pii_eval.render_markdown(
        {"generated_at": "t", "reference_date": "d", "git_commit": "c", "python": "p",
         "machine": "m", "protocol": pii_eval.PROTOCOL,
         "detector": {"path": "medbridge/privacy/redaction.py", "sha256": post["detector_sha256"]}},
        {"blind_v2": ds},
    )
    blind_at = md.index("## Blind held-out set (evaluated once, frozen detector): `blind_v2`")
    post_at = md.index(
        "## Post-hoc fixes (after blind evaluation; optimistic, not an independent estimate)"
    )
    assert blind_at < post_at
    assert "| blind_v2 — post-hoc | post-hoc (optimistic) |" in md
    assert f"`{post['detector_sha256'][:16]}…`" in md[post_at:]


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

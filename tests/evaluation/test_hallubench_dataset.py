"""Quality-control checks for the MedBridge-HalluBench v1 dataset.

Covers ``evaluation/datasets/hallubench_v1.json`` and its documentation
``evaluation/datasets/HALLUBENCH.md``: schema validity, unique sequential ids,
per-category counts against the study protocol (CLAUDE_AGENT_PROMPT §6.1),
the seeded stratified dev/test split, gold-label semantics per category,
gold evidence that points into ``data/corpus_manifest.json``, gold notes for
the judges, topic/style coverage and the absence of crisis-level BP readings
outside the emergency category.

Everything here is hermetic except ``test_evidence_pages_are_indexed``, which
reads chunk payloads from Qdrant (``qdrant`` marker, skipped when the service
is unreachable).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import pytest

from evaluation.datasets.schema import (
    ACTIONS,
    ANSWERABLE_CATEGORIES,
    CATEGORIES,
    UNANSWERABLE_CATEGORIES,
    HalluBench,
    HalluItem,
    load_dataset,
    stratified_split,
)
from evaluation.paths import DATASETS_DIR, DEFAULT_DATASET, REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "data" / "corpus_manifest.json"
DOC_PATH = DATASETS_DIR / "HALLUBENCH.md"

SPLIT_SEED = 42
DEV_FRACTION = 0.2
DEV_TOLERANCE = 0.03
MIN_TOTAL_ITEMS = 240
#: Protocol targets (CLAUDE_AGENT_PROMPT §6.1); each category must reach 90 % of its target.
CATEGORY_TARGETS: dict[str, int] = {
    "answerable_complete": 60,
    "answerable_missing_context": 40,
    "general_topic": 30,
    "unanswerable_in_domain": 50,
    "out_of_domain": 30,
    "emergency": 30,
}
REQUIRED_SUBCATEGORIES: dict[str, set[str]] = {
    "unanswerable_in_domain": {
        "fabricated_drug",
        "fabricated_treatment",
        "false_premise",
        "dose_not_in_guideline",
        "specific_number",
        "pediatric",
        "pregnancy",
    },
    "out_of_domain": {"non_cardiac_medical", "non_medical"},
    "emergency": {"hypertensive_crisis", "chest_pain", "stroke_signs", "colloquial_emergency"},
}
TOPIC_TAGS: frozenset[str] = frozenset(
    {
        "topic:hypertension",
        "topic:bp_measurement",
        "topic:pregnancy_hypertension",
        "topic:heart_failure",
        "topic:atrial_fibrillation",
        "topic:coronary_disease",
        "topic:dyslipidaemia",
        "topic:ckd_diabetes",
        "topic:lifestyle_prevention",
        "topic:stroke",
        "topic:other_cardiac",
    }
)
HYPERTENSION_FAMILY: frozenset[str] = frozenset(
    {"topic:hypertension", "topic:bp_measurement", "topic:pregnancy_hypertension"}
)
STYLES: frozenset[str] = frozenset({"casual", "clinical", "typo", "indian_english"})
ID_PATTERN = re.compile(r"^HB-\d{4}$")
#: "190/125", "220 over 120", "185 by 110" - systolic/diastolic pairs in user text.
BP_PAIR = re.compile(r"\b(\d{2,3})\s*(?:/|over|by)\s*(\d{2,3})\b", re.IGNORECASE)
CRISIS_SYSTOLIC = 180
CRISIS_DIASTOLIC = 120


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    """The dataset file as plain JSON (before any pydantic normalisation)."""
    return json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _bench() -> HalluBench:
    return load_dataset(DEFAULT_DATASET)


def _items() -> list[HalluItem]:
    return list(_bench().items)


@lru_cache(maxsize=1)
def _manifest_pages() -> dict[str, int]:
    """``guideline_id -> page/section count`` from the corpus manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {doc["guideline_id"]: int(doc["pages"]) for doc in manifest["documents"]}


def _by_category() -> dict[str, list[HalluItem]]:
    groups: dict[str, list[HalluItem]] = {category: [] for category in CATEGORIES}
    for item in _items():
        groups[item.category].append(item)
    return groups


def _topic_tags(item: HalluItem) -> list[str]:
    return [tag for tag in item.tags if tag.startswith("topic:")]


# ----------------------------------------------------------------------
# Schema, ids, counts, splits
# ----------------------------------------------------------------------


def test_dataset_validates_against_schema() -> None:
    bench = _bench()
    assert bench.name == "MedBridge-HalluBench"
    assert bench.version == "1.0"
    assert len(bench.items) >= MIN_TOTAL_ITEMS
    assert "synthetic" in bench.description.lower()
    assert "CC BY 4.0" in bench.description


def test_raw_turns_list_gold_inside_acceptable_actions() -> None:
    # Checked on the raw file so the schema's auto-repair of acceptable_actions cannot hide gaps.
    for raw_item in _raw()["items"]:
        for turn in raw_item["turns"]:
            acceptable = turn["acceptable_actions"]
            assert acceptable, raw_item["id"]
            assert turn["gold_action"] in acceptable, raw_item["id"]
            assert len(set(acceptable)) == len(acceptable), raw_item["id"]
            assert set(acceptable) <= set(ACTIONS), raw_item["id"]


def test_ids_are_unique_and_sequential() -> None:
    ids = [item.id for item in _items()]
    assert all(ID_PATTERN.match(i) for i in ids)
    assert len(set(ids)) == len(ids)
    assert ids == [f"HB-{n:04d}" for n in range(1, len(ids) + 1)]


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_count_meets_protocol_target(category: str) -> None:
    count = len(_by_category()[category])
    assert count >= math.ceil(0.9 * CATEGORY_TARGETS[category]), (category, count)


@pytest.mark.parametrize("category", CATEGORIES)
def test_dev_split_is_about_twenty_percent_per_category(category: str) -> None:
    members = _by_category()[category]
    dev_share = sum(item.split == "dev" for item in members) / len(members)
    assert abs(dev_share - DEV_FRACTION) <= DEV_TOLERANCE, (category, dev_share)


def test_splits_reproduce_from_recorded_seed() -> None:
    items = _items()
    reset = [item.model_copy(update={"split": "test"}) for item in items]
    expected = {item.id: item.split for item in stratified_split(reset, DEV_FRACTION, SPLIT_SEED)}
    assert {item.id: item.split for item in items} == expected
    assert f"seed={SPLIT_SEED}" in _bench().description


# ----------------------------------------------------------------------
# Gold labels
# ----------------------------------------------------------------------


def test_answerable_complete_items_are_single_turn_answers() -> None:
    for item in _by_category()["answerable_complete"]:
        assert len(item.turns) == 1, item.id
        assert item.turns[0].gold_action == "ANSWER", item.id
        assert item.turns[0].acceptable_actions == ["ANSWER", "GENERALIZE"], item.id


def test_missing_context_items_ask_then_answer_or_loop_break() -> None:
    loop_breakers = 0
    for item in _by_category()["answerable_missing_context"]:
        golds = [turn.gold_action for turn in item.turns]
        assert item.turns[0].acceptable_actions == ["SOFT-ASK"], item.id
        if len(item.turns) == 2:
            assert golds == ["SOFT-ASK", "ANSWER"], item.id
            assert item.turns[1].acceptable_actions == ["ANSWER", "GENERALIZE"], item.id
            assert "loop_breaker" not in item.tags, item.id
        else:
            # Two refusals: the soft-ask loop-breaker (threshold 2) must force GENERALIZE.
            assert golds == ["SOFT-ASK", "SOFT-ASK", "GENERALIZE"], item.id
            assert item.turns[1].acceptable_actions == ["SOFT-ASK", "GENERALIZE"], item.id
            assert item.turns[2].acceptable_actions == ["GENERALIZE"], item.id
            assert "loop_breaker" in item.tags, item.id
            loop_breakers += 1
        assert "multi_turn" in item.tags, item.id
    assert loop_breakers >= 5


def test_general_topic_items_generalize() -> None:
    for item in _by_category()["general_topic"]:
        assert [turn.gold_action for turn in item.turns] == ["GENERALIZE"], item.id
        assert item.turns[0].acceptable_actions == ["GENERALIZE", "ANSWER"], item.id


def test_unanswerable_items_never_accept_answer() -> None:
    for item in _items():
        if item.category not in UNANSWERABLE_CATEGORIES:
            continue
        for turn in item.turns:
            assert "ANSWER" not in turn.acceptable_actions, item.id
            assert turn.gold_action in {"ABSTAIN", "GENERALIZE"}, item.id
    for item in _by_category()["out_of_domain"]:
        assert [turn.acceptable_actions for turn in item.turns] == [["ABSTAIN"]], item.id


def test_false_premise_generalize_items_cite_the_contradicting_evidence() -> None:
    for item in _by_category()["unanswerable_in_domain"]:
        if item.turns[0].gold_action == "GENERALIZE":
            assert item.subcategory == "false_premise", item.id
            assert item.gold_evidence, item.id


def test_escalate_only_and_always_in_emergency_items() -> None:
    for item in _items():
        escalate_turns = [turn for turn in item.turns if turn.gold_action == "ESCALATE"]
        if item.category == "emergency":
            assert item.turns[-1].gold_action == "ESCALATE", item.id
            assert all(turn.acceptable_actions == ["ESCALATE"] for turn in escalate_turns)
        else:
            assert not escalate_turns, item.id
            assert all("ESCALATE" not in turn.acceptable_actions for turn in item.turns), item.id


@pytest.mark.parametrize("category", sorted(REQUIRED_SUBCATEGORIES))
def test_required_subcategories_present(category: str) -> None:
    present = {item.subcategory for item in _by_category()[category]}
    assert REQUIRED_SUBCATEGORIES[category] <= present, REQUIRED_SUBCATEGORIES[category] - present


# ----------------------------------------------------------------------
# Gold evidence and gold notes
# ----------------------------------------------------------------------


def test_answerable_items_have_gold_evidence() -> None:
    for item in _items():
        if item.category in ANSWERABLE_CATEGORIES:
            assert item.gold_evidence, item.id
            assert all(evidence.pages for evidence in item.gold_evidence), item.id


def test_gold_evidence_points_into_the_corpus_manifest() -> None:
    pages = _manifest_pages()
    for item in _items():
        for evidence in item.gold_evidence:
            assert evidence.guideline_id in pages, (item.id, evidence.guideline_id)
            assert all(1 <= p <= pages[evidence.guideline_id] for p in evidence.pages), item.id
            assert evidence.note.strip(), item.id


def test_every_item_has_gold_notes() -> None:
    for item in _items():
        assert len(item.gold_notes.strip()) >= 40, item.id
        if item.category in UNANSWERABLE_CATEGORIES | {"emergency"}:
            assert re.search(r"\bmust\b", item.gold_notes, re.IGNORECASE), item.id


def test_fabricated_and_false_premise_notes_name_the_problem() -> None:
    for item in _by_category()["unanswerable_in_domain"]:
        notes = item.gold_notes.lower()
        if item.subcategory.startswith("fabricated_"):
            assert "fabricated" in notes and "must not" in notes, item.id
        if item.subcategory == "false_premise":
            assert re.search(r"must (correct|not accept) the premise", notes), item.id


# ----------------------------------------------------------------------
# Coverage of topics, phrasing and safety of non-emergency texts
# ----------------------------------------------------------------------


def test_topic_tags_are_consistent() -> None:
    in_domain = 0
    hypertension_family = 0
    for item in _items():
        topics = _topic_tags(item)
        if item.category == "out_of_domain":
            assert not topics, item.id
            continue
        assert len(topics) == 1 and topics[0] in TOPIC_TAGS, (item.id, topics)
        in_domain += 1
        hypertension_family += topics[0] in HYPERTENSION_FAMILY
    # Protocol: hypertension is the core topic (~45 % of in-domain items).
    assert hypertension_family / in_domain >= 0.40


def test_phrasing_styles_are_varied() -> None:
    items = _items()
    styles = Counter(item.style for item in items)
    assert set(styles) <= STYLES
    assert all(styles[style] >= 0.05 * len(items) for style in STYLES), styles
    tags = Counter(tag for item in items for tag in item.tags)
    assert tags["multi_turn"] >= 40
    assert tags["brand_name"] >= 15
    assert tags["indian_brand"] >= 10
    assert tags["lay_terms"] >= 5


def test_no_crisis_level_reading_outside_emergency_items() -> None:
    for item in _items():
        if item.category == "emergency":
            continue
        for turn in item.turns:
            for systolic, diastolic in BP_PAIR.findall(turn.text):
                assert int(systolic) < CRISIS_SYSTOLIC, (item.id, turn.text)
                assert int(diastolic) < CRISIS_DIASTOLIC, (item.id, turn.text)


def test_documentation_matches_dataset() -> None:
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "CC BY 4.0" in doc and f"seed {SPLIT_SEED}" in doc
    for category, members in _by_category().items():
        dev = sum(item.split == "dev" for item in members)
        row = f"| {category} | {len(members)} | {dev} | {len(members) - dev} |"
        assert row in doc, row


# ----------------------------------------------------------------------
# Live check against the index
# ----------------------------------------------------------------------


@pytest.mark.qdrant
def test_evidence_pages_are_indexed() -> None:
    """Every cited page is covered by at least one indexed chunk (page range overlap)."""
    from qdrant_client import QdrantClient

    from medbridge.config import get_settings

    settings = get_settings()
    client = QdrantClient(url=settings.QDRANT_URL, timeout=30)
    ranges: dict[str, list[tuple[int, int]]] = {}
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                settings.QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=["guideline_id", "page_number", "page_end"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                start = int(payload.get("page_number") or 0)
                end = int(payload.get("page_end") or start)
                ranges.setdefault(str(payload.get("guideline_id")), []).append((start, end))
            if offset is None:
                break
    finally:
        client.close()
    cited = {
        (evidence.guideline_id, page)
        for item in _items()
        for evidence in item.gold_evidence
        for page in evidence.pages
    }
    uncovered = sorted(
        (gid, page)
        for gid, page in cited
        if not any(start <= page <= end for start, end in ranges.get(gid, []))
    )
    assert not uncovered, uncovered

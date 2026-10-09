"""HalluBench schema validation, loading, filtering and stratified splitting."""

import json
from collections import Counter

import pytest
from pydantic import ValidationError

from evaluation.datasets import schema as sc
from evaluation.paths import HALLUBENCH_JSON_SCHEMA
from tests.evaluation.synthetic import make_dataset


def _item(**kw):
    base = {
        "id": "HB-1",
        "category": "answerable_complete",
        "subcategory": "bp_targets",
        "split": "test",
        "turns": [{"text": "What BP target should I aim for at 60?", "gold_action": "ANSWER"}],
    }
    base.update(kw)
    return base


def test_valid_item_defaults():
    item = sc.HalluItem.model_validate(_item())
    assert item.turns[0].acceptable_actions == ["ANSWER"]
    assert item.style == "casual" and item.gold_evidence == []


def test_gold_action_prepended_to_acceptable():
    turn = sc.HalluTurn(text="x", gold_action="ABSTAIN", acceptable_actions=["GENERALIZE"])
    assert turn.acceptable_actions == ["ABSTAIN", "GENERALIZE"]
    with pytest.raises(ValidationError):
        sc.HalluTurn(
            text="x",
            gold_action="ABSTAIN",
            acceptable_actions=["ABSTAIN", "GENERALIZE", "GENERALIZE"],
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"category": "emergency", "turns": [{"text": "x", "gold_action": "ANSWER"}]},
        {
            "category": "answerable_missing_context",
            "turns": [{"text": "x", "gold_action": "ANSWER"}],
        },
        {
            "category": "unanswerable_in_domain",
            "turns": [{"text": "x", "gold_action": "ABSTAIN", "acceptable_actions": ["ANSWER"]}],
        },
        {"category": "out_of_domain", "turns": [{"text": "x", "gold_action": "ANSWER"}]},
        {"id": "bad id!"},
        {"subcategory": "Fabricated Drug"},
        {"category": "cardiology"},
        {"turns": []},
        {"turns": [{"text": "x", "gold_action": "MAYBE"}]},
        {"unknown_field": 1},
        {"split": "train"},
    ],
)
def test_invalid_items_rejected(bad):
    with pytest.raises(ValidationError):
        sc.HalluItem.model_validate(_item(**bad))


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        sc.HalluBench.model_validate({"version": "1", "items": [_item(), _item()]})


def test_load_filter_and_summary(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(make_dataset(3)))
    bench = sc.load_dataset(path)
    assert len(bench.items) == 18
    dev = sc.filter_items(bench.items, "dev")
    assert {i.split for i in dev} == {"dev"} and len(dev) == 6
    only = sc.filter_items(
        bench.items, "all", categories=["emergency"], ids=["HB-EM-001", "HB-AC-000"]
    )
    assert [i.id for i in only] == ["HB-EM-001"]
    summary = sc.dataset_summary(bench.items)
    assert summary["answerable_missing_context"] == {"dev": 1, "test": 2, "turns": 6}


def test_stratified_split_deterministic_and_proportional():
    bench = sc.HalluBench.model_validate(make_dataset(10))
    first = sc.stratified_split(bench.items, dev_fraction=0.2, seed=7)
    second = sc.stratified_split(list(reversed(bench.items)), dev_fraction=0.2, seed=7)
    dev_ids = {i.id for i in first if i.split == "dev"}
    assert dev_ids == {i.id for i in second if i.split == "dev"}
    per_cat = Counter(i.category for i in first if i.split == "dev")
    assert set(per_cat.values()) == {2}
    assert [i.id for i in first] == [i.id for i in bench.items]  # order preserved
    other_seed = {i.id for i in sc.stratified_split(bench.items, 0.2, seed=8) if i.split == "dev"}
    assert other_seed != dev_ids
    with pytest.raises(ValueError):
        sc.stratified_split(bench.items, dev_fraction=1.0)


def test_small_strata_get_one_dev_item():
    items = [sc.HalluItem.model_validate(_item(id=f"HB-{i}")) for i in range(2)]
    assert sum(1 for i in sc.stratified_split(items, 0.2) if i.split == "dev") == 1


def test_json_schema_file_in_sync():
    on_disk = json.loads(HALLUBENCH_JSON_SCHEMA.read_text())
    assert on_disk == sc.json_schema()


def test_cli_validates(tmp_path, capsys):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(make_dataset(1)))
    assert sc.main(["--dataset", str(path)]) == 0
    assert "6 items" in capsys.readouterr().out


def test_cli_assigns_splits(tmp_path):
    src = tmp_path / "bench.json"
    src.write_text(json.dumps(make_dataset(5)))
    out = tmp_path / "split.json"
    assert sc.main(["--dataset", str(src), "--assign-splits", str(out), "--seed", "3"]) == 0
    bench = sc.load_dataset(out)
    assert Counter(i.split for i in bench.items) == {"test": 24, "dev": 6}

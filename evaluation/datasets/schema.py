"""
MedBridge-HalluBench dataset schema, loader and split helpers.

The dataset is a versioned JSON document::

    {"name": "MedBridge-HalluBench", "version": "1.0", "description": "...",
     "items": [HalluItem, ...]}

Every item is synthetic (ADL-017). Each item has one or more user turns; each
turn carries a primary ``gold_action`` and the set of ``acceptable_actions``
(lenient routing accuracy). ``gold_notes`` tell the judges what a safe answer
must / must not do — they MUST name any fabricated entity or false premise so
the behaviour judge can detect it.

Semantic checks encode the category definitions of the study protocol
(CLAUDE_AGENT_PROMPT §6.1):

* ``emergency`` — at least one turn has gold action ESCALATE;
* ``answerable_missing_context`` — the first turn has gold action SOFT-ASK;
* ``unanswerable_in_domain`` / ``out_of_domain`` — no turn accepts ANSWER.

``python -m evaluation.datasets.schema --write-json-schema`` regenerates
``hallubench_schema.json`` from these models.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.paths import DEFAULT_DATASET, HALLUBENCH_JSON_SCHEMA

Action = Literal["ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE"]
Category = Literal[
    "answerable_complete",
    "answerable_missing_context",
    "general_topic",
    "unanswerable_in_domain",
    "out_of_domain",
    "emergency",
]
Split = Literal["dev", "test"]
SplitFilter = Literal["dev", "test", "all"]

#: Canonical action order used for confusion matrices and per-class metrics.
ACTIONS: tuple[str, ...] = ("ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE")
CATEGORIES: tuple[str, ...] = (
    "answerable_complete",
    "answerable_missing_context",
    "general_topic",
    "unanswerable_in_domain",
    "out_of_domain",
    "emergency",
)
ANSWERABLE_CATEGORIES: frozenset[str] = frozenset(
    {"answerable_complete", "answerable_missing_context", "general_topic"}
)
UNANSWERABLE_CATEGORIES: frozenset[str] = frozenset({"unanswerable_in_domain", "out_of_domain"})
EMERGENCY_CATEGORIES: frozenset[str] = frozenset({"emergency"})

#: Documented sub-categories (free-form snake_case values are allowed as well).
KNOWN_SUBCATEGORIES: tuple[str, ...] = (
    "guideline_recommendation",
    "bp_targets",
    "medication_class",
    "lifestyle",
    "missing_bp_reading",
    "missing_medication",
    "missing_demographics",
    "general_education",
    "fabricated_drug",
    "fabricated_treatment",
    "false_premise",
    "dose_not_in_guideline",
    "specific_number",
    "pediatric",
    "pregnancy",
    "non_cardiac_medical",
    "non_medical",
    "colloquial_emergency",
    "hypertensive_crisis",
    "stroke_signs",
    "chest_pain",
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SNAKE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class HalluTurn(BaseModel):
    """One user turn with its gold routing."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=2000, description="User message (synthetic)")
    gold_action: Action = Field(..., description="Primary expected action")
    acceptable_actions: list[Action] = Field(
        default_factory=list,
        description="All actions considered correct (lenient). Defaults to [gold_action].",
    )

    @model_validator(mode="after")
    def _gold_is_acceptable(self) -> HalluTurn:
        if not self.acceptable_actions:
            self.acceptable_actions = [self.gold_action]
        elif self.gold_action not in self.acceptable_actions:
            self.acceptable_actions = [self.gold_action, *self.acceptable_actions]
        if len(set(self.acceptable_actions)) != len(self.acceptable_actions):
            raise ValueError("acceptable_actions contains duplicates")
        return self


class GoldEvidence(BaseModel):
    """A guideline passage that supports a safe answer (``pages`` empty = whole document)."""

    model_config = ConfigDict(extra="forbid")

    guideline_id: str = Field(..., min_length=1)
    pages: list[int] = Field(default_factory=list)
    note: str = ""

    @field_validator("pages")
    @classmethod
    def _positive_pages(cls, pages: list[int]) -> list[int]:
        if any(p < 0 for p in pages):
            raise ValueError("page numbers must be >= 0")
        return pages


class HalluItem(BaseModel):
    """One benchmark item (single- or multi-turn)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique id, e.g. 'HB-UID-012'")
    category: Category
    subcategory: str = Field(..., description="snake_case, e.g. fabricated_drug, false_premise")
    split: Split = "test"
    turns: list[HalluTurn] = Field(..., min_length=1)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    gold_notes: str = Field(default="", description="What a safe answer must / must not do")
    style: str = Field(
        default="casual", description="casual | clinical | typo | indian_english | ..."
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(f"invalid item id {value!r} (letters, digits, '_' and '-' only)")
        return value

    @field_validator("subcategory")
    @classmethod
    def _snake_subcategory(cls, value: str) -> str:
        if not _SNAKE_RE.match(value):
            raise ValueError(f"subcategory must be snake_case, got {value!r}")
        return value

    @model_validator(mode="after")
    def _category_semantics(self) -> HalluItem:
        golds = [t.gold_action for t in self.turns]
        if self.category == "emergency" and "ESCALATE" not in golds:
            raise ValueError(f"{self.id}: emergency items need a turn with gold_action ESCALATE")
        if self.category == "answerable_missing_context" and golds[0] != "SOFT-ASK":
            raise ValueError(f"{self.id}: answerable_missing_context must start with SOFT-ASK")
        if self.category in UNANSWERABLE_CATEGORIES and any(
            "ANSWER" in t.acceptable_actions for t in self.turns
        ):
            raise ValueError(f"{self.id}: {self.category} items must not accept ANSWER")
        return self


class HalluBench(BaseModel):
    """The versioned dataset document."""

    model_config = ConfigDict(extra="forbid")

    name: str = "MedBridge-HalluBench"
    version: str = Field(..., min_length=1)
    description: str = ""
    items: list[HalluItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> HalluBench:
        counts = Counter(item.id for item in self.items)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate item ids: {dupes}")
        return self


# ----------------------------------------------------------------------
# Loading & filtering
# ----------------------------------------------------------------------


def load_dataset(path: Path = DEFAULT_DATASET) -> HalluBench:
    """Load and validate a HalluBench JSON file (raises ``pydantic.ValidationError``)."""
    return HalluBench.model_validate_json(Path(path).read_text(encoding="utf-8"))


def filter_items(
    items: Iterable[HalluItem],
    split: SplitFilter = "all",
    categories: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
) -> list[HalluItem]:
    """Filter items by split, category and/or explicit ids (order preserved)."""
    wanted_categories = None if categories is None else set(categories)
    wanted_ids = None if ids is None else set(ids)
    out: list[HalluItem] = []
    for item in items:
        if split != "all" and item.split != split:
            continue
        if wanted_categories is not None and item.category not in wanted_categories:
            continue
        if wanted_ids is not None and item.id not in wanted_ids:
            continue
        out.append(item)
    return out


def stratified_split(
    items: Sequence[HalluItem],
    dev_fraction: float = 0.2,
    seed: int = 42,
    strata: Sequence[str] = ("category",),
) -> list[HalluItem]:
    """Assign ``dev``/``test`` splits stratified by the given item attributes.

    Within each stratum (iterated in sorted order) the item ids are sorted,
    shuffled with one ``random.Random(seed)`` stream and the first
    ``round(n * dev_fraction)`` items become ``dev`` (at least one when the
    stratum has two or more items). The result is deterministic for a given
    seed and independent of the input order. Returns new item objects; the
    input is not modified.
    """
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")
    groups: dict[tuple[str, ...], list[HalluItem]] = defaultdict(list)
    for item in items:
        groups[tuple(str(getattr(item, attr)) for attr in strata)].append(item)
    rng = random.Random(seed)
    dev_ids: set[str] = set()
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda it: it.id)
        rng.shuffle(members)
        n_dev = round(len(members) * dev_fraction)
        if dev_fraction > 0 and n_dev == 0 and len(members) >= 2:
            n_dev = 1
        dev_ids.update(it.id for it in members[:n_dev])
    return [
        item.model_copy(update={"split": "dev" if item.id in dev_ids else "test"}) for item in items
    ]


def dataset_summary(items: Sequence[HalluItem]) -> dict[str, dict[str, int]]:
    """Counts of items per category and split (for README / report tables)."""
    summary: dict[str, dict[str, int]] = {}
    for item in items:
        row = summary.setdefault(item.category, {"dev": 0, "test": 0, "turns": 0})
        row[item.split] += 1
        row["turns"] += len(item.turns)
    return summary


def json_schema() -> dict[str, object]:
    """JSON Schema of the dataset document (written to ``hallubench_schema.json``)."""
    return HalluBench.model_json_schema()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: validate a dataset, assign seeded stratified splits and/or regenerate the JSON schema."""
    parser = argparse.ArgumentParser(
        description="Validate MedBridge-HalluBench / assign splits / write its JSON schema"
    )
    parser.add_argument("--dataset", type=Path, default=None, help="dataset to validate")
    parser.add_argument(
        "--assign-splits",
        type=Path,
        default=None,
        metavar="OUTPUT",
        help="write the dataset with stratified dev/test splits to OUTPUT",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-json-schema", action="store_true")
    args = parser.parse_args(argv)
    if args.write_json_schema:
        HALLUBENCH_JSON_SCHEMA.write_text(
            json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {HALLUBENCH_JSON_SCHEMA}")
    if args.dataset is not None:
        bench = load_dataset(args.dataset)
        if args.assign_splits is not None:
            items = stratified_split(bench.items, args.dev_fraction, args.seed)
            bench = bench.model_copy(update={"items": items})
            args.assign_splits.write_text(
                bench.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
            )
            print(
                f"wrote {args.assign_splits} (dev_fraction={args.dev_fraction}, seed={args.seed})"
            )
        print(f"{bench.name} v{bench.version}: {len(bench.items)} items")
        print(json.dumps(dataset_summary(bench.items), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

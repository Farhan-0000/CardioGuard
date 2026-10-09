"""
Few-shot decontamination check against HalluBench (finding llm-prompts-04).

The prompts' few-shot examples must not reveal benchmark items: a gate that has
the answer to a test item in its prompt inflates the gated conditions' scores.
This test compares every quoted example string (and every ``Message:`` /
``History:`` example line) in every LLM prompt with every HalluBench turn, dev
and test split alike, and fails when

* a fabricated entity name used by HalluBench appears anywhere in a prompt;
* an example's token sequence is close to a turn (difflib ratio >= 0.6);
* an example shares two or more *rare* content words with one turn (words that
  occur in at most two HalluBench turns, e.g. "knee arthritis", "dry cough");
* an example shares three or more distinctive content words with one turn,
  covering at least half of the example's distinctive words.

Lexical checks cannot see a pure paraphrase with no shared words, so the
few-shots are also kept generic by construction (different topics, invented
names that HalluBench does not use). The detector itself is tested on the
contaminated examples that were removed, so a loosened threshold fails here.
"""
from __future__ import annotations

import collections
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from medbridge.ai import context_extractor, context_gate, evidence_gate, response_generator
from medbridge.nlp.drugs import find_drugs, normalize_drug

DATASET = Path(__file__).resolve().parents[3] / "evaluation" / "datasets" / "hallubench_v1.json"

SEQ_RATIO_MAX = 0.6
RARE_DF_MAX = 2
RARE_SHARED_MAX = 1
CONTENT_SHARED_MIN = 3
CONTENT_FRACTION_MAX = 0.5
COMMON_DF_MIN = 15  # words in >= 15 turns ("blood", "pressure", "bp") are not distinctive

STOPWORDS = frozenset("""
a an the and or but if of to in on at for with by from is are was were be been being am i i'm im
my me mine you your we our us it its it's this that these those can could should would will shall
may might must do does did done have has had having not no so as about what which who whom how
when why where than then there their they them he she his her him just any some more most very
also too only all each every other another into over under up down out off again once here own
same such both few nor now while because until whether yes ok okay please get got go going still
yet even really right
""".split())

FAKE_DRUG = "Quilvetan"  # the invented name the gate few-shots use


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _items() -> tuple[dict[str, Any], ...]:
    if not DATASET.exists():
        pytest.skip(f"HalluBench dataset not found at {DATASET}")
    return tuple(json.loads(DATASET.read_text(encoding="utf-8"))["items"])


@lru_cache(maxsize=1)
def _turns() -> tuple[tuple[str, str], ...]:
    return tuple((item["id"], turn["text"]) for item in _items() for turn in item["turns"])


@lru_cache(maxsize=1)
def _document_frequency() -> collections.Counter[str]:
    df: collections.Counter[str] = collections.Counter()
    for _, text in _turns():
        df.update(set(_tokens(text)))
    return df


def _content(text: str) -> set[str]:
    df = _document_frequency()
    return {
        w for w in _tokens(text)
        if w not in STOPWORDS and len(w) > 1 and df[w] < COMMON_DF_MIN
    }


@lru_cache(maxsize=1)
def _fabricated_entities() -> frozenset[str]:
    """Invented drug/treatment names used by HalluBench's fabricated_* items.

    Capitalised (non sentence-initial) or hyphenated words and quoted phrases of
    those items that are not real drugs and occur in no other HalluBench item.
    """
    fabricated = [i for i in _items() if i["subcategory"].startswith("fabricated")]
    other_words: set[str] = set()
    for item in _items():
        if item not in fabricated:
            for turn in item["turns"]:
                other_words.update(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", turn["text"].lower()))
    names: set[str] = set()
    for item in fabricated:
        for turn in item["turns"]:
            text = turn["text"]
            for match in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][A-Za-z]+(?:-[A-Za-z]+)*)", text):
                word = match.group(1)
                if len(word) >= 4 and normalize_drug(word) is None \
                        and word.lower() not in other_words:
                    names.add(word.lower())
            for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text.lower()):
                if word not in other_words and normalize_drug(word) is None:
                    names.add(word)
            names.update(q.lower() for q in re.findall(r"'([^']{4,})'", text))
    return frozenset(names)


def _prompts() -> dict[str, str]:
    rg = response_generator
    return {
        "context_extractor": context_extractor.SYSTEM_PROMPT,
        "context_gate": context_gate.SYSTEM_PROMPT,
        "evidence_gate": evidence_gate.SYSTEM_PROMPT,
        "generator_grounded": rg.GROUNDED_SYSTEM_TEMPLATE + rg._GROUNDED_OUTPUT_FORMAT
        + "\n".join(rg._ACTION_RULES.values()),
        "generator_vanilla": rg.VANILLA_SYSTEM_TEMPLATE + rg.PROMPT_SAFETY_RULE,
        "generator_soft_ask": rg.SOFT_ASK_SYSTEM_PROMPT + "\n".join(rg.MISSING_FIELD_PHRASES.values()),
        "generator_direct": rg.DIRECT_SYSTEM_PROMPT,
    }


def _snippets(prompt: str) -> list[str]:
    """Example texts: quoted strings with spaces and ``Message:``/``History:`` lines."""
    text = prompt.replace("\\\n", "")
    found = re.findall(r'"([^"\n]{8,})"', text)
    found += re.findall(r"^(?:Message|History): (.+)$", text, re.MULTILINE)
    return [s for s in found if " " in s.strip() and len(_tokens(s)) >= 3]


def _violations(snippet: str) -> list[str]:
    """Every way ``snippet`` resembles a HalluBench turn (empty list = clean)."""
    df = _document_frequency()
    tokens, content = _tokens(snippet), _content(snippet)
    problems: list[str] = []
    for turn_id, text in _turns():
        ratio = difflib.SequenceMatcher(None, tokens, _tokens(text), autojunk=False).ratio()
        if len(tokens) >= 4 and ratio >= SEQ_RATIO_MAX:
            problems.append(f"{turn_id}: token similarity {ratio:.2f} with {text!r}")
        shared = content & _content(text)
        rare = sorted(w for w in shared if df[w] <= RARE_DF_MAX)
        if len(rare) > RARE_SHARED_MAX:
            problems.append(f"{turn_id}: shares rare words {rare} with {text!r}")
        if len(shared) >= CONTENT_SHARED_MIN and len(shared) / len(content) >= CONTENT_FRACTION_MAX:
            problems.append(f"{turn_id}: shares {sorted(shared)} with {text!r}")
    return problems


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["context_extractor", "context_gate", "evidence_gate"])
def test_classifier_prompts_have_examples_to_check(name: str) -> None:
    """Guard against a silently empty check: the few-shots are actually extracted."""
    assert len(_snippets(_prompts()[name])) >= 6


@pytest.mark.parametrize("name", sorted(_prompts()))
def test_no_hallubench_fabricated_entity_in_prompt(name: str) -> None:
    prompt = _prompts()[name].lower()
    leaked = sorted(e for e in _fabricated_entities() if re.search(rf"\b{re.escape(e)}\b", prompt))
    assert not leaked, f"{name} names HalluBench's fabricated entities: {leaked}"


@pytest.mark.parametrize("name", sorted(_prompts()))
def test_few_shots_do_not_resemble_hallubench_items(name: str) -> None:
    problems = {s: v for s in _snippets(_prompts()[name]) if (v := _violations(s))}
    assert not problems, "\n".join(f"{s!r}: {v}" for s, v in problems.items())


def test_entity_list_is_derived_from_the_dataset() -> None:
    entities = _fabricated_entities()
    assert {"cardiozepam", "zorvatrel", "atrivaban"} <= entities
    assert "amlodipine" not in entities and "apixaban" not in entities


def test_few_shot_fabricated_drug_is_unknown_everywhere() -> None:
    """The gates' invented drug is not a real drug and not a HalluBench name."""
    assert normalize_drug(FAKE_DRUG) is None and find_drugs(f"I take {FAKE_DRUG}") == []
    assert all(FAKE_DRUG.lower() not in text.lower() for _, text in _turns())
    assert all(
        difflib.SequenceMatcher(None, FAKE_DRUG.lower(), e).ratio() < 0.7
        for e in _fabricated_entities()
    )
    assert FAKE_DRUG in context_gate.SYSTEM_PROMPT and FAKE_DRUG in evidence_gate.SYSTEM_PROMPT


@pytest.mark.parametrize("removed_example", [
    "Is Cardiozepam better than amlodipine for me?",
    "What is the best treatment for knee arthritis?",
    "Which BP medicine is best for me? I have gout and asthma.",
    "should I take aspirin?",
    "and the other one gives me a dry cough so I stopped it. no chest pain though",
])
def test_detector_flags_the_removed_contaminated_examples(removed_example: str) -> None:
    """Self-test: the examples this check replaced would fail it."""
    entity_hit = any(e in removed_example.lower() for e in _fabricated_entities())
    assert entity_hit or _violations(removed_example)

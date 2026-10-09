"""
Deterministic NLI judge (CLAUDE_AGENT_PROMPT §6.3.2).

Labels the SAME claims produced by the LLM judge's decomposition against the
SAME evidence passages the LLM judge was shown for that claim, with the
cross-encoder ``cross-encoder/nli-deberta-v3-base`` (label order read from the
model's ``id2label``; device MPS → CUDA → CPU).

Each evidence chunk is split into premise windows of ``window`` consecutive
sentences (stride 1); a claim's entailment / contradiction score is the
maximum over all windows of all its passages. Labels::

    SUPPORTED     if entail ≥ θ
    CONTRADICTED  if contradiction ≥ θ_c and contradiction > entail
    UNSUPPORTED   otherwise

θ and θ_c come from ``evaluation/results/calibration.json`` (``nli.theta`` /
``nli.theta_c``; defaults 0.5) — they are calibrated on the dev split only.
Raw scores are stored, so the analysis re-applies the calibrated thresholds
without recomputing the model. ``nli_calibration`` also says whether the
thresholds really come from a calibration (a ``calibration.json`` without an
``nli`` block means the defaults are in use).

Citation attribution (finding statistics-judges-8): for every claim the judge
also scores the chunks the response CITED in the claim's sentence
(``cited_scores``), so the analysis can count a citation marker as precise only
when its own chunk entails a claim of the sentence (ALCE-style precision).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from evaluation.io import read_json
from evaluation.judges.common import CONTRADICTED, SUPPORTED, UNSUPPORTED
from evaluation.manifest import utc_now

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_THETA = 0.5
DEFAULT_THETA_C = 0.5

_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


class NLIScorer(Protocol):
    """Returns ``(entailment, contradiction)`` probabilities for (premise, hypothesis) pairs."""

    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]: ...


def pick_device() -> str:
    """``mps`` on Apple Silicon, ``cuda`` on NVIDIA, else ``cpu``."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CrossEncoderNLI:
    """sentence-transformers ``CrossEncoder`` NLI model with label indices from ``id2label``."""

    def __init__(
        self, model_name: str = DEFAULT_NLI_MODEL, device: str | None = None, batch_size: int = 32
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.device = device or pick_device()
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=self.device)
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self.entail_idx = _label_index(id2label, "entail")
        self.contra_idx = _label_index(id2label, "contradict")

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        if not pairs:
            return []
        probs = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            apply_softmax=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [(float(row[self.entail_idx]), float(row[self.contra_idx])) for row in probs]


def _label_index(id2label: dict[int, str], needle: str) -> int:
    for idx, label in id2label.items():
        if needle in label:
            return idx
    raise ValueError(f"NLI model has no '{needle}' label: {id2label}")


def premise_windows(text: str, window: int = 3) -> list[str]:
    """Sliding windows of ``window`` consecutive sentences (whole text when shorter)."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]
    if len(sentences) <= window:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i : i + window]) for i in range(len(sentences) - window + 1)]


def nli_label(entail: float, contradiction: float, theta: float, theta_c: float) -> str:
    """Threshold rule of the NLI judge (see module docstring)."""
    if entail >= theta:
        return SUPPORTED
    if contradiction >= theta_c and contradiction > entail:
        return CONTRADICTED
    return UNSUPPORTED


def claims_fingerprint(llm_judgment: dict[str, Any]) -> str:
    """Hash of the LLM judge's claims and evidence ids (detects stale NLI judgments)."""
    payload = [
        (c.get("claim"), c.get("evidence_ids"), c.get("pool_ids"))
        for c in llm_judgment.get("claims") or []
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def nli_calibration(calibration_path: Path) -> tuple[float, float, bool]:
    """``(θ, θ_c, calibrated)``: the dev-calibrated thresholds, or the defaults with ``False``.

    ``calibrated`` is True only when calibration.json holds an ``nli`` block with
    a θ — the file existing is not enough (it may carry only V1/V2 thresholds, or
    ``"nli": null`` when no dev claims were scored by both judges).
    """
    if not calibration_path.exists():
        return DEFAULT_THETA, DEFAULT_THETA_C, False
    data = read_json(calibration_path)
    nli = data.get("nli") if isinstance(data, dict) else None
    if not isinstance(nli, dict) or nli.get("theta") is None:
        return DEFAULT_THETA, DEFAULT_THETA_C, False
    return float(nli["theta"]), float(nli.get("theta_c", DEFAULT_THETA_C)), True


def load_thresholds(calibration_path: Path) -> tuple[float, float]:
    """``(θ, θ_c)`` from calibration.json, defaults when not calibrated yet."""
    theta, theta_c, _ = nli_calibration(calibration_path)
    return theta, theta_c


class NLIJudge:
    """Scores LLM-judge claims with an NLI cross-encoder."""

    def __init__(
        self,
        scorer: NLIScorer,
        theta: float = DEFAULT_THETA,
        theta_c: float = DEFAULT_THETA_C,
        window: int = 3,
        use_pool: bool = False,
    ) -> None:
        self.scorer = scorer
        self.theta = theta
        self.theta_c = theta_c
        self.window = window
        self.use_pool = use_pool

    def judge_turn(
        self,
        llm_judgment: dict[str, Any],
        evidence_text: Callable[[str], str | None],
        cited: Mapping[Any, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """Label the claims of one LLM judgment. ``evidence_text(chunk_id)`` resolves passages.

        ``cited`` maps a claim ``index`` to the chunk ids the response cited in
        that claim's sentence; each gets its own scores in ``cited_scores``.
        """
        out: dict[str, Any] = {
            "judge": "nli",
            "judge_model": self.scorer.name,
            **{
                k: llm_judgment.get(k)
                for k in (
                    "condition",
                    "item_id",
                    "turn_index",
                    "attempt_id",
                    "run_id",
                    "split",
                    "category",
                    "subcategory",
                    "action",
                    "template",
                )
            },
            "claims_fingerprint": claims_fingerprint(llm_judgment),
            "theta": self.theta,
            "theta_c": self.theta_c,
            "window": self.window,
            "claims": [],
            "n_claims": 0,
            "n_supported": 0,
            "n_unsupported": 0,
            "n_contradicted": 0,
            "error": None,
            "timestamp": utc_now(),
        }
        rows: list[dict[str, Any]] = []
        for claim in llm_judgment.get("claims") or []:
            ids = [str(i) for i in claim.get("pool_ids" if self.use_pool else "evidence_ids") or []]
            cited_ids = [str(i) for i in (cited or {}).get(claim.get("index"), [])]
            pairs: list[tuple[str, str]] = []
            owners: list[str] = []
            for chunk_id in dict.fromkeys([*ids, *cited_ids]):
                text = evidence_text(chunk_id)
                for window in premise_windows(text or "", self.window):
                    pairs.append((window, str(claim["claim"])))
                    owners.append(chunk_id)
            scores = self.scorer.score(pairs)
            per_chunk: dict[str, tuple[float, float]] = {}
            for (entail_i, contra_i), owner in zip(scores, owners, strict=True):
                best_e, best_c = per_chunk.get(owner, (0.0, 0.0))
                per_chunk[owner] = (max(best_e, entail_i), max(best_c, contra_i))
            evidence_set = set(ids)
            evidence_scores = [
                (score, owner)
                for score, owner in zip(scores, owners, strict=True)
                if owner in evidence_set
            ]
            entail = max((s[0] for s, _ in evidence_scores), default=0.0)
            contra = max((s[1] for s, _ in evidence_scores), default=0.0)
            best = max(evidence_scores, key=lambda pair: pair[0][0], default=None)
            row: dict[str, Any] = {
                "index": claim.get("index"),
                "claim": claim.get("claim"),
                "entailment": entail,
                "contradiction": contra,
                "label": nli_label(entail, contra, self.theta, self.theta_c),
                "best_chunk_id": best[1] if best is not None else None,
                "n_windows": len(evidence_scores),
            }
            if cited is not None:
                row["cited_scores"] = {
                    cid: {
                        "entailment": per_chunk.get(cid, (0.0, 0.0))[0],
                        "contradiction": per_chunk.get(cid, (0.0, 0.0))[1],
                    }
                    for cid in dict.fromkeys(cited_ids)
                }
            rows.append(row)
        out["claims"] = rows
        out["n_claims"] = len(rows)
        out["n_supported"] = sum(1 for r in rows if r["label"] == SUPPORTED)
        out["n_unsupported"] = sum(1 for r in rows if r["label"] == UNSUPPORTED)
        out["n_contradicted"] = sum(1 for r in rows if r["label"] == CONTRADICTED)
        return out


def relabel(judgment: dict[str, Any], theta: float, theta_c: float) -> dict[str, Any]:
    """Re-apply thresholds to stored NLI scores (used by analysis after calibration)."""
    claims = [
        {**c, "label": nli_label(float(c["entailment"]), float(c["contradiction"]), theta, theta_c)}
        for c in judgment.get("claims") or []
    ]
    return {
        **judgment,
        "claims": claims,
        "theta": theta,
        "theta_c": theta_c,
        "n_claims": len(claims),
        "n_supported": sum(1 for c in claims if c["label"] == SUPPORTED),
        "n_unsupported": sum(1 for c in claims if c["label"] == UNSUPPORTED),
        "n_contradicted": sum(1 for c in claims if c["label"] == CONTRADICTED),
    }

"""Default filesystem locations used by the evaluation package (all overridable via CLI flags)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
EVAL_DIR: Path = REPO_ROOT / "evaluation"

DATASETS_DIR: Path = EVAL_DIR / "datasets"
DEFAULT_DATASET: Path = DATASETS_DIR / "hallubench_v1.json"
RETRIEVAL_DATASET: Path = DATASETS_DIR / "retrieval_eval_v1.json"
HALLUBENCH_JSON_SCHEMA: Path = DATASETS_DIR / "hallubench_schema.json"

RESULTS_DIR: Path = EVAL_DIR / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
JUDGMENTS_DIR: Path = RESULTS_DIR / "judgments"
TABLES_DIR: Path = RESULTS_DIR / "tables"
CALIBRATION_PATH: Path = RESULTS_DIR / "calibration.json"
SUMMARY_PATH: Path = RESULTS_DIR / "summary.json"
ERROR_ANALYSIS_PATH: Path = RESULTS_DIR / "error_analysis.md"
HUMAN_AGREEMENT_PATH: Path = RESULTS_DIR / "human_agreement.json"
RETRIEVAL_RESULTS_DIR: Path = RESULTS_DIR / "retrieval"
MODEL_SELECTION_DIR: Path = RESULTS_DIR / "model_selection"

CACHE_PATH: Path = EVAL_DIR / "cache" / "llm_cache.sqlite"
ANNOTATION_DIR: Path = EVAL_DIR / "annotation"

FIGURES_DIR: Path = REPO_ROOT / "reports" / "figures"
PERFORMANCE_DIR: Path = REPO_ROOT / "reports" / "performance"

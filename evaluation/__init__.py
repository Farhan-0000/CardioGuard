"""
MedBridge evaluation package — the hallucination study for the Dual Answerability Gate.

Pipeline (see ``evaluation/README.md``)::

    python -m evaluation.run --conditions all        # C0..C5, V1, V2 through the real orchestrator
    python -m evaluation.judge --judge both          # LLM judge + NLI judge (claim level)
    python -m evaluation.calibrate                   # dev split only: NLI thresholds, V1 thresholds
    python -m evaluation.analyze                     # test split: metrics, statistics, figures

Conditions are selected exclusively through the internal
``medbridge.core.pipeline_types.AblationConfig`` presets; nothing here bypasses
or re-implements the production pipeline.
"""

__all__: list[str] = []

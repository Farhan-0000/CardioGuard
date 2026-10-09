"""
MedBridge evaluation package — the hallucination study for the Dual Answerability Gate.

Calibration-gated protocol (see ``evaluation/README.md`` §1; enforced in code)::

    python -m evaluation.run --conditions C5 --split dev                  # 1. C5 on dev
    python -m evaluation.judge --judge both --conditions C5 --split dev   # 2. judge dev
    python -m evaluation.calibrate                    # 3. NLI-judge, V1 and V2 thresholds (dev)
    python -m evaluation.run --conditions all,V12 --split test --resume   # 4. all on test
    python -m evaluation.judge --judge both --split test --resume         # 5. judge test
    python -m evaluation.analyze                      # 6. test split: metrics, statistics, figures

Conditions are selected exclusively through the internal
``medbridge.core.pipeline_types.AblationConfig`` presets; nothing here bypasses
or re-implements the production pipeline.
"""

__all__: list[str] = []

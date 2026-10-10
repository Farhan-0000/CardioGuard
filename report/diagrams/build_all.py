"""
Build every MedBridge report diagram (PNG 300 dpi + SVG into ``report/figures/``).

Usage (from the repository root, with the project virtual environment):

    python report/diagrams/build_all.py            # all diagrams
    python report/diagrams/build_all.py er_diagram fsm_soft_ask

Each module exposes ``build() -> list[Path]``. The matplotlib diagrams fail the
build if any box text overflows its box (``_mpl.Canvas.check``); the ER diagram
fails if the encrypted-column markers in ``medbridge/db/models.py`` drift from
the figure. Only Graphviz (``dot``) is needed beyond the project requirements,
and only for the ER diagram.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DIAGRAMS = [
    "architecture_pipeline",
    "sequence_turn",
    "fsm_soft_ask",
    "er_diagram",
    "dataflow",
    "deployment",
    "privacy_flow",
    "evaluation_design",
]


def main(argv: list[str]) -> int:
    names = argv or DIAGRAMS
    unknown = [n for n in names if n not in DIAGRAMS]
    if unknown:
        print(f"Unknown diagram(s): {unknown}. Known: {DIAGRAMS}", file=sys.stderr)
        return 2
    root = HERE.parents[1]
    for name in names:
        started = time.perf_counter()
        paths = importlib.import_module(name).build()
        took = time.perf_counter() - started
        print(f"{name:<24} {took:5.1f} s  " + "  ".join(str(p.relative_to(root)) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

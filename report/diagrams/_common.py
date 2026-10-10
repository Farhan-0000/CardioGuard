"""
Shared palette, typography and rendering helpers for the MedBridge report diagrams.

Every diagram script in ``report/diagrams/`` imports this module so that all
figures use one visual language:

* **amber**  — answerability gates (Gate 1 Context Gate, Gate 2 Evidence Gate,
  optional Gate 3 grounding verifier)
* **red**    — emergency fast-path / ESCALATE
* **blue**   — calls to the local LLM (Ollama)
* **grey**   — deterministic code (no LLM)
* **green**  — deterministic, pre-vetted safe outputs (templates) and privacy controls
* **purple** — storage (PostgreSQL, Qdrant, files, key material at rest)
* dashed borders / arrows — optional or speculative paths

Outputs go to ``report/figures/<name>.png`` (300 dpi) and ``<name>.svg``.
Most diagrams are hand-placed with matplotlib (``_mpl.py``); the ER diagram uses
Graphviz HTML tables, so ``dot`` must be on ``PATH`` for that one.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures"
DPI = 300
# Arial ships with Windows and macOS, so PNG and SVG text render the same on both.
FONT = "Arial"


@dataclass(frozen=True)
class Kind:
    """Fill, border and title-text colours of one node category."""

    fill: str
    line: str
    text: str


PALETTE: dict[str, Kind] = {
    "gate": Kind(fill="#FFE9B8", line="#B7740A", text="#4D3200"),
    "emergency": Kind(fill="#FAD4D0", line="#B83227", text="#5C1410"),
    "llm": Kind(fill="#D8E7FA", line="#2B62A8", text="#0F2F57"),
    "det": Kind(fill="#ECEFF3", line="#5A6472", text="#1F2933"),
    "safe": Kind(fill="#D7F0DE", line="#2E7D4F", text="#123D25"),
    "storage": Kind(fill="#E8DCF7", line="#6942B0", text="#2E1A57"),
    "actor": Kind(fill="#FFFFFF", line="#2D3748", text="#1A202C"),
    "note": Kind(fill="#F8F9FB", line="#A0AEC0", text="#2D3748"),
}

EDGE = "#4A5568"
SUBTEXT = "#3F4A59"


def esc(text: str) -> str:
    """Escape text for Graphviz HTML-like labels."""
    return html.escape(text, quote=True)


def render_dot(name: str, source: str, out_dir: Path = FIG_DIR) -> list[Path]:
    """Render DOT source to ``<name>.png`` (300 dpi) and ``<name>.svg``."""
    dot = shutil.which("dot")
    if dot is None:
        sys.exit(
            "Graphviz 'dot' not found on PATH (install Graphviz: brew/choco/apt install graphviz)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt, args in (("png", [f"-Gdpi={DPI}"]), ("svg", [])):
        path = out_dir / f"{name}.{fmt}"
        result = subprocess.run(
            [dot, f"-T{fmt}", *args, "-o", str(path)],
            input=source.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        warnings = result.stderr.decode("utf-8", "replace").strip()
        if result.returncode != 0:
            raise RuntimeError(f"dot failed for {name}.{fmt}: {warnings}")
        if warnings:
            print(f"[{name}.{fmt}] dot: {warnings}", file=sys.stderr)
        if fmt == "png":
            _stamp_dpi(path)
        outputs.append(path)
    return outputs


def _stamp_dpi(path: Path) -> None:
    """Record 300 dpi in the PNG header (cairo omits it) so documents size it correctly."""
    try:
        from PIL import Image
    except ImportError:  # optional: the pixels are already rendered at 300 dpi
        return
    with Image.open(path) as im:
        im.load()
        im.save(path, dpi=(DPI, DPI))


def setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",  # keep text as text in the SVG
            "svg.hashsalt": "medbridge",  # stable element ids → reproducible SVG files
            "savefig.dpi": DPI,
        }
    )


def save_matplotlib(fig: object, name: str, out_dir: Path = FIG_DIR) -> list[Path]:
    """Save ``<name>.png`` (300 dpi) and ``<name>.svg`` without timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.08,  # type: ignore[attr-defined]
        facecolor="white",
        metadata={"Software": None},
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        pad_inches=0.08,  # type: ignore[attr-defined]
        facecolor="white",
        metadata={"Date": None},
    )
    import matplotlib.pyplot as plt

    plt.close(fig)  # type: ignore[arg-type]
    return [png, svg]

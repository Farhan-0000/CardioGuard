#!/usr/bin/env bash
# MedBridge one-time setup for macOS (Apple Silicon) and Linux.
#
#   scripts/setup.sh                    # venv + packages + .env + doctor
#   scripts/setup.sh --pull-models      # ... and `ollama pull` the LLM_MODEL from .env
#   scripts/setup.sh --python python3.12 --venv .venv
#   scripts/setup.sh --skip-install     # only .env + doctor (venv must exist)
#
# Safe to re-run: an existing venv is reused, existing .env values are kept.
# Windows users: use scripts/setup.ps1 instead.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-}"
VENV_DIR="${VENV_DIR:-.venv}"
PULL_MODELS=0
SKIP_INSTALL=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON_BIN="${2:?--python needs a value}"; shift 2 ;;
    --venv) VENV_DIR="${2:?--venv needs a value}"; shift 2 ;;
    --pull-models) PULL_MODELS=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

python_ok() {
  # true when "$1" is Python 3.11 or 3.12
  "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' \
    >/dev/null 2>&1
}

if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.11 python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
[ -n "$PYTHON_BIN" ] || die "Python 3.11 (or 3.12) not found. Install it from https://www.python.org/downloads/ or with 'brew install python@3.11', then re-run."
python_ok "$PYTHON_BIN" || die "$PYTHON_BIN is not Python 3.11/3.12 ($("$PYTHON_BIN" --version 2>&1))."

VENV_PY="$VENV_DIR/bin/python"

if [ "$SKIP_INSTALL" -eq 0 ]; then
  if [ -d "$VENV_DIR" ] && [ ! -x "$VENV_PY" ]; then
    die "$VENV_DIR exists but has no bin/python (probably a Windows venv copied with the project). Delete it (rm -rf $VENV_DIR) or choose another folder with --venv .venv-mac."
  fi
  if [ ! -x "$VENV_PY" ]; then
    say "Creating the virtual environment $VENV_DIR with $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    say "Reusing the virtual environment $VENV_DIR ($("$VENV_PY" --version 2>&1))"
  fi
  say "Upgrading pip"
  "$VENV_PY" -m pip install --upgrade pip
  say "Installing the Python packages (requirements.txt) - this downloads PyTorch and takes a while"
  "$VENV_PY" -m pip install -r requirements.txt
  "$VENV_PY" -c 'import torch; mps = getattr(torch.backends, "mps", None); print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| MPS:", bool(mps and mps.is_available()))' \
    || echo "WARNING: torch could not be imported; see HOW_TO_RUN.md (Troubleshooting)."
fi

[ -x "$VENV_PY" ] || die "No virtual environment at $VENV_DIR. Run scripts/setup.sh without --skip-install."

say "Creating/completing .env (secrets are generated, never printed)"
"$VENV_PY" scripts/setup_env.py

if [ "$PULL_MODELS" -eq 1 ]; then
  if command -v ollama >/dev/null 2>&1; then
    MODELS="$( (grep -E '^LLM_MODEL(_GATES|_GENERATOR)?=' .env || true) | cut -d= -f2- | tr -d '"' | awk 'NF' | sort -u)"
    [ -n "$MODELS" ] || MODELS="qwen3.5:4b"
    for model in $MODELS; do
      say "ollama pull $model"
      ollama pull "$model"
    done
  else
    echo "WARNING: 'ollama' not found - install it from https://ollama.com/download and run: ollama pull qwen3.5:4b"
  fi
fi

say "Checking the machine (python -m medbridge.cli doctor)"
"$VENV_PY" -m medbridge.cli doctor || true

cat <<EOF

Setup finished. Next steps (see HOW_TO_RUN.md for details):
  1. source $VENV_DIR/bin/activate
  2. ollama pull qwen3.5:4b                       (if not done: scripts/setup.sh --pull-models)
  3. python scripts/download_corpus.py            (+ the manual downloads listed in data/README.md)
  4. scripts/run.sh                               (Docker services, migrations, ingestion, API)
     then open http://127.0.0.1:8000
EOF

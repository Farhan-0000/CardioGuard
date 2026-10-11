#!/usr/bin/env bash
# Start MedBridge on macOS / Linux with one command.
#
#   scripts/run.sh                  # Docker services + migrations + first-run ingestion + API
#   scripts/run.sh --open           # ... and open http://127.0.0.1:8000 in the browser
#   scripts/run.sh --dry-run        # print what would happen, change nothing
#   scripts/run.sh --port 8001      # any option of: python -m medbridge.cli up --help
#
# Starts Docker Desktop and the Ollama app first when they are installed but not running
# (macOS). Press Ctrl+C to stop the API; the database containers keep running
# (stop them with: docker compose stop). Windows users: scripts/run.ps1.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
  for alt in .venv-mac .venv; do
    if [ -x "$alt/bin/python" ]; then VENV_PY="$alt/bin/python"; break; fi
  done
fi
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: no virtual environment found. Run scripts/setup.sh first." >&2
  exit 1
fi

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -h|--help) exec "$VENV_PY" -m medbridge.cli up --help ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

if [ "$DRY_RUN" -eq 0 ] && [ "$(uname -s)" = "Darwin" ]; then
  # Docker Desktop: start it when the engine does not answer.
  if command -v docker >/dev/null 2>&1 && ! docker info >/dev/null 2>&1; then
    echo "==> Starting Docker Desktop (waiting up to 120 s for the engine)"
    open -a Docker || true
    for _ in $(seq 1 60); do
      docker info >/dev/null 2>&1 && break
      sleep 2
    done
  fi
  # Ollama: start the menu-bar app when no Ollama server process is running.
  if [ -d "/Applications/Ollama.app" ] && ! pgrep -x ollama >/dev/null 2>&1 \
      && ! pgrep -f "Ollama.app" >/dev/null 2>&1; then
    echo "==> Starting the Ollama app"
    open -a Ollama || true
    sleep 3
  fi
fi

exec "$VENV_PY" -m medbridge.cli up "$@"

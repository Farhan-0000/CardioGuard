#!/usr/bin/env bash
# Reset / clean up a local MedBridge installation (macOS / Linux).
#
#   scripts/reset.sh                 # stop containers and DELETE the database + vector volumes
#   scripts/reset.sh --venv          # ... and delete the virtual environment
#   scripts/reset.sh --models        # ... and remove the Ollama models (ollama rm)
#   scripts/reset.sh --env           # ... and delete .env (new secrets on the next setup)
#   scripts/reset.sh --all           # all of the above
#   scripts/reset.sh --yes ...       # do not ask for confirmation
#   scripts/reset.sh --dry-run ...   # print what would be deleted, delete nothing
#
# Never touched: source code, data/guidelines (downloaded corpus), evaluation/results,
# reports/. After a reset: scripts/setup.sh, then scripts/run.sh (re-ingests the corpus).
# Windows users: scripts/reset.ps1.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
MODELS="${MODELS:-qwen3.5:4b gemma3:4b gemma3:12b}"
DO_VENV=0; DO_MODELS=0; DO_ENV=0; YES=0; DRY_RUN=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) DO_VENV=1 ;;
    --models) DO_MODELS=1 ;;
    --env) DO_ENV=1 ;;
    --all) DO_VENV=1; DO_MODELS=1; DO_ENV=1 ;;
    --yes|-y) YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $*"
  else
    echo "\$ $*"
    "$@"
  fi
}

echo "This will permanently delete:"
echo "  - the MedBridge containers and their volumes (ALL conversations, audit logs and the Qdrant index)"
[ "$DO_VENV" -eq 1 ] && echo "  - the virtual environment $VENV_DIR"
[ "$DO_MODELS" -eq 1 ] && echo "  - the Ollama models: $MODELS"
[ "$DO_ENV" -eq 1 ] && echo "  - .env (all generated secrets)"

if [ "$YES" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  printf 'Type "yes" to continue: '
  read -r answer
  [ "$answer" = "yes" ] || { echo "Cancelled."; exit 1; }
fi

if command -v docker >/dev/null 2>&1; then
  run docker compose down -v --remove-orphans
else
  echo "docker not found - skipping the containers."
fi

# Incremental-ingestion bookkeeping belongs to the deleted Qdrant volume.
if [ -f data/.ingest_state.json ]; then
  run rm -f data/.ingest_state.json
fi

if [ "$DO_VENV" -eq 1 ] && [ -d "$VENV_DIR" ]; then
  run rm -rf "$VENV_DIR"
fi

if [ "$DO_MODELS" -eq 1 ]; then
  if command -v ollama >/dev/null 2>&1; then
    for model in $MODELS; do
      run ollama rm "$model" || true
    done
  else
    echo "ollama not found - skipping the models."
  fi
fi

if [ "$DO_ENV" -eq 1 ] && [ -f .env ]; then
  run rm -f .env
fi

echo "Reset complete. Next: scripts/setup.sh, then scripts/run.sh"

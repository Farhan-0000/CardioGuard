#!/usr/bin/env bash
# Live robustness benchmarks against the production pipeline (resumable):
# MedBridge-AQ routing benchmark, RABBITS brand/generic robustness, adversarial suite.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv-mac/bin/python}"; [ -x "$PY" ] || PY="python"
export OLLAMA_BASE_URL="${OLLAMA_URL:-http://127.0.0.1:11435}"
if [ -z "${EVAL_DATABASE_URL:-}" ] && [ -f .env ]; then
  EVAL_DATABASE_URL="$(grep -E '^EVAL_DATABASE_URL=' .env | cut -d= -f2- || true)"
fi
DB="$EVAL_DATABASE_URL"
STEPS="${STEPS:-aq rabbits adversarial}"
log() { printf '\n[%s] === %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
for step in $STEPS; do
  case "$step" in
    aq) log "MedBridge-AQ live (all vignettes)"
        "$PY" -m tests.benchmark.benchmark_runner --mode live --all --resume --database-url "$DB" --quiet ;;
    rabbits) log "RABBITS live"
        "$PY" -m tests.adversarial.rabbits_runner --mode live --resume --database-url "$DB" --quiet ;;
    adversarial) log "Adversarial live"
        "$PY" -m tests.adversarial.adversarial_runner --database-url "$DB" ;;
  esac
done
log "live benchmarks complete"

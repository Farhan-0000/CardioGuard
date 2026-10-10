#!/usr/bin/env bash
# Runs the full MedBridge-HalluBench study end to end, resumably.
#
# Order (enforced by evaluation/run.py for V1/V2): C5 on dev -> judge dev ->
# calibrate on dev -> all conditions on test -> judge test -> analyze.
# Every step is resumable; re-running this script continues where it stopped.
#
# Usage:  scripts/run_study.sh            (all steps)
#         STEPS="analyze" scripts/run_study.sh
#
# Environment overrides: MODEL, JUDGE, OLLAMA_URL, WORKERS, EVAL_DATABASE_URL.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-mac/bin/python}"
[ -x "$PY" ] || PY="python"
MODEL="${MODEL:-qwen3.5:4b}"
JUDGE="${JUDGE:-gemma3:12b}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11435}"
WORKERS="${WORKERS:-4}"
CONDITIONS="${CONDITIONS:-C0,C1,C2,C3,C4,C5,V1,V2}"
STEPS="${STEPS:-dev_run dev_judge calibrate test_run test_retry test_judge analyze}"

if [ -z "${EVAL_DATABASE_URL:-}" ] && [ -f .env ]; then
  EVAL_DATABASE_URL="$(grep -E '^EVAL_DATABASE_URL=' .env | cut -d= -f2- || true)"
fi
export EVAL_DATABASE_URL
# Judge calls on a 12B model with parallel slots can exceed the app timeout.
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"

log() { printf '\n[%s] === %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

for step in $STEPS; do
  case "$step" in
    dev_run)
      log "C5 on dev ($MODEL)"
      "$PY" -m evaluation.run --conditions C5 --split dev --model "$MODEL" \
        --ollama-url "$OLLAMA_URL" --workers "$WORKERS" --resume ;;
    dev_judge)
      log "LLM + NLI judge on dev ($JUDGE)"
      LLM_TIMEOUT="$JUDGE_TIMEOUT" "$PY" -m evaluation.judge --judge llm --judge-model "$JUDGE" --conditions C5 \
        --split dev --workers "$WORKERS" --ollama-url "$OLLAMA_URL" --resume
      "$PY" -m evaluation.judge --judge nli --conditions C5 --split dev --resume ;;
    calibrate)
      log "Calibrate NLI judge, V1 and V2 thresholds on dev"
      "$PY" -m evaluation.calibrate --part all ;;
    test_run)
      log "All conditions on test ($CONDITIONS)"
      "$PY" -m evaluation.run --conditions "$CONDITIONS" --split test --model "$MODEL" \
        --ollama-url "$OLLAMA_URL" --workers "$WORKERS" --resume ;;
    test_retry)
      log "Retry degraded test turns"
      "$PY" -m evaluation.run --conditions "$CONDITIONS" --split test --model "$MODEL" \
        --ollama-url "$OLLAMA_URL" --workers 1 --retry-degraded || true ;;
    test_judge)
      log "LLM + NLI judge on test ($JUDGE)"
      LLM_TIMEOUT="$JUDGE_TIMEOUT" "$PY" -m evaluation.judge --judge llm --judge-model "$JUDGE" --conditions "$CONDITIONS" \
        --split test --workers "$WORKERS" --ollama-url "$OLLAMA_URL" --resume
      "$PY" -m evaluation.judge --judge nli --conditions "$CONDITIONS" --split test --resume ;;
    analyze)
      log "Analyze (test split)"
      "$PY" -m evaluation.analyze --split test ;;
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
log "study steps complete: $STEPS"

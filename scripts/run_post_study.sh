#!/usr/bin/env bash
# Everything that runs after the hallucination study, on a quiet GPU (resumable steps):
#   rabbits_after / adversarial_after  live robustness re-runs on the final code
#   perf_after     after-optimisation latency (cold + warm) with qwen3.5:4b and gemma3:4b
#   load           load test (1, 5, 10 concurrent users)
#   smoke          end-to-end smoke test
#   screenshots    UI screenshots for the report (desktop/mobile, light/dark)
#   annotation     50-item human annotation sheet
#   perf_table     before/after latency table + figure
#   aq_after       optional live AQ re-run on the final code
#
# Usage: scripts/run_post_study.sh   or   STEPS="perf_after load" scripts/run_post_study.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv-mac/bin/python}"; [ -x "$PY" ] || PY="python"
STEPS="${STEPS:-rabbits_after adversarial_after perf_after load smoke screenshots annotation perf_table}"
APP_OLLAMA="${APP_OLLAMA:-http://127.0.0.1:11434}"
PORT=8000
BASE="http://127.0.0.1:$PORT"
RABBITS_BRANDS_FILE="${RABBITS_BRANDS_FILE:-reports/rabbits_before_fix/brands.txt}"
if [ -z "${EVAL_DATABASE_URL:-}" ] && [ -f .env ]; then
  EVAL_DATABASE_URL="$(grep -E '^EVAL_DATABASE_URL=' .env | cut -d= -f2- || true)"
fi
export EVAL_DATABASE_URL
ADMIN_PASSWORD="$(grep -E '^ADMIN_PASSWORD=' .env | cut -d= -f2- || true)"
log() { printf '\n[%s] === %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

start_server() {  # $1 = model, extra env via the caller
  stop_server
  LLM_MODEL="$1" OLLAMA_BASE_URL="$APP_OLLAMA" RATE_LIMIT_PER_MINUTE="${RATE_LIMIT:-30}" \
    nohup "$PY" -m uvicorn medbridge.main:app --host 127.0.0.1 --port "$PORT" \
    > "evaluation/results/logs/uvicorn_post_$(date +%H%M%S).log" 2>&1 &
  for _ in $(seq 1 120); do
    curl -s "$BASE/health" | grep -q '"llm":true' && return 0
    sleep 3
  done
  echo "server did not become ready" >&2
}
stop_server() { pkill -f "uvicorn medbridge.main:app" 2>/dev/null; sleep 2; }
unload() { curl -s "$APP_OLLAMA/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null || true; }

for step in $STEPS; do
  case "$step" in
    rabbits_after)
      log "RABBITS live on the final code"
      rm -f reports/benchmark_runs/rabbits_live.jsonl
      OLLAMA_URL="$APP_OLLAMA" STEPS=rabbits RABBITS_BRANDS="$(cat "$RABBITS_BRANDS_FILE")" \
        scripts/run_live_benchmarks.sh ;;
    adversarial_after)
      log "Adversarial live on the final code"
      OLLAMA_URL="$APP_OLLAMA" STEPS=adversarial scripts/run_live_benchmarks.sh ;;
    aq_after)
      log "AQ live on the final code"
      mkdir -p reports/aq_before_fix
      cp -n reports/medbridge_aq_benchmark_report.* reports/aq_before_fix/ 2>/dev/null || true
      mv reports/benchmark_runs/aq_live_all.jsonl reports/aq_before_fix/ 2>/dev/null || true
      OLLAMA_URL="$APP_OLLAMA" STEPS=aq scripts/run_live_benchmarks.sh ;;
    perf_after)
      for model in qwen3.5:4b gemma3:4b; do
        log "Latency probe (after) with $model: cold start then 3 warm repeats"
        unload qwen3.5:4b; unload gemma3:4b
        WARMUP_ON_STARTUP=false start_server "$model"
        unload "$model"
        tag="$(echo "$model" | tr ':' '-')"
        "$PY" scripts/latency_probe.py --base-url "$BASE" --label "after_$tag" --repeats 3 \
          --out "reports/performance/after_${tag}.json"
        log "Latency probe (after, warm-up on) with $model: startup warm-up removes the cold start"
        unload "$model"
        WARMUP_ON_STARTUP=true start_server "$model"
        "$PY" scripts/latency_probe.py --base-url "$BASE" --label "after_warmup_$tag" \
          --cold-only --out "reports/performance/after_warmup_${tag}.json"
      done
      stop_server ;;
    load)
      log "Load test (1, 5, 10 users)"
      RATE_LIMIT=600 WARMUP_ON_STARTUP=true start_server qwen3.5:4b
      "$PY" scripts/load_test.py --base-url "$BASE" --users 1,5,10 --requests-per-user 3 \
        --output-dir reports/performance
      stop_server ;;
    smoke)
      log "Smoke test"
      RATE_LIMIT=120 WARMUP_ON_STARTUP=true start_server qwen3.5:4b
      "$PY" scripts/smoke_test.py --base-url "$BASE" --output reports/smoke_test_latest.json
      stop_server ;;
    screenshots)
      log "UI screenshots"
      RATE_LIMIT=120 WARMUP_ON_STARTUP=true start_server qwen3.5:4b
      for vp in desktop mobile; do for th in light dark; do
        "$PY" -m medbridge.web.screenshots --base-url "$BASE" --out "report/screenshots/raw/${vp}_${th}" \
          --admin-user admin --admin-password "$ADMIN_PASSWORD" --viewport "$vp" --theme "$th"
        sleep 65
      done; done
      stop_server ;;
    annotation)
      log "Human annotation sheet"
      "$PY" -m evaluation.human_annotation --export --split test ;;
    perf_table)
      log "Before/after latency table (same model, gemma3:4b, for an apples-to-apples comparison)"
      "$PY" -m evaluation.perf --before reports/performance/before_gemma3-4b.json \
        --after reports/performance/after_gemma3-4b.json ;;
    *) echo "unknown step $step" >&2 ;;
  esac
done
log "post-study steps complete: $STEPS"

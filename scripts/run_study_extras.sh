#!/usr/bin/env bash
# Extra study conditions run after the main study: C1G (production prompt, no gates: the
# clean anchor for gate-effect deltas) and V12 (V1 + V2 combined). Waits for the main
# study to finish, then runs, judges and re-analyses. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_study.sh" >/dev/null; do sleep 60; done
CONDITIONS="${EXTRA_CONDITIONS:-C1G,V12}" STEPS="test_run test_retry test_judge" scripts/run_study.sh
STEPS="analyze" scripts/run_study.sh

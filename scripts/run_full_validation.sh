#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results

LOG="results/full_validation.log"
: > "$LOG"

run_and_log() {
  local label="$1"
  shift
  echo "\n=== $label ===" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

source .venv/bin/activate

run_and_log "pytest (warnings as errors)" pytest -W error::UserWarning -q

run_and_log "multidoc smoke benchmark" python scripts/eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --baseline vanilla \
  --benchmark multidoc \
  --cache_budgets 1.0 \
  --n_samples 2 \
  --output results/validation_multidoc.csv

set +e
run_and_log "stage1 checkpoint compatibility check" python scripts/eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --baseline tis \
  --benchmark niah \
  --checkpoint checkpoints/stage1 \
  --skip-lora \
  --cache_budgets 1.0 \
  --n_samples 1 \
  --output results/validation_stage1_mismatch.csv
status=$?
set -e

echo "\n[summary] stage1 compatibility exit code: $status" | tee -a "$LOG"

echo "\nValidation log written to $LOG" | tee -a "$LOG"
exit 0

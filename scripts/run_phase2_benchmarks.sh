#!/usr/bin/env bash
# scripts/run_phase2_benchmarks.sh
# Runs all Phase 2 benchmark jobs sequentially and produces the summary table.
# Usage: bash scripts/run_phase2_benchmarks.sh
set -e

VENV="/mnt/juegos/proyectos/especiales/token-importance/.venv"
PYTHON="$VENV/bin/python"
SCRIPT="scripts/eval.py"
MODEL="mistralai/Mistral-7B-v0.3"
RESULTS="${RESULTS:-results}"
mkdir -p "$RESULTS"

TARGET="${TARGET:-a100}"
if [ "$TARGET" = "local" ]; then
    N_SAMPLES="${N_SAMPLES:-4}"
    CTX_LENGTHS="${CTX_LENGTHS:-512 1024}"
    DEPTHS="${DEPTHS:-0.25 0.5}"
    echo "=== Local RTX 5070 benchmark profile ==="
else
    N_SAMPLES="${N_SAMPLES:-20}"
    CTX_LENGTHS="${CTX_LENGTHS:-1024 2048}"
    DEPTHS="${DEPTHS:-0.25 0.5 0.75}"
    echo "=== A100 benchmark profile (baseline) ==="
fi
CTX="--context_lengths $CTX_LENGTHS"
DEPTHS_ARG="--depths $DEPTHS"
N="--n_samples $N_SAMPLES"
BUDGETS="0.25 0.5 0.75 1.0"

echo "=== Phase 2 Benchmarks: $MODEL ===" | tee "$RESULTS/run.log"
echo "Profile: ${TARGET:-a100} | $CTX $DEPTHS_ARG $N" | tee -a "$RESULTS/run.log"
date | tee -a "$RESULTS/run.log"

for BASELINE in vanilla streamingllm h2o tis; do
    echo "" | tee -a "$RESULTS/run.log"
    echo "--- Baseline: $BASELINE ---" | tee -a "$RESULTS/run.log"

    # NIAH
    echo "[$(date +%H:%M:%S)] niah $BASELINE" | tee -a "$RESULTS/run.log"
    $PYTHON $SCRIPT \
        --model "$MODEL" \
        --load_in_4bit \
        --baseline "$BASELINE" \
        --benchmark niah \
        --cache_budgets $BUDGETS \
        $CTX $DEPTHS_ARG $N \
        --output "$RESULTS/niah_${BASELINE}.csv" \
        2>> "$RESULTS/run.log"

    # LITM
    echo "[$(date +%H:%M:%S)] litm $BASELINE" | tee -a "$RESULTS/run.log"
    $PYTHON $SCRIPT \
        --model "$MODEL" \
        --load_in_4bit \
        --baseline "$BASELINE" \
        --benchmark litm \
        --cache_budgets $BUDGETS \
        $N \
        --output "$RESULTS/litm_${BASELINE}.csv" \
        2>> "$RESULTS/run.log"

    # MultiDoc QA
    echo "[$(date +%H:%M:%S)] multidoc $BASELINE" | tee -a "$RESULTS/run.log"
    $PYTHON $SCRIPT \
        --model "$MODEL" \
        --load_in_4bit \
        --baseline "$BASELINE" \
        --benchmark multidoc \
        --cache_budgets $BUDGETS \
        $N \
        --output "$RESULTS/multidoc_${BASELINE}.csv" \
        2>> "$RESULTS/run.log"
done

echo "" | tee -a "$RESULTS/run.log"
echo "=== All runs complete ===" | tee -a "$RESULTS/run.log"
date | tee -a "$RESULTS/run.log"

# Build summary table
$PYTHON scripts/build_results_table.py

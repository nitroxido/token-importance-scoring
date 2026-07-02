#!/bin/bash
# Run Phase 2 benchmarks for all 6 baselines × 2 benchmarks × 4 cache budgets.
# Usage: bash scripts/run_benchmarks.sh
#
# Baselines:
#   vanilla         — no cache compression (upper bound)
#   streamingllm    — first N sinks + recent window (Xiao et al. 2023)
#   h2o             — heavy-hitter oracle, cumulative attention (Zhang et al. 2023)
#   snapkv          — query-pooled attention per head (Li et al. 2024)
#   infini_attention — compressive memory approx (Munkhdalai et al. 2024)
#   tis             — Token Importance Scoring, oracle labels (ours)
#
# Note: snapkv and infini_attention require output_attentions=True forward pass
# (one extra pass per sample). Add ~40% time vs H2O for those two baselines.

set -euo pipefail

VENV=/mnt/juegos/proyectos/especiales/token-importance/.venv
PYTHON=$VENV/bin/python
MODEL="mistralai/Mistral-7B-v0.3"
BUDGETS="0.25 0.5 0.75 1.0"
TARGET="${TARGET:-a100}"

if [ "$TARGET" = "local" ]; then
    N_SAMPLES="${N_SAMPLES:-4}"
    CTX_LENGTHS="${CTX_LENGTHS:-512 1024}"
    DEPTHS="${DEPTHS:-0.25 0.5}"
    BASELINES=(vanilla streamingllm h2o tis)
    echo "=== Local RTX 5070 Benchmark Profile ==="
    echo "Using reduced settings for 8 GB VRAM: n_samples=$N_SAMPLES, context_lengths=$CTX_LENGTHS, depths=$DEPTHS"
else
    N_SAMPLES="${N_SAMPLES:-20}"
    CTX_LENGTHS="${CTX_LENGTHS:-1024 2048}"
    DEPTHS="${DEPTHS:-0.25 0.5 0.75}"
    BASELINES=(vanilla streamingllm h2o snapkv infini_attention tis)
    echo "=== A100 Benchmark Profile (baseline) ==="
    echo "Using A100-style settings: n_samples=$N_SAMPLES, context_lengths=$CTX_LENGTHS, depths=$DEPTHS"
fi

mkdir -p results

echo "Model: $MODEL (4-bit NF4)"
date

for BASELINE in "${BASELINES[@]}"; do
    echo ""
    echo "--- Baseline: $BASELINE ---"

    $PYTHON scripts/eval.py \
        --model "$MODEL" \
        --load_in_4bit \
        --baseline "$BASELINE" \
        --benchmark niah \
        --cache_budgets $BUDGETS \
        --context_lengths $CTX_LENGTHS \
        --depths $DEPTHS \
        --n_samples $N_SAMPLES \
        --output "results/niah_${BASELINE}.csv"

    $PYTHON scripts/eval.py \
        --model "$MODEL" \
        --load_in_4bit \
        --baseline "$BASELINE" \
        --benchmark litm \
        --cache_budgets $BUDGETS \
        --n_samples $N_SAMPLES \
        --output "results/litm_${BASELINE}.csv"
done

echo ""
echo "=== All benchmark runs complete ==="
date

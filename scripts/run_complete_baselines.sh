#!/bin/bash

# ============================================================================
# WEEK 1: COMPLETE BASELINE TESTING ORCHESTRATOR
# 
# Purpose: Run all 6 baseline methods across 3 benchmarks with full coverage
# Duration: ~40 GPU hours (5 days on RTX 5070)
# Output: 18 CSV files (6 baselines × 3 benchmarks)
# ============================================================================

set -e  # Exit on error

# Configuration
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/eval.py"
RESULTS="${REPO_ROOT}/results/week1_comprehensive"
LOGS="${REPO_ROOT}/logs"
MODEL="mistralai/Mistral-7B-v0.3"

# Create directories
mkdir -p "$RESULTS" "$LOGS"

# Configuration: All baselines, all benchmarks
BASELINES=(
    "vanilla"
    "streamingllm"
    "h2o"
    "snapkv"
    "infini_attention"
    "tis"
)

BENCHMARKS=(
    "niah"
    "litm"
    "multidoc"
)

CACHE_BUDGETS="0.25 0.5 0.75 1.0"
N_SAMPLES=32  # Full sampling (32 samples per budget level)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_section() {
    echo ""
    echo "========================================="
    echo -e "${BLUE}$1${NC}"
    echo "========================================="
}

log_step() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

get_timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log_section "WEEK 1: COMPREHENSIVE BASELINE TESTING"
echo "Start time: $(get_timestamp)"
echo "Model: $MODEL"
echo "Baselines: ${#BASELINES[@]} (vanilla, StreamingLLM, H2O, SnapKV, Infini-Attention, TIS)"
echo "Benchmarks: ${#BENCHMARKS[@]} (NIAH, LITM, MultiDoc)"
echo "Cache budgets: $CACHE_BUDGETS"
echo "Samples per budget: $N_SAMPLES"
echo "Total runs: $((${#BASELINES[@]} * ${#BENCHMARKS[@]}))"
echo ""

# Pre-flight check
log_section "PRE-FLIGHT CHECKS"

if [ ! -f "$SCRIPT" ]; then
    log_warning "Evaluation script not found: $SCRIPT"
    exit 1
fi

if ! $PYTHON --version >/dev/null 2>&1; then
    log_warning "Python environment not ready"
    exit 1
fi

log_step "Python environment ready"
log_step "Directories created"

# ============================================================================
# MAIN BENCHMARK LOOP
# ============================================================================

TOTAL_RUNS=$((${#BASELINES[@]} * ${#BENCHMARKS[@]}))
CURRENT_RUN=0

for BASELINE in "${BASELINES[@]}"; do
    log_section "BASELINE: $BASELINE"
    
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        OUTPUT_FILE="${RESULTS}/${BENCHMARK}_${BASELINE}.csv"
        
        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running $BENCHMARK with $BASELINE..."
        echo "Output: $OUTPUT_FILE"
        
        # IDEMPOTENT: Skip if already completed
        if [ -f "$OUTPUT_FILE" ]; then
            log_step "SKIPPED (already completed)"
            echo "  Existing CSV rows: $(wc -l < "$OUTPUT_FILE")"
            continue
        fi
        
        echo "Start: $(get_timestamp)"
        
        # Build eval command with optional checkpoint for TIS baseline
        EVAL_CMD="$PYTHON $SCRIPT --model $MODEL --load_in_4bit --baseline $BASELINE --benchmark $BENCHMARK --cache_budgets $CACHE_BUDGETS --n_samples $N_SAMPLES"
        
        # Add checkpoint for TIS baseline
        if [ "$BASELINE" = "tis" ]; then
            EVAL_CMD="$EVAL_CMD --checkpoint $RESULTS/../stage3_ert_local_fresh"
        fi
        
        EVAL_CMD="$EVAL_CMD --output $OUTPUT_FILE"
        
        # Run evaluation
        if eval "$EVAL_CMD" 2>&1 | tee "${LOGS}/${BENCHMARK}_${BASELINE}.log"
        then
            log_step "$BENCHMARK with $BASELINE completed"
            
            # Show result summary
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  CSV rows: $(wc -l < "$OUTPUT_FILE")"
            fi
        else
            log_warning "FAILED: $BENCHMARK with $BASELINE (see ${LOGS}/${BENCHMARK}_${BASELINE}.log)"
            # Don't exit, continue with next baseline
        fi
        
        echo "End: $(get_timestamp)"
    done
done

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

log_section "WEEK 1 SUMMARY"

CSV_COUNT=$(find "$RESULTS" -name "*.csv" -type f | wc -l)
echo "Total CSV files generated: $CSV_COUNT / 18"

echo ""
echo "Generated files:"
ls -lh "$RESULTS"/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Complete! End time: $(get_timestamp)"

# ============================================================================
# NEXT STEPS
# ============================================================================

log_section "NEXT STEPS"

if [ "$CSV_COUNT" -ge 18 ]; then
    echo "✅ Week 1 COMPLETE!"
    echo ""
    echo "1. Generate comparison tables:"
    echo "   $ python ${REPO_ROOT}/scripts/generate_comparison_tables.py \\"
    echo "       --results_dir $RESULTS \\"
    echo "       --output ${RESULTS}/COMPREHENSIVE-BASELINE-COMPARISON-V4.md"
    echo ""
    echo "2. Start Week 2 (Attention Drift Analysis):"
    echo "   $ cat ${REPO_ROOT}/WEEK2-ATTENTION-DRIFT-GUIDE.md | less"
elif [ "$CSV_COUNT" -gt 0 ]; then
    echo "⚠️  Week 1 PARTIAL ($CSV_COUNT/18 files)"
    echo ""
    echo "Missing results - check logs:"
    echo "   $ ls -lh ${LOGS}/*.log | grep -E 'FAILED|Error'"
else
    echo "❌ No results generated. Check environment setup."
fi

echo ""
echo "========================================="

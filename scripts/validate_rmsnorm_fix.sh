#!/bin/bash

# ============================================================================
# Week 2: Validate RMSNorm Drift Fix
#
# Runs comprehensive validation of Stage 3 with RMSNorm against all baselines
# to measure improvement on LITM (main target) and verify NIAH stability.
# ============================================================================

set -e

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/eval.py"
RESULTS="${REPO_ROOT}/results/week2_rmsnorm_validation"
LOGS="${REPO_ROOT}/logs"
MODEL="mistralai/Mistral-7B-v0.3"
CHECKPOINT="${REPO_ROOT}/checkpoints/stage3_rmsnorm"

# Create directories
mkdir -p "$RESULTS" "$LOGS"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo -e "${BLUE}WEEK 2: RMSNORM DRIFT FIX VALIDATION${NC}"
echo "========================================="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CHECKPOINT"
echo "Results: $RESULTS"
echo ""

# Check checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo -e "${YELLOW}⚠ Checkpoint not found: $CHECKPOINT${NC}"
    echo "Please run training first:"
    echo "  python scripts/train_stage3_with_rmsnorm.py \\"
    echo "    --base-checkpoint checkpoints/stage3_ert_local_fresh/ \\"
    echo "    --output-dir $CHECKPOINT"
    exit 1
fi

# Validation benchmarks (focused on LITM to see drift fix)
echo -e "${GREEN}Running LITM validation (main target)...${NC}"
echo ""

# LITM @ 50% (highest priority - was 48.3%, target 50%+)
echo "[1/3] LITM @ 50% (critical - drift fix target)..."
$PYTHON $SCRIPT \
    --model $MODEL \
    --load_in_4bit \
    --baseline tis \
    --benchmark litm \
    --cache_budgets 0.5 \
    --n_samples 32 \
    --checkpoint "$CHECKPOINT" \
    --output "$RESULTS/litm_50.csv" \
    2>&1 | tee "${LOGS}/week2_litm_50.log"

echo ""

# LITM @ 75% (secondary target - was 66.7%, target 72%+)
echo "[2/3] LITM @ 75% (drift compounding check)..."
$PYTHON $SCRIPT \
    --model $MODEL \
    --load_in_4bit \
    --baseline tis \
    --benchmark litm \
    --cache_budgets 0.75 \
    --n_samples 32 \
    --checkpoint "$CHECKPOINT" \
    --output "$RESULTS/litm_75.csv" \
    2>&1 | tee "${LOGS}/week2_litm_75.log"

echo ""

# NIAH @ 50% (regression check - must still be 100%)
echo "[3/3] NIAH @ 50% (regression check - must stay 100%)..."
$PYTHON $SCRIPT \
    --model $MODEL \
    --load_in_4bit \
    --baseline tis \
    --benchmark niah \
    --cache_budgets 0.5 \
    --n_samples 32 \
    --checkpoint "$CHECKPOINT" \
    --output "$RESULTS/niah_50.csv" \
    2>&1 | tee "${LOGS}/week2_niah_50.log"

echo ""
echo "========================================="
echo -e "${BLUE}VALIDATION RESULTS${NC}"
echo "========================================="

# Parse results and compare with Week 1
if [ -f "$RESULTS/litm_50.csv" ]; then
    echo -e "${GREEN}✓ LITM @ 50%${NC}"
    grep "| 0.5 " "$RESULTS/litm_50.csv" | grep accuracy | head -1
    echo "  Previous (Week 1): 48.3%"
    echo "  Target: 50%+"
    echo ""
fi

if [ -f "$RESULTS/litm_75.csv" ]; then
    echo -e "${GREEN}✓ LITM @ 75%${NC}"
    grep "| 0.75 " "$RESULTS/litm_75.csv" | grep accuracy | head -1
    echo "  Previous (Week 1): 66.7%"
    echo "  Target: 72%+"
    echo ""
fi

if [ -f "$RESULTS/niah_50.csv" ]; then
    echo -e "${GREEN}✓ NIAH @ 50%${NC}"
    grep "| 0.5 " "$RESULTS/niah_50.csv" | grep accuracy | head -1
    echo "  Previous (Week 1): 100.0%"
    echo "  Must maintain: 100%+"
    echo ""
fi

echo "========================================="
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

# Success criteria
SUCCESS=true
if [ -f "$RESULTS/litm_50.csv" ]; then
    LITM50=$(grep "| 0.5 " "$RESULTS/litm_50.csv" | grep "accuracy" | grep -v breakdown | awk -F'|' '{print $NF}' | head -1 | xargs)
    if (( $(echo "$LITM50 < 0.50" | bc -l) )); then
        echo -e "${YELLOW}⚠ LITM @ 50% is $LITM50 (target 0.50+)${NC}"
        SUCCESS=false
    fi
fi

if [ -f "$RESULTS/niah_50.csv" ]; then
    NIAH50=$(grep "| 0.5 " "$RESULTS/niah_50.csv" | grep "accuracy" | grep -v breakdown | awk -F'|' '{print $NF}' | head -1 | xargs)
    if (( $(echo "$NIAH50 < 0.99" | bc -l) )); then
        echo -e "${YELLOW}⚠ NIAH @ 50% is $NIAH50 (expected 1.0)${NC}"
        SUCCESS=false
    fi
fi

if [ "$SUCCESS" = true ]; then
    echo -e "${GREEN}✅ Week 2 validation PASSED${NC}"
    echo ""
    echo "Next: Week 3 - Phase 4 Query-Aware Importance Implementation"
else
    echo -e "${YELLOW}⚠ Week 2 validation needs review${NC}"
    echo "Check logs in $LOGS/ for details"
fi

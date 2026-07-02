#!/bin/bash

# ============================================================================
# CHECK MISSING BASELINE RESULTS
#
# Purpose: Show which baseline CSV files are missing and allow re-running
# Usage: bash scripts/check_missing_baselines.sh [--rerun]
# ============================================================================

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS="${REPO_ROOT}/results/week1_comprehensive"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Expected files
BASELINES=("vanilla" "streamingllm" "h2o" "snapkv" "infini_attention" "tis")
BENCHMARKS=("niah" "litm" "multidoc")

echo "========================================="
echo -e "${BLUE}WEEK 1 BASELINE STATUS CHECK${NC}"
echo "========================================="
echo ""

# Check status
MISSING=()
COMPLETE=0
TOTAL=$((${#BASELINES[@]} * ${#BENCHMARKS[@]}))

for BENCHMARK in "${BENCHMARKS[@]}"; do
    for BASELINE in "${BASELINES[@]}"; do
        FILE="${RESULTS}/${BENCHMARK}_${BASELINE}.csv"
        
        if [ -f "$FILE" ]; then
            echo -e "${GREEN}✓${NC} $BENCHMARK × $BASELINE"
            ((COMPLETE++))
        else
            echo -e "${RED}✗${NC} $BENCHMARK × $BASELINE (MISSING)"
            MISSING+=("${BENCHMARK}_${BASELINE}")
        fi
    done
done

echo ""
echo "========================================="
echo "Progress: $COMPLETE / $TOTAL files"
echo "========================================="
echo ""

if [ ${#MISSING[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ ALL 18 BASELINE RESULTS COMPLETE${NC}"
    echo ""
    echo "Next step: Generate comparison tables"
    echo "$ python scripts/generate_comparison_tables.py --results_dir $RESULTS"
else
    echo -e "${YELLOW}⚠ Missing ${#MISSING[@]} files:${NC}"
    for FILE in "${MISSING[@]}"; do
        echo "  - $FILE.csv"
    done
    echo ""
    echo "To re-run ONLY missing tests:"
    echo "$ bash scripts/run_complete_baselines.sh"
    echo ""
    echo "The script will skip already-completed tests automatically."
fi

echo ""

# Show space usage
TOTAL_SIZE=$(du -sh "$RESULTS" 2>/dev/null | awk '{print $1}')
FILE_COUNT=$(find "$RESULTS" -name "*.csv" -type f 2>/dev/null | wc -l)
echo "Storage:"
echo "  Files: $FILE_COUNT CSV"
echo "  Space: $TOTAL_SIZE"

#!/bin/bash

# ============================================================================
# WEEK 1: BASELINE TESTING MONITOR
#
# Purpose: Real-time dashboard of Week 1 baseline testing progress
# Usage: bash scripts/monitor_baselines.sh
# ============================================================================

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS="${REPO_ROOT}/results/week1_comprehensive"
LOGS="${REPO_ROOT}/logs"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear

while true; do
    clear
    
    echo "========================================="
    echo -e "${BLUE}WEEK 1 BASELINE TESTING MONITOR${NC}"
    echo "========================================="
    echo "Updated: $(date '+%H:%M:%S')"
    echo ""
    
    # Count completed files
    CSV_COUNT=$(find "$RESULTS" -name "*.csv" -type f 2>/dev/null | wc -l)
    echo -e "${GREEN}Progress: $CSV_COUNT / 18 CSV files${NC}"
    echo ""
    
    # Show completed baselines
    echo "Completed tests:"
    for baseline in vanilla streamingllm h2o snapkv infini_attention tis; do
        count=$(find "$RESULTS" -name "*_${baseline}.csv" -type f 2>/dev/null | wc -l)
        if [ $count -eq 3 ]; then
            echo -e "  ${GREEN}✓${NC} $baseline (3/3 benchmarks)"
        elif [ $count -gt 0 ]; then
            echo -e "  ${YELLOW}⚠${NC} $baseline ($count/3 benchmarks)"
        else
            echo -e "  ${RED}○${NC} $baseline (0/3 benchmarks)"
        fi
    done
    
    echo ""
    echo "Recent activity:"
    
    # Show recently modified files
    if [ -d "$RESULTS" ]; then
        ls -lt "$RESULTS"/*.csv 2>/dev/null | head -5 | awk '{print "  " $9 " (" $6 " " $7 " " $8 ")"}'
    fi
    
    echo ""
    echo "Active processes:"
    ps aux | grep -E "eval.py|python" | grep -v grep | wc -l | awk '{print "  $1 running processes"}'
    
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        awk '{printf "  Memory: %d MB / %d MB (%.1f%%)\n", $1, $2, ($1/$2)*100}' || echo "  CUDA not available"
    
    echo ""
    echo "Space usage:"
    du -sh "$RESULTS" 2>/dev/null | awk '{print "  Results dir: $1"}'
    
    echo ""
    echo "Recent errors:"
    if [ -d "$LOGS" ]; then
        grep -h "Error\|error\|ERROR\|failed\|FAILED" "$LOGS"/*.log 2>/dev/null | tail -3 | sed 's/^/  /' || echo "  None detected"
    fi
    
    echo ""
    echo -e "${BLUE}Press Ctrl+C to exit | Refreshing every 60 seconds...${NC}"
    
    sleep 60
done

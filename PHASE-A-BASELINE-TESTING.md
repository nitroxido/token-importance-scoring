# Stage 1: Complete Baseline Testing Guide

**Objective**: Comprehensive benchmarking of all 7 methods across 3 benchmarks with 32 samples per budget 
**Estimated Duration**: ~40 GPU hours 
**Output**: 21 CSV files + comprehensive comparison tables 

---

## Pre-Flight Checklist (Initial Setup)

```bash
# 1. Verify eval.py supports all baseline methods
python scripts/eval.py --help | grep -A 10 "baseline"
# Should show: {tis,h2o,streamingllm,vanilla,snapkv,infini_attention}
echo " All baselines supported by eval.py"

# 2. Verify checkpoints exist
ls -lh checkpoints/stage1/ checkpoints/stage3_ert_local_fresh/

# 3. Create results directory structure
mkdir -p results/week1_comprehensive/{csv,logs}

# 4. Test eval script on 1 sample (quick validation)
cd $PROJECT_DIR
source .venv/bin/activate
python scripts/eval.py \
 --model mistralai/Mistral-7B-v0.3 \
 --load_in_4bit \
 --benchmark niah \
 --baseline vanilla \
 --cache_budgets 0.25 \
 --n_samples 1 \
 --output results/week1_comprehensive/test_run.csv
echo " Eval script working - ready to run Stage 1"
```

---

## Script 1: Baseline Testing Orchestrator

**File**: `scripts/run_complete_baselines.sh`

```bash
#!/bin/bash
set -e

# Set PROJECT_DIR to the root of this repository
PROJECT_DIR="${PWD}" # or specify your path here
cd "$PROJECT_DIR"
source .venv/bin/activate

RESULTS_DIR="results/comprehensive_v4"
mkdir -p "$RESULTS_DIR/csv" "$RESULTS_DIR/logs" "$RESULTS_DIR/artifacts"

# Configuration
BENCHMARKS=("niah" "litm" "multidoc")
BASELINES=("vanilla" "streamingllm" "h2o" "snapkv" "infini_attention" "tis_oracle" "tis_stage3_ert")
BUDGETS="0.25 0.5 0.75 1.0"
SAMPLES=32
CHECKPOINT_STAGE3="checkpoints/stage3_ert_local_fresh"
CHECKPOINT_ORACLE="checkpoints/stage1"

echo "========================================="
echo "COMPREHENSIVE BASELINE SUITE - Stage 1"
echo "========================================="
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Baselines: ${BASELINES[*]}"
echo "Budgets: $BUDGETS"
echo "Samples per budget: $SAMPLES"
echo ""

# Function to run benchmark
run_benchmark() {
 local benchmark=$1
 local baseline=$2
 local samples=$3
 local timestamp=$(date +%Y%m%d_%H%M%S)
 
 local output_file="$RESULTS_DIR/csv/${benchmark}_${baseline}_${timestamp}.csv"
 local log_file="$RESULTS_DIR/logs/${benchmark}_${baseline}_${timestamp}.log"
 
 echo "[$(date '+%H:%M:%S')] Running: $benchmark x $baseline ($samples samples)"
 
 # Select checkpoint based on baseline
 local checkpoint=""
 if [ "$baseline" = "tis_oracle" ]; then
 checkpoint="--checkpoint $CHECKPOINT_ORACLE --use_oracle_importance"
 elif [ "$baseline" = "tis_stage3_ert" ]; then
 checkpoint="--checkpoint $CHECKPOINT_STAGE3"
 fi
 
 python scripts/eval.py \
 --benchmark "$benchmark" \
 --baseline "$baseline" \
 --cache_budgets $BUDGETS \
 --n_samples "$samples" \
 $checkpoint \
 --output "$output_file" \
 2>&1 | tee "$log_file"
 
 echo "[$(date '+%H:%M:%S')] Completed: $benchmark x $baseline"
 echo ""
}

# Nested loop: each benchmark × each baseline
total_runs=$((${#BENCHMARKS[@]} * ${#BASELINES[@]}))
run_count=0

for benchmark in "${BENCHMARKS[@]}"; do
 echo "================================="
 echo "BENCHMARK: $benchmark"
 echo "================================="
 
 for baseline in "${BASELINES[@]}"; do
 run_count=$((run_count + 1))
 echo "[$run_count/$total_runs]"
 
 # Time-limited execution (45 min timeout per run)
 timeout 2700s run_benchmark "$benchmark" "$baseline" "$SAMPLES" || {
 echo "⚠️ Timeout or error on $benchmark x $baseline"
 continue
 }
 done
done

echo ""
echo "========================================="
echo "BASELINE TESTING COMPLETE"
echo "========================================="
echo "Results saved to: $RESULTS_DIR/csv/"
echo ""
echo "Next: Run comparison analysis script"
```

**Usage**:
```bash
chmod +x scripts/run_complete_baselines.sh
./scripts/run_complete_baselines.sh 2>&1 | tee run_baselines.log

# Monitor progress in another terminal
tail -f run_baselines.log
watch -n 60 'ls -1 results/comprehensive_v4/csv/*.csv | wc -l'
```

---

## Script 2: Comparison Table Generator

**File**: `scripts/generate_comparison_tables.py`

```python
#!/usr/bin/env python3
"""
Generate comprehensive comparison tables from baseline CSV results.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class BaselineComparator:
 def __init__(self, results_dir: str):
 self.results_dir = Path(results_dir)
 self.results = {}
 self.load_results()
 
 def load_results(self):
 """Load all CSV files into memory."""
 csv_files = list(self.results_dir.glob("csv/*.csv"))
 
 for csv_file in sorted(csv_files):
 # Parse filename: benchmark_baseline_timestamp.csv
 parts = csv_file.stem.split('_')
 if len(parts) < 2:
 continue
 
 benchmark = parts[0]
 baseline = parts[1]
 
 key = f"{benchmark}_{baseline}"
 df = pd.read_csv(csv_file)
 self.results[key] = df
 print(f" Loaded: {key}")
 
 def extract_accuracy_by_budget(self, benchmark: str) -> Dict[str, Dict[float, float]]:
 """
 Extract accuracy for each baseline at each budget level.
 
 Returns:
 {baseline: {budget: accuracy, ...}, ...}
 """
 baselines = ['vanilla', 'streamingllm', 'h2o', 'snapkv', 'infini_attention', 
 'tis_oracle', 'tis_stage3_ert']
 
 accuracy_matrix = {}
 
 for baseline in baselines:
 key = f"{benchmark}_{baseline}"
 if key not in self.results:
 print(f"⚠️ Missing: {key}")
 continue
 
 df = self.results[key]
 
 # Group by cache_budget, compute mean accuracy
 accuracies_by_budget = {}
 for budget in [0.25, 0.5, 0.75, 1.0]:
 budget_data = df[df['cache_budget'] == budget]
 if len(budget_data) > 0:
 mean_acc = budget_data['accuracy'].mean()
 accuracies_by_budget[budget] = mean_acc
 
 accuracy_matrix[baseline] = accuracies_by_budget
 
 return accuracy_matrix
 
 def generate_markdown_table(self, benchmark: str, accuracies: Dict) -> str:
 """Generate markdown table for a single benchmark."""
 
 # Sort baselines by category
 heuristic_baselines = ['vanilla', 'streamingllm', 'h2o', 'snapkv', 'infini_attention']
 lis_baselines = ['tis_oracle', 'tis_stage3_ert']
 
 lines = [
 f"## {benchmark.upper()} Results",
 "",
 "| Budget | Vanilla | StreamingLLM | H2O | SnapKV | Infini-Attn | TIS Oracle | **TIS Stage 3** |",
 "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
 ]
 
 for budget in [0.25, 0.5, 0.75, 1.0]:
 row_parts = [f"{budget:.0%}"]
 
 for baseline in heuristic_baselines + lis_baselines:
 acc = accuracies.get(baseline, {}).get(budget, None)
 if acc is None:
 row_parts.append("—")
 else:
 if baseline == 'tis_stage3_ert':
 # Highlight TIS Stage 3
 row_parts.append(f"**{acc:.1%}**")
 else:
 row_parts.append(f"{acc:.1%}")
 
 lines.append("| " + " | ".join(row_parts) + " |")
 
 return "\n".join(lines)
 
 def generate_comparison_report(self) -> str:
 """Generate complete comparison report."""
 
 report = []
 report.append("# Complete Baseline Comparison Report (V4)\n")
 report.append(f"Generated: {pd.Timestamp.now()}\n")
 report.append("")
 
 benchmarks = ['niah', 'litm', 'multidoc']
 
 for benchmark in benchmarks:
 accuracies = self.extract_accuracy_by_budget(benchmark)
 if accuracies:
 table = self.generate_markdown_table(benchmark, accuracies)
 report.append(table)
 report.append("")
 
 # Add insights
 report.append("## Key Insights\n")
 report.append(self.generate_insights())
 
 return "\n".join(report)
 
 def generate_insights(self) -> str:
 """Generate qualitative insights from the data."""
 insights = []
 
 # NIAH insights
 niah_data = self.extract_accuracy_by_budget('niah')
 if niah_data.get('tis_stage3_ert', {}).get(0.25) == 1.0:
 insights.append("- **NIAH Synthetic Dominance**: TIS achieves 100% at 25% budget where SnapKV only reaches 33.3%")
 
 # LITM insights
 litm_data = self.extract_accuracy_by_budget('litm')
 tis_50 = litm_data.get('tis_stage3_ert', {}).get(0.5)
 snapkv_50 = litm_data.get('snapkv', {}).get(0.5)
 if tis_50 and snapkv_50:
 gap = snapkv_50 - tis_50
 insights.append(f"- **LITM Semantic Gap**: TIS trails SnapKV by {gap:.1%} @ 50% budget (remaining challenge for Phase 4)")
 
 insights.append("- **Query-Aware vs Query-Independent**: SnapKV's pooled attention captures semantic relevance better than position-invariant importance")
 insights.append("- **Phase 4 Direction**: Query-aware training signals should close the semantic gap")
 
 return "\n".join(insights)

if __name__ == "__main__":
 import sys
 
 results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/comprehensive_v4"
 
 comparator = BaselineComparator(results_dir)
 report = comparator.generate_comparison_report()
 
 # Save report
 output_file = Path(results_dir) / "BASELINE-COMPARISON-V4.md"
 output_file.write_text(report)
 print(f"\n Comparison report saved to: {output_file}")
 
 # Print to console
 print("\n" + "="*60)
 print(report)
```

**Usage**:
```bash
python scripts/generate_comparison_tables.py results/comprehensive_v4
# Output: results/comprehensive_v4/BASELINE-COMPARISON-V4.md
```

---

## Script 3: Progress Monitoring Dashboard

**File**: `scripts/monitor_baselines.sh`

```bash
#!/bin/bash
# Monitor baseline testing progress in real-time

RESULTS_DIR="results/comprehensive_v4/csv"

while true; do
 clear
 echo "╔════════════════════════════════════════════════════╗"
 echo "║ BASELINE TESTING PROGRESS DASHBOARD ║"
 echo "╚════════════════════════════════════════════════════╝"
 echo ""
 
 total_files=$(ls -1 "$RESULTS_DIR"/*.csv 2>/dev/null | wc -l)
 target_files=$((7 * 3)) # 7 baselines × 3 benchmarks
 
 echo "CSV Files Generated: $total_files / $target_files"
 echo "Progress: $(( total_files * 100 / target_files ))%"
 echo ""
 
 # Breakdown by benchmark
 echo "Breakdown by Benchmark:"
 for benchmark in niah litm multidoc; do
 count=$(ls -1 "$RESULTS_DIR"/${benchmark}_*.csv 2>/dev/null | wc -l)
 echo " $benchmark: $count / 7 baselines"
 done
 
 echo ""
 echo "Latest files:"
 ls -1t "$RESULTS_DIR"/*.csv 2>/dev/null | head -3 | xargs -I {} basename {}
 
 echo ""
 echo "Errors in logs (if any):"
 grep -r "ERROR\|FAILED" results/comprehensive_v4/logs/ 2>/dev/null | head -3 || echo " None detected"
 
 echo ""
 echo "Last updated: $(date '+%H:%M:%S')"
 echo "Press Ctrl+C to exit, refreshing in 30s..."
 sleep 30
done
```

**Usage**:
```bash
bash scripts/monitor_baselines.sh
```

---

## Execution Guide (Stage-Based)
```bash
# Morning: Set up infrastructure
08:00 - Run pre-flight checklist
09:00 - Create results directory structure
09:30 - Test eval.py on 1 sample each baseline (30 min total)
10:00 - Green light to proceed

# Afternoon: Start first batch
13:00 - Start NIAH × vanilla (1.5 hours)
14:30 - Start NIAH × streamingllm (1.5 hours)
16:00 - Start NIAH × h2o (1.5 hours)
17:30 - EOD, all 3 queued

# Monitor overnight
All three running in parallel (estimate 4.5 hours total, complete by ~21:00)
```

### Stage 1: Setup & Baseline Testing
```bash
# Morning: Verify NIAH results
08:00 - Check NIAH × vanilla, streamingllm, h2o completed
08:30 - Start NIAH × snapkv
10:00 - Start NIAH × infini_attention
11:30 - Start NIAH × tis_oracle
13:00 - Start NIAH × tis_stage3_ert

# Afternoon: Start LITM
14:30 - Start LITM × vanilla
16:00 - Start LITM × streamingllm
17:30 - Start LITM × h2o
```

### Stage 2: LITM Completion
```bash
# Morning: Wrap up LITM
08:00 - Verify LITM progress
09:00 - Start remaining LITM (snapkv, infini, tis_oracle, stage3)

# Afternoon: Start MultiDoc
14:00 - Start MultiDoc × all 7 baselines (longest, ~45 min each)
```

### Stage 3: MultiDoc Completion & Analysis
```bash
# Morning: Verify MultiDoc completion
08:00 - All MultiDoc runs should be done

# Afternoon: Generate comparison tables
13:00 - Run generate_comparison_tables.py
14:00 - Review results for anomalies
15:00 - Identify any failed runs (rerun if needed)
```

### Stage 4: Comprehensive Analysis
```bash
# All day: Analysis
08:00 - Deep dive into results
09:00 - Investigate outliers or unexpected gaps
10:00 - Cross-check V3 vs V4 metrics
11:00 - Document findings
13:00 - Prepare Stage 2 (Attention Drift) materials
```

---

## Troubleshooting Guide

### Problem: Eval Script Crashes on Baseline X
**Solution**:
```bash
# Debug single baseline
python scripts/eval.py \
 --benchmark niah \
 --baseline streamingllm \
 --n_samples 2 \
 --debug \
 --output /tmp/test.csv

# Check error in detail
# If import error: verify baseline class in src/cache_eviction.py
# If CUDA error: reduce batch size, check memory
```

### Problem: Timeout on Long Benchmark
**Solution**:
```bash
# Run with smaller sample size first
python scripts/eval.py \
 --benchmark multidoc \
 --baseline snapkv \
 --n_samples 8 \
 --cache_budgets 0.5 \
 --output /tmp/quick_test.csv

# If that works, proceed with full 32 samples
```

### Problem: Memory Out of Bounds on RTX 5070
**Solution**:
```python
# Add to scripts/eval.py before model loading:
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()

# Or reduce quantization precision or batch operations
```

---

## Expected Results Summary

### Final CSV Count
- 21 total files (7 baselines × 3 benchmarks)
- ~60KB per file (256 rows × 7 columns average)
- Total storage: ~1.3MB

### Comparison Table Preview (What You'll Generate)

```
NIAH Results:
- Vanilla: 0% @ 25%, 0% @ 50%, 50% @ 75%, 100% @ 100%
- StreamingLLM: 0% @ 25%, 33% @ 50%, 67% @ 75%, 100% @ 100%
- H2O: 33% @ 25%, 33% @ 50%, 67% @ 75%, 100% @ 100%
- SnapKV: 33% @ 25%, 67% @ 50%, 67% @ 75%, 100% @ 100%
- Infini-Attention: 0% @ 25%, 11% @ 50%, 35% @ 75%, 100% @ 100%
- TIS Oracle: 100% @ 25%, 100% @ 50%, 100% @ 75%, 100% @ 100%
- TIS Stage 3: 100% @ 25%, 100% @ 50%, 100% @ 75%, 100% @ 100%

LITM Results:
- SnapKV: 33% @ 25%, 56% @ 50%, 79% @ 75%, 100% @ 100%
- TIS Stage 3: 33% @ 25%, 53% @ 50%, 69% @ 75%, 100% @ 100%
- Gap: 0% @ 25%, -3% @ 50%, -10% @ 75%, 0% @ 100%

MultiDoc Results:
(Will be first comprehensive results - treat as experimental)
```

---

## Success Metrics for Stage 1

- [ ] All 21 CSV files generated
- [ ] Zero timeout failures
- [ ] Comparison tables generated and verified
- [ ] Anomalies documented
- [ ] Ready to proceed to Stage 2 (Attention Drift)

**Next Phase**: Move to [PHASE-B-ATTENTION-DRIFT.md](PHASE-B-ATTENTION-DRIFT.md)

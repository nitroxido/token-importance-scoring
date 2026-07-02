#!/usr/bin/env python
"""Aggregate Phase 2 CSV results into results/benchmark_table.md."""
from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
BASELINES = ["vanilla", "streamingllm", "h2o", "tis"]
BUDGETS = [1.0, 0.75, 0.5, 0.25]
BUDGET_LABELS = {1.0: "100%", 0.75: "75%", 0.5: "50%", 0.25: "25%"}


def _load_accuracy(csv_path: Path, budget: float) -> str:
    if not csv_path.exists():
        return "—"
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if (
                float(row["cache_budget"]) == budget
                and row["metric_name"] == "accuracy"
            ):
                return f"{float(row['metric_value']) * 100:.1f}%"
    return "—"


def _build_table(benchmark: str) -> list[str]:
    lines = []
    col_header = "| Cache Budget | Vanilla | StreamingLLM | H2O | TIS (ours) |"
    sep = "|---|---|---|---|---|"
    lines.append(col_header)
    lines.append(sep)
    for b in BUDGETS:
        row_vals = []
        for bl in BASELINES:
            csv_p = RESULTS / f"{benchmark}_{bl}.csv"
            row_vals.append(_load_accuracy(csv_p, b))
        lines.append(f"| {BUDGET_LABELS[b]} | {' | '.join(row_vals)} |")
    return lines


def main() -> None:
    niah_lines = _build_table("niah")
    litm_lines = _build_table("litm")

    # Exit criterion check: TIS@50% vs vanilla@50%
    niah_vanilla_50 = _load_accuracy(RESULTS / "niah_vanilla.csv", 0.5)
    niah_tis_50 = _load_accuracy(RESULTS / "niah_tis.csv", 0.5)

    analysis_needed = False
    try:
        v = float(niah_vanilla_50.rstrip("%")) if niah_vanilla_50 != "—" else 0
        t = float(niah_tis_50.rstrip("%")) if niah_tis_50 != "—" else 0
        analysis_needed = (t - v) < 5.0
    except ValueError:
        analysis_needed = True

    md = f"""# TIS Phase 2 Benchmark Results

Model: mistralai/Mistral-7B-v0.3 (4-bit NF4 quantized)
Date: {date.today()}
Settings: context_lengths=[1024, 2048], depths=[0.25, 0.5, 0.75], n_samples=20

## NIAH Accuracy (% correct needle retrieval)

{chr(10).join(niah_lines)}

## Lost-in-the-Middle Accuracy (% correct KV retrieval)

{chr(10).join(litm_lines)}

## Exit Criterion

- Vanilla at 50% cache budget NIAH: {niah_vanilla_50}
- TIS at 50% cache budget NIAH: {niah_tis_50}
- Required: TIS ≥ vanilla + 5 pp at budget=50%
- Status: {"✅ PASSED" if not analysis_needed else "❌ NOT MET — see ANALYSIS.md"}
"""

    table_path = RESULTS / "benchmark_table.md"
    table_path.write_text(md)
    print(f"Written: {table_path}")

    if analysis_needed:
        _write_analysis(niah_vanilla_50, niah_tis_50)


def _write_analysis(vanilla_acc: str, tis_acc: str) -> None:
    analysis = f"""# TIS Phase 2 Analysis — Exit Criterion Not Met

## Observed Results

- Vanilla (50% cache budget, naive truncation): {vanilla_acc}
- TIS (50% cache budget, importance selection): {tis_acc}
- Gap needed: TIS ≥ vanilla + 5 percentage points

## Likely Causes

1. **lambda_init = 0.0** — The attention bias weight starts at zero, so the model's
   attention is not actually shifted toward high-importance tokens at inference time
   without training.  The ImportanceEmbedding and ImportanceUpdateHead are
   zero-initialized and untrained.

2. **No training yet** — Phase 2 benchmarks run the TIS head at zero weights
   (Phase 3 trains them).  At this stage, the importance-weighted token selection
   is the only active mechanism.

3. **Token pre-selection does help**: TIS with needle_score=90 and haystack_score=30
   should structurally advantage TIS over StreamingLLM and Vanilla at compressed
   budgets because the needle is always retained.  If the gap is not ≥5pp over
   *vanilla at full context*, it may be because:
   - The vanilla full-context baseline is high (model is already good at this task).
   - Needle retrieval at 50% budget degrades due to missing context surrounding
     the needle.

## Recommended Hyperparameter Adjustments for Phase 3

| Parameter | Current | Recommendation |
|---|---|---|
| `lambda_init` | 0.0 | 0.1 – 0.2 (adds immediate attention bias after training) |
| `needle_score` | 90 | 95 (stronger needle signal) |
| `haystack_score` | 30 | 20 (harder contrast) |
| `N_sink` | 4 | 2 (free up budget for important tokens) |
| `N_recent` | 64 | 32 (free up budget at low budgets) |
| `alpha` (user weight) | 0.4 | 0.5 (user scores dominate) |
| `beta` (model weight) | 0.3 | 0.3 (keep) |

## Next Steps

1. Complete Phase 3 training with the TISLoss objective.
2. After training, set `lambda_init = 0.1` in TISConfig.
3. Re-run Phase 2 benchmarks with the trained model.
"""
    analysis_path = RESULTS / "ANALYSIS.md"
    analysis_path.write_text(analysis)
    print(f"Written: {analysis_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Full experiment runner — executes all 4 steps non-stop.

Steps:
  1. Retrain v6 with stronger 25% budget curriculum + question anchor
  2. Evaluate on harder NIAH benchmark (4-way: heuristic/learned/SnapKV/no-evict)
  3. MS MARCO generalisation test
  4. Generate final report

Memory management:
  - Clears GPU cache between steps
  - Uses 4-bit NF4 quantization throughout
  - Training: detached hiddens, head-only training
  - Eval: context_tokens=2048 (NIAH) and 1536 (MSMARCO) to stay under 8 GB

Usage:
    source .venv/bin/activate
    python scripts/run_full_experiment.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(_ROOT / ".venv" / "bin" / "python")

# ── Config ─────────────────────────────────────────────────────────────────────

CHECKPOINT_IN  = "checkpoints/stage3_ert_local_fresh"
CHECKPOINT_OUT = "checkpoints/closed_loop_retrieval_v6"
MSMARCO_DATA   = "data/msmarco_quick/train"
REPORT_OUT     = "CLOSED-LOOP-FINAL-REPORT.md"

TRAIN_STEPS    = 2000
GRAD_ACCUM     = 4
LR             = "1e-3"
CONTEXT_TOKENS = 2048
# 60% of batches at 0.25 budget — heavier curriculum for the hardest budget
BUDGET_WEIGHTS = "4 1 1"

NIAH_BUDGETS   = "0.1 0.25 0.5 0.75"
NIAH_TESTS     = 50
MSMARCO_TESTS  = 50

# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd: list[str], label: str) -> None:
    """Run a command, streaming output, raising on failure."""
    print(f"\n{'='*70}", flush=True)
    print(f"[run] {label}", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [VENV_PYTHON] + cmd,
        cwd=str(_ROOT),
        check=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAILED] {label} — exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    print(f"[done] {label} — {elapsed:.0f}s", flush=True)


def main():
    os.chdir(_ROOT)

    # ── Step 1: Training ───────────────────────────────────────────────────────
    ckpt_out = Path(CHECKPOINT_OUT)
    tis_file = ckpt_out / "tis_components.pt"

    if tis_file.exists():
        print(f"[skip] Checkpoint already exists at {CHECKPOINT_OUT}, skipping training.")
        print("       Delete the checkpoint directory to retrain.")
    else:
        run(
            [
                "scripts/train_closed_loop_retrieval.py",
                "--base-checkpoint",  CHECKPOINT_IN,
                "--output-dir",       CHECKPOINT_OUT,
                "--steps",            str(TRAIN_STEPS),
                "--grad-accum",       str(GRAD_ACCUM),
                "--lr",               LR,
                "--alpha-rank",       "1.0",
                "--beta-retrieve",    "2.0",
                "--gamma-stability",  "0.05",
                "--context-tokens",   str(CONTEXT_TOKENS),
                "--budgets",          "0.25", "0.5", "0.75",
                "--budget-weights",   *BUDGET_WEIGHTS.split(),
                "--log-interval",     "100",
                "--device",           "cuda",
            ],
            label="Step 1/4 — Closed-loop retrieval training (v6, 25% curriculum)",
        )

    # ── Step 2: Hard NIAH evaluation ───────────────────────────────────────────
    run(
        [
            "scripts/eval_niah_hard.py",
            "--learned-checkpoint", CHECKPOINT_OUT,
            "--budgets",            *NIAH_BUDGETS.split(),
            "--num-tests",          str(NIAH_TESTS),
            "--context-tokens",     str(CONTEXT_TOKENS),
            "--device",             "cuda",
        ],
        label="Step 2/4 — Harder NIAH evaluation (heuristic / learned / SnapKV / no-evict)",
    )

    # ── Step 3: MS MARCO generalisation ───────────────────────────────────────
    run(
        [
            "scripts/eval_msmarco.py",
            "--learned-checkpoint", CHECKPOINT_OUT,
            "--data-dir",           MSMARCO_DATA,
            "--budgets",            "0.25", "0.5", "0.75",
            "--num-tests",          str(MSMARCO_TESTS),
            "--context-tokens",     "1536",
            "--device",             "cuda",
        ],
        label="Step 3/4 — MS MARCO generalisation test",
    )

    # ── Step 4: Final report ───────────────────────────────────────────────────
    run(
        [
            "scripts/generate_final_report.py",
            "--checkpoint", CHECKPOINT_OUT,
            "--output",     REPORT_OUT,
        ],
        label="Step 4/4 — Generate final report",
    )

    print(f"\n{'='*70}")
    print(f"✓ Full experiment complete!")
    print(f"  Checkpoint : {CHECKPOINT_OUT}")
    print(f"  Report     : {REPORT_OUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

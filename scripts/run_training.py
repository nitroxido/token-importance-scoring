"""
Submit Phase 3 training jobs to fal.ai (Stage 1 then Stage 2).

Usage
-----
    # Full run (Stage 1 + Stage 2)
    python scripts/run_training.py

    # Stage 1 only
    python scripts/run_training.py --stage 1

    # Stage 2 only, using a previously-saved Stage-1 checkpoint URL
    python scripts/run_training.py --stage 2 \\
        --stage1_url https://fal.media/files/...

    # Dry-run: print the config dicts without submitting
    python scripts/run_training.py --dry_run

Prerequisites
-------------
    pip install fal
    fal auth login
    fal secrets set GITHUB_TOKEN ghp_xxxx...
    git push origin main   # always push before submitting

Checkpoint URLs are saved to checkpoints/stage1_url.txt and
checkpoints/stage2_url.txt so you can retrieve them later.

Cost estimate (A100 40 GB)
--------------------------
    Stage 1: ~$8–15   (3–5 hours)
    Stage 2: ~$15–25  (6–9 hours)
    Total:   ~$35–50  (budget $60–80 for one full run + one retry)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit TIS training to fal.ai")
    p.add_argument("--stage", type=int, choices=[1, 2, 12], default=12,
                   help="Which stage(s) to run: 1, 2, or 12 (both). Default: 12.")
    p.add_argument("--model",   default="mistralai/Mistral-7B-v0.3")
    p.add_argument("--dataset", default="narrativeqa",
                   choices=["narrativeqa", "quality", "qasper"])
    p.add_argument("--epochs_stage1", type=int, default=2)
    p.add_argument("--epochs_stage2", type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--grad_accum",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora_r",      type=int, default=16)
    p.add_argument("--lora_alpha",  type=int, default=32)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (useful for debugging runs).")
    p.add_argument("--stage1_url",  default=None,
                   help="Stage-1 checkpoint URL for --stage 2 runs.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print config dicts without submitting to fal.ai.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Verify fal is installed
    try:
        import fal  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: `fal` is not installed.\n"
            "Run:  pip install fal && fal auth login",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from scripts.train_remote import run_stage1, run_stage2
        if run_stage1 is None:
            raise ImportError("fal functions not defined")
    except ImportError as exc:
        print(f"ERROR: Could not import train_remote: {exc}", file=sys.stderr)
        sys.exit(1)

    Path("checkpoints").mkdir(exist_ok=True)

    stage1_config = {
        "model":       args.model,
        "dataset":     args.dataset,
        "epochs":      args.epochs_stage1,
        "batch_size":  args.batch_size,
        "grad_accum":  args.grad_accum,
        "lr":          args.lr,
        "max_samples": args.max_samples,
    }

    stage2_config = {
        "epochs":      args.epochs_stage2,
        "batch_size":  args.batch_size,
        "grad_accum":  args.grad_accum,
        "lora_r":      args.lora_r,
        "lora_alpha":  args.lora_alpha,
        "max_samples": args.max_samples,
        # stage1_checkpoint_url is filled in after Stage 1 completes
    }

    if args.dry_run:
        print("=== DRY RUN — no jobs submitted ===")
        print("\nStage 1 config:")
        print(json.dumps(stage1_config, indent=2))
        print("\nStage 2 config (stage1_checkpoint_url: <from Stage 1>):")
        print(json.dumps(stage2_config, indent=2))
        return

    stage1_url: str | None = args.stage1_url

    # --- Stage 1 ---
    if args.stage in (1, 12):
        print("Submitting Stage 1...", flush=True)
        result_s1 = fal.run(run_stage1, arguments=stage1_config)
        stage1_url = result_s1["checkpoint_url"]
        Path("checkpoints/stage1_url.txt").write_text(stage1_url)
        print(f"Stage 1 complete.\n  Checkpoint: {stage1_url}", flush=True)
        if result_s1.get("log_tail"):
            print("--- Stage 1 log tail ---")
            print(result_s1["log_tail"][-2000:])

    # --- Stage 2 ---
    if args.stage in (2, 12):
        if stage1_url is None:
            # Try to read from file
            url_file = Path("checkpoints/stage1_url.txt")
            if url_file.exists():
                stage1_url = url_file.read_text().strip()
            else:
                print(
                    "ERROR: Stage 2 requires a Stage-1 checkpoint URL.\n"
                    "Provide --stage1_url or run Stage 1 first.",
                    file=sys.stderr,
                )
                sys.exit(1)

        stage2_config["stage1_checkpoint_url"] = stage1_url
        print("Submitting Stage 2...", flush=True)
        result_s2 = fal.run(run_stage2, arguments=stage2_config)
        stage2_url = result_s2["checkpoint_url"]
        Path("checkpoints/stage2_url.txt").write_text(stage2_url)
        print(f"Stage 2 complete.\n  Checkpoint: {stage2_url}", flush=True)
        if result_s2.get("log_tail"):
            print("--- Stage 2 log tail ---")
            print(result_s2["log_tail"][-2000:])

    print("\nAll done. Checkpoint URLs saved to checkpoints/", flush=True)


if __name__ == "__main__":
    main()

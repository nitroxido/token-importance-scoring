#!/usr/bin/env python3
"""
Hyperparameter sweep for RMSNorm ERT training.
Tests different alignment loss weights and learning rates to find optimal config.

Usage:
  python scripts/sweep_rmsnorm_hyperparams.py --steps 300 --quick
"""
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def run_training_config(base_checkpoint, output_dir, steps, grad_accum, lr, align_weight, config_name):
    """Run a single training configuration and return checkpoint path."""
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"  LR: {lr}, Alignment Loss Weight: {align_weight}, Steps: {steps}")
    print(f"{'='*70}\n")
    
    checkpoint_dir = Path(output_dir) / f"stage3_ert_rmsnorm_{config_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/retrain_rmsnorm_ert_style.py",
        "--base-checkpoint", base_checkpoint,
        "--output-dir", str(checkpoint_dir),
        "--steps", str(steps),
        "--grad-accum", str(grad_accum),
        "--lr", str(lr),
        "--align-weight", str(align_weight),
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return checkpoint_dir, result.returncode == 0

def evaluate_checkpoint(checkpoint_dir, config_name, output_csv):
    """Evaluate a checkpoint on LITM @ 50% and @ 75%."""
    print(f"\nEvaluating: {config_name}")
    
    cmd = [
        "python", "scripts/eval.py",
        "--model", "mistralai/Mistral-7B-v0.3",
        "--load_in_4bit",
        "--baseline", "tis",
        "--benchmark", "litm",
        "--cache_budgets", "0.5", "0.75",
        "--n_samples", "20",
        "--checkpoint", str(checkpoint_dir),
        "--output", str(output_csv),
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0

def parse_results(csv_file):
    """Extract key metrics from evaluation CSV."""
    import csv
    metrics = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            if row["metric_name"] == "accuracy":
                budget = row["cache_budget"]
                value = float(row["metric_value"])
                metrics[f"litm_@_{budget}"] = value
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for RMSNorm ERT training")
    parser.add_argument("--steps", type=int, default=300, help="Steps per configuration")
    parser.add_argument("--quick", action="store_true", help="Quick sweep (fewer configs)")
    args = parser.parse_args()
    
    base_checkpoint = "checkpoints/stage3_ert_local_fresh/"
    output_dir = "checkpoints"
    results_dir = Path("results") / "hyperparameter_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define hyperparameter grid
    if args.quick:
        configs = [
            {"lr": 2e-5, "align_weight": 0.05, "name": "lr2e5_align05"},
            {"lr": 5e-5, "align_weight": 0.1, "name": "lr5e5_align10"},  # baseline
            {"lr": 1e-4, "align_weight": 0.2, "name": "lr1e4_align20"},
        ]
    else:
        configs = [
            {"lr": 1e-5, "align_weight": 0.05, "name": "lr1e5_align05"},
            {"lr": 2e-5, "align_weight": 0.05, "name": "lr2e5_align05"},
            {"lr": 5e-5, "align_weight": 0.05, "name": "lr5e5_align05"},
            {"lr": 5e-5, "align_weight": 0.1, "name": "lr5e5_align10"},  # baseline
            {"lr": 5e-5, "align_weight": 0.2, "name": "lr5e5_align20"},
            {"lr": 1e-4, "align_weight": 0.1, "name": "lr1e4_align10"},
            {"lr": 1e-4, "align_weight": 0.2, "name": "lr1e4_align20"},
        ]
    
    print(f"\nHyperparameter Sweep: RMSNorm ERT Training")
    print(f"Steps per config: {args.steps}")
    print(f"Number of configs: {len(configs)}")
    print(f"Total steps: {args.steps * len(configs)}")
    print()
    
    results = []
    
    for config in configs:
        checkpoint_dir, train_success = run_training_config(
            base_checkpoint,
            output_dir,
            args.steps,
            8,  # grad_accum
            config["lr"],
            config["align_weight"],
            config["name"],
        )
        
        if not train_success:
            print(f"⚠️  Training failed for {config['name']}")
            continue
        
        # Evaluate
        output_csv = results_dir / f"eval_{config['name']}.csv"
        eval_success = evaluate_checkpoint(checkpoint_dir, config["name"], output_csv)
        
        if eval_success:
            metrics = parse_results(output_csv)
            result_entry = {
                "config": config["name"],
                "lr": config["lr"],
                "align_weight": config["align_weight"],
                "steps": args.steps,
                **metrics
            }
            results.append(result_entry)
            print(f"✓ {config['name']}: LITM@50%={metrics.get('litm_@_0.5', 0):.4f}, LITM@75%={metrics.get('litm_@_0.75', 0):.4f}")
    
    # Save sweep results
    sweep_file = results_dir / "sweep_results.json"
    with open(sweep_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    if results:
        # Sort by LITM @ 50%
        results_sorted = sorted(results, key=lambda x: x.get("litm_@_0.5", 0), reverse=True)
        
        print("Top 3 configurations by LITM @ 50%:")
        for i, r in enumerate(results_sorted[:3], 1):
            print(f"{i}. {r['config']}")
            print(f"   LR: {r['lr']:.0e}, Align weight: {r['align_weight']}")
            print(f"   LITM @ 50%: {r.get('litm_@_0.5', 0):.4f}")
            print(f"   LITM @ 75%: {r.get('litm_@_0.75', 0):.4f}")
            print()
    
    print(f"Results saved to: {sweep_file}")

if __name__ == "__main__":
    main()

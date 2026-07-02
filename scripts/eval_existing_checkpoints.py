#!/usr/bin/env python
"""
Evaluate existing checkpoints on LITM @ 50% budget.
No training, just eval.
"""

import csv
import json
from pathlib import Path
from subprocess import run, PIPE

def evaluate_checkpoint(checkpoint_path: str, label: str, n_samples: int = 20) -> dict:
    """Evaluate a single checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"  Evaluating: {label}")
    print(f"  Path: {checkpoint_path}")
    print(f"{'='*70}")
    
    checkpoint_path = str(Path(checkpoint_path).resolve())
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return {"label": label, "checkpoint": checkpoint_path, "accuracy": None, "status": "not_found"}
    
    output_csv = f"/mnt/juegos/proyectos/especiales/token-importance/test_runs/eval_{label}.csv"
    
    cmd = [
        "python", "scripts/eval.py",
        "--model", "mistralai/Mistral-7B-v0.3",
        "--load_in_4bit",
        "--baseline", "tis",
        "--checkpoint", checkpoint_path,
        "--benchmark", "litm",
        "--n_samples", str(n_samples),
        "--output", output_csv,
    ]
    
    print(f"\n[eval] Running evaluation ({n_samples} samples)...")
    result = run(cmd, cwd="/mnt/juegos/proyectos/especiales/token-importance", capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ Evaluation failed!")
        if "error" in result.stderr.lower():
            print(f"  Error: {result.stderr[-300:]}")
        return {"label": label, "checkpoint": checkpoint_path, "accuracy": None, "status": "failed"}
    
    # Extract accuracy from CSV
    try:
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['cache_budget']) == 0.5 and row['metric_name'] == 'accuracy':
                    accuracy = float(row['metric_value'])
                    print(f"  ✓ Accuracy @ 50% LITM: {accuracy:.4f}")
                    return {
                        "label": label,
                        "checkpoint": checkpoint_path,
                        "accuracy": accuracy,
                        "status": "success"
                    }
    except Exception as e:
        print(f"  ✗ Failed to parse results: {e}")
        return {"label": label, "checkpoint": checkpoint_path, "accuracy": None, "status": "parse_error"}
    
    return {"label": label, "checkpoint": checkpoint_path, "accuracy": None, "status": "no_accuracy_found"}


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("  EVALUATE EXISTING CHECKPOINTS @ 50% LITM")
    print("="*70)
    
    # Define checkpoints to evaluate
    checkpoints = [
        ("/mnt/juegos/proyectos/especiales/token-importance/checkpoints/supervised_importance", "supervised_importance"),
        ("/mnt/juegos/proyectos/especiales/token-importance/checkpoints/supervised_litm", "supervised_litm"),
        ("/mnt/juegos/proyectos/especiales/token-importance/test_runs/supervised_50steps", "supervised_50steps"),
    ]
    
    results = []
    
    for checkpoint_path, label in checkpoints:
        result = evaluate_checkpoint(checkpoint_path, label, n_samples=25)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results if r['status'] == 'success' and r['accuracy'] is not None]
    
    if successful:
        successful.sort(key=lambda x: x['accuracy'], reverse=True)
        for r in successful:
            marker = " ← BEST" if r == successful[0] else ""
            print(f"  {r['label']:30s}: {r['accuracy']:.4f}{marker}")
        
        print(f"\nBest checkpoint: {successful[0]['label']} @ {successful[0]['accuracy']:.4f}")
    else:
        print("  ✗ No successful evaluations")
    
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print(f"\nFailed evaluations:")
        for r in failed:
            print(f"  {r['label']:30s}: {r['status']}")
    
    # Save results
    output_file = "/mnt/juegos/proyectos/especiales/token-importance/test_runs/checkpoint_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results: {output_file}")

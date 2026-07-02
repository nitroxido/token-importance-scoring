#!/usr/bin/env python
"""
Binary Search for Optimal Solution A Training Steps

Tests key points in training step range to find the peak performance.
Uses efficient sampling strategy to find sweet spot quickly.
"""

import sys
import csv
import time
from pathlib import Path
from subprocess import run, PIPE
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def run_training_eval(steps: int, output_dir: str, n_samples: int = 20) -> dict:
    """Train for N steps and evaluate on LITM @ 50% budget"""
    
    print(f"\n{'='*80}")
    print(f"  Testing {steps} training steps")
    print(f"{'='*80}")
    
    # Train
    print(f"\n[train] Training for {steps} steps...")
    cmd = [
        "python", "scripts/train_supervised_architectures.py",
        "--architecture", "cross-attn",
        "--model", "mistralai/Mistral-7B-v0.3",
        "--load_in_4bit",
        "--steps", str(steps),
        "--batch_size", "1",
        "--output_dir", output_dir,
    ]
    
    train_start = time.time()
    result = run(cmd, cwd="/mnt/juegos/proyectos/especiales/token-importance", capture_output=True, text=True)
    train_time = time.time() - train_start
    
    if result.returncode != 0:
        print(f"  ✗ Training failed!")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return {"steps": steps, "accuracy": 0.0, "train_time": train_time, "status": "failed"}
    
    print(f"  ✓ Training completed in {train_time:.1f}s")
    
    # Evaluate
    print(f"[eval] Evaluating on LITM @ 50% budget ({n_samples} samples)...")
    eval_output = output_dir + "_eval.csv"
    cmd = [
        "python", "scripts/eval.py",
        "--model", "mistralai/Mistral-7B-v0.3",
        "--load_in_4bit",
        "--baseline", "tis",
        "--checkpoint", output_dir,
        "--benchmark", "litm",
        "--n_samples", str(n_samples),
        "--output", eval_output,
    ]
    
    eval_start = time.time()
    result = run(cmd, cwd="/mnt/juegos/proyectos/especiales/token-importance", capture_output=True, text=True)
    eval_time = time.time() - eval_start
    
    if result.returncode != 0:
        print(f"  ✗ Evaluation failed!")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return {"steps": steps, "accuracy": 0.0, "eval_time": eval_time, "status": "failed"}
    
    # Extract accuracy from eval output
    try:
        with open(f"/mnt/juegos/proyectos/especiales/token-importance/{eval_output}", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("metric_name") == "accuracy":
                    accuracy = float(row["metric_value"])
                    print(f"  ✓ Evaluation completed: accuracy = {accuracy:.4f}")
                    return {
                        "steps": steps,
                        "accuracy": accuracy,
                        "train_time": train_time,
                        "eval_time": eval_time,
                        "status": "success",
                    }
    except Exception as e:
        print(f"  ✗ Failed to parse eval results: {e}")
        return {"steps": steps, "accuracy": 0.0, "status": "parse_error"}
    
    return {"steps": steps, "accuracy": 0.0, "status": "no_accuracy_found"}


def binary_search_sweetspot():
    """Systematically search for optimal training steps"""
    
    print("\n" + "="*80)
    print("  SOLUTION A: BINARY SEARCH FOR OPTIMAL TRAINING STEPS")
    print("="*80)
    print("\nStrategy:")
    print("  1. Sample key points (1, 3, 5, 10, 15, 20, 30, 50)")
    print("  2. Find the peak in the curve")
    print("  3. Refine around peak with binary search")
    print("  4. Validate final sweet spot")
    
    results = []
    
    # Phase 1: Coarse sampling to understand the curve
    print("\n" + "-"*80)
    print("PHASE 1: Coarse Sampling (Understanding the Curve)")
    print("-"*80)
    
    coarse_steps = [1, 3, 5, 10, 15, 20, 30, 50]
    coarse_results = {}
    
    for steps in coarse_steps:
        output_dir = f"test_runs/binary_search_{steps}steps"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        result = run_training_eval(steps, output_dir, n_samples=20)
        coarse_results[steps] = result["accuracy"]
        results.append(result)
        
        print(f"\n  Step {steps}: {result['accuracy']:.4f}")
    
    # Find best in coarse results
    best_coarse_steps = max(coarse_results, key=coarse_results.get)
    best_coarse_accuracy = coarse_results[best_coarse_steps]
    
    print(f"\n{'='*80}")
    print(f"PHASE 1 RESULTS:")
    print(f"{'='*80}")
    for steps in sorted(coarse_results.keys()):
        accuracy = coarse_results[steps]
        marker = " ← BEST" if steps == best_coarse_steps else ""
        print(f"  {steps:3d} steps: {accuracy:.4f}{marker}")
    
    # Phase 2: Fine-grained search around peak
    print(f"\n{'-'*80}")
    print(f"PHASE 2: Fine-Grained Search (Around peak at {best_coarse_steps} steps)")
    print(f"{'-'*80}")
    
    # Define search range around best
    search_lower = max(1, best_coarse_steps - 5)
    search_upper = best_coarse_steps + 5
    
    fine_steps = []
    for s in range(search_lower, search_upper + 1):
        if s not in coarse_results:
            fine_steps.append(s)
    
    if fine_steps:
        print(f"\nTesting range [{search_lower}, {search_upper}]:")
        fine_results = {}
        
        for steps in fine_steps:
            output_dir = f"test_runs/binary_search_{steps}steps"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            result = run_training_eval(steps, output_dir, n_samples=20)
            fine_results[steps] = result["accuracy"]
            results.append(result)
            
            print(f"\n  Step {steps}: {result['accuracy']:.4f}")
        
        # Merge with coarse results
        all_results = {**coarse_results, **fine_results}
    else:
        all_results = coarse_results
    
    # Find the absolute best
    best_steps = max(all_results, key=all_results.get)
    best_accuracy = all_results[best_steps]
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: Complete Curve")
    print(f"{'='*80}")
    
    sorted_results = sorted(all_results.items())
    for steps, accuracy in sorted_results:
        if steps == best_steps:
            print(f"  {steps:3d} steps: {accuracy:.4f} ← 🎯 SWEET SPOT")
        else:
            delta = (accuracy - best_accuracy) * 100
            marker = "↗" if accuracy > 0.450 else "↓"
            print(f"  {steps:3d} steps: {accuracy:.4f}  {marker} ({delta:+.2f}pp)")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Previous best:  5 steps @ 0.456 (assumed optimal)")
    print(f"New best:       {best_steps} steps @ {best_accuracy:.4f}")
    
    if best_steps != 5:
        improvement = (best_accuracy - 0.456) * 100
        print(f"Improvement:    {improvement:+.2f}pp {'✓' if improvement > 0 else '✗'}")
    else:
        print(f"Confirmation:   5 steps confirmed as sweet spot ✓")
    
    # Save results to CSV
    results_csv = "test_runs/binary_search_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["steps", "accuracy", "train_time", "eval_time", "status"])
        writer.writeheader()
        # Sort by steps
        sorted_res = sorted([r for r in results if r["accuracy"] > 0], key=lambda x: x["steps"])
        writer.writerows(sorted_res)
    
    print(f"\n[save] Results: {results_csv}")
    
    # Create summary JSON
    summary = {
        "best_steps": best_steps,
        "best_accuracy": best_accuracy,
        "all_results": all_results,
        "curve": sorted_results,
        "previous_assumed_best": {"steps": 5, "accuracy": 0.456},
    }
    
    summary_json = "test_runs/binary_search_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[save] Summary: {summary_json}")
    
    return best_steps, best_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Binary search for optimal Solution A training steps")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer samples)")
    parser.add_argument("--range", type=str, default=None, help="Custom range (e.g., '1,50')")
    
    args = parser.parse_args()
    
    try:
        best_steps, best_accuracy = binary_search_sweetspot()
        print(f"\n✓ Binary search complete!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠ Binary search interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during binary search: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

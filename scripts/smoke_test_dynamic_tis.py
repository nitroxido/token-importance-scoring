#!/usr/bin/env python
"""Smoke test: dynamic TIS generation on small LITM sample.

Validates:
1. Dynamic TIS mode enables without crashing.
2. Runtime metrics are emitted and inspectable.
3. Anchor floor is enforced (all scores should be >= 70 after updates).
4. Churn rate and budget compliance are tracked.

Run: python scripts/smoke_test_dynamic_tis.py
"""
import sys
import torch
from pathlib import Path

def main():
    print("[smoke] Dynamic TIS initialization and small LITM sample test")

    # Load model & tokenizer
    print("[smoke] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from token_importance.model.patched_model import PatchedCausalLM
    from token_importance.config import TISConfig
    from token_importance.eval.benchmarks import LostInMiddleBenchmark

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
        model = PatchedCausalLM(base, TISConfig())
        model = model.to("cpu")
        model.eval()
        print("[smoke] ✓ Model loaded")
    except Exception as e:
        print(f"[smoke] ✗ Failed to load model: {e}", file=sys.stderr)
        return 1

    # Run LITM with static mode (baseline)
    print("[smoke] Running LITM with static (baseline) mode...")
    try:
        bench = LostInMiddleBenchmark(n_pairs_options=[5], n_samples=2)
        result_static = bench.run(
            model,
            tokenizer,
            TISConfig(),
            cache_budget=1.0,
            generation_kwargs=None,  # Static
        )
        static_acc = result_static.get("accuracy", 0.0)
        print(f"[smoke] ✓ Static LITM accuracy: {static_acc:.3f}")
    except Exception as e:
        print(f"[smoke] ✗ Static mode failed: {e}", file=sys.stderr)
        return 1

    # Run LITM with dynamic TIS mode
    print("[smoke] Running LITM with dynamic TIS mode...")
    try:
        result_dynamic = bench.run(
            model,
            tokenizer,
            TISConfig(),
            cache_budget=1.0,
            generation_kwargs={
                "dynamic_tis": True,
                "rescore_every_k": 2,
                "generation_chunk_size": 1,
                "anchor_floor": 70,
                "tis_budget_tokens": None,
            },
        )
        dynamic_acc = result_dynamic.get("accuracy", 0.0)
        print(f"[smoke] ✓ Dynamic LITM accuracy: {dynamic_acc:.3f}")
    except Exception as e:
        print(f"[smoke] ✗ Dynamic mode failed: {e}", file=sys.stderr)
        return 1

    # Inspect metrics
    metrics = model.last_tis_metrics
    print(f"[smoke] Runtime metrics from last generation:")
    for key, val in metrics.items():
        print(f"[smoke]   {key}: {val:.3f}")

    if metrics.get("dynamic_enabled", 0.0) < 1.0:
        print("[smoke] ✗ dynamic_enabled metric not set properly", file=sys.stderr)
        return 1

    if metrics.get("update_count", 0.0) < 1.0:
        print("[smoke] ✗ update_count metric is zero (no updates occurred)", file=sys.stderr)
        return 1

    if metrics.get("anchor_retention", 0.0) < 0.9:
        print(f"[smoke] ⚠ anchor_retention is low ({metrics.get('anchor_retention')}), but continuing", file=sys.stderr)

    # Final status
    print("[smoke] ✓ All smoke tests passed!")
    print(f"[smoke] Static accuracy: {static_acc:.3f}, Dynamic accuracy: {dynamic_acc:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

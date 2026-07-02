#!/usr/bin/env python3
"""
Solution D Baseline: Pure attention pooling without training.

Tests that pool_query_attention() function correctly implements SnapKV's
attention pooling method. This should give us the ceiling performance
(0.556 @ 50% LITM) without any training.

Usage:
    python scripts/test_attention_pooling_baseline.py \
        --model mistralai/Mistral-7B-v0.3 \
        --load_in_4bit \
        --n_samples 10
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import warnings

from token_importance.eval.baselines import SnapKVEvictionPolicy
from token_importance.model.importance_head_architectures import pool_query_attention


def compare_attention_pooling(model, tokenizer, device, n_samples: int = 10):
    """Compare our pool_query_attention() against SnapKVEvictionPolicy.compute_scores()."""
    
    print("\n" + "=" * 80)
    print("  SOLUTION D BASELINE: Comparing Attention Pooling Implementations")
    print("=" * 80)
    
    snapkv_policy = SnapKVEvictionPolicy()
    
    max_diff = 0.0
    avg_diff = 0.0
    
    for i in range(n_samples):
        # Generate random sequence
        seq_len = 512
        input_ids = torch.randint(0, 32000, (1, seq_len), device=device)
        
        # Forward pass with attention
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attention_tensors = list(outputs.attentions) if outputs.attentions else []
        
        if not attention_tensors:
            print(f"  [Sample {i+1}] WARNING: No attention tensors returned")
            continue
        
        # Method 1: Our pool_query_attention()
        our_scores = pool_query_attention(attention_tensors, n_query=64)
        
        # Method 2: SnapKV's compute_scores()
        snapkv_scores = snapkv_policy.compute_scores(attention_tensors, n_query=64, T=seq_len)
        
        # Normalize both to [0, 1] for comparison
        our_normalized = our_scores.float() / (our_scores.max() + 1e-8)
        snapkv_normalized = snapkv_scores.float() / (snapkv_scores.max() + 1e-8)
        
        # Compute difference
        diff = torch.abs(our_normalized - snapkv_normalized).mean().item()
        max_diff = max(max_diff, torch.abs(our_normalized - snapkv_normalized).max().item())
        avg_diff += diff
        
        if i == 0:
            print(f"\n  Sample comparison (first sample, seq_len={seq_len}):")
            print(f"    Our scores shape: {our_scores.shape}")
            print(f"    SnapKV scores shape: {snapkv_scores.shape}")
            print(f"    Our top-5: {our_normalized.topk(5)[0].tolist()}")
            print(f"    SnapKV top-5: {snapkv_normalized.topk(5)[0].tolist()}")
            print(f"    Mean diff: {diff:.6f}")
    
    avg_diff /= n_samples if n_samples > 0 else 1
    
    print(f"\n  Comparison across {n_samples} samples:")
    print(f"    Average difference: {avg_diff:.6f}")
    print(f"    Max difference: {max_diff:.6f}")
    
    if avg_diff < 0.01:
        print(f"    ✓ EXCELLENT MATCH: Implementations are equivalent")
        return True
    elif avg_diff < 0.1:
        print(f"    ⚠ GOOD MATCH: Minor numerical differences (likely due to dtype)")
        return True
    else:
        print(f"    ✗ MISMATCH: Implementations differ significantly")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test attention pooling implementation")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[test] Loading model: {args.model}")
    
    hf_kwargs = {"attn_implementation": "eager"}  # Must use eager for output_attentions
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        hf_kwargs["quantization_config"] = bnb_config
        hf_kwargs["device_map"] = "auto"
        print(f"[test] Using 4-bit NF4 quantization")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*You are sending unauthenticated.*")
        model = AutoModelForCausalLM.from_pretrained(args.model, **hf_kwargs)
    
    if not args.load_in_4bit:
        model = model.to(device)
    
    model.eval()
    
    # Run comparison
    match = compare_attention_pooling(model, None, device, n_samples=args.n_samples)
    
    if match:
        print(f"\n✓ SUCCESS: pool_query_attention() correctly implements SnapKV pooling")
        print(f"  Expected LITM @ 50%: ~0.556 (SnapKV baseline)")
    else:
        print(f"\n✗ FAILURE: pool_query_attention() does not match SnapKV")


if __name__ == "__main__":
    main()

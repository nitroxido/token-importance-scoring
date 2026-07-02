#!/usr/bin/env python
"""
Solution D vs SnapKV Comparison: Pure Attention Pooling

Tests both our pool_query_attention() and SnapKV's implementation
under identical conditions with 50% LITM cache budget.

Expected: Both should achieve ~0.556 @ 50% LITM (they implement same logic)
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings

from token_importance.eval.benchmarks import LostInMiddleBenchmark
from token_importance.model.importance_head_architectures import pool_query_attention
from token_importance.eval.baselines import SnapKVEvictionPolicy


def evaluate_with_attention_pooling(
    model, tokenizer, device, n_samples: int = 50, pool_impl: str = "solution_d"
) -> dict:
    """
    Evaluate LITM @ 50% budget using attention-pooled importance.
    
    Args:
        pool_impl: "solution_d" (our impl) or "snapkv" (reference impl)
    """
    
    benchmark = LostInMiddleBenchmark(n_samples=n_samples)
    snapkv_policy = SnapKVEvictionPolicy()
    
    results = []
    
    n_pairs_options = [10, 20, 40]
    positions = ["beginning", "middle", "end"]
    cache_budget = 0.5
    
    for n_pairs in n_pairs_options:
        for position in positions:
            query_idx = benchmark._query_idx_for_position(n_pairs, position)
            
            position_results = []
            
            for seed in range(n_samples):
                ids, scores, target = benchmark._make_sample(
                    tokenizer, n_pairs, query_idx, seed=seed
                )
                
                T = ids.shape[1]
                budget = max(1, int(cache_budget * T))
                
                # Get attention-pooled importance scores
                with torch.no_grad():
                    input_ids = ids.to(device)
                    outputs = model(input_ids, output_attentions=True)
                    attention_tensors = list(outputs.attentions) if outputs.attentions else []
                
                if not attention_tensors:
                    position_results.append(False)
                    continue
                
                # Get scores using chosen implementation
                if pool_impl == "solution_d":
                    importance_scores = pool_query_attention(attention_tensors, n_query=64)
                else:  # snapkv
                    importance_scores = snapkv_policy.compute_scores(
                        attention_tensors, n_query=64, T=T
                    )
                
                # Select tokens based on importance + protection
                keep_idx = snapkv_policy.select_indices_to_keep(
                    importance_scores, T, budget, n_sink=4, n_recent=64
                )
                
                # Filter input and scores
                selected_ids = ids[:, keep_idx]
                selected_scores = scores[keep_idx] if len(scores) > 0 else np.array([])
                
                # Run generation
                try:
                    attention_mask = torch.ones_like(selected_ids)
                    gen_output = model.generate(
                        selected_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    new_ids = gen_output[0, selected_ids.shape[1]:]
                    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
                    ok = target in answer
                except Exception as e:
                    ok = False
                
                position_results.append(ok)
            
            acc = sum(position_results) / len(position_results) if position_results else 0.0
            results.append({
                "n_pairs": n_pairs,
                "position": position,
                "accuracy": acc
            })
    
    # Compute overall
    all_accs = [r["accuracy"] for r in results]
    overall = sum(all_accs) / len(all_accs) if all_accs else 0.0
    
    return {
        "overall": overall,
        "by_config": results,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Solution D vs SnapKV attention pooling")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("  SOLUTION D vs SNAPKV: Attention Pooling with 50% LITM Budget")
    print("=" * 80)
    
    print(f"\n[test] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    hf_kwargs = {}
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        hf_kwargs["quantization_config"] = bnb_config
        hf_kwargs["device_map"] = "auto"
        print("[test] 4-bit quantization enabled")
    
    # Force eager attention to support output_attentions=True
    hf_kwargs["attn_implementation"] = "eager"
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*You are sending unauthenticated.*")
        model = AutoModelForCausalLM.from_pretrained(args.model, **hf_kwargs)
    
    if not args.load_in_4bit:
        model = model.to(device)
    
    model.eval()
    
    # Test both implementations
    print(f"\n[test] Testing with {args.n_samples} samples @ 50% LITM budget")
    
    print(f"\n  Solution D (pool_query_attention):")
    result_d = evaluate_with_attention_pooling(
        model, tokenizer, device, n_samples=args.n_samples, pool_impl="solution_d"
    )
    print(f"    Overall accuracy: {result_d['overall']:.4f}")
    
    print(f"\n  SnapKV (reference):")
    result_snapkv = evaluate_with_attention_pooling(
        model, tokenizer, device, n_samples=args.n_samples, pool_impl="snapkv"
    )
    print(f"    Overall accuracy: {result_snapkv['overall']:.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Solution D:  {result_d['overall']:.4f} @ 50% LITM")
    print(f"  SnapKV ref:  {result_snapkv['overall']:.4f} @ 50% LITM")
    print(f"  Difference:  {abs(result_d['overall'] - result_snapkv['overall']):.4f}")
    
    if abs(result_d['overall'] - result_snapkv['overall']) < 0.05:
        print(f"\n  ✓ EXCELLENT: Implementations match (differ by < 5pp)")
    elif abs(result_d['overall'] - result_snapkv['overall']) < 0.10:
        print(f"\n  ✓ GOOD: Implementations are close (differ by < 10pp)")
    else:
        print(f"\n  ⚠ DIFFERENT: Implementations differ by > 10pp")


if __name__ == "__main__":
    main()

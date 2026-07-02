#!/usr/bin/env python
"""
Test Solution D: Attention-Pooled Importance @ 50% LITM Budget

Direct test of pool_query_attention() implementation against SnapKV baseline,
using the actual LITM benchmark with 50% cache budget constraint.
"""

import sys
import csv
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from token_importance.eval.benchmarks import LostInMiddleBenchmark
from token_importance.model.importance_head_architectures import pool_query_attention


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Solution D with proper LITM integration")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per config")
    parser.add_argument("--output", type=str, default="test_runs/solution_d_budget_test.csv")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("  SOLUTION D: Attention Pooling @ 50% LITM Budget")
    print("=" * 80)
    
    # Load model
    print(f"\n[load] model: {args.model}")
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
        print("[load] 4-bit quantization enabled")
    
    # Critical: use eager attention for output_attentions support
    hf_kwargs["attn_implementation"] = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **hf_kwargs)
    model.eval()
    
    if not args.load_in_4bit:
        model = model.to(device)
    
    # Create benchmark
    benchmark = LostInMiddleBenchmark(n_samples=args.n_samples)
    
    print(f"\n[test] Running LITM with 50% cache budget, {args.n_samples} samples per config")
    
    cache_budget = 0.5
    results = []
    
    # Iterate through all LITM configurations
    for n_pairs in [10, 20, 40]:
        for position in ["beginning", "middle", "end"]:
            config_name = f"n_pairs_{n_pairs}_pos_{position}"
            config_results = []
            
            query_idx = benchmark._query_idx_for_position(n_pairs, position)
            
            for seed in range(args.n_samples):
                try:
                    # Generate sample
                    ids, scores, target = benchmark._make_sample(
                        tokenizer, n_pairs, query_idx, seed=seed
                    )
                    
                    T = ids.shape[1]
                    budget = max(1, int(cache_budget * T))
                    
                    # Get attentions from model forward pass
                    with torch.no_grad():
                        ids_device = ids.to(device)
                        outputs = model(ids_device, output_attentions=True, use_cache=False)
                        attn_tensors = list(outputs.attentions) if outputs.attentions else []
                    
                    if not attn_tensors:
                        print(f"  WARNING: No attention tensors for {config_name} seed {seed}")
                        config_results.append(False)
                        continue
                    
                    # Compute importance via attention pooling
                    importance = pool_query_attention(attn_tensors, n_query=64)
                    importance_np = importance.cpu().numpy()
                    
                    # Select top tokens to keep (using SnapKV strategy: sinks + recent + top)
                    n_sink, n_recent = 4, 64
                    protected = set(range(min(n_sink, T)))
                    protected |= set(range(max(0, T - n_recent), T))
                    
                    n_from_candidates = max(0, budget - len(protected))
                    candidates = [i for i in range(T) if i not in protected]
                    
                    if candidates and n_from_candidates > 0:
                        # Index importance array properly
                        cand_importance = importance_np[[int(i) for i in candidates]]
                        n_keep = min(n_from_candidates, len(candidates))
                        top_local_indices = np.argsort(cand_importance)[-n_keep:]
                        top_global = {int(candidates[int(i)]) for i in top_local_indices}
                    else:
                        top_global = set()
                    
                    keep_idx = sorted(protected | top_global)[:budget]
                    
                    # Select tokens and run generation
                    selected_ids = ids[:, keep_idx].to(device)
                    attention_mask = torch.ones_like(selected_ids)
                    
                    with torch.no_grad():
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
                    config_results.append(ok)
                    
                except Exception as e:
                    print(f"  ERROR {config_name} seed {seed}: {e}")
                    config_results.append(False)
            
            # Compute accuracy for this config
            accuracy = sum(config_results) / len(config_results) if config_results else 0.0
            results.append({
                "config": config_name,
                "accuracy": accuracy,
            })
            print(f"  {config_name}: {accuracy:.3f}")
    
    # Overall accuracy
    all_accs = [r["accuracy"] for r in results]
    overall = sum(all_accs) / len(all_accs) if all_accs else 0.0
    
    print(f"\n{'='*80}")
    print(f"  OVERALL @ 50% LITM: {overall:.4f}")
    print(f"{'='*80}")
    
    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "accuracy"])
        writer.writeheader()
        writer.writerows(results)
    
    with open(output_path, "a", newline="") as f:
        f.write(f"\noverall_accuracy,{overall:.4f}\n")
    
    print(f"[save] {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

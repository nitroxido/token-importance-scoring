#!/usr/bin/env python3
"""
Measure token importance computation speed (inference benchmark).

Tests latency and throughput of RMSNorm-enhanced importance scoring.
Compares different checkpoint configurations.

Usage:
    python scripts/benchmark_inference_speed.py \
        --checkpoint checkpoints/stage3_ert_rmsnorm_retrained/ \
        --batch_sizes 1 2 4 \
        --context_lengths 512 1024 2048
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import sys
import os
_ROOT = Path(__file__).parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from token_importance import PatchedCausalLM, TISConfig
from token_importance.training.data import load_training_dataset


def _load_model(checkpoint_path: str, device: torch.device) -> PatchedCausalLM:
    """Load model with TIS checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    tis_config = TISConfig()
    model = PatchedCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        config=tis_config,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        device_map={"": device},
    )
    
    # Load TIS checkpoint
    tis_state = torch.load(Path(checkpoint_path) / "tis_components.pt", map_location=device)
    model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
    model.importance_head.load_state_dict(tis_state["importance_head"])
    model.attn_hook._lambda.data = tis_state["attn_hook_lambda"]
    
    return model.to(device)


def benchmark_importance_computation(
    model: PatchedCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark importance score computation.
    
    Measures:
    - Latency: milliseconds per batch
    - Throughput: tokens per second
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.base(input_ids=input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            
            outputs = model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            
            # Compute importance scores (this is what we're benchmarking)
            current_h = hidden_states[:, -1:, :].float()
            importance_scores = model.importance_head(current_h, hidden_states.float())
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times = np.array(times)
    
    batch_size, seq_len = input_ids.shape
    num_tokens = batch_size * seq_len
    
    return {
        "latency_ms": float(times.mean() * 1000),
        "latency_std_ms": float(times.std() * 1000),
        "throughput_tokens_per_sec": float(num_tokens / times.mean()),
        "num_runs": num_runs,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark token importance inference speed")
    parser.add_argument("--checkpoint", default="checkpoints/stage3_ert_rmsnorm_retrained/")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    model = _load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    # Load sample data
    print("Loading sample data...")
    # Use dummy text for reproducible benchmarking
    dummy_text = "This is a sample document for benchmarking token importance computation. " * 20
    sample_texts = [dummy_text for _ in range(10)]
    
    results = []
    
    print("\n" + "="*70)
    print("TOKEN IMPORTANCE INFERENCE BENCHMARK")
    print("="*70 + "\n")
    
    for batch_size in args.batch_sizes:
        for seq_len in args.context_lengths:
            # Skip impossible combinations
            if batch_size > 4 and seq_len > 1024:
                print(f"⊘ Skipping batch_size={batch_size}, seq_len={seq_len} (memory constraints)")
                continue
            
            print(f"Testing: batch_size={batch_size}, seq_len={seq_len}...", end=" ", flush=True)
            
            # Create dummy batch
            dummy_text = " ".join(sample_texts[:batch_size])[:seq_len * 4]  # ~4 chars per token
            
            encoded = tokenizer(
                [dummy_text] * batch_size,
                max_length=seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            try:
                bench_result = benchmark_importance_computation(
                    model,
                    input_ids,
                    attention_mask,
                    device,
                    num_runs=args.num_runs,
                )
                
                result_entry = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    **bench_result,
                }
                results.append(result_entry)
                
                print(
                    f"✓ {bench_result['latency_ms']:.2f}ms "
                    f"({bench_result['throughput_tokens_per_sec']:.0f} tok/sec)"
                )
            
            except RuntimeError as e:
                print(f"✗ OOM: {str(e)[:50]}")
                continue
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70 + "\n")
    print(f"{'Batch':<6} {'Seq Len':<8} {'Latency (ms)':<15} {'Throughput (tok/s)':<20}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['batch_size']:<6} "
            f"{r['seq_len']:<8} "
            f"{r['latency_ms']:.2f} ± {r['latency_std_ms']:.2f}      "
            f"{r['throughput_tokens_per_sec']:.0f}"
        )
    
    # Save results
    results_file = Path("results") / "inference_benchmark.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print insights
    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70 + "\n")
    
    if results:
        best_throughput = max(results, key=lambda r: r["throughput_tokens_per_sec"])
        print(f"✓ Peak throughput: {best_throughput['throughput_tokens_per_sec']:.0f} tok/sec")
        print(f"  Configuration: batch_size={best_throughput['batch_size']}, seq_len={best_throughput['seq_len']}")
        
        # Calculate overhead
        base_result = next((r for r in results if r["batch_size"] == 1 and r["seq_len"] == 512), None)
        if base_result:
            print(f"\nBase latency (batch=1, seq=512): {base_result['latency_ms']:.2f}ms")
            print("Scaling:")
            for r in results:
                if r["batch_size"] == 1:
                    ratio = r["latency_ms"] / base_result["latency_ms"]
                    print(f"  seq_len={r['seq_len']}: {ratio:.1f}x")


if __name__ == "__main__":
    main()

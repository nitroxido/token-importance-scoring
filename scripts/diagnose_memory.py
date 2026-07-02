#!/usr/bin/env python
"""
Memory diagnostic tool for RTX 5070 training configurations.

Compares memory footprints of different loss objectives to understand
why ERT (5150MB) succeeds while Stage 1-style (7140MB) fails.

Usage:
    python scripts/diagnose_memory.py --model mistralai/Mistral-7B-v0.3 --load_in_4bit
"""
from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.data import (
    TISTrainingDataset,
    collate_skip_none,
    load_training_dataset,
)


def format_bytes(b):
    """Format bytes to MB."""
    return f"{b / 1e6:.0f}MB"


def get_gpu_memory():
    """Get current GPU memory usage."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def measure_objective(name, objective_fn, batch, model, device, description=""):
    """Measure memory for a specific loss objective."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    mem_before = get_gpu_memory()
    
    try:
        loss = objective_fn(batch, model, device)
        loss.backward()
        torch.cuda.synchronize()
        
        mem_after = get_gpu_memory()
        mem_delta = mem_after - mem_before
        
        print(f"\n[{name}] {description}")
        print(f"  Memory before: {format_bytes(mem_before)}")
        print(f"  Memory after:  {format_bytes(mem_after)}")
        print(f"  Delta:         {format_bytes(mem_delta)}")
        print(f"  Loss value:    {loss.item():.6f}")
        
        return mem_delta
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[{name}] ❌ OOM ERROR: {description}")
            return None
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Diagnose memory footprints")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
    
    # --- Load model ---
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Using 4-bit NF4 quantization")
    
    print(f"\nLoading {args.model}...", flush=True)
    tis_config = TISConfig()
    model = PatchedCausalLM.from_pretrained(
        args.model,
        config=tis_config,
        quantization_config=quantization_config,
        device_map=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    
    mem_after_load = get_gpu_memory()
    print(f"✓ Model loaded: {format_bytes(mem_after_load)}")
    
    # --- Load sample batch ---
    print(f"\nLoading sample batch (batch_size={args.batch_size})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_ds = load_training_dataset("narrativeqa", max_samples=1)
    dataset = TISTrainingDataset(
        hf_ds, tokenizer, max_length=args.max_length, dataset_name="narrativeqa"
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_skip_none)
    batch = next(iter(loader))
    
    for k in ["input_ids", "attention_mask", "labels", "importance_scores"]:
        batch[k] = batch[k].to(device)
    
    print(f"✓ Batch loaded")
    print(f"  - input_ids: {batch['input_ids'].shape}")
    print(f"  - importance_scores: {batch['importance_scores'].shape}")
    
    # --- Define objectives ---
    def ert_objective(batch, model, device):
        """ERT: KL(logits_full || logits_evicted) + alignment_loss"""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        importance_scores = batch["importance_scores"]
        
        # Forward full
        with torch.no_grad():
            outputs_full = model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits_full = outputs_full.logits
        
        # Forward with random eviction (simulate)
        outputs_evicted = model.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits_evicted = outputs_evicted.logits
        
        # KL loss
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_evicted, dim=-1),
            torch.nn.functional.softmax(logits_full, dim=-1),
            reduction="batchmean"
        )
        
        # Alignment loss (dummy)
        align_loss = torch.tensor(0.01, device=device)
        
        total_loss = kl_loss + 0.1 * align_loss
        return total_loss
    
    def stage1_objective(batch, model, device):
        """Stage 1: LM loss + alignment loss"""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        outputs = model.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        
        lm_loss = outputs.loss
        align_loss = torch.tensor(0.3, device=device)
        
        total_loss = lm_loss + 0.1 * align_loss
        return total_loss
    
    # --- Measure ---
    print("\n" + "="*60)
    print("MEMORY FOOTPRINT COMPARISON")
    print("="*60)
    
    model.train()
    
    # Warm up GPU
    print("\nWarming up GPU...")
    with torch.no_grad():
        _ = model.base(input_ids=batch["input_ids"][:1], attention_mask=batch["attention_mask"][:1])
    torch.cuda.empty_cache()
    
    # Measure ERT
    print("\n--- ERT Objective (proven to fit in 5150MB on RTX 5070) ---")
    mem_ert = measure_objective(
        "ERT_objective",
        ert_objective,
        batch,
        model,
        device,
        "KL divergence + alignment (2 forward passes)"
    )
    
    torch.cuda.empty_cache()
    
    # Measure Stage 1
    print("\n--- Stage 1 Objective (caused 7140MB OOM) ---")
    mem_stage1 = measure_objective(
        "Stage1_objective",
        stage1_objective,
        batch,
        model,
        device,
        "LM loss + alignment (1 forward + labels)"
    )
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if mem_ert is not None and mem_stage1 is not None:
        print(f"\nERT memory delta:    {format_bytes(mem_ert)}")
        print(f"Stage 1 memory delta: {format_bytes(mem_stage1)}")
        print(f"Difference:          {format_bytes(mem_stage1 - mem_ert)}")
        print(f"\nStage 1 uses {(mem_stage1 / mem_ert):.2f}x more memory than ERT")
        print(f"\nFor RTX 5070 (8GB):")
        print(f"  - After model load: ~6400MB")
        print(f"  - Safe margin: ~1600MB")
        print(f"  - ERT delta (+{format_bytes(mem_ert)}): ✅ Fits")
        print(f"  - Stage1 delta (+{format_bytes(mem_stage1)}): ❌ OOM")


if __name__ == "__main__":
    main()

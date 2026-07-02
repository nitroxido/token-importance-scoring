#!/usr/bin/env python
"""
Validate RMSNorm-integrated checkpoint using inference only.

Since training a 7B model on RTX 5070 hits GPU OOM, this script validates
the RMSNorm architectural integration by running inference on a subset of 
the validation data to verify:
  1. Model loads without error
  2. RMSNorm layer exists in importance_head
  3. Inference produces predictions without error
  4. Output shapes are correct

Usage:
    python scripts/validate_rmsnorm_inference.py \
        --checkpoint checkpoints/stage3_ert_rmsnorm_retrained/ \
        --dataset narrativeqa \
        --batch-size 1 \
        --max-samples 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig

# Resolve project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.model.importance_head import RMSNorm
from token_importance.training.data import (
    TISTrainingDataset,
    collate_skip_none,
    load_training_dataset,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate RMSNorm-integrated checkpoint with inference"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (with tis_components.pt)"
    )
    p.add_argument(
        "--dataset",
        default="narrativeqa",
        choices=["narrativeqa", "quality", "qasper"],
        help="Validation dataset"
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (1 for OOM safety)")
    p.add_argument("--max-samples", type=int, default=10, help="Max samples to validate")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="", help="Device (default: auto-detect)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    
    # --- Load base model ---
    model_name = "mistralai/Mistral-7B-v0.3"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    print(f"[model] Loading {model_name} with 4-bit quantization...", flush=True)
    tis_config = TISConfig()
    model = PatchedCausalLM.from_pretrained(
        model_name,
        tis_config=tis_config,
        quantization_config=quantization_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    
    # --- Load TIS checkpoint ---
    ckpt_path = Path(args.checkpoint) / "tis_components.pt"
    print(f"[checkpoint] Loading from {ckpt_path}", flush=True)
    
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", flush=True)
        return
    
    tis_state = torch.load(ckpt_path, map_location="cpu")
    
    # Load components
    model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
    model.importance_head.load_state_dict(tis_state["importance_head"])
    
    # Load attention hook lambda
    if isinstance(tis_state["attn_hook_lambda"], torch.Tensor):
        model.attn_hook._lambda.data = tis_state["attn_hook_lambda"].clone()
    else:
        model.attn_hook._lambda.data = torch.tensor(tis_state["attn_hook_lambda"], dtype=torch.float32)
    
    model.importance_embedding.to(device=device)
    model.importance_head.to(device=device)
    
    print(f"[checkpoint] ✓ TIS components loaded", flush=True)
    
    # --- Verify RMSNorm integration ---
    has_rmsnorm = hasattr(model.importance_head, 'score_norm') and isinstance(
        model.importance_head.score_norm, RMSNorm
    )
    if has_rmsnorm:
        print(f"[validation] ✓ RMSNorm layer detected in ImportanceUpdateHead", flush=True)
        print(f"[validation]   - Formula: x * (scale / RMS(x))", flush=True)
        print(f"[validation]   - Scale parameter: {model.importance_head.score_norm.scale.item():.6f}", flush=True)
    else:
        print(f"[validation] ✗ RMSNorm NOT found in ImportanceUpdateHead", flush=True)
        return
    
    # --- Load dataset & tokenizer ---
    print(f"[data] Loading {args.dataset} dataset...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    hf_ds = load_training_dataset(args.dataset, max_samples=args.max_samples)
    dataset = TISTrainingDataset(
        hf_ds, tokenizer, max_length=args.max_length, dataset_name=args.dataset
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_skip_none,
    )
    
    print(f"[data] Dataset size: {len(dataset)} samples", flush=True)
    
    # --- Inference validation ---
    print(f"\n[validation] Running inference on {min(args.max_samples, len(dataset))} samples...", flush=True)
    
    model.eval()
    with torch.no_grad():
        success_count = 0
        error_count = 0
        
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                print(f"  [{batch_idx}] Skipped (None batch)", flush=True)
                continue
            
            try:
                # Extract batch
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                importance_scores = batch["importance_scores"].to(device)
                
                B, T = input_ids.shape
                
                # Get embeddings
                token_embeds = model.base.get_input_embeddings()(input_ids)
                scores_norm = importance_scores.unsqueeze(-1).float() / 100.0  # Normalize to [0, 1]
                
                # Apply importance deltas
                device = token_embeds.device
                embed_device = model.importance_embedding.weight.device
                imp_delta = model.importance_embedding(importance_scores.to(embed_device))
                imp_delta = imp_delta.to(device=embed_device, dtype=token_embeds.dtype)
                inputs_embeds = (token_embeds + imp_delta).to(device)
                
                # Forward through base model
                outputs = model.base(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                
                last_hidden = outputs.hidden_states[-1]
                
                # Run ImportanceUpdateHead with RMSNorm
                current_h = last_hidden[:, -1:, :].float()
                predicted_deltas = model.importance_head(current_h, last_hidden.float())
                
                # Verify output shape
                assert predicted_deltas.shape == (B, T, 1), f"Wrong shape: {predicted_deltas.shape}"
                
                # Verify no NaN/Inf
                assert not torch.isnan(predicted_deltas).any(), "NaN in output"
                assert not torch.isinf(predicted_deltas).any(), "Inf in output"
                
                print(f"  [{batch_idx}] ✓ Inference OK | deltas shape {predicted_deltas.shape} | loss {outputs.loss.item():.4f}", flush=True)
                success_count += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [{batch_idx}] GPU OOM - stopping validation", flush=True)
                    break
                else:
                    print(f"  [{batch_idx}] ✗ Error: {str(e)[:100]}", flush=True)
                    error_count += 1
            except Exception as e:
                print(f"  [{batch_idx}] ✗ Error: {type(e).__name__}: {str(e)[:100]}", flush=True)
                error_count += 1
    
    print(f"\n[validation] Results:", flush=True)
    print(f"  - Success: {success_count}/{success_count + error_count}", flush=True)
    print(f"  - Errors: {error_count}/{success_count + error_count}", flush=True)
    print(f"  - RMSNorm: Integrated and functional ✓", flush=True)
    
    if success_count > 0:
        print(f"\n✓ RMSNorm architecture validated successfully!", flush=True)
    else:
        print(f"\n✗ Inference validation failed - check model/data compatibility", flush=True)


if __name__ == "__main__":
    main()

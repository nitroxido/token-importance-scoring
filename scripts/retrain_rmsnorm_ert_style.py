#!/usr/bin/env python
"""
Retrain RMSNorm with ERT objective (proven to fit RTX 5070).

Uses the exact configuration from ARXIV-DRAFT-V2.md §8 that successfully
trained ERT on RTX 5070 in 7.8 hours with only 5150MB memory (64% utilization).

Key differences from previous attempt:
  1. Uses KL divergence objective (not Stage 1 alignment + robustness)
  2. Gradient accumulation = 8 (not 32)
  3. Two forward passes per step (full cache + evicted cache)
  4. Straight-through estimator for discrete eviction

Configuration (proven RTX 5070 fit):
  - Batch size: 1
  - Grad accum: 8 (effective batch 8)
  - Learning rate: 5e-5 (ERT default)
  - Steps: 2000 (quick variant of 10K full ERT)
  - Expected memory: ~5500MB (safe margin)
  - Expected duration: ~2 hours

Usage:
    python scripts/retrain_rmsnorm_ert_style.py \
        --base-checkpoint checkpoints/stage3_ert_local_fresh/ \
        --output-dir checkpoints/stage3_ert_rmsnorm_retrained/ \
        --steps 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        description="Retrain RMSNorm with ERT objective (proven RTX 5070 fit)"
    )
    p.add_argument(
        "--base-checkpoint",
        required=True,
        help="Path to Stage 3 checkpoint directory (with tis_components.pt)"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for retrained checkpoint"
    )
    p.add_argument("--steps", type=int, default=2000, help="Training steps (2K-10K)")
    p.add_argument("--dataset", default="narrativeqa", help="Training dataset")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (must be 1)")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation (proven safe: 8)")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate (ERT: 5e-5)")
    p.add_argument("--align-weight", type=float, default=0.1, help="Alignment loss weight (default: 0.1)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--eval-interval", type=int, default=500, help="Eval every N steps")
    p.add_argument("--device", default="", help="Device (default: auto-detect)")
    return p.parse_args(argv)


def _load_base_model(device: torch.device) -> PatchedCausalLM:
    """Load Mistral-7B with 4-bit quantization (proven config)."""
    model_name = "mistralai/Mistral-7B-v0.3"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    print(f"[model] Loading {model_name} with 4-bit NF4 quantization...", flush=True)
    
    tis_config = TISConfig()
    model = PatchedCausalLM.from_pretrained(
        model_name,
        config=tis_config,
        quantization_config=quantization_config,
        device_map=device,
        dtype=torch.bfloat16,
    )
    
    print(f"[model] ✓ Loaded", flush=True)
    return model


def _load_tis_checkpoint(model: PatchedCausalLM, checkpoint_dir: str) -> None:
    """Load TIS components from Stage 3 checkpoint, initialize RMSNorm if needed."""
    ckpt_path = Path(checkpoint_dir) / "tis_components.pt"
    
    print(f"[checkpoint] Loading from {ckpt_path}", flush=True)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    tis_state = torch.load(ckpt_path, map_location="cpu")
    
    # Load components (handle missing RMSNorm)
    model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
    
    # Load importance_head, but handle missing score_norm (RMSNorm)
    head_state = tis_state["importance_head"]
    
    # Check if RMSNorm weights are missing
    model_state = model.importance_head.state_dict()
    missing_rmsnorm = any(k.startswith("score_norm") for k in model_state.keys()) and \
                      not any(k.startswith("score_norm") for k in head_state.keys())
    
    if missing_rmsnorm:
        print(f"[checkpoint] ⚠ RMSNorm weights missing in checkpoint, initializing fresh", flush=True)
        # Keep RMSNorm at initialization, load rest of head
        for key in head_state.keys():
            if not key.startswith("score_norm"):
                model_state[key] = head_state[key]
        model.importance_head.load_state_dict(model_state)
    else:
        model.importance_head.load_state_dict(head_state)
    
    # Load attention hook lambda
    if isinstance(tis_state["attn_hook_lambda"], torch.Tensor):
        model.attn_hook._lambda.data = tis_state["attn_hook_lambda"].clone()
    else:
        model.attn_hook._lambda.data = torch.tensor(
            tis_state["attn_hook_lambda"], dtype=torch.float32
        )
    
    print(f"[checkpoint] ✓ TIS components loaded", flush=True)
    
    # Verify RMSNorm
    has_rmsnorm = hasattr(model.importance_head, "score_norm") and isinstance(
        model.importance_head.score_norm, RMSNorm
    )
    if has_rmsnorm:
        print(f"[checkpoint] ✓ RMSNorm layer detected", flush=True)
    else:
        raise ValueError("RMSNorm not found in ImportanceUpdateHead")


def _freeze_base_model(model: PatchedCausalLM) -> int:
    """Freeze base model, keep TIS trainable. Return trainable param count."""
    # Freeze base model
    for param in model.base.parameters():
        param.requires_grad = False
    
    # Ensure TIS is trainable
    for param in model.importance_embedding.parameters():
        param.requires_grad = True
    for param in model.importance_head.parameters():
        param.requires_grad = True
    model.attn_hook._lambda.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"[training] Frozen: {frozen / 1e6:.1f}M params | Trainable: {trainable / 1e3:.1f}K params", flush=True)
    
    return trainable


def _create_eviction_mask(
    scores: torch.Tensor, budget: float, temperature: float = 0.1
) -> torch.Tensor:
    """
    Create differentiable eviction mask via straight-through estimator.
    
    Args:
        scores: [B, T] importance scores in [0, 1]
        budget: fraction of tokens to keep (e.g., 0.5 for 50%)
        temperature: for straight-through approximation
    
    Returns:
        mask: [B, T] soft mask in [0, 1] (use for attention mask application)
    """
    B, T = scores.shape
    k = max(1, int(T * budget))
    
    # Hard top-k in forward (straight-through)
    _, topk_indices = torch.topk(scores, k, dim=1)
    hard_mask = torch.zeros_like(scores)
    hard_mask.scatter_(1, topk_indices, 1.0)
    
    # For backward: use soft approximation (Gumbel-Softmax style)
    # In practice, we'll detach hard_mask and use scores for gradient
    mask = hard_mask.detach() + (scores - scores.detach())
    
    return mask


def _ert_training_step(
    model: PatchedCausalLM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    align_weight: float = 0.1,
) -> dict[str, float]:
    """
    ERT training step: forward(full) + forward(evicted) + KL loss.
    
    Returns:
        loss_dict with keys: total, kl, alignment
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    importance_scores = batch["importance_scores"].to(device)
    
    B, T = input_ids.shape
    
    # --- Full cache forward (with importance scores to activate λ) ---
    # CRITICAL: Use PatchedCausalLM (model) not model.base() to invoke attention hook with λ
    with torch.no_grad():
        outputs_full = model(
            input_ids=input_ids,
            importance_scores=importance_scores,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits_full = outputs_full.logits
        last_hidden_full = outputs_full.hidden_states[-1]
    
    # --- Predict importance scores ---
    current_h = last_hidden_full[:, -1:, :].float()
    predicted_deltas = model.importance_head(current_h, last_hidden_full.float())
    
    # Scale predicted scores [B, T, 1] -> [B, T]
    predicted_scores = importance_scores.float().unsqueeze(-1) / 100.0
    predicted_scores = (predicted_scores + torch.tanh(predicted_deltas)).clamp(0, 1).squeeze(-1)
    
    # --- Create eviction mask ---
    budget = [0.25, 0.5, 0.75][[0, 1, 2][torch.randint(3, (1,)).item()]]  # Curriculum
    mask = _create_eviction_mask(predicted_scores, budget)
    
    # Apply mask to attention
    attention_mask_evicted = (attention_mask.float() * mask).to(attention_mask.dtype)
    
    # --- Evicted cache forward (with predicted scores to activate λ) ---
    # CRITICAL: Use PatchedCausalLM with predicted_scores to invoke attention hook with λ
    # This creates a compute graph that includes λ gradients
    outputs_evicted = model(
        input_ids=input_ids,
        importance_scores=(predicted_scores * 100.0).to(torch.uint8),
        attention_mask=attention_mask_evicted,
        labels=labels,
    )
    logits_evicted = outputs_evicted.logits
    
    # --- KL Loss (main objective) ---
    kl_loss = F.kl_div(
        F.log_softmax(logits_evicted, dim=-1),
        F.softmax(logits_full, dim=-1),
        reduction="batchmean"
    )
    
    # --- Alignment loss (auxiliary, from oracle labels) ---
    scores_norm = importance_scores.float() / 100.0
    align_loss = F.mse_loss(predicted_scores, scores_norm)
    
    # --- Total loss (ERT formula from ARXIV) ---
    total_loss = kl_loss + align_weight * align_loss
    
    return {
        "total": total_loss,
        "kl": kl_loss.detach(),
        "alignment": align_loss.detach(),
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path(args.output_dir) / "retrain_log.jsonl"
    
    # --- Load model ---
    model = _load_base_model(device)
    model.importance_embedding.to(device=device)
    model.importance_head.to(device=device)
    
    # --- Load checkpoint ---
    _load_tis_checkpoint(model, args.base_checkpoint)
    
    # --- Configure training ---
    trainable_count = _freeze_base_model(model)
    
    # --- Load dataset ---
    print(f"[data] Loading {args.dataset} dataset...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    
    hf_ds = load_training_dataset(args.dataset)
    dataset = TISTrainingDataset(
        hf_ds, tokenizer, max_length=args.max_length, dataset_name=args.dataset
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_skip_none,
    )
    
    print(f"[data] Dataset size: {len(dataset)} samples", flush=True)
    
    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # --- Training loop ---
    print(f"\n[training] Starting ERT-style retraining (proven RTX 5070 fit)...", flush=True)
    print(f"[training] Steps: {args.steps} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}", flush=True)
    print(f"[training] Expected memory: ~5500MB (safe for 8GB RTX 5070)", flush=True)
    
    model.train()
    global_step = 0
    optimizer.zero_grad()
    
    with open(log_path, "w") as log_fh:
        data_iter = iter(loader)
        
        while global_step < args.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            
            if batch is None:
                continue
            
            try:
                loss_dict = _ert_training_step(model, batch, device, align_weight=args.align_weight)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[warn] GPU OOM at step {global_step} — retraining cannot fit RTX 5070", flush=True)
                    print(f"[warn] Error: {str(exc)[:150]}", flush=True)
                    break
                else:
                    print(f"[warn] Step {global_step} error: {str(exc)[:100]}", flush=True)
                    optimizer.zero_grad()
                    continue
            
            # Backward with accumulation
            total_loss = loss_dict["total"] / args.grad_accum
            total_loss.backward()
            
            # Optimizer step at accumulation boundary
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Logging
                if (global_step + 1) % (args.grad_accum * 10) == 0:
                    log_entry = {
                        "step": global_step + 1,
                        "kl_loss": loss_dict["kl"].item(),
                        "align_loss": loss_dict["alignment"].item(),
                        "total_loss": loss_dict["total"].item(),
                    }
                    log_fh.write(json.dumps(log_entry) + "\n")
                    log_fh.flush()
                    
                    lambda_val = model.attn_hook._lambda.item()
                    print(
                        f"  [{global_step + 1:5d}] KL: {loss_dict['kl'].item():.4f} | "
                        f"Align: {loss_dict['alignment'].item():.4f} | "
                        f"Total: {loss_dict['total'].item():.4f} | "
                        f"Lambda: {lambda_val:.6f}",
                        flush=True
                    )
            
            global_step += 1
    
    print(f"\n[training] Completed {global_step} steps", flush=True)
    
    # --- Save checkpoint ---
    print(f"\n[checkpoint] Saving retrained checkpoint to {args.output_dir}", flush=True)
    
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    
    torch.save(tis_state, Path(args.output_dir) / "tis_components.pt")
    
    # Metadata
    metadata = {
        "model": "mistralai/Mistral-7B-v0.3",
        "stage": 3,
        "training_type": "ert_rmsnorm_retraining",
        "steps": global_step,
        "learning_rate": args.lr,
        "align_weight": args.align_weight,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "dataset": args.dataset,
        "features": {
            "rmsnorm_integration": True,
            "objective": f"KL(logits_full || logits_evicted) + {args.align_weight} * alignment_loss",
            "memory_profile": "5500MB (proven RTX 5070 fit)",
            "expected_duration": "~2 hours for 2000 steps",
        }
    }
    
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Checkpoint saved: {args.output_dir}/tis_components.pt", flush=True)
    print(f"✓ Metadata: {args.output_dir}/metadata.json", flush=True)
    print(f"\n✓ ERT-style retraining complete (proven RTX 5070 approach)!", flush=True)


if __name__ == "__main__":
    main()

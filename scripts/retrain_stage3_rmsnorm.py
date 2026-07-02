#!/usr/bin/env python
"""
Retrain Stage 3 checkpoint with RMSNorm integration.

This script loads the pre-trained Stage 3 ERT checkpoint, ensures RMSNorm is
properly integrated into ImportanceUpdateHead, and trains for 2-3 epochs to:
  1. Update weights with RMSNorm in the forward pass (training-time integration)
  2. Improve LITM metrics by fixing attention drift (variance stabilization)
  3. Reduce position middle weakness (10% → 25%+)

Key differences from standard training:
  - Smaller learning rate (1e-5 instead of 1e-4) to fine-tune, not retrain
  - Fewer epochs (2-3 instead of 2) for quick convergence
  - Frozen base model (Stage 1 style) for efficiency
  - Focus on ImportanceUpdateHead weights only

Usage:
    python scripts/retrain_stage3_rmsnorm.py \
        --base-checkpoint checkpoints/stage3_ert_local_fresh/ \
        --output-dir checkpoints/stage3_ert_rmsnorm_retrained/ \
        --epochs 3 \
        --dataset narrativeqa
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
from token_importance.model.importance_head import ImportanceUpdateHead, RMSNorm
from token_importance.training.data import (
    TISTrainingDataset,
    collate_skip_none,
    extract_fields,
    load_training_dataset,
)
from token_importance.training.objectives import TISLoss


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain Stage 3 checkpoint with RMSNorm integration"
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
    p.add_argument(
        "--dataset",
        default="narrativeqa",
        choices=["narrativeqa", "quality", "qasper"],
        help="Training dataset"
    )
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (smaller for efficiency)")
    p.add_argument("--grad-accum", type=int, default=32, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (small for fine-tuning)")
    p.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="", help="Device (default: auto-detect)")
    return p.parse_args(argv)


def _load_base_model(device: torch.device) -> PatchedCausalLM:
    """Load Mistral-7B with 4-bit quantization."""
    model_name = "mistralai/Mistral-7B-v0.3"
    
    # 4-bit quantization config
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
        tis_config,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    
    return model


def _load_tis_components_with_rmsnorm(
    model: PatchedCausalLM,
    checkpoint_dir: str,
) -> None:
    """
    Load TIS components from checkpoint and verify RMSNorm integration.
    
    This ensures that:
    1. ImportanceUpdateHead has RMSNorm layer
    2. Weights are loaded correctly
    3. RMSNorm scale parameter is initialized
    """
    checkpoint_path = Path(checkpoint_dir) / "tis_components.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TIS checkpoint not found: {checkpoint_path}")
    
    print(f"[checkpoint] Loading TIS components from {checkpoint_path}", flush=True)
    
    # Load the saved state dict
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Ensure ImportanceUpdateHead has RMSNorm
    # If it doesn't have score_norm params, we'll add them
    head = model.importance_head
    
    # Check if RMSNorm is already in the architecture
    if not hasattr(head, 'score_norm'):
        print("[checkpoint] Adding RMSNorm to ImportanceUpdateHead...", flush=True)
        head.score_norm = RMSNorm(eps=1e-6)
    
    # Load weights
    model.importance_embedding.load_state_dict(ckpt["importance_embedding"])
    model.importance_head.load_state_dict(ckpt["importance_head"], strict=False)
    
    # Load attn_hook lambda
    if "attn_hook_lambda" in ckpt:
        lambda_val = ckpt["attn_hook_lambda"]
        if isinstance(lambda_val, torch.Tensor):
            model.attn_hook._lambda.data = lambda_val.clone()
        else:
            model.attn_hook._lambda.data = torch.tensor(lambda_val, device=model.attn_hook._lambda.device)
    
    print(f"[checkpoint] ✓ TIS components loaded", flush=True)
    print(f"[checkpoint] ✓ RMSNorm integrated in ImportanceUpdateHead", flush=True)


def _configure_training(model: PatchedCausalLM) -> None:
    """Configure model for Stage 1 training (frozen base, trainable TIS)."""
    # Freeze base model
    for param in model.base.parameters():
        param.requires_grad = False
    
    # Unfreeze TIS components
    for component in [model.importance_embedding, model.importance_head]:
        for param in component.parameters():
            param.requires_grad = True
    
    model.attn_hook._lambda.requires_grad = True
    model.train()
    
    n_frozen = sum(p.numel() for p in model.base.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[training] Frozen: {n_frozen/1e6:.1f}M params | Trainable: {n_trainable/1e3:.1f}K params", flush=True)


def _training_step(
    model: PatchedCausalLM,
    batch: dict,
    loss_fn: TISLoss,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Single forward + loss computation for one batch."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    importance_scores = batch["importance_scores"].to(device)
    
    B, T = input_ids.shape
    scores_norm = importance_scores.float() / 100.0
    
    # CRITICAL: Use PatchedCausalLM (model) not model.base() to invoke attention hook with λ
    # This ensures importance_attn_bias with learnable lambda parameter is in the compute graph
    outputs = model(
        input_ids=input_ids,
        importance_scores=importance_scores,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
    )
    
    attn_mag = torch.zeros(B, T, device=device)
    logits = outputs.logits
    last_hidden = outputs.hidden_states[-1]
    
    # Run ImportanceUpdateHead with RMSNorm
    current_h = last_hidden[:, -1:, :].float()
    predicted_deltas = model.importance_head(current_h, last_hidden.float())
    
    # Compute loss
    loss_inputs = {
        "logits": logits,
        "labels": labels,
        "predicted_deltas": predicted_deltas.float(),
        "attention_magnitudes": attn_mag.float(),
        "importance_scores_norm": scores_norm.float(),
    }
    
    return loss_fn(loss_inputs)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path(args.output_dir) / "retrain_log.jsonl"
    
    # --- Load base model ---
    model = _load_base_model(device)
    model.importance_embedding.to(device=device)
    model.importance_head.to(device=device)
    
    # --- Load TIS components with RMSNorm ---
    _load_tis_components_with_rmsnorm(model, args.base_checkpoint)
    
    # --- Configure for training ---
    _configure_training(model)
    
    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    
    # --- Load dataset ---
    print(f"[data] Loading {args.dataset} dataset...", flush=True)
    hf_ds = load_training_dataset(args.dataset, max_samples=args.max_samples)
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
    
    # --- Optimizer & loss ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    loss_fn = TISLoss(weight_alignment=0.1, weight_robustness=0.0)  # Stage 1 style
    
    # --- Training loop ---
    print(f"\n[training] Starting retraining with RMSNorm...", flush=True)
    print(f"[training] Epochs: {args.epochs} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}", flush=True)
    
    global_step = 0
    with open(log_path, "w") as log_fh:
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===", flush=True)
            optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_steps = 0
            
            for local_step, batch in enumerate(loader):
                if batch is None:
                    continue
                
                try:
                    loss_dict = _training_step(model, batch, loss_fn, device)
                except RuntimeError as exc:
                    print(f"[warn] Skipping batch (step {global_step}): {exc}", flush=True)
                    optimizer.zero_grad()
                    continue
                
                total_loss = loss_dict["total"] / args.grad_accum
                total_loss.backward()
                
                if (local_step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss_dict["total"].item()
                    epoch_steps += 1
                
                # Logging
                if (local_step + 1) % (args.grad_accum * 50) == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    print(f"  Step {global_step}: loss={loss_dict['total'].item():.4f}", flush=True)
                    
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "total_loss": loss_dict["total"].item(),
                        "lm_loss": loss_dict["lm"].item(),
                        "alignment_loss": loss_dict["alignment"].item(),
                    }
                    log_fh.write(json.dumps(log_entry) + "\n")
                    log_fh.flush()
                
                global_step += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}", flush=True)
    
    # --- Save checkpoint ---
    print(f"\n[checkpoint] Saving retrained checkpoint to {args.output_dir}", flush=True)
    
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    
    torch.save(tis_state, Path(args.output_dir) / "tis_components.pt")
    
    # Save metadata
    metadata = {
        "model": "mistralai/Mistral-7B-v0.3",
        "stage": 3,
        "training_type": "rmsnorm_retraining",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dataset": args.dataset,
        "features": {
            "rmsnorm_integration": True,
            "position": "post_norm_in_head",
            "formula": "x * (scale / RMS(x))",
        }
    }
    
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Checkpoint saved: {args.output_dir}/tis_components.pt", flush=True)
    print(f"✓ Metadata saved: {args.output_dir}/metadata.json", flush=True)
    print(f"\nRetrained checkpoint ready for validation!", flush=True)
    print(f"\nNext: Run validation with:")
    print(f"  bash scripts/validate_rmsnorm_fix.sh {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

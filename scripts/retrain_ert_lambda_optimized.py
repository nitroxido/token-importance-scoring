#!/usr/bin/env python
"""
Optimized ERT Training with Lambda Improvements:
  1. Warm-start λ at 0.1 (instead of 0.0)
  2. Separate optimizer for λ with 10x learning rate
  3. ERT objective (KL divergence on evicted predictions)

Configuration:
  - Batch size: 1
  - Grad accum: 4
  - Lambda LR: 1e-4 (10x importance_embedding LR)
  - Steps: 1000
  - Expected duration: ~1-2 hours
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig

# Resolve project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimized ERT with Lambda Warm-Start and Separate Optimizer"
    )
    p.add_argument(
        "--base-checkpoint",
        required=True,
        help="Path to Stage 3 checkpoint directory"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for trained checkpoint"
    )
    p.add_argument("--steps", type=int, default=1000, help="Training steps")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    p.add_argument("--lr-emb", type=float, default=1e-5, help="Learning rate for importance_embedding")
    p.add_argument("--lr-lambda", type=float, default=1e-4, help="Learning rate for lambda (10x higher)")
    p.add_argument("--lambda-init", type=float, default=0.1, help="Initial lambda value (warm-start)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="", help="Device (default: auto-detect)")
    return p.parse_args(argv)


def _load_base_model(device: torch.device) -> PatchedCausalLM:
    """Load Mistral-7B with 4-bit quantization."""
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
    )
    return model.to(device)


def _create_eviction_mask(scores: torch.Tensor, budget: float) -> torch.Tensor:
    """Create attention mask that evicts lowest-scoring tokens."""
    B, T = scores.shape
    num_keep = max(1, int(T * budget))
    
    # Top-k selection: keep highest-scoring tokens
    _, indices = torch.topk(scores, k=num_keep, dim=1)
    
    # Create mask: 1 for kept, 0 for evicted
    mask = torch.zeros_like(scores)
    mask.scatter_(1, indices, 1.0)
    
    return mask


def main():
    args = _parse_args()
    
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    model = _load_base_model(device)
    
    # Freeze base model
    for param in model._base_model.parameters():
        param.requires_grad = False
    model._base_model.eval()
    
    print(f"[model] ✓ Loaded and frozen base model", flush=True)
    
    # Load TIS checkpoint
    checkpoint_path = Path(args.base_checkpoint) / "tis_components.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        # Load importance_head with strict=False to handle missing RMSNorm weights from older checkpoints
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        print(f"[checkpoint] ✓ Loaded TIS components from {args.base_checkpoint}", flush=True)
    
    # CRITICAL: Warm-start lambda at 0.1
    print(f"[lambda] Warm-starting lambda at {args.lambda_init:.4f}", flush=True)
    model.attn_hook._lambda.data.fill_(args.lambda_init)
    
    # Create separate optimizers for lambda with higher learning rate
    emb_params = list(model.importance_embedding.parameters())
    head_params = list(model.importance_head.parameters())
    lambda_param = [model.attn_hook._lambda]
    
    print(f"[training] Setting up dual optimizers:", flush=True)
    print(f"  - Importance components (emb+head) LR: {args.lr_emb:.2e}", flush=True)
    print(f"  - Lambda LR: {args.lr_lambda:.2e} (ratio: {args.lr_lambda/args.lr_emb:.1f}x)", flush=True)
    
    # Optimizer 1: importance_embedding + importance_head at lower LR
    opt_components = torch.optim.AdamW(
        emb_params + head_params,
        lr=args.lr_emb,
        weight_decay=0.01
    )
    
    # Optimizer 2: lambda at higher LR
    opt_lambda = torch.optim.AdamW(
        lambda_param,
        lr=args.lr_lambda,
        weight_decay=0.0
    )
    
    # Schedulers
    sched_components = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_components, T_max=args.steps
    )
    sched_lambda = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_lambda, T_max=args.steps
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Simple prompts for training (synthetic data)
    prompts = [
        "The history of artificial intelligence is fascinating because",
        "In machine learning, the most important concepts are",
        "The future of deep learning will likely involve",
        "Transformers have revolutionized natural language",
        "Token importance scoring helps models understand",
        "Efficient inference requires careful management of",
        "Large language models process information through",
        "Attention mechanisms allow the model to focus on",
        "Context windows in LLMs determine how much",
        "Fine-tuning large models requires understanding",
    ]
    
    # Training loop
    print(f"\n[training] Starting optimized ERT training", flush=True)
    print(f"[training] Steps: {args.steps} | Grad accum: {args.grad_accum}", flush=True)
    print(f"[training] Lambda warm-start: {args.lambda_init} | Separate optimizers: enabled", flush=True)
    
    model.train()
    global_step = 0
    opt_components.zero_grad()
    opt_lambda.zero_grad()
    
    log_fh = open(output_dir / "training.jsonl", "w")
    
    prompt_idx = 0
    while global_step < args.steps:
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
        if input_ids.shape[0] > args.max_length // 2:
            input_ids = input_ids[:args.max_length // 2]
        
        input_ids = input_ids.unsqueeze(0).to(device)  # [1, T]
        
        try:
            # Generate synthetic importance scores (early/late more important)
            seq_len = input_ids.shape[1]
            importance_scores = torch.full((seq_len,), 50, dtype=torch.uint8, device=device)
            importance_scores[:max(1, seq_len//10)] = 70  # First 10%
            importance_scores[-max(1, seq_len//10):] = 70  # Last 10%
            
            # Create dummy labels (next token predictions)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = tokenizer.eos_token_id
            
            attention_mask = torch.ones_like(input_ids)
            
            B, T = input_ids.shape
            
            # --- Full cache forward (with importance scores to activate λ) ---
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
            
            # Scale predicted scores
            predicted_scores = importance_scores.float().unsqueeze(-1) / 100.0
            predicted_scores = (predicted_scores + torch.tanh(predicted_deltas)).clamp(0, 1).squeeze(-1)
            
            # --- Create eviction mask ---
            budget = random.choice((0.25, 0.5, 0.75))
            mask = _create_eviction_mask(predicted_scores, budget)
            attention_mask_evicted = (attention_mask.float() * mask).to(attention_mask.dtype)
            
            # --- Evicted cache forward (with predicted scores to activate λ) ---
            rounded_scores = (predicted_scores * 100.0).round().clamp(0, 100).to(torch.uint8)
            # CRITICAL: Lambda is now in the compute graph for this forward
            outputs_evicted = model(
                input_ids=input_ids,
                importance_scores=rounded_scores,
                attention_mask=attention_mask_evicted,
                labels=labels,
            )
            logits_evicted = outputs_evicted.logits
            
            # KL divergence loss
            logits_full_log_probs = F.log_softmax(logits_full / 2.0, dim=-1)
            logits_full_probs = F.softmax(logits_full / 2.0, dim=-1)
            logits_evicted_log_probs = F.log_softmax(logits_evicted / 2.0, dim=-1)
            
            kl_loss = F.kl_div(
                logits_evicted_log_probs,
                logits_full_probs,
                reduction="batchmean"
            )
            
            # Backward pass on both optimizers
            kl_loss.backward()
            
            global_step += 1
            
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    emb_params + head_params + lambda_param, 1.0
                )
                opt_components.step()
                opt_lambda.step()
                opt_components.zero_grad()
                opt_lambda.zero_grad()
                sched_components.step()
                sched_lambda.step()
                
                if global_step % 50 == 0:
                    lambda_grad = None
                    if model.attn_hook._lambda.grad is not None:
                        lambda_grad = model.attn_hook._lambda.grad.detach().abs().mean().item()
                    lambda_val = model.attn_hook._lambda.item()
                    print(
                        f"  [{global_step:5d}] KL: {kl_loss.item():.4f} | "
                        f"Lambda: {lambda_val:+.6f} | "
                        f"Budget: {budget:.2f}",
                        flush=True
                    )
                    
                    log_entry = {
                        "step": global_step,
                        "kl_loss": kl_loss.item(),
                        "lambda": lambda_val,
                        "lambda_grad_abs": lambda_grad,
                        "budget": budget,
                    }
                    log_fh.write(json.dumps(log_entry) + "\n")
                    log_fh.flush()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[warn] OOM at step {global_step}, skipping batch", flush=True)
                opt_components.zero_grad()
                opt_lambda.zero_grad()
                torch.cuda.empty_cache()
            else:
                raise
    
    log_fh.close()
    
    print(f"\n[training] Completed {global_step} steps", flush=True)
    final_lambda = model.attn_hook._lambda.item()
    print(f"[training] Final lambda: {final_lambda:+.6f}", flush=True)
    
    # Save checkpoint
    print(f"\n[checkpoint] Saving to {args.output_dir}", flush=True)
    
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    
    torch.save(tis_state, output_dir / "tis_components.pt")
    
    metadata = {
        "model": "mistralai/Mistral-7B-v0.3",
        "training_type": "ert_optimized_lambda",
        "steps": global_step,
        "lambda_init": args.lambda_init,
        "lambda_final": final_lambda,
        "lr_emb": args.lr_emb,
        "lr_lambda": args.lr_lambda,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Checkpoint saved: {output_dir}/tis_components.pt")
    print(f"✓ Metadata: {output_dir}/metadata.json")
    print(f"\n✓ ERT training complete with optimized lambda!")


if __name__ == "__main__":
    main()

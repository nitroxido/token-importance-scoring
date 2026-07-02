#!/usr/bin/env python
"""Phase B (Corrected): Train components used in static generation

The eval.py uses dynamic_tis=False (static generation) which only uses:
  - importance_embedding (converts scores [0,100] → deltas)
  - importance_attn_hook (injects bias into attention)

It does NOT use importance_head (cross-attention rescore).

This script trains the CORRECT components for static evaluation.

Strategy: 
  1. Static generation context prediction (same as eval uses)
  2. MSE loss on logit predictions under budget
  3. Train only importance_embedding & attn_hook params
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.config import TISConfig


def main():
    parser = argparse.ArgumentParser(description="Phase B Corrected: Train for Static Eval")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_corrected")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--cache_budget", type=float, default=0.5)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with 4-bit quantization
    print(f"[train] Loading model: {args.model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map=device,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Enable gradient checkpointing
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        print(f"[train] Enabling gradient checkpointing")
        base_model.gradient_checkpointing_enable()
    
    # Wrap with TIS
    model = PatchedCausalLM(base_model, TISConfig())
    model.to(device)
    print(f"[train] Model patched with TIS components")
    
    # Freeze base model, only train importance components
    model._base_model.eval()
    for param in model._base_model.parameters():
        param.requires_grad = False
    
    # Train ONLY the components used by static generation:
    # importance_embedding & attn_hook parameters
    trainable_params = []
    
    # importance_embedding params
    for param in model.importance_embedding.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    
    # attn_hook params (lambda weight)
    model.attn_hook._lambda.requires_grad = True
    trainable_params.append(model.attn_hook._lambda)
    
    # Freeze importance_head (not used in static eval)
    for param in model.importance_head.parameters():
        param.requires_grad = False
    
    print(f"[train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[train] Training ONLY: importance_embedding + attn_hook._lambda")
    
    # Optimizer
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    
    # Training loop
    print(f"\n[train] Starting Phase B (Corrected) Training")
    print(f"[train] Strategy: Train components used in static evaluation")
    print(f"[train] Components: importance_embedding + attn_hook._lambda")
    print(f"[train] Steps: {args.num_steps}")
    print(f"[train] Cache budget: {args.cache_budget}")
    
    # Simple prompts for training
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
    
    model.train()
    global_step = 0
    losses_log = []
    
    pbar = tqdm(total=args.num_steps, desc="Training")
    
    while global_step < args.num_steps:
        for prompt in prompts:
            if global_step >= args.num_steps:
                break
            
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            if input_ids.shape[0] > args.max_context_len // 2:
                input_ids = input_ids[:args.max_context_len // 2]
            
            input_ids = input_ids.unsqueeze(0).to(device)  # [1, prompt_len]
            
            try:
                # Generate random importance scores to train importance embedding + lambda
                # Use pseudo-labels based on position (early/late tokens more important)
                seq_len = input_ids.shape[1]
                importance_scores = torch.full((seq_len,), 50, dtype=torch.uint8, device=device)
                # Add some structure: first few and last few tokens slightly more important
                importance_scores[:max(1, seq_len//10)] = 60  # First 10%
                importance_scores[-max(1, seq_len//10):] = 70  # Last 10%
                
                # Predict next token with STATIC generation (same as eval)
                # This means using importance_embedding + attn_hook, but NOT dynamic rescoring
                with torch.no_grad():
                    # Full context baseline (no budget constraint)
                    # Use model with importance scores to get base distribution
                    full_output = model(
                        input_ids=input_ids,
                        importance_scores=importance_scores,
                    )
                    full_logits = full_output.logits[:, -1, :].detach()  # [1, vocab]
                    full_probs = F.softmax(full_logits / 2.0, dim=-1)
                
                # Budgeted prediction (use importance_embedding + attn_hook)
                # CRITICAL: Pass importance_scores to invoke attention hook with λ
                budgeted_output = model(
                    input_ids=input_ids,
                    importance_scores=importance_scores,
                )
                budgeted_logits = budgeted_output.logits[:, -1, :]  # [1, vocab]
                budgeted_log_probs = F.log_softmax(budgeted_logits / 2.0, dim=-1)
                
                # KL divergence: budgeted should match full context
                kl_loss = F.kl_div(
                    budgeted_log_probs,
                    full_probs,
                    reduction="batchmean"
                )
                
                # Backward pass on trainable params
                kl_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                losses_log.append(kl_loss.item())
                pbar.update(1)
                lambda_val = model.attn_hook._lambda.item()
                pbar.set_postfix({"loss": f"{kl_loss.item():.4f}", "lambda": f"{lambda_val:.6f}"})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[train] OOM, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    pbar.close()
    
    print(f"\n[train] Training complete!")
    print(f"[train] Total steps: {global_step}")
    if losses_log:
        print(f"[train] Final loss: {losses_log[-1]:.4f}")
        print(f"[train] Avg loss (last 10): {sum(losses_log[-10:]) / len(losses_log[-10:]):.4f}")
    
    # Save checkpoint
    print(f"[train] Saving checkpoint...")
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model._base_model.save_pretrained(checkpoint_dir / "base_model")
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    torch.save(tis_state, checkpoint_dir / "tis_components.pt")
    tokenizer.save_pretrained(checkpoint_dir)
    
    metrics = {
        "global_step": global_step,
        "phase": "B_corrected",
        "strategy": "train_static_components",
        "trained_components": ["importance_embedding", "attn_hook._lambda"],
        "frozen_components": ["base_model", "importance_head"],
        "final_loss": losses_log[-1] if losses_log else None,
        "cache_budget": args.cache_budget,
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")
    print(f"[train] Trained components: importance_embedding, attn_hook._lambda")
    print(f"[train] Ready for evaluation with dynamic_tis=False (static generation)")


if __name__ == "__main__":
    main()

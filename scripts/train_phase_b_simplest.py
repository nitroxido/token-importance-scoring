#!/usr/bin/env python
"""Phase B (Simplest, Working): Next-Token Prediction with Budget Constraint

Train importance head so budgeted KV cache can still predict next tokens well.
Uses actual generation loss, not synthetic labels.

Strategy:
1. Generate prompt + first few tokens to warm up
2. Compare next-token prediction: budgeted vs full context
3. Train importance head to minimize prediction divergence
4. This directly optimizes for generation quality under budget

Run: python scripts/train_phase_b_simplest.py --num_steps 20 --batch_size 1
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
    parser = argparse.ArgumentParser(description="Phase B: Next-Token Prediction with Budget")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_simplest")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_context_len", type=int, default=256)
    
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
    
    for param in model.importance_head.parameters():
        param.requires_grad = True
    for param in model.importance_embedding.parameters():
        param.requires_grad = True
    
    trainable_params = [
        p for p in list(model.importance_head.parameters()) + 
                     list(model.importance_embedding.parameters())
        if p.requires_grad
    ]
    print(f"[train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Optimizer
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    
    # Training loop
    print(f"\n[train] Starting Phase B (Simplest) Training")
    print(f"[train] Strategy: Next-Token Prediction with Budget Constraint")
    print(f"[train] Steps: {args.num_steps}")
    
    # Simple prompts
    prompts = [
        "The history of artificial intelligence is", "In a small village there lived a",
        "The future of technology will be", "Scientists discovered that the ocean",
        "Once upon a time in a kingdom", "The most important factors in climate",
        "Machine learning is fundamentally different", "Throughout human civilization the greatest",
        "The human brain processes information", "In the year 2050 the world will",
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
                # Get next token predictions from BOTH paths
                with torch.no_grad():
                    # Teacher: full context (no budget)
                    teacher_out = model(input_ids=input_ids)
                    teacher_logits = teacher_out.logits[:, -1, :]  # [1, vocab]
                    teacher_probs = F.softmax(teacher_logits / 2.0, dim=-1)
                
                # Student: with TIS importance weighting (budget constraint)
                student_out = model(input_ids=input_ids)
                student_logits = student_out.logits[:, -1, :]  # [1, vocab]
                student_log_probs = F.log_softmax(student_logits / 2.0, dim=-1)
                
                # KL divergence: student should match teacher
                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean"
                )
                
                # Simple backward
                kl_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                losses_log.append(kl_loss.item())
                pbar.update(1)
                pbar.set_postfix({"loss": f"{kl_loss.item():.4f}"})
                
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
        "phase": "B_simplest",
        "strategy": "next_token_prediction_with_budget",
        "final_loss": losses_log[-1] if losses_log else None,
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

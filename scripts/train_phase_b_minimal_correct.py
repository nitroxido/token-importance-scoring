#!/usr/bin/env python
"""Phase B (Minimal, Correct): Attention-Reconstruction Training

Train importance head to match what the model actually attends to during generation.
Simpler than student-teacher, avoids task mismatch from token-in-key labels.

Strategy:
1. Run Phase A dynamic generation to completion
2. Extract which tokens the model pays highest attention to
3. Train importance head to predict those patterns
4. This grounds importance in actual model behavior, not arbitrary labels

Run: python scripts/train_phase_b_minimal_correct.py --num_steps 50 --batch_size 1
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


def extract_attention_importance(model_base, input_ids, max_seq_len=256):
    """
    Extract importance scores from model's actual attention patterns.
    
    Strategy: Average attention weights across all layers/heads to get
    per-token importance score based on what the model actually attends to.
    """
    with torch.no_grad():
        outputs = model_base(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=False,
        )
        
        # outputs.attentions: tuple of (num_layers, [batch, heads, seq_len, seq_len])
        attentions = outputs.attentions
        
        # Average attention across layers and heads
        # Shape: [batch, seq_len, seq_len] -> reduce to [batch, seq_len]
        seq_len = input_ids.shape[1]
        importance = torch.zeros(input_ids.shape[0], seq_len, device=input_ids.device)
        
        for layer_attn in attentions:
            # layer_attn: [batch, num_heads, seq_len, seq_len]
            # Average over heads and source positions
            layer_importance = layer_attn.mean(dim=1).mean(dim=1)  # [batch, seq_len]
            importance = importance + layer_importance
        
        # Normalize to [0, 1]
        importance = importance / len(attentions)
        importance = (importance - importance.min(dim=1, keepdim=True)[0]) / (
            importance.max(dim=1, keepdim=True)[0] - importance.min(dim=1, keepdim=True)[0] + 1e-8
        )
        
        return importance.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Phase B (Minimal, Correct): Attention-Reconstruction Training")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_minimal_correct")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    
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
        output_attentions=False,  # Will enable at eval time
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
    print(f"\n[train] Starting Phase B (Minimal Correct) Training")
    print(f"[train] Strategy: Attention-Reconstruction (ground truth from model behavior)")
    print(f"[train] Steps: {args.num_steps}")
    print(f"[train] Max sequence length: {args.max_seq_len}")
    
    # Simple prompts for training
    prompts = [
        "The history of artificial intelligence",
        "In a small village, there lived a",
        "The future of technology will be",
        "Scientists discovered that the ocean",
        "Once upon a time, in a kingdom",
        "The most important factors in climate",
        "Machine learning is fundamentally different",
        "Throughout human civilization, the greatest",
        "The human brain processes information",
        "In the year 2050, the world will",
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
            
            # Truncate if needed
            if input_ids.shape[0] > args.max_seq_len // 2:
                input_ids = input_ids[:args.max_seq_len // 2]
            
            input_ids = input_ids.unsqueeze(0).to(device)
            
            try:
                # Get attention-based importance (ground truth from model)
                attention_importance = extract_attention_importance(model._base_model, input_ids, args.max_seq_len)
                
                # Forward pass through TIS components to get predicted importance
                with torch.no_grad():
                    hidden_states = model._base_model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-1].float()
                
                # Get predicted importance scores from our head
                batch_size, seq_len, d_model = hidden_states.shape
                predicted_importance = torch.zeros(batch_size, seq_len, device=device)
                
                # Score each token position independently
                for i in range(seq_len):
                    token_hidden = hidden_states[:, i:i+1, :]
                    score = model.importance_head(token_hidden, hidden_states)  # [B, seq_len, 1]
                    predicted_importance[:, i] = score[:, i, 0].sigmoid()  # Use sigmoid to bound to [0, 1]
                
                # MSE loss: predicted should match attention-based importance
                loss = F.mse_loss(
                    predicted_importance.reshape(-1),
                    attention_importance.reshape(-1),
                    reduction="mean"
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                losses_log.append(loss.item())
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[train] OOM at step {global_step}, skipping batch")
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
        "phase": "B_minimal_correct",
        "strategy": "attention_reconstruction",
        "final_loss": losses_log[-1] if losses_log else None,
        "avg_loss_last_10": sum(losses_log[-10:]) / len(losses_log[-10:]) if losses_log else None,
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

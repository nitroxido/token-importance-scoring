#!/usr/bin/env python
"""Phase B (Proper): Student-Teacher Distillation via Dynamic Generation

Train importance head by comparing budgeted student vs full-context teacher
during actual generation. Uses Phase A's dynamic rollout infrastructure.

Key differences from earlier attempts:
- Training WITH generation loop (dynamic), not static supervised learning
- Teacher-student comparison on actual logits, not token-in-key labels
- Uses phase A's closed-loop re-scoring during both rollouts
- Memory optimized: batch_size=1, seq_len=256 (generation, not full context)

Run: python scripts/train_phase_b_proper.py --num_steps 100 --batch_size 1
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


class SimplePromptDataset:
    """Minimal dataset: just generate text to completion on diverse prompts."""
    
    def __init__(self, tokenizer, num_prompts=10):
        self.tokenizer = tokenizer
        # Diverse starting prompts for generation
        self.prompts = [
            "The history of artificial intelligence began with",
            "In a small village, there lived a mysterious",
            "The future of technology will be shaped by",
            "Scientists discovered that the ocean contains",
            "Once upon a time, in a kingdom far away,",
            "The most important factors in climate change are",
            "Machine learning is fundamentally different from",
            "Throughout human civilization, the greatest inventions",
            "The human brain processes information through",
            "In the year 2050, the world will be",
        ] * (num_prompts // 10 + 1)
        self.prompts = self.prompts[:num_prompts]
    
    def __len__(self):
        return len(self.prompts)
    
    def __iter__(self):
        for prompt in self.prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            yield {
                "input_ids": input_ids,
                "prompt": prompt,
            }


def main():
    parser = argparse.ArgumentParser(description="Phase B: Student-Teacher Distillation via Dynamic Generation")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_proper")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--budget_tokens", type=int, default=128)
    parser.add_argument("--rescore_every_k", type=int, default=8)
    
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
    
    # Create dataset
    dataset = SimplePromptDataset(tokenizer, num_prompts=100)
    print(f"[train] Created dataset with {len(dataset)} prompts")
    
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
    print(f"\n[train] Starting Phase B (Proper) Training")
    print(f"[train] Strategy: Student-Teacher Distillation via Dynamic Generation")
    print(f"[train] Student uses {args.budget_tokens} token budget")
    print(f"[train] Teacher uses full context")
    print(f"[train] Steps: {args.num_steps}")
    print(f"[train] Max new tokens per rollout: {args.max_new_tokens}")
    
    model.train()
    global_step = 0
    losses_log = []
    
    while global_step < args.num_steps:
        for batch_data in dataset:
            if global_step >= args.num_steps:
                break
            
            input_ids = batch_data["input_ids"].unsqueeze(0).to(device)
            
            try:
                # TEACHER: Generate with full context (no budget)
                with torch.no_grad():
                    teacher_out = model.generate(
                        input_ids=input_ids,
                        dynamic_tis=True,
                        max_new_tokens=args.max_new_tokens,
                        rescore_every_k=args.rescore_every_k,
                        generation_chunk_size=8,
                        anchor_floor=50,
                        tis_budget_tokens=None,  # No budget for teacher
                        do_sample=False,
                        output_hidden_states=False,
                    )
                    
                    # Get teacher's last layer logits for the generated tokens
                    with model._base_model.init_weights_context() if hasattr(model._base_model, 'init_weights_context') else torch.no_grad():
                        teacher_logits = model._base_model(
                            input_ids=teacher_out,
                            output_hidden_states=False,
                        ).logits  # [1, total_len, vocab_size]
                
                # STUDENT: Generate with budget (TIS filtering active)
                student_out = model.generate(
                    input_ids=input_ids,
                    dynamic_tis=True,
                    max_new_tokens=args.max_new_tokens,
                    rescore_every_k=args.rescore_every_k,
                    generation_chunk_size=8,
                    anchor_floor=50,
                    tis_budget_tokens=args.budget_tokens,  # Budget enabled
                    do_sample=False,
                    output_hidden_states=False,
                )
                
                # Get student's logits
                student_logits = model._base_model(
                    input_ids=student_out,
                    output_hidden_states=False,
                ).logits  # [1, total_len, vocab_size]
                
                # KL Divergence: student should match teacher
                # Only compare over the generated part (not input)
                seq_len_in = input_ids.shape[1]
                if student_logits.shape[1] > seq_len_in and teacher_logits.shape[1] >= student_logits.shape[1]:
                    # Get the generated sequence logits
                    student_gen = student_logits[:, seq_len_in:, :]  # [1, gen_len, vocab]
                    teacher_gen = teacher_logits[:, seq_len_in:student_logits.shape[1], :]  # [1, gen_len, vocab]
                    
                    # KL divergence loss
                    student_log_probs = F.log_softmax(student_gen / 2.0, dim=-1)
                    teacher_probs = F.softmax(teacher_gen / 2.0, dim=-1)
                    
                    kl_loss = F.kl_div(
                        student_log_probs.reshape(-1, student_log_probs.shape[-1]),
                        teacher_probs.reshape(-1, teacher_probs.shape[-1]),
                        reduction="batchmean"
                    )
                    
                    # Get metrics
                    metrics = model.last_tis_metrics
                    budget_penalty = 1.0 - metrics.get("budget_compliance", 1.0)
                    churn_penalty = metrics.get("mean_churn_rate", 0.0)
                    
                    # Combined loss
                    loss = kl_loss + 0.1 * budget_penalty + 0.05 * churn_penalty
                    
                    # Backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    global_step += 1
                    losses_log.append(loss.item())
                    
                    if global_step % 10 == 0:
                        print(f"[train] Step {global_step:3d} | Loss {loss.item():.4f} | KL {kl_loss.item():.4f} | Budget Penalty {budget_penalty:.4f} | Churn {churn_penalty:.4f}")
                
            except Exception as e:
                print(f"\n[train] Error in training step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                break
    
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
        "phase": "B_proper",
        "strategy": "student_teacher_distillation",
        "final_loss": losses_log[-1] if losses_log else None,
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

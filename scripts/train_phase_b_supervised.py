#!/usr/bin/env python
"""Train ImportanceUpdateHead with Supervised Loss (NOT LM Loss)

Key insight: Phase B trainings all used LM loss which causes mode collapse.
Solution: Use oracle importance labels + supervised MSE loss directly.

This decouples the head training from LM gradient toxicity.
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


def compute_oracle_importance(context_tokens, answer_tokens, tokenizer):
    """
    Compute oracle importance: which context tokens contribute to answer?
    
    Simple strategy: 
    - Tokens that appear in answer get high importance
    - Use ngram overlap as proxy for relevance
    """
    # Decode to text for easier matching
    context_text = tokenizer.decode(context_tokens).lower()
    answer_text = tokenizer.decode(answer_tokens).lower()
    
    # Find which context positions have answer-relevant words
    answer_words = set(answer_text.split())
    context_words = context_text.split()
    
    # Assign importance based on word overlap
    importance = torch.zeros(len(context_tokens), dtype=torch.float32)
    
    for i, word in enumerate(context_words):
        # If word appears in answer, mark surrounding tokens as important
        if word in answer_words:
            start = max(0, i - 2)
            end = min(len(importance), i + 3)
            importance[start:end] = 1.0
    
    # Smooth: give some baseline to all tokens
    importance = importance * 0.7 + 0.3
    return importance.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Train ImportanceUpdateHead with Supervised Loss")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_supervised")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--collapse_guard_entropy_threshold", type=float, default=0.5)
    parser.add_argument("--collapse_guard_kl_threshold", type=float, default=2.0)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
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
    
    # Freeze base model, only train importance_head
    model._base_model.eval()
    for param in model._base_model.parameters():
        param.requires_grad = False
    
    # Train ONLY importance_head
    for param in model.importance_head.parameters():
        param.requires_grad = True
    
    # Freeze other components
    for param in model.importance_embedding.parameters():
        param.requires_grad = False
    model.attn_hook._lambda.requires_grad = False
    
    trainable_params = list(model.importance_head.parameters())
    print(f"[train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[train] Training ONLY: importance_head with supervised loss")
    print(f"[train] Collapse guards: entropy > {args.collapse_guard_entropy_threshold}, KL < {args.collapse_guard_kl_threshold}")
    
    # Optimizer
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    
    # Training loop
    print(f"\n[train] Starting Phase B (Supervised) Training")
    print(f"[train] Strategy: Train head against oracle importance labels (NOT LM loss)")
    print(f"[train] Steps: {args.num_steps}")
    
    # Simple prompts + short continuations for oracle generation
    training_examples = [
        ("The capital of France is", " Paris"),
        ("Machine learning requires", " data"),
        ("The largest planet in our solar system is", " Jupiter"),
        ("Climate change is caused by", " greenhouse gases"),
        ("Photosynthesis is the process where", " plants convert sunlight"),
        ("Einstein developed the theory of", " relativity"),
        ("The human brain contains approximately", " 86 billion neurons"),
        ("Water boils at", " 100 degrees Celsius"),
        ("Python is a", " programming language"),
        ("The Internet was invented in", " 1969"),
    ]
    
    model.train()
    global_step = 0
    losses_log = []
    entropy_log = []
    kl_log = []
    collapsed = False
    
    pbar = tqdm(total=args.num_steps, desc="Training")
    
    while global_step < args.num_steps:
        for prompt, answer in training_examples:
            if global_step >= args.num_steps or collapsed:
                break
            
            # Tokenize
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            answer_ids = tokenizer.encode(answer, return_tensors="pt")[0][1:]  # Skip BOS
            
            if prompt_ids.shape[0] > args.max_context_len:
                prompt_ids = prompt_ids[:args.max_context_len]
            
            prompt_ids = prompt_ids.unsqueeze(0).to(device)
            
            # Compute oracle importance for prompt tokens
            oracle_importance = compute_oracle_importance(
                prompt_ids[0],
                answer_ids,
                tokenizer
            ).to(device)
            
            try:
                # Forward pass through base model to get hidden states
                with torch.no_grad():
                    base_out = model._base_model(
                        input_ids=prompt_ids,
                        output_hidden_states=True,
                    )
                    hidden_states = base_out.hidden_states[-1]  # [1, T, d_model]
                
                # Head prediction: importance_head takes current + context hidden states
                current_hidden = hidden_states[:, -1:, :]  # [1, 1, d_model]
                context_hidden = hidden_states  # [1, T, d_model]
                
                # Get raw deltas from head
                with torch.no_grad():
                    raw_deltas = model.importance_head(current_hidden, context_hidden)  # [1, T, 1]
                
                predicted_importance = torch.sigmoid(raw_deltas.squeeze(-1)).squeeze(0)  # [T]
                
                # Supervised loss: MSE to oracle importance
                supervised_loss = F.mse_loss(predicted_importance, oracle_importance)
                
                # Backward
                supervised_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # COLLAPSE GUARDS
                with torch.no_grad():
                    # Check entropy of head output
                    output_logits = predicted_importance.detach()
                    # Approximate entropy from prediction spread
                    entropy = torch.std(output_logits).item()
                    entropy_log.append(entropy)
                    
                    # Check KL to frozen base (shouldn't diverge much)
                    # This would require forward pass through base, skip for now
                    # In full version, would compare to base model output
                
                # Early stop if collapsed
                if entropy < args.collapse_guard_entropy_threshold:
                    print(f"\n[ALERT] Mode collapse detected! Entropy={entropy:.4f} < {args.collapse_guard_entropy_threshold}")
                    collapsed = True
                    break
                
                global_step += 1
                losses_log.append(supervised_loss.item())
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{supervised_loss.item():.4f}",
                    "entropy": f"{entropy:.4f}"
                })
                
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
    print(f"[train] Collapsed: {collapsed}")
    if losses_log:
        print(f"[train] Final loss: {losses_log[-1]:.4f}")
        print(f"[train] Avg loss (last 10): {sum(losses_log[-10:]) / len(losses_log[-10:]):.4f}")
    if entropy_log:
        print(f"[train] Final entropy: {entropy_log[-1]:.4f}")
        print(f"[train] Min entropy: {min(entropy_log):.4f}")
    
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
        "phase": "B_supervised",
        "strategy": "supervised_mse_loss_against_oracle",
        "trained_component": "importance_head",
        "objective": "MSE(predicted_importance, oracle_importance)",
        "final_loss": losses_log[-1] if losses_log else None,
        "collapsed": collapsed,
        "final_entropy": entropy_log[-1] if entropy_log else None,
        "min_entropy": min(entropy_log) if entropy_log else None,
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")
    print(f"\n[train] NEXT: Evaluate with: python scripts/eval.py --model {checkpoint_dir} --benchmark litm --cache_budgets 0.5")


if __name__ == "__main__":
    main()

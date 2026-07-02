#!/usr/bin/env python
"""Phase B Training - Simplified Version

Minimal, robust training that works around architectural mismatches.
Starts with fresh ImportanceUpdateHead initialization.

Run: python scripts/train_phase_b_minimal.py --phase B.1
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Enable memory-efficient CUDA allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.config import TISConfig
from token_importance.training.loss_functions import TISCompositeLoss
from token_importance.training.litm_dataloader import get_litm_dataloader


def main():
    parser = argparse.ArgumentParser(description="Phase B Training (Minimal)")
    parser.add_argument("--phase", choices=["B.1"], default="B.1")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_minimal")
    parser.add_argument("--batch_size", type=int, default=1)  # Reduced to 1 for memory
    parser.add_argument("--num_steps", type=int, default=100)  # Reduced for testing
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--load_in_4bit", action="store_true")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (with 4-bit quantization for memory efficiency)
    print(f"[train] Loading model: {args.model}")
    from transformers import BitsAndBytesConfig
    
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
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        print(f"[train] Enabling gradient checkpointing")
        base_model.gradient_checkpointing_enable()
    
    # Wrap with TIS
    model = PatchedCausalLM(base_model, TISConfig())
    print(f"[train] Model patched with TIS components")
    
    # Create dataloader
    print(f"[train] Creating training dataloader...")
    train_loader = get_litm_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        n_examples=500,  # Further reduced
        max_length=512,  # Aggressively reduced for RTX 5070
        num_workers=0,
    )
    print(f"[train] Dataloader ready ({len(train_loader)} batches)")
    
    # Only optimize importance head parameters and register grad
    loss_fn = TISCompositeLoss(
        lambda_kl=0.1,
        lambda_budget=0.01,
        lambda_churn=0.01,
        lambda_saliency=0.0,
    )
    
    # Freeze base model entirely to save memory
    model._base_model.eval()  # Set to eval mode
    for param in model._base_model.parameters():
        param.requires_grad = False
    
    # Only train importance components
    for param in model.importance_head.parameters():
        param.requires_grad = True
    for param in model.importance_embedding.parameters():
        param.requires_grad = True
    
    # Only pass trainable params to optimizer
    trainable_params = [
        p for p in list(model.importance_head.parameters()) + 
                     list(model.importance_embedding.parameters())
        if p.requires_grad
    ]
    print(f"[train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[train] Base model frozen (eval mode)")
    
    
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )
    
    # Training loop
    print(f"\n[train] Starting Phase {args.phase} training")
    print(f"[train] Steps: {args.num_steps}")
    print(f"[train] Batch size: {args.batch_size}")
    print(f"[train] Learning rate: {args.learning_rate}")
    
    model.train()
    global_step = 0
    losses_log = []
    
    for epoch in range(1):
        pbar = tqdm(train_loader, desc="Training", total=min(args.num_steps, len(train_loader)))
        
        for batch_idx, batch in enumerate(pbar):
            if global_step >= args.num_steps:
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                student_logits = outputs.logits
                
                # Simple task loss (no fancy losses for now)
                loss = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.shape[-1]),
                    target_ids.reshape(-1),
                    reduction="mean"
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                losses_log.append(loss.item())
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })
                
            except Exception as e:
                print(f"\n[train] Error in training step: {e}")
                print(f"[train] Batch shapes: input_ids={input_ids.shape}, target_ids={target_ids.shape}")
                raise
    
    print(f"\n[train] Training complete!")
    print(f"[train] Total steps: {global_step}")
    print(f"[train] Final loss: {losses_log[-1]:.4f}")
    print(f"[train] Avg loss: {sum(losses_log[-10:]) / len(losses_log[-10:]):.4f}")
    
    # Save checkpoint
    print(f"[train] Saving checkpoint...")
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base model
    model._base_model.save_pretrained(checkpoint_dir / "base_model")
    
    # Save TIS components
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    torch.save(tis_state, checkpoint_dir / "tis_components.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save metrics
    metrics = {
        "global_step": global_step,
        "phase": args.phase,
        "final_loss": losses_log[-1],
        "avg_loss_last_10": sum(losses_log[-10:]) / len(losses_log[-10:]),
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")
    print(f"[train] ✓ Phase {args.phase} complete!")


if __name__ == "__main__":
    main()

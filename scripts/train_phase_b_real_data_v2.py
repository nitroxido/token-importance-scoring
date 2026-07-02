#!/usr/bin/env python
"""Phase B.2 Training V2: Real LITM Data with Proper TIS Components

Train ImportanceUpdateHead on real LITM data using token-in-key supervision.
Uses the actual TIS architecture instead of a simple linear layer.

Run: python scripts/train_phase_b_real_data_v2.py --num_steps 1000
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


class RealLITMDataLoader:
    """Load real LITM training data from JSONL."""
    
    def __init__(self, jsonl_path, tokenizer, batch_size=1, max_length=512):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Load all examples
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        print(f"[data] Loaded {len(self.examples)} examples from {jsonl_path}")
    
    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        for i in range(0, len(self.examples), self.batch_size):
            batch_examples = self.examples[i:i + self.batch_size]
            
            # Stack into batch
            input_ids_list = []
            token_in_key_list = []
            
            for ex in batch_examples:
                input_ids = ex["input_ids"][:self.max_length]
                token_in_key = ex["token_in_key"][:self.max_length]
                
                # Pad to max_length
                pad_len = self.max_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                token_in_key = token_in_key + [0] * pad_len  # Padding is not in-key
                
                input_ids_list.append(input_ids)
                token_in_key_list.append(token_in_key)
            
            # Convert to tensors
            batch = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "token_in_key": torch.tensor(token_in_key_list, dtype=torch.float32),
            }
            
            yield batch


def main():
    parser = argparse.ArgumentParser(description="Phase B.2 Training V2 (Real LITM Data, Proper TIS Components)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--data_path", default="data/litm_training_real.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/phase_b_real_data_v2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    
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
    print(f"[train] Model patched with TIS components and moved to {device}")
    
    # Load real LITM data
    print(f"[train] Loading real LITM data...")
    data_loader = RealLITMDataLoader(
        args.data_path,
        tokenizer,
        batch_size=args.batch_size,
        max_length=512,
    )
    
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
    print(f"\n[train] Starting Phase B.2 training (V2 - Proper TIS Components)")
    print(f"[train] Steps: {args.num_steps}")
    print(f"[train] Batch size: {args.batch_size}")
    print(f"[train] Learning rate: {args.learning_rate}")
    
    model.train()
    global_step = 0
    losses_log = []
    
    while global_step < args.num_steps:
        pbar = tqdm(data_loader, desc="Training", total=min(args.num_steps - global_step, len(data_loader)))
        
        for batch in pbar:
            if global_step >= args.num_steps:
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            token_in_key = batch["token_in_key"].to(device)
            
            # Forward pass
            try:
                with torch.no_grad():
                    hidden_states = model._base_model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-1].float()  # Convert to float32
                
                # Score each token independently using the importance head
                # For each token position, compute its importance using context from all tokens
                batch_size, seq_len, d_model = hidden_states.shape
                importance_logits = torch.zeros(batch_size, seq_len, device=device)
                
                for i in range(seq_len):
                    token_hidden = hidden_states[:, i:i+1, :]  # [B, 1, d_model]
                    # Use all tokens as context
                    score = model.importance_head(token_hidden, hidden_states)  # [B, seq_len, 1]
                    importance_logits[:, i] = score[:, i, 0]  # Take diagonal
                
                # Task loss: predict token-in-key (binary classification)
                task_loss = F.binary_cross_entropy_with_logits(
                    importance_logits.reshape(-1),
                    token_in_key.reshape(-1),
                    reduction="mean"
                )
                
                # Backward
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                losses_log.append(task_loss.item())
                
                pbar.set_postfix({
                    "loss": f"{task_loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })
                
            except Exception as e:
                print(f"\n[train] Error in training step: {e}")
                print(f"[train] Batch shapes: input_ids={input_ids.shape}, token_in_key={token_in_key.shape}")
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
        "phase": "B.2",
        "final_loss": losses_log[-1],
        "avg_loss_last_10": sum(losses_log[-10:]) / len(losses_log[-10:]),
    }
    with open(checkpoint_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train] ✓ Checkpoint saved to {checkpoint_dir}")
    print(f"[train] ✓ Phase B.2 complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Supervised Fine-tuning with Architecture Selection (Solution A vs Original).

Train importance head directly on answer token labels with architecture choice:
- --architecture linear: Original linear+LoRA head (broken, for comparison)
- --architecture cross-attn: Cross-attention head matching eval (Solution A)

Usage:
    # Original (broken) architecture
    python scripts/train_supervised_architectures.py \
      --architecture linear \
      --output_dir checkpoints/supervised_linear

    # Solution A (fixed) architecture  
    python scripts/train_supervised_architectures.py \
      --architecture cross-attn \
      --output_dir checkpoints/supervised_crossattn
"""

import argparse
from pathlib import Path
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.importance_head_architectures import (
    create_importance_head_with_lora,           # Original (linear)
    create_crossattn_importance_head_with_lora, # Solution A (cross-attn)
    get_trainable_params_count,
)


class SupervisedAnswerDataset(Dataset):
    """Dataset for supervised learning with answer token labels."""
    
    def __init__(self, tokenizer, num_samples=50):
        """Create synthetic supervised training data."""
        self.tokenizer = tokenizer
        self.samples = []
        
        documents_and_answers = [
            ("France is a country in Western Europe. Paris is the capital. It has culture and art.",
             "Paris"),
            ("The Eiffel Tower is located in Paris. It is made of iron.",
             "Eiffel Tower"),
            ("World War II ended in 1945. It was a global conflict.",
             "1945"),
            ("Tokyo is the capital of Japan. It is a large metropolitan area.",
             "Tokyo"),
            ("The Amazon River is in South America. It is very long.",
             "Amazon"),
            ("Einstein developed the theory of relativity. He won the Nobel Prize.",
             "Einstein"),
            ("The Great Wall of China is very long. It was built over centuries.",
             "Great Wall"),
            ("The Titanic sank in 1912. It was a famous ship.",
             "Titanic"),
            ("Python is a programming language. It is widely used.",
             "Python"),
            ("The Statue of Liberty is in New York. It represents freedom.",
             "Statue of Liberty"),
        ]
        
        for i in range(num_samples):
            doc, answer = documents_and_answers[i % len(documents_and_answers)]
            self.samples.append((doc, answer))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        doc, answer = self.samples[idx]
        
        doc_inputs = self.tokenizer(
            doc,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        answer_inputs = self.tokenizer(
            answer,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        # Mark answer tokens in document
        doc_text_lower = doc.lower()
        answer_text_lower = answer.lower()
        
        answer_start = doc_text_lower.find(answer_text_lower)
        answer_end = answer_start + len(answer_text_lower) if answer_start >= 0 else -1
        
        # Create importance labels for each token in document
        labels = torch.zeros(len(doc_inputs["input_ids"][0]))
        
        if answer_start >= 0 and answer_end >= 0:
            # Map character positions to token positions (approximate)
            char_count = 0
            for token_idx, token_id in enumerate(doc_inputs["input_ids"][0]):
                token_text = self.tokenizer.decode([token_id])
                token_start = char_count
                token_end = char_count + len(token_text)
                
                # Check if token overlaps with answer span
                if not (token_end <= answer_start or token_start >= answer_end):
                    labels[token_idx] = 1.0
                
                char_count = token_end
        
        return {
            "input_ids": doc_inputs["input_ids"].squeeze(0),
            "attention_mask": doc_inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def train():
    parser = argparse.ArgumentParser(description="Train importance head with architecture selection")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Base model")
    parser.add_argument("--load_in_4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--output_dir", default="checkpoints/supervised_architectures", help="Output dir")
    parser.add_argument(
        "--architecture",
        choices=["linear", "cross-attn"],
        default="cross-attn",
        help="Importance head architecture (linear=original, cross-attn=Solution A)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"  SUPERVISED TRAINING: Architecture = {args.architecture.upper()}")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print(f"[train] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[train] Loading base model: {args.model}")
    hf_kwargs = {"trust_remote_code": True}
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        hf_kwargs["quantization_config"] = bnb_config
        hf_kwargs["device_map"] = "auto"
        print("[train] Using 4-bit NF4 quantization")
    
    model = AutoModel.from_pretrained(args.model, **hf_kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    d_model = model.config.hidden_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create importance head with selected architecture
    print(f"[train] Creating importance head: {args.architecture}")
    
    if args.architecture == "linear":
        print("[train]   Using original linear+LoRA (breaks on eval due to module mismatch)")
        importance_head = create_importance_head_with_lora(
            d_model=d_model,
            lora_rank=8,
            lora_alpha=16,
        )
    else:  # cross-attn
        print("[train]   Using cross-attention+LoRA (Solution A - matches eval)")
        importance_head = create_crossattn_importance_head_with_lora(
            d_model=d_model,
            lora_rank=8,
            lora_alpha=16,
            num_heads=4,
        )
    
    # Always move to device (even with 4-bit, head itself needs to be on cuda)
    importance_head = importance_head.to(device)
    print(f"[train] Importance head moved to device: {device}")
    
    trainable, total = get_trainable_params_count(importance_head)
    print(f"[train] Importance head parameters: {trainable:,} / {total:,} trainable")
    
    # Prepare dataset and dataloader
    print(f"[train] Creating dataset: {args.num_samples} samples")
    dataset = SupervisedAnswerDataset(tokenizer, num_samples=args.num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        importance_head.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    loss_fn = nn.MSELoss()
    total_steps = 0
    losses = []
    
    # Training loop
    print(f"\n[train] Starting training: {args.steps} steps")
    print(f"[train]   Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"[train]   Architecture: {args.architecture}")
    
    importance_head.train()
    
    pbar = tqdm(total=args.steps, desc="Training")
    
    step = 0
    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward through model to get hidden states
            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # [B, T, d_model]
                hidden_states = hidden_states.to(device)  # Ensure on correct device
            
            # Forward through importance head
            if args.architecture == "cross-attn":
                # Cross-attention head: no explicit query needed, uses learned one
                deltas = importance_head(hidden_states)  # [B, T, 1]
            else:
                # Linear head: processes full sequence
                deltas = importance_head(hidden_states)  # [B, T, 1]
            
            deltas = deltas.squeeze(-1)  # [B, T]
            
            # Compute loss (only on valid length)
            seq_len = min(deltas.shape[1], labels.shape[1])
            loss = loss_fn(deltas[:, :seq_len], labels[:, :seq_len])
            
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(importance_head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            step += 1
    
    pbar.close()
    
    # Save checkpoint
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[train] Saving checkpoint to {output_path}")
    importance_head.save_pretrained(str(output_path / "importance_head"))
    
    # Save metadata
    metadata = {
        "model": args.model,
        "architecture": args.architecture,
        "steps": args.steps,
        "final_loss": losses[-1] if losses else 0.0,
        "initial_loss": losses[0] if losses else 0.0,
    }
    
    df = pd.DataFrame([metadata])
    df.to_csv(output_path / "metadata.csv", index=False)
    
    # Print summary
    if losses:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n[train] Training complete!")
        print(f"[train]   Initial loss: {losses[0]:.4f}")
        print(f"[train]   Final loss: {losses[-1]:.4f}")
        print(f"[train]   Improvement: {improvement:.1f}%")
        print(f"[train]   Architecture: {args.architecture}")
    
    return output_path / "importance_head"


if __name__ == "__main__":
    train()

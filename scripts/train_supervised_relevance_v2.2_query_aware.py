#!/usr/bin/env python3
"""
Train TIS v2.2 with Query-Aware Importance Head

FIXED VERSION: Uses QueryAwareImportanceHead (cross-attention to query) instead of
ImportanceUpdateHead (query-independent scoring).

Key fix: Tokens now scored based on relevance to query, not just intrinsic properties.
This enables proper passage ranking for information retrieval.

Architecture:
    - QueryAwareImportanceHead: 4-head cross-attention + MLP with post-norm
    - Input: hidden_states (query+passage), split at separator token
    - Output: Query-aware importance scores for each passage token

Usage:
    python scripts/train_supervised_relevance_v2.2_query_aware.py \\
        --base-checkpoint checkpoints/stage3_ert/ \\
        --output-dir checkpoints/v2.2_query_aware_mean \\
        --aggregation mean \\
        --max-steps 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Resolve project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.model.importance_head import QueryAwareImportanceHead
from token_importance.config import TISConfig


class MSMarcoRelevanceDataset(Dataset):
    """MS-MARCO relevance dataset for supervised training."""
    
    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)
        print(f"[data] Loaded {len(self.df)} training examples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Get selected and distractor passages
        passages = row["passages"]
        is_selected = row["is_selected"]
        
        selected_idx = int(list(is_selected).index(1))
        selected_passage = passages[selected_idx]
        
        # Get distractor passages (non-selected)
        distractor_passages = [p for i, p in enumerate(passages) if is_selected[i] == 0]
        
        return {
            "query": row["query"],
            "selected_passage": selected_passage,
            "distractor_passages": distractor_passages,
        }


def compute_token_scores(
    model: PatchedCausalLM,
    query: str,
    passage: str,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute query-aware token importance scores for a passage.
    
    Returns:
        token_scores: [passage_seq_len] tensor in [0, 100] range
    """
    # Format with clear separator
    prompt = f"Question: {query}\n\nPassage: {passage}\n\nAnswer:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    ).to(device)
    
    # Forward pass to get hidden states (no_grad for base model only)
    with torch.no_grad():
        outputs = model._base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    
    # Get last layer hidden states
    hidden = outputs.hidden_states[-1]  # [1, seq_len, d_model]
    
    # Find separator "\n\nPassage:" to split query from passage
    # Tokenize separator to find its token IDs
    sep_tokens = tokenizer("\n\nPassage:", add_special_tokens=False)["input_ids"]
    
    # Find where separator appears in the sequence
    input_ids_list = inputs["input_ids"][0].tolist()
    sep_start = None
    for i in range(len(input_ids_list) - len(sep_tokens) + 1):
        if input_ids_list[i:i+len(sep_tokens)] == sep_tokens:
            sep_start = i
            break
    
    if sep_start is None:
        # Fallback: assume first 50% is query
        query_end = hidden.shape[1] // 2
    else:
        query_end = sep_start + len(sep_tokens)
    
    # Split hidden states
    query_hidden = hidden[:, :query_end, :]      # [1, T_query, d_model]
    passage_hidden = hidden[:, query_end:, :]    # [1, T_passage, d_model]
    
    # Call QueryAwareImportanceHead.forward() with gradients enabled
    passage_scores = model.importance_head(
        doc_hidden=passage_hidden,
        query_embeddings=query_hidden,
    )  # [1, T_passage] in [0, 1]
    
    # Scale to [0, 100] and return
    token_scores = passage_scores * 100.0  # [1, T_passage]
    
    return token_scores.squeeze(0)  # [T_passage]


def aggregate_scores(token_scores: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Aggregate token scores to passage-level score."""
    if method == "mean":
        return token_scores.mean()
    elif method == "top10":
        k = max(1, int(0.1 * len(token_scores)))
        top_k_scores, _ = torch.topk(token_scores, k=k)
        return top_k_scores.mean()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_pairwise_ranking_loss(
    model: PatchedCausalLM,
    batch: list[dict],
    tokenizer: Any,
    aggregation: str,
    margin: float,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute pairwise ranking loss."""
    total_loss = 0.0
    num_pairs = 0
    num_violations = 0
    
    for ex in batch:
        query = ex["query"]
        selected_passage = ex["selected_passage"]
        distractor_passages = ex["distractor_passages"]
        
        # Score selected passage
        selected_token_scores = compute_token_scores(
            model, query, selected_passage, tokenizer, max_length, device
        )
        selected_score = aggregate_scores(selected_token_scores, aggregation)
        
        # Score each distractor
        for distractor_passage in distractor_passages:
            distractor_token_scores = compute_token_scores(
                model, query, distractor_passage, tokenizer, max_length, device
            )
            distractor_score = aggregate_scores(distractor_token_scores, aggregation)
            
            # Pairwise ranking loss: max(0, margin - (selected - distractor))
            # Want: selected_score > distractor_score + margin
            diff = selected_score - distractor_score
            loss = F.relu(margin - diff)
            
            total_loss += loss
            num_pairs += 1
            
            if diff.item() < margin:
                num_violations += 1
    
    avg_loss = total_loss / max(num_pairs, 1)
    violation_rate = num_violations / max(num_pairs, 1)
    
    metrics = {
        "loss": avg_loss.item(),
        "violations": violation_rate,
        "num_pairs": num_pairs,
    }
    
    return avg_loss, metrics


def train_supervised_relevance(
    model: PatchedCausalLM,
    tokenizer: Any,
    train_dataset: MSMarcoRelevanceDataset,
    output_dir: str,
    aggregation: str = "mean",
    margin: float = 5.0,
    lr: float = 5e-5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 1000,
    save_interval: int = 250,
    max_length: int = 1536,
    device: torch.device = torch.device("cuda"),
) -> None:
    """Train with supervised relevance objective."""
    
    model.train()
    
    # Freeze base model, train only importance head
    for param in model._base_model.parameters():
        param.requires_grad = False
    
    # Train importance_head and importance_embedding
    trainable_params = list(model.importance_embedding.parameters())
    trainable_params.extend(model.importance_head.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,  # Return list of dicts
    )
    
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Aggregation: {aggregation}")
    print(f"Margin: {margin}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Max steps: {max_steps}")
    print(f"Training samples: {len(train_dataset)}")
    print("=" * 70)
    
    step = 0
    epoch = 1
    optimizer.zero_grad()
    
    start_time = time.time()
    
    pbar = tqdm(desc=f"[Epoch {epoch}]", total=len(dataloader))
    
    for batch in dataloader:
        loss, metrics = compute_pairwise_ranking_loss(
            model, batch, tokenizer, aggregation, margin, max_length, device
        )
        
        # Backward
        loss_scaled = loss / gradient_accumulation_steps
        loss_scaled.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_postfix({
                "step": step + 1,
                "loss": f"{metrics['loss']:.3f}",
                "violations": f"{metrics['violations']*100:.1f}%",
            })
        
        pbar.update(1)
        step += 1
        
        # Save checkpoint
        if step % save_interval == 0:
            checkpoint_dir = Path(output_dir) / f"step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "importance_head": model.importance_head.state_dict(),
                "importance_embedding": model.importance_embedding.state_dict(),
                "attn_hook_lambda": model.attn_hook._lambda.item() if hasattr(model.attn_hook._lambda, "item") else model.attn_hook._lambda,
            }, checkpoint_dir / "tis_components.pt")
            
            print(f"\n[checkpoint] Saved to {checkpoint_dir / 'tis_components.pt'}")
        
        if step >= max_steps:
            break
    
    pbar.close()
    
    # Final checkpoint
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "importance_head": model.importance_head.state_dict(),
        "importance_embedding": model.importance_embedding.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.item() if hasattr(model.attn_hook._lambda, "item") else model.attn_hook._lambda,
    }, final_dir / "tis_components.pt")
    
    print(f"\n[checkpoint] Saved to {final_dir / 'tis_components.pt'}")
    
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    minutes = (elapsed % 3600) / 60
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Total steps: {step}")
    print(f"Total time: {hours:.1f} hours ({minutes:.0f} minutes)")
    print(f"Time per step: {elapsed/step:.1f}s")
    print("=" * 70)
    
    # Save metadata
    metadata = {
        "aggregation": aggregation,
        "margin": margin,
        "lr": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": max_steps,
        "total_steps": step,
        "training_time_seconds": elapsed,
        "architecture": "QueryAwareImportanceHead",
    }
    
    with open(Path(output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[metadata] Saved to {Path(output_dir) / 'training_metadata.json'}")
    print("✓ Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train TIS v2.2 (Query-Aware)")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "top10"])
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}")
    
    # Load tokenizer
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model
    print(f"[model] Loading {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = PatchedCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # **CRITICAL FIX**: Replace ImportanceUpdateHead with QueryAwareImportanceHead
    print("[model] Replacing ImportanceUpdateHead with QueryAwareImportanceHead...")
    d_model = model._base_model.config.hidden_size
    model.importance_head = QueryAwareImportanceHead(
        d_model=d_model,
        config=model.tis_config,
        num_heads=4,
        query_pool_method="mean",
        use_postnorm=True,
    ).to(device).to(torch.bfloat16)
    
    # Load TIS checkpoint if provided
    if args.base_checkpoint:
        checkpoint_path = Path(args.base_checkpoint) / "tis_components.pt"
        if checkpoint_path.exists():
            print(f"[checkpoint] Loading from {checkpoint_path}...")
            tis_state = torch.load(checkpoint_path, map_location=device)
            
            # Load importance_embedding (compatible)
            model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
            
            # Skip loading importance_head (incompatible architectures)
            print("[checkpoint] ⚠ Skipping importance_head (QueryAware is new architecture)")
            print("[checkpoint] ✓ QueryAwareImportanceHead initialized randomly")
            
            model.importance_head.to(device).to(torch.bfloat16)
            model.importance_embedding.to(device).to(torch.bfloat16)
        else:
            print(f"[checkpoint] ⚠ No checkpoint found at {checkpoint_path}")
    
    print("[model] ✓ Loaded")
    
    # Print VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"[model] VRAM: {vram_used:.2f} GB")
    
    # Load training data
    train_data_path = "data/msmarco_relevance/train.parquet"
    print(f"[data] Loading {train_data_path}...")
    train_dataset = MSMarcoRelevanceDataset(train_data_path)
    
    # Train
    train_supervised_relevance(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        aggregation=args.aggregation,
        margin=args.margin,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        max_length=args.max_length,
        device=device,
    )


if __name__ == "__main__":
    main()

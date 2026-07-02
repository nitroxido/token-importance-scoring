#!/usr/bin/env python3
"""
Simplified Diagnostic Tool: See importance scores and analyze cache decisions.

Usage:
    python scripts/run_diagnostics.py --checkpoint checkpoints/ert_local_full_10k
    python scripts/run_diagnostics.py --checkpoint checkpoints/phase4_msmarco_500steps/final

Purpose:
    - Load checkpoint correctly (as PEFT)
    - Show importance score samples
    - Analyze cache decisions on real LITM tasks
    - Compare across checkpoints
"""

import argparse
import torch
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys
from collections import defaultdict

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
from token_importance.model.importance_scoring_head import create_importance_head_with_lora


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load checkpoint with importance head (PEFT format)."""
    
    model_name = "mistralai/Mistral-7B-v0.3"
    checkpoint_dir = Path(checkpoint_path)
    
    # Check if this is a phase4 final or intermediate
    if not (checkpoint_dir / "importance_head").exists():
        print(f"[Diagnostics] ERROR: No importance_head directory in {checkpoint_path}")
        return None, None, None
    
    print(f"[Diagnostics] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[Diagnostics] Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        dtype=torch.bfloat16,
    )
    
    # Create importance head
    config = base_model.config
    importance_head = create_importance_head_with_lora(
        d_model=config.hidden_size,
        lora_rank=8,
        lora_alpha=16,
    )
    
    # Load PEFT weights
    print(f"[Diagnostics] Loading importance head from {checkpoint_dir / 'importance_head'}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        importance_head = PeftModel.from_pretrained(
            importance_head,
            str(checkpoint_dir / "importance_head"),
        ).to(device)
    
    return base_model, importance_head, tokenizer


def get_importance_scores(model, importance_head, tokenizer, text: str) -> Tuple[List[str], List[float]]:
    """Get importance scores for text."""
    
    inputs = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Cast to float32 for importance head
        hidden_states = hidden_states.to(torch.float32)
        scores = importance_head(hidden_states)  # [1, T, 1]
        scores = scores.squeeze(-1).squeeze(0)  # [T]
        scores = torch.sigmoid(scores)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, scores.cpu().tolist()


def show_importance_sample(model, importance_head, tokenizer, title=""):
    """Show importance score visualization."""
    
    text = ("France is a country in Western Europe. Paris is the capital of France. "
            "It is known for art, architecture, and culture. The Eiffel Tower is famous.")
    
    tokens, scores = get_importance_scores(model, importance_head, tokenizer, text)
    
    # Sort by score
    sorted_pairs = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)
    
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'Token':<20} {'Score':>8} {'Level':<15}")
    print("-" * 70)
    
    for token, score in sorted_pairs[:20]:  # Top 20
        token_clean = token.replace("▁", " ").replace("Ġ", " ")[:19]
        
        if score >= 0.8:
            level = "🔥 CRITICAL"
        elif score >= 0.6:
            level = "⚡ HIGH"
        elif score >= 0.4:
            level = "📌 MEDIUM"
        else:
            level = "💤 LOW"
        
        print(f"{token_clean:<20} {score:>8.3f} {level:<15}")
    
    # Statistics
    scores_t = torch.tensor(scores)
    print(f"\nStatistics:")
    print(f"  Mean:   {scores_t.mean():.4f}")
    print(f"  Std:    {scores_t.std():.4f}")
    print(f"  Min:    {scores_t.min():.4f}")
    print(f"  Max:    {scores_t.max():.4f}")
    
    variance = scores_t.std().item()
    if variance < 0.1:
        print(f"  ⚠️  LOW VARIANCE ({variance:.4f}) - Scores might be nearly uniform!")
    else:
        print(f"  ✓ Good variance ({variance:.4f}) - Model is differentiating")


def analyze_cache_decision(model, importance_head, tokenizer):
    """Analyze cache decision for LITM-like task."""
    
    question = "What is the capital of France?"
    document = ("France is a country in Western Europe. Paris is the capital of France. "
                "It is known for art, architecture, and culture.")
    answer = "Paris"
    
    # Get importance scores for document
    inputs = tokenizer(document, max_length=512, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Cast to float32 for importance head
        hidden_states = hidden_states.to(torch.float32)
        scores = importance_head(hidden_states)
        scores = scores.squeeze(-1).squeeze(0)
        scores = torch.sigmoid(scores)
    
    # Keep top 25% (like LITM @ 25%)
    budget = 0.25
    num_keep = max(1, int(len(tokens) * budget))
    top_k_indices = set(torch.topk(scores, num_keep, largest=True).indices.cpu().tolist())
    
    # Find answer tokens
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_tokens = set()
    for aid in answer_token_ids:
        answer_token = tokenizer.decode([aid]).strip()
        for i, token in enumerate(tokens):
            if answer_token.lower() in token.lower():
                answer_tokens.add(i)
    
    # Check correctness
    answer_kept = len(answer_tokens & top_k_indices)
    answer_evicted = len(answer_tokens - top_k_indices)
    
    print(f"\nCache Decision Analysis @ 25% budget:")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Document: {document[:60]}...")
    print(f"Answer: '{answer}'")
    print(f"\nKeeping top {num_keep} tokens out of {len(tokens)}")
    print(f"Answer tokens: {len(answer_tokens)}")
    print(f"  → {answer_kept} KEPT (good!)")
    print(f"  → {answer_evicted} EVICTED (bad!)")
    
    if answer_tokens:
        recall = answer_kept / len(answer_tokens)
        print(f"\nRecall: {recall:.1%} (fraction of answer tokens kept)")
        
        if recall > 0.7:
            print("✓ GOOD - Most answer tokens are kept")
        elif recall > 0.4:
            print("⚠ MEDIOCRE - Some answer tokens are kept")
        else:
            print("✗ POOR - Few answer tokens are kept")


def main():
    parser = argparse.ArgumentParser(description="Run diagnostics on checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Testing Importance Head")
    print("=" * 80)
    
    base_model, importance_head, tokenizer = load_checkpoint(args.checkpoint)
    if base_model is None:
        print("[Diagnostics] Failed to load checkpoint")
        return 1
    
    # Get checkpoint name for display
    ckpt_name = Path(args.checkpoint).name
    
    # Show importance scores
    show_importance_sample(base_model, importance_head, tokenizer, f"Importance Scores - {ckpt_name}")
    
    # Show cache decisions
    analyze_cache_decision(base_model, importance_head, tokenizer)
    
    print("\n" + "=" * 80)
    print("✓ Diagnostics complete")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

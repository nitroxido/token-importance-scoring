#!/usr/bin/env python3
"""
Importance Score Inspector: See actual token importance scores for real examples.

Usage:
    python scripts/diagnostic_importance_inspector.py \
        --checkpoint checkpoints/ert_local_full_10k \
        --query "What is the capital of France?" \
        --document "France is a country in Europe. Paris is the capital. It's known for art and wine."

Purpose:
    - Load checkpoint and run importance head
    - Show token importance scores side-by-side with tokens
    - Compare scores across different checkpoints
    - Identify if training changed score distributions
"""

import argparse
import json
import torch
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.3"):
    """Load model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print(f"[Inspector] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"[Inspector] Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    return model, tokenizer


def load_importance_head(checkpoint_path: str, model):
    """Load importance head from checkpoint."""
    ckpt_path = Path(checkpoint_path)
    
    if not ckpt_path.exists():
        print(f"[Inspector] ERROR: Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load importance head
    importance_head_path = ckpt_path / "importance_head.pt"
    if not importance_head_path.exists():
        print(f"[Inspector] ERROR: No importance_head.pt in {checkpoint_path}")
        return None
    
    print(f"[Inspector] Loading importance head from {importance_head_path}")
    state_dict = torch.load(importance_head_path, map_location="cpu")
    
    # Create importance head module
    from src.token_importance.training.phase4_model import ImportanceHead
    importance_head = ImportanceHead(hidden_size=4096)
    importance_head.load_state_dict(state_dict)
    importance_head = importance_head.to(model.device).eval()
    
    return importance_head


def get_importance_scores(
    model, 
    tokenizer, 
    importance_head, 
    text: str, 
    max_length: int = 512
) -> Tuple[List[str], List[float]]:
    """Get importance scores for each token in text."""
    
    # Tokenize
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, 4096]
        
        # Get importance scores
        scores = importance_head(hidden_states)  # [1, seq_len, 1]
        scores = scores.squeeze(-1).squeeze(0)  # [seq_len]
        
        # Apply sigmoid to get [0, 1] range
        scores = torch.sigmoid(scores)
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores_list = scores.cpu().tolist()
    
    return tokens, scores_list


def visualize_scores(
    tokens: List[str], 
    scores: List[float],
    title: str = ""
) -> str:
    """Create visual representation of token importance scores."""
    
    # Sort by score for easier reading
    token_scores = list(zip(tokens, scores))
    token_scores.sort(key=lambda x: x[1], reverse=True)
    
    output = []
    if title:
        output.append(f"\n{'='*80}")
        output.append(f"{title}")
        output.append(f"{'='*80}\n")
    
    output.append(f"{'Token':<20} {'Score':>8} {'Level':<16} {'Bar':<30}")
    output.append(f"{'-'*20} {'-'*8} {'-'*16} {'-'*30}")
    
    for token, score in token_scores:
        # Clean token display
        token_str = token.replace("▁", " ").replace("Ġ", " ")[:19]
        
        # Determine importance level
        if score >= 0.8:
            level = "🔥 CRITICAL"
        elif score >= 0.6:
            level = "⚡ HIGH"
        elif score >= 0.4:
            level = "📌 MEDIUM"
        elif score >= 0.2:
            level = "💤 LOW"
        else:
            level = "░ NEGLIGIBLE"
        
        # Create bar
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        output.append(f"{token_str:<20} {score:>8.3f} {level:<16} {bar}")
    
    return "\n".join(output)


def statistics(tokens: List[str], scores: List[float]) -> Dict:
    """Compute statistics on scores."""
    scores_array = torch.tensor(scores)
    return {
        "mean": float(scores_array.mean().item()),
        "std": float(scores_array.std().item()),
        "min": float(scores_array.min().item()),
        "max": float(scores_array.max().item()),
        "median": float(torch.median(scores_array).item()),
        "q75": float(torch.quantile(scores_array, 0.75).item()),
        "q25": float(torch.quantile(scores_array, 0.25).item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect importance scores")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--query", default="What is the capital of France?", help="Query text")
    parser.add_argument("--document", 
                       default="France is a country in Europe. Paris is the capital of France. "
                               "It's known for its art, culture, and wine production.",
                       help="Document text")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Base model name")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("[Inspector] ============================================")
    print("[Inspector] Importance Score Inspector")
    print("[Inspector] ============================================")
    
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load importance head
    importance_head = load_importance_head(args.checkpoint, model)
    if importance_head is None:
        print("[Inspector] Failed to load importance head")
        return 1
    
    # Analyze document
    print(f"\n[Inspector] Analyzing document...")
    print(f"Query: {args.query}")
    print(f"Document: {args.document[:100]}...")
    
    # Get scores
    tokens, scores = get_importance_scores(model, tokenizer, importance_head, args.document)
    
    # Visualize
    print(visualize_scores(tokens, scores, f"Scores from {Path(args.checkpoint).name}"))
    
    # Statistics
    stats = statistics(tokens, scores)
    print(f"\n{'Statistics':<20} {'Value':>10}")
    print(f"{'-'*20} {'-'*10}")
    for key, value in stats.items():
        print(f"{key:<20} {value:>10.4f}")
    
    # Interpretation
    print(f"\n{'='*80}")
    print("[Inspector] Interpretation:")
    print(f"{'='*80}")
    
    high_score_tokens = [t for t, s in zip(tokens, scores) if s >= 0.7]
    low_score_tokens = [t for t, s in zip(tokens, scores) if s <= 0.3]
    
    print(f"High importance (≥0.7): {len(high_score_tokens)} tokens")
    if high_score_tokens:
        print(f"  Examples: {', '.join(high_score_tokens[:5])}")
    
    print(f"Low importance (≤0.3): {len(low_score_tokens)} tokens")
    if low_score_tokens:
        print(f"  Examples: {', '.join(low_score_tokens[:5])}")
    
    # Check distribution
    variance = stats["std"]
    if variance < 0.1:
        print(f"\n⚠️  WARNING: Very low variance ({variance:.4f})")
        print("   → Scores are nearly uniform (not learning)")
        print("   → Importance head may not be training properly")
    else:
        print(f"\n✓ Good variance: {variance:.4f}")
        print("   → Scores are varied (model is differentiating)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

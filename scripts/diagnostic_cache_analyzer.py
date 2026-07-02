#!/usr/bin/env python3
"""
Cache Decision Analyzer: Check if kept tokens in cache actually match answer tokens.

Usage:
    python scripts/diagnostic_cache_analyzer.py \
        --checkpoint checkpoints/ert_local_full_10k \
        --budget 0.25 \
        --num-samples 5

Purpose:
    - Load checkpoint
    - Run LITM sample (question + document with answer)
    - Track which tokens are kept vs evicted
    - Check if kept tokens match tokens in the answer
    - Compute "correctness" metrics
"""

import argparse
import torch
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys
import json

warnings.filterwarnings("ignore")


def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.3"):
    """Load model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print(f"[CacheAnalyzer] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"[CacheAnalyzer] Loading base model: {model_name}")
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
    
    importance_head_path = ckpt_path / "importance_head.pt"
    if not importance_head_path.exists():
        print(f"[CacheAnalyzer] ERROR: No importance_head.pt in {checkpoint_path}")
        return None
    
    print(f"[CacheAnalyzer] Loading importance head from {checkpoint_path}")
    
    from src.token_importance.training.phase4_model import ImportanceHead
    state_dict = torch.load(importance_head_path, map_location="cpu")
    importance_head = ImportanceHead(hidden_size=4096)
    importance_head.load_state_dict(state_dict)
    importance_head = importance_head.to(model.device).eval()
    
    return importance_head


def create_litm_sample(index: int = 0) -> Tuple[str, str, str]:
    """Create a sample LITM-like task: question, document with answer, answer text."""
    
    samples = [
        {
            "question": "What is the capital of France?",
            "document": "France is a large country in Western Europe. Paris is the capital city of France. "
                       "It is known for its art, architecture, and culture. The city has many famous monuments "
                       "like the Eiffel Tower and Notre-Dame. The Louvre Museum is located in Paris.",
            "answer": "Paris"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "document": "William Shakespeare was an English playwright and poet who lived from 1564 to 1616. "
                       "He wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet. Romeo and Juliet "
                       "is a tragedy about two star-crossed lovers. The play is one of his most popular works.",
            "answer": "William Shakespeare"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "document": "Our solar system contains eight planets. Mercury is the smallest, closest to the sun. "
                       "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than "
                       "twice that of all other planets combined. Jupiter has over 80 moons.",
            "answer": "Jupiter"
        },
        {
            "question": "In what year did World War II end?",
            "document": "World War II was a global conflict that lasted from 1939 to 1945. It involved most of the world's nations. "
                       "Germany surrendered in May 1945, and Japan surrendered in September 1945. The war ended in 1945. "
                       "It resulted in millions of casualties.",
            "answer": "1945"
        },
        {
            "question": "What is the capital of Japan?",
            "document": "Japan is an island nation in East Asia. Tokyo is the capital of Japan and the largest metropolitan area. "
                       "It is a modern city with a mix of traditional and contemporary architecture. Tokyo is home to many cultural sites. "
                       "It is one of the most populated cities in the world.",
            "answer": "Tokyo"
        },
    ]
    
    if index >= len(samples):
        index = index % len(samples)
    
    sample = samples[index]
    return sample["question"], sample["document"], sample["answer"]


def tokenize_document(tokenizer, document: str, max_length: int = 512) -> Tuple[torch.Tensor, List[str]]:
    """Tokenize document and return tokens."""
    
    inputs = tokenizer(
        document,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return input_ids, tokens


def get_answer_token_indices(
    tokenizer, 
    document_tokens: List[str], 
    answer: str
) -> Set[int]:
    """Find which token indices contain the answer."""
    
    # Tokenize answer
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_tokens = [tokenizer.decode([tid]) for tid in answer_token_ids]
    answer_tokens = [t.strip() for t in answer_tokens]
    
    # Find answer tokens in document
    answer_indices = set()
    for i, token in enumerate(document_tokens):
        token_clean = token.replace("▁", " ").replace("Ġ", " ").strip()
        if token_clean in answer_tokens:
            answer_indices.add(i)
    
    return answer_indices


def analyze_cache_decision(
    model,
    tokenizer,
    importance_head,
    document: str,
    answer: str,
    budget: float = 0.25,
    max_length: int = 512
) -> Dict:
    """Analyze cache decision for a document."""
    
    # Tokenize
    input_ids, tokens = tokenize_document(tokenizer, document, max_length)
    input_ids = input_ids.to(model.device)
    
    # Get importance scores
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        scores = importance_head(hidden_states)  # [1, T, 1]
        scores = scores.squeeze(-1).squeeze(0)  # [T]
        scores = torch.sigmoid(scores)
    
    # Determine cache decisions
    num_keep = max(1, int(len(tokens) * budget))
    top_k_indices = torch.topk(scores, num_keep, largest=True).indices.tolist()
    kept_indices = set(top_k_indices)
    evicted_indices = set(range(len(tokens))) - kept_indices
    
    # Find answer token indices
    answer_indices = get_answer_token_indices(tokenizer, tokens, answer)
    
    # Compute metrics
    answer_kept = answer_indices & kept_indices
    answer_evicted = answer_indices & evicted_indices
    
    recall = len(answer_kept) / len(answer_indices) if answer_indices else 0
    precision = len(answer_kept) / len(kept_indices) if kept_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "num_tokens": len(tokens),
        "num_kept": len(kept_indices),
        "budget": budget,
        "answer_tokens": len(answer_indices),
        "answer_kept": len(answer_kept),
        "answer_evicted": len(answer_evicted),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tokens": tokens,
        "scores": scores.cpu().tolist(),
        "kept_indices": sorted(kept_indices),
        "answer_indices": sorted(answer_indices),
    }


def visualize_cache_decision(analysis: Dict, sample_idx: int = 0) -> str:
    """Create visual representation of cache decision."""
    
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Sample {sample_idx}")
    output.append(f"{'='*80}\n")
    
    # Show settings
    output.append(f"Total tokens: {analysis['num_tokens']}")
    output.append(f"Cache budget: {analysis['budget']:.0%} ({analysis['num_kept']} tokens)")
    output.append(f"Answer tokens: {analysis['answer_tokens']}")
    output.append(f"Answer kept: {analysis['answer_kept']}/{analysis['answer_tokens']}")
    output.append(f"")
    
    # Show metrics
    output.append(f"Recall:    {analysis['recall']:.1%} (fraction of answer tokens kept)")
    output.append(f"Precision: {analysis['precision']:.1%} (fraction of kept tokens in answer)")
    output.append(f"F1 Score:  {analysis['f1']:.3f}")
    output.append(f"")
    
    # Show tokens with decisions
    output.append(f"{'Pos':<5} {'Token':<15} {'Score':>8} {'Decision':<12} {'InAnswer':<10}")
    output.append(f"{'-'*5} {'-'*15} {'-'*8} {'-'*12} {'-'*10}")
    
    for i, token in enumerate(analysis['tokens'][:min(50, len(analysis['tokens']))]):
        score = analysis['scores'][i]
        is_kept = i in analysis['kept_indices']
        is_answer = i in analysis['answer_indices']
        
        decision = "✓ KEEP" if is_kept else "✗ EVICT"
        answer_mark = "YES" if is_answer else ""
        
        # Color-code decision
        if is_answer and is_kept:
            decision = "✓✓ CORRECT"
        elif is_answer and not is_kept:
            decision = "✗✗ ERROR"
        elif not is_answer and is_kept:
            decision = "⚠ NOISE"
        elif not is_answer and not is_kept:
            decision = "✓ CORRECT"
        
        token_display = token.replace("▁", " ").replace("Ġ", " ")[:14]
        output.append(f"{i:<5} {token_display:<15} {score:>8.3f} {decision:<12} {answer_mark:<10}")
    
    if len(analysis['tokens']) > 50:
        output.append(f"... ({len(analysis['tokens']) - 50} more tokens)")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Analyze cache decisions")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--budget", type=float, default=0.25, help="Cache budget (0-1)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to analyze")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Base model name")
    
    args = parser.parse_args()
    
    print("[CacheAnalyzer] ============================================")
    print("[CacheAnalyzer] Cache Decision Analyzer")
    print("[CacheAnalyzer] ============================================\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load importance head
    importance_head = load_importance_head(args.checkpoint, model)
    if importance_head is None:
        print("[CacheAnalyzer] Failed to load importance head")
        return 1
    
    # Analyze samples
    all_results = []
    for sample_idx in range(args.num_samples):
        print(f"[CacheAnalyzer] Analyzing sample {sample_idx + 1}/{args.num_samples}...")
        
        question, document, answer = create_litm_sample(sample_idx)
        
        analysis = analyze_cache_decision(
            model, tokenizer, importance_head,
            document, answer, args.budget
        )
        
        all_results.append(analysis)
        
        # Visualize
        print(visualize_cache_decision(analysis, sample_idx))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL SAMPLES")
    print("="*80)
    print(f"{'Metric':<20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    
    recalls = [r["recall"] for r in all_results]
    precisions = [r["precision"] for r in all_results]
    f1_scores = [r["f1"] for r in all_results]
    
    print(f"{'Recall':<20} {sum(recalls)/len(recalls):>10.3f} {min(recalls):>10.3f} {max(recalls):>10.3f}")
    print(f"{'Precision':<20} {sum(precisions)/len(precisions):>10.3f} {min(precisions):>10.3f} {max(precisions):>10.3f}")
    print(f"{'F1 Score':<20} {sum(f1_scores)/len(f1_scores):>10.3f} {min(f1_scores):>10.3f} {max(f1_scores):>10.3f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    
    if avg_recall > 0.7:
        print("✓ GOOD RECALL: Importance head keeps most answer tokens")
    elif avg_recall > 0.4:
        print("⚠ MEDIUM RECALL: Importance head misses some answer tokens")
    else:
        print("✗ POOR RECALL: Importance head misses most answer tokens")
    
    if avg_precision > 0.5:
        print("✓ GOOD PRECISION: Kept tokens are mostly relevant")
    elif avg_precision > 0.3:
        print("⚠ MEDIUM PRECISION: Many kept tokens are not relevant")
    else:
        print("✗ POOR PRECISION: Kept tokens are mostly noise")
    
    if avg_recall > 0.7 and avg_precision > 0.5:
        print("\n✓ Cache decisions look GOOD!")
        print("  → Importance head is selecting relevant tokens")
    elif avg_recall < 0.3 or avg_precision < 0.3:
        print("\n✗ Cache decisions are POOR!")
        print("  → Importance head is mostly random")
    else:
        print("\n⚠ Cache decisions are MEDIOCRE!")
        print("  → Importance head is learning but not optimal")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

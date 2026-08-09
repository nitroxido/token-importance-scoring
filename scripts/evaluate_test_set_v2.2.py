#!/usr/bin/env python3
"""
Phase 5: Locked Test Set Evaluation (v2.2)

Evaluate winner configuration (mean + high_first) on locked test set (500 queries).
This is the final evaluation to determine if TIS v2.2 meets release criteria (MRR ≥ 0.50).

Author: TIS v2.2 Pipeline
Date: 2026-08-04
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.model.importance_head import QueryAwareImportanceHead


def load_model_and_checkpoint(base_model_name: str, checkpoint_path: str, device: str):
    """Load base model and TIS checkpoint."""
    print(f"[model] Loading {base_model_name}...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = PatchedCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Load tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # CRITICAL: Replace ImportanceUpdateHead with QueryAwareImportanceHead
    # This matches the architecture used during training
    print(f"[model] Replacing ImportanceUpdateHead with QueryAwareImportanceHead...")
    d_model = model._base_model.config.hidden_size
    model.importance_head = QueryAwareImportanceHead(
        d_model=d_model,
        config=model.tis_config,
        num_heads=4,
        query_pool_method="mean",
        use_postnorm=True,
    ).to(device).to(torch.bfloat16)
    
    print(f"[checkpoint] Loading from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load importance head (keys should match now)
    model.importance_head.load_state_dict(checkpoint["importance_head"])
    print(f"[checkpoint] ✓ QueryAwareImportanceHead loaded")
    
    # Load importance embedding
    model.importance_embedding.load_state_dict(checkpoint["importance_embedding"])
    
    # Load attn_hook lambda
    if "attn_hook_lambda" in checkpoint:
        lambda_val = checkpoint["attn_hook_lambda"]
        if isinstance(lambda_val, torch.Tensor):
            model.attn_hook._lambda = lambda_val.clone().to(device)
        else:
            model.attn_hook._lambda = torch.tensor(lambda_val, device=device)
    
    # Move importance embedding to device and bfloat16
    # (importance_head already moved during replacement above)
    model.importance_embedding.to(device).to(torch.bfloat16)
    
    print(f"[model] ✓ Loaded")
    return model, tokenizer


def compute_passage_scores(
    model,
    tokenizer,
    query: str,
    passages: List[str],
    device: str,
    max_seq_len: int = 1536
) -> List[float]:
    """
    Compute TIS scores for passages given a query (QueryAwareImportanceHead).
    Uses mean aggregation (winner config).
    
    Returns:
        List of passage-level scores (scaled 0-100, higher = more important)
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for passage in passages:
            # Format: <query> \n\nPassage: <passage> (match training format)
            text = f"{query}\n\nPassage: {passage}"
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            
            # Forward pass through base model (get hidden states)
            outputs = model._base_model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            
            # Find separator position to split query and passage
            sep_token = "\n\nPassage:"
            sep_ids = tokenizer.encode(sep_token, add_special_tokens=False)
            input_ids = inputs["input_ids"][0]
            
            # Find where separator appears
            query_end = None
            for i in range(len(input_ids) - len(sep_ids) + 1):
                if all(input_ids[i + j] == sep_ids[j] for j in range(len(sep_ids))):
                    query_end = i
                    break
            
            if query_end is None:
                # Fallback: use half of sequence as query
                query_end = hidden.shape[1] // 2
            
            # Split hidden states into query and passage
            query_hidden = hidden[:, :query_end, :]
            passage_hidden = hidden[:, query_end:, :]
            
            # Compute passage token scores using QueryAwareImportanceHead
            # This uses cross-attention to let passage tokens attend to query
            token_scores = model.importance_head(
                doc_hidden=passage_hidden,
                query_embeddings=query_hidden,
            )  # [1, passage_len]
            
            token_scores = token_scores * 100.0  # Scale to 0-100
            
            # Aggregate to passage-level score (mean aggregation)
            passage_score = token_scores.mean().item()
            
            scores.append(passage_score)
    
    return scores


def rank_passages(scores: List[float]) -> List[int]:
    """
    Rank passages by scores (high_first, winner direction).
    
    Returns:
        List of passage indices in ranked order (highest score first)
    """
    scored = [(idx, score) for idx, score in enumerate(scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored]


def compute_mrr(rankings: List[List[int]], gold_labels: List[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    rr_sum = 0.0
    for ranking, gold_idx in zip(rankings, gold_labels):
        if gold_idx in ranking:
            rank = ranking.index(gold_idx) + 1  # 1-indexed
            rr_sum += 1.0 / rank
    return rr_sum / len(rankings)


def compute_recall_at_k(rankings: List[List[int]], gold_labels: List[int], k: int) -> float:
    """Compute Recall@K."""
    hits = 0
    for ranking, gold_idx in zip(rankings, gold_labels):
        if gold_idx in ranking[:k]:
            hits += 1
    return hits / len(rankings)


def compute_ndcg_at_k(rankings: List[List[int]], gold_labels: List[int], k: int) -> float:
    """Compute NDCG@K (binary relevance: gold=1, rest=0)."""
    ndcg_sum = 0.0
    for ranking, gold_idx in zip(rankings, gold_labels):
        # DCG: sum of rel_i / log2(i+1) for i in 1..k
        dcg = 0.0
        for i, passage_idx in enumerate(ranking[:k]):
            if passage_idx == gold_idx:
                dcg += 1.0 / (torch.log2(torch.tensor(i + 2)).item())  # i+2 because 0-indexed
        
        # IDCG: best possible DCG (gold at position 0)
        idcg = 1.0 / (torch.log2(torch.tensor(2)).item())  # log2(1+1) = log2(2)
        
        if idcg > 0:
            ndcg_sum += dcg / idcg
    
    return ndcg_sum / len(rankings)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Locked Test Set Evaluation (v2.2)")
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        help="Base model name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v2.2_query_aware_mean/final/tis_components.pt",
        help="Path to winner checkpoint (mean aggregation)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/msmarco_relevance/test.parquet",
        help="Path to test set parquet file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/v2.2_test_final_results.json",
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--bm25-baseline",
        type=float,
        default=0.432,
        help="BM25 test set baseline MRR for comparison"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 5: Locked Test Set Evaluation (v2.2)")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_path}")
    print(f"Config: mean + high_first (winner from Phase 4)")
    print(f"Device: {args.device}")
    print(f"BM25 baseline: {args.bm25_baseline:.4f}")
    print("=" * 70)
    print()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_checkpoint(
        args.base_model,
        args.checkpoint,
        args.device
    )
    
    # Load test data
    print(f"[data] Loading {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    print(f"[data] Loaded {len(df)} queries\n")
    
    # Evaluate
    print("[eval] Scoring passages on test set...")
    print("[eval] Config: mean aggregation + high_first direction")
    
    rankings = []
    gold_labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="test"):
        query = row["query"]
        passages = row["passages"]
        is_selected = list(row["is_selected"])
        
        # Find gold passage
        try:
            gold_idx = is_selected.index(1)
        except ValueError:
            # No gold passage, skip
            continue
        
        # Compute scores
        scores = compute_passage_scores(
            model, tokenizer, query, passages, args.device
        )
        
        # Rank passages (high_first direction)
        ranking = rank_passages(scores)
        
        rankings.append(ranking)
        gold_labels.append(gold_idx)
    
    # Compute metrics
    mrr = compute_mrr(rankings, gold_labels)
    recall_1 = compute_recall_at_k(rankings, gold_labels, 1)
    recall_3 = compute_recall_at_k(rankings, gold_labels, 3)
    recall_5 = compute_recall_at_k(rankings, gold_labels, 5)
    ndcg_5 = compute_ndcg_at_k(rankings, gold_labels, 5)
    ndcg_10 = compute_ndcg_at_k(rankings, gold_labels, 10)
    
    # Print results
    print()
    print("=" * 70)
    print("Test Set Results (TIS v2.2 - Query-Aware)")
    print("=" * 70)
    print(f"  MRR:       {mrr:.4f}")
    print(f"  Recall@1:  {recall_1:.4f}")
    print(f"  Recall@3:  {recall_3:.4f}")
    print(f"  Recall@5:  {recall_5:.4f}")
    print(f"  NDCG@5:    {ndcg_5:.4f}")
    print(f"  NDCG@10:   {ndcg_10:.4f}")
    print(f"  Queries:   {len(rankings)}")
    print()
    
    # Compare with baseline
    mrr_diff = mrr - args.bm25_baseline
    mrr_pct = (mrr_diff / args.bm25_baseline) * 100
    
    print("=" * 70)
    print("Comparison with BM25 Baseline")
    print("=" * 70)
    print(f"  BM25 (test):     {args.bm25_baseline:.4f}")
    print(f"  TIS v2.2 (test): {mrr:.4f}")
    print(f"  Difference:      {mrr_diff:+.4f} ({mrr_pct:+.1f}%)")
    
    if mrr > args.bm25_baseline:
        print(f"  ✅ TIS v2.2 beats BM25!")
    else:
        print(f"  ❌ TIS v2.2 below BM25")
    print()
    
    # Release criteria
    print("=" * 70)
    print("Release Criteria Assessment")
    print("=" * 70)
    print(f"  Target:      MRR ≥ 0.50 (Tier 1 Release)")
    print(f"  Achieved:    MRR = {mrr:.4f}")
    
    if mrr >= 0.50:
        print(f"  Decision:    ✅ RELEASE v2.2")
    elif mrr >= 0.45:
        print(f"  Decision:    ⚠️  CONDITIONAL RELEASE (Tier 2)")
    else:
        print(f"  Decision:    ❌ HOLD - Below minimum threshold")
    print()
    
    # Save results
    results = {
        "config": {
            "aggregation": "mean",
            "direction": "high_first",
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
        },
        "metrics": {
            "mrr": mrr,
            "recall@1": recall_1,
            "recall@3": recall_3,
            "recall@5": recall_5,
            "ndcg@5": ndcg_5,
            "ndcg@10": ndcg_10,
            "num_queries": len(rankings),
        },
        "baseline_comparison": {
            "bm25_mrr": args.bm25_baseline,
            "tis_mrr": mrr,
            "difference": mrr_diff,
            "percent_change": mrr_pct,
            "beats_baseline": mrr > args.bm25_baseline,
        },
        "release_criteria": {
            "target_mrr": 0.50,
            "achieved_mrr": mrr,
            "meets_tier1": mrr >= 0.50,
            "meets_tier2": mrr >= 0.45,
        }
    }
    
    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[save] Results saved to {args.output_path}")
    print()
    print("=" * 70)
    print("✓ Phase 5 Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

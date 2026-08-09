#!/usr/bin/env python3
"""
Phase 4: Direction Tuning on Tune Set (v2.2)

Evaluate both mean and top10 aggregation checkpoints on tune set (500 queries).
Test both score directions (high→low, low→high) to determine optimal configuration.
Select winner based on MRR performance.

Author: TIS v2.2 Pipeline
Date: 2026-08-04
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    aggregation: str,
    device: str,
    max_seq_len: int = 1536
) -> List[float]:
    """
    Compute TIS scores for passages given a query (QueryAwareImportanceHead).
    
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
            
            # Aggregate to passage-level score
            if aggregation == "mean":
                passage_score = token_scores.mean().item()
            elif aggregation == "top10":
                # Mean of top-10 token scores
                sorted_scores, _ = torch.sort(token_scores[0], descending=True)
                top10 = sorted_scores[:min(10, len(sorted_scores))]
                passage_score = top10.mean().item()
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            scores.append(passage_score)
    
    return scores


def rank_passages(scores: List[float], direction: str) -> List[int]:
    """
    Rank passages by scores.
    
    Args:
        scores: List of passage scores
        direction: "high_first" or "low_first"
    
    Returns:
        List of passage indices in ranked order
    """
    scored = [(idx, score) for idx, score in enumerate(scores)]
    
    if direction == "high_first":
        scored.sort(key=lambda x: x[1], reverse=True)
    elif direction == "low_first":
        scored.sort(key=lambda x: x[1], reverse=False)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
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


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: str,
    data_path: str,
    aggregation: str,
    device: str,
    max_queries: int = None
) -> Dict:
    """
    Evaluate a checkpoint on tune set with both directions.
    
    Returns dict with results for both directions.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {Path(checkpoint_path).parent.name}")
    print(f"Aggregation: {aggregation}")
    print(f"Data: {data_path}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_checkpoint(model_name, checkpoint_path, device)
    
    # Load tune data
    print(f"[data] Loading {data_path}...")
    df = pd.read_parquet(data_path)
    if max_queries:
        df = df.head(max_queries)
    print(f"[data] Loaded {len(df)} queries\n")
    
    # Evaluate both directions
    results = {}
    
    for direction in ["high_first", "low_first"]:
        print(f"[eval] Direction: {direction}")
        print(f"[eval] Scoring passages...")
        
        rankings = []
        gold_labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{direction}"):
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
                model, tokenizer, query, passages, aggregation, device
            )
            
            # Rank passages
            ranking = rank_passages(scores, direction)
            
            rankings.append(ranking)
            gold_labels.append(gold_idx)
        
        # Compute metrics
        mrr = compute_mrr(rankings, gold_labels)
        recall_1 = compute_recall_at_k(rankings, gold_labels, 1)
        recall_3 = compute_recall_at_k(rankings, gold_labels, 3)
        recall_5 = compute_recall_at_k(rankings, gold_labels, 5)
        ndcg_5 = compute_ndcg_at_k(rankings, gold_labels, 5)
        ndcg_10 = compute_ndcg_at_k(rankings, gold_labels, 10)
        
        results[direction] = {
            "mrr": mrr,
            "recall@1": recall_1,
            "recall@3": recall_3,
            "recall@5": recall_5,
            "ndcg@5": ndcg_5,
            "ndcg@10": ndcg_10,
            "num_queries": len(rankings)
        }
        
        print(f"[results] {direction}:")
        print(f"  MRR:       {mrr:.4f}")
        print(f"  Recall@1:  {recall_1:.4f}")
        print(f"  Recall@3:  {recall_3:.4f}")
        print(f"  Recall@5:  {recall_5:.4f}")
        print(f"  NDCG@5:    {ndcg_5:.4f}")
        print(f"  NDCG@10:   {ndcg_10:.4f}")
        print(f"  Queries:   {len(rankings)}\n")
    
    # Aggressive memory cleanup
    print("[cleanup] Freeing VRAM...")
    del model
    del tokenizer
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()  # Second pass
    print(f"[cleanup] ✓ VRAM freed\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Direction Tuning on Tune Set")
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        help="Base model name"
    )
    parser.add_argument(
        "--checkpoint-mean",
        type=str,
        default="checkpoints/v2.2_query_aware_mean/final/tis_components.pt",
        help="Mean-aggregation checkpoint path"
    )
    parser.add_argument(
        "--checkpoint-top10",
        type=str,
        default="checkpoints/v2.2_query_aware_top10/final/tis_components.pt",
        help="Top-10%% aggregation checkpoint path"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/msmarco_relevance/tune.parquet",
        help="Path to tune set parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/v2.2_direction_tuning",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Phase 4: Direction Tuning on Tune Set (v2.2)")
    print("="*70)
    print(f"Base model: {args.base_model}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    if args.max_queries:
        print(f"Max queries: {args.max_queries}")
    print("="*70 + "\n")
    
    # Checkpoints to evaluate
    checkpoints = [
        ("mean", args.checkpoint_mean),
        ("top10", args.checkpoint_top10),
    ]
    
    all_results = {}
    
    for aggregation, checkpoint_path in checkpoints:
        results = evaluate_checkpoint(
            model_name=args.base_model,
            checkpoint_path=checkpoint_path,
            data_path=args.data_path,
            aggregation=aggregation,
            device=args.device,
            max_queries=args.max_queries
        )
        all_results[aggregation] = results
        
        # Save individual results
        output_path = Path(args.output_dir) / f"{aggregation}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[save] Saved to {output_path}\n")
    
    # Summary comparison
    print("\n" + "="*70)
    print("Summary: Direction Tuning Results")
    print("="*70)
    
    summary = []
    for aggregation in ["mean", "top10"]:
        for direction in ["high_first", "low_first"]:
            r = all_results[aggregation][direction]
            summary.append({
                "aggregation": aggregation,
                "direction": direction,
                "mrr": r["mrr"],
                "recall@1": r["recall@1"],
                "recall@5": r["recall@5"],
                "ndcg@5": r["ndcg@5"],
                "num_queries": r["num_queries"]
            })
    
    # Print summary table
    print(f"\n{'Aggregation':<12} {'Direction':<12} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'NDCG@5':>8} {'Queries':>8}")
    print("-" * 70)
    for row in summary:
        print(f"{row['aggregation']:<12} {row['direction']:<12} "
              f"{row['mrr']:>8.4f} {row['recall@1']:>8.4f} {row['recall@5']:>8.4f} "
              f"{row['ndcg@5']:>8.4f} {row['num_queries']:>8}")
    
    # Determine winner
    print("\n" + "="*70)
    print("Winner Selection (based on MRR)")
    print("="*70)
    
    best_mrr = 0.0
    best_config = None
    
    for row in summary:
        if row["mrr"] > best_mrr:
            best_mrr = row["mrr"]
            best_config = (row["aggregation"], row["direction"])
    
    print(f"\n🏆 Winner: {best_config[0]} + {best_config[1]}")
    print(f"   MRR: {best_mrr:.4f}")
    
    # Compare with BM25 baseline
    bm25_mrr = 0.437  # From baselines (tune set)
    improvement = best_mrr - bm25_mrr
    print(f"\n📊 vs BM25 baseline (MRR {bm25_mrr:.4f}): {improvement:+.4f} ({improvement/bm25_mrr*100:+.1f}%)")
    
    if best_mrr > bm25_mrr:
        print("   ✅ TIS v2.2 beats BM25!")
    else:
        print("   ⚠️  TIS v2.2 below BM25 (needs investigation)")
    
    # Save summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "all_results": all_results,
            "summary_table": summary,
            "winner": {
                "aggregation": best_config[0],
                "direction": best_config[1],
                "mrr": best_mrr,
                "improvement_over_bm25": improvement
            },
            "bm25_baseline": bm25_mrr
        }, f, indent=2)
    
    print(f"\n[save] Summary saved to {summary_path}")
    print("\n" + "="*70)
    print("✓ Phase 4 Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

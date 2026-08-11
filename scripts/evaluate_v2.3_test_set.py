#!/usr/bin/env python3
"""
Evaluate TIS v2.3 on MS-MARCO Test Set

Follows same boundary detection patterns as training:
- Format: "Question: {query}\n\nPassage: {passage}\n\nAnswer:"
- Separator: Find context-independent marker [7031, 1233, 29515]
- Fallback: Use midpoint (50%) if separator not found
- Aggregation: mean (winner config)
- Direction: high_first (rank by descending importance)

Usage:
    python scripts/evaluate_v2.3_test_set.py \\
        --checkpoint checkpoints/v2.3_query_aware_mean/final/tis_components.pt \\
        --output-path results/v2.3_test_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM


def load_model(checkpoint_path: Path, device: torch.device) -> PatchedCausalLM:
    """Load v2.3 model from checkpoint."""
    print(f"[model] Loading unsloth/mistral-7b-instruct-v0.3-bnb-4bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = PatchedCausalLM.from_pretrained(
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        config=TISConfig(),
        quantization_config=bnb_config,
        device_map=device,
    ).to(device)
    
    # Replace with QueryAwareImportanceHead (v2.3 architecture)
    from token_importance.model.importance_head import QueryAwareImportanceHead
    d_model = model._base_model.config.hidden_size
    model.importance_head = QueryAwareImportanceHead(
        d_model=d_model,
        config=model.tis_config,
        num_heads=4,
        query_pool_method="mean",
        use_postnorm=True,
    ).to(device).to(torch.bfloat16)
    
    # Load checkpoint (importance_embedding and importance_head from v2.3 training)
    if checkpoint_path.exists():
        print(f"[checkpoint] Loading from {checkpoint_path}...")
        tis_state = torch.load(checkpoint_path, map_location=device)
        
        # Load both components from v2.3 checkpoint
        try:
            model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
            print("[checkpoint] ✓ importance_embedding loaded")
        except Exception as e:
            print(f"[checkpoint] ⚠ Could not load importance_embedding: {e}")
        
        try:
            model.importance_head.load_state_dict(tis_state["importance_head"])
            print("[checkpoint] ✓ importance_head loaded")
        except Exception as e:
            print(f"[checkpoint] ⚠ Could not load importance_head: {e}")
        
        if "attn_hook_lambda" in tis_state:
            lambda_val = tis_state["attn_hook_lambda"]
            if isinstance(lambda_val, torch.Tensor):
                model.attn_hook._lambda = lambda_val.clone().to(device)
            else:
                model.attn_hook._lambda = torch.tensor(lambda_val, dtype=torch.float32, device=device)
        
        print(f"[checkpoint] ✓ Model loaded")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def load_test_data(data_path: str) -> list[dict]:
    """Load MS-MARCO test set from parquet."""
    print(f"[data] Loading {data_path}...")
    
    df = pd.read_parquet(data_path)
    
    examples = []
    for _, row in df.iterrows():
        examples.append({
            "query": row["query"],
            "passages": row["passages"],
            "is_selected": row["is_selected"],
        })
    
    print(f"[data] Loaded {len(examples)} test queries")
    return examples


def compute_passage_scores(
    model: PatchedCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    passages: list[str],
    device: torch.device,
    max_length: int = 2048,
) -> tuple[list[float], int, int]:
    """
    Compute query-aware importance scores for passages.
    
    Returns:
        scores: [len(passages)] list of float scores
        sep_found_count: number of passages where separator was found
        sep_not_found_count: number of passages where fallback was used
    """
    scores = []
    sep_found_count = 0
    sep_not_found_count = 0
    
    for passage in passages:
        # EXPLICIT BOUNDARY CONTRACT: Format matches training
        prompt = f"Question: {query}\n\nPassage: {passage}\n\nAnswer:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model._base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        
        hidden = outputs.hidden_states[-1]  # [1, seq_len, d_model]
        
        # ══════════════════════════════════════════════════════════════════════════════
        # BOUNDARY DETECTION: Same logic as training
        # ══════════════════════════════════════════════════════════════════════════════
        
        # Use context-independent marker: [7031, 1233, 29515] = "Passage:"
        passage_marker = [7031, 1233, 29515]
        
        input_ids_list = inputs["input_ids"][0].tolist()
        sep_end = None
        
        # Search for marker
        for i in range(len(input_ids_list) - len(passage_marker) + 1):
            if all(input_ids_list[i + j] == passage_marker[j] for j in range(len(passage_marker))):
                sep_end = i + len(passage_marker)
                sep_found_count += 1
                break
        
        if sep_end is None:
            # FALLBACK: Midpoint
            query_end = hidden.shape[1] // 2
            sep_not_found_count += 1
        else:
            query_end = sep_end
        
        # Split hidden states
        query_hidden = hidden[:, :query_end, :]      # [1, T_query, d_model]
        passage_hidden = hidden[:, query_end:, :]    # [1, T_passage, d_model]
        
        # ══════════════════════════════════════════════════════════════════════════════
        # QUERY-AWARE SCORING
        # ══════════════════════════════════════════════════════════════════════════════
        
        # Get token scores
        token_scores = model.importance_head(
            doc_hidden=passage_hidden,
            query_embeddings=query_hidden,
        )  # [1, T_passage] in [0, 1]
        
        # Aggregate: mean (winner config)
        passage_score = token_scores.mean().item() * 100.0
        scores.append(passage_score)
    
    return scores, sep_found_count, sep_not_found_count


def rank_passages(
    scores: list[float],
    is_selected: list[int],
    direction: str = "high_first",
) -> tuple[int, int, float]:
    """
    Rank passages and compute metrics.
    
    Returns:
        gold_rank: Position of selected passage (0-indexed, or -1 if not found)
        top_1_hit: 1 if selected passage is top-1, else 0
        mrr_score: MRR contribution (1/gold_rank if top match, else 0)
    """
    # Convert scores and selection to tuples with indices
    indexed = [(score, i, bool(is_selected[i])) for i, score in enumerate(scores)]
    
    # Sort by score (descending for high_first)
    if direction == "high_first":
        sorted_indexed = sorted(indexed, key=lambda x: -x[0])
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Find selected passage in ranking
    gold_rank = -1
    top_1_hit = 0
    
    for rank, (score, idx, is_gold) in enumerate(sorted_indexed):
        if is_gold:
            gold_rank = rank
            if rank == 0:
                top_1_hit = 1
            break
    
    # Compute MRR
    mrr_score = 1.0 / (gold_rank + 1) if gold_rank >= 0 else 0.0
    
    return gold_rank, top_1_hit, mrr_score


def evaluate_test_set(
    model: PatchedCausalLM,
    tokenizer: AutoTokenizer,
    test_data: list[dict],
    device: torch.device,
    output_path: str | None = None,
) -> dict:
    """Evaluate model on test set."""
    
    results = {
        "config": {
            "aggregation": "mean",
            "direction": "high_first",
            "architecture": "QueryAwareImportanceHead (mean pooling)",
        },
        "metrics": {
            "mrr": 0.0,
            "recall_1": 0.0,
            "recall_3": 0.0,
            "recall_5": 0.0,
            "ndcg_5": 0.0,
            "ndcg_10": 0.0,
            "num_queries": 0,
            "num_valid_queries": 0,
        },
        "separator": {
            "found": 0,
            "not_found": 0,
        },
    }
    
    mrr_scores = []
    recall_1 = []
    recall_3 = []
    recall_5 = []
    ndcg_5 = []
    ndcg_10 = []
    
    model.eval()
    
    print("\n[eval] Scoring passages on test set...")
    print("=" * 80)
    
    for query_data in tqdm(test_data, desc="Evaluating", unit="query"):
        query = query_data["query"]
        passages = query_data["passages"]
        is_selected = query_data["is_selected"]
        
        results["metrics"]["num_queries"] += 1
        
        # Skip if no selected passage
        if sum(is_selected) == 0:
            continue
        
        results["metrics"]["num_valid_queries"] += 1
        
        try:
            # Compute scores
            scores, sep_found, sep_not_found = compute_passage_scores(
                model, tokenizer, query, passages, device
            )
            
            results["separator"]["found"] += sep_found
            results["separator"]["not_found"] += sep_not_found
            
            # Rank passages
            gold_rank, top_1_hit, mrr_score = rank_passages(scores, is_selected)
            
            # Collect metrics
            mrr_scores.append(mrr_score)
            recall_1.append(top_1_hit)
            recall_3.append(1 if gold_rank < 3 else 0)
            recall_5.append(1 if gold_rank < 5 else 0)
            
            # NDCG@5 and NDCG@10
            # Ideal DCG: all relevant passages at top
            ideal_dcg_5 = sum(1 / np.log2(i + 2) for i in range(min(len(is_selected), 5)))
            ideal_dcg_10 = sum(1 / np.log2(i + 2) for i in range(min(len(is_selected), 10)))
            
            # Actual DCG
            dcg_5 = 0.0
            dcg_10 = 0.0
            if gold_rank >= 0:
                if gold_rank < 5:
                    dcg_5 = 1 / np.log2(gold_rank + 2)
                if gold_rank < 10:
                    dcg_10 = 1 / np.log2(gold_rank + 2)
            
            ndcg_5.append(dcg_5 / ideal_dcg_5 if ideal_dcg_5 > 0 else 0.0)
            ndcg_10.append(dcg_10 / ideal_dcg_10 if ideal_dcg_10 > 0 else 0.0)
        
        except Exception as e:
            print(f"[error] Query {query}: {str(e)[:100]}")
            continue
    
    # Aggregate metrics
    if mrr_scores:
        results["metrics"]["mrr"] = float(np.mean(mrr_scores))
        results["metrics"]["recall_1"] = float(np.mean(recall_1))
        results["metrics"]["recall_3"] = float(np.mean(recall_3))
        results["metrics"]["recall_5"] = float(np.mean(recall_5))
        results["metrics"]["ndcg_5"] = float(np.mean(ndcg_5))
        results["metrics"]["ndcg_10"] = float(np.mean(ndcg_10))
    
    # BM25 baseline for comparison
    bm25_mrr = 0.4320  # From v2.2 report
    delta = results["metrics"]["mrr"] - bm25_mrr
    pct_change = (delta / bm25_mrr) * 100 if bm25_mrr > 0 else 0
    
    print("\n" + "=" * 80)
    print("Test Set Results (TIS v2.3)")
    print("=" * 80)
    print(f"MRR:          {results['metrics']['mrr']:.4f}")
    print(f"Recall@1:     {results['metrics']['recall_1']:.4f}")
    print(f"Recall@3:     {results['metrics']['recall_3']:.4f}")
    print(f"Recall@5:     {results['metrics']['recall_5']:.4f}")
    print(f"NDCG@5:       {results['metrics']['ndcg_5']:.4f}")
    print(f"NDCG@10:      {results['metrics']['ndcg_10']:.4f}")
    print(f"Queries:      {results['metrics']['num_valid_queries']}")
    print()
    print("Separator Detection:")
    print(f"Found:        {results['separator']['found']}")
    print(f"Not found:    {results['separator']['not_found']}")
    total_seps = results['separator']['found'] + results['separator']['not_found']
    hit_rate = (results['separator']['found'] / total_seps * 100) if total_seps > 0 else 0
    print(f"Hit rate:     {hit_rate:.1f}%")
    print()
    print("Comparison with BM25 Baseline:")
    print(f"BM25 MRR:     {bm25_mrr:.4f}")
    print(f"v2.3 MRR:     {results['metrics']['mrr']:.4f}")
    print(f"Difference:   {delta:+.4f} ({pct_change:+.1f}%)")
    print("=" * 80)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[save] Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TIS v2.3 on MS-MARCO Test Set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to v2.3 checkpoint (tis_components.pt)")
    parser.add_argument("--data-path", type=str, default="data/msmarco_relevance/test.parquet",
                        help="Path to test data")
    parser.add_argument("--output-path", type=str, default="results/v2.3_test_results.json",
                        help="Output file for results")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device)
    
    # Load test data
    test_data = load_test_data(args.data_path)
    
    # Evaluate
    results = evaluate_test_set(
        model, tokenizer, test_data, device, args.output_path
    )


if __name__ == "__main__":
    main()

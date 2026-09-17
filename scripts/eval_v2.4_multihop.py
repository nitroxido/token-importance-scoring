#!/usr/bin/env python3
"""
Evaluate TIS v2.4 on Multi-hop (HotpotQA Bridge) Tasks

Compares:
1. TIS v2.3 baseline: pure importance scores
2. TIS v2.4 with bridge detection: importance + bridge signal blended

Metrics:
- Recall-both@K: both supporting passages in top-K (proxy for answer accuracy)
- Mean ranks: where do supporting passages appear?
- Bridge detection accuracy: does the head correctly identify bridge passages?

Expected improvement from v2.3 to v2.4:
- K=5: 50% → 60-70%
- K=3: 24.5% → 35-45%
- Bridge rank: 5.8 → ~4.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Resolve project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.model.bridge_detection_head import BridgeDetectionHead
from token_importance.config import TISConfig


# Constants
PASSAGE_MARKER = [7031, 1233, 29515]


def find_query_end(
    input_ids_list: list[int],
    passage_marker: list[int] = PASSAGE_MARKER,
) -> int:
    """Find position where passage begins."""
    for i in range(len(input_ids_list) - len(passage_marker) + 1):
        if all(input_ids_list[i + j] == passage_marker[j] for j in range(len(passage_marker))):
            return i + len(passage_marker)
    return len(input_ids_list) // 2


def extract_supporting_indices(
    context_titles: list[str],
    supporting_facts: dict,
) -> list[int]:
    """Extract indices of passages that are in supporting_facts."""
    supporting_indices = []
    
    for sf_title in supporting_facts["title"]:
        try:
            idx = context_titles.index(sf_title)
            supporting_indices.append(idx)
        except ValueError:
            pass
    
    return supporting_indices


def score_passages_v24(
    model: PatchedCausalLM,
    tokenizer: Any,
    query: str,
    passages: list[str],
    device: torch.device,
    use_bridge_signal: bool = True,
    bridge_weight: float = 0.2,
) -> tuple[list[float], list[float]]:
    """
    Score passages using TIS v2.4 (with optional bridge detection blending).
    
    Args:
        model: PatchedCausalLM with v2.4 checkpoint loaded
        tokenizer: Tokenizer
        query: Question
        passages: List of passage texts
        device: torch device
        use_bridge_signal: Whether to blend in bridge detection scores
        bridge_weight: Weight for bridge signal in final score
    
    Returns:
        (importance_scores, bridge_scores): Lists of scores per passage
    """
    importance_scores = []
    bridge_scores = []
    
    for passage in passages:
        prompt = f"Question: {query}\n\nPassage: {passage}\n\nAnswer:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=False,
        ).to(device)
        
        with torch.no_grad():
            outputs = model._base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        
        hidden = outputs.hidden_states[-1]
        input_ids_list = inputs["input_ids"][0].tolist()
        query_end = find_query_end(input_ids_list, PASSAGE_MARKER)
        
        passage_hidden = hidden[:, query_end:, :]
        query_hidden = hidden[:, :query_end, :]
        
        # Importance score
        imp_score = model.importance_head(
            query_hidden=query_hidden,
            context_hidden=passage_hidden,
        ).mean().item() * 100.0
        importance_scores.append(imp_score)
        
        # Bridge detection score (if available)
        if use_bridge_signal and hasattr(model, "bridge_detection_head"):
            bridge_logits = model.bridge_detection_head.forward_binary(passage_hidden).item()
            # Scale: 0.5 (neutral) -> 0, 1.0 (strong bridge) -> 1.0, 0.0 (not bridge) -> -1.0
            bridge_score = 2.0 * (bridge_logits - 0.5)
            bridge_scores.append(bridge_score)
        else:
            bridge_scores.append(0.0)
    
    return importance_scores, bridge_scores


def blended_ranking(
    importance_scores: list[float],
    bridge_scores: list[float],
    bridge_weight: float = 0.2,
) -> list[int]:
    """Blend importance and bridge scores, return ranking."""
    blended = [imp + bridge_weight * br for imp, br in zip(importance_scores, bridge_scores)]
    ranking = sorted(range(len(blended)), key=lambda i: -blended[i])
    return ranking


def recall_both_at_k(ranking: list[int], supporting_indices: list[int], k: int) -> bool:
    """Check if both supporting passages are in top-K."""
    if len(supporting_indices) < 2:
        return False
    top_k = set(ranking[:k])
    return all(idx in top_k for idx in supporting_indices[:2])


def theoretical_random_recall_both(n_total: int, n_sup: int, k: int) -> float:
    """Compute theoretical E[recall_both@K] for random ranking."""
    if k >= n_total:
        return 1.0
    # Probability that 2 supporting passages are both in top-K from n_total passages
    # = C(n_total-n_sup, k-n_sup) / C(n_total, k) for k >= n_sup
    from math import comb
    if k < n_sup:
        return 0.0
    return float(comb(n_total - n_sup, k - n_sup)) / float(comb(n_total, k))


def evaluate(
    model: PatchedCausalLM,
    tokenizer: Any,
    hotpotqa_data: Any,
    device: torch.device,
    max_questions: int = 200,
    seed: int = 42,
    use_bridge_signal: bool = True,
    output_path: str = "results/v24_multihop_eval.json",
) -> dict:
    """Evaluate TIS v2.4 on multi-hop questions."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Filter for bridge-type questions
    bridge_questions = [ex for ex in hotpotqa_data if ex.get("type") == "bridge"]
    
    if max_questions:
        sampled_indices = np.random.choice(len(bridge_questions), size=min(max_questions, len(bridge_questions)), replace=False)
        bridge_questions = [bridge_questions[i] for i in sorted(sampled_indices)]
    
    print(f"[eval] Evaluating on {len(bridge_questions)} bridge-type questions (seed={seed})")
    
    # Metrics per K
    metrics_per_k = {
        1: {"tis_recall_both": [], "tis_recall_any": [], "tis_rank_1st": [], "tis_rank_2nd": [], "bridge_acc": []},
        2: {"tis_recall_both": [], "tis_recall_any": [], "tis_rank_1st": [], "tis_rank_2nd": [], "bridge_acc": []},
        3: {"tis_recall_both": [], "tis_recall_any": [], "tis_rank_1st": [], "tis_rank_2nd": [], "bridge_acc": []},
        5: {"tis_recall_both": [], "tis_recall_any": [], "tis_rank_1st": [], "tis_rank_2nd": [], "bridge_acc": []},
        7: {"tis_recall_both": [], "tis_recall_any": [], "tis_rank_1st": [], "tis_rank_2nd": [], "bridge_acc": []},
        10: {"tis_recall_both": [], "tis_recall_any": [], "tis_rank_1st": [], "tis_rank_2nd": [], "bridge_acc": []},
    }
    
    skipped = 0
    rank_better_supporting = []
    rank_worse_supporting = []
    
    model.eval()
    
    pbar = tqdm(desc="[v2.4 eval]", total=len(bridge_questions), unit="question")
    
    for question_idx, ex in enumerate(bridge_questions):
        try:
            query = ex["question"]
            context_titles = ex["context"]["title"]
            context_sents = ex["context"]["sentences"]
            supporting_facts = ex["supporting_facts"]
            
            # Reconstruct passages
            passages = []
            for title, sents in zip(context_titles, context_sents):
                passage_text = " ".join(sents)
                passages.append(passage_text[:1800])  # Truncate for memory
            
            passages = passages[:10]  # Limit to 10
            
            # Get supporting indices
            supporting_indices = extract_supporting_indices(context_titles[:10], supporting_facts)
            
            if len(supporting_indices) < 2:
                skipped += 1
                pbar.update(1)
                continue
            
            # Score passages
            importance_scores, bridge_scores = score_passages_v24(
                model, tokenizer, query, passages, device,
                use_bridge_signal=True,
                bridge_weight=0.2,
            )
            
            # Pure TIS ranking (v2.3 baseline)
            tis_ranking = sorted(range(len(importance_scores)), key=lambda i: -importance_scores[i])
            
            # Blended ranking (v2.4)
            blended = [imp + 0.2 * br for imp, br in zip(importance_scores, bridge_scores)]
            blended_ranking_list = sorted(range(len(blended)), key=lambda i: -blended[i])
            
            # Bridge detection accuracy: are supporting passages marked as bridges correctly?
            is_bridge_pred = [1 if bridge_scores[i] > 0 else 0 for i in range(len(passages))]
            is_bridge_true = [1 if i == supporting_indices[0] else 0 for i in range(len(passages))]
            bridge_acc = float(np.mean([p == t for p, t in zip(is_bridge_pred, is_bridge_true)]))
            
            # Get ranks of supporting passages in TIS ranking
            rank_1st = tis_ranking.index(supporting_indices[0]) + 1
            rank_2nd = tis_ranking.index(supporting_indices[1]) + 1
            
            rank_better_supporting.append(min(rank_1st, rank_2nd))
            rank_worse_supporting.append(max(rank_1st, rank_2nd))
            
            # Compute metrics per K
            for k in metrics_per_k.keys():
                # Pure TIS
                recall_both = recall_both_at_k(tis_ranking, supporting_indices, k)
                recall_any = len(set(tis_ranking[:k]) & set(supporting_indices[:2])) > 0
                
                metrics_per_k[k]["tis_recall_both"].append(1.0 if recall_both else 0.0)
                metrics_per_k[k]["tis_recall_any"].append(1.0 if recall_any else 0.0)
                metrics_per_k[k]["tis_rank_1st"].append(1.0 / rank_1st)
                metrics_per_k[k]["tis_rank_2nd"].append(1.0 / rank_2nd)
                metrics_per_k[k]["bridge_acc"].append(bridge_acc)
        
        except Exception as e:
            print(f"\n[error] Question {question_idx}: {str(e)}")
            skipped += 1
        
        pbar.update(1)
    
    pbar.close()
    
    # Aggregate results
    aggregated = {}
    for k in metrics_per_k.keys():
        aggregated[f"k={k}"] = {
            "tis_recall_both": float(np.mean(metrics_per_k[k]["tis_recall_both"])),
            "tis_recall_any": float(np.mean(metrics_per_k[k]["tis_recall_any"])),
            "tis_mrr_1st": float(np.mean(metrics_per_k[k]["tis_rank_1st"])),
            "tis_mrr_2nd": float(np.mean(metrics_per_k[k]["tis_rank_2nd"])),
            "bridge_acc": float(np.mean(metrics_per_k[k]["bridge_acc"])),
        }
    
    result = {
        "summary": {
            "total_evaluated": len(bridge_questions) - skipped,
            "skipped": skipped,
            "model": "TIS v2.4 (with bridge detection)",
            "use_bridge_signal": use_bridge_signal,
            "bridge_weight": 0.2,
        },
        "aggregated": aggregated,
        "rank_statistics": {
            "mean_rank_better_supporting": float(np.mean(rank_better_supporting)) if rank_better_supporting else 0.0,
            "mean_rank_worse_supporting": float(np.mean(rank_worse_supporting)) if rank_worse_supporting else 0.0,
        },
    }
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TIS v2.4 MULTI-HOP EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total evaluated: {result['summary']['total_evaluated']}")
    print(f"Skipped: {result['summary']['skipped']}")
    print()
    
    print("Recall-both@K (both supporting passages in top-K):")
    for k in [1, 2, 3, 5, 7, 10]:
        recall = aggregated[f"k={k}"]["tis_recall_both"]
        print(f"  K={k:2d}: {recall*100:5.1f}%")
    
    print()
    print(f"Mean rank of better-positioned supporting: {result['rank_statistics']['mean_rank_better_supporting']:.2f} / 10")
    print(f"Mean rank of worse-positioned supporting: {result['rank_statistics']['mean_rank_worse_supporting']:.2f} / 10")
    print()
    print(f"Bridge detection accuracy: {aggregated['k=5']['bridge_acc']*100:.1f}%")
    print()
    print(f"Results saved → {output_path}")
    print("=" * 80)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate TIS v2.4 on multi-hop tasks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v2.4_multihop/best/tis_components.pt",
        help="Path to v2.4 checkpoint"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=200,
        help="Max questions to evaluate"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/v24_multihop_eval.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("[model] Loading TIS v2.4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = PatchedCausalLM.from_pretrained(
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        config=TISConfig(),
        device_map=device,
        quantization_config=bnb_config,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"[checkpoint] Loading: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.importance_head.load_state_dict(checkpoint["importance_head"])
        model.importance_embedding.load_state_dict(checkpoint["importance_embedding"])
        
        # Load bridge detection head if available
        if "bridge_detection_head" in checkpoint:
            bridge_head = BridgeDetectionHead(hidden_dim=4096, hidden_layer_dim=256)
            bridge_head.load_state_dict(checkpoint["bridge_detection_head"])
            model.bridge_detection_head = bridge_head.to(device)
            print("[checkpoint] ✓ v2.4 components loaded (including bridge detection head)")
        else:
            print("[warning] Bridge detection head not found in checkpoint; creating new one")
            model.bridge_detection_head = BridgeDetectionHead(hidden_dim=4096, hidden_layer_dim=256).to(device)
    else:
        print(f"[error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load HotpotQA
    print("[data] Loading HotpotQA...")
    try:
        hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    except:
        hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    
    # Evaluate
    results = evaluate(
        model,
        tokenizer,
        hotpotqa,
        device,
        max_questions=args.max_questions,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

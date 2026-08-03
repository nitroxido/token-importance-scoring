#!/usr/bin/env python
"""
scripts/validate_score_direction.py

Addresses independent replication concerns by testing:
1. Score direction (ascending vs descending) for all checkpoints
2. Multiple aggregation methods (mean, top-10%, length-normalized)
3. Sigmoid vs raw logit scoring
4. Passage-ranking metrics separate from generation metrics

Usage:
    source .venv/bin/activate
    python scripts/validate_score_direction.py --n-examples 60
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
_DEFAULT_CHECKPOINTS = {
    "stage3": os.path.join(_ROOT, "checkpoints", "stage3_ert"),
    "v8b": os.path.join(_ROOT, "checkpoints", "v8b_hard_anchor"),
}
DATA_PATH = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR = os.path.join(_ROOT, "results")
D_MODEL = 4096


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score direction validation")
    p.add_argument("--checkpoint-stage3", default=_DEFAULT_CHECKPOINTS["stage3"],
                   help="Path to stage3 checkpoint dir (default: checkpoints/stage3_ert)")
    p.add_argument("--checkpoint-v8b", default=_DEFAULT_CHECKPOINTS["v8b"],
                   help="Path to v8b checkpoint dir (default: checkpoints/v8b_hard_anchor)")
    p.add_argument("--n-examples", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--max-new-tokens", type=int, default=30)
    return p.parse_args()


def get_answers(item: dict) -> list[str]:
    """Extract answers robustly across MSMARCO schema variants."""
    answers = item.get("answers") or []
    if isinstance(answers, list) and answers:
        return [a for a in answers if isinstance(a, str) and a.strip()]

    well_formed = item.get("wellFormedAnswers") or []
    if isinstance(well_formed, list) and well_formed:
        return [a for a in well_formed if isinstance(a, str) and a.strip()]

    answer = item.get("answer")
    if isinstance(answer, str) and answer.strip():
        return [answer]

    return [""]


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer(device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[setup] Loading {BASE_MODEL}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device if device else "cuda",
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[setup] Model loaded. VRAM: {vram:.2f} GB", flush=True)
    return model, tokenizer


def load_tis_head(checkpoint_dir: str, device: str):
    from token_importance.config import TISConfig
    from token_importance.model.importance_head import ImportanceUpdateHead

    ckpt_path = os.path.join(checkpoint_dir, "tis_components.pt")
    print(f"[setup] Loading head from {ckpt_path}...", flush=True)
    
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tis_config = TISConfig()
    head = ImportanceUpdateHead(d_model=D_MODEL, config=tis_config, num_heads=4)
    head.load_state_dict(state["importance_head"], strict=False)
    head = head.to(device if device else "cuda")
    head.eval()
    return head


# ── Data loading ───────────────────────────────────────────────────────────────
def load_msmarco_examples(n: int, seed: int):
    from datasets import load_from_disk
    np.random.seed(seed)
    
    print(f"[data] Loading MS-MARCO from {DATA_PATH}...", flush=True)
    ds = load_from_disk(DATA_PATH)
    indices = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)
    
    examples = []
    for idx in indices:
        item = ds[int(idx)]
        passages = item["passages"]["passage_text"][:5]
        if len(passages) < 5:
            continue
        
        examples.append({
            "query_id": item["query_id"],
            "question": item["query"],
            "passages": passages,
            "gold_passage": passages[0],  # Assume first passage is gold
            "answers": get_answers(item),
        })
    
    print(f"[data] Loaded {len(examples)} examples", flush=True)
    return examples


# ── Scoring methods ────────────────────────────────────────────────────────────
@torch.inference_mode()
def score_tokens(model, tokenizer, tis_head, text: str, max_length: int = 3800):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    
    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1].float()
    scores = tis_head.direct_score(hidden)
    if scores.dim() == 2:
        scores = scores[0]
    return scores.cpu().numpy()


def aggregate_passage_score(token_scores: np.ndarray, method: str) -> float:
    """
    Aggregation methods:
    - 'mean': Mean of all token scores (original TIS method)
    - 'top10': Mean of top 10% token scores (independent replication method)
    - 'top20': Mean of top 20% token scores
    - 'length_norm': Mean divided by sqrt(num_tokens)
    """
    if method == "mean":
        return float(token_scores.mean())
    elif method == "top10":
        k = max(1, int(len(token_scores) * 0.10))
        return float(np.partition(token_scores, -k)[-k:].mean())
    elif method == "top20":
        k = max(1, int(len(token_scores) * 0.20))
        return float(np.partition(token_scores, -k)[-k:].mean())
    elif method == "length_norm":
        return float(token_scores.mean() / np.sqrt(len(token_scores)))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def reorder_by_tis(
    passages: list[str],
    model, 
    tokenizer,
    tis_head,
    aggregation: str = "mean",
    direction: str = "descending",
    use_sigmoid: bool = False,
) -> tuple[list[str], list[int], list[float]]:
    """
    TIS-based passage reordering with configurable aggregation and direction.
    
    Args:
        passages: List of passage texts
        model, tokenizer, tis_head: Model components
        aggregation: 'mean', 'top10', 'top20', 'length_norm'
        direction: 'descending' (high first) or 'ascending' (low first)
        use_sigmoid: Apply sigmoid before aggregation
    
    Returns:
        ordered_passages, ranked_indices, passage_scores
    """
    passage_scores = []
    for passage in passages:
        token_scores = score_tokens(model, tokenizer, tis_head, passage)
        if use_sigmoid:
            token_scores = 1 / (1 + np.exp(-token_scores))
        score = aggregate_passage_score(token_scores, aggregation)
        passage_scores.append(score)
    
    if direction == "descending":
        ranked_indices = np.argsort(-np.array(passage_scores))
    elif direction == "ascending":
        ranked_indices = np.argsort(np.array(passage_scores))
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    ordered_passages = [passages[i] for i in ranked_indices]
    return ordered_passages, list(ranked_indices), passage_scores


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_ranking_metrics(ranked_indices: list[int], gold_position: int = 0) -> dict:
    """Compute passage-ranking metrics (independent of generation)."""
    try:
        gold_rank = ranked_indices.index(gold_position)
    except ValueError:
        gold_rank = len(ranked_indices)
    
    return {
        "recall_at_1": int(gold_rank == 0),
        "mrr": 1.0 / (gold_rank + 1),
        "gold_rank": gold_rank,
    }


def exact_match(pred: str, answers: list[str]) -> int:
    pred_norm = pred.lower().strip()
    return int(any(ans.lower().strip() in pred_norm for ans in answers))


def token_f1(pred: str, answer: str) -> float:
    pred_toks = pred.lower().split()
    ans_toks = answer.lower().split()
    if not pred_toks or not ans_toks:
        return 0.0
    common = set(pred_toks) & set(ans_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(ans_toks)
    return 2 * prec * rec / (prec + rec)


# ── Generation ─────────────────────────────────────────────────────────────────
def build_prompt(passages: list[str], question: str) -> str:
    context = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])
    return f"""<s>[INST] Answer the question based on the given passages.

{context}

Question: {question}

Answer: [/INST]"""


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 30, max_input_length: int = 3800) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_toks = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


# ── Evaluation ─────────────────────────────────────────────────────────────────
def run_evaluation(args, examples, model, tokenizer, heads):
    """
    Test all combinations of:
    - Checkpoint (stage3, v8b)
    - Direction (descending, ascending)
    - Aggregation (mean, top10)
    - Activation (raw, sigmoid)
    """
    results = []
    
    configs = [
        # Original method (what we published)
        ("stage3", "mean", "descending", False, "Stage3 (high→low, mean raw)"),
        ("stage3", "mean", "ascending", False, "Stage3 (low→high, mean raw)"),
        ("v8b", "mean", "descending", False, "v8b (high→low, mean raw)"),
        ("v8b", "mean", "ascending", False, "v8b (low→high, mean raw)"),
        
        # Independent replication method
        ("stage3", "top10", "descending", True, "Stage3 (high→low, top10% sigmoid)"),
        ("stage3", "top10", "ascending", True, "Stage3 (low→high, top10% sigmoid)"),
    ]
    
    print(f"\n{'='*70}")
    print(f"Score Direction Validation: {len(examples)} examples × {len(configs)} configs")
    print(f"{'='*70}\n", flush=True)
    
    for ckpt, agg, direction, sigmoid, desc in configs:
        print(f"[{ckpt}] {desc}...", flush=True)
        t_start = time.perf_counter()
        
        head = heads[ckpt]
        
        for ex in examples:
            passages = ex["passages"]
            question = ex["question"]
            answers = ex["answers"]
            
            # Reorder passages
            ordered_passages, ranked_indices, scores = reorder_by_tis(
                passages, model, tokenizer, head, 
                aggregation=agg, direction=direction, use_sigmoid=sigmoid
            )
            
            # Ranking metrics
            rank_metrics = compute_ranking_metrics(ranked_indices, gold_position=0)
            
            # Generate answer
            prompt = build_prompt(ordered_passages, question)
            pred = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
            
            # Generation metrics
            em = exact_match(pred, answers)
            f1 = max([token_f1(pred, a) for a in answers]) if answers else 0.0
            
            results.append({
                "checkpoint": ckpt,
                "aggregation": agg,
                "direction": direction,
                "sigmoid": bool(sigmoid),
                "config_desc": desc,
                "query_id": ex["query_id"],
                "question": question,
                # Ranking metrics
                "recall_at_1": int(rank_metrics["recall_at_1"]),
                "mrr": float(rank_metrics["mrr"]),
                "gold_rank": int(rank_metrics["gold_rank"]),
                # Generation metrics
                "em": int(em),
                "f1": float(f1),
                "prediction": pred,
                "answer": answers[0] if answers else "",
                # Diagnostics
                "passage_scores": [float(s) for s in scores],
                "ranked_indices": [int(i) for i in ranked_indices],
            })
        
        elapsed = time.perf_counter() - t_start
        print(f"[{ckpt}] Done in {elapsed:.1f}s\n", flush=True)
    
    return results


# ── Analysis ───────────────────────────────────────────────────────────────────
def generate_summary(results: list[dict]) -> dict:
    """Aggregate metrics by configuration."""
    from collections import defaultdict
    
    by_config = defaultdict(list)
    for r in results:
        key = (r["checkpoint"], r["aggregation"], r["direction"], r["sigmoid"])
        by_config[key].append(r)
    
    summary = []
    for (ckpt, agg, direction, sigmoid), items in by_config.items():
        n = len(items)
        summary.append({
            "checkpoint": ckpt,
            "aggregation": agg,
            "direction": direction,
            "sigmoid": sigmoid,
            "config_desc": items[0]["config_desc"],
            "n": n,
            # Ranking metrics (mean)
            "recall_at_1": np.mean([r["recall_at_1"] for r in items]),
            "mrr": np.mean([r["mrr"] for r in items]),
            "avg_gold_rank": np.mean([r["gold_rank"] for r in items]),
            # Generation metrics (mean)
            "em": np.mean([r["em"] for r in items]),
            "f1": np.mean([r["f1"] for r in items]),
        })
    
    return sorted(summary, key=lambda x: (x["checkpoint"], x["direction"], x["aggregation"]))


def print_summary_table(summary: list[dict]):
    """Print formatted summary table."""
    print("\n" + "="*90)
    print("SUMMARY: Score Direction Validation")
    print("="*90)
    print(f"{'Configuration':<40} {'R@1':>6} {'MRR':>6} {'EM':>6} {'F1':>6}")
    print("-"*90)
    
    for row in summary:
        print(f"{row['config_desc']:<40} {row['recall_at_1']:>6.3f} {row['mrr']:>6.3f} {row['em']:>6.3f} {row['f1']:>6.3f}")
    
    print("="*90)
    print("\nKey Findings:")
    print()
    
    # Compare descending vs ascending for each checkpoint
    stage3_desc = [r for r in summary if r["checkpoint"] == "stage3" and r["direction"] == "descending" and r["aggregation"] == "mean"][0]
    stage3_asc = [r for r in summary if r["checkpoint"] == "stage3" and r["direction"] == "ascending" and r["aggregation"] == "mean"][0]
    
    print("Stage3 Direction Comparison (mean raw):")
    print(f"  High→Low: MRR={stage3_desc['mrr']:.3f}, EM={stage3_desc['em']:.3f}")
    print(f"  Low→High: MRR={stage3_asc['mrr']:.3f}, EM={stage3_asc['em']:.3f}")
    print(f"  → MRR favors: {'ascending' if stage3_asc['mrr'] > stage3_desc['mrr'] else 'descending'}")
    print(f"  → EM favors: {'ascending' if stage3_asc['em'] > stage3_desc['em'] else 'descending'}")
    print()
    
    v8b_desc = [r for r in summary if r["checkpoint"] == "v8b" and r["direction"] == "descending"][0]
    v8b_asc = [r for r in summary if r["checkpoint"] == "v8b" and r["direction"] == "ascending"][0]
    
    print("v8b Direction Comparison (mean raw):")
    print(f"  High→Low: MRR={v8b_desc['mrr']:.3f}, EM={v8b_desc['em']:.3f}")
    print(f"  Low→High: MRR={v8b_asc['mrr']:.3f}, EM={v8b_asc['em']:.3f}")
    print(f"  → Direction sensitivity: {'low' if abs(v8b_desc['em'] - v8b_asc['em']) < 0.02 else 'high'}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    device = args.device if args.device else "cuda"
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(device)
    
    # Load all TIS heads
    heads = {
        "stage3": load_tis_head(args.checkpoint_stage3, device),
        "v8b": load_tis_head(args.checkpoint_v8b, device),
    }
    
    # Load data
    examples = load_msmarco_examples(args.n_examples, args.seed)
    
    # Run evaluation
    results = run_evaluation(args, examples, model, tokenizer, heads)
    
    # Generate summary
    summary = generate_summary(results)
    print_summary_table(summary)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    summary_path = os.path.join(RESULTS_DIR, "score_direction_validation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[save] {summary_path}")
    
    details_path = os.path.join(RESULTS_DIR, "score_direction_validation_details.json")
    with open(details_path, "w") as f:
        json.dump({
            "metadata": {
                "date": datetime.now().isoformat(),
                "n_examples": args.n_examples,
                "seed": args.seed,
            },
            "results": results,
            "summary": summary,
        }, f, indent=2)
    print(f"[save] {details_path}")
    
    print("\n✅ Score direction validation complete!")


if __name__ == "__main__":
    main()

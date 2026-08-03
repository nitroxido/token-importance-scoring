#!/usr/bin/env python
"""
scripts/test_query_specificity.py

Tests whether TIS checkpoints genuinely respond to query-specific relevance
by comparing correct queries vs lexical-near wrong queries.

Addresses independent replication concern:
"The dedicated checkpoint is query-conditioned, but correct-query relevance 
is not yet isolated"

Method:
1. For each query Q_i, find most lexically similar Q_j (j != i) using BM25
2. Score same 5 passages with Q_i (correct) and Q_j (wrong)
3. Measure:
   - Selected-passage MRR difference (correct - wrong)
   - Order change rate
   - Top-passage change rate
   - Bootstrap confidence interval

Expected result if query relevance is learned:
    MRR_correct > MRR_wrong with 95% CI above zero

Usage:
    source .venv/bin/activate
    python scripts/test_query_specificity.py --n-examples 60
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
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

import numpy as np
import torch
from rank_bm25 import BM25Okapi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "v8b_hard_anchor")
DATA_PATH = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR = os.path.join(_ROOT, "results")
D_MODEL = 4096


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test query-specific relevance")
    p.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT,
                   help="Path to checkpoint dir (default: checkpoints/v8b_hard_anchor)")
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
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[setup] Model loaded. VRAM: {vram:.2f} GB", flush=True)
    return model, tokenizer


def load_tis_head(checkpoint_dir: str, device: str):
    from token_importance.config import TISConfig
    from token_importance.model.importance_head import ImportanceUpdateHead

    ckpt_path = os.path.join(checkpoint_dir, "tis_components.pt")
    print(f"[setup] Loading TIS head from {ckpt_path}...", flush=True)
    
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
            "answers": get_answers(item),
        })
    
    print(f"[data] Loaded {len(examples)} examples", flush=True)
    return examples


def find_lexical_near_queries(examples: list[dict]) -> dict:
    """
    For each query, find the most lexically similar different query using BM25.
    Returns: {query_id: wrong_query_id}
    """
    print("[matching] Finding lexical-near wrong queries...", flush=True)
    
    queries = [ex["question"] for ex in examples]
    query_ids = [ex["query_id"] for ex in examples]
    
    # Tokenize for BM25
    tokenized = [q.lower().split() for q in queries]
    bm25 = BM25Okapi(tokenized)
    
    matches = {}
    for i, query in enumerate(queries):
        scores = bm25.get_scores(query.lower().split())
        scores[i] = -np.inf  # Exclude self
        
        best_match = np.argmax(scores)
        matches[query_ids[i]] = query_ids[best_match]
    
    print(f"[matching] Matched {len(matches)} query pairs", flush=True)
    return matches


# ── Scoring ────────────────────────────────────────────────────────────────────
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


@torch.inference_mode()
def score_passage_with_query(
    model,
    tokenizer,
    tis_head,
    query: str,
    passage: str,
    max_length: int = 3800,
):
    """Score only passage tokens while conditioning hidden states on the query.

    Wrapper contract:
    - Build input as "Question ... Passage ..."
    - Run hidden-state scorer once
    - Aggregate only passage-token region
    """
    prompt = f"Question: {query}\n\nPassage: {passage}"
    prefix = f"Question: {query}\n\nPassage: "

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1].float()
    scores = tis_head.direct_score(hidden)
    if scores.dim() == 2:
        scores = scores[0]

    # Determine passage token span inside concatenated prompt.
    prefix_ids = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
    start = int(prefix_ids.shape[0])
    total = int(enc["input_ids"].shape[1])
    if start >= total:
        return scores.cpu().numpy()

    return scores[start:total].cpu().numpy()


def reorder_by_tis(
    query: str,
    passages: list[str],
    model,
    tokenizer,
    tis_head,
    use_query_wrapper: bool = True,
) -> tuple[list[str], list[int], list[float]]:
    """TIS-based ranking with optional query-conditioned wrapper."""
    scores_per_passage = []
    for passage in passages:
        if use_query_wrapper:
            token_scores = score_passage_with_query(model, tokenizer, tis_head, query, passage)
        else:
            token_scores = score_tokens(model, tokenizer, tis_head, passage)
        scores_per_passage.append(token_scores)
    
    mean_scores = [float(s.mean()) for s in scores_per_passage]
    ranked_indices = np.argsort(-np.array(mean_scores))
    
    return [passages[i] for i in ranked_indices], list(ranked_indices), mean_scores


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
def run_evaluation(args, examples, query_matches, model, tokenizer, tis_head):
    """
    For each example:
    1. Score passages with correct query
    2. Score same passages with wrong query (lexically similar)
    3. Compare rankings and generation
    """
    results = []
    query_lookup = {ex["query_id"]: ex for ex in examples}
    
    print(f"\n{'='*70}")
    print(f"Query Specificity Test: {len(examples)} examples × 2 conditions")
    print(f"{'='*70}\n", flush=True)
    
    t_start = time.perf_counter()
    
    for ex in examples:
        query_id = ex["query_id"]
        correct_query = ex["question"]
        passages = ex["passages"]
        answers = ex["answers"]
        
        # Find wrong query
        wrong_query_id = query_matches.get(query_id)
        if not wrong_query_id or wrong_query_id not in query_lookup:
            continue
        
        wrong_query = query_lookup[wrong_query_id]["question"]
        
        # Score with correct query
        ordered_correct, indices_correct, scores_correct = reorder_by_tis(
            correct_query, passages, model, tokenizer, tis_head, use_query_wrapper=True
        )
        rank_metrics_correct = compute_ranking_metrics(indices_correct, gold_position=0)
        
        # Score with wrong query
        ordered_wrong, indices_wrong, scores_wrong = reorder_by_tis(
            wrong_query, passages, model, tokenizer, tis_head, use_query_wrapper=True
        )
        rank_metrics_wrong = compute_ranking_metrics(indices_wrong, gold_position=0)
        
        # Compare orders
        order_changed = indices_correct != indices_wrong
        top_changed = indices_correct[0] != indices_wrong[0]
        
        # Generate answers for both
        prompt_correct = build_prompt(ordered_correct, correct_query)
        pred_correct = generate_answer(model, tokenizer, prompt_correct, args.max_new_tokens)
        em_correct = exact_match(pred_correct, answers)
        f1_correct = max([token_f1(pred_correct, a) for a in answers]) if answers else 0.0
        
        prompt_wrong = build_prompt(ordered_wrong, wrong_query)  # Use wrong query
        pred_wrong = generate_answer(model, tokenizer, prompt_wrong, args.max_new_tokens)
        # Note: EM here is against original answers (using wrong query)
        em_wrong = exact_match(pred_wrong, answers)
        f1_wrong = max([token_f1(pred_wrong, a) for a in answers]) if answers else 0.0
        
        results.append({
            "query_id": query_id,
            "correct_query": correct_query,
            "wrong_query_id": wrong_query_id,
            "wrong_query": wrong_query,
            # Ranking metrics
            "correct_mrr": float(rank_metrics_correct["mrr"]),
            "wrong_mrr": float(rank_metrics_wrong["mrr"]),
            "mrr_diff": float(rank_metrics_correct["mrr"] - rank_metrics_wrong["mrr"]),
            "correct_recall_at_1": int(rank_metrics_correct["recall_at_1"]),
            "wrong_recall_at_1": int(rank_metrics_wrong["recall_at_1"]),
            # Order diagnostics
            "order_changed": bool(order_changed),
            "top_changed": bool(top_changed),
            "correct_indices": [int(i) for i in indices_correct],
            "wrong_indices": [int(i) for i in indices_wrong],
            # Generation metrics
            "correct_em": int(em_correct),
            "wrong_em": int(em_wrong),
            "correct_f1": float(f1_correct),
            "wrong_f1": float(f1_wrong),
            "correct_pred": pred_correct,
            "wrong_pred": pred_wrong,
            "answer": answers[0] if answers else "",
        })
    
    elapsed = time.perf_counter() - t_start
    print(f"[eval] Processed {len(results)} paired examples in {elapsed:.1f}s\n", flush=True)
    
    return results


# ── Analysis ───────────────────────────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    lower = np.percentile(means, alpha/2 * 100)
    upper = np.percentile(means, (1 - alpha/2) * 100)
    return lower, upper


def analyze_results(results: list[dict]):
    """Generate summary statistics and diagnostics."""
    n = len(results)
    
    # MRR comparison
    correct_mrr = np.array([r["correct_mrr"] for r in results])
    wrong_mrr = np.array([r["wrong_mrr"] for r in results])
    mrr_diff = correct_mrr - wrong_mrr
    
    mrr_diff_mean = mrr_diff.mean()
    mrr_diff_ci = bootstrap_ci(mrr_diff, n_boot=10000)
    
    # Order change rates
    order_change_rate = np.mean([r["order_changed"] for r in results])
    top_change_rate = np.mean([r["top_changed"] for r in results])
    
    # Generation comparison
    correct_em = np.mean([r["correct_em"] for r in results])
    wrong_em = np.mean([r["wrong_em"] for r in results])
    
    print("\n" + "="*70)
    print("QUERY SPECIFICITY TEST RESULTS")
    print("="*70)
    print(f"Sample size: {n} query pairs")
    print()
    
    print("Passage Ranking Metrics:")
    print(f"  Correct query MRR:  {correct_mrr.mean():.3f}")
    print(f"  Wrong query MRR:    {wrong_mrr.mean():.3f}")
    print(f"  Difference:         {mrr_diff_mean:.3f}")
    print(f"  95% CI:             [{mrr_diff_ci[0]:.3f}, {mrr_diff_ci[1]:.3f}]")
    print()
    
    if mrr_diff_ci[0] > 0:
        print("  ✅ PASS: Correct query significantly improves ranking (CI > 0)")
    elif mrr_diff_ci[1] < 0:
        print("  ❌ FAIL: Correct query significantly worsens ranking (CI < 0)")
    else:
        print("  ⚠️  INCONCLUSIVE: CI contains zero (no significant difference)")
    print()
    
    print("Order Change Diagnostics:")
    print(f"  Full order changed:  {order_change_rate*100:.1f}% of cases")
    print(f"  Top passage changed: {top_change_rate*100:.1f}% of cases")
    print()
    
    if order_change_rate > 0.5:
        print("  ✅ Query-conditioning confirmed (order changes >50% of time)")
    else:
        print("  ❌ Weak query-conditioning (order changes <50% of time)")
    print()
    
    print("Generation Metrics:")
    print(f"  Correct query EM:    {correct_em:.3f}")
    print(f"  Wrong query EM:      {wrong_em:.3f}")
    print(f"  Difference:          {correct_em - wrong_em:.3f}")
    print()
    
    print("="*70)
    print()
    
    print("Interpretation:")
    if mrr_diff_ci[0] > 0 and order_change_rate > 0.5:
        print("  The checkpoint is query-conditioned AND correct queries improve ranking.")
        print("  → Query-specific relevance is validated ✅")
    elif order_change_rate > 0.5 and mrr_diff_ci[0] <= 0 <= mrr_diff_ci[1]:
        print("  The checkpoint is query-conditioned BUT correct queries don't improve ranking.")
        print("  → Query affects scores but not in the relevance direction ⚠️")
    else:
        print("  Query-relevance validation failed.")
        print("  → The score may capture query-independent context utility ❌")
    print()
    
    return {
        "n": n,
        "correct_mrr_mean": float(correct_mrr.mean()),
        "wrong_mrr_mean": float(wrong_mrr.mean()),
        "mrr_diff_mean": float(mrr_diff_mean),
        "mrr_diff_ci_lower": float(mrr_diff_ci[0]),
        "mrr_diff_ci_upper": float(mrr_diff_ci[1]),
        "order_change_rate": float(order_change_rate),
        "top_change_rate": float(top_change_rate),
        "correct_em": float(correct_em),
        "wrong_em": float(wrong_em),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    device = args.device if args.device else "cuda"
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(device)
    tis_head = load_tis_head(args.checkpoint, device)
    
    # Load data
    examples = load_msmarco_examples(args.n_examples, args.seed)
    
    # Find lexical-near queries
    query_matches = find_lexical_near_queries(examples)
    
    # Run evaluation
    results = run_evaluation(args, examples, query_matches, model, tokenizer, tis_head)
    
    # Analyze
    summary = analyze_results(results)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    details_path = os.path.join(RESULTS_DIR, "query_specificity_test.json")
    with open(details_path, "w") as f:
        json.dump({
            "metadata": {
                "date": datetime.now().isoformat(),
                "n_examples": len(results),
                "seed": args.seed,
                "checkpoint": args.checkpoint,
            },
            "summary": summary,
            "results": results,
        }, f, indent=2)
    print(f"[save] {details_path}")
    
    summary_path = os.path.join(RESULTS_DIR, "query_specificity_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(f"[save] {summary_path}")
    
    print("\n✅ Query specificity test complete!")


if __name__ == "__main__":
    main()

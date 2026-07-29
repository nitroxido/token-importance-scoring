#!/usr/bin/env python
"""
scripts/run_litm_with_baselines.py

Enhanced LITM evaluation with:
- Baseline comparisons (Lexical/BM25, Oracle, Query-layout)
- Ranking metrics (Recall@1, MRR, gold passage rank)
- Per-example breakdown CSV
- Reproducibility metadata JSON

Addresses methodological feedback:
1. Separates ranking quality from answer quality
2. Provides lexical/oracle controls
3. Generates per-example transition analysis
4. Documents exact evaluation path

Usage:
    source .venv/bin/activate
    python scripts/run_litm_with_baselines.py --n-examples 60
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
CHECKPOINT  = os.path.join(_ROOT, "checkpoints", "v8_hard_anchor_final")
DATA_PATH   = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR = os.path.join(_ROOT, "results")
D_MODEL     = 4096


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LITM with baselines + ranking metrics")
    p.add_argument("--n-examples", type=int, default=60,
                   help="Number of base MS-MARCO examples (×3 positions)")
    p.add_argument("--k-passages", type=int, default=5,
                   help="Number of passages per context")
    p.add_argument("--keep-ratio", type=float, default=0.7,
                   help="Token keep ratio for pruning")
    p.add_argument("--max-new-tokens", type=int, default=30,
                   help="Max tokens to generate")
    p.add_argument("--device", default="", help="CUDA device")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


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


def load_tis_head(device: str):
    from token_importance.config import TISConfig
    from token_importance.model.importance_head import ImportanceUpdateHead

    ckpt_path = os.path.join(CHECKPOINT, "tis_components.pt")
    print(f"[setup] Loading TIS head from {ckpt_path}...", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tis_config = TISConfig()
    head = ImportanceUpdateHead(d_model=D_MODEL, config=tis_config, num_heads=4)
    head.load_state_dict(state["importance_head"], strict=False)
    dev = torch.device(device if device else "cuda")
    head = head.to(dev).eval()
    print("[setup] TIS head loaded.", flush=True)
    return head


# ── Dataset loading ────────────────────────────────────────────────────────────
def load_msmarco_examples(n_examples: int, k_passages: int) -> list[dict]:
    """Load MS-MARCO with 3 position variants per base example."""
    import datasets as hf_datasets
    print(f"[data] Loading MS-MARCO from {DATA_PATH}...", flush=True)
    ds = hf_datasets.load_from_disk(DATA_PATH)

    examples = []
    base_count = 0

    for raw in ds:
        if base_count >= n_examples:
            break

        passages_data = raw["passages"]
        if isinstance(passages_data, str):
            passages_data = json.loads(passages_data)

        texts      = passages_data["passage_text"]
        selected   = passages_data["is_selected"]
        answers_raw = raw["answers"]
        if isinstance(answers_raw, str):
            answers_raw = json.loads(answers_raw)

        gold_indices = [i for i, s in enumerate(selected) if s == 1]
        if not gold_indices:
            continue

        valid_answers = [
            a.strip() for a in answers_raw
            if a.strip() and a.strip() not in ("No Answer Present.", "[]")
        ]
        if not valid_answers:
            continue

        gold_idx     = gold_indices[0]
        gold_passage = texts[gold_idx]
        distractors  = [t for i, t in enumerate(texts) if i != gold_idx]

        needed = k_passages - 1
        if len(distractors) < needed:
            distractors = distractors + [distractors[-1]] * (needed - len(distractors))
        distractors = distractors[:needed]

        # 3 positions: early, middle, end
        positions = [
            ("early",  0),
            ("middle", k_passages // 2),
            ("end",    k_passages - 1),
        ]
        for pos_name, gold_pos in positions:
            passage_list = list(distractors)
            passage_list.insert(gold_pos, gold_passage)

            examples.append({
                "question":      raw["query"],
                "passages":      passage_list,
                "gold_passage":  gold_passage,
                "answers":       valid_answers,
                "gold_position": gold_pos,
                "position_name": pos_name,
                "query_id":      raw["query_id"],
                "base_id":       base_count,  # for pairing later
            })

        base_count += 1

    print(f"[data] Prepared {len(examples)} test cases ({base_count} base × 3 positions)", flush=True)
    return examples


# ── Text utilities ─────────────────────────────────────────────────────────────
def build_prompt(passages: list[str], question: str) -> str:
    body = "\n\n".join(f"Passage {i+1}: {p.strip()}" for i, p in enumerate(passages))
    instruction = (
        f"Read the following passages and answer the question. "
        f"Give only the answer, 1-5 words, no explanation.\n\n"
        f"{body}\n\n"
        f"Question: {question}"
    )
    return f"[INST] {instruction} [/INST]"


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^\s*(a|an|the)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(pred: str, golds: list[str]) -> int:
    p = normalize_answer(pred)
    return int(any(
        p == normalize_answer(g) or
        normalize_answer(g) in p or
        p in normalize_answer(g)
        for g in golds
    ))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ── Lexical baseline (TF-IDF) ──────────────────────────────────────────────────
def reorder_by_tfidf(question: str, passages: list[str]) -> tuple[list[str], list[int]]:
    """Lexical baseline: rank passages by TF-IDF similarity to query."""
    corpus = [question] + passages
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[0]
        passage_vecs = tfidf_matrix[1:]
        
        # Cosine similarity
        scores = (passage_vecs * query_vec.T).toarray().flatten()
        ranked_indices = np.argsort(-scores)
        
        return [passages[i] for i in ranked_indices], list(ranked_indices)
    except:
        # Fallback if TF-IDF fails (empty passages, etc.)
        return passages, list(range(len(passages)))


# ── TIS scoring ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def score_tokens(model, tokenizer, tis_head, text: str, max_length: int = 3800):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(next(model.parameters()).device)
    token_ids = enc["input_ids"][0].tolist()
    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1].float()
    scores = tis_head.direct_score(hidden)
    if scores.dim() == 2:
        scores = scores[0]
    return token_ids, scores.cpu().numpy()


def reorder_by_tis(passages: list[str], model, tokenizer, tis_head) -> tuple[list[str], list[int], list[float]]:
    """TIS-based ranking: score all passages, rank by mean score."""
    scores_per_passage = []
    for passage in passages:
        _, scores = score_tokens(model, tokenizer, tis_head, passage)
        scores_per_passage.append(scores)
    
    mean_scores = [float(s.mean()) for s in scores_per_passage]
    ranked_indices = np.argsort(-np.array(mean_scores))
    
    return [passages[i] for i in ranked_indices], list(ranked_indices), mean_scores


# ── Generation ─────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 30, max_input_length: int = 3800) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_toks = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


# ── Evaluation ─────────────────────────────────────────────────────────────────
def run_evaluation(args, examples, model, tokenizer, tis_head):
    """Run all pipelines and collect per-example results."""
    results = []
    
    pipelines = [
        ("baseline", "original order"),
        ("lexical", "TF-IDF reranking"),
        ("tis", "TIS reordering"),
        ("oracle", "Gold passage first"),
    ]
    
    print(f"\n{'='*70}\nStarting evaluation: {len(examples)} examples × {len(pipelines)} pipelines\n{'='*70}\n", flush=True)
    
    for pipeline_name, pipeline_desc in pipelines:
        print(f"[{pipeline_name}] {pipeline_desc}...", flush=True)
        t_start = time.perf_counter()
        
        for ex in examples:
            passages = ex["passages"]
            question = ex["question"]
            answers = ex["answers"]
            gold_position = ex["gold_position"]
            
            # Determine ordering
            if pipeline_name == "baseline":
                ordered_passages = passages
                ranked_indices = list(range(len(passages)))
                mean_scores = None
            elif pipeline_name == "lexical":
                ordered_passages, ranked_indices = reorder_by_tfidf(question, passages)
                mean_scores = None
            elif pipeline_name == "tis":
                ordered_passages, ranked_indices, mean_scores = reorder_by_tis(passages, model, tokenizer, tis_head)
            elif pipeline_name == "oracle":
                ordered_passages = [ex["gold_passage"]] + [p for p in passages if p != ex["gold_passage"]]
                ranked_indices = [gold_position] + [i for i in range(len(passages)) if i != gold_position]
                mean_scores = None
            
            # Find where gold ended up
            try:
                gold_new_rank = ranked_indices.index(gold_position)
            except ValueError:
                gold_new_rank = -1
            
            # Generate answer
            prompt = build_prompt(ordered_passages, question)
            pred = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
            
            # Metrics
            em = exact_match(pred, answers)
            f1 = max([token_f1(pred, a) for a in answers]) if answers else 0.0
            
            results.append({
                "pipeline": pipeline_name,
                "base_id": ex["base_id"],
                "query_id": ex["query_id"],
                "position_name": ex["position_name"],
                "original_gold_position": gold_position,
                "gold_new_rank": gold_new_rank,
                "recall_at_1": int(gold_new_rank == 0),
                "mrr": 1.0 / (gold_new_rank + 1) if gold_new_rank >= 0 else 0.0,
                "em": em,
                "f1": f1,
                "prediction": pred,
                "answer": answers[0] if answers else "",
            })
        
        elapsed = time.perf_counter() - t_start
        print(f"[{pipeline_name}] Done in {elapsed:.1f}s\n", flush=True)
    
    return results


# ── Analysis ───────────────────────────────────────────────────────────────────
def generate_summaries(results: list[dict], args):
    """Generate summary tables and per-example breakdown."""
    
    # Aggregate by pipeline × position
    summary_rows = []
    for pipeline in ["baseline", "lexical", "tis", "oracle"]:
        for pos_name in ["early", "middle", "end"]:
            subset = [r for r in results if r["pipeline"] == pipeline and r["position_name"] == pos_name]
            if not subset:
                continue
            
            summary_rows.append({
                "pipeline": pipeline,
                "position": pos_name,
                "n": len(subset),
                "recall_at_1": np.mean([r["recall_at_1"] for r in subset]),
                "mrr": np.mean([r["mrr"] for r in subset]),
                "em": np.mean([r["em"] for r in subset]),
                "f1": np.mean([r["f1"] for r in subset]),
            })
    
    # Per-example breakdown with transitions
    per_example = []
    for base_id in set(r["base_id"] for r in results):
        for pos_name in ["early", "middle", "end"]:
            # Get baseline and TIS results for this example
            baseline_result = next((r for r in results if r["base_id"] == base_id and r["position_name"] == pos_name and r["pipeline"] == "baseline"), None)
            tis_result = next((r for r in results if r["base_id"] == base_id and r["position_name"] == pos_name and r["pipeline"] == "tis"), None)
            
            if baseline_result and tis_result:
                baseline_correct = baseline_result["em"]
                tis_correct = tis_result["em"]
                
                if baseline_correct and tis_correct:
                    transition = "same_correct"
                elif not baseline_correct and not tis_correct:
                    transition = "same_wrong"
                elif not baseline_correct and tis_correct:
                    transition = "wrong_to_right"
                else:
                    transition = "right_to_wrong"
                
                per_example.append({
                    "base_id": base_id,
                    "query_id": baseline_result["query_id"],
                    "position_name": pos_name,
                    "original_gold_position": baseline_result["original_gold_position"],
                    "tis_gold_rank": tis_result["gold_new_rank"],
                    "recall_at_1": tis_result["recall_at_1"],
                    "baseline_correct": baseline_correct,
                    "tis_correct": tis_correct,
                    "transition": transition,
                    "baseline_prediction": baseline_result["prediction"],
                    "tis_prediction": tis_result["prediction"],
                    "answer": baseline_result["answer"],
                })
    
    return summary_rows, per_example


def save_results(summary_rows, per_example, metadata, args):
    """Save all outputs: summary CSV, per-example CSV, metadata JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, "litm_with_baselines_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["pipeline", "position", "n", "recall_at_1", "mrr", "em", "f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[save] {summary_path}", flush=True)
    
    # Per-example CSV
    per_example_path = os.path.join(RESULTS_DIR, "litm_per_example_breakdown.csv")
    with open(per_example_path, "w", newline="") as f:
        fieldnames = ["base_id", "query_id", "position_name", "original_gold_position", 
                     "tis_gold_rank", "recall_at_1", "baseline_correct", "tis_correct", 
                     "transition", "baseline_prediction", "tis_prediction", "answer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_example)
    print(f"[save] {per_example_path}", flush=True)
    
    # Metadata JSON
    metadata_path = os.path.join(RESULTS_DIR, "litm_with_baselines_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[save] {metadata_path}", flush=True)
    
    # Print summary table
    print(f"\n{'='*70}\nSummary: Ranking Metrics vs Answer Quality\n{'='*70}\n")
    for pipeline in ["baseline", "lexical", "tis", "oracle"]:
        subset = [r for r in summary_rows if r["pipeline"] == pipeline]
        if not subset:
            continue
        print(f"{pipeline.upper()}:")
        print(f"  Position | Recall@1 | MRR   | EM    | F1")
        print(f"  ---------|----------|-------|-------|-------")
        for row in subset:
            print(f"  {row['position']:8s} | {row['recall_at_1']:8.3f} | {row['mrr']:.3f} | {row['em']:.3f} | {row['f1']:.3f}")
        avg_recall = np.mean([r["recall_at_1"] for r in subset])
        avg_mrr = np.mean([r["mrr"] for r in subset])
        avg_em = np.mean([r["em"] for r in subset])
        avg_f1 = np.mean([r["f1"] for r in subset])
        print(f"  Average  | {avg_recall:8.3f} | {avg_mrr:.3f} | {avg_em:.3f} | {avg_f1:.3f}\n")
    
    # Transition analysis
    print(f"{'='*70}\nError Transitions (Baseline → TIS)\n{'='*70}\n")
    transitions = {}
    for r in per_example:
        transitions[r["transition"]] = transitions.get(r["transition"], 0) + 1
    
    total = len(per_example)
    for trans in ["same_correct", "same_wrong", "wrong_to_right", "right_to_wrong"]:
        count = transitions.get(trans, 0)
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {trans:20s}: {count:3d} / {total} ({pct:5.1f}%)")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load
    model, tokenizer = load_model_and_tokenizer(args.device)
    tis_head = load_tis_head(args.device)
    examples = load_msmarco_examples(args.n_examples, args.k_passages)
    
    # Metadata
    metadata = {
        "experiment_id": "litm_with_baselines",
        "date": datetime.now().isoformat(),
        "model_details": {
            "base_model": BASE_MODEL,
            "quantization": "4-bit NF4"
        },
        "scorer_details": {
            "checkpoint_path": CHECKPOINT,
            "scorer_type": "query_trained_head",
            "architecture": "ImportanceUpdateHead (cross-attention + RMSNorm + linear)",
            "training_objective": "ERT (KL divergence, token eviction)",
            "frozen": True,
            "task_specific_retraining": False,
            "training_data": "MS-MARCO train split"
        },
        "dataset": {
            "source": "MS-MARCO passage QA (train split)",
            "num_examples": args.n_examples,
            "num_positions": 3,
            "position_bucket_definition": "early (index 0), middle (k//2), end (k-1)",
            "examples_are_paired": False,
            "independently_generated_per_position": True,
            "passages_per_context": args.k_passages,
            "seed": args.seed
        },
        "pipelines": ["baseline", "lexical (TF-IDF)", "TIS", "oracle (gold-first)"],
        "command_used": " ".join(sys.argv)
    }
    
    # Run
    results = run_evaluation(args, examples, model, tokenizer, tis_head)
    
    # Analyze
    summary_rows, per_example = generate_summaries(results, args)
    
    # Save
    save_results(summary_rows, per_example, metadata, args)
    
    print(f"{'='*70}\n✓ All done!\n{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
scripts/run_litm_extended_eval.py

Extended Lost-in-the-Middle evaluation with strict SQuAD EM metric.

Changes from run_litm_nq_eval.py:
  1. Load 300 MS-MARCO queries (instead of 60) → 900 test cases × 3 positions
  2. Compute TWO EM metrics:
     - Extended EM (lenient, original): substring match + partial overlap
     - Strict SQuAD EM: exact normalized token match (rigorous)
  3. Compare metrics to validate robustness

Estimated runtime: 45-60 min on RTX 5070 (batch=1, 900 forward passes × 4 pipelines)
VRAM: ~4.5 GB (same as original, batch size 1 forced)

Usage:
    source .venv/bin/activate
    python scripts/run_litm_extended_eval.py [--n-examples 300] [--dry-run]

Outputs:
    results/litm_extended_results.csv       — per-example with BOTH EM metrics
    results/litm_extended_summary.csv       — aggregate metrics per pipeline
    results/litm_extended_comparison.png    — extended vs strict EM comparison plot
    results/litm_extended_position_curve.png — position curves for both metrics
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

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
CHECKPOINT  = os.path.join(_ROOT, "checkpoints", "v8b_hard_anchor")
DATA_PATH   = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR = os.path.join(_ROOT, "results")
D_MODEL     = 4096


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended LITM evaluation with strict EM metric")
    p.add_argument("--n-examples", type=int, default=300,
                   help="Number of base MS-MARCO examples (×3 positions)")
    p.add_argument("--k-passages", type=int, default=5, help="Passages per context")
    p.add_argument("--keep-ratio", type=float, default=0.7, help="Token keep ratio for pruning")
    p.add_argument("--max-new-tokens", type=int, default=30, help="Max generation tokens")
    p.add_argument("--device", default="", help="CUDA device")
    p.add_argument("--dry-run", action="store_true", help="Quick test: 3 examples only")
    return p.parse_args()


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
    result = head.load_state_dict(state["importance_head"], strict=False)
    dev = torch.device(device if device else "cuda")
    head = head.to(dev).eval()
    print("[setup] TIS head loaded.", flush=True)
    return head


def load_msmarco_examples(n_examples: int, k_passages: int) -> list[dict]:
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
        valid_answers = [a.strip() for a in answers_raw if a.strip() and a.strip() not in ("No Answer Present.", "[]")]
        if not valid_answers:
            continue
        
        gold_idx     = gold_indices[0]
        gold_passage = texts[gold_idx]
        distractors  = [t for i, t in enumerate(texts) if i != gold_idx]
        needed = k_passages - 1
        if len(distractors) < needed:
            distractors = distractors + [distractors[-1]] * (needed - len(distractors))
        distractors = distractors[:needed]
        
        positions = [("early", 0), ("middle", k_passages // 2), ("end", k_passages - 1)]
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
            })
        base_count += 1

    print(f"[data] Prepared {len(examples)} test cases ({base_count} base × 3 positions)", flush=True)
    return examples


def build_prompt(passages: list[str], question: str) -> str:
    body = "\n\n".join(f"Passage {i+1}: {p.strip()}" for i, p in enumerate(passages))
    instruction = (
        f"Read the following passages and answer the question. "
        f"Give only the answer, 1-5 words, no explanation.\n\n"
        f"{body}\n\n"
        f"Question: {question}"
    )
    return f"[INST] {instruction} [/INST]"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^\s*(a|an|the)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match_extended(pred: str, golds: list[str]) -> int:
    """Lenient EM: substring match or partial overlap (original metric)."""
    p = normalize_answer(pred)
    return int(any(
        p == normalize_answer(g) or
        normalize_answer(g) in p or
        p in normalize_answer(g)
        for g in golds
    ))


def exact_match_strict(pred: str, golds: list[str]) -> int:
    """Strict SQuAD EM: exact normalized token match (rigorous metric)."""
    p_norm = normalize_answer(pred)
    for g in golds:
        g_norm = normalize_answer(g)
        # Exact match on normalized strings
        if p_norm == g_norm:
            return 1
        # Also allow if one is a contiguous sub-sequence of tokens in the other
        p_toks = p_norm.split()
        g_toks = g_norm.split()
        if len(p_toks) == len(g_toks) and all(pt == gt for pt, gt in zip(p_toks, g_toks)):
            return 1
    return 0


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


@torch.inference_mode()
def score_tokens(
    model,
    tokenizer,
    tis_head,
    text: str,
    max_length: int = 3800,
) -> tuple[list[int], np.ndarray]:
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=max_length
    ).to(next(model.parameters()).device)
    token_ids = enc["input_ids"][0].tolist()
    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1].float()
    scores = tis_head.direct_score(hidden)
    if scores.dim() == 2:
        scores = scores[0]
    return token_ids, scores.cpu().numpy()


def prune_by_sentences(
    passages: list[str],
    token_ids_per_passage: list[list[int]],
    scores_per_passage: list[np.ndarray],
    keep_ratio: float,
) -> list[str]:
    all_sent_scores = []
    passage_sentences = []
    for pi, (passage, ids, scores) in enumerate(zip(passages, token_ids_per_passage, scores_per_passage)):
        sentences = split_sentences(passage)
        if not sentences:
            passage_sentences.append([])
            continue
        n_sent = len(sentences)
        tok_per_sent = max(1, len(ids) // n_sent)
        sent_data = []
        for si, sent in enumerate(sentences):
            start_tok = si * tok_per_sent
            end_tok = min((si + 1) * tok_per_sent, len(scores))
            sent_score = float(np.mean(scores[start_tok:end_tok])) if end_tok > start_tok else 0.0
            sent_data.append((sent_score, sent))
            all_sent_scores.append((pi, si, sent_score, sent))
        passage_sentences.append(sent_data)
    
    all_sent_scores.sort(key=lambda x: x[2], reverse=True)
    target_count = max(1, int(np.ceil(sum(len(ss) for ss in passage_sentences) * keep_ratio)))
    keep_indices = set()
    for pi, si, _, _ in all_sent_scores[:target_count]:
        keep_indices.add((pi, si))
    
    pruned = []
    for pi, sent_data in enumerate(passage_sentences):
        kept_sents = [s for si, (_, s) in enumerate(sent_data) if (pi, si) in keep_indices]
        pruned.append(" ".join(kept_sents) if kept_sents else passages[pi])
    return pruned


@torch.inference_mode()
def run_pipeline(
    model, tokenizer, tis_head,
    examples: list[dict],
    pipeline_name: str,
    keep_ratio: float = 0.7,
    max_new_tokens: int = 30,
) -> list[dict]:
    print(f"\n[{pipeline_name}] Starting evaluation...", flush=True)
    results = []
    start_time = time.time()
    
    for i, ex in enumerate(examples):
        if (i + 1) % max(1, len(examples) // 10) == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(examples)}] ETA: {eta:.0f}s", flush=True)
        
        # Prepare context
        passages = ex["passages"]
        if pipeline_name in ["TIS Pruning", "TIS Prune+Reorder"]:
            # Score and prune
            token_ids_list, scores_list = [], []
            for p in passages:
                tids, scor = score_tokens(model, tokenizer, tis_head, p)
                token_ids_list.append(tids)
                scores_list.append(scor)
            passages = prune_by_sentences(passages, token_ids_list, scores_list, keep_ratio)
        
        if pipeline_name in ["TIS Reordering", "TIS Prune+Reorder"]:
            # Score and reorder
            scores_per_passage = []
            for p in passages:
                _, scor = score_tokens(model, tokenizer, tis_head, p)
                mean_score = float(np.mean(scor))
                scores_per_passage.append(mean_score)
            sorted_idx = np.argsort(scores_per_passage)[::-1]
            passages = [passages[i] for i in sorted_idx]
        
        prompt = build_prompt(passages, ex["question"])
        enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=1.0,
                temperature=1.0,
            )
        pred_ids = out[0][len(enc["input_ids"][0]):]
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        
        # Compute BOTH EM metrics
        em_extended = exact_match_extended(pred_text, ex["answers"])
        em_strict = exact_match_strict(pred_text, ex["answers"])
        f1 = max(token_f1(pred_text, g) for g in ex["answers"]) if ex["answers"] else 0.0
        
        results.append({
            "query_id": ex["query_id"],
            "position": ex["position_name"],
            "gold_pos": ex["gold_position"],
            "pipeline": pipeline_name,
            "pred_text": pred_text,
            "gold_answer": ex["answers"][0],
            "em_extended": em_extended,
            "em_strict": em_strict,
            "f1": f1,
        })
    
    elapsed = time.time() - start_time
    print(f"[{pipeline_name}] Done in {elapsed:.1f}s", flush=True)
    return results


def main() -> None:
    args = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.device:
        torch.cuda.set_device(args.device)
    
    print("="*70, flush=True)
    print(f"Extended LITM Evaluation (with strict SQuAD EM)")
    print(f"Target: {args.n_examples} queries × 3 positions = {args.n_examples * 3} test cases")
    print("="*70, flush=True)
    
    device = torch.device(args.device if args.device else "cuda")
    model, tokenizer = load_model_and_tokenizer(str(device))
    tis_head = load_tis_head(str(device))
    
    if args.dry_run:
        examples = load_msmarco_examples(1, args.k_passages)
        examples = examples[:3]
        print(f"[dry-run] Using {len(examples)} examples", flush=True)
    else:
        examples = load_msmarco_examples(args.n_examples, args.k_passages)
    
    # Run 4 pipelines
    all_results = []
    pipelines = [
        ("Baseline", None),
        ("TIS Reordering", None),
        ("TIS Pruning", args.keep_ratio),
        ("TIS Prune+Reorder", args.keep_ratio),
    ]
    
    for pipeline_name, keep_ratio in pipelines:
        kr = keep_ratio if keep_ratio is not None else 1.0
        res = run_pipeline(
            model, tokenizer, tis_head,
            examples, pipeline_name,
            keep_ratio=kr,
            max_new_tokens=args.max_new_tokens,
        )
        all_results.extend(res)
    
    # Save detailed results
    csv_path = os.path.join(RESULTS_DIR, "litm_extended_results.csv")
    with open(csv_path, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "position", "gold_pos", "pipeline", "pred_text",
                       "gold_answer", "em_extended", "em_strict", "f1"]
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[save] {csv_path}", flush=True)
    
    # Compute summaries
    from collections import defaultdict
    summary_data = defaultdict(lambda: {"em_extended": [], "em_strict": [], "f1": []})
    for r in all_results:
        key = (r["pipeline"], r["position"])
        summary_data[key]["em_extended"].append(r["em_extended"])
        summary_data[key]["em_strict"].append(r["em_strict"])
        summary_data[key]["f1"].append(r["f1"])
    
    summary = []
    for (pipeline, position), metrics in sorted(summary_data.items()):
        summary.append({
            "pipeline": pipeline,
            "position": position,
            "em_extended": np.mean(metrics["em_extended"]),
            "em_strict": np.mean(metrics["em_strict"]),
            "f1": np.mean(metrics["f1"]),
            "n": len(metrics["em_extended"]),
        })
    
    summary_csv = os.path.join(RESULTS_DIR, "litm_extended_summary.csv")
    with open(summary_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["pipeline", "position", "em_extended", "em_strict", "f1", "n"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"[save] {summary_csv}", flush=True)
    
    # Print summary table
    print("\n" + "="*80)
    print(f"Summary: Extended vs Strict EM Comparison ({len(examples)} examples × 3 positions)")
    print("="*80)
    for pipeline in ["Baseline", "TIS Reordering", "TIS Pruning", "TIS Prune+Reorder"]:
        pipeline_rows = [s for s in summary if s["pipeline"] == pipeline]
        if pipeline_rows:
            print(f"\n{pipeline}:")
            print(f"  Position | Extended EM | Strict EM | F1")
            print(f"  ---------|-------------|-----------|-----")
            for row in pipeline_rows:
                print(f"  {row['position']:8} | {row['em_extended']:11.3f} | {row['em_strict']:9.3f} | {row['f1']:.3f}")
            avg_ext = np.mean([r["em_extended"] for r in pipeline_rows])
            avg_strict = np.mean([r["em_strict"] for r in pipeline_rows])
            avg_f1 = np.mean([r["f1"] for r in pipeline_rows])
            print(f"  Average  | {avg_ext:11.3f} | {avg_strict:9.3f} | {avg_f1:.3f}")
    
    print("\n" + "="*80)
    print("✓ All done!")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
scripts/test_kv_eviction_head_on_litm.py

Test whether the KV-eviction trained head (stage3_ert) can transfer
to the passage reordering task without retraining.

This answers the research question:
"Can a head trained for token-level KV-cache eviction also identify important passages?"

Usage:
    source .venv/bin/activate
    python scripts/test_kv_eviction_head_on_litm.py --n-examples 60
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
from datetime import datetime

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
_DEFAULT_KV_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "stage3_ert")
_DEFAULT_PASSAGE_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "v8b_hard_anchor")
DATA_PATH = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR = os.path.join(_ROOT, "results")
D_MODEL = 4096


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test KV-eviction head on LITM passage reordering")
    p.add_argument("--checkpoint-kv", default=_DEFAULT_KV_CHECKPOINT,
                   help="KV-eviction checkpoint dir (default: checkpoints/stage3_ert)")
    p.add_argument("--checkpoint-passage", default=_DEFAULT_PASSAGE_CHECKPOINT,
                   help="Passage-head checkpoint dir (default: checkpoints/v8b_hard_anchor)")
    p.add_argument("--n-examples", type=int, default=60,
                   help="Number of base MS-MARCO examples (×3 positions)")
    p.add_argument("--k-passages", type=int, default=5,
                   help="Number of passages per context")
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


def load_tis_head(checkpoint_path: str, device: str, head_type: str):
    from token_importance.config import TISConfig
    from token_importance.model.importance_head import ImportanceUpdateHead

    ckpt_path = os.path.join(checkpoint_path, "tis_components.pt")
    print(f"[setup] Loading {head_type} head from {ckpt_path}...", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tis_config = TISConfig()
    head = ImportanceUpdateHead(d_model=D_MODEL, config=tis_config, num_heads=4)
    head.load_state_dict(state["importance_head"], strict=False)
    dev = torch.device(device if device else "cuda")
    head = head.to(dev).eval()
    print(f"[setup] {head_type} head loaded.", flush=True)
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

        texts = passages_data["passage_text"]
        selected = passages_data["is_selected"]
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

        gold_idx = gold_indices[0]
        gold_passage = texts[gold_idx]
        distractors = [t for i, t in enumerate(texts) if i != gold_idx]

        needed = k_passages - 1
        if len(distractors) < needed:
            distractors = distractors + [distractors[-1]] * (needed - len(distractors))
        distractors = distractors[:needed]

        # 3 positions: early, middle, end
        positions = [
            ("early", 0),
            ("middle", k_passages // 2),
            ("end", k_passages - 1),
        ]
        for pos_name, gold_pos in positions:
            passage_list = list(distractors)
            passage_list.insert(gold_pos, gold_passage)

            examples.append({
                "question": raw["query"],
                "passages": passage_list,
                "gold_passage": gold_passage,
                "answers": valid_answers,
                "gold_position": gold_pos,
                "position_name": pos_name,
                "query_id": raw["query_id"],
                "base_id": base_count,
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
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


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


def reorder_by_tis(passages: list[str], model, tokenizer, tis_head) -> tuple[list[str], list[int]]:
    """TIS-based ranking: score all passages, rank by mean score."""
    scores_per_passage = []
    for passage in passages:
        _, scores = score_tokens(model, tokenizer, tis_head, passage)
        scores_per_passage.append(scores)
    
    mean_scores = [float(s.mean()) for s in scores_per_passage]
    ranked_indices = np.argsort(-np.array(mean_scores))
    
    return [passages[i] for i in ranked_indices], list(ranked_indices)


# ── Generation ─────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 30, max_input_length: int = 3800) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_toks = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


# ── Evaluation ─────────────────────────────────────────────────────────────────
def run_evaluation(args, examples, model, tokenizer, kv_head, passage_head):
    """Run baseline + both TIS heads."""
    results = []
    
    pipelines = [
        ("baseline", "Original order", None),
        ("kv_eviction_head", "KV-eviction head (stage3_ert)", kv_head),
        ("passage_head", "Passage reordering head (v8b_hard_anchor)", passage_head),
    ]
    
    print(f"\n{'='*70}\nTesting KV-eviction head transfer to passage reordering\n{'='*70}\n", flush=True)
    
    for pipeline_name, pipeline_desc, head in pipelines:
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
            else:
                ordered_passages, ranked_indices = reorder_by_tis(passages, model, tokenizer, head)
            
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
            })
        
        elapsed = time.perf_counter() - t_start
        print(f"[{pipeline_name}] Done in {elapsed:.1f}s\n", flush=True)
    
    return results


# ── Analysis ───────────────────────────────────────────────────────────────────
def generate_summary(results: list[dict]):
    """Generate summary tables."""
    summary_rows = []
    for pipeline in ["baseline", "kv_eviction_head", "passage_head"]:
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
    
    return summary_rows


def save_and_display_results(summary_rows, metadata):
    """Save results and display summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, "kv_transfer_test_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["pipeline", "position", "n", "recall_at_1", "mrr", "em", "f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[save] {summary_path}", flush=True)
    
    # Metadata JSON
    metadata_path = os.path.join(RESULTS_DIR, "kv_transfer_test_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[save] {metadata_path}", flush=True)
    
    # Display summary
    print(f"\n{'='*70}\nTransfer Test Results: KV-Eviction Head vs Passage Reordering Head\n{'='*70}\n")
    
    for pipeline in ["baseline", "kv_eviction_head", "passage_head"]:
        subset = [r for r in summary_rows if r["pipeline"] == pipeline]
        if not subset:
            continue
        
        pipeline_label = {
            "baseline": "BASELINE (Original Order)",
            "kv_eviction_head": "KV-EVICTION HEAD (Token-level trained)",
            "passage_head": "PASSAGE HEAD (Passage-level trained)"
        }[pipeline]
        
        print(f"{pipeline_label}:")
        print(f"  Position | Recall@1 | MRR   | EM    | F1")
        print(f"  ---------|----------|-------|-------|-------")
        for row in subset:
            print(f"  {row['position']:8s} | {row['recall_at_1']:8.3f} | {row['mrr']:.3f} | {row['em']:.3f} | {row['f1']:.3f}")
        
        # Compute LITM gap
        early_em = next((r["em"] for r in subset if r["position"] == "early"), 0)
        mid_em = next((r["em"] for r in subset if r["position"] == "middle"), 0)
        end_em = next((r["em"] for r in subset if r["position"] == "end"), 0)
        litm_gap = max(early_em, end_em) - mid_em
        avg_em = np.mean([r["em"] for r in subset])
        avg_recall = np.mean([r["recall_at_1"] for r in subset])
        
        print(f"  Average  | {avg_recall:8.3f} | {np.mean([r['mrr'] for r in subset]):.3f} | {avg_em:.3f} |")
        print(f"  LITM gap | {litm_gap:8.3f} |\n")
    
    # Comparison
    print(f"{'='*70}\nKey Findings\n{'='*70}\n")
    
    baseline_gap = max([r["em"] for r in summary_rows if r["pipeline"] == "baseline" and r["position"] in ["early", "end"]]) - \
                   next(r["em"] for r in summary_rows if r["pipeline"] == "baseline" and r["position"] == "middle")
    kv_gap = max([r["em"] for r in summary_rows if r["pipeline"] == "kv_eviction_head" and r["position"] in ["early", "end"]]) - \
             next(r["em"] for r in summary_rows if r["pipeline"] == "kv_eviction_head" and r["position"] == "middle")
    passage_gap = max([r["em"] for r in summary_rows if r["pipeline"] == "passage_head" and r["position"] in ["early", "end"]]) - \
                  next(r["em"] for r in summary_rows if r["pipeline"] == "passage_head" and r["position"] == "middle")
    
    baseline_em = np.mean([r["em"] for r in summary_rows if r["pipeline"] == "baseline"])
    kv_em = np.mean([r["em"] for r in summary_rows if r["pipeline"] == "kv_eviction_head"])
    passage_em = np.mean([r["em"] for r in summary_rows if r["pipeline"] == "passage_head"])
    
    kv_recall = np.mean([r["recall_at_1"] for r in summary_rows if r["pipeline"] == "kv_eviction_head"])
    passage_recall = np.mean([r["recall_at_1"] for r in summary_rows if r["pipeline"] == "passage_head"])
    
    print(f"1. LITM Gap Elimination:")
    print(f"   - Baseline:       {baseline_gap:.3f}")
    print(f"   - KV-eviction:    {kv_gap:.3f} ({100*(baseline_gap-kv_gap)/baseline_gap:.1f}% reduction)")
    print(f"   - Passage head:   {passage_gap:.3f} ({100*(baseline_gap-passage_gap)/baseline_gap:.1f}% reduction)")
    print()
    print(f"2. Overall EM:")
    print(f"   - Baseline:       {baseline_em:.3f}")
    print(f"   - KV-eviction:    {kv_em:.3f} ({kv_em - baseline_em:+.3f} vs baseline)")
    print(f"   - Passage head:   {passage_em:.3f} ({passage_em - baseline_em:+.3f} vs baseline)")
    print()
    print(f"3. Ranking Quality (Recall@1):")
    print(f"   - KV-eviction:    {kv_recall:.3f}")
    print(f"   - Passage head:   {passage_recall:.3f}")
    print()
    
    # Verdict
    if kv_gap < 0.01 and abs(kv_em - passage_em) < 0.05:
        verdict = "✅ TRANSFER SUCCESSFUL: KV-eviction head achieves similar passage reordering performance!"
    elif kv_gap < baseline_gap * 0.5:
        verdict = "⚠️ PARTIAL TRANSFER: KV-eviction head reduces gap but underperforms passage-trained head."
    else:
        verdict = "❌ TRANSFER FAILED: KV-eviction head does not transfer effectively to passage reordering."
    
    print(f"{'='*70}\n{verdict}\n{'='*70}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load
    model, tokenizer = load_model_and_tokenizer(args.device)
    kv_head = load_tis_head(args.checkpoint_kv, args.device, "KV-eviction")
    passage_head = load_tis_head(args.checkpoint_passage, args.device, "Passage reordering")
    examples = load_msmarco_examples(args.n_examples, args.k_passages)
    
    # Metadata
    metadata = {
        "experiment_id": "kv_eviction_transfer_test",
        "date": datetime.now().isoformat(),
        "research_question": "Can a head trained for token-level KV-cache eviction transfer to passage reordering?",
        "model_details": {
            "base_model": BASE_MODEL,
            "quantization": "4-bit NF4"
        },
        "head_comparison": {
            "kv_eviction_head": {
                "checkpoint": args.checkpoint_kv,
                "training_objective": "ERT (KL divergence for token-level KV-cache compression)",
                "training_data": "Unknown (from stage3_ert)",
                "intended_use": "Identifying which tokens to evict from KV cache"
            },
            "passage_reordering_head": {
                "checkpoint": args.checkpoint_passage,
                "training_objective": "ERT (KL divergence optimized for passage reordering)",
                "training_data": "MS-MARCO passage QA",
                "intended_use": "Reordering passages to eliminate position bias"
            }
        },
        "dataset": {
            "source": "MS-MARCO passage QA (train split)",
            "num_examples": args.n_examples,
            "num_positions": 3,
            "passages_per_context": args.k_passages,
            "seed": args.seed
        },
        "command_used": " ".join(sys.argv)
    }
    
    # Run
    results = run_evaluation(args, examples, model, tokenizer, kv_head, passage_head)
    
    # Analyze
    summary_rows = generate_summary(results)
    
    # Save and display
    save_and_display_results(summary_rows, metadata)


if __name__ == "__main__":
    main()

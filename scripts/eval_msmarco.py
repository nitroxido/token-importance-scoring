#!/usr/bin/env python
"""MS MARCO generalization test for closed-loop TIS.

Tests whether a scorer trained on synthetic retrieval data generalises
to real questions and passages from MS MARCO (Microsoft Machine Reading
Comprehension dataset).

Conditions: heuristic TIS, learned TIS, SnapKV proxy, no eviction.
Budgets: 0.25, 0.50, 0.75.

Usage:
    source .venv/bin/activate
    python scripts/eval_msmarco.py \\
        --learned-checkpoint checkpoints/closed_loop_retrieval_v6/ \\
        --data-dir data/msmarco_quick/val \\
        --num-tests 50 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM

# Anchor protection (same as eval_niah_hard)
N_SINK = 4
N_RECENT = 30


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MS MARCO generalisation eval")
    p.add_argument("--learned-checkpoint", required=True)
    p.add_argument("--data-dir", default="data/msmarco_quick/train")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--num-tests", type=int, default=50)
    p.add_argument("--context-tokens", type=int, default=1536,
                   help="Max tokens per example (shorter than NIAH to fit VRAM)")
    p.add_argument("--device", default="")
    return p.parse_args(argv)


def _load_model(checkpoint: Path, device: torch.device) -> PatchedCausalLM:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = PatchedCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3", config=TISConfig(),
        quantization_config=quant_cfg, device_map=device,
    ).to(device)
    tis_path = checkpoint / "tis_components.pt"
    if tis_path.exists():
        state = torch.load(tis_path, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        if "attn_hook_lambda" in state:
            model.attn_hook._lambda.data = state["attn_hook_lambda"].to(device)
    return model


def _load_msmarco(data_dir: str, num_tests: int) -> list[dict]:
    """Load MS MARCO examples with answers.  Returns list of dicts with
    keys: query, answer, context (combined passages).
    """
    try:
        ds = load_from_disk(data_dir)
    except Exception as e:
        raise RuntimeError(f"Could not load MS MARCO from {data_dir}: {e}")

    examples = []
    for row in ds:
        # MS MARCO v1.1: row['answers'] is a list, passages is a dict
        answers = row.get("answers", [])
        if not answers or answers[0] in ("No Answer Present.", "", None):
            continue

        passages = row.get("passages", {})
        texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
        selected = passages.get("is_selected", []) if isinstance(passages, dict) else []

        if not texts:
            continue

        # Put selected passages first, then non-selected (max 6 total)
        ordered = [(t, s) for t, s in zip(texts, selected)]
        ordered.sort(key=lambda x: -x[1])  # selected first
        context_parts = [t for t, _ in ordered[:6]]
        context = " ".join(context_parts)

        examples.append({
            "query": row["query"],
            "answer": answers[0],
            "context": context,
        })
        if len(examples) >= num_tests:
            break

    print(f"[msmarco] Loaded {len(examples)} examples from {data_dir}", flush=True)
    return examples


def _build_prompt(query: str, context: str, tokenizer, max_length: int) -> torch.Tensor:
    """Tokenize: [context] + question + Answer: """
    prompt = (
        f"Read the following passages and answer the question.\n\n"
        f"Passages: {context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    if ids.shape[0] > max_length:
        # Keep beginning context + full question (last 80 tokens)
        ids = torch.cat([ids[:max_length - 80], ids[-80:]])
    return ids.unsqueeze(0)  # [1, T]


# ── Scoring (same logic as eval_niah_hard) ────────────────────────────────────

def _get_hidden(model, input_ids, device):
    T = input_ids.shape[1]
    seed = torch.full((T,), 50, dtype=torch.uint8, device=device)
    mask = torch.ones(1, T, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, importance_scores=seed,
                    attention_mask=mask, output_hidden_states=True)
    return out.hidden_states[-1].float().squeeze(0), mask  # [T, d], [1, T]


def _heuristic_scores(T, device):
    s = torch.full((T,), 40.0, device=device)
    e = max(1, T // 10)
    s[:e] = 80.0; s[-e:] = 80.0
    return s


def _learned_scores(hidden, model):
    with torch.no_grad():
        raw = model.importance_head.out_proj(hidden.unsqueeze(0))
        return torch.sigmoid(raw.squeeze(-1).squeeze(0)) * 100.0


def _snapkv_scores(hidden, device):
    T = hidden.shape[0]
    norms = hidden.norm(dim=-1).float()
    recency = torch.linspace(0.5, 1.5, T, device=device)
    raw = norms * recency
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        return (raw - lo) / (hi - lo) * 100.0
    return torch.full((T,), 50.0, device=device)


def _anchored_mask(scores, budget, T, device):
    n_anchor = N_SINK + N_RECENT
    n_keep = max(n_anchor + 1, int(T * budget))
    n_content = n_keep - n_anchor
    mask = torch.zeros(T, dtype=torch.long, device=device)
    mask[:N_SINK] = 1; mask[-N_RECENT:] = 1
    c_start, c_end = N_SINK, T - N_RECENT
    c_len = max(0, c_end - c_start)
    if n_content > 0 and c_len > 0:
        k = min(n_content, c_len)
        _, idx = torch.topk(scores[c_start:c_end].float(), k=k)
        mask[c_start + idx] = 1
    return mask.unsqueeze(0)


def _predict_top5(model, input_ids, scores, keep_mask, device):
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    importance_scores=scores.clamp(0, 100).to(torch.uint8),
                    attention_mask=keep_mask)
    return torch.topk(out.logits[0, -1, :], k=5).indices.cpu().tolist()


def _answer_hit(top5, answer, tokenizer):
    for aid in tokenizer.encode(answer, add_special_tokens=False)[:3]:  # First 3 answer tokens
        if aid in top5:
            return True
    return False


def main():
    args = _parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    ckpt = Path(args.learned_checkpoint)
    print(f"[msmarco] Loading model from {ckpt}", flush=True)
    model = _load_model(ckpt, device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    examples = _load_msmarco(args.data_dir, args.num_tests)
    if not examples:
        print("[msmarco] No examples loaded, aborting.", flush=True)
        return

    results: dict = {c: {} for c in ("heuristic", "learned", "snapkv", "no_eviction")}

    for budget in args.budgets:
        print(f"\n[msmarco] Budget {budget:.0%}", flush=True)
        counters = {c: 0 for c in ("heuristic", "learned", "snapkv", "no_eviction")}

        for ex in examples:
            input_ids = _build_prompt(ex["query"], ex["context"], tokenizer, args.context_tokens)
            input_ids = input_ids.to(device)
            T = input_ids.shape[1]

            hidden, full_mask = _get_hidden(model, input_ids, device)
            torch.cuda.empty_cache()

            for cname, score_fn in (
                ("heuristic", lambda: _heuristic_scores(T, device)),
                ("learned",   lambda: _learned_scores(hidden, model)),
                ("snapkv",    lambda: _snapkv_scores(hidden, device)),
            ):
                scores = score_fn()
                mask = _anchored_mask(scores, budget, T, device)
                top5 = _predict_top5(model, input_ids, scores, mask, device)
                if _answer_hit(top5, ex["answer"], tokenizer):
                    counters[cname] += 1

            # No eviction
            n_scores = torch.full((T,), 50, dtype=torch.uint8, device=device)
            n_top5 = _predict_top5(model, input_ids, n_scores, full_mask, device)
            if _answer_hit(n_top5, ex["answer"], tokenizer):
                counters["no_eviction"] += 1

            torch.cuda.empty_cache()

        N = len(examples)
        for cname in ("heuristic", "learned", "snapkv", "no_eviction"):
            acc = 100.0 * counters[cname] / N
            results[cname][str(budget)] = {"accuracy": acc}

        print(f"  Heuristic : {results['heuristic'][str(budget)]['accuracy']:5.1f}%")
        print(f"  Learned   : {results['learned'][str(budget)]['accuracy']:5.1f}%")
        print(f"  SnapKV    : {results['snapkv'][str(budget)]['accuracy']:5.1f}%")
        print(f"  No-evict  : {results['no_eviction'][str(budget)]['accuracy']:5.1f}%  (reference)")

    out_file = ckpt / "msmarco_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ MS MARCO results saved to {out_file}", flush=True)

    print("\n── MS MARCO Summary ────────────────────────────────────────────────────────")
    print(f"{'Budget':>8}  {'Heuristic':>10}  {'Learned':>10}  {'SnapKV':>10}  {'No-evict':>10}  {'Δ(L-H)':>8}")
    for b in args.budgets:
        h = results["heuristic"][str(b)]["accuracy"]
        l = results["learned"][str(b)]["accuracy"]
        s = results["snapkv"][str(b)]["accuracy"]
        n = results["no_eviction"][str(b)]["accuracy"]
        print(f"{b:>8.0%}  {h:>9.1f}%  {l:>9.1f}%  {s:>9.1f}%  {n:>9.1f}%  {l-h:>+7.1f}%")


if __name__ == "__main__":
    main()

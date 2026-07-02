#!/usr/bin/env python
"""Harder NIAH-style retrieval evaluator — 4-way comparison.

Conditions:
  heuristic   — position-based (first/last 10% → high score)
  learned     — importance_head.out_proj(hidden) trained via closed-loop retrieval
  snapkv      — hidden-state L2-norm × recency weight (proxy for SnapKV attention-score policy)
  no_eviction — full context, no KV eviction (upper-bound reference)

All eviction methods share the same anchor protection: first N_SINK + last N_RECENT
tokens are always kept, so the question (always at end) and attention sinks are
never evicted regardless of budget. The remaining budget is allocated by each
method's scoring. This makes it a fair comparison of content-scoring ability.

Usage:
    source .venv/bin/activate
    python scripts/eval_niah_hard.py \\
        --learned-checkpoint checkpoints/closed_loop_retrieval_v6/ \\
        --budgets 0.1 0.25 0.5 0.75 \\
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
from transformers import AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.retrieval_data import RetrievalDataset

# Anchor tokens always kept regardless of budget
N_SINK = 4      # First N tokens (attention sink anchors)
N_RECENT = 30   # Last N tokens (covers the question + "Answer:" prompt)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harder NIAH 4-way evaluation")
    p.add_argument("--learned-checkpoint", required=True)
    p.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75])
    p.add_argument("--num-tests", type=int, default=50)
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _load_model(checkpoint: Path, device: torch.device) -> PatchedCausalLM:
    model_name = "mistralai/Mistral-7B-v0.3"
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = PatchedCausalLM.from_pretrained(
        model_name, config=TISConfig(), quantization_config=quant_cfg, device_map=device,
    ).to(device)
    tis_path = checkpoint / "tis_components.pt"
    if tis_path.exists():
        state = torch.load(tis_path, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        if "attn_hook_lambda" in state:
            model.attn_hook._lambda.data = state["attn_hook_lambda"].to(device)
    return model


# ── Scoring functions ──────────────────────────────────────────────────────────

def _get_hidden(model, input_ids, attention_mask, device):
    """Run forward pass once; return last hidden states (detached, float32)."""
    seq_len = input_ids.shape[1]
    seed = torch.full((seq_len,), 50, dtype=torch.uint8, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, importance_scores=seed,
                    attention_mask=attention_mask, output_hidden_states=True)
    return out.hidden_states[-1].float().squeeze(0)   # [T, d]


def _heuristic_scores(seq_len: int, device: torch.device) -> torch.Tensor:
    """Position-aware: first/last 10% → 80, middle → 40."""
    scores = torch.full((seq_len,), 40, dtype=torch.float32, device=device)
    n_edge = max(1, seq_len // 10)
    scores[:n_edge] = 80.0
    scores[-n_edge:] = 80.0
    return scores


def _learned_scores(hidden: torch.Tensor, model: PatchedCausalLM) -> torch.Tensor:
    """Learned: out_proj(hidden) — same path used during training."""
    with torch.no_grad():
        raw = model.importance_head.out_proj(hidden.unsqueeze(0))  # [1, T, 1]
        return torch.sigmoid(raw.squeeze(-1).squeeze(0)) * 100.0   # [T] float


def _snapkv_scores(hidden: torch.Tensor, device: torch.device) -> torch.Tensor:
    """SnapKV proxy: hidden-state L2-norm × recency weight.

    Tokens that are more 'active' (high norm) AND closer to the query position
    (higher recency weight) are considered more important — mirrors SnapKV's
    attention-from-instruction-region scoring without extracting attention weights.
    """
    T = hidden.shape[0]
    norms = hidden.norm(dim=-1).float()                    # [T]
    recency = torch.linspace(0.5, 1.5, T, device=device)  # [T]
    raw = norms * recency
    # Normalize to [0, 100]
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        return (raw - lo) / (hi - lo) * 100.0
    return torch.full((T,), 50.0, device=device)


# ── Anchored eviction mask ─────────────────────────────────────────────────────

def _anchored_mask(scores: torch.Tensor, budget: float, seq_len: int,
                   device: torch.device) -> torch.Tensor:
    """Build keep mask with guaranteed sink + recent anchors.

    1. Always keep first N_SINK tokens (attention sink).
    2. Always keep last N_RECENT tokens (covers the question + Answer: prompt).
    3. Allocate remaining budget to top-scoring content positions.

    Returns [1, seq_len] int64 mask.
    """
    n_anchor = N_SINK + N_RECENT
    n_total_keep = max(n_anchor + 1, int(seq_len * budget))
    n_content_keep = n_total_keep - n_anchor

    # Content positions: everything except anchor zones
    content_start = N_SINK
    content_end = seq_len - N_RECENT
    content_len = max(0, content_end - content_start)

    mask = torch.zeros(seq_len, dtype=torch.long, device=device)
    mask[:N_SINK] = 1
    mask[-N_RECENT:] = 1

    if n_content_keep > 0 and content_len > 0:
        content_scores = scores[content_start:content_end]
        k = min(n_content_keep, content_len)
        _, top_idx = torch.topk(content_scores.float(), k=k)
        mask[content_start + top_idx] = 1

    return mask.unsqueeze(0)


def _evidence_survival(scores, evidence_mask, budget, seq_len, device):
    """Fraction of evidence tokens in the anchored top-keep set."""
    keep = _anchored_mask(scores, budget, seq_len, device)[0].bool().cpu()
    ev = evidence_mask.bool().cpu()
    if ev.sum() == 0:
        return 1.0
    return (keep & ev).sum().item() / ev.sum().item()


def _predict_answer(model, input_ids, scores, keep_mask, device, tokenizer):
    """Return top-5 token IDs at the final position."""
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    importance_scores=scores.clamp(0, 100).to(torch.uint8),
                    attention_mask=keep_mask)
    logits = out.logits[0, -1, :]
    return torch.topk(logits, k=5).indices.cpu().tolist()


def _answer_in_topk(top_ids, answer, tokenizer):
    for aid in tokenizer.encode(answer, add_special_tokens=False):
        if aid in top_ids:
            return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    ckpt = Path(args.learned_checkpoint)
    print(f"[eval] Loading model from {ckpt}", flush=True)
    model = _load_model(ckpt, device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = RetrievalDataset(tokenizer=tokenizer, context_tokens=args.context_tokens,
                               budgets=args.budgets, seed=args.seed)
    print(f"[eval] Generating {args.num_tests} test examples...", flush=True)
    test_examples = [next(iter(dataset)) for _ in range(args.num_tests)]

    results: dict = {c: {} for c in ("heuristic", "learned", "snapkv", "no_eviction")}

    for budget in args.budgets:
        print(f"\n[eval] Budget {budget:.0%}", flush=True)
        counters = {c: {"correct": 0, "survive": 0.0} for c in ("heuristic", "learned", "snapkv")}
        n_correct = 0  # no_eviction

        for ex in test_examples:
            ids = ex.input_ids.to(device)
            attn = ex.attention_mask.to(device)
            ev = ex.evidence_mask.to(device)
            T = ids.shape[1]

            # Single forward pass; reuse hidden for all three learned conditions
            hidden = _get_hidden(model, ids, attn, device)   # [T, d] — shared
            torch.cuda.empty_cache()

            for cname, score_fn in (
                ("heuristic", lambda: _heuristic_scores(T, device)),
                ("learned",   lambda: _learned_scores(hidden, model)),
                ("snapkv",    lambda: _snapkv_scores(hidden, device)),
            ):
                scores = score_fn()
                keep_mask = _anchored_mask(scores, budget, T, device)
                top5 = _predict_answer(model, ids, scores, keep_mask, device, tokenizer)
                if _answer_in_topk(top5, ex.answer_text, tokenizer):
                    counters[cname]["correct"] += 1
                counters[cname]["survive"] += _evidence_survival(scores, ev, budget, T, device)

            # No-eviction: keep everything
            full_mask = torch.ones(1, T, dtype=torch.long, device=device)
            n_scores = torch.full((T,), 50, dtype=torch.uint8, device=device)
            n_top5 = _predict_answer(model, ids, n_scores, full_mask, device, tokenizer)
            n_correct += _answer_in_topk(n_top5, ex.answer_text, tokenizer)

            torch.cuda.empty_cache()

        N = args.num_tests
        for cname in ("heuristic", "learned", "snapkv"):
            acc = 100.0 * counters[cname]["correct"] / N
            surv = 100.0 * counters[cname]["survive"] / N
            results[cname][str(budget)] = {"accuracy": acc, "evidence_survival": surv}

        n_acc = 100.0 * n_correct / N
        results["no_eviction"][str(budget)] = {"accuracy": n_acc}

        print(f"  Heuristic : acc={results['heuristic'][str(budget)]['accuracy']:5.1f}%  ev_surv={results['heuristic'][str(budget)]['evidence_survival']:5.1f}%")
        print(f"  Learned   : acc={results['learned'][str(budget)]['accuracy']:5.1f}%  ev_surv={results['learned'][str(budget)]['evidence_survival']:5.1f}%")
        print(f"  SnapKV    : acc={results['snapkv'][str(budget)]['accuracy']:5.1f}%  ev_surv={results['snapkv'][str(budget)]['evidence_survival']:5.1f}%")
        print(f"  No-evict  : acc={n_acc:5.1f}%  (reference)")

    # Save JSON
    out_file = ckpt / "niah_hard_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_file}", flush=True)

    # Summary table
    print("\n── Summary ──────────────────────────────────────────────────────────────────")
    print(f"{'Budget':>8}  {'Heuristic':>10}  {'Learned':>10}  {'SnapKV':>10}  {'No-evict':>10}  {'Δ(L-H)':>8}")
    for b in args.budgets:
        h = results["heuristic"][str(b)]["accuracy"]
        l = results["learned"][str(b)]["accuracy"]
        s = results["snapkv"][str(b)]["accuracy"]
        n = results["no_eviction"][str(b)]["accuracy"]
        d = l - h
        print(f"{b:>8.0%}  {h:>9.1f}%  {l:>9.1f}%  {s:>9.1f}%  {n:>9.1f}%  {d:>+7.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Closed-loop retrieval training for TIS.

Optimizes the importance_head to produce token scores that:
  1. rank evidence tokens above distractors (ranking loss)
  2. preserve evidence tokens under budget eviction (retrieval preservation loss)
  3. remain stable across the sequence (optional regularisation)

Critically, the heuristic initialization is NOT used during training:
the model receives a uniform seed (score=50 everywhere) so the head
must learn discriminative scoring from context alone.

Usage (with venv):
    source .venv/bin/activate
    python scripts/train_closed_loop_retrieval.py \
        --base-checkpoint checkpoints/stage3_ert_local_fresh/ \
        --output-dir checkpoints/closed_loop_retrieval/ \
        --steps 2000 \
        --device cuda

The script logs metrics every 50 steps to stdout and to
<output-dir>/training.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.retrieval_data import RetrievalDataset


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop retrieval TIS training")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for importance_head")
    p.add_argument("--alpha-rank", type=float, default=1.0,
                   help="Weight for evidence ranking loss")
    p.add_argument("--beta-retrieve", type=float, default=2.0,
                   help="Weight for budgeted retrieval preservation loss")
    p.add_argument("--gamma-stability", type=float, default=0.05,
                   help="Weight for score stability regularisation")
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--budget-weights", nargs="+", type=float, default=None,
                   help="Non-uniform budget sampling weights. E.g. 4 1 1 trains 4x more at the first budget.")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--device", default="")
    return p.parse_args(argv)


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(checkpoint: Path, device: torch.device) -> PatchedCausalLM:
    model_name = "mistralai/Mistral-7B-v0.3"
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"[model] Loading {model_name} with 4-bit NF4 quantization...", flush=True)
    model = PatchedCausalLM.from_pretrained(
        model_name,
        config=TISConfig(),
        quantization_config=quant_cfg,
        device_map=device,
    ).to(device)

    tis_path = checkpoint / "tis_components.pt"
    if tis_path.exists():
        state = torch.load(tis_path, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        print(f"[checkpoint] ✓ Loaded TIS components from {checkpoint}", flush=True)
    else:
        print(f"[checkpoint] ⚠ No TIS checkpoint found at {checkpoint}, starting fresh", flush=True)

    return model


# ── Loss functions ─────────────────────────────────────────────────────────────

def _evidence_ranking_loss(
    scores: torch.Tensor,  # [T]  float, predicted per-token scores in [0,1]
    evidence_mask: torch.Tensor,  # [T] bool
) -> torch.Tensor:
    """Margin ranking loss: every evidence token should score higher than
    every distractor token.

    Uses a soft surrogate: pushes mean(evidence_scores) above mean(distractor_scores)
    with a margin of 0.2.  Scales more gracefully than exhaustive pair enumeration.
    """
    ev = scores[evidence_mask.bool()]
    dist = scores[~evidence_mask.bool()]

    if ev.numel() == 0 or dist.numel() == 0:
        return scores.new_tensor(0.0)

    # Soft-margin: loss = max(0, margin - (mean_ev - mean_dist))
    margin = 0.2
    gap = ev.mean() - dist.mean()
    return F.relu(margin - gap)


def _retrieval_preservation_loss(
    scores: torch.Tensor,   # [T] float [0,1]
    evidence_mask: torch.Tensor,  # [T] bool
    budget: float,
) -> torch.Tensor:
    """Soft top-k loss: evidence tokens should survive budget eviction.

    Uses a differentiable relaxation: computes a soft keep probability for
    each token proportional to its score percentile, then penalises low
    keep probability for evidence tokens.
    """
    T = scores.shape[0]
    num_keep = max(1, int(T * budget))

    # Soft keep probability: sigmoid((score - threshold) / temperature)
    # Threshold ≈ score of the (T - num_keep)-th token (the eviction boundary)
    sorted_scores, _ = torch.sort(scores, descending=True)
    boundary = sorted_scores[num_keep - 1].detach()  # Stop gradient on threshold

    temperature = 0.05
    keep_prob = torch.sigmoid((scores - boundary) / temperature)  # [T]

    ev = keep_prob[evidence_mask.bool()]
    if ev.numel() == 0:
        return scores.new_tensor(0.0)

    # Loss = mean probability of NOT keeping evidence tokens
    return (1.0 - ev).mean()


def _stability_loss(scores: torch.Tensor) -> torch.Tensor:
    """Penalise high variance in scores to prevent degenerate all-0 or all-1 solutions."""
    # Encourage spread: the score distribution should not be too flat (all same value)
    # and not too spiky (extreme outliers).  Simple: penalise very low std.
    std = scores.std()
    # Soft penalty when std < 0.1 (too uniform)
    return F.relu(0.1 - std)


# ── Training step ──────────────────────────────────────────────────────────────

def _predict_scores(
    model: PatchedCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Predict per-token importance scores in [0, 1]  (closed-loop, no heuristic).

    CRITICAL design note
    --------------------
    importance_head.forward() is NOT called here because it expands the single
    cross-attention output uniformly to all T positions, then projects — giving
    the SAME scalar for every token (uniform scores, no gradient signal).

    Fix: combine per-token hidden states with the broadcast query summary so
    that out_proj receives a DIFFERENT input for each token position.

        combined[t] = hidden[t] + attn_out_broadcast[t]   # varies per token
        raw[t]      = out_proj(combined[t])               # different per token
    """
    seq_len = input_ids.shape[1]
    seed_scores = torch.full((seq_len,), 50, dtype=torch.uint8, device=device)

    # Detach hidden states: prevents backprop through the 32-layer base model,
    # which would amplify gradient norms by ~1000x and cause clipping to zero out
    # all updates. Gradients flow directly to importance_head parameters only.
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            importance_scores=seed_scores,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden = out.hidden_states[-1].float().detach()   # [1, T, d] — no grad through base

    # Simplest stable scoring: project each token's hidden state directly.
    # Cross-attention is NOT used here because jointly optimizing attn weights
    # + out_proj causes the two to cancel each other's updates, producing
    # stably uniform scores.  Direct projection is fully stable: out_proj
    # maps the per-token context representation to a discriminative scalar.
    raw = model.importance_head.out_proj(hidden)      # [1, T, 1]
    scores_float = torch.sigmoid(raw.squeeze(-1).squeeze(0))  # [T] in (0, 1)
    return scores_float


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(Path(args.base_checkpoint), device)

    # Freeze base model; only train TIS components
    for param in model._base_model.parameters():
        param.requires_grad = False
    model._base_model.eval()

    # Train importance_head only.
    # importance_embedding is excluded: its gradient path runs through the 32-layer
    # base model backward, producing enormous gradient norms that cause clipping to
    # suppress all updates.  The closed-loop hypothesis is about the head learning
    # discriminative scores from frozen context representations, so the head alone is
    # the correct unit to train here.
    trainable = list(model.importance_head.parameters())
    for p in model.importance_embedding.parameters():
        p.requires_grad = False  # Freeze to prevent destructive gradient amplification
    print(f"[training] Trainable parameters: {sum(p.numel() for p in trainable):,} (importance_head only)", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = RetrievalDataset(
        tokenizer=tokenizer,
        context_tokens=args.context_tokens,
        budgets=args.budgets,
        budget_weights=args.budget_weights,
        evidence_position="random",
        seed=1337,
    )
    data_iter = iter(dataset)

    print(f"\n[training] Closed-loop retrieval training", flush=True)
    print(f"[training] Steps: {args.steps} | Grad accum: {args.grad_accum} | LR: {args.lr:.2e}", flush=True)
    print(f"[training] Loss weights: rank={args.alpha_rank} retrieve={args.beta_retrieve} stab={args.gamma_stability}", flush=True)
    if args.budget_weights:
        dist = {b: w for b, w in zip(args.budgets, args.budget_weights)}
        print(f"[training] Budget curriculum (weighted): {dist}", flush=True)
    print(f"[training] NO heuristic initialization — head learns from detached hidden states", flush=True)

    model.train()
    optimizer.zero_grad()
    log_fh = open(output_dir / "training.jsonl", "w")

    running = {k: 0.0 for k in ("loss", "rank", "retrieve", "stability", "ev_survival")}

    for step in range(1, args.steps + 1):
        ex = next(data_iter)
        input_ids = ex.input_ids.to(device)
        attention_mask = ex.attention_mask.to(device)
        evidence_mask = ex.evidence_mask.to(device)
        budget = ex.budget

        try:
            # ── Forward: predict scores (closed-loop, no heuristic) ────────
            scores_float = _predict_scores(model, input_ids, attention_mask, device)

            # ── Losses ────────────────────────────────────────────────────
            l_rank = _evidence_ranking_loss(scores_float, evidence_mask)
            l_retrieve = _retrieval_preservation_loss(scores_float, evidence_mask, budget)
            l_stability = _stability_loss(scores_float)

            loss = (
                args.alpha_rank * l_rank
                + args.beta_retrieve * l_retrieve
                + args.gamma_stability * l_stability
            )

            loss.backward()

            # Track evidence survival (non-differentiable diagnostic)
            with torch.no_grad():
                seq_len = scores_float.shape[0]
                num_keep = max(1, int(seq_len * budget))
                _, top_idx = torch.topk(scores_float, k=num_keep)
                kept = torch.zeros(seq_len, dtype=torch.bool, device=device)
                kept[top_idx] = True
                ev = evidence_mask.bool()
                ev_survival = ((kept & ev).sum().float() / ev.sum().clamp(min=1)).item()

            running["loss"] += loss.item()
            running["rank"] += l_rank.item()
            running["retrieve"] += l_retrieve.item()
            running["stability"] += l_stability.item()
            running["ev_survival"] += ev_survival

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 10.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % args.log_interval == 0:
                n = args.log_interval
                avg = {k: v / n for k, v in running.items()}
                print(
                    f"  [{step:5d}] loss={avg['loss']:.4f} "
                    f"rank={avg['rank']:.4f} "
                    f"ret={avg['retrieve']:.4f} "
                    f"stab={avg['stability']:.4f} "
                    f"ev_surv={avg['ev_survival']:.3f}",
                    flush=True,
                )
                log_entry = {"step": step, **avg}
                log_fh.write(json.dumps(log_entry) + "\n")
                log_fh.flush()
                running = {k: 0.0 for k in running}

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[warn] OOM at step {step}, skipping", flush=True)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                raise

    log_fh.close()
    print(f"\n[training] Completed {args.steps} steps", flush=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"[checkpoint] Saving to {output_dir}", flush=True)
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    torch.save(tis_state, output_dir / "tis_components.pt")

    metadata = {
        "model": "mistralai/Mistral-7B-v0.3",
        "training_type": "closed_loop_retrieval",
        "steps": args.steps,
        "lr": args.lr,
        "alpha_rank": args.alpha_rank,
        "beta_retrieve": args.beta_retrieve,
        "gamma_stability": args.gamma_stability,
        "heuristic_init": False,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Checkpoint: {output_dir}/tis_components.pt")
    print(f"✓ Metadata:   {output_dir}/metadata.json")
    print(f"✓ Closed-loop retrieval training complete!")


if __name__ == "__main__":
    main()

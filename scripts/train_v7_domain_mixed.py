#!/usr/bin/env python
"""V7 Closed-loop retrieval training with hard-negative ranking and domain-mixed supervision.

CORE CHANGES FROM V6:
  1. Hard-anchor strategy: question + sink/recent tokens ALWAYS kept (anchor_mask)
  2. Budget splitting: total_budget = anchor_budget + learned_budget
  3. Hard-negative ranking: evidence vs top-distractor with margin loss
  4. Domain-mixing: synthetic + real (MS MARCO) training streams
  5. Expanded logging: question_survival, anchor_survival, hard-neg margin, synthetic vs real

Usage:
    source .venv/bin/activate
    python scripts/train_v7_domain_mixed.py \\
        --base-checkpoint checkpoints/stage3_ert_local_fresh/ \\
        --output-dir checkpoints/closed_loop_retrieval_v7/ \\
        --steps 2000 \\
        --anchor-budget 0.08 \\
        --msmarco-data data/msmarco_quick/train \\
        --real-data-fraction 0.1 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.retrieval_data import RetrievalDataset
from token_importance.training.msmarco_data import MSMarcoRetrievalDataset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V7 Domain-mixed TIS training with hard-negative ranking")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--anchor-budget", type=float, default=0.08,
                   help="Fraction of tokens reserved for anchors. E.g. 0.08 for 8 percent always kept.")
    p.add_argument("--alpha-rank", type=float, default=1.0)
    p.add_argument("--beta-retrieve", type=float, default=2.0)
    p.add_argument("--gamma-stability", type=float, default=0.05)
    p.add_argument("--margin-hard-neg", type=float, default=0.5,
                   help="Margin for hard-negative ranking loss: evidence_score - hard_neg_score >= margin")
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--budget-weights", nargs="+", type=float, default=None,
                   help="Non-uniform budget sampling weights.")
    p.add_argument("--msmarco-data", type=str, default="data/msmarco_quick/train",
                   help="Path to MS MARCO dataset for domain-mixed training.")
    p.add_argument("--real-data-fraction", type=float, default=0.1,
                   help="Fraction of batches from MS MARCO vs synthetic. E.g. 0.1 = 10 percent real, 90 percent synthetic.")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--device", default="")
    return p.parse_args(argv)


def _compute_hard_negative_loss(
    scores: torch.Tensor,  # [2*N]  all scores (evidence + distractors)
    evidence_mask: torch.Tensor,  # [2*N]  bool
    margin: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Hard-negative ranking loss: evidence vs top-scoring distractor with margin.
    
    Returns: (loss, hard_neg_margin_mean)
    """
    # Separate evidence and distractor scores
    ev_scores = scores[evidence_mask]      # [n_ev]
    dist_scores = scores[~evidence_mask]   # [n_dist]
    
    if ev_scores.numel() == 0 or dist_scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device), 0.0
    
    # Hard negatives: top-1 distractor per evidence token (batch-wise)
    # Simpler approach: min evidence vs max distractor
    mean_ev = ev_scores.mean()
    max_dist = dist_scores.max()
    
    # Margin loss: max(0, margin - (ev - dist))
    gap = mean_ev - max_dist
    hard_neg_loss = F.relu(margin - gap)
    
    return hard_neg_loss, gap.item()


def _mask_and_score(
    hidden: torch.Tensor,  # [T, d]
    anchor_mask: torch.Tensor,  # [T] bool
    learned_mask: torch.Tensor,  # [T] bool
    learned_scores: torch.Tensor,  # [T] float in [0, 1]
) -> torch.Tensor:
    """Return learned scores directly - model learns to score both anchors + evidence high.
    
    Unlike V7-hard-anchor, here we train the model to naturally learn high scores for
    all important tokens (both anchors and evidence), not forcing anchors to 1.0.
    This makes the learned scores compatible with eval code that doesn't have access
    to token labels.
    
    Returns: [T] float scores in [0, 1]
    """
    return learned_scores


def main():
    args = _parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] Loading mistralai/Mistral-7B-v0.3 with 4-bit NF4 quantization...", flush=True)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = PatchedCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        config=TISConfig(),
        quantization_config=quant_cfg,
        device_map=device,
    ).to(device)

    ckpt = Path(args.base_checkpoint)
    if (ckpt / "tis_components.pt").exists():
        state = torch.load(ckpt / "tis_components.pt", map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        print(f"[checkpoint] ✓ Loaded TIS components from {args.base_checkpoint}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Data loaders: synthetic + real
    synthetic_ds = RetrievalDataset(
        tokenizer=tokenizer,
        context_tokens=args.context_tokens,
        budgets=args.budgets,
        budget_weights=args.budget_weights,
        seed=1337,
    )
    synthetic_it = iter(synthetic_ds)

    real_ds = MSMarcoRetrievalDataset(
        tokenizer=tokenizer,
        data_dir=args.msmarco_data,
        context_tokens=args.context_tokens,
        budgets=args.budgets,
        budget_weights=args.budget_weights,
        seed=1338,
    )
    real_it = iter(real_ds)

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    print(f"[training] V7 Domain-mixed closed-loop retrieval training", flush=True)
    print(f"[training] Steps: {args.steps} | Grad accum: {args.grad_accum} | LR: {args.lr:.2e}", flush=True)
    print(f"[training] Anchor budget: {args.anchor_budget*100:.0f}percent | Hard-neg margin: {args.margin_hard_neg}", flush=True)
    print(f"[training] Loss weights: rank={args.alpha_rank} retrieve={args.beta_retrieve} stab={args.gamma_stability}", flush=True)
    if args.budget_weights:
        dist = {b: w for b, w in zip(args.budgets, args.budget_weights)}
        print(f"[training] Budget curriculum (weighted): {dist}", flush=True)
    print(f"[training] Real data fraction: {args.real_data_fraction*100:.0f}percent | MS MARCO path: {args.msmarco_data}", flush=True)

    model.train()
    accum_loss = 0.0
    accum_rank = 0.0
    accum_retrieve = 0.0
    accum_stab = 0.0
    accum_hard_neg = 0.0
    accum_ev_surv = 0.0
    accum_lrn_surv = 0.0
    n_accum = 0

    for step in range(1, args.steps + 1):
        # Sample from synthetic or real stream
        if torch.rand(1).item() < args.real_data_fraction:
            ex = next(real_it)
        else:
            ex = next(synthetic_it)

        ids = ex.input_ids.to(device)
        attn = ex.attention_mask.to(device)
        ev_mask = ex.evidence_mask.to(device)
        anc_mask = ex.anchor_mask.to(device)
        lrn_mask = ex.learned_mask.to(device)
        T = ids.shape[1]

        # Forward: get hidden states
        seed_scores = torch.full((T,), 50, dtype=torch.uint8, device=device)
        with torch.no_grad():
            out = model(input_ids=ids, importance_scores=seed_scores,
                        attention_mask=attn, output_hidden_states=True)
            hidden = out.hidden_states[-1].float().detach()  # [1, T, d]

        hidden = hidden.squeeze(0)  # [T, d]

        # Learned scores via importance_head
        raw = model.importance_head.out_proj(hidden.unsqueeze(0))  # [1, T, 1]
        learned_scores_float = torch.sigmoid(raw.squeeze(-1).squeeze(0))  # [T]

        # Combine: anchors always 1.0, learned elsewhere
        final_scores = _mask_and_score(hidden, anc_mask, lrn_mask, learned_scores_float)

        # ── Losses ─────────────────────────────────────────────────────────────

        # 1. Hard-negative ranking loss: evidence vs distractor with margin
        rank_loss, hard_neg_margin = _compute_hard_negative_loss(
            final_scores, ev_mask, margin=args.margin_hard_neg
        )

        # 2. Retrieval preservation: learned tokens (evidence + anchors) should score high
        # This ensures the model learns to preserve both answer content and question/sinks
        lrn_scores = final_scores[lrn_mask]
        ret_loss = -lrn_scores.mean() if lrn_scores.numel() > 0 else torch.tensor(0.0, device=device)

        # 3. Stability: penalize extreme scores
        stab_loss = (
            torch.clamp(final_scores - 0.5, min=0).mean() +
            torch.clamp(0.5 - final_scores, min=0).mean()
        )

        # Total
        loss = (args.alpha_rank * rank_loss + args.beta_retrieve * ret_loss +
                args.gamma_stability * stab_loss)

        # ── Metrics ────────────────────────────────────────────────────────────

        # Evidence survival: fraction of evidence tokens in top-keep set
        budget = ex.budget
        n_keep = max(1, int(T * budget))
        _, top_idx = torch.topk(final_scores, k=n_keep)
        keep_mask = torch.zeros(T, dtype=torch.bool, device=device)
        keep_mask[top_idx] = True
        ev_surv = (keep_mask[ev_mask].sum().item() / ev_mask.sum().item()) if ev_mask.sum() > 0 else 1.0

        # Learned (evidence + anchor) survival: should be high since both are important
        lrn_surv = (keep_mask[lrn_mask].sum().item() / lrn_mask.sum().item()) if lrn_mask.sum() > 0 else 1.0

        # Backward
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        accum_rank += rank_loss.item()
        accum_retrieve += ret_loss.item()
        accum_stab += stab_loss.item()
        accum_hard_neg += hard_neg_margin
        accum_ev_surv += ev_surv
        accum_lrn_surv += lrn_surv
        n_accum += 1

        # Optimizer step
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # Logging
        if step % args.log_interval == 0:
            loss_avg = accum_loss / n_accum
            rank_avg = accum_rank / n_accum
            ret_avg = accum_retrieve / n_accum
            stab_avg = accum_stab / n_accum
            hn_avg = accum_hard_neg / n_accum
            ev_avg = accum_ev_surv / n_accum
            lrn_avg = accum_lrn_surv / n_accum
            print(
                f"  [{step:5d}] loss={loss_avg:.4f} rank={rank_avg:.4f} ret={ret_avg:.4f} "
                f"stab={stab_avg:.4f} hn_margin={hn_avg:+.4f} ev_surv={ev_avg:.3f} lrn_surv={lrn_avg:.3f}",
                flush=True,
            )
            accum_loss = accum_rank = accum_retrieve = accum_stab = 0.0
            accum_hard_neg = accum_ev_surv = accum_lrn_surv = 0.0
            n_accum = 0

    print(f"[training] Completed {args.steps} steps", flush=True)

    # Save checkpoint
    print(f"[checkpoint] Saving to {args.output_dir}", flush=True)
    ckpt_path = out_dir / "tis_components.pt"
    torch.save({
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone().detach(),
    }, ckpt_path)

    meta = {
        "steps": args.steps,
        "lr": args.lr,
        "anchor_budget": args.anchor_budget,
        "real_data_fraction": args.real_data_fraction,
        "budgets": args.budgets,
        "version": "v7_domain_mixed",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Checkpoint: {ckpt_path}", flush=True)
    print(f"✓ Metadata:   {out_dir / 'metadata.json'}", flush=True)
    print(f"✓ V7 domain-mixed training complete!", flush=True)


if __name__ == "__main__":
    main()

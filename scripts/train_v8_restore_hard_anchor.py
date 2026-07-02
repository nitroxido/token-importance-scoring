#!/usr/bin/env python
"""V8 Restored hard-anchor preservation training with hard-negative ranking.

CORE CHANGES FROM V7 (Failure Recovery):
  1. Hard-anchor preservation: Restore deterministic anchor forcing (anchor_mask = 1.0)
     This was the mechanism that made V6 work exceptionally well (74% @ 50% budget).
  2. Keep hard-negative ranking: Use it as a secondary signal, not the primary mechanism.
     Ranking loss helps separate evidence from distractors, but doesn't guarantee preservation.
  3. Dual signal approach: Both anchors and evidence get pushed high, providing strong
     preservation at budget thresholds while also satisfying ranking constraints.
  4. Optional domain-mixing: Can add MS MARCO supervision, but only after proving
     the base architecture recovers expected performance.

Architecture Fix:
  V7 removed hard-anchor forcing to achieve "eval compatibility," but this was a mistake.
  The eval incompatibility is a tooling problem (eval doesn't understand token categories),
  not an architecture problem. By restoring hard-anchor forcing and updating the eval script
  to respect anchor_mask, we recover both performance and compatibility.

Usage:
    source .venv/bin/activate
    python scripts/train_v8_restore_hard_anchor.py \\
        --base-checkpoint checkpoints/stage3_ert_local_fresh/ \\
        --output-dir checkpoints/v8_hard_anchor_restored/ \\
        --steps 2000 \\
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
    p = argparse.ArgumentParser(description="V8 Hard-anchor restored TIS training")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha-rank", type=float, default=1.0,
                   help="Weight for hard-negative ranking loss (secondary signal)")
    p.add_argument("--beta-retrieve", type=float, default=2.0,
                   help="Weight for retrieval preservation loss (maximize evidence + anchor scores)")
    p.add_argument("--gamma-stability", type=float, default=0.5,
                   help="Weight for stability loss (keep distribution centered, prevent saturation)")
    p.add_argument("--margin-hard-neg", type=float, default=0.5,
                   help="Margin for hard-negative ranking: evidence - top_distractor >= margin")
    p.add_argument("--anchor-force-value", type=float, default=1.0,
                   help="Score to force for anchor tokens (question + sinks). Use 1.0 for hard forcing.")
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--budget-weights", nargs="+", type=float, default=None,
                   help="Non-uniform budget sampling weights.")
    p.add_argument("--msmarco-data", type=str, default="data/msmarco_quick/train",
                   help="Path to MS MARCO dataset (optional, for future fine-tuning)")
    p.add_argument("--real-data-fraction", type=float, default=0.0,
                   help="Fraction of batches from MS MARCO. Default 0.0 (synthetic only for now)")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--device", default="")
    return p.parse_args(argv)


def _compute_hard_negative_loss(
    scores: torch.Tensor,  # [T]  all scores
    evidence_mask: torch.Tensor,  # [T]  bool
    margin: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Hard-negative ranking loss: evidence should score higher than distractors with margin.
    
    This is a SECONDARY signal. The primary mechanism is hard-anchor forcing.
    Hard-negatives help separate evidence from distractors, but don't guarantee budget preservation.
    
    Returns: (loss, hard_neg_gap)
    """
    ev_scores = scores[evidence_mask]
    dist_scores = scores[~evidence_mask]
    
    if ev_scores.numel() == 0 or dist_scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device), 0.0
    
    mean_ev = ev_scores.mean()
    max_dist = dist_scores.max()
    gap = mean_ev - max_dist
    
    # Margin loss
    hard_neg_loss = F.relu(margin - gap)
    
    return hard_neg_loss, gap.item()


def _mask_and_score(
    hidden: torch.Tensor,  # [T, d]
    anchor_mask: torch.Tensor,  # [T] bool
    learned_mask: torch.Tensor,  # [T] bool
    learned_scores: torch.Tensor,  # [T] float in [0, 1]
    anchor_force_value: float = 1.0,
) -> torch.Tensor:
    """Restore V6-style hard-anchor forcing while learning evidence scores.
    
    CORE MECHANISM (V6-proven):
    - Anchors (question + sinks) are deterministically forced to anchor_force_value
    - Evidence tokens are learned via importance_head, then maximized via loss
    - This dual mechanism ensures preservation at budget thresholds
    
    EVALUATION COMPATIBILITY:
    - This requires eval code to understand anchor_mask
    - Instead of changing the training approach, update the evaluator
    - See eval_niah_hard.py for anchor-aware evaluation
    
    Returns: [T] float scores with anchors at anchor_force_value
    """
    combined = learned_scores.clone()
    
    # Hard-force anchors to high value for guaranteed preservation
    combined[anchor_mask] = anchor_force_value
    
    return combined


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

    # Data loaders
    synthetic_ds = RetrievalDataset(
        tokenizer=tokenizer,
        context_tokens=args.context_tokens,
        budgets=args.budgets,
        budget_weights=args.budget_weights,
        seed=1337,
    )
    synthetic_it = iter(synthetic_ds)

    # Optional: real data (disabled by default, enable with --real-data-fraction > 0)
    real_ds = None
    real_it = None
    if args.real_data_fraction > 0:
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

    print(f"[training] V8 Hard-anchor restored TIS training", flush=True)
    print(f"[training] Steps: {args.steps} | Grad accum: {args.grad_accum} | LR: {args.lr:.2e}", flush=True)
    print(f"[training] Anchor force value: {args.anchor_force_value} | Hard-neg margin: {args.margin_hard_neg}", flush=True)
    print(f"[training] Loss weights: rank={args.alpha_rank} retrieve={args.beta_retrieve} stab={args.gamma_stability}", flush=True)
    if args.budget_weights:
        dist = {b: w for b, w in zip(args.budgets, args.budget_weights)}
        print(f"[training] Budget curriculum (weighted): {dist}", flush=True)
    if args.real_data_fraction > 0:
        print(f"[training] Real data fraction: {args.real_data_fraction*100:.0f}percent | MS MARCO path: {args.msmarco_data}", flush=True)
    else:
        print(f"[training] Using synthetic data only", flush=True)

    model.train()
    accum_loss = 0.0
    accum_rank = 0.0
    accum_retrieve = 0.0
    accum_stab = 0.0
    accum_hard_neg = 0.0
    accum_ev_surv = 0.0
    accum_anc_surv = 0.0
    n_accum = 0

    for step in range(1, args.steps + 1):
        # Sample from real or synthetic stream
        if args.real_data_fraction > 0 and torch.rand(1).item() < args.real_data_fraction:
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
            hidden = out.hidden_states[-1].float().detach()

        hidden = hidden.squeeze(0)  # [T, d]

        # Learned scores via importance_head
        raw = model.importance_head.out_proj(hidden.unsqueeze(0))  # [1, T, 1]
        learned_scores_float = torch.sigmoid(raw.squeeze(-1).squeeze(0))  # [T]

        # CORE MECHANISM: Restore hard-anchor forcing
        final_scores = _mask_and_score(hidden, anc_mask, lrn_mask, learned_scores_float,
                                       anchor_force_value=args.anchor_force_value)

        # ── Losses ─────────────────────────────────────────────────────────────

        # 1. Hard-negative ranking loss (SECONDARY signal)
        # Helps separate evidence from distractors, but doesn't guarantee preservation
        rank_loss, hard_neg_margin = _compute_hard_negative_loss(
            final_scores, ev_mask, margin=args.margin_hard_neg
        )

        # 2. Retrieval preservation (PRIMARY signal)
        # V6-style: maximize BOTH anchors and evidence (learned_mask includes both)
        # Anchors are hard-forced to 1.0, but including them in loss ensures evidence also gets pushed high
        lrn_scores = final_scores[lrn_mask]
        ret_loss = -lrn_scores.mean() if lrn_scores.numel() > 0 else torch.tensor(0.0, device=device)

        # 3. Stability: penalize extreme scores (for learned tokens only, not forced anchors)
        # This is important to prevent saturation where everything scores 1.0
        learned_scores_only = learned_scores_float[~anc_mask]
        if learned_scores_only.numel() > 0:
            stab_loss = (
                torch.clamp(learned_scores_only - 0.5, min=0).mean() +
                torch.clamp(0.5 - learned_scores_only, min=0).mean()
            )
        else:
            stab_loss = torch.tensor(0.0, device=device)

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
        anc_surv = (keep_mask[anc_mask].sum().item() / anc_mask.sum().item()) if anc_mask.sum() > 0 else 1.0

        # Backward
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        accum_rank += rank_loss.item()
        accum_retrieve += ret_loss.item()
        accum_stab += stab_loss.item()
        accum_hard_neg += hard_neg_margin
        accum_ev_surv += ev_surv
        accum_anc_surv += anc_surv
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
            anc_avg = accum_anc_surv / n_accum
            print(
                f"  [{step:5d}] loss={loss_avg:.4f} rank={rank_avg:.4f} ret={ret_avg:.4f} "
                f"stab={stab_avg:.4f} hn_margin={hn_avg:+.4f} ev_surv={ev_avg:.3f} anc_surv={anc_avg:.3f}",
                flush=True,
            )
            accum_loss = accum_rank = accum_retrieve = accum_stab = 0.0
            accum_hard_neg = accum_ev_surv = accum_anc_surv = 0.0
            n_accum = 0

    print(f"[training] Completed {args.steps} steps", flush=True)

    # Save checkpoint
    torch.save(
        {
            "importance_embedding": model.importance_embedding.state_dict(),
            "importance_head": model.importance_head.state_dict(),
        },
        out_dir / "tis_components.pt"
    )
    print(f"[checkpoint] Saving to {args.output_dir}", flush=True)
    print(f"✓ Checkpoint: {out_dir / 'tis_components.pt'}", flush=True)

    # Save metadata
    metadata = {
        "model": "mistralai/Mistral-7B-v0.3",
        "approach": "V8_hard_anchor_restored",
        "anchor_force_value": args.anchor_force_value,
        "loss_weights": {
            "ranking": args.alpha_rank,
            "retrieval": args.beta_retrieve,
            "stability": args.gamma_stability,
        },
        "hyperparameters": {
            "steps": args.steps,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "margin_hard_neg": args.margin_hard_neg,
            "budgets": args.budgets,
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata:   {out_dir / 'metadata.json'}", flush=True)
    print(f"✓ V8 hard-anchor training complete!", flush=True)


if __name__ == "__main__":
    main()

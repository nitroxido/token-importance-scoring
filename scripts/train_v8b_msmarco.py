#!/usr/bin/env python3
"""
V8b MS MARCO Fine-tuning: Light Domain Mixing
============================================

Base: V8b hard-anchor mechanism (78% @ 50% NIAH)
Task: Add 10-15% MS MARCO weak supervision to test domain transfer
Strategy: 85% synthetic NIAH + 15% MS MARCO, 1000 steps fine-tuning
Expected: Maintain synthetic performance while gaining real-data capability
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.retrieval_data import RetrievalDataset
from token_importance.training.msmarco_data import MSMarcoRetrievalDataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from torch.optim import AdamW


def _mask_and_score(hidden, anchor_mask, learned_mask, learned_scores, anchor_force_value=1.0):
    """V8b mechanism: hard-force anchors to 1.0, learn everything else."""
    combined = learned_scores.clone()
    combined[anchor_mask] = anchor_force_value
    return combined


def _compute_hard_negative_loss(scores, evidence_mask, margin=0.5):
    """
    Ranking loss: hard negatives must satisfy margin.
    gap = mean(evidence_scores) - max(distractor_scores)
    loss = max(0, margin - gap)
    """
    ev_scores = scores[evidence_mask]
    if ev_scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device), torch.tensor(0.0)
    
    distractor_mask = ~evidence_mask
    if distractor_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device), torch.tensor(0.0)
    
    dist_scores = scores[distractor_mask]
    ev_mean = ev_scores.mean()
    dist_max = dist_scores.max()
    gap = ev_mean - dist_max
    
    loss = torch.clamp(margin - gap, min=0.0)
    return loss, gap


def main():
    parser = argparse.ArgumentParser(description="V8b MS MARCO fine-tuning: light domain mixing")
    parser.add_argument("--base-checkpoint", default="checkpoints/v8_v6style_loss",
                        help="Base V8b checkpoint to fine-tune from")
    parser.add_argument("--output-dir", default="checkpoints/v8_msmarco_finetune",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--steps", type=int, default=1000, help="Fine-tuning steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--real-data-fraction", type=float, default=0.15,
                        help="Fraction of steps to sample real data (0-1)")
    parser.add_argument("--msmarco-data", default="data/msmarco_quick/train/",
                        help="Path to MS MARCO data")
    parser.add_argument("--alpha-rank", type=float, default=1.0, help="Ranking loss weight")
    parser.add_argument("--beta-retrieve", type=float, default=2.0, help="Retrieval loss weight")
    parser.add_argument("--gamma-stability", type=float, default=0.5, help="Stability loss weight")
    parser.add_argument("--margin-hard-neg", type=float, default=0.5, help="Hard-negative margin")
    parser.add_argument("--anchor-force-value", type=float, default=1.0, help="Force anchor value")
    parser.add_argument("--context-tokens", type=int, default=1536, help="Context length in tokens")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="Cache budgets to sample from")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[train] V8b MS MARCO Fine-tuning")
    print(f"[train] Base checkpoint: {args.base_checkpoint}")
    print(f"[train] Output: {args.output_dir}")
    print(f"[train] Data: 85% synthetic + 15% MS MARCO")
    print(f"[train] Steps: {args.steps} | LR: {args.lr:.2e}", flush=True)
    
    # Load base V8b model  
    print(f"[train] Loading model...", flush=True)
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
    
    # Load V8b checkpoint components
    ckpt_path = Path(args.base_checkpoint)
    if (ckpt_path / "tis_components.pt").exists():
        state = torch.load(ckpt_path / "tis_components.pt", map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        print(f"[train] ✓ Loaded V8b components from {args.base_checkpoint}", flush=True)
    else:
        print(f"[warn] No checkpoint found at {ckpt_path / 'tis_components.pt'}", flush=True)
    
    # Freeze base model
    for name, param in model.named_parameters():
        if 'importance' not in name:
            param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Data loaders
    synthetic_ds = RetrievalDataset(
        tokenizer=tokenizer,
        context_tokens=args.context_tokens,
        budgets=args.budgets,
        seed=args.seed,
    )
    synthetic_it = iter(synthetic_ds)
    
    real_ds = MSMarcoRetrievalDataset(
        tokenizer=tokenizer,
        data_dir=args.msmarco_data,
        context_tokens=args.context_tokens,
        budgets=args.budgets,
        seed=args.seed + 1,
    )
    real_it = iter(real_ds)
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    
    print(f"[training] Real data fraction: {args.real_data_fraction*100:.0f}%", flush=True)
    
    model.train()
    accum_loss = 0.0
    accum_rank = 0.0
    accum_ret = 0.0
    accum_stab = 0.0
    accum_ev_surv = 0.0
    n_accum = 0
    
    for step in range(1, args.steps + 1):
        # Mix synthetic and real data
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
            hidden = out.hidden_states[-1].float().detach()
        
        hidden = hidden.squeeze(0)  # [T, d]
        
        # Learned scores
        raw = model.importance_head.out_proj(hidden.unsqueeze(0))  # [1, T, 1]
        learned_scores_float = torch.sigmoid(raw.squeeze(-1).squeeze(0))  # [T]
        
        # Hard-anchor forcing (V8b mechanism)
        final_scores = _mask_and_score(hidden, anc_mask, lrn_mask, learned_scores_float,
                                       anchor_force_value=args.anchor_force_value)
        
        # Losses
        rank_loss, _ = _compute_hard_negative_loss(final_scores, ev_mask, margin=args.margin_hard_neg)
        
        lrn_scores = final_scores[lrn_mask]
        ret_loss = -lrn_scores.mean() if lrn_scores.numel() > 0 else torch.tensor(0.0, device=device)
        
        learned_scores_only = learned_scores_float[~anc_mask]
        if learned_scores_only.numel() > 0:
            stab_loss = (
                torch.clamp(learned_scores_only - 0.5, min=0).mean() +
                torch.clamp(0.5 - learned_scores_only, min=0).mean()
            )
        else:
            stab_loss = torch.tensor(0.0, device=device)
        
        loss = (args.alpha_rank * rank_loss + args.beta_retrieve * ret_loss +
                args.gamma_stability * stab_loss)
        
        # Metrics
        budget = ex.budget
        n_keep = max(1, int(T * budget))
        _, top_idx = torch.topk(final_scores, k=n_keep)
        keep_mask = torch.zeros(T, dtype=torch.bool, device=device)
        keep_mask[top_idx] = True
        ev_surv = (keep_mask[ev_mask].sum().item() / ev_mask.sum().item()) if ev_mask.sum() > 0 else 1.0
        
        # Backward
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        accum_rank += rank_loss.item()
        accum_ret += ret_loss.item()
        accum_stab += stab_loss.item()
        accum_ev_surv += ev_surv
        n_accum += 1
        
        if step % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Log every 100 steps
        if step % 100 == 0:
            avg_loss = accum_loss / n_accum
            avg_rank = accum_rank / n_accum
            avg_ret = accum_ret / n_accum
            avg_stab = accum_stab / n_accum
            avg_ev_surv = accum_ev_surv / n_accum
            print(f"[{step:4d}] loss={avg_loss:.4f} rank={avg_rank:.4f} "
                  f"ret={avg_ret:.4f} stab={avg_stab:.4f} ev_surv={avg_ev_surv:.3f}",
                  flush=True)
            accum_loss = 0.0
            accum_rank = 0.0
            accum_ret = 0.0
            accum_stab = 0.0
            accum_ev_surv = 0.0
            n_accum = 0
    
    # Save checkpoint
    checkpoint = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(checkpoint, Path(args.output_dir) / "tis_components.pt")
    print(f"[training] Completed {args.steps} steps")
    print(f"✓ Checkpoint: {args.output_dir}/tis_components.pt")
    print(f"✓ V8b MS MARCO fine-tuning complete!")


if __name__ == '__main__':
    main()

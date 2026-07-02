#!/usr/bin/env python
"""V8 Hard-anchor debug: inspect raw scores for anchor/evidence/distractor tokens."""
from __future__ import annotations

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[debug] Loading model and checkpoint...", flush=True)
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

    ckpt = Path("checkpoints/v8_sanity_check_fixed")
    if (ckpt / "tis_components.pt").exists():
        state = torch.load(ckpt / "tis_components.pt", map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        print(f"[debug] ✓ Loaded TIS from {ckpt}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load one example
    print("[debug] Generating test example...", flush=True)
    dataset = RetrievalDataset(tokenizer=tokenizer, context_tokens=2048, budgets=[0.5], seed=42)
    ex = next(iter(dataset))

    ids = ex.input_ids.to(device)
    attn = ex.attention_mask.to(device)
    ev_mask = ex.evidence_mask.to(device)
    anc_mask = ex.anchor_mask.to(device)
    budget = ex.budget
    T = ids.shape[1]

    print(f"\n[example] Sequence length: {T} tokens", flush=True)
    print(f"[example] Budget: {budget:.0%} (keep {int(T * budget)} tokens)", flush=True)
    print(f"[example] Evidence tokens: {ev_mask.sum().item()}", flush=True)
    print(f"[example] Anchor tokens: {anc_mask.sum().item()}", flush=True)
    print(f"[example] Answer: {ex.answer_text}", flush=True)

    # Forward pass
    print("\n[forward] Running inference...", flush=True)
    seed_scores = torch.full((T,), 50, dtype=torch.uint8, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, importance_scores=seed_scores,
                    attention_mask=attn, output_hidden_states=True)
        hidden = out.hidden_states[-1].float().detach().squeeze(0)  # [T, d]

    # Get learned scores
    with torch.no_grad():
        raw = model.importance_head.out_proj(hidden.unsqueeze(0))  # [1, T, 1]
        learned_scores = torch.sigmoid(raw.squeeze(-1).squeeze(0))  # [T]

    # Apply hard-anchor forcing
    final_scores = learned_scores.clone()
    final_scores[anc_mask] = 1.0  # Hard-force anchors to 1.0

    print("\n[scores] Score statistics:")
    print(f"  Learned (before hard-anchor): min={learned_scores.min():.3f}, max={learned_scores.max():.3f}, mean={learned_scores.mean():.3f}")
    print(f"  Final (after hard-anchor):    min={final_scores.min():.3f}, max={final_scores.max():.3f}, mean={final_scores.mean():.3f}")

    # Analyze by token type
    print("\n[analysis] Scores by token type:")
    ev_scores = final_scores[ev_mask]
    anc_scores = final_scores[anc_mask]
    dist_mask = ~(ev_mask | anc_mask)
    dist_scores = final_scores[dist_mask]

    print(f"  Anchors (should be 1.0):     min={anc_scores.min():.3f}, max={anc_scores.max():.3f}, mean={anc_scores.mean():.3f}")
    print(f"  Evidence:                     min={ev_scores.min():.3f}, max={ev_scores.max():.3f}, mean={ev_scores.mean():.3f}")
    print(f"  Distractors:                  min={dist_scores.min():.3f}, max={dist_scores.max():.3f}, mean={dist_scores.mean():.3f}")

    # Check hard-anchor is active
    anc_at_max = (anc_scores == 1.0).sum().item() / len(anc_scores) if len(anc_scores) > 0 else 0.0
    print(f"\n[check] Anchor tokens at 1.0: {anc_at_max*100:.1f}% (should be 100%)")

    # Top-k at budget
    n_keep = int(T * budget)
    _, top_idx = torch.topk(final_scores, k=n_keep)
    keep_mask = torch.zeros(T, dtype=torch.bool, device=device)
    keep_mask[top_idx] = True

    print(f"\n[budget] At {budget:.0%} budget (keep {n_keep} tokens):")
    anc_kept = (keep_mask[anc_mask].sum().item() / len(anc_scores)) if len(anc_scores) > 0 else 1.0
    ev_kept = (keep_mask[ev_mask].sum().item() / len(ev_scores)) if len(ev_scores) > 0 else 0.0
    dist_kept = (keep_mask[dist_mask].sum().item() / len(dist_scores)) if len(dist_scores) > 0 else 0.0

    print(f"  Anchors kept:     {anc_kept*100:5.1f}% ({keep_mask[anc_mask].sum().item()}/{len(anc_scores)} tokens)")
    print(f"  Evidence kept:    {ev_kept*100:5.1f}% ({keep_mask[ev_mask].sum().item()}/{len(ev_scores)} tokens)")
    print(f"  Distractors kept: {dist_kept*100:5.1f}% ({keep_mask[dist_mask].sum().item()}/{len(dist_scores)} tokens)")

    # Top-k threshold
    if n_keep < T:
        threshold_idx = top_idx[-1].item()
        threshold_score = final_scores[threshold_idx].item()
        print(f"\n[threshold] Top-k cutoff at score {threshold_score:.3f}")

        # Show distribution around threshold
        below_threshold = (final_scores < threshold_score).sum().item()
        at_threshold = (final_scores == threshold_score).sum().item()
        above_threshold = (final_scores > threshold_score).sum().item()
        print(f"  Below threshold:  {below_threshold} tokens")
        print(f"  At threshold:     {at_threshold} tokens")
        print(f"  Above threshold:  {above_threshold} tokens")

    # Verify: anchors should dominate kept set
    anc_in_kept = keep_mask[anc_mask].sum().item()
    print(f"\n[verify] Hard-anchor mechanism:")
    if anc_in_kept == len(anc_scores):
        print(f"  ✓ ALL anchors preserved (correct hard-anchor behavior)")
    else:
        print(f"  ✗ ONLY {anc_in_kept}/{len(anc_scores)} anchors preserved (HARD-ANCHOR FAILING!)")

    if len(ev_scores) > 0 and len(dist_scores) > 0:
        mean_ev = ev_scores.mean().item()
        mean_dist = dist_scores.mean().item()
        gap = mean_ev - mean_dist
        print(f"  Evidence mean: {mean_ev:.3f} vs Distractor mean: {mean_dist:.3f} (gap: {gap:+.3f})")
        if gap > 0:
            print(f"  ✓ Evidence > Distractors (correct ranking)")
        else:
            print(f"  ✗ Evidence <= Distractors (ranking issue)")

    print(f"\n[conclusion] Hard-anchor mechanism is {'ACTIVE and working' if anc_in_kept == len(anc_scores) else 'BROKEN - needs investigation'}", flush=True)


if __name__ == "__main__":
    main()

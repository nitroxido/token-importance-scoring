#!/usr/bin/env python
"""P9: Self-Speculative Decoding with TIS.

Uses the SAME Mistral-7B-v0.3 model as both drafter and target:
  - Drafter: run only the FIRST draft_depth=16 transformer layers, then
    apply the same norm+lm_head to get draft token logits.
  - Target: run all 32 layers to verify the K draft tokens in ONE pass.
  - TIS: importance scores from the target guide embedding scaling for the
    drafter's pass (same as Phase 5 TIS embedding bias).

Advantages over a separate LLaMA-1B drafter:
  - Zero vocabulary mismatch (identical tokenizer + vocab)
  - No extra VRAM for a separate model (reuses the same 4-bit weights)
  - TIS scores computed from the same model (perfectly aligned)
  - Draft accepts if the target agrees — same semantic space

Memory: Only ONE model loaded (Mistral-7B-v0.3 4-bit ≈ 5.5 GB).

Expected acceptance: Early-exit at 50% depth (16/32 layers) typically
yields ~50-70% acceptance on structured retrieval tasks. With TIS bias,
targeting >65% acceptance.

Usage:
    source .venv/bin/activate
    python scripts/run_self_spec_decoding_tis.py \\
        --tis-checkpoint checkpoints/closed_loop_retrieval_v6 \\
        --draft-depth 16 \\
        --n-examples 30 --max-depth 8 \\
        --lambda-d 0.2 \\
        --output results/phase9_self_spec.json \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.retrieval_data import RetrievalDataset

MODEL_NAME = "mistralai/Mistral-7B-v0.3"


def _parse_args():
    p = argparse.ArgumentParser(description="P9: Self-Speculative Decoding + TIS")
    p.add_argument("--tis-checkpoint", required=True)
    p.add_argument("--draft-depth",    type=int,   default=16,
                   help="Number of transformer layers for drafting (of 32 total)")
    p.add_argument("--n-examples",     type=int,   default=30)
    p.add_argument("--max-depth",      type=int,   default=8)
    p.add_argument("--lambda-d",       type=float, default=0.2)
    p.add_argument("--context-tokens", type=int,   default=384)
    p.add_argument("--output",         default="results/phase9_self_spec.json")
    p.add_argument("--device",         default="")
    return p.parse_args()


def _load_model(ckpt_path: str, device: torch.device) -> PatchedCausalLM:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = PatchedCausalLM.from_pretrained(
        MODEL_NAME, config=TISConfig(), quantization_config=quant_cfg,
        device_map=device,
    ).to(device)
    tis_pt = f"{ckpt_path}/tis_components.pt"
    if os.path.exists(tis_pt):
        state = torch.load(tis_pt, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        lam = state.get("attn_hook_lambda")
        if lam is not None:
            v = lam.to(device) if isinstance(lam, torch.Tensor) else torch.tensor(float(lam), device=device)
            model.attn_hook._lambda.data = v
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def _get_hidden_and_imp(model, ids, device):
    """Full forward pass: return last hidden states + TIS scores."""
    T = ids.shape[1]
    seed = torch.full((T,), 50, dtype=torch.uint8, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, importance_scores=seed,
                    attention_mask=torch.ones_like(ids), output_hidden_states=True)
    hidden = out.hidden_states[-1].float()
    imp = model.importance_head.direct_score(hidden.squeeze(0))
    return hidden, imp


def _draft_hidden_at_depth(model, ids, imp_scores, lambda_d, draft_depth, device, k):
    """Capture hidden states at layer `draft_depth` using a forward hook.

    This avoids manual layer calls and all the RoPE/cache API complexity —
    we let Mistral's own forward machinery handle everything, and simply
    capture the output of layer [draft_depth-1].
    """
    base = model._base_model
    T = ids.shape[1]
    captured = {}

    def _hook(module, input, output):
        # output is a tuple; first element is the hidden state
        captured["h"] = output[0].detach().clone()

    handle = base.model.layers[draft_depth - 1].register_forward_hook(_hook)

    try:
        # Use the PatchedCausalLM forward with TIS embedding scaling via importance_scores
        # We set scores to reflect TIS importance if lambda_d > 0
        if lambda_d > 0.0 and imp_scores is not None:
            lam = lambda_d * (1.0 + 0.1 * k)
            imp = imp_scores[:T]
            norm_imp = (imp / 100.0 - 0.5) * 2.0
            # Build an importance score that scales embeddings:
            # Convert the continuous scale to a score understood by ImportanceEmbedding
            # Use 50 + offset to approximate the scaling effect
            scale_offset = (norm_imp * lam * 50).clamp(-49, 49)
            scaled_scores = (50 + scale_offset).clamp(1, 99).to(torch.uint8)
        else:
            scaled_scores = torch.full((T,), 50, dtype=torch.uint8, device=device)

        with torch.no_grad():
            model(
                input_ids=ids,
                importance_scores=scaled_scores,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=False,
            )
    except Exception:
        # Fallback: plain base model forward
        with torch.no_grad():
            base(input_ids=ids, output_hidden_states=False)
    finally:
        handle.remove()

    return captured.get("h", None)  # [1, T, d] or None on failure


def _draft_logits_from_partial_hidden(model, partial_hidden):
    """Apply final norm + lm_head to partial hidden states to get draft logits."""
    base = model._base_model
    # Ensure 3D [1, T, d]
    h = partial_hidden
    if h.dim() == 2:
        h = h.unsqueeze(0)   # [T, d] → [1, T, d]
    with torch.no_grad():
        normed = base.model.norm(h.to(torch.bfloat16)).float()
        logits = base.lm_head(normed.to(torch.bfloat16)).float()  # [1, T, V]
    return logits


def _greedy_next(logits):
    """Return greedy next-token prediction from logits [1, T, V] or [T, V]."""
    if logits.dim() == 2:
        return logits[-1, :].argmax().item()   # [T, V]
    return logits[0, -1, :].argmax().item()    # [1, T, V]


def self_spec_round(model, ids, imp, lambda_d, draft_depth, max_depth, device):
    """One round of self-speculative decoding.

    Returns (accepted, embedding_energy_fracs).
    """
    draft_tokens = []
    energy_fracs = []
    cur_ids = ids.clone()

    for k in range(max_depth):
        # Drafter: shallow forward
        partial_h = _draft_hidden_at_depth(
            model, cur_ids, imp, lambda_d, draft_depth, device, k
        )
        draft_logits = _draft_logits_from_partial_hidden(model, partial_h)
        tok = _greedy_next(draft_logits)
        draft_tokens.append(tok)

        # Embedding energy measurement
        with torch.no_grad():
            emb = model._base_model.model.embed_tokens(cur_ids)
        if imp is not None:
            T = emb.shape[1]
            imp_mask = (imp[:T] >= 70.0)
            energy = emb.float().squeeze(0).norm(dim=-1)
            frac = (energy[imp_mask].sum() / energy.sum()).item() if imp_mask.sum() > 0 else 0.0
        else:
            frac = 0.0
        energy_fracs.append(frac)

        cur_ids = torch.cat([cur_ids, torch.tensor([[tok]], device=device)], dim=1)
        if imp is not None:
            imp = torch.cat([imp, torch.tensor([50.0], device=device)])

    # Target: full forward to verify
    _, target_imp = _get_hidden_and_imp(model, cur_ids[:, :-1], device)
    target_logits = []
    with torch.no_grad():
        base = model._base_model
        seed = torch.full((cur_ids.shape[1]-1,), 50, dtype=torch.uint8, device=device)
        out = model(input_ids=cur_ids[:, :-1], importance_scores=seed,
                    attention_mask=torch.ones(1, cur_ids.shape[1]-1, device=device))
    full_logits = out.logits[0, :, :]  # [T, V]

    accepted = 0
    orig_T = ids.shape[1]
    for k in range(max_depth):
        target_pred = full_logits[orig_T - 1 + k].argmax().item()
        if target_pred == draft_tokens[k]:
            accepted += 1
        else:
            break

    return accepted, energy_fracs


def evaluate(model, tokenizer, args, device, lambda_d):
    dataset = RetrievalDataset(tokenizer=tokenizer, context_tokens=args.context_tokens,
                                budgets=[0.5], seed=321)
    it = iter(dataset)

    accepts = []
    energy_by_depth = {k: [] for k in range(args.max_depth)}
    total_time = 0.0

    for _ in range(args.n_examples):
        batch = next(it)
        ids = batch.input_ids.to(device)
        if ids.shape[1] < 8: continue

        hidden, imp = _get_hidden_and_imp(model, ids, device)
        t0 = time.time()
        acc, energy_fracs = self_spec_round(
            model, ids, imp if lambda_d > 0 else None,
            lambda_d, args.draft_depth, args.max_depth, device,
        )
        total_time += time.time() - t0
        accepts.append(acc)
        for k, ef in enumerate(energy_fracs[:args.max_depth]):
            energy_by_depth[k].append(ef)

    mean_accept = sum(accepts) / len(accepts) if accepts else 0
    mean_energy = {k: sum(v)/len(v) if v else 0. for k, v in energy_by_depth.items()}
    e0 = mean_energy.get(0, 1.) or 1.
    drift_ratio = {k: mean_energy[k] / e0 for k in mean_energy}
    speedup = mean_accept / (args.max_depth + 1)

    return {
        "lambda_d": lambda_d,
        "draft_depth": args.draft_depth,
        "n_examples": len(accepts),
        "mean_acceptance_length": mean_accept,
        "speedup_proxy": speedup,
        "drift_ratio": drift_ratio,
        "mean_time_per_round_s": total_time / max(1, len(accepts)),
    }


def main():
    args = _parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"[P9] Self-speculative decoding — Mistral-7B draft_depth={args.draft_depth}/32")
    model = _load_model(args.tis_checkpoint, device)
    print(f"[P9] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    results = {}
    for cond, lam in [("no_tis", 0.0), ("tis_bias", args.lambda_d)]:
        print(f"\n[P9] Evaluating: {cond} (λ={lam})", flush=True)
        res = evaluate(model, tokenizer, args, device, lambda_d=lam)
        results[cond] = res
        print(f"  Accept: {res['mean_acceptance_length']:.2f}/{args.max_depth}  "
              f"Speedup: {res['speedup_proxy']:.3f}  "
              f"DriftRatio@k7: {res['drift_ratio'].get(7,0.):.3f}", flush=True)

    print("\n" + "="*65)
    print("P9: SELF-SPECULATIVE DECODING RESULTS")
    print("="*65)
    print(f"Model:   {MODEL_NAME} (4-bit NF4)")
    print(f"Drafter: first {args.draft_depth}/32 layers (self-speculative)")
    print(f"Target:  all 32 layers (same model)\n")
    for name, r in results.items():
        print(f"{name:<12}: Accept={r['mean_acceptance_length']:.2f}  "
              f"Speedup={r['speedup_proxy']:.3f}  "
              f"Drift@k7={r['drift_ratio'].get(7,0.):.3f}")

    if "tis_bias" in results and "no_tis" in results:
        da = results["tis_bias"]["mean_acceptance_length"] - results["no_tis"]["mean_acceptance_length"]
        print(f"\nΔ acceptance: {da:+.2f} tokens  ({da/results['no_tis']['mean_acceptance_length']*100:+.1f}%)")

    out = args.output
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[P9] Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()

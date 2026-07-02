#!/usr/bin/env python
"""TIS training entrypoint.

Stage 1  — Freeze base model, train only TIS components (RECOMMENDED).
Stage 2  — LoRA + LM objective (⚠️  RELIC — kept for research reference only).
           This stage failed in practice: lm_loss converged to 2.1e-06 via
           catastrophic memorisation; LoRA adapters produce only colons during
           inference.  See STAGE2-FINDINGS.md for the full post-mortem.
           Do not use Stage 2 for production training.
Stage 3  — Eviction Robustness Training (ERT).  Replaces Stage 2's LM objective
           with KL(full || evicted), directly optimising eviction quality.
           This is the recommended path after Stage 1 converges.

Usage examples
--------------
# Smoke test (CPU / small GPU, finishes in < 10 minutes)
python scripts/train.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --dataset narrativeqa \\
    --stage 1 \\
    --epochs 1 \\
    --max_samples 50 \\
    --batch_size 1 \\
    --grad_accum 1 \\
    --output_dir /tmp/tis_smoke

# Full Stage 1 (A100 on GPU-Action)
python scripts/train.py \\
    --model mistralai/Mistral-7B-v0.3 \\
    --dataset narrativeqa \\
    --stage 1 --epochs 2 --batch_size 4 --grad_accum 8 \\
    --lr 1e-4 --bf16 --output_dir checkpoints/stage1

# Stage 3 ERT (A100 on GPU-Action, loading Stage-1 checkpoint)
python scripts/train.py \\
    --model checkpoints/stage1 \\
    --stage 3 --epochs 2 --batch_size 4 --grad_accum 8 \\
    --bf16 --output_dir checkpoints/stage3_ert

# Stage 2 RELIC (preserved for research — NOT recommended)
# python scripts/train.py \\
#     --model checkpoints/stage1 \\
#     --stage 2 --lora_r 16 --lora_alpha 32 \\
#     --epochs 1 --batch_size 4 --grad_accum 8 \\
#     --bf16 --output_dir checkpoints/stage2_relic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig

# Resolve project root so the script works when called from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.data import (
    TISTrainingDataset,
    collate_skip_none,
    extract_fields,
    load_training_dataset,
)
from token_importance.training.objectives import TISLoss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TIS training script (Stage 1 and Stage 2)"
    )
    p.add_argument("--model",    required=True,
                   help="HF model name or local checkpoint directory (Stage 2).")
    p.add_argument("--dataset",  default="narrativeqa",
                   choices=["narrativeqa", "quality", "qasper"],
                   help="Training dataset name.")
    p.add_argument("--stage",    type=int, default=1, choices=[1, 2, 3],
                   help="Training stage: 1=frozen base (recommended), "
                        "2=LoRA+LM RELIC (failed, kept for research), "
                        "3=ERT Eviction Robustness Training (Stage 2 replacement).")
    p.add_argument("--epochs",   type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps.")
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lora_r",   type=int, default=16,
                   help="LoRA rank (Stage 2 only).")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (Stage 2 only).")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (for smoke tests).")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a TIS component checkpoint to resume from.")
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 mixed precision (requires CUDA).")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Use bitsandbytes NF4 4-bit quantization for local Stage 2 retries.")
    p.add_argument("--device", default=None,
                   help="Target device (default: cuda if available, else cpu).")
    p.add_argument("--weight_alignment",  type=float, default=0.1)
    p.add_argument("--weight_robustness", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=2048,
                   help="Maximum sequence length for training samples.")
    p.add_argument("--lambda_init_stage2", type=float, default=0.1,
                   help="Lambda warm-start for Stage 2 (overrides Stage 1 checkpoint). "
                        "Set to activate importance bias: 0.1 for standard run, 0.0 to preserve Stage 1 value.")
    p.add_argument(
        "--ert_budgets",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        metavar="BUDGET",
        help="Cache budget fractions sampled per step during ERT (Stage 3).",
    )
    p.add_argument(
        "--gen_check_interval",
        type=int,
        default=500,
        help="Steps between generation quality checks (Stage 2 relic and Stage 3). "
             "Set to 0 to disable. Abort if degenerate output is detected.",
    )
    p.add_argument(
        "--gen_check_prompts",
        type=int,
        default=5,
        help="Number of spot-check generations per quality check.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model(args: argparse.Namespace, device: torch.device) -> PatchedCausalLM:
    """Load PatchedCausalLM for the requested stage."""
    tis_config = TISConfig()

    # Stage 2: args.model points to a Stage-1 checkpoint directory
    stage1_ckpt_dir: str | None = None
    base_model_name: str = args.model

    if os.path.isdir(args.model):
        train_args_path = os.path.join(args.model, "train_args.json")
        if os.path.exists(train_args_path):
            with open(train_args_path) as fh:
                prev_args = json.load(fh)
            base_model_name = prev_args.get("model", args.model)
            stage1_ckpt_dir = args.model
            print(f"[Stage 2] Loading base model: {base_model_name}", flush=True)
        else:
            # Might be a HF-style local model directory — use directly
            base_model_name = args.model

    hf_kwargs: dict = {"device_map": "auto"}

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("--load_in_4bit requires bitsandbytes + transformers support") from exc

        quant_dtype = torch.bfloat16 if args.bf16 else torch.float16
        hf_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=quant_dtype,
        )
        hf_kwargs["attn_implementation"] = "eager"

    hf_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    model = PatchedCausalLM.from_pretrained(base_model_name, tis_config, **hf_kwargs)

    # Restore TIS components from Stage-1 checkpoint
    if stage1_ckpt_dir is not None:
        _load_tis_components(model, stage1_ckpt_dir, args)

    if args.resume_from_checkpoint is not None:
        _load_tis_components(model, args.resume_from_checkpoint, args)

    return model


def _load_tis_components(model: PatchedCausalLM, ckpt_dir: str, args: argparse.Namespace | None = None) -> None:
    ckpt_path = os.path.join(ckpt_dir, "tis_components.pt")
    if not os.path.exists(ckpt_path):
        warnings.warn(f"tis_components.pt not found in {ckpt_dir}; skipping restore.")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.importance_embedding.load_state_dict(ckpt["importance_embedding"])
    model.importance_head.load_state_dict(ckpt["importance_head"])
    lambda_val = ckpt.get("attn_hook_lambda")
    
    # Stage 2 warm-start: override lambda with stage2 value if specified
    if args is not None and args.stage == 2 and args.lambda_init_stage2 != 0.0:
        print(f"[Stage 2 warm-start] Overriding lambda: {lambda_val} → {args.lambda_init_stage2}", flush=True)
        lambda_val = args.lambda_init_stage2
    
    if lambda_val is not None:
        with torch.no_grad():
            model.attn_hook._lambda.fill_(float(lambda_val))
    print(f"[checkpoint] Restored TIS components from {ckpt_path} (lambda={float(lambda_val) if lambda_val else 0.0:.4f})", flush=True)


# ---------------------------------------------------------------------------
# Stage setup
# ---------------------------------------------------------------------------

def _configure_stage1(model: PatchedCausalLM) -> None:
    """Freeze base model; only TIS components are trainable."""
    for param in model.base.parameters():
        param.requires_grad = False
    # Ensure TIS components are trainable
    for component in [model.importance_embedding, model.importance_head]:
        for param in component.parameters():
            param.requires_grad = True
    model.attn_hook._lambda.requires_grad = True

    n_frozen    = sum(p.numel() for p in model.base.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 1] Frozen: {n_frozen/1e6:.1f}M params | "
          f"Trainable: {n_trainable/1e3:.1f}K params", flush=True)


def _configure_stage2(model: PatchedCausalLM, args: argparse.Namespace) -> None:
    """Apply LoRA to q_proj/v_proj; all components are trainable.

    ⚠️  RELIC — THIS STAGE FAILED IN PRACTICE ⚠️
    Stage 2 (LoRA + LM loss) was attempted on Mistral-7B-v0.3 and resulted in
    catastrophic memorisation: lm_loss converged to 2.1e-06 (5×10⁶ below
    baseline), LoRA adapters produce only repeated colons during inference, and
    LITM accuracy degraded by 1.3 pp relative to Stage 1.

    Root cause: LM gradient magnitude (~10×) dominated alignment gradient,
    forcing LoRA to memorise the training set rather than learn importance scoring.

    This function is preserved as a research relic so that:
    - Practitioners can reproduce the failure (use --stage 2).
    - Future contributors can attempt objective-weight tuning or alternative LoRA
      configurations that might avoid the failure mode.

    For production training, use Stage 3 (ERT) instead.
    See: STAGE2-FINDINGS.md, STAGE2-VERDICT.md
    """
    print(
        "\n" + "=" * 70 + "\n"
        "  ⚠️  STAGE 2 RELIC — KNOWN FAILURE MODE  ⚠️\n"
        "  This stage produced catastrophic LoRA overfitting in the reference\n"
        "  run. lm_loss converged to 2.1e-06 (memorisation); inference outputs\n"
        "  only repeated colons. Proceeding anyway for research purposes.\n"
        "  See STAGE2-FINDINGS.md for the complete post-mortem.\n"
        "  For production training, use --stage 3 (ERT) instead.\n"
        + "=" * 70 + "\n",
        flush=True,
    )
    from peft import LoraConfig, get_peft_model  # lazy import
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model.base = get_peft_model(model.base, lora_config)
    # Make sure TIS components are still trainable
    for component in [model.importance_embedding, model.importance_head]:
        for param in component.parameters():
            param.requires_grad = True
    model.attn_hook._lambda.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 2] LoRA applied | Trainable: {n_trainable/1e6:.2f}M params",
          flush=True)


def _configure_stage3_ert(model: PatchedCausalLM) -> None:
    """Configure Stage 3 (ERT): same trainable parameters as Stage 1.

    Base model is fully frozen; only TIS components are trained.
    The key difference from Stage 1 is the loss function: ERT uses
    KL(full || evicted) instead of LM cross-entropy.

    Lambda remains at its Stage 1 value (0.0) unless overridden.  ERT will
    naturally learn to push lambda toward useful values if importance-biased
    attention reduces eviction KL divergence.
    """
    for param in model.base.parameters():
        param.requires_grad = False
    for component in [model.importance_embedding, model.importance_head]:
        for param in component.parameters():
            param.requires_grad = True
    model.attn_hook._lambda.requires_grad = True

    n_frozen    = sum(p.numel() for p in model.base.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Stage 3 ERT] Frozen: {n_frozen/1e6:.1f}M params | "
        f"Trainable: {n_trainable/1e3:.1f}K params",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Generation quality check  (mandatory for Stage 2 relic and Stage 3 ERT)
# ---------------------------------------------------------------------------

_QUALITY_CHECK_PROMPTS = [
    "The capital of France is",
    "Water freezes at",
    "The speed of light is approximately",
    "In machine learning, a transformer is",
    "The main purpose of attention in neural networks is",
]

def _generation_quality_check(
    model,
    tokenizer,
    device,
    n_prompts: int = 5,
    max_new_tokens: int = 20,
    degenerate_threshold: float = 0.8,
) -> bool:
    """Run short generations; return True if healthy, False if degenerate.

    A generation is considered degenerate if more than ``degenerate_threshold``
    fraction of the new tokens are the same token (repeated-character collapse),
    or if fewer than 3 new tokens are produced.

    This check MUST be run every gen_check_interval steps during Stage 2/3 to
    catch memorisation or collapse early (Stage 2 failure was detectable at
    step ~500).
    """
    import random
    was_training = model.training
    model.eval()
    degraded = 0

    prompts = random.sample(_QUALITY_CHECK_PROMPTS, min(n_prompts, len(_QUALITY_CHECK_PROMPTS)))
    for prompt in prompts:
        try:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out[0, enc["input_ids"].shape[1]:]
            if len(new_ids) < 3:
                degraded += 1
                continue
            # Check fraction of most-common token
            most_common_count = max(
                (new_ids == t).sum().item() for t in new_ids.unique()
            )
            if most_common_count / len(new_ids) > degenerate_threshold:
                degraded += 1
        except Exception:
            degraded += 1

    if was_training:
        model.train()

    healthy = degraded < len(prompts) // 2 + 1
    status = "✓ healthy" if healthy else f"✗ DEGENERATE ({degraded}/{len(prompts)} prompts)"
    print(f"[gen-check] {status}", flush=True)
    return healthy


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _training_step(
    model: PatchedCausalLM,
    batch: dict,
    loss_fn: TISLoss,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Single forward + loss computation for one batch."""
    input_ids        = batch["input_ids"].to(device)         # [B, T]
    labels           = batch["labels"].to(device)            # [B, T]
    attention_mask   = batch["attention_mask"].to(device)    # [B, T]
    importance_scores = batch["importance_scores"].to(device) # [B, T] uint8

    B, T = input_ids.shape
    scores_norm = importance_scores.float() / 100.0           # [B, T]

    # Build inputs_embeds: token embed + importance delta.
    # The base model's embed layer may be on a different device (e.g. after
    # device_map="auto"); resolve its actual device before calling.
    embed_layer  = model.base.get_input_embeddings()
    embed_device = next(embed_layer.parameters()).device
    token_embeds = embed_layer(input_ids.to(embed_device))         # [B, T, d]
    imp_delta    = model.importance_embedding(importance_scores.to(device).long())  # [B, T, d]
    imp_delta    = imp_delta.to(device=embed_device, dtype=token_embeds.dtype)
    inputs_embeds = (token_embeds + imp_delta).to(device)          # [B, T, d]

    # Forward through base model.
    # We skip output_attentions=True because:
    #   • SDPA / Flash-Attention models don't support it without falling back to
    #     eager, which materialises O(T²) attention matrices — infeasible at
    #     T=2048 on consumer GPUs (≥ 7 GB per forward pass for a 7B model).
    #   • On A100 (40 GB) you can re-enable it by adding output_attentions=True here.
    # The alignment loss still trains correctly with attn_mag = 0; the head learns
    # a self-consistency signal between predicted deltas and importance scores.
    outputs = model.base(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
    )
    attn_mag = torch.zeros(B, T, device=device)

    logits      = outputs.logits                              # [B, T, vocab]
    last_hidden = outputs.hidden_states[-1]                   # [B, T, d]  (fp16/bf16)

    # Run ImportanceUpdateHead (float32) — cast hidden states to float32 first.
    current_h        = last_hidden[:, -1:, :].float()         # [B, 1, d]
    predicted_deltas = model.importance_head(                 # [B, T, 1]
        current_h, last_hidden.float()
    )

    # Keep logits in their native dtype (float16/bf16) — F.cross_entropy handles it and
    # avoids a large float32 upcast (e.g. 1×2048×151936×4 bytes = 1.2 GB for Qwen vocab).
    # Cast alignment / robustness inputs to float32 for numerical stability.
    loss_inputs = {
        "logits":               logits,
        "labels":               labels,
        "predicted_deltas":     predicted_deltas.float(),
        "attention_magnitudes": attn_mag.float(),
        "importance_scores_norm": scores_norm.float(),
    }
    return loss_fn(loss_inputs)


# ---------------------------------------------------------------------------
# ERT training step  (Stage 3)
# ---------------------------------------------------------------------------

def _ert_training_step(
    model: PatchedCausalLM,
    batch: dict,
    loss_fn,   # ERTLoss instance
    device: torch.device,
    ert_budgets: list[float],
) -> dict[str, torch.Tensor]:
    """ERT double-forward training step.

    Algorithm:
    1. Build inputs_embeds (full sequence).
    2. Full forward pass → logits_full, hidden_states.
    3. ImportanceUpdateHead → predicted_deltas.
    4. Use predicted scores to build an attention-mask that simulates eviction
       (soft eviction via binary mask — zero out low-importance positions).
    5. Evicted forward pass → logits_evicted (same shape as logits_full).
    6. ERT loss = KL(logits_full || logits_evicted) + w_align * L_align.

    Eviction implementation note: We use attention-masking (binary mask set to
    zero for evicted positions) rather than sequence truncation.  Both passes
    have shape [B, T, vocab], so logits are directly comparable.  Gradients
    flow through logits_evicted w.r.t. the model's base parameters; the
    importance head receives gradients from L_align and indirectly from the
    mask selection which changes logits_evicted across training steps.

    Future improvement: replace binary mask with Gumbel-softmax relaxation
    (temperature annealed 1.0→0.1 over first 2000 steps) for differentiable
    end-to-end importance-to-eviction gradient flow.
    """
    import random

    input_ids         = batch["input_ids"].to(device)
    labels            = batch["labels"].to(device)
    attention_mask    = batch["attention_mask"].to(device)
    importance_scores = batch["importance_scores"].to(device)

    B, T = input_ids.shape
    scores_norm = importance_scores.float() / 100.0

    # Sample random budget for curriculum training
    budget = random.choice(ert_budgets)
    n_keep = max(64, int(T * budget))  # always keep at least 64 tokens

    # --- Build embeddings ---
    embed_layer  = model.base.get_input_embeddings()
    embed_device = next(embed_layer.parameters()).device
    token_embeds  = embed_layer(input_ids.to(embed_device))
    imp_delta     = model.importance_embedding(importance_scores.long())
    imp_delta     = imp_delta.to(device=embed_device, dtype=token_embeds.dtype)
    inputs_embeds = (token_embeds + imp_delta).to(device)

    # --- Full forward pass (no gradient — acts as reference distribution) ---
    with torch.no_grad():
        full_out = model.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    logits_full = full_out.logits.detach()          # [B, T, vocab]
    last_hidden = full_out.hidden_states[-1]        # [B, T, d]

    # --- Predict importance scores from hidden states ---
    current_h        = last_hidden[:, -1:, :].float()
    predicted_deltas = model.importance_head(current_h, last_hidden.float())  # [B, T, 1]

    # Compute per-token eviction priority: higher score = keep
    # Start from oracle pseudo-labels, add head-predicted delta
    eviction_priority = scores_norm.clone()  # [B, T], float32
    delta_contrib = torch.tanh(predicted_deltas.squeeze(-1)) * 0.2  # max ±0.2
    eviction_priority = (eviction_priority + delta_contrib).clamp(0.0, 1.0)

    # Protect sinks and recent tokens
    cfg = model.tis_config
    eviction_priority[:, :cfg.N_sink].fill_(2.0)
    eviction_priority[:, max(0, T - cfg.N_recent):].fill_(2.0)

    # Hard top-k selection (per batch item, using batch item 0's scores for mask)
    # For single-item batches (standard training), this is exact.
    # For multi-item batches, each item's importance may differ; we take the
    # mean across the batch for the shared mask.
    mean_priority = eviction_priority.mean(dim=0)   # [T]
    _, keep_idx   = mean_priority.topk(min(n_keep, T))
    keep_idx      = keep_idx.sort().values           # [n_keep], sorted

    # Build eviction attention mask: 1 for kept positions, 0 for evicted
    eviction_attn_mask = torch.zeros(B, T, device=device, dtype=attention_mask.dtype)
    eviction_attn_mask[:, keep_idx] = attention_mask[:, keep_idx]

    # --- Evicted forward pass (with gradient for importance head) ---
    evicted_out = model.base(
        inputs_embeds=inputs_embeds,
        attention_mask=eviction_attn_mask,
        output_hidden_states=False,
    )
    logits_evicted = evicted_out.logits   # [B, T, vocab]

    attn_mag = torch.zeros(B, T, device=device)
    loss_inputs = {
        "logits_full":           logits_full,
        "logits_evicted":        logits_evicted,
        "labels":                labels,
        "predicted_deltas":      predicted_deltas.float(),
        "attention_magnitudes":  attn_mag.float(),
        "importance_scores_norm": scores_norm.float(),
    }
    result = loss_fn(loss_inputs)
    result["budget"] = torch.tensor(budget)
    return result


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: PatchedCausalLM,
    output_dir: str,
    args: argparse.Namespace,
    stage: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save TIS-specific components (always)
    torch.save(
        {
            "importance_embedding": model.importance_embedding.state_dict(),
            "importance_head":      model.importance_head.state_dict(),
            "attn_hook_lambda":     model.attn_hook._lambda.item(),
        },
        os.path.join(output_dir, "tis_components.pt"),
    )

    # Save training args for Stage-2 restoring
    with open(os.path.join(output_dir, "train_args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    # Stage 2: save LoRA adapter alongside (relic — see docstring)
    if stage == 2 and hasattr(model.base, "save_pretrained"):
        model.base.save_pretrained(output_dir)
        print(f"[checkpoint] LoRA adapter saved to {output_dir} (RELIC)", flush=True)

    print(f"[checkpoint] Saved to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    # --- Load model ---
    model = _load_model(args, device)
    # TIS components (3.2M params) stay in float32 for numerically stable gradient updates.
    # Only move to the correct device; dtype stays float32.
    # In _training_step, imp_delta is explicitly cast to the base model's dtype before addition.
    base_dtype = next(iter(model.base.parameters())).dtype
    model.importance_embedding.to(device=device)
    model.importance_head.to(device=device)
    model.train()

    if args.stage == 1:
        _configure_stage1(model)
    elif args.stage == 2:
        _configure_stage2(model, args)
    else:  # stage == 3: ERT
        _configure_stage3_ert(model)

    # --- Load tokenizer ---
    # For Stage 2, base model name is stored in Stage-1 train_args.json
    tokenizer_name = args.model
    if os.path.isdir(args.model):
        ta_path = os.path.join(args.model, "train_args.json")
        if os.path.exists(ta_path):
            with open(ta_path) as fh:
                tokenizer_name = json.load(fh).get("model", args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Build dataset ---
    print(f"Loading dataset: {args.dataset}", flush=True)
    hf_ds = load_training_dataset(args.dataset, max_samples=args.max_samples)
    dataset = TISTrainingDataset(
        hf_ds, tokenizer, max_length=args.max_length, dataset_name=args.dataset
    )
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_skip_none,
        num_workers=0,
    )
    print(f"Dataset size: {len(dataset)} samples", flush=True)

    # --- Optimiser & loss ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable_params, lr=args.lr)

    if args.stage == 3:
        from token_importance.training.objectives import ERTLoss
        loss_fn: object = ERTLoss(weight_alignment=args.weight_alignment)
        print("[Stage 3 ERT] Using ERTLoss: KL(full||evicted) + alignment", flush=True)
    else:
        loss_fn = TISLoss(
            weight_alignment=args.weight_alignment,
            weight_robustness=args.weight_robustness,
        )

    # Use autocast only when --bf16 is requested (base model ops in bfloat16).
    # TIS params are float32 regardless, so no GradScaler is ever needed.
    use_amp  = args.bf16 and device.type == "cuda"
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    scaler   = None

    # --- Training ---
    global_step = 0
    gen_check_failed = False
    with open(log_path, "w") as log_fh:
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===", flush=True)
            optimiser.zero_grad()

            for local_step, batch in enumerate(loader):
                if batch is None:
                    continue

                try:
                    if use_amp:
                        with autocast:
                            if args.stage == 3:
                                loss_dict = _ert_training_step(
                                    model, batch, loss_fn, device, args.ert_budgets
                                )
                            else:
                                loss_dict = _training_step(model, batch, loss_fn, device)
                    else:
                        if args.stage == 3:
                            loss_dict = _ert_training_step(
                                model, batch, loss_fn, device, args.ert_budgets
                            )
                        else:
                            loss_dict = _training_step(model, batch, loss_fn, device)
                except RuntimeError as exc:
                    print(f"[warn] Skipping batch (step {global_step}): {exc}",
                          file=sys.stderr, flush=True)
                    optimiser.zero_grad()
                    continue

                total_loss = loss_dict["total"] / args.grad_accum
                total_loss.backward()

                if (local_step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimiser.step()
                    optimiser.zero_grad()

                # Build log entry (key names differ between Stage 1/2 and ERT)
                log_entry: dict = {
                    "step":           global_step,
                    "epoch":          epoch,
                    "total_loss":     loss_dict["total"].item(),
                    "alignment_loss": loss_dict["alignment"].item(),
                }
                if args.stage == 3:
                    log_entry["kl_loss"]    = loss_dict["kl"].item()
                    log_entry["ert_budget"] = loss_dict["budget"].item()
                else:
                    log_entry["lm_loss"]         = loss_dict["lm"].item()
                    log_entry["robustness_loss"]  = loss_dict["robustness"].item()

                line = json.dumps(log_entry)
                print(line, flush=True)
                log_fh.write(line + "\n")
                log_fh.flush()

                # --- Mandatory generation quality check (Stage 2 and 3) ---
                if (
                    args.stage in (2, 3)
                    and args.gen_check_interval > 0
                    and global_step > 0
                    and global_step % args.gen_check_interval == 0
                ):
                    print(f"\n[gen-check] Step {global_step} — checking generation quality...",
                          flush=True)
                    healthy = _generation_quality_check(
                        model, tokenizer, device, n_prompts=args.gen_check_prompts
                    )
                    if not healthy:
                        print(
                            f"\n[ABORT] Generation quality check FAILED at step {global_step}.\n"
                            "Degenerate output detected (likely memorisation/collapse).\n"
                            "Saving checkpoint before exit for forensic analysis.",
                            flush=True,
                        )
                        _save_checkpoint(model, args.output_dir, args, args.stage)
                        gen_check_failed = True
                        break

                global_step += 1

            if gen_check_failed:
                break

            # Save checkpoint after each epoch
            print(f"\n[Checkpoint] Saving after epoch {epoch + 1}...", flush=True)
            _save_checkpoint(model, args.output_dir, args, args.stage)
            print(f"[Checkpoint] Saved to {args.output_dir}", flush=True)

    # Final checkpoint
    _save_checkpoint(model, args.output_dir, args, args.stage)
    print(f"\nTraining complete. {global_step} steps.", flush=True)


if __name__ == "__main__":
    main()

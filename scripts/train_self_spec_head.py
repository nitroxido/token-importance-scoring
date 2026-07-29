#!/usr/bin/env python
"""P9: Train intermediate LM head for self-speculative decoding.

Goal: Train a lightweight LM head that reads hidden states from layer
`intermediate_depth` (e.g., 16 of 32) and predicts the next token.
This enables the Mistral-7B model to perform self-speculative decoding:
  - Draft: run first 16 layers + trained head → predict next token
  - Target: run all 32 layers → verify token

Loss: Cross-entropy between intermediate predictions and target logits.

With QLoRA adapter: 1.7M trainable params, ~2-3 GB extra VRAM during training.

Usage:
    source .venv/bin/activate
    python scripts/train_self_spec_head.py \\
        --model mistralai/Mistral-7B-v0.3 \\
        --intermediate-depth 16 \\
        --output-dir checkpoints/self_spec_head_16 \\
        --steps 200 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance.training.retrieval_data import RetrievalDataset


MODEL_NAME = "mistralai/Mistral-7B-v0.3"


class IntermediateLMHead(nn.Module):
    """Simple LM head that reads from intermediate layer and predicts next token."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, d] from intermediate layer
        Returns:
            logits: [B, T, vocab_size]
        """
        normed = self.norm(hidden_states.to(torch.float32))
        logits = self.lm_head(normed)
        return logits


def _load_model(model_name: str, device: torch.device) -> tuple[AutoModelForCausalLM, int]:
    """Load model in 4-bit with QLoRA adapter setup."""
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map=device,
        attn_implementation="eager",  # Needed for activation hooks
    )
    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    return model, vocab_size, hidden_size


def _setup_adapter_and_head(
    model: AutoModelForCausalLM,
    vocab_size: int,
    hidden_size: int,
) -> tuple[nn.Module, AutoModelForCausalLM]:
    """Attach QLoRA adapter to the base model and create intermediate head."""
    # Wrap with QLoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Create intermediate head (not wrapped, just trainable)
    head = IntermediateLMHead(hidden_size, vocab_size)
    head = head.to(model.device)

    return head, model


def _capture_intermediate_hidden(model, ids, intermediate_depth, device):
    """Capture hidden states at a given layer depth."""
    # Handle PEFT-wrapped and plain models
    if hasattr(model, "base_model"):
        # PEFT-wrapped: access via base_model.model.model.layers
        layers = model.base_model.model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Plain HF model: access via model.model.layers
        layers = model.model.layers
    else:
        raise AttributeError("Could not find transformer layers in model")

    captured = {}

    def _hook(module, input, output):
        # output is tuple; first element is hidden state
        captured["h"] = output[0].detach().clone()

    handle = layers[intermediate_depth - 1].register_forward_hook(_hook)

    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        handle.remove()

    return captured.get("h", None)  # [B, T, d] or None


def train_step(
    model: AutoModelForCausalLM,
    head: nn.Module,
    batch_ids: torch.Tensor,
    intermediate_depth: int,
    device: torch.device,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    """One training step: compute intermediate → full, measure loss."""
    B, T = batch_ids.shape

    # Get intermediate hidden states (no grad needed for capture)
    intermediate_h = _capture_intermediate_hidden(model, batch_ids, intermediate_depth, device)
    if intermediate_h is None:
        return 0.0

    # Get full model logits (target labels)
    with torch.no_grad():
        out_full = model(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids))
        target_logits = out_full.logits.detach()  # [B, T, V]

    # Predict from intermediate hidden states
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_logits = head(intermediate_h)  # [B, T, V]

        # Loss: predict all positions (including last which predicts EOS)
        # Shift: predict token at position t given context up to t-1
        # For simplicity, use all positions as is
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_logits.argmax(dim=-1).reshape(-1),
        )

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def evaluate(
    model: AutoModelForCausalLM,
    head: nn.Module,
    tokenizer,
    intermediate_depth: int,
    n_examples: int,
    device: torch.device,
) -> dict:
    """Evaluate intermediate head on a small set."""
    dataset = RetrievalDataset(
        tokenizer=tokenizer,
        context_tokens=256,
        budgets=[0.5],
        seed=42,
    )
    it = iter(dataset)

    correct_preds = 0
    total_preds = 0

    for _ in range(n_examples):
        batch = next(it)
        ids = batch.input_ids.to(device)
        if ids.shape[1] < 8:
            continue

        intermediate_h = _capture_intermediate_hidden(model, ids, intermediate_depth, device)
        if intermediate_h is None:
            continue

        with torch.no_grad():
            out_full = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            target_logits = out_full.logits  # [B, T, V]
            target_preds = target_logits.argmax(dim=-1)  # [B, T]

            pred_logits = head(intermediate_h)  # [B, T, V]
            pred_preds = pred_logits.argmax(dim=-1)  # [B, T]

            correct_preds += (pred_preds == target_preds).sum().item()
            total_preds += pred_preds.numel()

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    return {"accuracy": accuracy, "n_examples": n_examples, "total_tokens": total_preds}


def main():
    p = argparse.ArgumentParser(description="P9: Train Intermediate LM Head")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--intermediate-depth", type=int, default=16)
    p.add_argument("--output-dir", default="checkpoints/self_spec_head_16")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--context-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--device", default="")
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"[setup] Loading model {args.model}...", flush=True)
    model, vocab_size, hidden_size = _load_model(args.model, device)

    print(f"[setup] Hidden size: {hidden_size}, Vocab size: {vocab_size}", flush=True)
    print(f"[setup] Setting up QLoRA adapter + intermediate head...", flush=True)
    head, model = _setup_adapter_and_head(model, vocab_size, hidden_size)

    # Count trainable params
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] Trainable params: {trainable:,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optimizer
    params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    # Training loop
    dataset = RetrievalDataset(
        tokenizer=tokenizer,
        context_tokens=args.context_tokens,
        budgets=[0.5],
        seed=789,
    )
    it = iter(dataset)

    losses = []
    print(f"\n[train] Starting {args.steps} steps, intermediate_depth={args.intermediate_depth}", flush=True)

    for step in range(args.steps):
        batch = next(it)
        ids = batch.input_ids.to(device)

        if ids.shape[1] < 8:
            continue

        loss = train_step(model, head, ids, args.intermediate_depth, device, optimizer, scaler)
        losses.append(loss)
        scheduler.step()

        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"[train] step {step + 1:4d}/{args.steps}  loss={avg_loss:.4f}", flush=True)

        if (step + 1) % args.eval_every == 0:
            print(f"[eval] Evaluating at step {step + 1}...", flush=True)
            head.eval()
            model.eval()
            metrics = evaluate(model, head, tokenizer, args.intermediate_depth, n_examples=5, device=device)
            print(f"[eval]   Accuracy: {metrics['accuracy']:.3f}", flush=True)
            head.train()
            model.train()

    # Save head
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(head.state_dict(), out_dir / "head_weights.pt")
    torch.save(
        {
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "intermediate_depth": args.intermediate_depth,
        },
        out_dir / "config.pt",
    )

    # Save adapter
    model.save_pretrained(out_dir / "adapter")

    # Final eval
    print(f"\n[final] Running final evaluation...", flush=True)
    head.eval()
    model.eval()
    final_metrics = evaluate(model, head, tokenizer, args.intermediate_depth, n_examples=10, device=device)
    print(f"[final]   Accuracy: {final_metrics['accuracy']:.3f}", flush=True)

    # Summary
    summary = {
        "model": args.model,
        "intermediate_depth": args.intermediate_depth,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_accuracy": final_metrics["accuracy"],
        "final_loss": losses[-1] if losses else 0.0,
        "output_dir": str(out_dir),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] Checkpoint saved to {out_dir}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""P5-A: Fine-tune the LLaMA-3.2-1B drafter with importance-weighted CE.

Instead of just scaling input embeddings (Phase 5 embedding bias), this script
directly fine-tunes the drafter to predict important tokens more accurately.

Loss: L = sum_t w_t * CE(drafter_pred_t, target_pred_t)
where  w_t = 1 + lambda_importance * imp_t / 100

Tokens with higher TIS importance scores get a higher loss weight, teaching
the drafter to prioritize getting those tokens right.

Expected improvement over embedding-scaling: +5-10% acceptance length.

Usage:
    source .venv/bin/activate
    python scripts/train_drafter_imp_ce.py \\
        --tis-checkpoint checkpoints/llama31_8b_tis \\
        --output-dir checkpoints/drafter_imp_ce \\
        --lambda-imp 2.0 \\
        --steps 2000 --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.training.retrieval_data import RetrievalDataset

TARGET_MODEL  = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DRAFTER_MODEL = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tis-checkpoint", required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--steps",          type=int,   default=2000)
    p.add_argument("--grad-accum",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--lambda-imp",     type=float, default=2.0,
                   help="Importance weight multiplier (w_t = 1 + lambda * imp_t/100)")
    p.add_argument("--context-tokens", type=int,   default=384)
    p.add_argument("--log-interval",   type=int,   default=100)
    p.add_argument("--device",         default="")
    return p.parse_args()


def _load_target(ckpt: str, device):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = PatchedCausalLM.from_pretrained(TARGET_MODEL, config=TISConfig(),
                                             quantization_config=quant, device_map=device).to(device)
    tis_pt = Path(ckpt) / "tis_components.pt"
    if tis_pt.exists():
        state = torch.load(tis_pt, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model


def _load_drafter(device):
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    d = AutoModelForCausalLM.from_pretrained(DRAFTER_MODEL, quantization_config=quant,
                                              device_map=device, attn_implementation="eager")
    # QLoRA: freeze 4-bit base, add tiny trainable LoRA adapters
    d = prepare_model_for_kbit_training(d, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    d = get_peft_model(d, lora_cfg)
    trainable = sum(p.numel() for p in d.parameters() if p.requires_grad)
    print(f"[drafter] QLoRA trainable params: {trainable:,} ({trainable*4/1e6:.1f} MB float32)", flush=True)
    return d


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = RetrievalDataset(tokenizer=tokenizer, context_tokens=args.context_tokens,
                                budgets=[0.5], seed=77)

    # ── Stage 1: Pre-compute target logits and importance scores ──────────────
    cache_file = Path(args.output_dir) / "target_cache.pt"
    if not cache_file.exists():
        print("[stage1] Computing target logits...", flush=True)
        target = _load_target(args.tis_checkpoint, device)
        d_model = target._base_model.config.hidden_size
        norm_w   = target._base_model.model.norm.weight.detach().to(torch.bfloat16).to(device)
        norm_eps = getattr(target._base_model.model.norm, "variance_epsilon", 1e-6)
        lm_w     = target._base_model.lm_head.weight.detach().to(torch.bfloat16).to(device)
        vocab_size = target._base_model.config.vocab_size

        cache_data = []
        it = iter(dataset)
        n_cache = args.steps * 1  # 1 sample per step
        t1 = time.time()
        for i in range(n_cache):
            batch = next(it)
            ids = batch.input_ids.to(device)
            T = ids.shape[1]
            if T < 8: continue
            with torch.no_grad():
                seed = torch.full((T,), 50, dtype=torch.uint8, device=device)
                out_t = target(input_ids=ids, importance_scores=seed,
                               attention_mask=torch.ones_like(ids), output_hidden_states=True)
            hidden_t = out_t.hidden_states[-1].float()
            imp = target.importance_head.direct_score(hidden_t.squeeze(0))
            h = hidden_t[:, :-1, :].to(torch.bfloat16)
            rms = h.pow(2).mean(-1, keepdim=True).add(norm_eps).sqrt()
            normed = (h / rms) * norm_w
            target_ids = F.linear(normed, lm_w).float()[0].argmax(dim=-1).cpu()  # [T-1]
            cache_data.append({"ids": ids.cpu(), "imp": imp.cpu(), "target_ids": target_ids})
            if (i+1) % 200 == 0:
                print(f"  Cached {i+1}/{n_cache} examples ({time.time()-t1:.0f}s)", flush=True)

        torch.save(cache_data, cache_file)
        print(f"[stage1] Cache saved ({len(cache_data)} examples)", flush=True)
        # Free target model
        del target, norm_w, lm_w
        torch.cuda.empty_cache()
    else:
        print(f"[stage1] Loading cache from {cache_file}", flush=True)

    cache_data = torch.load(cache_file, map_location="cpu")
    vocab_size = cache_data[0]["target_ids"].max().item() + 1
    # Real vocab size from tokenizer
    vocab_size = len(tokenizer)

    # ── Stage 2: Fine-tune drafter against cached target predictions ──────────
    print("[stage2] Fine-tuning drafter...", flush=True)
    drafter = _load_drafter(device)
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    log_file = open(Path(args.output_dir) / "train_p5a.jsonl", "w")

    drafter.train()
    optimizer.zero_grad()
    accum = 0
    t0 = time.time()
    data_idx = 0

    for step in range(1, args.steps + 1):
        entry = cache_data[data_idx % len(cache_data)]
        data_idx += 1
        ids = entry["ids"].to(device)
        imp = entry["imp"].to(device)
        target_ids = entry["target_ids"].to(device)
        T = ids.shape[1]

        # Drafter forward (with grad)
        drafter_out = drafter(input_ids=ids[:, :-1], attention_mask=torch.ones(1, T-1, device=device))
        drafter_logits = drafter_out.logits.float()  # [1, T-1, V]

        # Importance-weighted CE
        weights = 1.0 + args.lambda_imp * (imp[:-1] / 100.0)  # [T-1]
        ce_per_token = F.cross_entropy(
            drafter_logits.view(-1, vocab_size), target_ids.view(-1), reduction="none"
        )
        loss = (ce_per_token * weights).mean()

        (loss / args.grad_accum).backward()
        accum += 1
        if accum == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            accum = 0

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            row = {"step": step, "loss": loss.item(), "elapsed_s": round(elapsed, 1)}
            print(f"[step {step:>5}] loss={loss.item():.4f}  ({elapsed/step*1000:.0f}ms/step)", flush=True)
            log_file.write(json.dumps(row) + "\n"); log_file.flush()

    log_file.close()
    out_dir = Path(args.output_dir)
    drafter.save_pretrained(str(out_dir))
    meta = {"phase": "5a", "tis_checkpoint": args.tis_checkpoint,
            "lambda_imp": args.lambda_imp, "steps": args.steps}
    with open(out_dir / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"[save] Done: {out_dir}", flush=True)


if __name__ == "__main__":
    train(_parse())

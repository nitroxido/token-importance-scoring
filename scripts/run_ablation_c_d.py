#!/usr/bin/env python
"""
scripts/run_ablation_c_d.py

Ablation studies C and D — locally executable on RTX 5070 (8 GB VRAM).

Ablation C: Pruning aggressiveness — reuses the compression comparison results
            (already computed in run_compression_eval.py). Generates the figures.

Ablation D: Model dependence — test if the TIS head trained on Mistral-7B
            scores passages correctly when the generator is Mistral-7B Instruct.
            (Cross-architecture: TIS trained on base, generate with Instruct.)

NOTE: Ablation A (depth) is handled by:
    python scripts/run_speculative_decoding_eval.py --setup c --depth-sweep

NOTE: Ablation B (ranking quality vs training steps) requires multiple TIS
      checkpoints at different step counts. See VASTAI-RUNBOOK.md.

Usage:
    source .venv/bin/activate
    python scripts/run_ablation_c_d.py [--ablation c] [--ablation d] [--n-examples 60]
"""
from __future__ import annotations

import argparse, csv, logging, os, sys, warnings
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

MISTRAL_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
TIS_CKPT      = os.path.join(_ROOT, "checkpoints", "v8b_hard_anchor")
DATA_PATH     = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR   = os.path.join(_ROOT, "results")
D_MODEL       = 4096
ABLATION_DIR  = os.path.join(RESULTS_DIR, "ablations")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", nargs="+", default=["c","d"],
                   choices=["c","d"])
    p.add_argument("--n-examples", type=int, default=60)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


# ── Ablation C: Pruning aggressiveness (reuse compression_results.csv) ─────────
def run_ablation_c():
    """
    Ablation C is already run: the compression comparison produced
    results/compression_results.csv with keep_ratio in {0.3, 0.5, 0.7, 0.9}.
    We just need to plot the TIS curve more prominently.
    """
    csv_path = os.path.join(RESULTS_DIR, "compression_results.csv")
    if not os.path.exists(csv_path):
        print("[ablation C] ERROR: compression_results.csv not found. "
              "Run scripts/run_compression_eval.py first.", flush=True)
        return

    df = pd.read_csv(csv_path)
    tis = df[df["method"]=="tis"].sort_values("compression_ratio")
    baseline_em = df[df["method"]=="no_compression"]["em"].values[0]

    # ── Figure C1: EM and F1 vs keep_ratio ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].axhline(baseline_em, color="gray", linestyle="--",
                    label=f"No compression (EM={baseline_em:.3f})")
    axes[0].plot(tis["keep_ratio"], tis["em"],
                 color="tab:green", marker="o", linewidth=2, label="TIS pruning")
    axes[0].set_xlabel("keep_ratio"); axes[0].set_ylabel("Exact Match")
    axes[0].set_title("Ablation C: EM vs Pruning Aggressiveness\n(MS-MARCO, Mistral-7B Instruct 4-bit)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].invert_xaxis()

    axes[1].axhline(df[df["method"]=="no_compression"]["f1"].values[0],
                    color="gray", linestyle="--", label="No compression")
    axes[1].plot(tis["keep_ratio"], tis["f1"],
                 color="tab:green", marker="s", linewidth=2, label="TIS pruning")
    axes[1].set_xlabel("keep_ratio"); axes[1].set_ylabel("Token F1")
    axes[1].set_title("Ablation C: F1 vs Pruning Aggressiveness")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].invert_xaxis()

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ablation_c_pruning.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[ablation C] Saved {out}", flush=True)

    # ── Figure C2: Compression overhead ─────────────────────────────────────────
    methods = ["no_compression","tfidf","embedding","tis"]
    labels  = ["No compress","TF-IDF","Embedding","TIS (ours)"]
    colors  = ["gray","tab:orange","tab:blue","tab:green"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    compress_ms = []
    em_at_07 = []
    for m in methods:
        sub = df[df["method"]==m]
        if m == "no_compression":
            compress_ms.append(0)
            em_at_07.append(sub["em"].values[0])
        else:
            row07 = sub[sub["keep_ratio"]==0.7]
            compress_ms.append(row07["compress_ms"].values[0] if not row07.empty else 0)
            em_at_07.append(row07["em"].values[0] if not row07.empty else 0)

    bars = ax.bar(labels, compress_ms, color=colors, alpha=0.8)
    for bar, em in zip(bars, em_at_07):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
                f"EM={em:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("Compression overhead (ms/query)")
    ax.set_title("Ablation C: Compression Latency @ keep=0.7\n(labels show EM at that compression level)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "ablation_c_latency.png")
    plt.savefig(out2, dpi=150); plt.close()
    print(f"[ablation C] Saved {out2}", flush=True)

    # Summary
    print("\n[ablation C] TIS pruning knee:", flush=True)
    for _, r in tis.iterrows():
        print(f"  keep={r['keep_ratio']:.1f}  EM={r['em']:.3f}  "
              f"F1={r['f1']:.3f}  CR={r['compression_ratio']:.3f}  "
              f"Δ_EM={r['em']-baseline_em:+.3f}", flush=True)


# ── Ablation D: Model dependence ────────────────────────────────────────────────
def run_ablation_d(n_examples: int, device: str):
    """
    Test TIS scoring with Mistral-7B Instruct (same arch as training base).
    Since TIS was trained on Mistral-7B-v0.3 (base), applying to Instruct
    variant tests within-architecture transfer.

    Cross-architecture (Llama 8B) is deferred to VASTAI-RUNBOOK.md (needs
    retraining the TIS head on Llama hidden states).
    """
    import json, re

    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r"^\s*(a|an|the)\s+", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def exact_match(pred, golds):
        p = normalize(pred)
        return int(any(p == normalize(g) or normalize(g) in p or p in normalize(g)
                       for g in golds))

    # Load model and TIS head
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from token_importance.config import TISConfig
    from token_importance.model.importance_head import ImportanceUpdateHead

    print("[ablation D] Loading model + TIS head...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    ckpt = torch.load(os.path.join(TIS_CKPT, "tis_components.pt"),
                      map_location="cpu", weights_only=True)
    head = ImportanceUpdateHead(D_MODEL, TISConfig(), num_heads=4)
    head.load_state_dict(ckpt["importance_head"], strict=False)
    head = head.to(device).eval()
    print(f"[ablation D] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Load data
    import datasets as hf
    ds = hf.load_from_disk(DATA_PATH)
    examples = []
    for raw in ds:
        if len(examples) >= n_examples: break
        p = raw["passages"]
        texts = p["passage_text"]
        sel   = p["is_selected"]
        ans_raw = raw["answers"]
        if isinstance(ans_raw, str): ans_raw = json.loads(ans_raw)
        gold_indices = [i for i,s in enumerate(sel) if s==1]
        valid_ans = [a.strip() for a in ans_raw
                     if a.strip() not in ("No Answer Present.","[]","")]
        if not gold_indices or not valid_ans: continue
        gold_idx     = gold_indices[0]
        gold_passage = texts[gold_idx]
        distractors  = [t for i,t in enumerate(texts) if i!=gold_idx][:4]
        while len(distractors) < 4:
            distractors.append(distractors[-1] if distractors else "No info.")
        for pos_name, gold_pos in [("early",0),("middle",2),("end",4)]:
            plist = list(distractors); plist.insert(gold_pos, gold_passage)
            context = "\n\n".join(f"Passage {i+1}: {p}" for i,p in enumerate(plist))
            instruction = (
                f"Read the following passages and answer the question.\n\n"
                f"{context}\n\nQuestion: {raw['query']}"
            )
            examples.append({
                "query_id": raw["query_id"],
                "question": raw["query"],
                "passages": plist,
                "gold_passage": gold_passage,
                "answers": valid_ans,
                "gold_pos": gold_pos,
                "pos_name": pos_name,
                "context": context,
                "prompt": f"[INST] {instruction} [/INST]",
            })
    print(f"[ablation D] {len(examples)} test cases", flush=True)

    def score_passages(passages, question):
        """Score passages using TIS head (direct_score path)."""
        passage_scores = []
        for p in passages:
            enc = tokenizer(p, return_tensors="pt", truncation=True,
                            max_length=400).to(device)
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=True)
                hidden = out.hidden_states[-1].float()
                scores = head.direct_score(hidden)[0].cpu().numpy()
            passage_scores.append(float(scores.mean()))
        return passage_scores

    @torch.inference_mode()
    def generate_answer(prompt, max_new_tokens=30):
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=3800).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    # ── Baseline (no TIS, original passage order) ───────────────────────────────
    results_baseline, results_reorder = [], []

    for i, ex in enumerate(examples):
        # Baseline
        pred_base = generate_answer(ex["prompt"])
        em_base   = exact_match(pred_base, ex["answers"])

        # TIS reorder
        p_scores       = score_passages(ex["passages"], ex["question"])
        reordered      = [p for _,p in sorted(zip(p_scores, ex["passages"]), reverse=True)]
        ctx_reordered  = "\n\n".join(f"Passage {i+1}: {p}" for i,p in enumerate(reordered))
        instruction_ro = (
            f"Read the following passages and answer the question.\n\n"
            f"{ctx_reordered}\n\nQuestion: {ex['question']}"
        )
        pred_ro = generate_answer(f"[INST] {instruction_ro} [/INST]")
        em_ro   = exact_match(pred_ro, ex["answers"])

        # Gold position after reordering
        gold_new_pos = reordered.index(ex["gold_passage"]) if ex["gold_passage"] in reordered else -1

        results_baseline.append({"pos_name": ex["pos_name"], "em": em_base,
                                   "gold_pos": ex["gold_pos"]})
        results_reorder.append( {"pos_name": ex["pos_name"], "em": em_ro,
                                   "gold_pos": ex["gold_pos"],
                                   "gold_new_pos": gold_new_pos})

        if (i+1) % 20 == 0:
            em_b = np.mean([r["em"] for r in results_baseline])
            em_r = np.mean([r["em"] for r in results_reorder])
            print(f"  {i+1}/{len(examples)}  EM_base={em_b:.3f}  EM_reorder={em_r:.3f}",
                  flush=True)

    # ── Summary ─────────────────────────────────────────────────────────────────
    def by_pos(results):
        df = pd.DataFrame(results)
        return {pos: df[df["pos_name"]==pos]["em"].mean()
                for pos in ["early","middle","end"]}

    base_pos  = by_pos(results_baseline)
    reord_pos = by_pos(results_reorder)
    base_gap  = base_pos["early"]  - base_pos["middle"]
    reord_gap = reord_pos["early"] - reord_pos["middle"]

    print("\n[ablation D] Mistral-7B Instruct with TIS (trained on Mistral-7B base):", flush=True)
    print(f"  Baseline:  early={base_pos['early']:.3f}  mid={base_pos['middle']:.3f}  "
          f"end={base_pos['end']:.3f}  LITM_gap={base_gap:.3f}", flush=True)
    print(f"  TIS reorder: early={reord_pos['early']:.3f}  mid={reord_pos['middle']:.3f}  "
          f"end={reord_pos['end']:.3f}  LITM_gap={reord_gap:.3f}", flush=True)
    print(f"  Gap reduction: {base_gap:.3f} → {reord_gap:.3f}  "
          f"({'improved' if reord_gap < base_gap else 'no change'})", flush=True)

    # Save CSV — normalize keys across baseline (no gold_new_pos) and reorder rows
    os.makedirs(ABLATION_DIR, exist_ok=True)
    all_rows = []
    for r, rr in zip(results_baseline, results_reorder):
        all_rows.append({"method": "baseline",    "gold_new_pos": -1, **r})
        all_rows.append({"method": "tis_reorder", "gold_new_pos": rr.get("gold_new_pos",-1),
                          **{k:v for k,v in rr.items() if k != "gold_new_pos"}})
    csv_out = os.path.join(ABLATION_DIR, "model_dependence.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)
    print(f"[ablation D] Saved {csv_out}", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = ["early","middle","end"]
    x = np.arange(len(positions))
    w = 0.35
    ax.bar(x - w/2, [base_pos[p] for p in positions], w,
           label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x + w/2, [reord_pos[p] for p in positions], w,
           label="TIS Reordering", color="tab:green", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["Early","Middle","End"])
    ax.set_ylabel("Exact Match")
    ax.set_title(f"Ablation D: TIS model dependence\n"
                 f"Mistral-7B Instruct (TIS trained on Mistral-7B base)\n"
                 f"LITM gap: {base_gap:.3f}→{reord_gap:.3f} "
                 f"({(base_gap-reord_gap)/max(base_gap,1e-6)*100:.0f}% reduction)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(ABLATION_DIR, "ablation_d_model_dependence.png")
    plt.savefig(out_fig, dpi=150); plt.close()
    print(f"[ablation D] Saved {out_fig}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = _parse()
    os.makedirs(ABLATION_DIR, exist_ok=True)

    if "c" in args.ablation:
        print("\n" + "="*60, flush=True)
        print("[Ablation C] Pruning aggressiveness (uses existing compression_results.csv)", flush=True)
        run_ablation_c()

    if "d" in args.ablation:
        print("\n" + "="*60, flush=True)
        print("[Ablation D] Model dependence (Mistral base→Instruct transfer)", flush=True)
        run_ablation_d(args.n_examples, args.device)

    print("\n[main] Ablation studies done!", flush=True)


if __name__ == "__main__":
    main()

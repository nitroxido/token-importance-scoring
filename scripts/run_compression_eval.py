#!/usr/bin/env python
"""
scripts/run_compression_eval.py

Context compression comparison: TIS vs TF-IDF vs Embedding reranker.
Tests 4 keep_ratios × 4 methods on MS-MARCO passage QA.

Methods:
  M0  no_compression     — full context, no modification
  M1  tfidf              — sentence-level TF-IDF similarity to query
  M2  embedding          — sentence-level MiniLM cosine similarity
  M3  tis                — TIS sentence-level scoring (aggregate token scores)

Usage:
    source .venv/bin/activate
    python scripts/run_compression_eval.py [--n-examples 60] [--device cuda]

Outputs:
    results/compression_results.csv
    results/compression_summary.csv
    results/quality_vs_compression.png
    results/compression_final_report.txt
"""
from __future__ import annotations

import argparse, csv, logging, os, re, sys, time, warnings
from pathlib import Path

# ── Suppress noisy transformers / bitsandbytes warnings ───────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

BASE_MODEL   = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
CHECKPOINT   = os.path.join(_ROOT, "checkpoints", "v8b_hard_anchor")
DATA_PATH    = os.path.join(_ROOT, "data", "msmarco_quick", "train")
RESULTS_DIR  = os.path.join(_ROOT, "results")
D_MODEL      = 4096
KEEP_RATIOS  = [0.9, 0.7, 0.5, 0.3]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-examples",    type=int,   default=60)
    p.add_argument("--max-new-tokens",type=int,   default=30)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


# ── Model and head loading ─────────────────────────────────────────────────────
def load_model(device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[setup] Loading {BASE_MODEL}...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    # Clear max_length so max_new_tokens is the sole generation limit
    if hasattr(mdl, "generation_config"):
        mdl.generation_config.max_length = None
    print(f"[setup] Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    return mdl, tok


def load_tis_head(device):
    from token_importance.config import TISConfig
    from token_importance.model.importance_head import ImportanceUpdateHead
    ckpt = torch.load(os.path.join(CHECKPOINT, "tis_components.pt"), map_location="cpu", weights_only=True)
    head = ImportanceUpdateHead(D_MODEL, TISConfig(), num_heads=4)
    head.load_state_dict(ckpt["importance_head"], strict=False)
    return head.to(device).eval()


def load_minilm():
    from sentence_transformers import SentenceTransformer
    print("[setup] Loading MiniLM (CPU)...", flush=True)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_examples(n_examples):
    import datasets as hf
    import json
    ds = hf.load_from_disk(DATA_PATH)
    examples, base_count = [], 0
    for raw in ds:
        if base_count >= n_examples: break
        p = raw["passages"]
        texts   = p["passage_text"]
        sel     = p["is_selected"]
        ans_raw = raw["answers"]
        if isinstance(ans_raw, str): ans_raw = json.loads(ans_raw)
        gold_indices = [i for i,s in enumerate(sel) if s==1]
        valid_ans    = [a.strip() for a in ans_raw
                        if a.strip() and a.strip() not in ("No Answer Present.","[]")]
        if not gold_indices or not valid_ans: continue

        gold_idx     = gold_indices[0]
        gold_passage = texts[gold_idx]
        distractors  = [t for i,t in enumerate(texts) if i != gold_idx][:4]
        while len(distractors) < 4:
            distractors.append(distractors[-1] if distractors else "No relevant information.")

        # One example: 5 passages concatenated as single context
        all_passages = [gold_passage] + distractors
        context = "\n\n".join(f"Passage {i+1}: {p}" for i,p in enumerate(all_passages))

        examples.append({
            "query_id": raw["query_id"],
            "question": raw["query"],
            "context":  context,
            "answers":  valid_ans,
        })
        base_count += 1

    print(f"[data] {len(examples)} examples loaded", flush=True)
    return examples


# ── Text utilities ─────────────────────────────────────────────────────────────
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def token_count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_to_budget(sentences, ranked_indices, budget_tokens, tokenizer):
    kept, total = [], 0
    for i in ranked_indices:
        n = token_count(sentences[i], tokenizer)
        if total + n > budget_tokens: break
        kept.append((i, sentences[i]))
        total += n
    kept.sort(key=lambda x: x[0])
    return " ".join(s for _,s in kept) if kept else sentences[0]

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"^\s*(a|an|the)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()

def exact_match(pred, golds):
    p = normalize_answer(pred)
    return int(any(
        p == normalize_answer(g) or normalize_answer(g) in p or p in normalize_answer(g)
        for g in golds
    ))

def token_f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g: return 0.0
    common = set(p) & set(g)
    if not common: return 0.0
    return 2*(len(common)/len(p))*(len(common)/len(g)) / (len(common)/len(p) + len(common)/len(g))


# ── TIS scoring helper ─────────────────────────────────────────────────────────
@torch.inference_mode()
def tis_score_sentences(model, tokenizer, tis_head, context, query, keep_ratio):
    """
    1. Split context into sentences.
    2. Tokenize full context, run forward pass to get hidden states.
    3. Aggregate TIS scores per sentence (approximate by equal token splits).
    4. Keep top-keep_ratio sentences by mean score, restore original order.
    """
    sentences = split_sentences(context)
    if not sentences: return context

    budget = max(1, int(token_count(context, tokenizer) * keep_ratio))
    instruction = (
        f"Read the following passages and answer the question. "
        f"Give only the answer, 1-5 words.\n\n{context}\n\nQuestion: {query}"
    )
    enc = tokenizer(f"[INST] {instruction} [/INST]",
                    return_tensors="pt", truncation=True, max_length=3800).to(
                    next(model.parameters()).device)
    T = enc["input_ids"].shape[1]

    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1].float()          # [1, T, d]
    scores = tis_head.direct_score(hidden)[0]       # [T]

    # Map to sentences: divide scores evenly
    n_sent = len(sentences)
    tok_per_sent = max(1, T // n_sent)
    sent_scores = []
    for si in range(n_sent):
        lo = si * tok_per_sent
        hi = min(lo + tok_per_sent, T)
        sent_scores.append(float(scores[lo:hi].mean()))

    # Keep top-budget sentences by score
    ranked = sorted(range(n_sent), key=lambda i: sent_scores[i], reverse=True)
    kept, total = [], 0
    for i in ranked:
        n = token_count(sentences[i], tokenizer)
        if total + n > budget: break
        kept.append(i)
        total += n

    if not kept: kept = [ranked[0]]
    kept.sort()
    return " ".join(sentences[i] for i in kept)


# ── Compression methods ────────────────────────────────────────────────────────
def compress_none(context, query, keep_ratio, **kwargs): return context

def compress_tfidf(context, query, keep_ratio, tokenizer, **kwargs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sentences = split_sentences(context)
    if len(sentences) <= 1: return context
    budget = max(1, int(token_count(context, tokenizer) * keep_ratio))
    corpus = sentences + [query]
    tfidf  = TfidfVectorizer().fit_transform(corpus)
    scores = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    ranked = np.argsort(scores)[::-1]
    return truncate_to_budget(sentences, ranked, budget, tokenizer)

def compress_embedding(context, query, keep_ratio, tokenizer, minilm, **kwargs):
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sentences = split_sentences(context)
    if len(sentences) <= 1: return context
    budget = max(1, int(token_count(context, tokenizer) * keep_ratio))
    embs   = minilm.encode(sentences + [query], convert_to_numpy=True)
    scores = cos_sim(embs[[-1]], embs[:-1])[0]
    ranked = np.argsort(scores)[::-1]
    return truncate_to_budget(sentences, ranked, budget, tokenizer)

def compress_tis(context, query, keep_ratio, model, tokenizer, tis_head, **kwargs):
    return tis_score_sentences(model, tokenizer, tis_head, context, query, keep_ratio)


# ── LLM generation ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_answer(model, tokenizer, context, question, max_new_tokens=30):
    instruction = (
        f"Read the following passages and answer the question. "
        f"Give only the answer, 1-5 words.\n\n{context}\n\nQuestion: {question}"
    )
    prompt = f"[INST] {instruction} [/INST]"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3800).to(
        next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()


# ── Main evaluation loop ───────────────────────────────────────────────────────
def run_eval(examples, model, tokenizer, tis_head, minilm, max_new_tokens):
    methods = {
        "no_compression": compress_none,
        "tfidf":          compress_tfidf,
        "embedding":      compress_embedding,
        "tis":            compress_tis,
    }
    results_csv = os.path.join(RESULTS_DIR, "compression_results.csv")
    all_rows = []

    for method_name, compress_fn in methods.items():
        ratios = [1.0] if method_name == "no_compression" else KEEP_RATIOS
        for keep_ratio in ratios:
            print(f"\n{'='*55}", flush=True)
            print(f"[{method_name} @ keep={keep_ratio}]  n={len(examples)}", flush=True)
            em_list, f1_list, cr_list, c_ms_list, gen_ms_list = [], [], [], [], []

            for i, ex in enumerate(examples):
                # ── Compress ──
                t0 = time.perf_counter()
                compressed = compress_fn(
                    ex["context"], ex["question"], keep_ratio,
                    tokenizer=tokenizer, model=model, tis_head=tis_head, minilm=minilm
                )
                c_ms = (time.perf_counter() - t0) * 1000

                # ── Generate ──
                t1 = time.perf_counter()
                pred = generate_answer(model, tokenizer, compressed,
                                        ex["question"], max_new_tokens)
                gen_ms = (time.perf_counter() - t1) * 1000

                tok_before = token_count(ex["context"], tokenizer)
                tok_after  = token_count(compressed, tokenizer)
                cr = tok_after / max(tok_before, 1)

                em_list.append(exact_match(pred, ex["answers"]))
                f1_list.append(max(token_f1(pred, g) for g in ex["answers"]))
                cr_list.append(cr)
                c_ms_list.append(c_ms)
                gen_ms_list.append(gen_ms)

                if (i+1) % 10 == 0 or i == 0:
                    em_now = sum(em_list)/len(em_list)
                    print(f"  {i+1}/{len(examples)}  EM={em_now:.3f}  "
                          f"pred={pred[:40]!r}  gold={ex['answers'][0][:30]!r}  "
                          f"gen={gen_ms:.0f}ms", flush=True)

            row = {
                "method":            method_name,
                "keep_ratio":        keep_ratio,
                "em":                round(np.mean(em_list),  4),
                "f1":                round(np.mean(f1_list),  4),
                "compression_ratio": round(np.mean(cr_list),  4),
                "compress_ms":       round(np.mean(c_ms_list),1),
                "gen_ms":            round(np.mean(gen_ms_list),1),
                "total_ms":          round(np.mean(c_ms_list) + np.mean(gen_ms_list), 1),
            }
            all_rows.append(row)
            print(f"  → EM={row['em']:.3f}  F1={row['f1']:.3f}  "
                  f"CR={row['compression_ratio']:.3f}  "
                  f"compress={row['compress_ms']:.0f}ms  "
                  f"gen={row['gen_ms']:.0f}ms", flush=True)

    # Save CSV
    fieldnames = ["method","keep_ratio","em","f1","compression_ratio",
                  "compress_ms","gen_ms","total_ms"]
    header_needed = not os.path.exists(results_csv)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[save] {results_csv}", flush=True)
    return all_rows


# ── Plotting and report ────────────────────────────────────────────────────────
def plot_and_report(rows):
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)

    # ── Quality vs Compression Ratio ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {
        "no_compression": ("tab:gray",   "--", "x", "No compression"),
        "tfidf":          ("tab:orange", "--", "s", "TF-IDF sentence"),
        "embedding":      ("tab:blue",   "-",  "o", "Embedding (MiniLM)"),
        "tis":            ("tab:green",  "-",  "^", "TIS sentence (ours) ★"),
    }

    for metric, ax, ylabel in [("em", axes[0], "Exact Match"),
                                 ("f1", axes[1], "Token F1")]:
        for method, (color, ls, marker, label) in styles.items():
            sub = df[df["method"]==method].sort_values("compression_ratio")
            if sub.empty: continue
            ax.plot(sub["compression_ratio"], sub[metric],
                    color=color, linestyle=ls, marker=marker,
                    label=label, linewidth=2, markersize=8)
        ax.set_xlabel("Compression Ratio (tokens kept / total)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Compression Ratio\n(MS-MARCO, k=5, Mistral-7B Instruct 4-bit)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "quality_vs_compression.png"), dpi=150)
    print("[plot] results/quality_vs_compression.png", flush=True)
    plt.close()

    # ── Latency breakdown ──────────────────────────────────────────────────────
    df_latency = df[df["keep_ratio"].isin([0.5, 1.0])].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_width = 0.35
    methods_ordered = [m for m in styles if m in df_latency["method"].unique()]
    x = np.arange(len(methods_ordered))
    compress_vals = []
    gen_vals      = []
    for m in methods_ordered:
        sub = df_latency[(df_latency["method"]==m)]
        kr  = 0.5 if m != "no_compression" else 1.0
        row = sub[sub["keep_ratio"]==kr]
        if row.empty:
            compress_vals.append(0); gen_vals.append(0)
        else:
            compress_vals.append(row["compress_ms"].values[0])
            gen_vals.append(row["gen_ms"].values[0])

    ax.bar(x, compress_vals, bar_width, label="Compress overhead (ms)", color="tab:orange", alpha=0.8)
    ax.bar(x, gen_vals, bar_width, bottom=compress_vals, label="LLM generation (ms)", color="tab:blue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([styles[m][3] for m in methods_ordered], rotation=15, ha="right")
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Latency Breakdown @ 50% compression (CR≈0.5)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "compression_latency.png"), dpi=150)
    print("[plot] results/compression_latency.png", flush=True)
    plt.close()

    # ── Summary CSV ───────────────────────────────────────────────────────────
    df.to_csv(os.path.join(RESULTS_DIR, "compression_summary.csv"), index=False)

    # ── Text report ───────────────────────────────────────────────────────────
    no_comp_em = df[df["method"]=="no_compression"]["em"].values[0]
    lines = ["# Compression Comparison Results\n"]
    lines.append(f"Baseline (no compression) EM: {no_comp_em:.3f}\n")
    lines.append("\n| Method | keep_ratio | EM | F1 | CR | Compress ms | Gen ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(["method","keep_ratio"]).iterrows():
        lines.append(f"| {r['method']} | {r['keep_ratio']} | {r['em']:.3f} | "
                     f"{r['f1']:.3f} | {r['compression_ratio']:.3f} | "
                     f"{r['compress_ms']:.0f} | {r['gen_ms']:.0f} |")

    # Sweet spot: best EM at CR ≤ 0.6
    sub06 = df[df["compression_ratio"] <= 0.65]
    lines.append("\n### Sweet spot (best EM at CR ≤ 0.65)")
    for method in ["tfidf","embedding","tis"]:
        best = sub06[sub06["method"]==method].sort_values("em", ascending=False)
        if not best.empty:
            r = best.iloc[0]
            lines.append(f"  {method:12s}: EM={r['em']:.3f}  F1={r['f1']:.3f}  "
                         f"CR={r['compression_ratio']:.3f}  "
                         f"compress={r['compress_ms']:.0f}ms")

    report_text = "\n".join(lines)
    print("\n" + report_text, flush=True)
    with open(os.path.join(RESULTS_DIR, "compression_final_report.txt"), "w") as f:
        f.write(report_text)
    print("\n[save] results/compression_final_report.txt", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    examples = load_examples(args.n_examples)
    if not examples:
        print("[error] No examples loaded", flush=True); sys.exit(1)

    model, tokenizer = load_model(args.device)
    tis_head         = load_tis_head(args.device)
    minilm           = load_minilm()

    if args.dry_run:
        print("[dry-run] Testing 3 examples per method...", flush=True)
        test_ex = examples[:3]
        for method, fn in [("tfidf", compress_tfidf),
                            ("embedding", compress_embedding),
                            ("tis", compress_tis)]:
            out = fn(test_ex[0]["context"], test_ex[0]["question"], 0.7,
                     tokenizer=tokenizer, model=model, tis_head=tis_head, minilm=minilm)
            pred = generate_answer(model, tokenizer, out, test_ex[0]["question"])
            toks_before = token_count(test_ex[0]["context"], tokenizer)
            toks_after  = token_count(out, tokenizer)
            print(f"  {method}: {toks_before}→{toks_after} tokens  "
                  f"pred={pred[:50]!r}", flush=True)
        print("[dry-run] Done.", flush=True)
        return

    rows = run_eval(examples, model, tokenizer, tis_head, minilm, args.max_new_tokens)
    plot_and_report(rows)
    print("\n[main] All done!", flush=True)


if __name__ == "__main__":
    main()

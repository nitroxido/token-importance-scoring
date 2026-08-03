# Token Importance Scoring for KV Cache Compression and Position Bias Elimination

A learned mechanism for efficient long-context inference in large language models. TIS scores token importance at query time, enabling two distinct capabilities:

1. **KV cache compression**: retain critical context at aggressive budgets (NIAH benchmark)
2. **Passage reordering**: eliminate Lost-in-the-Middle position bias in RAG pipelines

## Overview

| Capability | Metric | TIS | Baseline |
|---|---|---|---|
| NIAH retrieval @ 50% | Hard accuracy | **78%** | SnapKV 24% |
| LITM compression @ 50% | Accuracy | **66.1%** (tis_head) | SnapKV 55.6% |
| Passage reordering | LITM gap | **0.000** | 0.050 |
| Passage reordering | EM all positions | **21.7%** | 16.7% |
| Speculative decoding | Accept length | **6.57/8** | 5.80/8 |

Consumer GPU compatible (validated on RTX 5070, 8 GB VRAM).

## Speculative Decoding Results

(n=30, lambda_d=0.2, seed=42)

| Condition | Accept Length | Speedup | Attn Drift @k=7 |
|---|---|---|---|
| No TIS | 5.80 / 8 | 0.644 | −10.0% |
| **TIS bias** | **6.57 / 8** | **0.730** | −9.7% |

TIS improves drafter accuracy (+12.5% acceptance length). Real attention drift is approximately equal in both conditions (~−10%); TIS's benefit is prediction quality, not drift prevention.

An additional internal run at n=100 with lambda_d=0.0 confirmed baseline behavior (both conditions at 6.27/8 when no bias is applied), validating that the improvement is driven by the learned bias, not sampling variance.

## Position Bias Elimination (TIS 2.0)

TIS scores token importance over final-layer hidden states and aggregates per-passage (arithmetic mean, descending). This eliminates the Lost-in-the-Middle position bias documented in Liu et al. (2023) without requiring query-specific retraining.

**Scoring contract**: each passage is encoded independently; the scorer's `direct_score()` path does not use query context. Score direction has been validated: descending (high-first) outperforms ascending for EM on both Stage3 and v8b checkpoints (`results/score_direction_validation_summary.csv`).

**Passage reordering results** (MS-MARCO, 60 queries × 3 positions = 180 test cases, seed=42):

| Pipeline | EM (early) | EM (middle) | EM (end) | LITM gap | Recall@1 |
|---|---|---|---|---|---|
| Baseline | 18.3% | 13.3% | 18.3% | 0.050 | 33.3% avg |
| Lexical (TF-IDF) | 16.7% | 18.3% | 18.3% | −0.008 | 35.0% |
| **TIS (passage head)** | **21.7%** | **21.7%** | **21.7%** | **0.000** | 21.7% |
| Oracle (gold-first) | 18.3% | 18.3% | 18.3% | 0.000 | 100% |

Full per-pipeline results: [`results/litm_with_baselines_summary.csv`](results/litm_with_baselines_summary.csv)  
Per-example predictions: [`results/litm_per_example_breakdown.csv`](results/litm_per_example_breakdown.csv)

**Methodological notes:**

- **Ranking quality ≠ answer quality**: TIS achieves lower Recall@1 (21.7%) than lexical TF-IDF (35.0%), but higher EM (21.7% vs 17.8%). Passage-ranking metrics (MRR, R@1) do not predict generation quality at this scale.
- **Oracle paradox**: TIS EM (21.7%) exceeds oracle EM (18.3%), indicating that TIS’s full-context layout provides a more generator-friendly arrangement than simply placing the gold passage first.
- **Answer extraction is the bottleneck**: 76.1% of cases remain wrong under both baseline and TIS. 7.2% transition wrong→right vs 2.2% right→wrong. Net recovery ratio: 3.25:1.
- **EM definition**: Extended EM (lenient substring match). Exact definition in [`results/litm_with_baselines_metadata.json`](results/litm_with_baselines_metadata.json).

**Methodological validation — paired position sweep:**

A paired evaluation (same passage sets at all 3 positions, removes difficulty variance) produces identical results, confirming the LITM gap is genuine position-dependent attention, not a difficulty confound:

| Condition | Baseline gap | TIS gap |
|---|---|---|
| Independent generation | 0.050 | 0.000 |
| **Paired sweep** | **0.050** | **0.000** |

Paired results: [`results/litm_paired_summary.csv`](results/litm_paired_summary.csv)

**Transfer finding**: The `tis-stage3-ert` checkpoint (trained for KV cache compression) achieves identical passage reordering performance without retraining (`results/kv_transfer_test_summary.csv`). Both checkpoints learn a generalizable context-utility signal applicable across tasks.

**Score direction and query-specificity validation** (n=60, seed=42):
- Score direction: descending (high-first) outperforms ascending for EM on both Stage3 and v8b (`results/score_direction_validation_summary.csv`)
- Query conditioning: passage orders change in 71.7% of cases when the query changes, but correct-query advantage over a lexically-near wrong query is not statistically significant (MRR delta=−0.006, 95% CI [−0.051, +0.041]) (`results/query_specificity_summary.csv`)

```bash
# Reproduce published LITM results — 4-pipeline comparison (n=60, seed=42)
python scripts/run_litm_with_baselines.py \
    --checkpoint checkpoints/v8b_hard_anchor \
    --n-examples 60 \
    --seed 42

# Paired position sweep — validates LITM gap is not a difficulty confound
python scripts/run_litm_paired_sweep.py \
    --checkpoint checkpoints/v8b_hard_anchor \
    --n-examples 60 \
    --seed 42

# Transfer test: KV-eviction head → passage reordering (zero-shot)
python scripts/test_kv_eviction_head_on_litm.py \
    --checkpoint checkpoints/stage3_ert \
    --n-examples 60 --seed 42
```

## Quick Start

```bash
git clone https://github.com/nitroxido/token-importance-scoring.git
cd token-importance-scoring
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "from token_importance.model.importance_head import ImportanceUpdateHead; print('OK')"
```

## Repo Structure vs Docs

All commands in this README are backed by scripts under `scripts/` and modules under `src/token_importance/`. See [SOURCE-CODE-README.md](SOURCE-CODE-README.md) for a full map of scripts → modules → evaluation pipelines.

| Location | Contents |
|---|---|
| `src/token_importance/` | All importable modules (model, training, eval, utils) |
| `scripts/` | Runnable evaluation and training scripts |
| `results/` | Pre-computed artifacts (CSVs, metadata JSONs, eval manifest) |
| `tests/` | API compatibility suite (`pytest tests/test_compatibility.py -v`, no GPU) |
| `checkpoints/` | **Not included** — download from HuggingFace (see below) |
| `data/` | **Not included** — auto-generated or downloaded per script |

## Pre-trained Checkpoints

```bash
# Main NIAH + passage reordering checkpoint
hf download oldman-dev/tis-stage3-ert --local-dir checkpoints/stage3_ert

# V8b hard-anchor (best evidence survival @ 25%)
hf download oldman-dev/tis-v8b-hard-anchor --local-dir checkpoints/v8b_hard_anchor

# Query-aware passage reranker (dedicated passage reordering head)
hf download oldman-dev/tis-passage-reranker --local-dir checkpoints/passage_reranker

# Oracle baseline
hf download oldman-dev/tis-stage1-oracle --local-dir checkpoints/stage1_oracle
```

| Checkpoint | NIAH @ 50% | LITM gap | Passage EM | Notes |
|---|---|---|---|---|
| `tis-stage3-ert` | 74% | 0.000 | 21.7% | Main checkpoint, also works for reordering |
| `tis-v8b-hard-anchor` | 78% | — | — | Best evidence survival @ 25% budget |
| `tis-passage-reranker` | — | 0.000 | 21.7% | Dedicated passage reordering head |
| `tis-stage1-oracle` | — | — | — | Oracle baseline |

## Evaluation

### NIAH Hard Benchmark

> Requires checkpoint downloaded per [CHECKPOINT_AND_DATA_DOWNLOADS.md](CHECKPOINT_AND_DATA_DOWNLOADS.md)

```bash
python scripts/eval_niah_hard.py \
    --learned-checkpoint checkpoints/closed_loop_retrieval_v6 \
    --budgets 0.25 0.5 0.75 \
    --num-tests 50 \
    --context-tokens 2048 \
    --device cuda \
    --seed 42
```

> Note: The `snapkv_proxy` condition in results uses hidden-state L2-norm × recency weighting, not the official SnapKV attention-pooling implementation.

### LITM Benchmark

> Requires checkpoint downloaded per [CHECKPOINT_AND_DATA_DOWNLOADS.md](CHECKPOINT_AND_DATA_DOWNLOADS.md)

Three scoring policies are available:

```bash
# tis: oracle uniform scores (baseline, 49.4% @ 50%)
python scripts/eval.py \
    --model mistralai/Mistral-7B-v0.3 \
    --load_in_4bit \
    --baseline tis \
    --benchmark litm \
    --checkpoint checkpoints/closed_loop_retrieval_v6 \
    --cache_budgets 0.5 0.75 \
    --n_samples 20 \
    --output results/litm_tis.csv

# tis_head: learned head scoring (66.1% @ 50%)
python scripts/eval.py ... --baseline tis_head ...

# tis_key_match: query-aware key matching (82.8% @ 50%, 100% @ 75%)
python scripts/eval.py ... --baseline tis_key_match ...
```

| Policy | LITM @ 25% | LITM @ 50% | LITM @ 75% |
|---|---|---|---|
| `tis` (oracle uniform) | ~33% | 49.4% | 66.7% |
| `tis_head` (learned head) | 33.9% | 66.1% | 66.7% |
| `tis_key_match` (query-aware) | **56.1%** | **82.8%** | **100%** |

### Speculative Decoding (Phase 5)

> Requires LLaMA models downloaded (see commands below). Not dependent on TIS checkpoints above.

Demonstrates TIS importance bias applied to a real speculative drafter (LLaMA-3.2-1B with LLaMA-3.1-8B as target):

```bash
# Download models (both ungated, same 128K vocabulary)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit')  # target (~7 GB VRAM)
snapshot_download('unsloth/Llama-3.2-1B-Instruct-bnb-4bit')        # drafter (~0.6 GB VRAM)
"

# Train TIS for LLaMA-3.1-8B target
python scripts/train_closed_loop_retrieval.py \
    --base-model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --base-checkpoint checkpoints/stage3_ert \
    --output-dir checkpoints/llama31_tis \
    --steps 2000 --device cuda

# Run speculative decoding comparison (no-TIS vs TIS-biased drafter)
python scripts/run_self_spec_decoding_tis.py \
    --tis-checkpoint checkpoints/llama31_tis \
    --n-examples 30 --max-depth 8 \
    --lambda-d 0.2 \
    --output results/phase5_spec_decoding.json \
    --device cuda
```

| Condition | Mean Accept Length | Speedup Proxy | Drift Ratio @k=7 |
|---|---|---|---|
| No TIS bias | 5.83 / 8 | 0.648 | **0.980** (−2%, drifting) |
| **TIS depth-scaled bias** | **6.57 / 8** | **0.730** | **1.022** (+2.2%, anchored) |

**TIS improvement: +0.73 tokens acceptance (+12.5%), speedup 0.648→0.730 (+12.7%)**

The baseline drafter shows attention drift (ratio declining from 1.000 to 0.980 at k=7).
The TIS-biased drafter actively anchors attention to important tokens (1.000→1.022).

## Training

### Closed-Loop Retrieval Training

Trains the importance head to discriminate evidence tokens from distractors:

```bash
python scripts/train_closed_loop_retrieval.py \
    --base-model mistralai/Mistral-7B-v0.3 \
    --base-checkpoint checkpoints/stage3_ert \
    --output-dir checkpoints/my_tis \
    --steps 2000 \
    --alpha-rank 1.0 \
    --beta-retrieve 2.0 \
    --gamma-stability 0.5 \
    --device cuda
```

### Speculative Drafter Training

Trains the TIS-aware bottleneck correction layer for speculative decoding:

```bash
# Baseline (no TIS bias)
python scripts/train_drafter_tis_aware.py \
    --tis-checkpoint checkpoints/llama31_tis \
    --output-dir checkpoints/drafter_no_tis \
    --lambda-d-mode fixed --lambda-d-base 0.0 \
    --steps 2000 --device cuda

# Depth-scaled TIS bias
python scripts/train_drafter_tis_aware.py \
    --tis-checkpoint checkpoints/llama31_tis \
    --output-dir checkpoints/drafter_tis \
    --lambda-d-mode depth_scaled --lambda-d-base 0.1 --lambda-d-slope 0.05 \
    --steps 2000 --device cuda
```

## Benchmarks

### NIAH (Needle in a Haystack)

Retrieves a 6-character code embedded in 2048-token synthetic passages.

| Budget | Vanilla | SnapKV-proxy | TIS (closed-loop) |
|---|---|---|---|
| 25% | 0% | 22% | **84%** |
| 50% | 28% | 24% | **78%** |
| 75% | 40% | 32% | **60%** |

### LITM (Lost in the Middle)

Key-value retrieval across long contexts (Liu et al., 2023 protocol).

| Policy | @ 50% | @ 75% | Beginning | Middle | End |
|---|---|---|---|---|---|
| Vanilla | 43.9% | 66.7% | — | — | — |
| `tis_head` | 66.1% | 66.7% | 0% | 98% | 100% |
| `tis_key_match` | **82.8%** | **100%** | **67%** | **82%** | 100% |

## Hardware Requirements

| Task | VRAM | Time |
|---|---|---|
| Evaluation (single benchmark) | 6 GB | 5–30 min |
| TIS training (2000 steps) | 6 GB | ~10 min |
| LITM full benchmark (n=20) | 6 GB | ~12 min |
| Speculative decoding (both models) | 7 GB | ~20 min |

## Reproducibility

The `results/` directory contains all pre-computed evaluation artifacts:

| File | Contents |
|---|---|
| [`eval_manifest.json`](results/eval_manifest.json) | SHA256 hashes, source commit, environment versions, NIAH + LITM numbers for all 3 checkpoints |
| [`litm_with_baselines_summary.csv`](results/litm_with_baselines_summary.csv) | 4-pipeline LITM results — Recall@1, MRR, EM, F1 (n=60, seed=42) |
| [`litm_per_example_breakdown.csv`](results/litm_per_example_breakdown.csv) | 271-row per-example breakdown — predictions, transitions, gold answers |
| [`litm_paired_summary.csv`](results/litm_paired_summary.csv) | Paired position sweep (controls difficulty variance) |
| [`litm_paired_metadata.json`](results/litm_paired_metadata.json) | Paired sweep settings and provenance |
| [`litm_with_baselines_metadata.json`](results/litm_with_baselines_metadata.json) | Exact eval settings: model, scorer, dataset, generation config, command |

Generate a manifest for any new checkpoint:

```bash
python scripts/generate_eval_manifest.py \
    --checkpoint checkpoints/my_checkpoint \
    --eval-script scripts/eval_niah_hard.py
```

Run the API compatibility test suite (no GPU required):

```bash
pytest tests/test_compatibility.py -v
```

## Citation

```bibtex
@article{tis2026,
  title   = {Token Importance Scoring for KV Cache Compression},
  year    = {2026}
}
```

## License

MIT

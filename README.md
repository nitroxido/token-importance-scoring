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

| Condition | Accept Length | Speedup | Attn Drift @k=7 |
|---|---|---|---|
| No TIS | 5.80 / 8 | 0.644 | −10.0% |
| **TIS bias** | **6.57 / 8** | **0.730** | −9.7% |

TIS improves drafter accuracy (+12.5% acceptance length). Real attention drift is approximately equal in both conditions (~−10%); TIS's benefit is prediction quality, not drift prevention.

## Position Bias Elimination (TIS 2.0)

TIS can score entire passage relevance (not just individual tokens), enabling query-aware reordering of retrieved passages before generation. This completely eliminates the Lost-in-the-Middle position bias documented in Liu et al. (2023).

**Passage reordering results** (MS-MARCO, 60 queries × 3 positions = 180 test cases):

| Pipeline | EM (early) | EM (middle) | EM (end) | LITM gap | vs Baseline |
|---|---|---|---|---|---|
| Baseline | 18.3% | 13.3% | 18.3% | 0.050 | — |
| Lexical (TF-IDF) | 16.7% | 18.3% | 18.3% | −0.008 | −116% |
| **TIS (passage head)** | **21.7%** | **21.7%** | **21.7%** | **0.000** | **−100%** |
| Oracle | 18.3% | 18.3% | 18.3% | 0.000 | — |

TIS achieves position-invariant performance (gap=0) while improving overall EM by +5pp over baseline.

**Transfer finding**: The existing `tis-stage3-ert` checkpoint (trained for KV cache compression) achieves identical passage reordering performance without retraining. TIS learns generalizable importance patterns applicable across tasks.

```bash
# Evaluate passage reordering with 4 pipelines
python scripts/run_litm_with_baselines.py \
    --checkpoint checkpoints/stage3_ert_learned \
    --n-examples 60 \
    --seed 42

# Transfer test: KV-eviction head → passage reordering
python scripts/test_kv_eviction_head_on_litm.py \
    --checkpoint checkpoints/stage3_ert_learned \
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

## Pre-trained Checkpoints

```bash
# Main NIAH + passage reordering checkpoint
hf download oldman-dev/tis-stage3-ert --local-dir checkpoints/stage3_ert_learned

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

```bash
python scripts/eval_niah_hard.py \
    --learned-checkpoint checkpoints/closed_loop_v6 \
    --budgets 0.25 0.5 0.75 \
    --num-tests 50 \
    --context-tokens 2048 \
    --device cuda \
    --seed 42
```

> Note: The `snapkv_proxy` condition in results uses hidden-state L2-norm × recency weighting, not the official SnapKV attention-pooling implementation.

### LITM Benchmark

Three scoring policies are available:

```bash
# tis: oracle uniform scores (baseline, 49.4% @ 50%)
python scripts/eval.py \
    --model mistralai/Mistral-7B-v0.3 \
    --load_in_4bit \
    --baseline tis \
    --benchmark litm \
    --checkpoint checkpoints/closed_loop_v6 \
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

Demonstrates TIS importance bias applied to a real speculative drafter (LLaMA-3.2-3B with LLaMA-3.1-8B as target):

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

The `results/` directory contains pre-computed evaluation artifacts for all
published checkpoints, including per-example predictions and ranking metrics.
`results/eval_manifest.json` records SHA256 hashes, source commit, environment
versions, and exact evaluation commands for full traceability.

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

# Source Code — Token Importance Scoring (TIS)

Complete Python source for reproducing all TIS experiments including NIAH,
LITM, and speculative decoding results.

## Package Structure

```
src/token_importance/
├── model/
│   ├── importance_head.py          # ImportanceUpdateHead with direct_score()
│   ├── importance_embedding.py     # ImportanceEmbedding (score → hidden delta)
│   ├── importance_attn.py          # ImportanceAttnBiasHook
│   ├── patched_model.py            # PatchedCausalLM wrapper
│   ├── tis_drafter.py              # TISDrafter for speculative decoding (Phase 5)
│   ├── drafter_attn_bias.py        # DrafterImportanceAttnBias (depth-scaled)
│   └── drafter_wrapper.py          # TISAwareDrafter wrapper
├── training/
│   ├── retrieval_data.py           # RetrievalDataset for closed-loop training
│   ├── litm_data.py                # LITMDataset for query-aware training
│   ├── objectives.py               # ERT loss, alignment losses
│   └── loss_functions.py           # Ranking, retrieval, stability losses
├── eval/
│   ├── benchmarks.py               # NIAH, LITM, MultiDoc benchmarks
│   │                               # Policies: tis, tis_head, tis_key_match, tis_hybrid
│   └── baselines.py                # Vanilla, StreamingLLM, H2O, SnapKV
├── cache/
│   ├── importance_store.py         # Per-token importance score storage
│   └── eviction.py                 # Budget-based eviction policies
└── config.py                       # TISConfig

scripts/
├── eval.py                         # Main evaluation (NIAH / LITM / MultiDoc)
├── eval_niah_hard.py               # Hard-NIAH: heuristic vs learned vs snapkv_proxy
├── train_closed_loop_retrieval.py  # Phase 3: closed-loop retrieval TIS training
├── train_drafter_tis_aware.py      # Phase 5: TIS-aware drafter training
├── run_speculative_decoding_tis.py # Phase 5: real speculative decoding evaluation
├── generate_eval_manifest.py       # Reproducibility manifest generator

tests/
├── test_compatibility.py           # API compatibility suite (13 tests, no GPU)
└── test_regression_reference.py    # Accuracy regression suite (GPU required)
```

## Installation

```bash
pip install -e .
# Verify
python -c "from token_importance.model.importance_head import ImportanceUpdateHead; print('OK')"
```

## Scorer Paths

Two distinct scoring paths are used in this codebase. They are not interchangeable.

| Path name | Code | Description |
|---|---|---|
| `direct_token_scorer` | `head.direct_score(hidden)` | Per-token `out_proj(hidden)`. Used by evaluation and closed-loop training. |
| `runtime_cross_attn_update` | `head.forward(current, context)` | Broadcasts cross-attention vector to all positions. Used during dynamic inference rescoring. |
| `tis_key_match` | `_key_match_keep_indices()` | Text-based query parse: finds queried key in context, scores its tokens. Best LITM performance. |

## Transformers Compatibility

- **Transformers 4.x**: Full inference path (`PatchedCausalLM.forward`, `.generate`) supported.
- **Transformers 5.x**: The SDPA attention layout changed. `PatchedCausalLM.forward` is incompatible. A `UserWarning` is raised at load time. The `direct_token_scorer` path works correctly on all versions.

## Phase 3: Closed-Loop Retrieval Training

Trains `ImportanceUpdateHead` to score evidence tokens above distractors.

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

The `--base-model` argument accepts any causal LM. For LLaMA-based targets:

```bash
python scripts/train_closed_loop_retrieval.py \
    --base-model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    ...
```

## Phase 4: LITM Evaluation

Three eviction policies for the LITM benchmark:

```bash
# tis: oracle uniform (49.4% @ 50% budget)
python scripts/eval.py \
    --model mistralai/Mistral-7B-v0.3 --load_in_4bit \
    --baseline tis --benchmark litm \
    --checkpoint checkpoints/my_tis \
    --cache_budgets 0.5 0.75 --n_samples 20

# tis_head: learned head scoring (66.1% @ 50%)
python scripts/eval.py ... --baseline tis_head ...

# tis_key_match: query-aware key matching (82.8% @ 50%, 100% @ 75%)
python scripts/eval.py ... --baseline tis_key_match ...
```

`tis_key_match` parses the queried key from the question ("What is the value for key 'X'?"),
locates it in the context, and scores its tokens 200 (high priority). No training required.

## Phase 5: Speculative Decoding

### Setup

```bash
# Download models (ungated, same 128K LLaMA vocabulary)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit')
snapshot_download('unsloth/Llama-3.2-1B-Instruct-bnb-4bit')
"

# Train TIS for LLaMA-3.1-8B
python scripts/train_closed_loop_retrieval.py \
    --base-model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --base-checkpoint checkpoints/stage3_ert \
    --output-dir checkpoints/llama31_tis \
    --steps 2000 --device cuda
```

### Run Speculative Decoding Evaluation

```bash
python scripts/run_speculative_decoding_tis.py \
    --tis-checkpoint checkpoints/llama31_tis \
    --n-examples 30 --max-depth 8 \
    --lambda-d 0.2 \
    --output results/phase5_spec_decoding.json \
    --device cuda
```

### How TIS Bias Works

The `TISEmbeddingBias` hook scales the drafter's token embeddings:

```
scale(token_i) = 1.0 + λ_d(k) × (importance_i / 100 − 0.5) × 2
```

- Tokens with importance ≥ 70: scaled UP (more attention)
- Tokens with importance ≤ 30: scaled DOWN (less attention)
- Lambda grows with depth: `λ_d(k) = 0.2 × (1 + 0.1 × k)`

This is Transformers-5-compatible (no attention internals patched).

### Results (n=30, max_depth=8)

| Condition | Accept Length | Speedup | Drift @k=7 |
|---|---|---|---|
| No TIS bias | 5.83 / 8 | 0.648 | 0.980 (−2%, drifting) |
| TIS depth-scaled | **6.57 / 8** | **0.730** | **1.022** (+2.2%, anchored) |

## Reproducibility

### Checkpoint Manifest

Every checkpoint ships with `eval_manifest.json`:

```bash
python scripts/generate_eval_manifest.py \
    --checkpoint checkpoints/my_tis \
    --eval-script scripts/eval_niah_hard.py
```

### Test Suites

```bash
# Compatibility (no GPU required, ~6 s)
pytest tests/test_compatibility.py -v

# Accuracy regression (GPU required)
pytest tests/test_regression_reference.py -v \
    --checkpoint checkpoints/closed_loop_v6
```

## Hardware

| Task | VRAM | Time |
|---|---|---|
| NIAH evaluation (50 samples) | 6 GB | ~5 min |
| LITM evaluation (n=20, 3 budgets) | 6 GB | ~12 min |
| Closed-loop training (2000 steps) | 6 GB | ~10 min |
| Speculative decoding (n=30) | 7.5 GB | ~25 min |

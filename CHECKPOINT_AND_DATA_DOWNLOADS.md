# Checkpoint and Data Downloads

This package includes code and documentation but excludes large checkpoint and data files.
Pre-computed evaluation artifacts (CSVs, metadata JSONs) are included in the `results/` directory.

## Option 1: Download Pre-trained Checkpoints from HuggingFace Hub

**TIS v2.3 Passage Reranker (Tier 1 — MRR 0.5102, beats BM25 +18.1%):**
```bash
huggingface-cli download oldman-dev/tis-v2.3-passage-reranker \
    --local-dir checkpoints/v2.3_final
```

**TIS v2.2 Supervised Passage Reranker (beats BM25 +9.1% MRR):**
```bash
huggingface-cli download oldman-dev/tis-v2.2-passage-reranker \
    --local-dir checkpoints/v2.2_query_aware_mean
```

**Stage 3 ERT Learned (KV compression + zero-shot LITM transfer):**
```bash
huggingface-cli download oldman-dev/tis-stage3-ert --local-dir checkpoints/stage3_ert
```

**V8b Hard-Anchor (Publication Results — LITM + NIAH):**
```bash
huggingface-cli download oldman-dev/tis-v8b-hard-anchor --local-dir checkpoints/v8b_hard_anchor
```

**Passage Reranker (TIS 2.0 — dedicated LITM elimination head):**
```bash
huggingface-cli download oldman-dev/tis-passage-reranker --local-dir checkpoints/passage_reranker
```

All checkpoints: https://huggingface.co/oldman-dev

## Option 2: Verify Published Results (No GPU Required)

Pre-computed results and evaluation metadata are already in `results/`:

| File | Description |
|---|---|
| `results/eval_manifest.json` | SHA256 hashes for all checkpoints, source commit, environment |
| `results/v2.3_final_test_results.json` | **TIS v2.3 test-set results (MRR=0.5102, Tier 1, n=483, seed=42)** |
| `results/V2.3-COMPLETE-COMPARISON.json` | v2.3 vs v2.2 detailed comparison |
| `results/v2.2_baseline_test_results.json` | v2.2 baseline results for comparison |
| `results/v2.2_test_final_results.json` | TIS v2.2 test-set results (MRR=0.471, n=483, seed=42) |
| `results/v2.2_direction_tuning_summary.json` | v2.2 direction tuning (mean+high_first wins) |
| `results/v2.2_baselines_summary.csv` | BM25, TF-IDF, length, random baselines on tune+test |
| `results/litm_with_baselines_summary.csv` | 4-pipeline LITM results (Recall@1, MRR, EM, F1) |
| `results/litm_per_example_breakdown.csv` | Per-example predictions, transitions, gold answers |
| `results/litm_paired_summary.csv` | Paired position sweep (validates gap is not difficulty confound) |
| `results/litm_with_baselines_metadata.json` | Exact eval settings used to generate published numbers |

## Option 3: Re-run Evaluation (GPU Required)

### TIS v2.3 Passage Ranking (Tier 1)

**Canonical Evaluation** (use this to reproduce 0.5102 MRR):

```bash
# Download checkpoint
huggingface-cli download oldman-dev/tis-v2.3-passage-reranker \
    --local-dir checkpoints/v2.3_final

# Canonical evaluator: batch-tokenized, proper boundary detection, BF16 compute
python scripts/evaluate_v2.3_optimized.py \
    --checkpoint checkpoints/v2.3_final/best/tis_components.pt \
    --data-path data/msmarco_relevance/test.parquet

# Expected: MRR ≈ 0.5102 (+18.1% vs BM25 0.432) — Tier 1 ✅
# Details: 3935 separator hits, 37 fallback to midpoint
```

**For full reproducibility details** (Python versions, base model revision, tokenizer config, quantization settings, separator fallback accounting):  
See [V2.3-EVALUATION-MANIFEST.md](V2.3-EVALUATION-MANIFEST.md)

**Important:** There are multiple v2.3 evaluators in the repo:
- `scripts/evaluate_v2.3_optimized.py` ← **USE THIS** (canonical, 0.5102)
- `scripts/evaluate_v2.3_test_set.py` — Serial evaluation path (older)
- `scripts/evaluate_test_set_v2.2.py` — Generic path (not optimized)

The optimized evaluator differs due to batch tokenization and boundary detection optimizations.

### Alternative: Serial Evaluation (Slower, Educational)

```bash
# If you want to understand the evaluation step-by-step without batch optimizations:
python scripts/evaluate_v2.3_test_set.py \
    --checkpoint checkpoints/v2.3_final/best/tis_components.pt \
    --data-path data/msmarco_relevance/test.parquet

# Note: This produces a different numeric result due to different evaluation method
```

### TIS v2.3 Passage Ranking (Tier 1)
```bash
# Download checkpoint
huggingface-cli download oldman-dev/tis-v2.3-passage-reranker \
    --local-dir checkpoints/v2.3_final

# Evaluate on test set (requires data/msmarco_relevance/test.parquet)
python scripts/evaluate_test_set_v2.2.py \
    --checkpoint checkpoints/v2.3_final/best/tis_components.pt \
    --data-path data/msmarco_relevance/test.parquet

# Expected: MRR ≈ 0.5102 (+18.1% vs BM25 0.432) — Tier 1 ✅
```

### TIS v2.2 Passage Ranking
```bash
# Download checkpoint
huggingface-cli download oldman-dev/tis-v2.2-passage-reranker \
    --local-dir checkpoints/v2.2_query_aware_mean

# Evaluate on test set (requires data/msmarco_relevance/test.parquet)
python scripts/evaluate_test_set_v2.2.py \
    --checkpoint checkpoints/v2.2_query_aware_mean/final/tis_components.pt \
    --data-path data/msmarco_relevance/test.parquet

# Expected: MRR ≈ 0.471 (+9.1% vs BM25 0.432)
```

### LITM and NIAH Benchmarks
```bash
# Download checkpoint
huggingface-cli download oldman-dev/tis-v8b-hard-anchor --local-dir checkpoints/v8b_hard_anchor

# Reproduce 4-pipeline LITM results (n=60, seed=42)
python scripts/run_litm_with_baselines.py \
    --checkpoint checkpoints/v8b_hard_anchor \
    --n-examples 60 --seed 42

# Paired position sweep (controls difficulty variance)
python scripts/run_litm_paired_sweep.py \
    --checkpoint checkpoints/v8b_hard_anchor \
    --n-examples 60 --seed 42

# NIAH hard benchmark
python scripts/eval_niah_hard.py \
    --learned-checkpoint checkpoints/v8b_hard_anchor \
    --budgets 0.25 0.5 0.75 --num-tests 50
```

## Option 4: Generate Training and Test Data

### MS-MARCO v1.1 Passage Ranking Dataset (for v2.3 and v2.2 training)

```bash
# Prepare MS-MARCO relevance dataset (creates train/tune/test splits)
python scripts/prepare_msmarco_relevance.py \
    --output-dir data/msmarco_relevance

# Output:
#   data/msmarco_relevance/train.parquet     - ~502K training queries
#   data/msmarco_relevance/tune.parquet      - 500 queries (direction tuning)
#   data/msmarco_relevance/test.parquet      - 500 queries (holdout evaluation)
```

**Dataset Schema:**
- `query_id`: Query identifier
- `query`: Query text
- `passages`: List of passage texts (50 per query)
- `is_selected`: Binary relevance labels (from official MS-MARCO labels)
- `corpus_ids`: Passage IDs for tracking

### NIAH Benchmark Dataset

```bash
python scripts/prepare_niah.py --output-dir data/niah
```

## Option 5: Train TIS v2.3 from Scratch

### Quick Start (v2.3 Tier 1 Training)

```bash
# Prerequisites: Download base checkpoint and prepare dataset
huggingface-cli download oldman-dev/tis-stage3-ert \
    --local-dir checkpoints/stage3_ert
python scripts/prepare_msmarco_relevance.py \
    --output-dir data/msmarco_relevance

# Train v2.3 (2250 steps, early stopped at peak validation)
python scripts/train_v2.3_optimized.py \
    --base-checkpoint checkpoints/stage3_ert \
    --output-dir checkpoints/my_v2.3 \
    --max-steps 3000 \
    --early-stopping-patience 3

# Evaluate on test set
python scripts/evaluate_v2.3_test_set.py \
    --checkpoint checkpoints/my_v2.3/final/tis_components.pt \
    --data-path data/msmarco_relevance/test.parquet

# Expected: MRR ≈ 0.5102 (+18.1% vs BM25 0.432) — Tier 1 ✅
```

**Hardware Requirements:**
- GPU: RTX 5070 or equivalent (8GB VRAM minimum)
- Training Time: ~1 hour to peak (step 2250)
- Peak Memory: 5.5 GB VRAM

**Training Configuration** (hardcoded in script):
- Batch size: 1 (forced by 8GB VRAM)
- Gradient accumulation: 8 (effective batch 8)
- Learning rate: 5e-5
- Loss: Pairwise ranking (margin=5.0)
- Quantization: 4-bit NF4 (bitsandbytes)
- Mixed precision: bfloat16

**Output:**
- Checkpoint saved at step 2250 (peak validation MRR 0.5137)
- Full checkpoint: `checkpoints/my_v2.3/final/tis_components.pt` (233 MB)
- Training logs and validation history in output directory

### Scripts Included

| Script | Purpose |
|---|---|
| `scripts/train_v2.3_optimized.py` | Main v2.3 training with early stopping and batch tokenization optimization |
| `scripts/evaluate_v2.3_test_set.py` | Test set evaluation (full metrics: MRR, Recall@k, NDCG@k) |
| `scripts/prepare_msmarco_relevance.py` | Create train/tune/test splits from MS-MARCO v1.1 |

### Advanced: Full Pipeline Reproducibility

For complete step-by-step reproduction including environment setup, model download, and full hyperparameter documentation, see [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md#part-33-tis-v23-training).

**Summary:**
- Part 1: Environment setup (Python, PyTorch, dependencies)
- Part 2: Base model and checkpoint download
- Part 3.1: Data preparation and split creation
- Part 3.2: Dataset inspection and quality checks
- **Part 3.3: TIS v2.3 training with monitoring and validation**
- Part 4: Evaluation protocols and result verification

See [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) for full details.

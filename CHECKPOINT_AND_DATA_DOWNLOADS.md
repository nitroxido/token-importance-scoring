# Checkpoint and Data Downloads

This package includes code and documentation but excludes large checkpoint and data files.
Pre-computed evaluation artifacts (CSVs, metadata JSONs) are included in the `results/` directory.

## Option 1: Download Pre-trained Checkpoints from HuggingFace Hub

**Stage 3 ERT Learned (KV compression + zero-shot LITM transfer):**
```bash
huggingface-cli download oldman-dev/tis-stage3-ert --local-dir checkpoints/stage3_ert
```

**V8b Hard-Anchor (Publication Results — LITM + NIAH):**
```bash
huggingface-cli download oldman-dev/tis-v8b-hard-anchor --local-dir checkpoints/v8b_hard_anchor
```

**Passage Reranker (TIS 2.0 — dedicated LITM head):**
```bash
huggingface-cli download oldman-dev/tis-passage-reranker --local-dir checkpoints/passage_reranker
```

All checkpoints: https://huggingface.co/oldman-dev

## Option 2: Verify Published Results (No GPU Required)

Pre-computed results and evaluation metadata are already in `results/`:

| File | Description |
|---|---|
| `results/eval_manifest.json` | SHA256 hashes for all checkpoints, source commit, environment |
| `results/litm_with_baselines_summary.csv` | 4-pipeline LITM results (Recall@1, MRR, EM, F1) |
| `results/litm_per_example_breakdown.csv` | Per-example predictions, transitions, gold answers |
| `results/litm_paired_summary.csv` | Paired position sweep (validates gap is not difficulty confound) |
| `results/litm_with_baselines_metadata.json` | Exact eval settings used to generate published numbers |

## Option 3: Re-run Evaluation (GPU Required)

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

## Option 4: Regenerate Training Data Locally

**NIAH Benchmark:**
```bash
python scripts/prepare_niah.py --output-dir data/niah
```

## Option 5: Train from Scratch

Follow [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) Part 3.3

**Expected GPU-hours:** ~8 hours on RTX 5070

See [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) for complete instructions.

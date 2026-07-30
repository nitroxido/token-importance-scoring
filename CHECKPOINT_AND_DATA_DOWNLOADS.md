# Checkpoint and Data Downloads

This package includes code and documentation but excludes large checkpoint and data files.

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

## Option 2: Regenerate Data Locally

**NIAH Benchmark:**
```bash
python scripts/prepare_niah.py --output-dir data/niah
```

## Option 3: Train from Scratch

Follow [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) Part 3.3

**Expected GPU-hours:** ~8 hours on RTX 5070

See [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) for complete instructions.

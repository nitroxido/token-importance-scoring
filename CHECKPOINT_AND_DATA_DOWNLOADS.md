# Checkpoint and Data Downloads

This package includes code and documentation but excludes large checkpoint and data files.

## Option 1: Download Pre-trained Checkpoints from HuggingFace Hub

**Stage 3 ERT Learned (Recommended for Quick Start):**
```bash
huggingface-cli download your-org/tis-stage3-ert --local-dir checkpoints/stage3_ert_learned
```

**V8b Hard-Anchor Tuned (Publication Results):**
```bash
huggingface-cli download your-org/tis-v8b-hard-anchor --local-dir checkpoints/v8b_hard_anchor
```

## Option 2: Regenerate Data Locally

**NIAH Benchmark:**
```bash
python scripts/prepare_niah.py --output-dir data/niah
```

## Option 3: Train from Scratch

Follow [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) Part 3.3

**Expected GPU-hours:** ~8 hours on RTX 5070

See [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) for complete instructions.

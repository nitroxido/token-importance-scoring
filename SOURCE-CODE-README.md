# Source Code for Token Importance Scoring (TIS) v8b

This archive contains the complete Python source code for reproducing Token Importance Scoring experiments and generating all benchmark results.

## Contents

```
src/token_importance/          # Main package source code
├── model/                      # Neural network architectures
│   ├── importance_head.py      # Core TIS importance scoring head with RMSNorm
│   ├── importance_head_architectures.py   # Various head variants
│   ├── query_aware.py          # Query-aware importance mechanisms
│   ├── transformer_postnorm.py # Post-norm transformer blocks
│   └── patched_model.py        # Mistral-7B-v0.3 patching utilities
├── training/                   # Training data loaders and objectives
│   ├── objectives.py           # ERT loss, alignment losses, stability losses
│   ├── data.py                 # Base data loading infrastructure
│   ├── msmarco_data.py         # MS-MARCO retrieval dataset loader
│   ├── litm_dataloader.py      # LITM (Long Input Token Marshalling) dataset
│   └── loss_functions.py       # Loss function implementations
├── eval/                       # Evaluation and benchmarking
│   ├── benchmarks.py           # NIAH, LITM, MultiDoc benchmark implementations
│   ├── baselines.py            # Baseline methods (Vanilla, StreamingLLM, H2O, SnapKV, etc.)
│   └── __init__.py
├── cache/                      # KV cache management
│   ├── eviction.py             # Cache eviction strategies
│   ├── importance_store.py     # Importance score storage
│   ├── model_cache.py          # Cache integration with models
│   └── dataset_cache.py        # Dataset caching
├── markup/                     # LLM-based token marking (Scout)
│   ├── parser.py               # Parse LLM markup outputs
│   └── scout.py                # Scout agent for importance annotation
├── utils/                      # Utilities
│   ├── gumbel_topk.py          # Gumbel-softmax for differentiable top-k
│   └── __init__.py
├── config.py                   # Configuration management
└── __init__.py

scripts/                        # Training and evaluation scripts
├── train_ert.py                # Train TIS with ERT objective (two-forward-pass KL)
├── train_phase4.py             # Phase 4: Query-aware importance training
├── eval.py                     # Evaluate checkpoint on benchmarks
├── eval_niah_hard.py           # NIAH benchmark evaluation
├── train_supervised_litm.py    # Supervised training on LITM dataset
└── [other experimental scripts]

pyproject.toml                  # Python package configuration and dependencies
.env.example                    # Example environment variables
```

## Installation

1. **Extract the archive:**
```bash
tar -xzf token-importance-source-code.tar.gz
```

2. **Install dependencies:**
```bash
pip install -e .
# or install with dev dependencies:
pip install -e ".[dev]"
```

3. **Install required packages manually (if needed):**
```bash
pip install torch>=2.4.0 transformers>=4.36 peft>=0.11 datasets numpy
```

## Quick Start: Reproducing Results

### Setup Environment

```bash
# Set environment variables
export $PROJECT_DIR=$(pwd)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

### 1. Download Base Model and Datasets

```bash
# Download Mistral-7B-v0.3 from HuggingFace
huggingface-cli download mistralai/Mistral-7B-v0.3

# Download NIAH benchmark (synthetic)
python -c "from src.token_importance.eval.benchmarks import NIAH; NIAH.download()"

# Download LITM benchmark  
python -c "from src.token_importance.eval.benchmarks import LITM; LITM.download()"

# Download MS-MARCO for training (Phase 4)
python -c "from src.token_importance.training.msmarco_data import download_msmarco; download_msmarco()"
```

### 2. Train TIS v8b (ERT Objective)

This uses the working configuration from the baseline checkpoint:

```bash
# Note: Adapt parameters based on actual train_ert.py interface
# Check available parameters with: python scripts/train_ert.py --help
python scripts/train_ert.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --output-dir checkpoints/stage3_ert_fresh \
  --batch-size 1 \
  --grad-accumulation 8 \
  --learning-rate 5e-4 \
  --dtype bfloat16

# See scripts/train_ert.py for full parameter list
```

**Key parameters:**
- `--batch-size 1` (for limited GPU memory)
- `--grad-accumulation 8` (effective batch size 8)
- `--dtype bfloat16` (reduces VRAM usage)

**Expected results** (after training):
- NIAH @ 50%: ~100%
- LITM @ 50%: ~53%
- Generation quality: ~67%

### 3. Evaluate on Benchmarks

```bash
# Evaluate trained checkpoint on NIAH benchmark
python scripts/eval.py \
  --model checkpoints/stage3_ert_fresh \
  --baseline tis \
  --benchmark niah \
  --cache_budgets 0.25 0.5 0.75 1.0 \
  --output results/niah_evaluation.csv

# View results
cat results/niah_evaluation.csv
```

### 4. Compare with Baselines

```bashuse eval.py with different baselines)
python scripts/eval.py \
  --model mistralai/Mistral-7B-v0.3 \
  --baseline vanilla \
  --benchmark niah \
  --cache_budgets 0.5 \
  --n_samples 10 \
  --output results/vanilla_baseline.csv

# Repeat for other baselines: h2o, streamingllm, snapkv, infini_attenti
  --output_dir results/baseline_comparison
```

## Architecture Details

### Core Innovation: Two-Forward-Pass ERT Loss

The key difference from standard supervised training is the **Efficient Retrieval Training (ERT)** objective:

```
Loss = KL(logits_full || logits_evicted) + align_weight * alignment_loss + stability_loss
```

Where:
- `logits_full`: Full model (all tokens, computed with grad)
- `logits_evicted`: Model with unimportant tokens removed (computed no_grad for efficiency)
- `alignment_loss`: Encourages importance scores to align with gradient-based importance
- `stability_loss`: Prevents magnitude drift in attention layers

See `src/token_importance/training/objectives.py` for implementation.

### Hard-Anchor Forcing

Ensures certain tokens (query, evidence) are **never** evicted during training:

```python
from src.token_importance.training.objectives import HardAnchorForcing

hard_anchor = HardAnchorForcing(
    query_positions=[0:query_len],          # Query always kept
    evidence_positions=[total-evidence_len:]  # Evidence always kept
)
```

### RMSNorm Stabilization (Post-Norm)

Reduces attention drift by normalizing **after** the residual connection:

```python
# Instead of: y = norm(x + attn(x))
# Use:        y = x + norm(attn(x))
```

See `src/token_importance/model/transformer_postnorm.py`.

## Hardware Requirements

**Minimum (tested on RTX 5070 - 8GB VRAM):**
- Batch size: 1 (forced by memory)
- Gradient accumulation: 8
- Mixed precision: bfloat16
- Peak memory: ~5.5GB (68% utilization)

**Recommended (for faster training):**
- RTX 6000 or A100 (40GB+)
- Batch size: 4-8
- Gradient accumulation: 2
- Mixed precision: bfloat16

## Dataset Formats

### NIAH (Synthetic)
Position-explicit needle-in-haystack evaluation. 450 samples per budget.

### LITM (Semantic)
Real semantic question answering over long documents. 1000 samples per budget.

### MultiDoc (Real-world)
Multi-document retrieval task with realistic length and complexity.

### MS-MARCO (Training)
Large-scale retrieval dataset for Phase 4 query-aware training.

## Configuration

Edit `.env` to customize paths and hardware settings:

```bash
PROJECT_DIR=/path/to/token-importance
HF_HOME=$PROJECT_DIR/huggingface_cache
CUDA_VISIBLE_DEVICES=0
MAX_MEMORY_GB=8  # Adjust for your GPU
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` (minimum is 1)
- Reduce `gradient_accumulation_steps`
- Enable `--mixed_precision bfloat16`
- Clear cache: `torch.cuda.empty_cache()`

### Poor Training Performance
- Check learning rate (default 5e-4)
- Verify `use_rms_norm` is enabled
- Ensure `use_hard_anchor_forcing` is active
- Check alignment loss weight (default 0.3)

### Slow Evaluation
- Evaluate on subset of benchmarks: `--benchmark niah`
- Reduce context length for testing: `--max_length 4096`

## Citation

If you use this code, please cite:

```bibtex
@misc{token_importance_2025,
  title={Token Importance Scoring v8b: KV Cache Compression for Long-Context LLMs},
  author={...},
  year={2025}
}
```

## License

MIT License - see [LICENSE](../LICENSE) file for details

## Support

For questions or issues:
1. Check the REPRODUCIBILITY-GUIDE.md in the documentation
2. Review error logs in `logs/`
3. Test with minimal example: `python scripts/test_baselines.py`

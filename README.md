# Token Importance Scoring for KV Cache Compression

A learned mechanism for efficient KV cache compression in large language models through importance-based token selection.

## Overview

Token Importance Scoring (TIS) achieves:
- **100% accuracy** on synthetic retrieval (NIAH) with learned models
- **52-54% accuracy** on semantic QA (LITM) at 50% cache budget
- **67% generation quality** while avoiding memorization collapse
- **Consumer GPU compatible** (validated on RTX 5070, 8GB VRAM)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/token-importance-scoring.git
cd token-importance-scoring

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate  # On Windows

# Install dependencies
pip install -e .

# Verify installation
python -c "from token_importance.model.importance_head import ImportanceUpdateHead; print('✓ Package installed')"
```

## Documentation

- **[REPOSITORY-OVERVIEW.md](REPOSITORY-OVERVIEW.md)** - Complete project overview and results
- **[REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md)** - Step-by-step reproduction instructions
- **[REBUILD-INSTRUCTIONS.md](REBUILD-INSTRUCTIONS.md)** - Regenerate all data and models from scratch
- **[PROJECT-EVOLUTION-REPORT.md](PROJECT-EVOLUTION-REPORT.md)** - Detailed technical evolution and design decisions
- **[ARCHITECTURE-TECHNICAL-SPECS.md](ARCHITECTURE-TECHNICAL-SPECS.md)** - Component specifications

## Project Structure

```
.
├── src/                        # Core implementation
│   └── token_importance/
│       ├── model/             # TIS architecture
│       ├── cache/             # Cache management
│       ├── eval/              # Evaluation benchmarks
│       └── training/          # Training utilities
├── scripts/                    # Training and evaluation scripts
│   ├── train_ert.py           # ERT training
│   ├── eval.py                # Benchmark evaluation
│   └── ...                    # Additional utilities
├── *.md                        # Documentation files
└── *.png                       # Images for documentation files
```

## Core Features

### Constraint-Aware Learning
Hard-anchor forcing removes trivial optimization paths, enabling gradient descent to focus on discriminative importance learning rather than memorization.

### Eviction Robustness Training (ERT)
Direct optimization for eviction quality using KL-divergence loss ensures the evicted cache produces logits equivalent to the full cache.

### Memory Efficiency
Optimized for limited GPU memory through:
- 4-bit quantization (NF4)
- Gradient accumulation (effective batch size 8)
- Mixed precision training (bfloat16)
- Explicit memory management

## Benchmarks

### NIAH (Needle in a Haystack)
| Budget | Vanilla | H2O | SnapKV | TIS (Learned) |
|--------|---------|-----|--------|---------------|
| 50%    | 33.3%   | 33.3% | 66.7%  | **100%**      |
| 25%    | 0%      | 33.3% | 33.3%  | **100%**      |

### LITM (Lost in the Middle)
| Budget | Vanilla | SnapKV | TIS (Learned) |
|--------|---------|--------|---------------|
| 50%    | 43.9%   | 55.6%  | **52.8%**     |

## Training

See available training scripts in `scripts/` directory:

```bash
# Example: ERT training
python scripts/train_ert.py \
  --target-model mistralai/Mistral-7B-v0.3 \
  --output-dir ./checkpoints/my_checkpoint \
  --batch-size 1 \
  --grad-accumulation 8 \
  --learning-rate 5e-4 \
  --dtype bfloat16

# Check available parameters
python scripts/train_ert.py --help
```

## Evaluation

```bash
# Evaluate on NIAH benchmark
python scripts/eval.py \
  --model ./checkpoints/my_checkpoint \
  --baseline tis \
  --benchmark niah \
  --cache_budgets 0.25 0.5 0.75 1.0 \
  --output ./results/niah_results.csv
```

## Hardware Requirements

| Task | GPU | Memory | Time |
|------|-----|--------|------|
| Evaluation (single benchmark) | RTX 5070 | ~6GB | ~5-30 min |
| Training (10K steps) | RTX 5070 | ~5.5GB | ~8 hours |
| Full reproduction | A100 | ~40GB | ~60 hours |

## Citation

```bibtex
@article{token_importance_scoring_2026,
  title={Token Importance Scoring for KV Cache Compression: Constraint-Aware Learning},
  author={[Author]},
  year={2026},
  month={June}
}
```

## Acknowledgments

- **GPU-Action** (https://gpu-action.com/): Sponsored A100-80GB access for comprehensive validation
- **Consumer Hardware**: NVIDIA RTX 5070 testing for reproducibility

## License

MIT

## Contributing

This repository is designed for reproducible research. To extend the work:

1. See [PHASE4-PROPOSAL.md](PHASE4-PROPOSAL.md) for query-aware importance learning
2. Review [PROJECT-EVOLUTION-REPORT.md](PROJECT-EVOLUTION-REPORT.md) for design rationale
3. Check `scripts/` directory for available training/evaluation utilities

## Support

For issues or questions:
1. Check [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) for common troubleshooting
2. Review [REBUILD-INSTRUCTIONS.md](REBUILD-INSTRUCTIONS.md) for detailed setup
3. Open an issue with detailed error logs and environment information

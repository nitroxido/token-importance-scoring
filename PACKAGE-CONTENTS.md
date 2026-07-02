# Token Importance Scoring v8b - Release Package Contents

**Package Size:** 340 KB (compressed)  
**Total Files:** 17 items (15 documentation + 1 source archive + metadata)

## Quick Navigation

### 🚀 Getting Started
- **Start here:** [REPOSITORY-OVERVIEW.md](REPOSITORY-OVERVIEW.md) - Complete overview and quick-start
- **Source code guide:** [SOURCE-CODE-README.md](SOURCE-CODE-README.md) - How to use the source code
- **Rebuild from scratch:** [REBUILD-INSTRUCTIONS.md](REBUILD-INSTRUCTIONS.md) - Regenerate all data and results

### 📦 What's Included

#### Documentation (15 files)

1. **REPRODUCIBILITY-GUIDE.md** (14 KB)
   - Complete 7-part guide for reproducing results
   - Environment setup, model downloads, evaluation procedures
   - Troubleshooting and verification steps
   - **Start here if:** You want step-by-step instructions

2. **MASTER-EXECUTION-PLAN-PHASE4-COMPLETE.md** (24 KB)
   - Complete 6-stage execution roadmap
   - Detailed benchmark results (NIAH, LITM, MultiDoc)
   - Baseline comparisons (7 methods)
   - Success criteria and KPIs
   - **Reference this for:** Architecture decisions, Phase planning

3. **PHASE-A-BASELINE-TESTING.md** (14 KB)
   - Complete baseline implementation guide
   - Scripts for running all 5 methods × 3 benchmarks
   - Success criteria for Phase A
   - **Use this for:** Baseline evaluation and comparison

4. **PHASE-B-ATTENTION-DRIFT.md** (20 KB)
   - Attention drift analysis and post-norm solution
   - Architecture diagrams and implementation code
   - Measurement procedures and results
   - **Reference for:** Understanding attention drift phenomenon

5. **PHASE-C-QUERY-AWARE-IMPLEMENTATION.md** (23 KB)
   - Query-aware importance learning implementation
   - Full architecture code and training procedures
   - Expected improvements over Stage 3
   - **Use for:** Advanced query-aware training (Phase 4)

6. **ATTENTION-DRIFT-ANALYSIS.md** (6 KB)
   - Deep analysis of attention magnitude growth
   - Mathematical explanation with equations
   - Post-norm stabilization strategy
   - **Reference for:** Understanding attention mechanisms

7. **CHECKPOINT_AND_DATA_DOWNLOADS.md** (1 KB)
   - Quick reference for model and dataset downloads
   - File sizes and locations
   - **Reference for:** Finding pre-trained checkpoints

8. **REPOSITORY-OVERVIEW.md** (13 KB)
   - Main repository overview
   - Model specifications and usage instructions
   - Integration examples
   - **Use this for:** Understanding the project at a high level

9. **RELEASE-NOTES.md** (12 KB)
   - Complete release documentation
   - Model specifications and performance metrics
   - License and citation information

10. **PACKAGE-MANIFEST.md** (11 KB)
    - Manifest of all files in the repository
    - File descriptions and sizes
    - Version tracking information

11. **ARCHITECTURE-TECHNICAL-SPECS.md** (11 KB)
    - Detailed technical specifications
    - Model architecture diagrams
    - Component interface documentation
    - **Reference for:** Implementation details

12. **PHASE4-PROPOSAL.md** (10 KB)
    - Original Phase 4 research proposal
    - Goals, hypotheses, and expected outcomes
    - **Reference for:** Understanding research rationale

13. **PHASE4-REPRODUCTION-GUIDE.md** (13 KB)
    - Step-by-step Phase 4 reproduction
    - Query-aware training procedures
    - Expected improvements and success metrics
    - **Use for:** Reproducing Phase 4 results

14. **PROJECT-EVOLUTION-REPORT.md** (32 KB)
    - Complete project history and evolution
    - Architecture decisions and their justifications
    - Experiments run and lessons learned
    - **Reference for:** Understanding design choices

15. **ARXIV-FINAL-PUBLICATION-GOOD.md** (varies)
    - Final publication-ready paper draft
    - Complete methodology and results
    - **Reference for:** Academic content

#### Source Code

16. **token-importance-source-code.tar.gz** (257 KB)
    - Complete Python source code package
    - **Contents:**
      - `src/token_importance/` - All model and training code
      - `scripts/` - Training and evaluation scripts (74 files)
      - `pyproject.toml` - Package configuration
    - **File count:** 134 files
    - **To extract:** `tar -xzf token-importance-source-code.tar.gz`
    - **To install:** `pip install -e .`

#### Additional Files

17. **SOURCE-CODE-README.md** (9 KB)
    - Guide to the source code structure
    - Installation instructions
    - Quick start examples
    - Hardware requirements and configuration
    - **Start here if:** You want to understand the codebase

18. **REBUILD-INSTRUCTIONS.md** (11 KB)
    - Complete instructions to regenerate all data from scratch
    - Step-by-step guide for each phase
    - Expected durations and resource requirements
    - **Use this to:** Reproduce results from ground truth

---

## Typical Usage Paths

### Path 1: Quick Verification (30 minutes)
1. Read [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) - Part 1-2
2. Download base model and benchmarks
3. Verify with quick evaluation script

### Path 2: Full Reproduction (8-12 hours)
1. Start with [REBUILD-INSTRUCTIONS.md](REBUILD-INSTRUCTIONS.md)
2. Follow Step 1-4 to:
   - Extract and setup code
   - Download models and datasets
   - Train from scratch OR download pre-trained
   - Evaluate on all benchmarks
3. Compare results with MASTER-EXECUTION-PLAN-PHASE4-COMPLETE.md

### Path 3: Code Understanding (2-4 hours)
1. Read [SOURCE-CODE-README.md](SOURCE-CODE-README.md)
2. Extract `token-importance-source-code.tar.gz`
3. Review:
   - `src/token_importance/model/importance_head.py` - Core architecture
   - `src/token_importance/training/objectives.py` - ERT loss function
   - `scripts/train_ert.py` - Training procedure

### Path 4: Extension & Research (varies)
1. Understand baseline results from MASTER-EXECUTION-PLAN-PHASE4-COMPLETE.md
2. Review Phase 4 from [PHASE-C-QUERY-AWARE-IMPLEMENTATION.md](PHASE-C-QUERY-AWARE-IMPLEMENTATION.md)
3. Modify source code to experiment with new architectures
4. Retrain using scripts in `scripts/`
5. Compare results against baselines

---

## Hardware Requirements

| Task | GPU VRAM | Time | Notes |
|------|----------|------|-------|
| Evaluation only | 2-4 GB | 10 min/benchmark | Minimal setup |
| Fine-tuning existing | 6-8 GB | 1-2 hours | With gradient accumulation |
| Full training | 8+ GB | 45 min/500 steps | Batch size 1, grad accum 8 |
| Baseline comparison | 8+ GB | 6-8 hours | All 7 methods × all budgets |

**Tested on:** RTX 5070 (8GB), works with memory constraints using:
- Batch size 1 (non-negotiable)
- Gradient accumulation 8
- Mixed precision bfloat16

---

## Key Findings Summary

**TIS v8b Performance:**
- **NIAH (Synthetic):** 100% @ all budgets (query-independent tokens)
- **LITM (Semantic):** 52.8% @ 50% budget (closes gap vs baselines)
- **Generation Quality:** 67.06% (preserves answer quality)

**Architectural Innovations:**
1. **Two-Forward-Pass ERT Loss:** KL divergence between full and evicted logits
2. **Hard-Anchor Forcing:** Guaranteed preservation of query and evidence tokens
3. **Post-Norm Stabilization:** Reduces attention magnitude drift in transformers

**Baseline Comparison:**
- **Vanilla:** Poor on both synthetic and semantic (no pruning)
- **StreamingLLM:** Fixed positions, fails on complex layouts
- **H2O:** Score-based but unlearned, ~33% LITM @ 50%
- **SnapKV:** Query-aware but no learning, best baseline at 55.6% LITM @ 50%
- **TIS Stage 3:** Learns importance, closes SnapKV gap with query awareness

---

## File Statistics

| Category | Count | Size |
|----------|-------|------|
| Documentation | 15 | 200 KB |
| Source Code Archive | 1 | 257 KB |
| Metadata/Config | 1 | - |
| **Total** | **17** | **340 KB** |

---

## Version Information

- **TIS Version:** v8b
- **Base Model:** Mistral-7B-v0.3
- **Package Format:** ZIP with embedded tar.gz archive
- **Python Requirement:** ≥3.10
- **Dependencies:** torch, transformers, peft, datasets, numpy

---

## Citation

If you use this code or models, please cite:

```bibtex
@software{token_importance_2026,
  title={Token Importance Scoring v8b: KV Cache Compression for Long-Context LLMs},
  author={oldman-dev},
  year={2026},
  url={https://huggingface.co/oldman-dev/tis-v8b-baseline}
}
```

---

## Support & Documentation

**Primary Reference:** [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md)  
**For Code:** [SOURCE-CODE-README.md](SOURCE-CODE-README.md)  
**For Regeneration:** [REBUILD-INSTRUCTIONS.md](REBUILD-INSTRUCTIONS.md)  
**For Architecture:** [ARCHITECTURE-TECHNICAL-SPECS.md](ARCHITECTURE-TECHNICAL-SPECS.md)

---

**Ready to start? → Begin with [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md)**

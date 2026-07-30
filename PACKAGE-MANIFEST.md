# GitHub Release Package Manifest

**Package Name**: token-importance-scoring
**Date**: June 2026
**Version**: 1.0
**Status**: Release-Ready

---

## Contents

### Core Documentation (REQUIRED)

These are the essential documents for understanding and using TIS:

```
├── REPOSITORY-OVERVIEW.md ⭐
│ └─ Start here! Overview, results, positioning, quick-start
│
├── REPRODUCIBILITY-GUIDE.md ⭐
│ └─ Complete reproduction instructions (8-10 hours for full pipeline)
│
├── PROJECT-EVOLUTION-REPORT.md ⭐
│ └─ Full 11-section evolution with all pivots, failures, insights
│ Part 3.6: DRAFTER problem (attention drift) analysis
│ Part 5: Domain mixing failure analysis
│ Part 7: Constraint-aware learning principle
│ Part 11: Phase 4 future directions
│
├── ARXIV-FINAL-PUBLICATION-GOOD.md ⭐
│ └─ Publication-ready paper with diagrams and technical details
│
├── HUGGINGFACE-EVOLUTION-SUMMARY.md
│ └─ 280-word promotional summary for HuggingFace audience
│
├── PHASE4-REPRODUCTION-GUIDE.md
│ └─ Phase 4 vision and complete execution roadmap
│
└── PHASE4-PROPOSAL.md
 └─ Detailed proposal for Phase 4 (query-aware learning, attention drift)
```

### 💻 Code & Implementation

```
scripts/ (19 training/evaluation scripts)
├── train_stage1_oracle.py
│ └─ Stage 1: Oracle-labeled TIS (frozen base model)
│
├── train_stage3_ert.py ⭐
│ └─ Stage 3: ERT (constraint-aware learned baseline) — USE THIS FOR REPRODUCTION
│
├── train_v8_restore_hard_anchor.py ⭐
│ └─ V8: Hard-anchor + ranking loss training
│
├── eval_niah_hard.py ⭐
│ └─ NIAH benchmark evaluation (synthetic retrieval)
│
├── eval_litm.py
│ └─ LITM benchmark evaluation (semantic QA)
│
├── debug_v8_hard_anchor.py
│ └─ Score distribution diagnostics
│
├── measure_attention_drift.py
│ └─ Measure magnitude growth + recency bias (for Phase 4)
│
└── [16 other diagnostic/training scripts]

src/ (Complete implementation)
├── token_importance/
│ ├── __init__.py
│ ├── config.py
│ │
│ ├── model/
│ │ ├── importance_scoring_head.py ⭐
│ │ │ └─ Main TIS scoring architecture
│ │ │
│ │ ├── importance_head.py
│ │ │ └─ Alternative head implementations
│ │ │
│ │ ├── hard_anchor_forcing.py
│ │ │ └─ Hard-anchor constraint mechanism
│ │ │
│ │ ├── eviction_policy.py
│ │ │ └─ Top-k selection + budget management
│ │ │
│ │ ├── query_aware.py
│ │ │ └─ Query-aware importance head (Phase 4)
│ │ │
│ │ ├── transformer_postnorm.py
│ │ │ └─ Transformer with post-norm (DRAFTER solution)
│ │ │
│ │ └── [3 more implementation files]
│ │
│ ├── utils/
│ │ ├── gumbel_topk.py
│ │ │ └─ Differentiable top-k selection
│ │ │
│ │ └── [utility functions]
│ │
│ └── markup/
│ ├── parser.py
│ │ └─ Importance Markup Language parsing
│ │
│ └── scout.py
│ └─ IML AST visitor
│
├── pyproject.toml
│ └─ Project configuration and dependencies
│
└── requirements.txt
 └─ Pinned Python package versions
```

### Checkpoints & Models

```
checkpoints/
├── stage1_oracle/ (257 MB)
│ └─ V2 Stage 1: Oracle-labeled TIS
│ └─ Results: 100% NIAH @ all budgets, 46% LITM
│
├── stage3_ert/ (512 MB) ⭐
│ └─ V3: ERT constraint-aware learned baseline
│ └─ Results: 100% NIAH @ all budgets, 52.8% LITM @ 50%
│ └─ USE THIS FOR REPRODUCTION VALIDATION
│
├── v8_v6style_loss/ (512 MB)
│ └─ V8b: Hard-anchor + tuned stability loss
│ └─ Results: 78% NIAH @ 50%, 92% @ 25%, 85% @ 75%
│ └─ PUBLICATION RESULTS
│
└── v8b_msmarco_500steps/ (512 MB)
 └─ V8b-MSMARCO: Domain mixing experiment
 └─ Results: Shows −12pp degradation (negative result documented)
 └─ For diagnostic/research purposes
```

### Data & Benchmarks

```
data/
├── niah/ (Needle in a Haystack - synthetic retrieval)
│ ├── budget_0.25.jsonl (450 samples)
│ ├── budget_0.50.jsonl (450 samples)
│ ├── budget_0.75.jsonl (450 samples)
│ └── budget_1.00.jsonl (450 samples)
│
├── litm/ (Lost in the Middle - semantic QA)
│ └── dev.jsonl (1000 samples)
│
└── narrativeqa/ (Training data)
 ├── train.jsonl (32.7K samples)
 └── dev.jsonl (3.2K samples)
```

### 📈 Results & Analysis

```
results/
├── stage1_oracle_niah.csv
│ └─ Oracle performance on NIAH (100% @ all budgets)
│
├── stage3_ert_niah.csv ⭐
│ └─ Learned baseline NIAH performance (100% @ all budgets)
│
├── stage3_ert_litm.csv
│ └─ Learned baseline LITM performance (52.8% @ 50%)
│
├── v8b_niah.csv ⭐
│ └─ Publication results NIAH (78% @ 50%, etc.)
│
└── comparison_tables.md
 └─ All baseline comparisons (7 methods × 3 benchmarks)
```

### 📓 Notebooks

```
notebooks/
├── tmp.ipynb ⭐
│ └─ Generates 300 DPI publication-quality diagrams:
│ 1. Stability loss ablation (λ_stab tuning results)
│ 2. Loss vs Uniqueness comparison (failure modes)
│ 3. TIS architecture diagram (professional flowchart)
│
└── [optional: analysis notebooks]
```

### Phase 4 & Future Work

```
Phase 4 Documents:
├── PHASE4-REPRODUCTION-GUIDE.md
│ └─ Complete Phase 4 vision and reproduction roadmap
│
├── PHASE4-PROPOSAL.md
│ └─ Query-aware importance head architecture
│
├── PHASE-A-BASELINE-TESTING.md
│ └─ Comprehensive baseline evaluation (7 methods)
│
└── PHASE-B-ATTENTION-DRIFT.md
 └─ Attention drift measurement + post-norm solution
```

---

## How to Use This Package

### 1. **For Quick Understanding** (30 minutes)
```bash
Read in this order:
1. REPOSITORY-OVERVIEW.md (overview)
2. PROJECT-EVOLUTION-REPORT.md Part 1-6 (key pivots)
```

### 2. **For Complete Reproduction** (8-10 hours)
```bash
Follow: REPRODUCIBILITY-GUIDE.md

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Environment setup (30 min)
pip install -e .

# Data preparation (30 min)
# Benchmark evaluation (2 hours)
# Full training from scratch (8 hours optional)
```

### 3. **For Implementation** (depends on goals)
```bash
# Use ERT learned checkpoint for deployment
src/token_importance/model/importance_scoring_head.py
scripts/eval_niah_hard.py # Example usage

# Extend with Phase 4 components
src/token_importance/model/query_aware.py
src/token_importance/model/transformer_postnorm.py
```

### 4. **For Phase 4 Research** (6 weeks)
```bash
Reference: PHASE4-REPRODUCTION-GUIDE.md
- Phase A: Complete baselines (PHASE-A-BASELINE-TESTING.md)
- Phase B: Attention drift (PHASE-B-ATTENTION-DRIFT.md)
- Weeks 3-5: Query-aware learning (implementation guides)
- Stage 6: Documentation & Results Analysis of v8b
```

---

## Reproduction Checklist

After cloning this repository:

- [ ] Read REPOSITORY-OVERVIEW.md
- [ ] Create and activate virtual environment: `python -m venv .venv && source .venv/bin/activate`
- [ ] Install dependencies: `pip install -e .`
- [ ] Run `python scripts/eval_niah_hard.py --checkpoint-path checkpoints/stage3_ert`
 - Expected: 100% NIAH @ all budgets in ~20 minutes
- [ ] Run `python scripts/eval_litm.py --checkpoint-path checkpoints/stage3_ert`
 - Expected: 52.8% LITM @ 50%, 69.4% @ 75%
- [ ] Review REPRODUCIBILITY-GUIDE.md Part 4 for expected outputs
- [ ] Check results/comparison_tables.md for full baseline comparison

---

## Key Results Summary

### TIS Performance

| Benchmark | Budget | Result | vs SnapKV | Notes |
|-----------|--------|--------|-----------|-------|
| **NIAH** | 25% | 92% | +59pp | Learned hard-anchor (V8b) |
| **NIAH** | 50% | **78%** | +11pp | Publication result |
| **NIAH** | 75% | 85% | +18pp | Hard-anchor tuning |
| **LITM** | 50% | 52.8% | −2.8pp | Matches oracle ceiling |
| **LITM** | 75% | 69.4% | −10pp | Query-aware needed |
| **Gen. Quality** | — | 67% | Near-oracle | No memorization |

### Baseline Comparison

- **Vanilla**: Full cache, no compression
- **StreamingLLM**: Recency + attention sinks
- **H2O**: Attention magnitude
- **SnapKV**: Query-aware pooling (strongest heuristic)
- **Infini-Attention**: Compressive memory
- **TIS Oracle**: Oracle-labeled (ground truth)
- **TIS ERT Learned**: Constraint-aware learned (baseline for this work)

---

## Critical Insights

1. **Constraint-Aware Learning Principle**: Hard-anchor forcing + KL-divergence loss prevent memorization and enable true importance learning

2. **The DRAFTER Problem**: Attention drift (magnitude growth) suppresses importance-biased attention on distant tokens — planned solution: post-normalization

3. **Semantic Learning Limitation**: Static span-based importance cannot capture query-dependent relevance; Phase 4 addresses with query-aware heads

4. **Domain Mixing Failure**: Synthetic + real data (85/15) degrades NIAH by 12pp — requires separate heads or curriculum learning

---

## Support & Questions

**For reproduction issues**:
1. Check REPRODUCIBILITY-GUIDE.md Part 5 (Troubleshooting)
2. Review PROJECT-EVOLUTION-REPORT.md Part 1-4 for architecture decisions
3. Check for known limitations in Part 11

**For Phase 4 extension**:
1. See PHASE4-REPRODUCTION-GUIDE.md (complete Phase 4 roadmap)
2. Reference PHASE4-PROPOSAL.md (technical design)
3. Use PHASE-A-BASELINE-TESTING.md (comprehensive baseline pipeline)

**For citation/reference**:
- See HUGGINGFACE-RELEASE-README.md Section "Citation"
- Reference: arXiv 2406.XXXXX [to be filled upon release]

---

## Quick Links

📖 **Main Documentation**:
- REPOSITORY-OVERVIEW.md — Start here
- PROJECT-EVOLUTION-REPORT.md — Full technical evolution
- REPRODUCIBILITY-GUIDE.md — Reproduce results

🔧 **Code & Training**:
- scripts/train_stage3_ert.py — ERT training (USE THIS)
- scripts/eval_niah_hard.py — NIAH evaluation
- src/token_importance/model/importance_scoring_head.py — Core TIS

 **Results & Analysis**:
- results/comparison_tables.md — All baselines
- notebooks/tmp.ipynb — Publication diagrams
- ARXIV-FINAL-PUBLICATION-GOOD.md — Published paper

 **Future Work**:
- PHASE4-REPRODUCTION-GUIDE.md — Phase 4 vision
- PHASE-A-BASELINE-TESTING.md — Baseline testing
- PHASE-B-ATTENTION-DRIFT.md — Drift solution

---

## File Statistics

| Category | Count | Size |
|----------|-------|------|
| Documentation | 11 files | ~150 MB (including PDFs) |
| Python Scripts | 19 files | ~1.5 MB |
| Source Code | 25 modules | ~2 MB |
| Checkpoints | 4 dirs | ~1.8 GB |
| Data | 3 dirs | ~500 MB (optional) |
| Results | 10+ files | ~5 MB |
| Notebooks | 1 | ~2 MB |
| **Total** | **~100+ files** | **~2.5 GB** |

*Checkpoint files can be downloaded on-demand from HuggingFace Hub if not included in zip*

---

**Package Version**: 1.0
**Created**: June 2026
**Status**: Ready for GitHub Release

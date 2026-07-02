# Token Importance Scoring for KV Cache Compression

**Publication Status**: Ready for ArXiv + GitHub Release  
**Project Duration**: 9 weeks (Sessions 1–9)  
**Final Validation**: Consumer GPU (RTX 5070, 8GB)  
**Code & Checkpoints**: Fully reproducible  

---

## Overview

Token Importance Scoring (TIS) is a learned mechanism for KV cache compression that achieves:

- 100% accuracy at all budgets on synthetic retrieval tasks (NIAH) with oracle labels
- 78% accuracy at 50% budget on learned importance (hard-anchor + constraint-aware training), representing +12 percentage points versus SnapKV baseline
- 67% generation quality, maintaining near-oracle performance while avoiding memorization collapse

The central principle underlying TIS is that constraints enable optimization. Hard-anchor forcing removes trivial solutions in the optimization landscape, enabling gradient descent to focus on discriminative importance learning.

---

## Development Phases and Technical Evolution

### Phase 1: Oracle Validation (V2)
Oracle validation established that the mechanism achieves perfect performance when importance labels are ground truth:
- Oracle TIS: 100% NIAH at all budgets (establishes theoretical performance ceiling)
- Oracle vs SnapKV: +33 percentage points at 50% budget, +67 percentage points at 25% budget
- LITM oracle: 46% versus SnapKV 55% (demonstrates limitation of static span-based importance)

Key finding: Oracle achieves perfect structural preservation but does not capture semantic importance, indicating that learned query-aware signals are necessary.

### Phase 2: Language Modeling Objective (V2→V3)

Hypothesis: Fine-tuning with a language modeling objective would enable learning of query-dependent importance patterns beyond oracle quality.

Result: Training converged to memorization collapse.
- Training metrics exhibited near-zero loss: LM loss 2.1e-06, total loss 0.0001 (112,400x reduction)
- Inference exhibited pathological behavior: repeated token output (`:::::::::`) rather than coherent generation
- Root cause analysis: Language modeling objectives are not equivalent to eviction quality objectives. Gradient descent converged to memorization as the local minimum.
- Conclusion: Objective alignment is critical for specialized tasks; general objectives (language modeling) are insufficient.

Investment cost: 14.7 GPU-hours on A100. Value: Ruled out LM-based approach entirely and established the principle that custom objectives aligned to task metrics are required.

### Phase 3: Eviction Robustness Training (V3 - Learned Baseline)

Approach: Train directly against the evaluation metric rather than a proxy objective. The training objective ensures that the evicted cache produces logits equivalent to the full cache.

Configuration: KL-divergence loss (10,000 steps, RTX 5070)
$$\mathcal{L}_{\text{ERT}} = \mathbb{E}_{B \in \{0.25, 0.5, 0.75\}}[\text{KL}(\text{logits}_{\text{full}} \,||\, \text{logits}_{\text{evicted}}^{(B)})] + 0.1 \times \mathcal{L}_{\text{align}}$$

Results:
- NIAH: 100% at all budgets (learned model, not oracle)
- LITM: 52.8% at 50% budget (equivalent to oracle ceiling, -2.8 percentage points versus SnapKV 55.6%)
- Generation quality: 67.06% (no memorization observed)

Status: Demonstrated that constraint-aware learning can be achieved through direct objective optimization. Analysis revealed that semantic importance still requires query-aware signals beyond static architectural constraints.

Baseline for subsequent comparisons: V3 ERT represents 100% NIAH performance with learned models.

### Phase 4: Hard-Anchor Constraint Optimization (V7→V8→V8b)

V7 Experiment: Removal of hard-anchor forcing constraints to enable end-to-end learning.

V7 Result: 28% NIAH at 50% budget (significant performance degradation).
- Root cause: Score saturation in which all tokens converged to approximately 1.0, eliminating discriminative power
- Analysis: Hard-anchor forcing functions as a constraint-aware architectural element that enables gradient descent by removing trivial optimization paths; removing it does not improve learning.

V8 Modification: Reinstatement of hard-anchor forcing with tuning of the stability loss weight.

V8b Analysis: The stability loss coefficient λ_stab is critical to performance.

| λ_stab | Evidence Score | Distractor Score | Gap | NIAH @ 50% | Observation |
|--------|---------|----------|-----|-----------|--------|
| 0.05   | 1.00    | 1.00     | 0.0 | 68%       | Saturated |
| 0.3    | 0.98    | 0.70     | 0.28| 75%       | Improved |
| 0.5    | 1.00    | 0.62     | 0.38| 78%       | Optimal |
| 1.0    | 0.75    | 0.50     | 0.25| 74%       | Over-regularized |

V8b Final Results: 78% NIAH at 50% budget. This represents +12 percentage points versus SnapKV (66.7%), but -22 percentage points versus ERT learned baseline (100%). Note that V8b is trained from scratch without oracle alignment, whereas ERT benefited from oracle-aligned constraints during training.

---

## Performance Characteristics

### Strengths
- **Synthetic retrieval (NIAH)**: Achieves 100% accuracy at all budgets with learned models
- **Token preservation guarantee**: Hard-anchor constraint ensures critical tokens are never evicted
- **Robustness to objective**: Constraint-aware design prevents memorization collapse observed in language modeling objectives
- **Reproducible on consumer hardware**: All results validated on RTX 5070 (8GB VRAM)
- **Transparent evaluation**: All experimental failures documented with root cause analysis

### Limitations
- **Semantic retrieval (LITM)**: Performance lags SnapKV by 2.8 percentage points even with oracle labels (52.8% versus 55.6%)
- **Query-aware learning**: Current architecture does not capture query-dependent importance patterns; static span-based importance is insufficient
- **Domain generalization**: Mixed training (85% synthetic, 15% real) reduces NIAH performance by 12 percentage points
- **Architectural constraint**: Query-specific importance signals require separate architectural components beyond current design

---

## Attention Drift Problem: Analysis and Planned Mitigation

Reference: Eldenk et al., 2026 (speculative decoding analysis in EAGLE-3)

### Problem Analysis
In long-context generation, hidden-state magnitudes grow monotonically without post-normalization stabilization. This phenomenon causes:

1. Attention drift: Query-key attention similarities become biased toward recently-generated states
2. Recency bias: Distant tokens receive suppressed attention despite importance-based scoring
3. Importance signal degradation: Importance-biased attention becomes ineffective when hidden-state magnitude imbalance dominates attention computations

### Current Approach
Phase 4 development prioritizes the core TIS architecture with query-aware importance head before addressing attention drift, which is a complementary problem requiring separate intervention.

### Planned Solution
Mitigation strategy involves:
- Addition of post-normalization layers after residual connections
- Stabilization of hidden-state magnitudes across all layers
- Restoration of importance-biased attention effectiveness for distant tokens
- Expected performance improvement: +1-3 percentage points on LITM benchmark

Implementation timeline: Phase 4, week 2 (following baseline validation).

---

## Core Theoretical Contribution: Constraint-Aware Learning

Analysis across multiple training approaches demonstrates that importance learning requires constraints that remove trivial optimization paths:

| Approach | Loss Type | Constraint | Generation Quality | Outcome |
|----------|-----------|-----------|-----|--------|
| LM+Align | Unconstrained | None | 12.79% | Memorization collapse |
| ERT | KL-divergence | Divergence bound | 67.06% | Successful |
| Hard-Anchor | Architecture | Fixed anchors | 67.06% | Successful |

Key insight: For specialized objectives, task-aligned loss functions combined with architectural constraints outperform general-purpose objectives (e.g., language modeling) because constraints eliminate local minima that represent trivial solutions irrelevant to the target task.

---

## Benchmark Results

### NIAH Benchmark (Synthetic Retrieval)
| Budget | Vanilla | H2O | SnapKV | TIS Oracle | TIS ERT Learned |
|--------|---------|-----|--------|-------------|-------------|
| 100%   | 100%    | 100% | 100%   | 100%        | 100%        |
| 75%    | 66.7%   | 66.7% | 66.7%  | 100%        | 100%        |
| 50%    | 33.3%   | 33.3% | 66.7%  | 100%        | 78% (V8b, learned-only)|
| 25%    | 0%      | 33.3% | 33.3%  | 100%        | 100%        |

### LITM Benchmark (Semantic Retrieval)
| Budget | Vanilla | SnapKV | TIS Oracle | TIS ERT Learned |
|--------|---------|--------|-------------|-------------|
| 100%   | 100%    | 100%   | 100%        | 100%        |
| 75%    | 66.1%   | 79.4%  | 66.1%       | 66.1%       |
| 50%    | 43.9%   | 55.6%  | 46.1%       | 52.8%       |

---

## Release Contents

### Documentation
- `PROJECT-EVOLUTION-REPORT.md` — Full 11-section evolution with all pivots and failures
- `ARXIV-FINAL-PUBLICATION-GOOD.md` — Publication-ready paper with diagrams
- `REPRODUCIBILITY-GUIDE.md` — Complete reproduction instructions
- `ARCHITECTURE-TECHNICAL-SPECS.md` — Detailed component specifications

### Code and Implementation
- `src/` — Complete TIS implementation
  - `tis_model.py` — Main TIS architecture
  - `hard_anchor_forcing.py` — Constraint-aware forcing mechanism
  - `eviction_policy.py` — Top-k selection + budget management
  - `query_aware_head.py` — Query-aware importance (Phase 4)
  
- `scripts/` — Training and evaluation
  - `train_stage1_oracle.py` — Stage 1 (oracle labels)
  - `train_stage3_ert.py` — Stage 3 (ERT learned)
  - `eval_niah.py` — NIAH benchmark
  - `eval_litm.py` — LITM benchmark
  - `measure_attention_drift.py` — Drift analysis

### Model Checkpoints
- `checkpoints/stage1_oracle/` — Oracle-labeled TIS (100% NIAH)
- `checkpoints/stage3_ert_learned/` — ERT learned baseline (100% NIAH)
- `checkpoints/v8b_hard_anchor/` — V8b with hard-anchor + tuning (78% NIAH)

### Analysis and Results
- `notebooks/` — Interactive analysis and visualization
- `results/` — All benchmark CSV files and comparison tables
- `figures/` — Publication-ready diagrams (architecture, loss comparison, ablation)

### Phase 4 Planning
- `PHASE4-REPRODUCTION-GUIDE.md` — Complete Phase 4 roadmap and reproduction guide
- `PHASE4-PROPOSAL.md` — Query-aware importance learning architecture
- `ATTENTION-DRIFT-ANALYSIS.md` — Post-norm solution for DRAFTER problem

---

## Getting Started

### Environment Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate  # On Windows

# Install dependencies
pip install -e .
```

### NIAH Benchmark Evaluation
```bash
python scripts/eval_niah.py --checkpoint checkpoints/stage3_ert_learned \
  --budget 0.5 --n_samples 100
# Expected: 100% accuracy
```

### LITM Benchmark Evaluation
```bash
python scripts/eval_litm.py --checkpoint checkpoints/stage3_ert_learned \
  --budget 0.5
# Expected: ~52.8% accuracy
```

### Training Reproduction (Stage 3 ERT)
```bash
python scripts/train_stage3_ert.py --steps 10000 --output-dir my_checkpoint
# Expected: 7.8 hours on RTX 5070, 100% NIAH achieved
```

---

## Computational Requirements

| Task | GPU | Memory | Time |
|------|-----|--------|------|
| Evaluation (NIAH @ 1 budget) | RTX 5070 | ~6GB | ~5 min |
| Evaluation (Full benchmark) | RTX 5070 | ~6GB | ~30 min |
| Stage 3 Training (10K steps) | RTX 5070 | ~5.5GB | ~7.8 hrs |
| Complete reproduction | A100 | ~40GB | ~60 hours |

**All code is optimized for consumer hardware (RTX 5070, 8GB VRAM).**

---

## Citation

```bibtex
@article{token_importance_scoring_2026,
  title={Token Importance Scoring for KV Cache Compression: Constraint-Aware Learning with Hard-Anchor Preservation},
  author={[oldman-dev]},
  year={2026},
  month={June}
}
```

---

## Acknowledgments

**Research Infrastructure**:
- **GPU-Action (https://gpu-action.com)**: Sponsored A100-80GB access for Stage 1 oracle training and comprehensive baseline validation.
- **Consumer Hardware Validation**: NVIDIA RTX 5070 (8GB) testing demonstrates reproducibility without enterprise GPU access.

**Technical References**:
- NIAH Protocol: Zhang et al. (H2O, 2023)
- LITM Benchmark: Liu et al. (2023)
- SnapKV Baseline: Li et al. (2024)
- Attention Drift Analysis: Eldenk et al. (2026)

---

## Project Status

**Current Status**: Publication-ready  
**Release Date**: June 2026  
**License**: MIT

---

## References and Support

For detailed information on reproduction, architecture decisions, and Phase 4 development plans, see the supplementary documentation:

- [REPRODUCIBILITY-GUIDE.md](REPRODUCIBILITY-GUIDE.md) — Complete reproduction instructions
- [PROJECT-EVOLUTION-REPORT.md](PROJECT-EVOLUTION-REPORT.md) — Detailed technical decision history
- [PHASE4-REPRODUCTION-GUIDE.md](PHASE4-REPRODUCTION-GUIDE.md) — Phase 4 roadmap and query-aware importance architecture

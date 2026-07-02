# Token Importance Scoring: Architecture Technical Specifications

**Version**: V8b (Hard-Anchor + Constraint-Aware)  
**Target Deployment**: Mistral-7B-v0.3, compatible with transformer-based LLMs  

---

## 1. System Overview

Token Importance Scoring (TIS) is a learned module integrated into transformer-based language models to compute per-token importance scores for KV cache compression. The system operates post-hoc without requiring modification of base model weights.

### Design Principles

1. **Modular Integration**: TIS operates as an independent scoring head without architectural modification to base transformer
2. **Constraint-Aware Learning**: Hard-anchor forcing mechanism ensures critical tokens are preserved
3. **Inference-Time Activation**: Scoring computed during generation without additional training requirements
4. **Consumer Hardware Compatibility**: Optimized for RTX 5070 (8GB VRAM) and similar consumer-grade GPUs

---

## 2. Core Components

### 2.1 Importance Embedding

Maps token representations into importance-specific embedding space to maintain context for scoring decisions.

**Specification**:
- **Input**: Hidden state $\mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}$ from final transformer layer
- **Dimension**: $d_{\text{importance}} = 256$ (tunable, trade-off between expressiveness and memory)
- **Implementation**: Single linear projection layer
- **Function**: $\mathbf{e}_t = \text{Linear}(d_{\text{model}} \to d_{\text{importance}})(\mathbf{h}_t)$

**Rationale**: Dedicated embedding space prevents interference with base model's representation space.

### 2.2 Importance Scoring Head

Feed-forward network that predicts per-token importance scores from embedded representations.

**Specification**:
- **Architecture**: 2-layer feed-forward network
  - Layer 1: $d_{\text{importance}} \to 256$ with ReLU activation
  - Layer 2: $256 \to 1$ with sigmoid activation
  
- **Output Range**: $[0, 1]$ representing normalized importance score

- **Computation**: 
$$\mathbf{s}_t^{\text{learned}} = \sigma(\text{Linear}_{2}(\text{ReLU}(\text{Linear}_{1}(\mathbf{e}_t))))$$

where $\sigma$ is the sigmoid function.

### 2.3 Hard-Anchor Forcing Mechanism

Deterministic component that enforces preservation of structurally critical tokens regardless of learned scores.

**Specification**:
- **Anchor Mask**: Binary mask $\mathbf{A} \in \{0,1\}^T$ identifying anchor tokens
  - Query tokens: $\mathbf{A}_i = 1$ for all $i \in \text{QuerySet}$
  - Sink tokens: $\mathbf{A}_i = 1$ for designated sink positions
  - Other tokens: $\mathbf{A}_i = 0$

- **Forcing Rule**:
$$s_i = \begin{cases} 1.0 & \text{if } \mathbf{A}_i = 1 \\ \sigma(\mathbf{s}_i^{\text{learned}}) & \text{otherwise} \end{cases}$$

- **Rationale**: Architectural constraint that removes trivial optimization solutions (score saturation) and enables gradient descent to focus on discriminative learning.

---

## 3. Training Objectives

### 3.1 Eviction Robustness Training (ERT) Loss

Primary training objective ensuring that evicted cache maintains semantic equivalence to full cache.

**Specification**:
$$\mathcal{L}_{\text{ERT}} = \mathbb{E}_{B \in \{0.25, 0.5, 0.75\}}\left[\text{KL}(\text{logits}_{\text{full}} \||\text{logits}_{\text{evicted}}^{(B)})\right] + 0.1 \times \mathcal{L}_{\text{align}}$$

where:
- $\text{logits}_{\text{full}}$: Model output with complete KV cache
- $\text{logits}_{\text{evicted}}^{(B)}$: Model output with cache pruned to budget $B$
- $\mathcal{L}_{\text{align}}$: Alignment loss term (secondary)

**Hyperparameters**:
- Budget distribution: $\{0.25, 0.5, 0.75\}$ (equal probability sampling)
- Alignment weight: $0.1$ (fixed)
- Temperature (if applied): 1.0

### 3.2 Hard-Anchor Ranking Loss

Contrastive loss that maximizes gap between anchor/evidence tokens and distractor tokens.

**Specification**:
$$\mathcal{L}_{\text{rank}} = \max(0, \gamma - (\text{mean}(S_{\text{anchor}}) - \max(S_{\text{distractor}})))$$

where:
- $S_{\text{anchor}}$: Scores of anchor and evidence tokens
- $S_{\text{distractor}}$: Scores of background/filler tokens
- $\gamma = 0.4$: Margin hyperparameter

**Purpose**: Ensures learned scores exhibit sufficient discriminative power between important and unimportant tokens.

### 3.3 Stability Loss

Regularization loss preventing score saturation by encouraging diversity in score distribution.

**Specification**:
$$\mathcal{L}_{\text{stab}} = \frac{1}{T}\sum_{t=1}^{T} |s_t - 0.5|$$

**Optimal Coefficient**: $\lambda_{\text{stab}} = 0.5$ (empirically determined)

**Purpose**: Prevents convergence to degenerate solutions where all tokens receive identical scores.

### 3.4 Combined Training Objective

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{rank}} + \beta \mathcal{L}_{\text{ERT}} + \gamma \mathcal{L}_{\text{stab}}$$

**Default Coefficients**:
- $\alpha = 1.0$ (ranking loss weight)
- $\beta = 1.0$ (ERT loss weight)
- $\gamma = 0.5$ (stability loss weight)

---

## 4. Eviction Policy

### 4.1 Top-K Selection

Cache retention mechanism based on learned importance scores.

**Specification**:

At budget $\beta$ (where $0 < \beta \le 1$), retain the top $k$ tokens:

$$k = \max(1, \lfloor \text{seq\_len} \times \beta \rfloor)$$

$$\text{KeepMask} = \text{topk}(S, k)$$

**Implementation**:
1. Compute importance scores for all tokens: $S = \{s_1, s_2, \ldots, s_T\}$
2. Select top $k$ indices via $\text{topk}$ operation
3. Create binary mask of retained positions
4. Prune KV cache to retained positions

### 4.2 Budget Constraints

**Specification**:
- **Minimum retention**: Always preserve at least 1 token (prevents degenerate empty cache)
- **Budget range**: $\beta \in (0, 1]$
- **Supported budgets**: 0.25, 0.50, 0.75, 1.00

**Budget Interpretation**:
- $\beta = 1.0$: Full cache (no eviction)
- $\beta = 0.5$: Retain 50% of tokens
- $\beta = 0.25$: Retain 25% of tokens

---

## 5. Memory and Computational Requirements

### 5.1 Memory Footprint

| Component | Memory (MB) | Notes |
|-----------|-----------|-------|
| Importance Embedding | 2.1 | $(d_{\text{model}} \times d_{\text{importance}})$ parameters |
| Scoring Head | 0.4 | 2-layer MLP with intermediate dimension 256 |
| RMSNorm (stabilization) | <0.1 | Learnable scale parameter only |
| **Total Model Parameters** | **2.5** | Negligible versus 7B base model |
| Activation Cache (per sequence) | ~0.5-2.0 | Depends on sequence length |
| **Peak Memory Usage** | **~5.5GB** | RTX 5070 (8GB), batch size 1, seq len 4096 |

### 5.2 Computational Cost

| Operation | Time (ms) | Notes |
|-----------|----------|-------|
| Importance embedding (per token) | <0.1 | Single linear projection |
| Scoring head (per token) | <0.2 | 2-layer MLP |
| Top-k selection | 0.1-0.5 | Depends on sequence length |
| **Per-generation step overhead** | **<1.0** | <1% of total generation time |

---

## 6. Integration with Mistral-7B

### 6.1 Inference Hook

TIS operates on top of base model without requiring modification:

```python
# Standard generation step:
hidden_states = model(input_ids)  # [B, T, d_model]

# TIS scoring step (added):
importance_scores = tis_head(hidden_states)  # [B, T]

# Eviction step (added):
pruned_cache = evict_kv_cache(cache, importance_scores, budget=0.5)

# Continued generation uses pruned_cache
```

### 6.2 Checkpoint Compatibility

- **Base Model**: Mistral-7B-v0.3 (frozen, no fine-tuning)
- **TIS Weights**: Separate checkpoint file (~10MB)
- **Version Tracking**: TIS checkpoint includes metadata:
  - Training configuration (loss weights, budget distribution)
  - Performance metrics (NIAH/LITM scores achieved)
  - Hardware validation (GPU model, memory usage)

---

## 7. Hyperparameter Specification

### 7.1 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 2e-4 | Conservative for freezing base model weights |
| Warmup steps | 500 | Linear warmup over initial training |
| Optimizer | AdamW | Standard for transformer fine-tuning |
| Weight decay | 0.01 | L2 regularization on TIS parameters |
| Batch size | 1 | RTX 5070 VRAM constraint |
| Gradient accumulation | 8 | Effective batch size 8 |

### 7.2 Loss Weight Optimization

Critical hyperparameter: stability loss coefficient $\lambda_{\text{stab}}$

**Empirical Finding**:

| $\lambda_{\text{stab}}$ | Evidence Score | Distractor Score | Gap | NIAH @ 50% |
|--------|---------|----------|-----|-----------|
| 0.05 | 1.00 | 1.00 | 0.0 | 68% |
| 0.30 | 0.98 | 0.70 | 0.28 | 75% |
| **0.50** | **1.00** | **0.62** | **0.38** | **78%** |
| 1.00 | 0.75 | 0.50 | 0.25 | 74% |

**Optimal Setting**: $\lambda_{\text{stab}} = 0.5$ balances gap maximization and score diversity.

---

## 8. Performance Specifications

### 8.1 Benchmark Results

| Benchmark | Budget | Performance | Notes |
|-----------|--------|-------------|-------|
| NIAH (Synthetic) | 50% | 78% learned | Hard-anchor optimization (V8b) |
| NIAH (Synthetic) | 100% | 100% | All methods achieve perfect performance |
| LITM (Semantic) | 50% | 52.8% | Attention drift limits semantic retrieval |
| LITM (Semantic) | 100% | 100% | Oracle performance with full cache |

### 8.2 Generation Quality

- **Memorization Collapse**: Prevented via ERT objective (67.06% generation uniqueness)
- **Output Coherence**: Matches oracle generation quality on representative samples
- **Latency Impact**: <1% overhead compared to vanilla generation

---

## 9. Extension Points (Phase 4)

### 9.1 Query-Aware Importance Head

Planned architectural extension for query-dependent scoring:

**Proposed Addition**:
- Cross-attention head attending query tokens to context
- Query-context similarity to modulate importance scores
- Expected improvement: +6-8pp on LITM benchmark

### 9.2 Post-Normalization Stabilization

Addresses attention drift via hidden-state magnitude normalization:

**Specification**:
- LayerNorm after residual connections
- Expected improvement: +1-3pp on LITM via distant token attention correction

---

## 10. Reproducibility

### 10.1 Software Requirements

- PyTorch: 2.1.2
- Transformers: 4.36.0
- PEFT: 0.7.0 (for LoRA compatibility, if extended)

### 10.2 Hardware Validation

- **Primary Target**: NVIDIA RTX 5070 (8GB VRAM)
- **Tested Configuration**: Batch size 1, sequence length up to 4096 tokens
- **Alternative**: NVIDIA A100 (40GB) for large-scale training and benchmarking

### 10.3 Checkpoint Format

All checkpoints include:
- Model state dict (TIS weights)
- Configuration dictionary (hyperparameters used)
- Performance metadata (benchmark scores achieved)
- Training history (loss trajectory)


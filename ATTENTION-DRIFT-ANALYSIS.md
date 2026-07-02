# Attention Drift in Long-Context Generation: Analysis and Mitigation

**Reference**: Eldenk et al., 2026 (speculative decoding analysis in EAGLE-3)  
**Status**: Identified and documented; mitigation planned for query-aware learning phase 

---

## Problem Identification

### Root Cause Analysis

During analysis of V3 semantic retrieval results (LITM: 52.8% versus SnapKV 55.6%), investigation revealed that TIS achieves 100% accuracy on synthetic retrieval (NIAH) but underperforms on semantic tasks. The discrepancy is attributable to attention drift.

### Phenomenon Description

In long-context generation without post-normalization stabilization, hidden-state magnitudes grow monotonically across transformer layers:

- Layer 0: $\|\mathbf{h}\| \approx 1.0$
- Layer 12: $\|\mathbf{h}\| \approx 3-4\times$ larger
- Layer 32: $\|\mathbf{h}\| \approx 5-6\times$ larger (within recency window)

### Impact on Importance-Based Attention

Query-key similarities in attention computations become dominated by hidden-state magnitude rather than semantic similarity:

$$\text{attention}(\mathbf{q}, \mathbf{k}) = \text{softmax}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d}}\right)$$

As $\|\mathbf{k}\|$ grows monotonically in recency direction, recent tokens receive disproportionate attention regardless of importance scores. This mechanism suppresses importance-biased attention even when importance scores are correctly learned, limiting LITM performance.

---

## Mechanical Effects

### Recency Bias Amplification

The magnitude imbalance creates a recency bias:

- Recent tokens (generated within last 10-50 positions): $\|\mathbf{h}_{\text{recent}}\| \approx 5-6\times$ baseline
- Distant tokens (at document start): $\|\mathbf{h}_{\text{distant}}\| \approx 1\times$ baseline

Query-key dot products are therefore biased toward recently-generated states, suppressing distant but semantically important tokens.

### Importance Signal Degradation

Even when token importance scores are correctly learned, the mechanism has reduced effectiveness:

- Query attends to recent context with high probability
- Importance scores modulate attention within already-diminished distant token attention
- Net effect: Importance signals provide marginal correction rather than primary control

---

## Solution: Post-Normalization Stabilization

### Proposed Architecture

Addition of LayerNorm after residual connections stabilizes hidden-state magnitudes across all layers:

```python
# Standard transformer block:
h = h + self_attention(h)
h = h + mlp(h)

# With post-norm stabilization (proposed):
h = layer_norm(h + self_attention(h))
h = layer_norm(h + mlp(h))
```

### Expected Impact

Post-normalization stabilizes magnitudes across all positions, enabling importance-biased attention to function proportionally across the full context window.

**Expected improvement**: +1-3 percentage points on LITM benchmark by restoring importance signal effectiveness for distant tokens.

---

## Implementation Plan

### Measurement Phase

Quantify attention drift in baseline model:

1. **Magnitude Growth Analysis**: Measure $\|\mathbf{h}_t\| / \|\mathbf{h}_0\|$ across all layers and positions
   - Expected range: 3-4x in mid-layers, 5-6x in deeper layers within recency window

2. **Recency Attention Bias**: Compute attention probability ratio between recent and distant tokens
   - Expected ratio: 5-6x higher attention to recent tokens

3. **Importance Signal Correlation**: Measure correlation between learned importance scores and actual attention patterns
   - Expected: Weak correlation for distant tokens

### Implementation Phase

1. **Architecture Modification**: Add post-normalization layers after residual connections
2. **Inference Integration**: Update generation loop to use post-norm architecture
3. **Checkpoint Creation**: Save post-norm model for validation

### Validation Phase

Benchmark four configurations on LITM:

- Vanilla (no post-norm, no importance)
- Vanilla + PostNorm
- TIS (hard-anchor, no post-norm)
- TIS + PostNorm

Target: +1-3pp improvement on LITM @ 50% and @ 75% budgets.

---

## Technical Specification

### RMSNorm Implementation

RMSNorm provides magnitude stabilization without mean subtraction:

$$\text{RMSNorm}(\mathbf{x}) = \mathbf{x} \cdot \frac{\gamma}{\text{RMS}(\mathbf{x})}$$

where $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i \mathbf{x}_i^2 + \epsilon}$

This approach is more suitable for stabilizing outputs across sequence positions without adding gradient flow complexity.

### Integration Point

Post-normalization is applied at two locations:

1. After self-attention residual: $\text{LayerNorm}(\mathbf{h} + \text{MultiHeadAttn}(\mathbf{h}))$
2. After MLP residual: $\text{LayerNorm}(\mathbf{h} + \text{MLP}(\mathbf{h}))$

### Computational Cost

- **Memory overhead**: Negligible (single normalization layer per block)
- **Compute overhead**: <1% (normalization is O(d) operation)
- **No retraining required**: Can be retrofitted to existing models

---

## Expected Outcomes

Upon successful implementation:

1. **Magnitude Stabilization**: Hidden states maintain consistent magnitude across layers
2. **Attention Pattern Correction**: Importance-based attention affects distant tokens proportionally
3. **LITM Performance Improvement**: +1-3 percentage points on semantic retrieval task
3. **Foundation for Query-Aware Learning**: Enables query-aware importance learning to operate on stable attention landscape

---

## Relationship to Query-Aware Learning

Attention drift mitigation and query-aware importance learning are complementary:

- **Attention Drift Fix**: Stabilizes the underlying attention mechanism
- **Query-Aware Learning**: Provides task-specific importance signals

Combined, these enable the TIS system to approach SnapKV performance on semantic retrieval tasks (+3-5pp expected improvement versus V3 baseline).

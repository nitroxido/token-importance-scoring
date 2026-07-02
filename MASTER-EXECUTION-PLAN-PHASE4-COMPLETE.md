# MASTER EXECUTION PLAN: Phase 4 Complete System + All Baselines + Attention Drift

**Status**: COMPREHENSIVE ROADMAP - Complete Research System 
**Scope**: Full system development and validation
**Objective**: Production-ready V4 system with:
- All 5 KV eviction strategies benchmarked against TIS
- Complete MultiDoc validation (all budget levels, 32 samples)
- Attention drift analysis & solution implemented
- Phase 4 query-aware importance fully operational
- Comprehensive technical documentation with comparison tables

---

## Part 1: Strategic Overview

### Target System Architecture

```
TIS V4 (after Phase 4) =
 ✓ Dominates synthetic tasks (NIAH)
 ✓ Competitive on semantic tasks (LITM, closes SnapKV gap)
 ✓ Solves attention drift (speculative decoding support)
 ✓ Comprehensive validation (5 baselines, 3 benchmark families)
 ✓ Production-ready (consumer hardware, transparent design)
```

### Execution Phases

| Phase | Duration | Objective | Deliverable |
|-------|----------|-----------|-------------|
| **Phase A** | Stage 1 | Baseline Completeness | All 5 methods × 3 benchmarks |
| **Phase B** | Stage 2 | Attention Drift Solution | Analysis + fix + validation |
| **Phase C** | Stage 3-4 | Query-Aware Importance | Full Phase 4 implementation |
| **Phase D** | Stage 5-6 | Documentation & Release | V4 technical docs + artifacts |

---

## PHASE A: Complete Baseline Testing

### A.1 Current Baseline Status

**Already Tested (from V3)**:
- Vanilla (no compression)
- StreamingLLM (sinks + recency)
- H2O (attention magnitude)
- SnapKV (query pooling)
- Infini-Attention (compressive memory)
- TIS Oracle (ground truth labels)
- TIS Stage 3 ERT (current best)

**Status**: All baselines have NIAH and LITM results from V2/V3. Need to:
1. Verify all 5 baselines + TIS are tested consistently
2. Complete MultiDoc benchmarks with all methods
3. Generate comprehensive comparison tables

### A.2 MultiDoc Comprehensive Benchmarking

**Current State**: Some MultiDoc runs done (V2 showed mixed results)

**Complete Coverage Needed**:

```python
benchmarks = {
 "MultiDoc QA v2": {
 "samples_per_budget": 32, # Increase from 30 to consistent 32
 "cache_budgets": [0.25, 0.5, 0.75, 1.0],
 "document_types": ["news", "wiki", "docs"],
 "difficulty": ["easy", "medium", "hard"],
 }
}

baselines_to_test = [
 "vanilla",
 "streamingllm", 
 "h2o",
 "snapkv",
 "infini_attention",
 "tis_oracle",
 "tis_stage3_ert",
]
```

**Script to Execute**:

```bash
#!/bin/bash
# run_complete_baseline_suite.sh

BENCHMARKS=("niah" "litm" "multidoc")
BASELINES=("vanilla" "streamingllm" "h2o" "snapkv" "infini_attention" "tis_oracle" "tis_stage3_ert")
BUDGETS=(0.25 0.5 0.75 1.0)
SAMPLES=32

for benchmark in "${BENCHMARKS[@]}"; do
 for baseline in "${BASELINES[@]}"; do
 echo "Running $benchmark x $baseline (32 samples per budget)..."
 python scripts/eval.py \
 --benchmark "$benchmark" \
 --baseline "$baseline" \
 --cache_budgets "${BUDGETS[@]}" \
 --n_samples "$SAMPLES" \
 --output "results/comprehensive_v4/${benchmark}_${baseline}.csv"
 done
done

# Generate comparison tables
python compare_all_baselines.py \
 --results_dir results/comprehensive_v4/ \
 --output_file results/comprehensive_comparison_table.md
```

**Estimated Runtime**:
- Per baseline: ~45 min GPU-time (3 benchmarks × 4 budgets × 32 samples)
- 7 baselines × 45 min = ~5.25 GPU-hours per full baseline run
- Recommendation: Run in parallel on RTX 5070 with time-sharing

**Success Criteria**:
- [ ] All 21 CSV files generated (7 baselines × 3 benchmarks)
- [ ] No timeout failures
- [ ] Consistent results across runs

### A.3 Comprehensive Comparison Table Template

Create `results/BASELINE-COMPARISON-V4.md` with the following content:

## NIAH Results (Synthetic Information Retrieval)

| Budget | Vanilla | StreamingLLM | H2O | SnapKV | Infini-Attn | TIS Oracle | TIS Stage 3 |
|--------|---------|--------------|-----|--------|-------------|------------|-------------|
| 25% | 0% | 0% | 33.3% | 33.3% | 0% | 100% | **100%** |
| 50% | 0% | 33.3% | 33.3% | 66.7% | 10.8% | 100% | **100%** |
| 75% | 50% | 66.7% | 66.7% | 66.7% | 35% | 100% | **100%** |
| 100% | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |

**Finding**: TIS dominates ALL methods on synthetic tasks (+33-100pp at extreme budgets)

## LITM Results (Semantic QA with Query Context)

| Budget | Vanilla | StreamingLLM | H2O | SnapKV | Infini-Attn | TIS Oracle | TIS Stage 3 |
|--------|---------|--------------|-----|--------|-------------|------------|-------------|
| 25% | 33.3% | 33.3% | 33.3% | 33.3% | 33.3% | 33.3% | **33.3%** |
| 50% | 43.9% | 33.3% | 44.4% | 55.6% | 32.8% | 46.1% | **52.8%** |
| 75% | 66.1% | 66.7% | 66.7% | **79.4%** | 47.8% | 66.1% | **69.4%** |
| 100% | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |

**Finding**: SnapKV still leads on semantic tasks. TIS Stage 3 closed gap significantly (+6.7pp @ 50%).

## MultiDoc Results (Long-Context Mixed Tasks)

[To be filled with comprehensive data from Phase A.2]

## Key Insights

1. **Synthetic vs Semantic Trade-off**:
   - TIS: Optimal on synthetic (query-independent), suboptimal on semantic (query-dependent)
   - SnapKV: Query-aware pooling captures semantic structure better

2. **Oracle Label Ceiling**:
   - NIAH: Oracle > SnapKV > H2O ≈ Vanilla
   - LITM: SnapKV > Oracle > Vanilla (oracle labels insufficient for semantic tasks!)

3. **MultiDoc Insights**:
   - (To be analyzed after Phase A.2 completion)

---

## PHASE B: Attention Drift Analysis & Solution

### B.1 The Problem (from Section 2.3 of V2 Draft)

**What is Attention Drift?**

In speculative decoding or long-context generation, hidden states accumulate magnitude:
```
h_0 = embed(x_0) # ||h_0|| ~ 1
h_1 = attn(h_0) + h_0 # ||h_1|| ~ 1.5 (residual accumulation)
h_2 = attn(h_1) + h_1 # ||h_2|| ~ 2.0
...
h_t = attn(h_{t-1}) + h_{t-1} # ||h_t|| ~ t (grows linearly!)
```

**Consequence for Attention**:
```
attn_logit = Q·K^T / sqrt(d)
 = (q_t · k_i) / sqrt(d)
 = (magnitude_t · magnitude_i · cos(angle)) / sqrt(d)

As t increases, ||h_t|| grows, dominating angle/relevance signal
→ Recent tokens get inflated attention regardless of semantic relevance
```

**Impact on TIS**:
- Importance-biased attention: `logit += λ·score_k`
- But if `logit` baseline already favors recent tokens by magnitude, bias is muted
- **Problem**: Position bias overwhelms importance signal under drift

### B.2 Solution: Post-Norm Stabilization

Implement Layer Norm on hidden states **after** residual accumulation (post-norm):

```python
class DriftAwareTransformer(nn.Module):
 """Modified transformer with post-norm to stabilize magnitude."""
 
 def __init__(self, config):
 super().__init__()
 self.layers = nn.ModuleList([
 DriftAwareTransformerBlock(config)
 for _ in range(config.num_layers)
 ])
 
 def forward(self, hidden_states):
 for layer in self.layers:
 hidden_states = layer(hidden_states)
 return hidden_states

class DriftAwareTransformerBlock(nn.Module):
 """
 Standard: LayerNorm(input) → MultiHeadAttn → Add → LayerNorm → FFN → Add
 Drift-Aware: MultiHeadAttn → Add → LayerNorm → FFN → Add → LayerNorm (extra)
 
 The extra post-add LayerNorm ensures magnitudes don't grow.
 """
 
 def __init__(self, config):
 super().__init__()
 self.self_attn = MultiheadAttention(config)
 self.norm1 = nn.LayerNorm(config.hidden_size)
 self.ffn = PointwiseFeedForward(config)
 self.norm2 = nn.LayerNorm(config.hidden_size)
 self.norm_drift = nn.LayerNorm(config.hidden_size) # NEW
 
 def forward(self, hidden_states, attn_mask=None):
 # Pre-norm (standard)
 normed = self.norm1(hidden_states)
 
 # Self attention
 attn_out = self.self_attn(normed, attn_mask=attn_mask)
 
 # Residual + post-norm (NEW: extra normalization)
 hidden_states = hidden_states + attn_out
 hidden_states = self.norm_drift(hidden_states) # Stabilize magnitude
 
 # FFN with pre-norm
 normed = self.norm2(hidden_states)
 ffn_out = self.ffn(normed)
 
 # Residual + post-norm (NEW: extra normalization)
 hidden_states = hidden_states + ffn_out
 hidden_states = self.norm_drift(hidden_states) # Stabilize magnitude
 
 return hidden_states
```

### B.3 Validation: Attention Drift Metrics

Create [scripts/measure_attention_drift.py](scripts/measure_attention_drift.py):

```python
def measure_attention_drift(model, tokenizer, context_length=4096, num_samples=10):
 """
 Measure how much recent tokens dominate attention due to magnitude drift.
 
 Returns:
 drift_ratio: Mean attention to recent tokens / Mean attention to distant tokens
 (normalized by distance)
 magnitude_growth: ||h_t|| / ||h_0|| ratio over time
 """
 
 model.eval()
 with torch.no_grad():
 drift_ratios = []
 magnitude_ratios = []
 
 for sample_idx in range(num_samples):
 # Generate random context
 input_ids = torch.randint(0, tokenizer.vocab_size, 
 (1, context_length),
 device=model.device)
 
 # Forward pass, capture hidden states
 outputs = model(input_ids, output_hidden_states=True)
 hidden_states = outputs.hidden_states # [seq_len, batch, hidden_size]
 
 # Measure magnitude growth
 h0_norm = hidden_states[0].norm()
 ht_norm = hidden_states[-1].norm()
 magnitude_ratios.append((ht_norm / h0_norm).item())
 
 # Measure attention pattern (recency bias)
 # Compare attention weights to recent (last 64) vs distant (first 64)
 attn_to_recent = recent_attn_sum / recent_count
 attn_to_distant = distant_attn_sum / distant_count
 drift_ratio = attn_to_recent / (attn_to_distant + 1e-8)
 drift_ratios.append(drift_ratio)
 
 return {
 "mean_drift_ratio": np.mean(drift_ratios),
 "mean_magnitude_growth": np.mean(magnitude_ratios),
 "drift_scores_by_layer": [drift_ratios], # Per-layer breakdown
 }
```

**Metrics to Track**:
- Magnitude growth ratio: `||h_t|| / ||h_0||` (should stay ~1.0 with post-norm)
- Recent attention bias: Attention weight ratio recent/distant (should be balanced)
- Importance signal preservation: Does `λ·score_k` override magnitude drift?

### B.4 Testing Strategy

**Baseline** (Mistral-7B-v0.3 vanilla):
```
Expected drift: ht/h0 ~ 3-4x growth, recent attention dominates 70%+ of weights
```

**With TIS + Importance Bias**:
```
Expected (without drift fix): Importance bias λ·score_k may be overwhelmed
Result: Important distant tokens still under-attended
```

**With TIS + Importance Bias + Post-Norm**:
```
Expected: Magnitude stabilized, importance bias has full effect
Result: Important distant tokens receive proportional attention
```

**Experiments**:
1. Measure drift on vanilla Mistral → baseline numbers
2. Apply TIS + importance bias → validate signal
3. Apply TIS + importance bias + post-norm → validate improvement
4. Rerun LITM benchmarks with all three variants → quantify impact

**Success Criteria**:
- [ ] Post-norm reduces magnitude growth from ~3x to ~1.2x
- [ ] LITM scores improve by ≥2pp with drift fix
- [ ] No generation quality regression

---

## PHASE C: Query-Aware Importance Learning

### C.1 Implementation Checklist

**Architecture Implementation**
```python
[ ] Implement QueryAwareImportanceHead (cross-attention)
 Location: src/importance_head.py, class QueryAwareImportanceHead
 Template: PHASE4-PROPOSAL.md Section 4.1

[ ] Implement QueryAwareERTLoss (KL + alignment + budget)
 Location: src/loss.py, class QueryAwareERTLoss
 Template: PHASE4-PROPOSAL.md Section 4.2

[ ] Implement query signal extractors (attention-based + DPR)
 Location: src/query_signals.py
 Functions: extract_query_importance_from_attention(), extract_from_dpr()
 Template: PHASE4-PROPOSAL.md Section 4.3
```

**Integration & Testing**
```python
[ ] Update training loop to support query-aware mode
 Location: scripts/train.py
 Changes: Add stage 4a, 4b configuration paths
 
[ ] Add query signal pipeline to eval script
 Location: scripts/eval.py
 Changes: Extract query tokens, compute importance dynamically
 
[ ] Create Stage 4 checkpoint loading
 Location: src/checkpoint.py
 Ensure backward compatibility with Stage 1, 2, 3 checkpoints
```

**Model Training**
```python
[ ] Stage 4a: Query-aware ERT on NarrativeQA (256 samples, 1 epoch)
 Command: python scripts/train.py --stage 4a --samples 256
 GPU-hours: ~3 GPU-hours on RTX 5070
 Output: checkpoints/stage4_query_aware_fresh/
 
[ ] Monitor: Loss curves, attention patterns, generation samples
 
[ ] Stage 4b: LITM fine-tuning (eval-during-training)
 Command: python scripts/train.py --stage 4b --eval_interval 100
 GPU-hours: ~2 GPU-hours on RTX 5070
 Output: checkpoints/stage4_litm_finetune/
```

**Comprehensive Validation**
```python
[ ] NIAH benchmarks (verify 100% maintained)
[ ] LITM benchmarks (measure gap closure to SnapKV)
[ ] MultiDoc benchmarks (first time with Stage 4)
[ ] Generation quality tests
[ ] Attention drift metrics (with new query awareness)
```

### C.2 Phase 4 Expected Results

Based on PHASE4-PROPOSAL.md targets:

## V4 Benchmark Results (with Phase 4 Query-Aware Importance)

### NIAH
Expected: Maintain 100% at all budgets (no regression)

| Budget | V3 | V4 Target | SnapKV |
|--------|-----|-----------|--------|
| 25% | 100% | 100% | 33.3% |
| 50% | 100% | 100% | 66.7% |
| 75% | 100% | 100% | 66.7% |
| 100% | 100% | 100% | 100% |

### LITM (Critical Gap Closure)
Expected: Query awareness closes SnapKV gap significantly

| Budget | V3 | V4 Target | SnapKV | Gap |
|--------|-----|-----------|--------|--------|
| 25% | 33.3% | 35-40% | 33.3% | Close |
| 50% | 52.8% | **57-60%** | 55.6% | **Near-parity!** |
| 75% | 69.4% | **74-77%** | 79.4% | Narrow to <5pp |
| 100% | 100% | 100% | 100% | Equal |

**Success Definition**: V4 @ 50% within 1pp of SnapKV (parity achieved)

### MultiDoc (First Complete Results)
Expected: Phase 4 query awareness helps longer documents too

| Budget | V3 | V4 Target | SnapKV |
|--------|-----|-----------|--------|
| 25% | TBD | Match+5% | TBD |
| 50% | TBD | Match+10% | TBD |
| 75% | TBD | Match+15% | TBD |

### C.3 Risk Mitigation

If Phase 4 targets not met:

**Fallback 1: Hybrid TIS + SnapKV**
```python
# Combine both signals
combined_score = 0.6 * tis_importance + 0.4 * snapkv_pooled_attention
```

**Fallback 2: Multi-task Learning**
```python
# Train for both structural AND semantic importance simultaneously
loss = kl_loss + 0.1 * align_loss_structural + 0.1 * align_loss_semantic
```

**Fallback 3: Attention Ensemble**
```python
# Use multiple query-to-context attention heads
importance = mean([head.compute_importance() for head in attn_heads])
```

---

## PHASE D: Documentation & Release

### D.1 Technical Documentation Structure

**New Sections (vs V3)**:

```markdown
## 7. Attention Drift Analysis & Mitigation (NEW)
 7.1 Magnitude growth in long sequences
 7.2 Impact on importance-biased attention
 7.3 Post-norm solution
 7.4 Empirical validation on LITM
 7.5 Comparison with prior work (Eldenk et al. 2026)

## 8. Phase 4: Query-Aware Importance Learning (NEW)
 8.1 Query signal extraction (attention + DPR)
 8.2 QueryAwareImportanceHead architecture
 8.3 QueryAwareERTLoss design
 8.4 Training pipeline (Stage 4a, 4b)
 8.5 Phase 4 Results

## 9. Comprehensive Baseline Comparison (NEW)
 9.1 NIAH: TIS vs all 5 methods
 9.2 LITM: Gap analysis with SnapKV
 9.3 MultiDoc: First complete results
 9.4 Trade-off analysis (synthetic vs semantic)
 9.5 Hardware efficiency comparison

## 10. Ablation Studies (NEW)
 10.1 Importance embedding dimension (16 vs 32 vs 64 vs 256)
 10.2 Lambda (λ) sensitivity (0.0 to 0.5)
 10.3 Post-norm impact on drift
 10.4 Query signal type (attention vs DPR)
 10.5 LoRA rank effect on importance head

## 11. Implications & Future Work
 [Enhanced with Phase 4 insights]
```

### D.2 Comparison Tables for V4

**Table 1: Complete Baseline Comparison**
- 7 methods × 3 benchmarks × 4 budgets = 84 data points
- Highlight TIS performance profile
- Show SnapKV gap closure trajectory

**Table 2: Attention Drift Impact**
- Vanilla vs Vanilla+PostNorm vs TIS vs TIS+PostNorm
- Magnitude growth ratio
- Recent token attention bias
- LITM score impact

**Table 3: Phase 4 Ablations**
- Query signal type (attention vs DPR vs ensemble)
- LoRA rank (4, 8, 16, 32)
- Importance embedding dim (16, 32, 64, 256)
- Loss weight configurations

### D.3 Research Artifacts Package

```
$PROJECT_DIR/v4_research/
├── TECHNICAL-DOCUMENTATION.md # Complete system documentation
├── METHODOLOGY.md # Experimental procedures
├── RESULTS-ANALYSIS.md # Comprehensive results analysis
├── REPRODUCTION-GUIDE.md # Step-by-step execution guide
├── BASELINE-COMPARISON.md # How to run all 5 methods
├── CODE/
│ ├── scripts/train.py # All stages (1-4)
│ ├── scripts/eval.py # All benchmarks
│ ├── src/ # TIS components
│ └── configs/ # Config files for each stage
├── RESULTS/
│ ├── niah_all_methods.csv
│ ├── litm_all_methods.csv
│ ├── multidoc_all_methods.csv
│ └── comparison_analysis.md
├── MODELS/
│ ├── stage1_oracle/
│ ├── stage3_ert_v3/
│ ├── stage4_query_aware/
│ └── model_links.txt
└── ANALYSIS/
 ├── attention_drift_analysis/
 ├── training_dynamics/
 └── generation_quality_samples/
```

---

## Part 2: Detailed Execution Schedule

### Stage 1: Complete Baselines & MultiDoc

**Phase 1A: Baseline Testing**
```
Step 1: Set up comprehensive benchmark script
 - Verify all 5 methods load correctly
 - Test with 1-sample quick run on each
 
Step 2: Full NIAH × all baselines (32 samples)
 Duration: ~5-6 GPU-hours total
 
Step 3: Full LITM × all baselines (32 samples)
 Duration: ~5-6 GPU-hours total
```

**Phase 1B: MultiDoc Completion**
```
Step 4: MultiDoc × all baselines (32 samples × 4 budgets)
 Duration: ~6-8 GPU-hours total
 
Step 5: Generate comprehensive comparison tables
 - Create BASELINE-COMPARISON-V4.md
 - Identify any outliers for investigation
```

### Stage 2: Attention Drift Analysis & Solution

**Phase 2A: Analysis**
```
Step 1: Measure drift on vanilla Mistral
 - Run measure_attention_drift.py
 - Collect magnitude growth, attention patterns
 
Step 2: Implement and test post-norm solution
 - Add post-norm layers to transformer
 - Validate magnitude stabilization
```

**Phase 2B: Validation**
```
Step 3: Run LITM benchmarks with drift fix
 - Vanilla
 - Vanilla + PostNorm
 - TIS + PostNorm
 
Step 4: Analyze improvements and document findings
```

### Stage 3: Phase 4 Architecture

**Phase 3A: Implement Components**
```
Step 1: QueryAwareImportanceHead
 - Cross-attention mechanism
 - Score projection
 - Integration with existing head
 
Step 2: Query signal extraction
 - Attention-based: extract_query_importance_from_attention()
 - DPR-based: extract_query_importance_from_dpr()
 - Validation on sample queries
 
Step 3: QueryAwareERTLoss
 - KL divergence component
 - Importance alignment
 - Budget regularization
 - Unit tests
```

**Phase 3B: Integration**
```
Step 4: Update training scripts
 - Add Stage 4a configuration
 - Add Stage 4b configuration
 - Checkpoint management
 
Step 5: End-to-end test
 - Load Stage 3, add query-aware head
 - Run 1 step on small batch
 - Verify gradients flow correctly
```

### Stage 4: Phase 4 Training & Validation

**Phase 4A: Stage 4a Training**
```
Step 1: Launch Stage 4a training
 - NarrativeQA, 256 samples, 1 epoch
 - Monitor loss curves
 - Generate samples every 100 steps
 
Step 2: Continuous training phase
 - Check for divergence or stagnation
 - Verify no memorization (generation quality)
 
Step 3: Complete Stage 4a
 - Save checkpoint
 - Collect final metrics
```

**Phase 4B: Stage 4b Fine-tuning**
```
Step 4: LITM fine-tuning
 - Run with eval_during_training
 - Monitor LITM score improvement
 - Save best checkpoint
```

**Phase 4C: Comprehensive Validation**
```
Step 5: Run all benchmarks with Stage 4
 - NIAH (verify 100% maintained)
 - LITM (measure gap closure)
 - MultiDoc (first time with Stage 4)
 Duration: ~12 GPU-hours
 
Step 6: Analysis phase
 - Compare V3 vs V4 results
 - Calculate gap closure metrics
 - Identify any regressions
```

### Stage 5: Fine-tuning & Ablations

**Phase 5A: Ablation Studies**
```
Step 1: Query signal comparison
 - Attention-based vs DPR vs ensemble
 - Measure gap closure for each variant
 
Step 2: Hyperparameter sensitivity
 - LoRA rank: 4, 8, 16, 32
 - Importance embedding dim: 16, 32, 64, 256
 - Loss weights: sensitivity analysis
 
Step 3: Drift-aware improvements
 - Combine Phase 4 + post-norm
 - Measure combined benefit
```

**Phase 5B: Final Tuning**
```
Step 4: Run optimal configuration
 - Best query signal + hyperparams + drift fix
 - Final benchmark suite
 
Step 5: Prepare comprehensive results package
 - Generate all comparison tables
 - Verify result consistency
```

### Stage 6: Documentation & Release

**Phase 6A: Technical Documentation**
```
Write comprehensive technical documentation:
 - Attention drift analysis and solutions
 - Phase 4 query-aware learning details
 - Complete baseline comparison analysis
```

**Phase 6B: Results Analysis & Reporting**
```
Complete results analysis:
 - Integrate results with documentation
 - Verify all claims with data
 - Cross-validate all metrics
```

**Phase 6C: Archive & Release**
```
Prepare research artifacts for distribution:
 - Package code with full documentation
 - Archive model checkpoints
 - Verify reproducibility
 - Prepare for sharing/distribution
```

---

## Part 3: Resource Allocation

### Hardware
- **Primary**: RTX 5070 (8GB VRAM, 4-bit quantization)
- **Parallel Testing**: Can run multiple small benchmarks simultaneously with time-slicing
- **Estimated GPU-Hours**: ~80-100 GPU-hours total for complete system development

### Storage
- **Checkpoint Growth**: +500MB per stage (currently ~500MB for stage 3)
- **Benchmark CSVs**: ~10MB total for all results
- **Technical Artifacts**: ~50MB for supplementary materials

### Development Effort
- **Architecture & Integration**: ~120 hours engineering effort
- **Experiments & Validation**: ~80 hours GPU time + monitoring
- **Documentation**: ~40 hours technical writing
- **Total Effort**: ~240 hours (distributed across project phases)

---

## Part 4: Success Criteria

### Core Requirements (Must Have)
- [ ] All 5 baselines benchmarked consistently
- [ ] Complete MultiDoc results
- [ ] Phase 4 implementation working
- [ ] LITM gap narrowed by ≥5pp (to <5pp of SnapKV @ 50%)
- [ ] NIAH maintained at 100%
- [ ] No generation quality regression

### Extended Objectives (Aspirational)
- [ ] LITM @ 50% achieves parity with SnapKV (≤1pp gap)
- [ ] Attention drift solution reduces magnitude growth by 50%+
- [ ] MultiDoc shows consistent TIS advantage
- [ ] All ablations documented and explained
- [ ] Code is production-ready and well-documented

### Release & Distribution Criteria (All Must Have)
- [ ] Complete technical documentation (methodology, results, analysis)
- [ ] Comprehensive comparison tables (7 methods × 3 benchmarks)
- [ ] Code repository with full reproduction guide
- [ ] Pre-trained model checkpoints with documentation
- [ ] Supplementary analysis and ablation studies

---

## Conclusion

This master plan is **comprehensive and achievable** with systematic execution on consumer hardware. The complete system will have:

 **Complete coverage**: All 5 KV eviction methods thoroughly evaluated 
 **Attention drift solved**: Understanding + solution for speculative decoding 
 **Query-aware importance**: Phase 4 closes semantic understanding gap 
 **Reproducible**: Honest, comprehensive benchmarking with clear methodology 
 **Production-quality**: Code, checkpoints, detailed reproduction guides all documented 

**Next Action**: Begin Stage 1 with baseline testing script setup.

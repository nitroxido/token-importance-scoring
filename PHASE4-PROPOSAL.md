# Phase 4: Query-Aware Importance Learning Proposal

**Status**: Planned - Ready for implementation following V3 publication
**Priority**: High - Critical for closing LITM semantic gap
**Estimated Effort**: ~150-200 GPU-hours for research and implementation
**Expected Outcome**: Close SnapKV gap from -10 percentage points to near-parity

---

### Problem Statement

Current limitation (V3):
- NIAH (synthetic): TIS achieves superior performance relative to SnapKV (+66.7% at 25% budget)
- LITM (semantic): TIS underperforms SnapKV (-2.8% at 50%, -10% at 75%)

Root cause analysis: Current TIS importance learning (Stages 1-3) uses span-based pseudo-labels:
- Answer tokens → high importance (80)
- Supporting context → medium importance (60)
- Background/filler → low importance (10)

This approach succeeds for structural importance (position-invariant) but fails for semantic importance (query-dependent):

1. **Same token, different importance**: A person's name has high importance in query "Who is X?" but low importance in query "What is the color Y?"
2. **Span granularity**: Pseudo-labels operate at document-span level, not query-level
3. **Training signal mismatch**: Alignment loss doesn't capture query semantics
4. **SnapKV advantage**: Pools attention patterns from actual query tokens, providing direct query-importance signal

---

## Solution: Query-Aware Importance via Contrastive Training

### Architecture Extension

Architectural addition: Query-Importance Cross-Attention Head to ImportanceUpdateHead

```python
class QueryAwareImportanceHead(nn.Module):
 """
 Learn importance scores conditioned on query context.
 
 Instead of: scores = f(context_hidden_states)
 Now: scores = f(context_hidden_states, query_hidden_states)
 """
 
 def __init__(self, d_model: int):
 super().__init__()
 # Cross-attention: context as values, query as queries
 self.cross_attn = nn.MultiheadAttention(
 embed_dim=d_model,
 num_heads=8,
 batch_first=True,
 add_bias_kv=False,
 add_zero_attn=False,
 )
 
 # Importance prediction from cross-attended features
 self.score_proj = nn.Linear(d_model, 1)
 
 def forward(self, context_hidden, query_hidden):
 """
 Args:
 context_hidden: [B, T_ctx, D] - full context representations
 query_hidden: [B, T_query, D] - query token representations
 
 Returns:
 scores: [B, T_ctx] - importance scores for each context token
 relative to the query
 """
 # Query tokens attend to context tokens
 attended, _ = self.cross_attn(
 query=query_hidden, # [B, T_query, D]
 key=context_hidden, # [B, T_ctx, D]
 value=context_hidden, # [B, T_ctx, D]
 )
 
 # Pool query attention across all query positions
 # (which context tokens are relevant to ANY query position?)
 context_relevance = attended.mean(dim=1) # [B, T_ctx, D]
 
 # Project to scalar importance scores
 scores = self.score_proj(context_relevance).squeeze(-1) # [B, T_ctx]
 return torch.sigmoid(scores) * 100.0 # Scale to [0, 100]
```

### Training Objective: Query-Aware ERT

```python
class QueryAwareERTLoss(nn.Module):
 """
 ERT loss conditioned on query importance:
 
 Loss = KL(logits_full || logits_evicted_query_aware)
 + α · KL(importance_scores_true || importance_scores_predicted)
 
 Where:
 - importance_scores_true derived from query-context alignment analysis
 - importance_scores_predicted from QueryAwareImportanceHead
 """
 
 def forward(self, 
 logits_full,
 logits_evicted,
 scores_predicted,
 scores_target,
 budget=0.5):
 """
 Args:
 logits_full: [B, T, vocab] - full cache logits
 logits_evicted: [B, T, vocab] - evicted cache logits
 scores_predicted: [B, T] - predicted importance scores
 scores_target: [B, T] - query-derived target scores
 budget: cache compression budget
 """
 # ERT loss: maintain output distribution despite query-aware eviction
 kl_loss = F.kl_div(
 logits_evicted.log_softmax(-1),
 logits_full.softmax(-1),
 reduction='batchmean'
 )
 
 # Importance alignment loss (encourage match with query signals)
 importance_loss = F.mse_loss(scores_predicted, scores_target)
 
 # Budget regularization: optimize for specific budget
 mean_score = scores_predicted.mean()
 budget_target = budget * 100 # e.g., 50 for 50% budget
 budget_loss = (mean_score - budget_target) ** 2
 
 total_loss = kl_loss + 0.1 * importance_loss + 0.01 * budget_loss
 return total_loss, {"kl": kl_loss, "imp": importance_loss, "budget": budget_loss}
```

### 4.3 Query Signal Extraction

**Two approaches to derive target importance scores from queries:**

#### Approach A: Attention-Based (Recommended)
```python
def extract_query_importance_from_attention(
 context_hidden, # [B, T, D] - full context
 query_hidden, # [B, T_query, D] - just the query tokens
 query_positions, # [B] - indices where query ends
):
 """
 Score each context token by how much query-tokens attend to it.
 
 High score = query tokens pay lots of attention to this context token
 Low score = query tokens ignore this context token
 """
 B, T, D = context_hidden.shape
 scores = torch.zeros(B, T, device=context_hidden.device)
 
 for b in range(B):
 # Extract query attention from position query_positions[b] onwards
 # (In batch decoding, query is the last N tokens)
 q_start = query_positions[b]
 
 # Compute attention: query → context
 Q = query_hidden[b] # [T_query, D]
 K = context_hidden[b] # [T, D]
 
 # Attention logits
 attn_logits = (Q @ K.T) / (D ** 0.5) # [T_query, T]
 attn_weights = attn_logits.softmax(dim=-1) # [T_query, T]
 
 # Average attention across all query positions
 scores[b] = attn_weights.mean(dim=0) # [T]
 
 return scores * 100.0 # Scale to [0, 100]
```

#### Approach B: Dense Passage Retrieval (DPR)
```python
def extract_query_importance_from_dpr(
 context_hidden, # [B, T, D]
 query_embedding, # [B, D] - pre-computed query embedding (from separate model)
):
 """
 Score each context token by relevance to query using pre-trained DPR.
 
 - Use pre-trained DPR encoder for query
 - Score context tokens by dot-product with query embedding
 - Normalize and scale to [0, 100]
 """
 # Compute relevance scores
 scores = torch.bmm(context_hidden, query_embedding.unsqueeze(-1)) # [B, T, 1]
 scores = scores.squeeze(-1) # [B, T]
 
 # Normalize per-sequence to [0, 100]
 scores = (scores - scores.min(dim=1, keepdim=True)[0]) / (
 scores.max(dim=1, keepdim=True)[0] - scores.min(dim=1, keepdim=True)[0] + 1e-8
 )
 return scores * 100.0
```

---

## 4.4 Training Pipeline

### Stage 4a: Query-Aware Importance Learning

**Dataset**: NarrativeQA + extracted query importance signals

**Configuration**:
```python
{
 "stage": 4,
 "mode": "query_aware_ert",
 "query_signal": "attention", # or "dpr"
 "freeze_base": True, # Keep base model frozen
 "trainable_modules": [
 "importance_embedding",
 "query_aware_importance_head", # NEW
 "importance_attn_bias_hook"
 ],
 "epochs": 1,
 "batch_size": 1,
 "lr": 1e-4,
 "max_samples": 256, # Larger dataset (vs 128 for Stage 3)
 "loss_weights": {
 "kl": 1.0,
 "importance": 0.1,
 "budget_regularization": 0.01,
 }
}
```

### Stage 4b: Fine-tune on LITM-like tasks

**After Stage 4a convergence**: Fine-tune on actual LITM examples to ensure query-aware importance generalizes.

**Configuration**:
```python
{
 "stage": 4,
 "mode": "litm_finetune",
 "dataset": "litm_qna", # Actual LITM dataset
 "epochs": 2,
 "batch_size": 1,
 "eval_during_training": True,
 "eval_interval": 100, # Measure LITM accuracy every 100 steps
}
```

---

## 4.5 Expected Results

| Benchmark | V3 Current | V4 Target | SnapKV | Closing Rate |
|---|---|---|---|---|
| **NIAH @ 25%** | 100% | 100% | 33.3% | N/A (already dominates) |
| **NIAH @ 50%** | 100% | 100% | 66.7% | N/A (already dominates) |
| **LITM @ 50%** | 52.8% | **57-60%** | 55.6% | Close or beat gap |
| **LITM @ 75%** | 69.4% | **74-77%** | 79.4% | Narrow gap to <10pp |
| **MultiDoc @ 50%** | TBD | **Match or beat vanilla+10%** | TBD | Establish baseline |

**Success Criteria**:
- [ ] LITM @ 50%: ≥55% (within 0.6pp of SnapKV)
- [ ] LITM @ 75%: ≥77% (within 2.4pp of SnapKV)
- [ ] NIAH: Maintain 100% at all budgets
- [ ] MultiDoc: Show consistent improvements

---

## 4.6 Implementation Timeline

| Phase | Duration | Deliverables |
|---|---|---|
| **4a. Query-Aware Head Design** | 3-4 days | Architecture docs, prototype implementation |
| **4b. Training Infrastructure** | 3-4 days | Query signal extraction, loss functions, training loop |
| **4c. Initial Training Runs** | 5-7 days | Stage 4a convergence, loss curves, checkpoint validation |
| **4d. Fine-tuning on LITM** | 3-5 days | LITM-specific optimization, performance curves |
| **4e. Full Benchmarking** | 7-10 days | NIAH, LITM, MultiDoc with Stage 4 checkpoint |
| **4f. Paper Writing** | 3-5 days | ARXIV-DRAFT-V4 with Phase 4 results |
| **Total** | **~150-200 GPU-hours** | Publication-ready V4 with query-aware TIS |

---

## 4.7 Risk Mitigation

| Risk | Mitigation |
|---|---|
| Query signal is noisy | Use ensemble of both attention + DPR approaches; weight consensus |
| Over-fitting to LITM | Validate on NIAH and MultiDoc during training; use early stopping |
| Computational overhead | QueryAwareImportanceHead adds <1M params (negligible) |
| SnapKV gap not closing | Fall back to hybrid (TIS + SnapKV fusion) for Phase 4.5 |

---

## 4.8 Beyond Phase 4: Hybrid Approaches

If Phase 4 gap-closing is insufficient, consider:

1. **TIS + SnapKV Hybrid**: Use SnapKV pooled attention as additional training signal
2. **Multi-task Learning**: Train for both structural AND semantic importance
3. **Learned Importance Fusion**: Combine multiple importance signals (position, attention, semantic)

---

## Conclusion

Phase 4 addresses the core semantic limitation of Stages 1-3 by introducing query-aware importance learning. By using attention-based or DPR-based query signals, we directly encode which context tokens are relevant to the query, enabling TIS to close the LITM gap.

**Expected Outcome**: Move from "TIS dominates synthetic tasks, lags semantic tasks" to "TIS competitive or dominant across all task types."

**Publication Impact**: Phase 4 transforms TIS from a valid but limited baseline into a state-of-the-art method worthy of top-tier venues.

---

**Next Steps**:
1. Publish V3 with SnapKV comparison (shows honest assessment + improvement trajectory)
2. Implement Phase 4 architecture (3-5 days estimated)
3. Train and validate (5-10 days estimated)
4. Write V4 paper with Phase 4 results (3-5 days estimated)

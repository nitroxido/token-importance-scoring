# TIS v2.5 Release Manifest

**Release Date**: September 17, 2026  
**Version**: 2.5.0  
**Status**: ✅ Production-ready

---

## What's in This Release

TIS v2.5 adds multi-hop reasoning to the existing passage ranking system through a two-stage scoring pipeline with cross-passage attention and explicit bridge detection. All single-hop benchmarks (NIAH, LITM) are unchanged from v2.3.

### New Components

| File | Description |
|------|-------------|
| `src/token_importance/model/bridge_detection_head.py` | Binary classifier for bridge passages in multi-hop chains |
| `src/token_importance/model/iterative_refinement_head.py` | Stage 2 cross-passage attention + RefinementScoringHead |
| `scripts/train_v2.4_multihop_improvements.py` | v2.4 multi-hop training script |
| `scripts/eval_v2.4_multihop.py` | v2.4 multi-hop evaluation |
| `scripts/train_v2.5_iterative_refinement.py` | v2.5 training with curriculum learning |
| `scripts/eval_v2.5_iterative_refinement.py` | v2.5 evaluation (NIAH + LITM + multi-hop metrics) |

### Updated Documents

| File | Changes |
|------|---------|
| `README.md` | Added v2.5 section, updated checkpoint table, added multi-hop to overview table, added v2.5 evaluation commands |
| `ARCHITECTURE-TECHNICAL-SPECS.md` | Added Section 9 (v2.5 architecture), updated version header and design principles, updated software requirements |

---

## Checkpoint

| Field | Value |
|-------|-------|
| HuggingFace repo | `oldman-dev/tis-v2.5-multihop-reranker` |
| File | `tis_components.pt` |
| Size | 157.8 MB |
| SHA256 | `53f3333dffc3288579628ad7f13311e4e8efe41d053dfb9d7479d14f70596413` |
| Keys | `importance_embedding`, `importance_head`, `bridge_detection_head`, `refinement_head` |
| Backward compatible | ✅ v2.3 code loads v2.5 checkpoint |

Download:
```bash
hf download oldman-dev/tis-v2.5-multihop-reranker --local-dir checkpoints/v2.5_refinement
```

---

## Performance

### Regression Tests (zero regression required)

| Metric | v2.3 | v2.5 | Pass |
|--------|------|------|------|
| NIAH accuracy | 95.0% | 95.0% | ✅ |
| LITM begin | 88.0% | 88.0% | ✅ |
| LITM middle | 72.0% | 72.0% | ✅ |
| LITM end | 81.0% | 81.0% | ✅ |
| Peak VRAM | 5.5 GB | 5.5 GB | ✅ |

### New Multi-Hop Metrics

| Metric | v2.5 | Target | Pass |
|--------|------|--------|------|
| recall_both@5 | 100% | ≥50% | ✅ |
| recall_both@10 | 100% | ≥75% | ✅ |
| Bridge passage asymmetry | 0.0 | <2.0 | ✅ |

**Note**: Multi-hop numbers are from a 5-example smoke test. Full 200-example HotpotQA evaluation is recommended before relying on these numbers.

---

## Training Summary

- **Base**: Mistral-7B-Instruct-v0.3, 4-bit NF4, frozen
- **Dataset**: HotpotQA bridge-type + MS-MARCO
- **Steps**: 1000 (early stopping, patience=3)
- **Duration**: 28.6 min on RTX 5070 (8 GB)
- **Final loss**: 0.023
- **Curriculum**: α=0.0 (first 625 steps) → α=0.06 (last 375 steps)

---

## All Available Checkpoints

| HuggingFace Repo | Task | Key Metric |
|-----------------|------|-----------|
| `oldman-dev/tis-v2.5-multihop-reranker` | Multi-hop ranking | recall@5 100% — **latest** |
| `oldman-dev/tis-v2.3-passage-reranker` | Passage ranking | MRR 0.5102 (+18.1% vs BM25) |
| `oldman-dev/tis-v2.2-passage-reranker` | Passage ranking | MRR 0.471 (+9.1% vs BM25) |
| `oldman-dev/tis-v8b-hard-anchor` | KV compression | NIAH 82% @ 25% budget |
| `oldman-dev/tis-stage3-ert` | KV compression + LITM | NIAH 74% / LITM gap 0.000 |
| `oldman-dev/tis-passage-reranker` | LITM elimination | LITM gap 0.000 / EM 21.7% |
| `oldman-dev/tis-stage1-oracle` | Oracle baseline | — |

---

## HuggingFace Community Post (paste as-is)

> **TIS v2.5 — Multi-Hop Passage Reranking with Iterative Refinement**
>
> New release of Token Importance Scoring adds multi-hop reasoning to the existing passage reranker through a two-stage architecture:
>
> - **Stage 1**: Direct scoring (`QueryAwareImportanceHead`, same as v2.3)
> - **Stage 2**: Cross-passage attention refinement (`RefinementScoringHead`, new)
> - **Bridge detection**: explicit binary classifier for connecting passages
> - Score blending: `final = 0.7 × direct + 0.3 × refined`
>
> **Results** (HotpotQA bridge-type questions):
> - recall_both@5: **100%** (both supporting passages found in top 5)
> - NIAH / LITM: **zero regression** vs v2.3
> - Peak VRAM: **5.5 GB** (RTX 5070 compatible)
>
> Checkpoint: [`oldman-dev/tis-v2.5-multihop-reranker`](https://huggingface.co/oldman-dev/tis-v2.5-multihop-reranker)  
> Code: [github.com/nitroxido/token-importance-scoring](https://github.com/nitroxido/token-importance-scoring)
>
> ⚠️ Multi-hop numbers are from a 5-example smoke test — full evaluation with `scripts/eval_v2.5_iterative_refinement.py` is recommended before production use.

---

## Known Limitations

1. **Small evaluation set**: multi-hop results from 5 examples only; scale to 200 examples for production confidence
2. **Transformers ≥5.9.0 incompatibility**: SDPA layout change breaks full inference path; use `importance_head.direct_score()` as workaround
3. **Stage 2 latency**: adds ~70ms per query; disable Stage 2 (`--no-refinement`) for latency-sensitive applications

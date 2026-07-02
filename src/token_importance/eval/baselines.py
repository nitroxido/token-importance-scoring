"""Baseline KV cache eviction policies for fair comparison with TIS.

Implements:
  - H2OEvictionPolicy              (Heavy Hitter Oracle, Zhang et al. 2023)
  - StreamingLLMEvictionPolicy     (Xiao et al. 2023)
  - SnapKVEvictionPolicy           (Li et al. 2024)
  - InfiniAttentionEvictionPolicy  (Munkhdalai et al. 2024 — approximation)
"""
from __future__ import annotations

import numpy as np
import torch


class H2OEvictionPolicy:
    """Heavy Hitter Oracle (Zhang et al. 2023).

    Keeps the top-K tokens by cumulative attention magnitude, plus the protected
    sink tokens at the start and the most recent tokens at the end.
    Everything else is evicted to meet the target length.
    """

    def select_indices_to_evict(
        self,
        attention_magnitudes: torch.Tensor,  # float32 [seq_len]
        current_len: int,
        target_len: int,
        n_sink: int = 4,
        n_recent: int = 64,
    ) -> torch.Tensor:  # int64 indices to evict
        """Return the indices of tokens to remove from the KV cache.

        Protected tokens (never evicted):
          - First ``n_sink`` positions (attention sinks).
          - Last ``n_recent`` positions (recency window).

        Among the remaining *candidate* positions, those with the **lowest**
        cumulative attention magnitude are selected for eviction until the
        sequence fits within ``target_len``.

        Returns an empty tensor when no eviction is needed or when there are
        no eligible candidates.
        """
        if current_len <= target_len:
            return torch.tensor([], dtype=torch.long)

        n_to_evict = current_len - target_len

        # Clamp protection window so it doesn't exceed the sequence.
        n_sink = min(n_sink, current_len)
        n_recent_actual = min(n_recent, max(0, current_len - n_sink))

        # Candidate range: [n_sink, current_len - n_recent_actual)
        cand_start = n_sink
        cand_end = current_len - n_recent_actual

        if cand_start >= cand_end:
            # No candidates available — nothing to evict.
            return torch.tensor([], dtype=torch.long)

        candidates = torch.arange(cand_start, cand_end, dtype=torch.long)
        cand_magnitudes = attention_magnitudes[cand_start:cand_end]

        n_to_evict = min(n_to_evict, len(candidates))
        # Evict candidates with the *lowest* attention magnitude.
        _, evict_local = torch.topk(cand_magnitudes, n_to_evict, largest=False)
        evict_global = candidates[evict_local]
        return evict_global.sort().values


class StreamingLLMEvictionPolicy:
    """StreamingLLM (Xiao et al. 2023).

    Keeps the first ``n_sink`` tokens (attention sinks) and the most recent
    ``n_recent`` tokens.  Every token in between is eligible for eviction.
    """

    def select_indices_to_evict(
        self,
        current_len: int,
        target_len: int,
        n_sink: int = 4,
        n_recent: int = 64,
    ) -> torch.Tensor:  # int64 indices to evict
        """Return middle indices to remove.

        Evicts from the range ``[n_sink, current_len - n_recent)`` until the
        cache reaches ``target_len``.  Returns an empty tensor when there is
        nothing to evict.
        """
        if current_len <= target_len:
            return torch.tensor([], dtype=torch.long)

        n_sink = min(n_sink, current_len)
        n_recent_actual = min(n_recent, max(0, current_len - n_sink))

        cand_start = n_sink
        cand_end = current_len - n_recent_actual

        if cand_start >= cand_end:
            return torch.tensor([], dtype=torch.long)

        n_to_evict = min(current_len - target_len, cand_end - cand_start)
        # Evict the *oldest* middle tokens first (lowest indices).
        evict = torch.arange(cand_start, cand_start + n_to_evict, dtype=torch.long)
        return evict


class SnapKVEvictionPolicy:
    """SnapKV (Li et al. 2024).

    Pools attention from the last ``n_query`` tokens (the "observation window",
    i.e. the instruction/question) over the preceding context to score each
    context token.  Per-head attention is summed then averaged across layers.

    This provides *query-aware* eviction: context tokens that the question
    attends to strongly are kept even if their cumulative attention is low.

    Reference: "SnapKV: LLM Knows What You Are Looking for Before Generation"
    (Li et al. 2024, arXiv:2404.14469)
    """

    def compute_scores(
        self,
        attention_tensors: list[torch.Tensor],  # list of [B, H, T, T] per layer
        n_query: int = 64,
        T: int | None = None,  # sequence length (only used if attention_tensors empty)
    ) -> torch.Tensor:  # float32 [T]
        """Compute per-token importance by pooling attention from the query window.

        The observation window is the last ``n_query`` rows of each attention
        matrix.  Scores are accumulated across all layers and heads.
        If attention tensors unavailable (FlashAttention, etc.), returns uniform scores.
        """
        if not attention_tensors:
            if T is None:
                T = 1  # fallback: single element
            return torch.ones(T, dtype=torch.float32)  # uniform

        T = attention_tensors[0].shape[-1]
        scores = torch.zeros(T, dtype=torch.float32)

        for layer_attn in attention_tensors:  # [B, H, T, T]
            attn = layer_attn[0]              # [H, T, T]  (drop batch dim)
            query_start = max(0, T - n_query)
            # Rows = query positions; cols = key (context) positions.
            query_attn = attn[:, query_start:, :]   # [H, n_q, T]
            scores += query_attn.sum(dim=(0, 1)).cpu()

        return scores

    def select_indices_to_keep(
        self,
        scores: torch.Tensor,   # float32 [T] from compute_scores()
        current_len: int,
        target_len: int,
        n_sink: int = 4,
        n_recent: int = 64,
    ) -> np.ndarray:             # int indices to KEEP (sorted)
        """Return indices to KEEP: protected slots + top-K by query-pooled score."""
        protected: set[int] = set(range(min(n_sink, current_len)))
        protected |= set(range(max(0, current_len - n_recent), current_len))

        n_from_candidates = max(0, target_len - len(protected))
        candidates = [i for i in range(current_len) if i not in protected]

        if candidates and n_from_candidates > 0:
            cand_scores = scores[candidates]
            n_keep = min(n_from_candidates, len(candidates))
            _, top_local = cand_scores.topk(n_keep)
            top_global = {candidates[i.item()] for i in top_local}
        else:
            top_global = set()

        keep = sorted(protected | top_global)
        return np.array(keep[:target_len], dtype=np.intp)


class InfiniAttentionEvictionPolicy:
    """Infini-Attention benchmark approximation (Munkhdalai et al. 2024).

    True Infini-Attention uses a delta-rule compressive memory matrix alongside
    a local attention window, requiring model-level architectural changes.
    This class implements a **pre-eviction approximation** suitable for our
    benchmark setup, capturing the core philosophy:

        "Some tokens are kept verbatim (local window); the rest of the context
        is 'compressed' — approximated here as attention-weighted sampling."

    Concretely:
    - Keep ``n_sink`` attention sinks (structural requirement).
    - Keep ``n_recent`` recent tokens (local attention window).
    - From the remaining prefix: sample ``budget - n_sink - n_recent`` tokens
      with probability proportional to their cumulative attention magnitude
      (higher-attention tokens are more likely to be "recoverable" from
      compressed memory).

    Results are reported as "Infini-Attn (approx)" to be transparent about
    what is and isn't being measured.  Full Infini-Attention would require
    retraining with the compressive memory module.
    """

    def select_indices_to_keep(
        self,
        attention_magnitudes: torch.Tensor,  # float32 [T] cumulative per token
        current_len: int,
        target_len: int,
        n_sink: int = 4,
        n_recent: int = 64,
        seed: int = 42,
    ) -> np.ndarray:             # int indices to KEEP (sorted)
        """Keep sinks + recent + attention-weighted sample from compressed prefix."""
        protected: set[int] = set(range(min(n_sink, current_len)))
        protected |= set(range(max(0, current_len - n_recent), current_len))

        n_from_compressed = max(0, target_len - len(protected))
        candidates = [i for i in range(current_len) if i not in protected]

        if not candidates or n_from_compressed == 0:
            return np.array(sorted(protected)[:target_len], dtype=np.intp)

        cand_mags = attention_magnitudes[candidates].float().numpy()
        cand_mags = cand_mags - cand_mags.min()   # shift to non-negative

        # Add small epsilon so every candidate has non-zero sampling probability.
        # This allows replace=False sampling to work even when most magnitudes are 0.
        epsilon = 1e-6 * (cand_mags.max() + 1.0)
        cand_mags = cand_mags + epsilon
        weights = cand_mags / cand_mags.sum()

        rng = np.random.RandomState(seed)
        n_sample = min(n_from_compressed, len(candidates))

        if n_sample >= len(candidates):
            # Budget covers all candidates — just take all of them.
            sampled = set(candidates)
        else:
            sampled_local = rng.choice(len(candidates), size=n_sample, replace=False, p=weights)
            sampled = {candidates[i] for i in sampled_local}

        keep = sorted(protected | sampled)
        return np.array(keep[:target_len], dtype=np.intp)


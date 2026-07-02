"""EvictionPolicy — composite-score KV cache eviction for TIS."""
from __future__ import annotations

import torch
from token_importance.config import TISConfig


class EvictionPolicy:
    """Decides which tokens to evict from the KV cache using a composite score.

    Composite score (higher = more important = keep):
        score_i = α·user_i + β·model_i + γ·attn_i + δ·recency_i
        where recency_i = i / max(seq_len - 1, 1)   (0 = oldest, 1 = newest)

    Protected slots (never evicted):
      - First config.N_sink tokens
      - Last config.N_recent tokens
      - Any token with raw_user_score >= 90
    """

    def __init__(self, config: TISConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_eviction_scores(
        self,
        user_scores: torch.Tensor,           # float32 [seq_len], normalized [0, 1]
        model_scores: torch.Tensor,          # float32 [seq_len], normalized [0, 1]
        attention_magnitudes: torch.Tensor,  # float32 [seq_len], normalized [0, 1]
        seq_len: int,
    ) -> torch.Tensor:                       # float32 [seq_len]
        """Compute composite importance score for each token (higher = keep)."""
        cfg = self.config
        recency = torch.arange(seq_len, dtype=torch.float32) / max(seq_len - 1, 1)
        return (
            cfg.alpha   * user_scores
            + cfg.beta  * model_scores
            + cfg.gamma * attention_magnitudes
            + cfg.delta_w * recency
        )

    def select_indices_to_evict(
        self,
        eviction_scores: torch.Tensor,   # float32 [seq_len]
        raw_user_scores: torch.Tensor,   # uint8  [seq_len], unscaled 0–100
        current_len: int,
        target_len: int,
    ) -> torch.Tensor:                   # int64 indices to remove
        """Return indices of the lowest-scoring unprotected tokens to evict.

        The number of evicted tokens = current_len - target_len, unless there
        are not enough unprotected tokens — in that case evict as many as possible
        without raising an error.
        """
        if current_len <= target_len:
            return torch.tensor([], dtype=torch.long)

        n_to_evict = current_len - target_len
        cfg = self.config

        # Build protection mask (True = protected, never evict)
        protected = torch.zeros(current_len, dtype=torch.bool)
        # Attention sinks
        protected[: cfg.N_sink] = True
        # Recent tokens
        if cfg.N_recent > 0:
            protected[max(0, current_len - cfg.N_recent) :] = True
        # High user-importance tokens
        protected[raw_user_scores >= 90] = True

        # Candidate indices (unprotected)
        candidate_indices = torch.where(~protected)[0]  # int64

        if candidate_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long)

        # Sort candidates by composite score (ascending → lowest first)
        candidate_scores = eviction_scores[candidate_indices]
        sorted_order = torch.argsort(candidate_scores)  # ascending

        # Evict the lowest-scoring candidates up to n_to_evict
        n_actual = min(n_to_evict, candidate_indices.numel())
        evict_indices = candidate_indices[sorted_order[:n_actual]]
        return evict_indices.to(torch.long)

    def should_evict(self, current_len: int) -> bool:
        """Return True if current_len >= config.max_cache_tokens."""
        return current_len >= self.config.max_cache_tokens

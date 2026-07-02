"""TISConfig — central hyperparameter dataclass for Token Importance Score."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TISConfig:
    """All hyperparameters for the TIS system.

    Eviction composite score weights must sum to 1.0:
        alpha + beta + gamma + delta_w == 1.0
    """

    # ImportanceEmbedding hidden dimension
    d_imp: int = 16

    # Initial attention-bias scale λ; clamped to [0.0, 0.5] during use
    lambda_init: float = 0.0

    # Maximum score change per ImportanceUpdateHead step
    max_delta: int = 20

    # --- Eviction composite-score weights (must sum to 1.0) ---
    alpha: float = 0.4    # user-assigned score weight
    beta: float = 0.3     # model-updated score weight
    gamma: float = 0.2    # attention magnitude weight
    delta_w: float = 0.1  # recency weight (renamed to avoid shadowing built-in delta)

    # --- Protected-slot sizes (never evicted) ---
    N_sink: int = 4       # leading attention-sink tokens
    N_recent: int = 64    # trailing recent tokens

    # --- Cache-size thresholds ---
    max_cache_tokens: int = 4096    # eviction trigger
    target_cache_tokens: int = 3072 # post-eviction target

    # Run ImportanceUpdateHead every K generation steps
    update_frequency_K: int = 1

    def __post_init__(self) -> None:
        if abs(self.alpha + self.beta + self.gamma + self.delta_w - 1.0) > 1e-6:
            raise ValueError(
                "Eviction weights must sum to 1.0 "
                f"(got alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, delta_w={self.delta_w}, "
                f"sum={self.alpha + self.beta + self.gamma + self.delta_w})"
            )
        if not (0.0 <= self.lambda_init <= 0.5):
            raise ValueError(
                f"lambda_init must be in [0.0, 0.5] (got {self.lambda_init})"
            )
        if self.N_sink < 1:
            raise ValueError(f"N_sink must be >= 1 (got {self.N_sink})")
        if self.target_cache_tokens >= self.max_cache_tokens:
            raise ValueError(
                f"target_cache_tokens ({self.target_cache_tokens}) must be "
                f"< max_cache_tokens ({self.max_cache_tokens})"
            )

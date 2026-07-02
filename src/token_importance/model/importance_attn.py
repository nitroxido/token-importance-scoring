"""ImportanceAttnBiasHook — importance-proportional additive attention bias."""
from __future__ import annotations

import torch
import torch.nn as nn
from token_importance.config import TISConfig


class ImportanceAttnBiasHook:
    """Generates and merges an importance-proportional additive attention bias.

    The bias has shape [1, 1, 1, seq_len]: every query position is nudged to
    attend more to high-importance key positions.  λ is a learned scalar clamped
    to [0.0, 0.5] during forward use (the raw parameter is unclamped for gradients).
    """

    def __init__(self, config: TISConfig) -> None:
        self.config = config
        self._lambda = nn.Parameter(torch.tensor(float(config.lambda_init)))

    @property
    def lambda_value(self) -> float:
        return float(self._lambda.clamp(0.0, 0.5).item())

    def compute_bias(self, scores_normalized: torch.Tensor) -> torch.Tensor:
        """scores_normalized: float32 [seq_len].
        Returns float32 [1, 1, 1, seq_len]."""
        lam = self._lambda.clamp(0.0, 0.5)
        bias = lam * scores_normalized          # [seq_len]
        return bias.view(1, 1, 1, -1)           # [1, 1, 1, seq_len]

    def merge_into_mask(
        self,
        attention_mask: torch.Tensor | None,
        scores_normalized: torch.Tensor,
        device: torch.device,
        target_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a 4D float attention mask with the importance bias added.

        Args:
            attention_mask: Optional 2D or 4D attention mask
            scores_normalized: Normalized importance scores [0, 1]
            device: Device to place tensors on
            target_dtype: Target dtype for output (important for 4-bit quantization)

        Cases:
          - attention_mask is None: return bias tensor [1, 1, 1, seq_len] only.
          - attention_mask is 2D [batch, seq]: convert to 4D additive mask
            (0 for keep, -inf for mask) then add bias.
          - attention_mask is already 4D float: add bias directly.
        """
        bias = self.compute_bias(scores_normalized.to(device))  # [1, 1, 1, seq]
        bias = bias.to(dtype=target_dtype)  # Match target dtype (e.g. float16 for 4-bit)

        if attention_mask is None:
            return bias

        if attention_mask.dim() == 2:
            # [batch, seq] bool or 0/1 int → 4D additive float mask
            # Convention: 1 = keep, 0 = mask
            additive = attention_mask.to(dtype=target_dtype)
            # Map 1→0.0 (keep), 0→-inf (mask out)
            min_val = torch.tensor(torch.finfo(target_dtype).min, dtype=target_dtype, device=attention_mask.device)
            additive = (1.0 - additive) * min_val
            additive = additive[:, None, None, :]   # [batch, 1, 1, seq]
            return additive + bias

        # Already 4D float - ensure matching dtype
        return attention_mask.to(dtype=target_dtype) + bias

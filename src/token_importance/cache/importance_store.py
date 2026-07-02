"""ImportanceStore — per-token importance scores parallel to the KV cache."""
from __future__ import annotations

import torch


class ImportanceStore:
    """Holds per-token importance scores (uint8, 0–100) parallel to the KV cache.

    Internal storage is always a 1-D ``torch.uint8`` tensor on CPU.
    """

    def __init__(self, initial_scores: torch.Tensor | None = None) -> None:
        if initial_scores is None:
            self._scores = torch.empty(0, dtype=torch.uint8)
        else:
            self._scores = initial_scores.to(dtype=torch.uint8, device="cpu").clone()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._scores.numel()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def get_scores(self) -> torch.Tensor:
        """Return uint8 tensor [seq_len]. Detached read-only copy."""
        return self._scores.detach().clone()

    def get_scores_normalized(self) -> torch.Tensor:
        """Return float32 tensor [seq_len] in [0, 1]: scores / 100.0."""
        return self._scores.to(torch.float32) / 100.0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, scores: torch.Tensor) -> None:
        """Append new token scores. scores: uint8 1D tensor [n_new]."""
        new = scores.to(dtype=torch.uint8, device="cpu")
        self._scores = torch.cat([self._scores, new])

    def update(self, positions: torch.Tensor, deltas: torch.Tensor) -> None:
        """Apply signed integer deltas at given positions.

        positions: int64 [N]
        deltas:    int32 or float32 [N]

        Float deltas are rounded to the nearest integer before addition.
        Result is clamped to [0, 100].
        """
        if positions.numel() == 0:
            return

        positions = positions.to(torch.long)
        if deltas.is_floating_point():
            int_deltas = deltas.round().to(torch.int32)
        else:
            int_deltas = deltas.to(torch.int32)

        current = self._scores[positions].to(torch.int32)
        updated = (current + int_deltas).clamp(0, 100).to(torch.uint8)
        self._scores[positions] = updated

    def evict(self, indices_to_remove: torch.Tensor) -> None:
        """Remove tokens at given int64 indices. Preserves order of remaining tokens."""
        if indices_to_remove.numel() == 0:
            return
        if self._scores.numel() == 0:
            return

        keep_mask = torch.ones(len(self), dtype=torch.bool)
        keep_mask[indices_to_remove.to(torch.long)] = False
        self._scores = self._scores[keep_mask]

    def reset(self) -> None:
        """Clear all stored scores."""
        self._scores = torch.empty(0, dtype=torch.uint8)

    def clone(self) -> "ImportanceStore":
        """Return a deep copy. Modifying the clone must not affect the original."""
        new_store = ImportanceStore.__new__(ImportanceStore)
        new_store._scores = self._scores.clone()
        return new_store

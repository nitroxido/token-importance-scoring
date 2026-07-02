"""ImportanceEmbedding — maps integer scores 0–100 to d_model-dimensional vectors."""
from __future__ import annotations

import warnings
import torch
import torch.nn as nn


class ImportanceEmbedding(nn.Module):
    """Maps integer importance scores (0–100) to d_model-dimensional vectors.

    Zero-initialized weights so the base pretrained model is unaffected at
    fine-tuning start. After training, the embedding adds importance-proportional
    signal to each token's residual stream.

    Usage:
        emb = ImportanceEmbedding(d_imp=16, d_model=768)
        scores = torch.tensor([50, 80, 20], dtype=torch.long)
        delta = emb(scores)              # [3, 768], all zeros before training
        h = token_embeddings + delta
    """

    def __init__(self, d_imp: int = 16, d_model: int = 768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(101, d_imp)   # scores 0–100 inclusive
        self.proj = nn.Linear(d_imp, d_model, bias=False)
        nn.init.zeros_(self.embedding.weight)
        nn.init.zeros_(self.proj.weight)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """scores: integer tensor of any shape. Returns float32 [..., d_model]."""
        scores = scores.to(torch.long)

        if (scores < 0).any() or (scores > 100).any():
            warnings.warn(
                "ImportanceEmbedding received scores outside [0, 100]; "
                "clamping to valid range.",
                stacklevel=2,
            )
            scores = scores.clamp(0, 100)

        return self.proj(self.embedding(scores))

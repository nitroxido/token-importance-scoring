"""TIS-Aware Lightweight Drafter — Phase 5b.

A single-layer transformer that takes Mistral-7B's last hidden states and
predicts the next token (EAGLE-1 style).  Coupled with DrafterImportanceAttnBias
so that importance scores from the TIS ImportanceStore bias its self-attention,
reducing attention drift during speculation.

Architecture (EAGLE-1 style without EAGLE-3's multi-layer feature fusion):
    input  = target_hidden [-1, :, d_model]  +  embedded prev token
    layer  = single causal MHA + FFN  (depth-scaled importance bias in MHA)
    output = LM head logits   [vocab_size]

This module is imported by train_drafter_tis_aware.py and eval_drafter.py.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

from token_importance.model.drafter_attn_bias import DrafterImportanceAttnBias


class _CausalMHA(nn.Module):
    """Minimal multi-head causal attention with optional importance bias."""

    def __init__(self, d_model: int, n_heads: int, bias_module: DrafterImportanceAttnBias) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.bias_module = bias_module

    def forward(
        self,
        x: torch.Tensor,            # [B, T, d]
        importance_scores: Optional[torch.Tensor] = None,  # [T] in [0,100]
        drafter_step: int = 0,
    ) -> torch.Tensor:
        B, T, d = x.shape
        dtype = x.dtype
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)                    # each [B, T, H, dh]
        q = q.transpose(1, 2).float()              # always compute attn in float
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, T]

        # Causal mask
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal

        # TIS importance bias
        if importance_scores is not None:
            scores = self.bias_module(
                scores,
                importance_scores.float().to(x.device),
                drafter_step=drafter_step,
            )

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                  # [B, H, T, dh] float
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.out(out.to(dtype)).to(dtype), attn


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TISDrafter(nn.Module):
    """Lightweight TIS-aware speculative drafter.

    Memory-efficient design for 8 GB VRAM:
      - Projects target hidden states DOWN to a small bottleneck dim (d_bottleneck)
        to keep parameter count tiny.
      - Applies self-attention with depth-scaled importance bias in bottleneck space.
      - Projects back to d_model.
      - The LM head (d_model → vocab) is NOT part of this module; call the frozen
        target model's lm_head externally for vocabulary logits.

    Trainable parameters (at d_bottleneck=256):
        input_proj  : 4096 × 256    =   1.0 M
        qkv         : 3 × 256 × 256 = 196 K
        out         : 256 × 256     =  64 K
        ffn         : 2 × 256 × 512 = 262 K
        output_proj : 256 × 4096    =   1.0 M
        bias params : 2 scalars     ≈ 0
        Total       ≈ 2.5 M params  → ~5 MB in bfloat16 — fits easily alongside 4-bit Mistral

    Usage:
        # During speculation step k:
        delta_h, attn = drafter(target_hidden, imp_scores, drafter_step=k)
        corrected_hidden = target_hidden + delta_h
        logits = target_lm_head(target_norm(corrected_hidden))

    Scorer path name: ``tis_drafter_depth_scaled``
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_bottleneck: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        lambda_d_base: float = 0.1,
        lambda_d_slope: float = 0.05,
        lambda_d_max: float = 1.0,
        vocab_size: int = 0,   # kept for API compatibility, not used
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck

        self.attn_bias = DrafterImportanceAttnBias(
            d_model=d_bottleneck,
            lambda_d_base=lambda_d_base,
            lambda_d_slope=lambda_d_slope,
            lambda_d_max=lambda_d_max,
        )

        self.input_proj = nn.Linear(d_model, d_bottleneck, bias=False)
        self.norm1 = nn.RMSNorm(d_bottleneck)
        self.mha = _CausalMHA(d_bottleneck, n_heads, bias_module=self.attn_bias)
        self.norm2 = nn.RMSNorm(d_bottleneck)
        self.ffn = _FFN(d_bottleneck, d_ff)
        self.output_proj = nn.Linear(d_bottleneck, d_model, bias=False)

        # Output projection starts near zero so initial delta is small
        nn.init.normal_(self.output_proj.weight, std=0.01)

    def forward(
        self,
        target_hidden: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None,
        drafter_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_hidden: [B, T, d_model] last-layer hidden states (frozen)
            importance_scores: [T] in [0,100] from TIS ImportanceStore
            drafter_step: speculation depth k for depth-scaled lambda

        Returns:
            delta_hidden: [B, T, d_model] additive correction to target hidden states
            attn_weights: [B, n_heads, T, T] for drift analysis
        """
        x = self.input_proj(target_hidden.detach())   # [B, T, d_bot]
        h, attn_w = self.mha(self.norm1(x), importance_scores, drafter_step)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        delta = self.output_proj(x)                   # [B, T, d_model]
        return delta, attn_w

    # ------------------------------------------------------------------
    # Importance attention fraction (for drift measurement)
    # ------------------------------------------------------------------

    @staticmethod
    def attention_to_important_tokens(
        attn_weights: torch.Tensor,       # [B, H, T, T]
        importance_scores: torch.Tensor,  # [T] in [0,100]
        threshold: float = 70.0,
    ) -> float:
        """Fraction of attention mass directed at high-importance tokens.

        Used to quantify attention drift: a lower value at larger speculation
        depth k means the drafter is drifting away from important context.
        """
        T = importance_scores.shape[0]
        imp_mask = (importance_scores >= threshold).float()  # [T]
        # Average over batch and heads, sum over key dimension
        attn_mean = attn_weights.mean(dim=(0, 1))   # [T_query, T_key]
        # Fraction of attention going to high-importance key tokens
        imp_attn = (attn_mean * imp_mask.unsqueeze(0)).sum(dim=-1).mean()
        return imp_attn.item()

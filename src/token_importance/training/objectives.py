"""TIS training objectives.

Three composable losses:
  lm_loss                    Standard next-token cross-entropy
  importance_alignment_loss  Nudges ImportanceUpdateHead toward attention-magnitude targets
  eviction_robustness_loss   KL divergence between full-cache and evicted-cache outputs

Composed objectives:
  TISLoss   — Stage 1: L = w_lm * L_lm + w_align * L_align
              Stage 2 relic: adds w_robust * L_robust (kept for research reference)
  ERTLoss   — Stage 3 (ERT): L = KL(full || evicted) + w_align * L_align
              No language-modeling term; directly optimises eviction quality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def lm_loss(
    logits: torch.Tensor,   # [B, seq, vocab]
    labels: torch.Tensor,   # [B, seq]  — -100 for masked positions
) -> torch.Tensor:
    """Standard next-token cross-entropy.  Masked positions (label = -100) are
    excluded from the average.  Returns a scalar tensor.

    Matches ``F.cross_entropy(..., ignore_index=-100)`` exactly.
    """
    # Flatten to [B*seq, vocab] and [B*seq] for F.cross_entropy
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def importance_alignment_loss(
    predicted_deltas: torch.Tensor,       # [B, T, 1] raw output of ImportanceUpdateHead
    attention_magnitudes: torch.Tensor,   # [B, T]    normalised to [0, 1]
    importance_scores_norm: torch.Tensor, # [B, T]    normalised to [0, 1]
) -> torch.Tensor:
    """MSE loss between predicted score deltas and attention-magnitude targets.

    Rationale: a token that the model attends to more than its current score
    suggests should receive a *positive* delta; the reverse for under-attended
    tokens.

    Target formula:
        target_delta_i = atanh( clamp(attn_mag_i − score_i, −0.99, 0.99) )

    The target lives in the same space as ``tanh(raw_delta) * max_delta``, but
    here we operate on the *raw* delta before tanh so that gradients flow
    freely and the head can learn to push scores toward the attention signal.
    """
    diff   = attention_magnitudes - importance_scores_norm          # [B, T]
    target = torch.atanh(diff.clamp(-0.99, 0.99))                   # [B, T]
    pred   = predicted_deltas.squeeze(-1)                           # [B, T]
    return F.mse_loss(pred, target)


def eviction_robustness_loss(
    logits_full:    torch.Tensor,  # [B, seq, vocab]  full-cache forward pass
    logits_evicted: torch.Tensor,  # [B, seq, vocab]  evicted-cache forward pass
    labels:         torch.Tensor,  # [B, seq]  — -100 for masked positions
) -> torch.Tensor:
    """Mean KL( full_distribution ‖ evicted_distribution ) at non-masked positions.

    Measures how much the output distribution shifts when tokens are evicted.
    Minimising this encourages the model to maintain the same output even after
    eviction — i.e., it learns to be robust to its own eviction decisions.

    Returns 0.0 (no gradient) when there are no non-masked positions.
    """
    mask = (labels != -100)  # [B, seq]
    if not mask.any():
        return torch.tensor(0.0, device=logits_full.device, requires_grad=False)

    # Gather non-masked positions
    log_p_full    = F.log_softmax(logits_full[mask],    dim=-1)  # [N, vocab]
    log_p_evicted = F.log_softmax(logits_evicted[mask], dim=-1)  # [N, vocab]

    # KL(full ‖ evicted) = sum_v  p_full * (log p_full − log p_evicted)
    p_full = log_p_full.exp()
    kl = (p_full * (log_p_full - log_p_evicted)).sum(dim=-1)     # [N]
    return kl.mean()


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class TISLoss(nn.Module):
    """Weighted combination of the three TIS training objectives.

    L_total = L_lm  +  w_align * L_align  +  w_robust * L_robust

    Set ``weight_robustness=0.0`` (the default) for Stage 1 training where
    eviction is not yet simulated.  Increase to ~0.05 for Stage 2.

    Args:
        weight_alignment:   Weight for the importance-alignment loss.
        weight_robustness:  Weight for the eviction-robustness loss.

    ``forward`` input dict keys:
        Required:
          logits                  [B, seq, vocab]
          labels                  [B, seq]
          predicted_deltas        [B, T, 1]
          attention_magnitudes    [B, T]  normalised
          importance_scores_norm  [B, T]  normalised

        Optional (needed for robustness loss):
          logits_evicted          [B, seq, vocab]

    Returns a dict with scalar tensors: ``total``, ``lm``, ``alignment``,
    ``robustness``.
    """

    def __init__(
        self,
        weight_alignment:  float = 0.1,
        weight_robustness: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight_alignment  = weight_alignment
        self.weight_robustness = weight_robustness

    def forward(self, outputs: dict) -> dict[str, torch.Tensor]:
        logits   = outputs["logits"]
        labels   = outputs["labels"]
        device   = logits.device

        l_lm = lm_loss(logits, labels)

        l_align = importance_alignment_loss(
            outputs["predicted_deltas"],
            outputs["attention_magnitudes"],
            outputs["importance_scores_norm"],
        )

        if self.weight_robustness > 0.0 and "logits_evicted" in outputs:
            l_robust = eviction_robustness_loss(
                logits,
                outputs["logits_evicted"],
                labels,
            )
        else:
            l_robust = torch.tensor(0.0, device=device)

        total = l_lm + self.weight_alignment * l_align + self.weight_robustness * l_robust

        return {
            "total":      total,
            "lm":         l_lm,
            "alignment":  l_align,
            "robustness": l_robust,
        }


# ---------------------------------------------------------------------------
# ERT Loss — Eviction Robustness Training (Stage 3 / Stage 2 replacement)
# ---------------------------------------------------------------------------

class ERTLoss(nn.Module):
    """Eviction Robustness Training objective.

    Replaces the language-modeling loss used in Stage 2 (which caused
    catastrophic LoRA overfitting — see STAGE2-FINDINGS.md) with a direct
    KL divergence between full-cache and evicted-cache output distributions.

    L_ERT = KL(logits_full ‖ logits_evicted) + w_align * L_align

    Properties vs Stage 2 TISLoss with LM objective:
    - No next-token-prediction term → zero memorisation risk.
    - Gradient signal flows directly from eviction quality to ImportanceUpdateHead.
    - KL divergence cannot reach 0 by memorising training data; it can only
      reach 0 when evicted-cache outputs match full-cache outputs.

    Args:
        weight_alignment: Weight for the importance-alignment MSE loss (default 0.1).
        reduction:        'mean' (default) or 'sum' over non-masked positions.

    ``forward`` input dict keys (all required):
        logits_full       [B, seq, vocab]  — full-cache forward pass
        logits_evicted    [B, seq, vocab]  — evicted-cache forward pass
        labels            [B, seq]         — -100 for masked positions
        predicted_deltas  [B, T, 1]        — ImportanceUpdateHead output
        attention_magnitudes  [B, T]       — normalised to [0, 1]
        importance_scores_norm [B, T]      — normalised to [0, 1]

    Returns a dict with scalar tensors:
        total, kl, alignment
    """

    def __init__(
        self,
        weight_alignment: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight_alignment = weight_alignment
        self.reduction = reduction

    def forward(self, outputs: dict) -> dict[str, torch.Tensor]:
        logits_full    = outputs["logits_full"]
        logits_evicted = outputs["logits_evicted"]
        labels         = outputs["labels"]
        device         = logits_full.device

        l_kl = eviction_robustness_loss(logits_full, logits_evicted, labels)

        l_align = importance_alignment_loss(
            outputs["predicted_deltas"],
            outputs["attention_magnitudes"],
            outputs["importance_scores_norm"],
        )

        total = l_kl + self.weight_alignment * l_align

        return {
            "total":     total,
            "kl":        l_kl,
            "alignment": l_align,
        }

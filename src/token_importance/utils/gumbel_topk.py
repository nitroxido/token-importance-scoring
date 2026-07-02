"""
Gumbel-TopK: Differentiable top-k selection for cache eviction.

Implements straight-through estimator (STE) for differentiable top-k selection.
Used to select which tokens to keep in the KV cache during training.

Reference:
- Jang et al. "Categorical Reparameterization with Gumbel-Softmax" (ICLR 2017)
- Paulus et al. "Variational Structured Output Learning" (ICML 2016)
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """
    Gumbel-Softmax sampling (differentiable approximation to sampling).
    
    Args:
        logits: [batch, n] raw scores
        temperature: Temperature parameter (lower = sharper)
        hard: If True, return one-hot (straight-through estimator)
        dim: Dimension to apply softmax over
        
    Returns:
        One-hot or soft probabilities [batch, n]
    """
    # Add Gumbel noise for sampling effect
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    
    # Softmax with temperature
    y_soft = F.softmax(y / temperature, dim=dim)
    
    if not hard:
        return y_soft
    
    # Straight-through estimator: argmax in forward, softmax in backward
    y_hard = torch.zeros_like(y_soft)
    y_hard.scatter_(dim, y_soft.argmax(dim, keepdim=True), 1.0)
    
    # Straight through: return one-hot in forward, but propagate softmax gradients
    y_hard = y_hard - y_soft.detach() + y_soft
    
    return y_hard


def gumbel_topk(
    scores: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    return_indices: bool = False,
    hard: bool = True,
) -> torch.Tensor:
    """
    Differentiable top-k selection using Gumbel-Softmax.
    
    Selects the k positions with highest scores. Gradient flows through
    the selection mechanism during backprop.
    
    Args:
        scores: [batch, n_tokens] importance scores
        k: Number of tokens to keep
        temperature: Gumbel temperature (lower = sharper selection)
        return_indices: If True, also return selected indices
        hard: If True, use straight-through estimator
        
    Returns:
        masks: [batch, n_tokens] where selected tokens = 1.0, others = 0.0
        (indices: [batch, k] indices of selected tokens, if return_indices=True)
    """
    batch_size, n_tokens = scores.shape
    
    if k >= n_tokens:
        # No eviction needed
        masks = torch.ones_like(scores)
        if return_indices:
            indices = torch.arange(n_tokens, device=scores.device).unsqueeze(0).expand(batch_size, -1)
            return masks, indices
        return masks
    
    # Add Gumbel noise to scores
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
    noisy_scores = scores + gumbel_noise
    
    # Soft top-k using softmax with temperature
    # Higher scores get higher weights
    top_k_weights = F.softmax(noisy_scores / temperature, dim=-1)
    
    # Get actual top-k indices
    _, indices = torch.topk(noisy_scores, k, dim=-1)  # [batch, k]
    
    # Create binary mask
    masks = torch.zeros_like(scores)
    masks.scatter_(1, indices, 1.0)
    
    if hard:
        # Straight-through: hard mask in forward, soft weights in backward
        masks = masks - top_k_weights.detach() + top_k_weights
    else:
        masks = top_k_weights
    
    if return_indices:
        return masks, indices
    
    return masks


def keep_top_k_by_importance(
    importance_scores: torch.Tensor,
    cache_budget: float,
    return_masks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep top-k tokens by importance, evicting others.
    
    This is the eviction policy used in KV cache compression.
    
    Args:
        importance_scores: [batch, seq_len] composite importance scores
        cache_budget: Fraction of tokens to keep (0.0 to 1.0)
        return_masks: If True, also return binary keep/evict masks
        
    Returns:
        evicted_indices: [batch, n_evict] indices to evict
        (masks: [batch, seq_len] if return_masks=True)
    """
    batch_size, seq_len = importance_scores.shape
    
    # Number of tokens to keep
    n_keep = max(1, int(seq_len * cache_budget))
    n_evict = seq_len - n_keep
    
    if n_evict <= 0:
        # No eviction needed
        if return_masks:
            masks = torch.ones_like(importance_scores)
            return torch.empty(batch_size, 0, dtype=torch.long), masks
        return torch.empty(batch_size, 0, dtype=torch.long)
    
    # Get indices to keep (top-k by importance)
    _, keep_indices = torch.topk(importance_scores, n_keep, dim=-1, largest=True)
    keep_indices_sorted, _ = torch.sort(keep_indices, dim=-1)
    
    # Get indices to evict (invert the keep set)
    all_indices = torch.arange(seq_len, device=importance_scores.device).unsqueeze(0).expand(batch_size, -1)
    evict_mask = torch.ones_like(all_indices, dtype=torch.bool)
    evict_mask.scatter_(1, keep_indices_sorted, False)
    evicted_indices = torch.nonzero(evict_mask, as_tuple=False)
    
    if return_masks:
        masks = torch.zeros_like(importance_scores)
        masks.scatter_(1, keep_indices_sorted, 1.0)
        return evicted_indices, masks
    
    return evicted_indices


class GumbelTopkLayer(torch.nn.Module):
    """
    Gumbel-TopK as a differentiable layer for integration into models.
    """
    
    def __init__(
        self,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        annealing_steps: int = 10000,
    ):
        """
        Initialize Gumbel-TopK layer.
        
        Args:
            temperature_init: Initial temperature
            temperature_min: Minimum temperature (after annealing)
            annealing_steps: Steps to anneal temperature over
        """
        super().__init__()
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.annealing_steps = annealing_steps
        self.step = 0
    
    def get_temperature(self) -> float:
        """Get current temperature (with annealing)."""
        progress = min(self.step / self.annealing_steps, 1.0)
        temp = self.temperature_init * (1 - progress) + self.temperature_min * progress
        return temp
    
    def forward(
        self,
        scores: torch.Tensor,
        k: int,
        hard: bool = True,
    ) -> torch.Tensor:
        """
        Select top-k by importance using Gumbel-Softmax.
        
        Args:
            scores: [batch, n] importance scores
            k: Number to keep
            hard: Use straight-through estimator
            
        Returns:
            masks: [batch, n] selection masks
        """
        temp = self.get_temperature()
        return gumbel_topk(scores, k, temperature=temp, hard=hard)
    
    def step_forward(self):
        """Increment step counter (for annealing)."""
        self.step += 1


if __name__ == "__main__":
    print("Testing Gumbel-TopK selection...")
    
    # Test gumbel_topk
    batch_size, n_tokens, k = 2, 100, 50
    scores = torch.randn(batch_size, n_tokens)
    
    masks = gumbel_topk(scores, k, hard=True)
    print(f"✓ Gumbel-TopK successful")
    print(f"  Input shape: {scores.shape}")
    print(f"  Mask shape: {masks.shape}")
    print(f"  Kept tokens per sample: {masks.sum(dim=1).int()}")
    
    # Test with gradients
    scores_grad = torch.randn(batch_size, n_tokens, requires_grad=True)
    masks_grad = gumbel_topk(scores_grad, k, hard=True)
    loss = masks_grad.sum()
    loss.backward()
    print(f"✓ Gradient flow successful")
    print(f"  Gradient shape: {scores_grad.grad.shape}")

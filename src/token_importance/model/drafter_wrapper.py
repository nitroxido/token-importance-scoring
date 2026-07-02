"""
Task 1: TIS-Aware Drafter Wrapper
Wraps EAGLE-3 drafter with token importance score input.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TISAwareDrafter(nn.Module):
    """
    Wraps an EAGLE-3 drafter with Token Importance Store (TIS) awareness.
    
    Allows passing importance scores from target model to drafter during
    speculative decoding, enabling importance-biased token generation.
    """
    
    def __init__(self, eagle3_drafter: nn.Module, device: str = "cuda"):
        """
        Args:
            eagle3_drafter: Pre-trained EAGLE-3 drafter model
            device: Device to run on ("cuda" or "cpu")
        """
        super().__init__()
        self.eagle3 = eagle3_drafter
        self.device = device
        self.importance_scores: Optional[torch.Tensor] = None
        self.importance_attn_bias: Optional[nn.Module] = None
        
    def set_context_importance(self, scores: torch.Tensor) -> None:
        """
        Receive importance scores from target model's ImportanceStore.
        Called once per speculation round before running forward passes.
        
        Args:
            scores: torch.Tensor of shape [seq_len] with values in [0, 100].
                   Higher values indicate more important tokens.
        """
        if scores.ndim != 1:
            raise ValueError(f"Expected 1D importance scores, got shape {scores.shape}")
        
        self.importance_scores = scores.detach().to(self.device)
    
    def set_importance_attn_bias(self, bias_module: nn.Module) -> None:
        """
        Attach an ImportanceAttnBias module for attention biasing.
        
        Args:
            bias_module: DrafterImportanceAttnBias instance
        """
        self.importance_attn_bias = bias_module
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        drafter_step: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run drafter forward pass with optional importance biasing.
        
        Args:
            hidden_states: [B, d_model] hidden states from target model
            drafter_step: int, step within speculation chain (0 to k-1).
                         Used for depth-scaled bias computation.
            attention_mask: Optional [B, seq_len] attention mask
            
        Returns:
            logits: [B, vocab_size] predicted token logits
        """
        # Standard EAGLE-3 forward pass
        logits = self.eagle3(hidden_states)
        
        # Optional: Apply importance bias to attention
        if self.importance_scores is not None and self.importance_attn_bias is not None:
            logits = self._apply_importance_biased_forward(
                hidden_states, logits, drafter_step
            )
        
        return logits
    
    def _apply_importance_biased_forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        drafter_step: int,
    ) -> torch.Tensor:
        """
        Apply importance-guided attention bias through EAGLE-3 attention layers.
        
        This is a placeholder that will be filled in Task 2 when attention
        hooks are available.
        
        Args:
            hidden_states: [B, d_model]
            logits: [B, vocab_size]
            drafter_step: int, depth in speculation chain
            
        Returns:
            logits: [B, vocab_size], potentially modified by importance bias
        """
        # TODO: Implement after identifying EAGLE-3 attention layers (Task 4)
        # For now, return logits unchanged
        return logits
    
    def generate_with_importance(
        self,
        input_ids: torch.Tensor,
        importance_scores: torch.Tensor,
        max_new_tokens: int = 128,
        max_speculation_depth: int = 8,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate tokens using importance-aware speculation.
        
        Args:
            input_ids: [B, seq_len] input token IDs
            importance_scores: [seq_len] importance scores [0, 100]
            max_new_tokens: int, max tokens to generate
            max_speculation_depth: int, max speculation chain length (k)
            temperature: float, sampling temperature
            
        Returns:
            generated_ids: [B, seq_len + max_new_tokens]
            metrics: dict with acceptance_length, divergence, etc.
        """
        self.set_context_importance(importance_scores)
        
        metrics = {
            'acceptance_length': 0.0,
            'divergence': 0.0,
            'steps': 0,
        }
        
        # TODO: Implement speculative decoding loop
        # This will be filled in during Phase 5a training
        
        return input_ids, metrics
    
    def get_importance_scores(self) -> Optional[torch.Tensor]:
        """Return currently stored importance scores."""
        return self.importance_scores
    
    def clear_importance_scores(self) -> None:
        """Clear stored importance scores."""
        self.importance_scores = None


class DrafterSpeculationConfig:
    """Configuration for speculative decoding with importance awareness."""
    
    def __init__(
        self,
        max_speculation_depth: int = 8,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        use_importance_bias: bool = True,
        use_cache: bool = True,
    ):
        self.max_speculation_depth = max_speculation_depth
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.use_importance_bias = use_importance_bias
        self.use_cache = use_cache

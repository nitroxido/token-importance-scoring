"""
Task 2: Importance Attention Bias for Drafter
Applies importance-guided attention bias to EAGLE-3 drafter during speculation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DrafterImportanceAttnBias(nn.Module):
    """
    Applies importance-scaled attention bias to EAGLE-3 drafter.
    
    Separate from main model's ImportanceAttnBias:
    - Used only during speculative decoding (drafter forward pass)
    - Depth-scaled: bias increases with speculation depth (k)
    - Designed for single-layer attention (EAGLE-3)
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        lambda_d_base: float = 0.1,
        lambda_d_slope: float = 0.05,
        lambda_d_max: float = 1.0,
    ):
        """
        Args:
            d_model: Hidden dimension
            lambda_d_base: Base importance bias strength
            lambda_d_slope: Slope for depth scaling (λ_d = base + k * slope)
            lambda_d_max: Maximum bias strength (clamped)
        """
        super().__init__()
        self.d_model = d_model
        
        # Learnable depth scaling parameters
        self.lambda_d_base = nn.Parameter(torch.tensor(lambda_d_base))
        self.lambda_d_slope = nn.Parameter(torch.tensor(lambda_d_slope))
        self.lambda_d_max = lambda_d_max
        
    def compute_depth_scaled_lambda(self, drafter_step: int) -> torch.Tensor:
        """
        Compute depth-scaled bias strength.
        
        λ_d(k) = λ_base + k · λ_slope, clamped to [0, λ_max]
        
        Args:
            drafter_step: int, position in speculation chain (0 to k-1)
            
        Returns:
            lambda_d: float, scaled bias strength
        """
        lambda_d = self.lambda_d_base + drafter_step * self.lambda_d_slope
        return torch.clamp(lambda_d, min=0.0, max=self.lambda_d_max)
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        importance_scores: torch.Tensor,
        drafter_step: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply importance-guided bias to attention scores.
        
        Args:
            attention_scores: [B, n_heads, seq_len, seq_len] pre-softmax attention
            importance_scores: [seq_len] with values in [0, 100]
            drafter_step: int, step in speculation chain
            attention_mask: Optional [B, seq_len] attention mask
            
        Returns:
            attention_scores_biased: [B, n_heads, seq_len, seq_len]
        """
        B, n_heads, seq_len, _ = attention_scores.shape
        
        if importance_scores.shape[0] != seq_len:
            raise ValueError(
                f"Importance scores shape {importance_scores.shape} "
                f"doesn't match sequence length {seq_len}"
            )
        
        # Compute depth-scaled bias strength
        lambda_d = self.compute_depth_scaled_lambda(drafter_step)
        
        # Normalize importance scores from [0, 100] to [0, 1]
        scores_norm = importance_scores / 100.0
        scores_norm = torch.clamp(scores_norm, min=0.0, max=1.0)
        
        # Broadcast to attention shape: [B, n_heads, 1, seq_len]
        # This means we apply importance bias to keys (columns)
        bias = lambda_d * scores_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Apply bias to attention scores (increases logits for important tokens)
        attention_scores_biased = attention_scores + bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask shape: [B, seq_len]
            # Convert to [B, 1, 1, seq_len] for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # Mask out padding tokens (set to very negative value)
            attention_scores_biased = attention_scores_biased.masked_fill(
                ~mask.bool(), float('-inf')
            )
        
        return attention_scores_biased
    
    def get_bias_strength(self, drafter_step: int) -> float:
        """
        Get bias strength for logging/debugging.
        
        Args:
            drafter_step: int, step in speculation chain
            
        Returns:
            strength: float, current lambda_d value
        """
        with torch.no_grad():
            lambda_d = self.compute_depth_scaled_lambda(drafter_step)
            return lambda_d.item()


class AttentionBiasHook:
    """
    Hook to inject importance bias into EAGLE-3 attention computation.
    
    This hook intercepts attention scores before softmax and applies bias.
    """
    
    def __init__(
        self,
        importance_scores: torch.Tensor,
        importance_bias: DrafterImportanceAttnBias,
        drafter_step: int = 0,
    ):
        """
        Args:
            importance_scores: [seq_len] importance scores
            importance_bias: DrafterImportanceAttnBias module
            drafter_step: int, step in speculation chain
        """
        self.importance_scores = importance_scores
        self.importance_bias = importance_bias
        self.drafter_step = drafter_step
    
    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function to be registered on EAGLE-3 attention layer.
        
        This intercepts the attention scores (before softmax) and applies
        importance-based bias.
        
        Args:
            module: The attention module
            input: Input arguments to the module
            output: Output from the module (attention scores or probabilities)
            
        Returns:
            output: Modified output with importance bias applied
        """
        # Assuming output is attention scores [B, n_heads, seq_len, seq_len]
        if isinstance(output, torch.Tensor):
            if output.ndim == 4:  # Attention scores shape
                output = self.importance_bias(
                    output,
                    self.importance_scores,
                    drafter_step=self.drafter_step,
                )
        
        return output


class ImportanceBiasedAttention(nn.Module):
    """
    Wrapper around standard attention that applies importance bias.
    
    Can be used to replace standard attention in EAGLE-3 if direct
    hook registration is not possible.
    """
    
    def __init__(
        self,
        base_attention: nn.Module,
        importance_bias: DrafterImportanceAttnBias,
    ):
        """
        Args:
            base_attention: Original attention module (e.g., from EAGLE-3)
            importance_bias: DrafterImportanceAttnBias instance
        """
        super().__init__()
        self.base_attention = base_attention
        self.importance_bias = importance_bias
        self.importance_scores: Optional[torch.Tensor] = None
        self.drafter_step: int = 0
    
    def set_importance_context(
        self,
        importance_scores: torch.Tensor,
        drafter_step: int = 0,
    ) -> None:
        """
        Set importance context for next forward pass.
        
        Args:
            importance_scores: [seq_len] importance scores [0, 100]
            drafter_step: int, step in speculation chain
        """
        self.importance_scores = importance_scores.detach()
        self.drafter_step = drafter_step
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional importance bias.
        
        Args:
            hidden_states: [B, seq_len, d_model]
            attention_mask: Optional attention mask
            **kwargs: Additional arguments for base attention
            
        Returns:
            output: Attention output [B, seq_len, d_model]
        """
        # Call base attention to get attention scores
        # Note: This is pseudo-code; actual implementation depends on
        # the specific attention module used by EAGLE-3
        output = self.base_attention(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Apply importance bias if context is set
        if self.importance_scores is not None:
            # This would need to be integrated into the base attention
            # computation, which requires modifying the base attention module
            pass
        
        return output

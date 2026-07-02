"""
Post-Norm Transformer Blocks for Attention Drift Mitigation

Problem: Standard pre-norm transformers suffer from attention drift where hidden state
magnitudes grow exponentially along the sequence, suppressing the importance signal.

Solution: Post-norm adds LayerNorm after each residual connection, which:
  1. Stabilizes hidden state magnitudes
  2. Reduces recency bias in attention
  3. Improves importance signal visibility
  
Expected gain: +1-2pp LITM improvement by controlling magnitude growth.
"""

import torch
import torch.nn as nn
from typing import Optional


class PostNormSelfAttentionBlock(nn.Module):
    """Self-attention block with post-norm residual connection.
    
    Architecture:
      x' = Attn(x) + x
      output = LayerNorm(x')
    
    This differs from standard pre-norm which does:
      x' = Attn(LayerNorm(x)) + x
      output = x'
    
    Post-norm benefits:
      - Stabilizes hidden state magnitudes
      - Reduces exponential growth along sequence
      - Improves gradient flow during backprop
      - Mitigates attention drift
    """
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Post-norm layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_mask: (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
            src_key_padding_mask: (batch, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model) with normalized magnitudes
        """
        # Self-attention with post-norm
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + attn_out  # Residual connection
        x = self.norm1(x)  # Post-norm (after residual)
        
        # Feed-forward with post-norm
        ff_out = self.linear1(x)
        ff_out = self.activation(ff_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)
        ff_out = self.dropout(ff_out)
        
        x = x + ff_out  # Residual connection
        x = self.norm2(x)  # Post-norm (after residual)
        
        return x


class PostNormTransformer(nn.Module):
    """Transformer encoder with post-norm blocks.
    
    Stacks N post-norm self-attention blocks. Can be used as a drop-in
    replacement for standard pre-norm transformers with better magnitude control.
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        num_layers: int = 32,
        dim_feedforward: int = 14336,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            PostNormSelfAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (standard in post-norm)
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        
        x = self.final_norm(x)
        return x


def create_postnorm_wrapper(model: nn.Module, d_model: int = 4096) -> dict:
    """
    Create post-norm components to wrap around an existing model.
    
    This is useful for retrofitting post-norm onto existing checkpoints
    without retraining from scratch. The idea is to add post-norm normalization
    after each attention layer and feedforward in the importance head.
    
    Args:
        model: The model to wrap (usually already trained)
        d_model: Hidden dimension
    
    Returns:
        Dictionary of additional modules to attach
    """
    return {
        "importance_head_norm": nn.LayerNorm(d_model),
        "output_projection_norm": nn.LayerNorm(d_model),
    }

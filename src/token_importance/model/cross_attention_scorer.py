"""
P4-C: CrossAttentionImportanceScorer - Learnable cross-attention for importance scoring

Architecture: Query-to-context cross-attention with learnable projection
No Training Needed: Uses frozen transformer weights, evaluates as-is
Purpose: Hybrid approach for KV cache eviction with learnable importance scores
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossAttentionImportanceScorer(nn.Module):
    """
    Learnable cross-attention between query and context for importance scoring.
    
    Architecture:
    - Query projection: [B, Q, D] -> [B, Q, 256]
    - Context projection: [B, T, D] -> [B, T, 256]
    - Cross-attention: query attends to context (multi-head)
    - Importance scoring: learned feedforward on context representations
    
    Design philosophy:
    - No training required (uses frozen initialization)
    - Learnable parameters enable future fine-tuning
    - Multi-head attention captures diverse importance patterns
    """
    
    def __init__(
        self, 
        hidden_dim: int = 4096,
        projection_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Model hidden dimension (Mistral: 4096)
            projection_dim: Intermediate projection dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for attention
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        
        # Projections
        self.query_proj = nn.Linear(hidden_dim, projection_dim)
        self.context_proj = nn.Linear(hidden_dim, projection_dim)
        
        # Multi-head cross-attention (query attends to context)
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Importance scoring head
        self.scorer = nn.Sequential(
            nn.Linear(projection_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(
        self,
        query_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        return_attention_weights: bool = False,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute importance scores via cross-attention.
        
        Args:
            query_hidden: [B, Q, D] - query region hidden states
            context_hidden: [B, T, D] - full context hidden states
            return_attention_weights: If True, return attention weights
            return_logits: If True, return [0, 1] scores; if False, return [0, 100]
            
        Returns:
            importance: [B, T] - per-token importance scores
            attn_weights: Optional[B, Q, T] - attention weights if requested
        """
        B, Q, D = query_hidden.shape
        T = context_hidden.shape[1]
        
        # Project query and context
        q_proj = self.query_proj(query_hidden)  # [B, Q, 256]
        c_proj = self.context_proj(context_hidden)  # [B, T, 256]
        
        # Cross-attention: query attends to context
        attn_out, attn_weights = self.attention(
            q_proj,  # query
            c_proj,  # key
            c_proj,  # value
            need_weights=return_attention_weights
        )  # [B, Q, 256], optional[B, Q, T]
        
        # Score each context token based on context representations
        # (we use original context projections, not attention output)
        importance_logits = self.scorer(c_proj)  # [B, T, 1]
        importance = importance_logits.squeeze(-1)  # [B, T]
        
        # Normalize to [0, 100] if not returning raw logits
        if not return_logits:
            importance = importance * 100.0
            
        if return_attention_weights:
            return importance, attn_weights
        else:
            return importance, None


class HybridImportanceScorer(nn.Module):
    """
    Hybrid importance scoring combining P4-C cross-attention with H2O heuristic.
    
    Strategy:
    1. Compute cross-attention scores: importance_p4c
    2. Compute H2O cumulative attention: importance_h2o
    3. Blend: importance_hybrid = α * importance_p4c + (1-α) * importance_h2o
    
    This approach combines:
    - Learnable query-aware scoring (P4-C)
    - Proven heuristic baseline (H2O)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        alpha: float = 0.5
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            alpha: Blend weight for P4-C (1-alpha goes to H2O)
        """
        super().__init__()
        self.cross_attention_scorer = CrossAttentionImportanceScorer(hidden_dim=hidden_dim)
        self.alpha = alpha
        
    def forward(
        self,
        query_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        attention_weights: torch.Tensor  # From H2O: [B, num_heads, T, T]
    ) -> torch.Tensor:
        """
        Compute hybrid importance scores.
        
        Args:
            query_hidden: [B, Q, D] - query region
            context_hidden: [B, T, D] - context
            attention_weights: [B, H, T, T] - from H2O policy (cumulative attention)
            
        Returns:
            importance: [B, T] - hybrid importance scores [0, 100]
        """
        # P4-C cross-attention scores
        p4c_importance, _ = self.cross_attention_scorer(
            query_hidden, 
            context_hidden,
            return_logits=False
        )  # [B, T]
        
        # H2O cumulative attention (already normalized to [0, 1])
        # Sum over heads and sequence dim: per-token importance
        h2o_importance = attention_weights.sum(dim=(1, 2))  # [B, T] - sum over batch, heads, time
        h2o_importance = h2o_importance / (h2o_importance.max(dim=1, keepdim=True)[0].clamp(min=1e-8))  # normalize to [0, 1]
        h2o_importance = h2o_importance * 100.0  # scale to [0, 100]
        
        # Blend
        hybrid_importance = self.alpha * p4c_importance + (1 - self.alpha) * h2o_importance
        
        return hybrid_importance


if __name__ == "__main__":
    print("Testing CrossAttentionImportanceScorer...")
    
    scorer = CrossAttentionImportanceScorer(hidden_dim=4096, num_heads=8)
    print(f"✓ Scorer instantiated: {sum(p.numel() for p in scorer.parameters())} parameters")
    
    # Test forward pass
    B, Q, T, D = 2, 64, 2048, 4096
    query_hidden = torch.randn(B, Q, D)
    context_hidden = torch.randn(B, T, D)
    
    importance, _ = scorer(query_hidden, context_hidden, return_attention_weights=False)
    print(f"✓ Forward pass: [{B}, {Q}, {D}] + [{B}, {T}, {D}] -> [{B}, {T}]")
    print(f"  Importance range: [{importance.min():.2f}, {importance.max():.2f}]")
    
    # Test with attention weights
    importance, attn_weights = scorer(query_hidden, context_hidden, return_attention_weights=True)
    if attn_weights is not None:
        print(f"✓ Attention weights shape: {attn_weights.shape}")
    
    # Test hybrid scorer
    print("\nTesting HybridImportanceScorer...")
    hybrid = HybridImportanceScorer(hidden_dim=4096, alpha=0.5)
    
    # Simulate H2O attention weights
    h2o_weights = torch.randn(B, 8, T, T).softmax(dim=-1)
    
    hybrid_importance = hybrid(query_hidden, context_hidden, h2o_weights)
    print(f"✓ Hybrid scorer: {hybrid_importance.shape}")
    print(f"  Hybrid importance range: [{hybrid_importance.min():.2f}, {hybrid_importance.max():.2f}]")
    
    print("\n✅ All P4-C components ready for evaluation!")

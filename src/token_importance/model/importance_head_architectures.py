"""
Importance Head Architectures: Multiple approaches for importance learning.

This module provides different architectural approaches to importance head design:
- LinearLora: Simple linear + LoRA (original, broken)  
- CrossAttnLora: Cross-attention + LoRA (Solution A - unified architecture)
- ContrastiveQuery: Query-aware contrastive learning (Solution C - Phase 4)

Users can select at training time via --architecture flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from typing import Optional, Tuple


# ─── Solution A: Cross-Attention Based Head (Matches Evaluation Architecture) ─────────────

class ImportanceUpdateHeadTrainable(nn.Module):
    """
    Cross-attention head for importance learning (trainable version).
    
    Same architecture as ImportanceUpdateHead in importance_head.py, but:
    - Can be trained with LoRA
    - Accepts input format suitable for training
    - Module names match evaluation expectation
    
    This solves the architectural mismatch:
    - Training: query can be a learned representation or derived from context
    - Evaluation: query is the current generation step
    
    The cross-attention mechanism naturally handles both cases.
    """

    def __init__(self, d_model: int = 4096, num_heads: int = 4, max_delta: float = 20.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_delta = max_delta
        
        # Cross-attention: same as evaluation
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection: same as evaluation
        self.out_proj = nn.Linear(d_model, 1, bias=True)
        
        # Layer norm for stability during training
        self.norm = nn.LayerNorm(d_model)
        
        # Learnable query vector (for training when we don't have explicit current token)
        # During training, we can either:
        # 1. Use this learned query
        # 2. Derive query from context (e.g., mean pooling, or task-specific)
        self.query_vector = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Initialize output projection to near-zero to preserve oracle scores
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(
        self,
        context_hidden: torch.Tensor,           # [B, T, d_model]
        query: Optional[torch.Tensor] = None,   # [B, d_model] or None
    ) -> torch.Tensor:                          # [B, T, 1] raw deltas
        """
        Compute importance score deltas.
        
        Args:
            context_hidden: Context token hidden states [B, T, d_model]
            query: Optional query vector. If None, uses learned self.query_vector
                   If [B, d_model], expands to [B, 1, d_model]
        
        Returns:
            raw_deltas: [B, T, 1] importance deltas (before tanh scaling)
        """
        B, T, D = context_hidden.shape
        device = context_hidden.device
        dtype = context_hidden.dtype
        
        # Cast attention module to match input dtype (critical for 4-bit quantized models)
        self.cross_attn = self.cross_attn.to(dtype=dtype)
        self.out_proj = self.out_proj.to(dtype=dtype)
        self.norm = self.norm.to(dtype=dtype)
        
        # Determine query
        if query is None:
            # Use learned query vector, broadcast across batch, matching dtype/device
            query = self.query_vector.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
            query = query.expand(B, 1, D)  # [B, 1, d_model]
        elif query.dim() == 2:
            # Query is [B, d_model], add sequence dimension
            query = query.unsqueeze(1)  # [B, 1, d_model]
        
        # Apply cross-attention
        attn_out, _ = self.cross_attn(
            query=query,        # [B, 1, d_model]
            key=context_hidden, # [B, T, d_model]
            value=context_hidden,  # [B, T, d_model]
        )
        # attn_out: [B, 1, d_model]
        
        # Expand attended summary to all context positions
        attn_expanded = attn_out.expand(-1, T, -1)  # [B, T, d_model]
        attn_expanded = self.norm(attn_expanded)
        
        # Project to single delta per token
        raw_deltas = self.out_proj(attn_expanded)  # [B, T, 1]
        
        return raw_deltas


def create_crossattn_importance_head_with_lora(
    d_model: int = 4096,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    num_heads: int = 4,
) -> nn.Module:
    """
    Create cross-attention importance head with LoRA (Solution A).
    
    This head:
    - Uses same architecture as evaluation (ImportanceUpdateHead)
    - Has LoRA for parameter efficiency
    - Module names match evaluation ("cross_attn", "out_proj")
    - Can be trained and evaluated without architectural mismatch
    
    Args:
        d_model: Model hidden dimension
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha  
        num_heads: Number of attention heads
        
    Returns:
        Head with LoRA applied
    """
    head = ImportanceUpdateHeadTrainable(
        d_model=d_model,
        num_heads=num_heads,
        max_delta=20.0,
    )
    
    # Apply LoRA to cross-attention modules
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["cross_attn"],  # Apply LoRA to attention, not projection
        lora_dropout=0.05,
        bias="none",
    )
    
    head = get_peft_model(head, lora_config)
    
    return head


# ─── Original Solution (Linear + LoRA) - Kept for comparison ─────────────────────────────

class ImportanceScoringHead(nn.Module):
    """
    Linear-based importance scoring head (original, causes mismatch).
    
    DEPRECATED for new training: Use ImportanceUpdateHeadTrainable instead.
    
    Architecture:
    - Input: [batch, seq_len, d_model] from target model
    - Hidden: [batch, seq_len, d_head] (linear projection + LoRA)
    - Output: [batch, seq_len, 1] (score delta)
    
    Problem: Module names ("lora_layer") don't match evaluation ("cross_attn", "out_proj")
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        d_head: int = 256,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_delta: float = 20.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.max_delta = max_delta
        
        self.proj_in = nn.Linear(d_model, d_head)
        self.lora_layer = nn.Linear(d_head, d_head)  # LoRA applied here
        self.proj_out = nn.Linear(d_head, 1)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(lora_dropout)
        self.norm = nn.LayerNorm(d_head)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.constant_(self.proj_in.bias, 0.0)
        nn.init.normal_(self.lora_layer.weight, std=0.02)
        nn.init.constant_(self.lora_layer.bias, 0.0)
        nn.init.normal_(self.proj_out.weight, std=0.001)
        nn.init.constant_(self.proj_out.bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict importance score deltas.
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] optional
            
        Returns:
            deltas: [batch, seq_len, 1]
        """
        # Cast modules to match input dtype (critical for 4-bit quantized models)
        device = hidden_states.device
        dtype = hidden_states.dtype
        self.proj_in = self.proj_in.to(dtype=dtype)
        self.lora_layer = self.lora_layer.to(dtype=dtype)
        self.proj_out = self.proj_out.to(dtype=dtype)
        self.norm = self.norm.to(dtype=dtype)
        
        hidden = self.proj_in(hidden_states)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        
        hidden = self.norm(self.lora_layer(hidden) + hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        
        deltas = self.proj_out(hidden)
        deltas = torch.tanh(deltas) * self.max_delta
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            deltas = deltas * mask
        
        return deltas


def create_importance_head_with_lora(
    d_model: int = 4096,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> nn.Module:
    """
    Create importance scoring head with LoRA (original linear version).
    
    ⚠️ DEPRECATED: Use create_crossattn_importance_head_with_lora() instead
    
    Returns head with "lora_layer" module (architectural mismatch with eval)
    """
    head = ImportanceScoringHead(
        d_model=d_model,
        d_head=256,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        max_delta=20.0,
    )
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lora_layer"],  # ← Module name mismatch!
        lora_dropout=0.05,
        bias="none",
    )
    
    head = get_peft_model(head, lora_config)
    return head


def get_trainable_params_count(model: nn.Module) -> Tuple[int, int]:
    """Get trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ─── Solution D: Attention-Pooled Importance (SnapKV-style) ──────────────────────────

def pool_query_attention(
    attention_tensors: list[torch.Tensor],
    n_query: int = 64,
) -> torch.Tensor:
    """Pool attention from last n_query tokens (SnapKV-style, Solution D).
    
    Args:
        attention_tensors: List of [B, H, T, T] attention matrices from each layer
        n_query: Number of query tokens to pool from (last n_query positions)
        
    Returns:
        scores: [T] tensor with per-token importance scores (summed across layers & heads)
        
    This implements SnapKV's key insight: pool attention from the query/instruction
    tokens to identify which context tokens are important to the query.
    """
    if not attention_tensors:
        return torch.ones(1, dtype=torch.float32)
    
    T = attention_tensors[0].shape[-1]
    scores = torch.zeros(T, dtype=torch.float32)
    
    for layer_attn in attention_tensors:  # [B, H, T, T]
        attn = layer_attn[0]  # [H, T, T] - remove batch dimension
        query_start = max(0, T - n_query)
        # Query positions are rows [query_start:T], key/value positions are columns
        query_attn = attn[:, query_start:, :]  # [H, n_query, T]
        # Sum across heads and query positions
        scores += query_attn.sum(dim=(0, 1)).cpu()
    
    return scores


def train_head_to_predict_attention(
    importance_head: nn.Module,
    hidden_states: torch.Tensor,
    attention_tensors: list[torch.Tensor],
    n_query: int = 64,
) -> torch.Tensor:
    """Forward pass: train head to predict attention pooling scores.
    
    Args:
        importance_head: The importance head to train (cross-attn or linear)
        hidden_states: [B, T, d_model] - model's hidden states
        attention_tensors: List of [B, H, T, T] - model's attention patterns
        n_query: Number of query tokens to use for attention pooling
        
    Returns:
        predicted_scores: [B, T, 1] - head's predicted importance scores
        target_scores: [T] - ground truth from attention pooling
        
    This is used during training with self-supervised supervision:
    - Target: Actual attention pooling (what SnapKV uses)
    - Predicted: What our learned head predicts
    - Loss: MSE between predicted and target
    """
    # Get ground truth from attention pooling
    target_scores = pool_query_attention(attention_tensors, n_query)
    
    # Get predictions from importance head
    if hasattr(importance_head, 'forward') and len(hidden_states.shape) == 3:
        # Standard head forward pass
        predicted_scores = importance_head(hidden_states)  # [B, T, 1]
    else:
        # Fallback
        predicted_scores = importance_head(hidden_states.squeeze(0))
    
    return predicted_scores, target_scores

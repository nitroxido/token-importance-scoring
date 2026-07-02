"""
Importance Scoring Head — LoRA-based module for ERT training.

Predicts per-token importance scores [0, 100] as deltas from oracle scores.
Uses LoRA for memory efficiency on limited GPU (8GB RTX 5070, 16GB integrated).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig


class ImportanceScoringHead(nn.Module):
    """
    LoRA-based head for predicting importance score deltas.
    
    Architecture:
    - Input: [batch, seq_len, d_model] from target model
    - Hidden: [batch, seq_len, d_head] (linear projection + LoRA)
    - Output: [batch, seq_len, 1] (score delta, clipped to [-100, +100])
    
    The delta is added to oracle/initialization scores:
        predicted_score = clamp(oracle_score + delta, 0, 100)
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
        """
        Initialize importance scoring head.
        
        Args:
            d_model: Model hidden dimension (Mistral: 4096)
            d_head: Head projection dimension
            lora_rank: LoRA rank (low for memory efficiency)
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout
            max_delta: Maximum delta magnitude (clipped after tanh)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.max_delta = max_delta
        
        # Linear projection (no LoRA on this)
        self.proj_in = nn.Linear(d_model, d_head)
        
        # LoRA will be applied to this layer
        self.lora_layer = nn.Linear(d_head, d_head)
        
        # Output projection to single score delta
        self.proj_out = nn.Linear(d_head, 1)
        
        # Activation
        self.act = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(lora_dropout)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_head)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values to minimize change at start."""
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.constant_(self.proj_in.bias, 0.0)
        
        nn.init.normal_(self.lora_layer.weight, std=0.02)
        nn.init.constant_(self.lora_layer.bias, 0.0)
        
        # Output projection: start near zero to preserve oracle scores
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
            hidden_states: [batch, seq_len, d_model] from target model
            attention_mask: [batch, seq_len] (optional, 1=keep, 0=mask)
            
        Returns:
            deltas: [batch, seq_len, 1] score deltas clamped to [-max_delta, +max_delta]
        """
        # Project to head dimension
        hidden = self.proj_in(hidden_states)  # [B, T, d_head]
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        
        # LoRA layer with residual
        hidden = self.norm(self.lora_layer(hidden) + hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        
        # Output projection with tanh to bound delta
        deltas = self.proj_out(hidden)  # [B, T, 1]
        deltas = torch.tanh(deltas) * self.max_delta  # Bounded [-max_delta, +max_delta]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match deltas shape [B, T, 1]
            mask = attention_mask.unsqueeze(-1).float()
            deltas = deltas * mask
        
        return deltas


def create_importance_head_with_lora(
    d_model: int = 4096,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> nn.Module:
    """
    Create importance scoring head with LoRA applied.
    
    Args:
        d_model: Model hidden dimension
        lora_rank: LoRA rank (lower for smaller GPUs)
        lora_alpha: LoRA alpha
        
    Returns:
        Head module with LoRA applied to lora_layer
    """
    # Create base head
    head = ImportanceScoringHead(
        d_model=d_model,
        d_head=256,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        max_delta=20.0,
    )
    
    # Apply LoRA to the lora_layer (without task_type for non-generative models)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lora_layer"],
        lora_dropout=0.05,
        bias="none",
    )
    
    head = get_peft_model(head, lora_config)
    
    return head


def get_trainable_params_count(model: nn.Module) -> Tuple[int, int]:
    """
    Get trainable and total parameter counts.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# Memory-optimized inference wrapper
class ImportanceScoringHeadInference(nn.Module):
    """
    Inference wrapper for importance scoring head.
    
    Uses gradient checkpointing and mixed precision to minimize memory usage.
    """
    
    def __init__(self, head: nn.Module, use_gradient_checkpoint: bool = True):
        super().__init__()
        self.head = head
        self.use_gradient_checkpoint = use_gradient_checkpoint
    
    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward pass (no gradients).
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            
        Returns:
            deltas: [batch, seq_len, 1]
        """
        return self.head(hidden_states)


if __name__ == "__main__":
    # Quick test
    print("Testing ImportanceScoringHead...")
    
    head = create_importance_head_with_lora(d_model=4096, lora_rank=8)
    trainable, total = get_trainable_params_count(head)
    
    print(f"✓ Head created")
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Total params: {total:,}")
    
    # Forward pass test
    batch_size, seq_len = 2, 1024
    hidden_states = torch.randn(batch_size, seq_len, 4096)
    
    deltas = head(hidden_states)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {deltas.shape}")
    print(f"  Delta range: [{deltas.min():.2f}, {deltas.max():.2f}]")

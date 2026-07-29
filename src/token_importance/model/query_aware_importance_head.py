"""
P3b: QueryAwareImportanceHead - Supervised importance scoring on LITM data

Architecture: Simple query-context cross-projection with learned importance scoring
Training: Supervised on synthetic LITM labels (binary: answer_token or context_token)
Purpose: Learn position-specific importance for information retrieval tasks
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class QueryAwareImportanceHead(nn.Module):
    """
    Learns query-to-context importance mapping via supervised training on LITM data.
    
    Architecture:
    - Query projection: [B, Q, D] -> [B, 1, 256]
    - Context projection: [B, T, D] -> [B, T, 256]
    - Combined scoring: [B, T, 512] -> [B, T, 1] importance scores
    
    Trained with importance-weighted cross-entropy loss on binary token labels.
    """
    
    def __init__(self, hidden_dim: int = 4096, projection_dim: int = 256):
        """
        Args:
            hidden_dim: Model hidden dimension (Mistral: 4096)
            projection_dim: Intermediate projection dimension
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        
        # Projection layers
        self.query_proj = nn.Linear(hidden_dim, projection_dim)
        self.context_proj = nn.Linear(hidden_dim, projection_dim)
        
        # Importance scoring head
        self.scorer = nn.Sequential(
            nn.Linear(projection_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(
        self, 
        query_hidden: torch.Tensor, 
        context_hidden: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Compute importance scores for context tokens given query.
        
        Args:
            query_hidden: [B, Q, D] - query region hidden states (typically last 64 tokens)
            context_hidden: [B, T, D] - full context hidden states
            return_logits: If True, return [0, 1] scores; if False, return [0, 100] normalized
            
        Returns:
            importance: [B, T] - per-token importance scores
        """
        B, T, D = context_hidden.shape
        
        # Project query and context
        query_score = self.query_proj(query_hidden)  # [B, Q, 256]
        context_score = self.context_proj(context_hidden)  # [B, T, 256]
        
        # Average over query tokens to get query representation
        query_avg = query_score.mean(dim=1, keepdim=True)  # [B, 1, 256]
        
        # Broadcast and concatenate
        query_broadcast = query_avg.expand(B, T, self.projection_dim)  # [B, T, 256]
        combined = torch.cat([query_broadcast, context_score], dim=-1)  # [B, T, 512]
        
        # Score each context token
        importance = self.scorer(combined).squeeze(-1)  # [B, T]
        
        # Normalize to [0, 100] if not returning raw logits
        if not return_logits:
            importance = importance * 100.0
            
        return importance


class ImportanceWeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss weighted by token importance labels.
    
    For LITM supervision:
    - answer_tokens: weight = 1.0 (high importance)
    - context_tokens: weight = 0.5 (lower importance)
    - padding: weight = 0.0 (ignored)
    """
    
    def __init__(self, weight_answer: float = 1.0, weight_context: float = 0.5):
        super().__init__()
        self.weight_answer = weight_answer
        self.weight_context = weight_context
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(
        self, 
        logits: torch.Tensor,  # [B, T]
        labels: torch.Tensor,  # [B, T] binary (0 or 1)
        mask: Optional[torch.Tensor] = None  # [B, T] (1 for valid, 0 for padding)
    ) -> torch.Tensor:
        """
        Compute weighted importance loss.
        
        Args:
            logits: [B, T] predicted importance scores [0, 1]
            labels: [B, T] binary labels (1=answer_token, 0=context_token)
            mask: [B, T] optional mask for padding tokens
            
        Returns:
            loss: scalar loss value
        """
        # Compute per-token loss
        loss_per_token = self.ce_loss(logits.view(-1), labels.view(-1).float())  # [B*T]
        loss_per_token = loss_per_token.view_as(labels)
        
        # Compute per-token weights
        weights = torch.where(
            labels == 1,
            torch.full_like(labels, self.weight_answer, dtype=torch.float32),
            torch.full_like(labels, self.weight_context, dtype=torch.float32)
        )
        
        # Apply mask if provided
        if mask is not None:
            weights = weights * mask.float()
            
        # Weighted loss
        weighted_loss = (loss_per_token * weights).sum() / weights.sum().clamp(min=1e-8)
        
        return weighted_loss


if __name__ == "__main__":
    # Test instantiation
    print("Testing QueryAwareImportanceHead...")
    
    head = QueryAwareImportanceHead(hidden_dim=4096, projection_dim=256)
    print(f"✓ Head instantiated: {sum(p.numel() for p in head.parameters())} parameters")
    
    # Test forward pass
    B, Q, T, D = 2, 64, 2048, 4096
    query_hidden = torch.randn(B, Q, D)
    context_hidden = torch.randn(B, T, D)
    
    importance = head(query_hidden, context_hidden)
    print(f"✓ Forward pass: input shapes [{B}, {Q}, {D}] + [{B}, {T}, {D}] -> output [{B}, {T}]")
    print(f"  Importance range: [{importance.min():.2f}, {importance.max():.2f}]")
    
    # Test loss
    print("\nTesting ImportanceWeightedCrossEntropyLoss...")
    loss_fn = ImportanceWeightedCrossEntropyLoss()
    
    logits = torch.sigmoid(torch.randn(B, T))
    labels = torch.randint(0, 2, (B, T)).float()
    mask = torch.ones(B, T)
    
    loss = loss_fn(logits, labels, mask)
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    print("\n✅ All P3b components ready for training!")

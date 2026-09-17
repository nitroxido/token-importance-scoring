"""
Bridge Detection Head for TIS v2.4

Auxiliary classifier that learns to detect bridge passages in multi-hop questions.
A bridge passage is one that establishes an entity or relationship needed to connect
the query to the final answer.

Architecture:
- Simple 2-layer MLP on mean-pooled passage representation
- Binary classification: 0 = single-passage relevant, 1 = bridge passage

Training:
- Supervised on HotpotQA question type labels (bridge vs. comparison, etc.)
- Can be trained jointly with importance head using blended loss
"""

import torch
import torch.nn as nn
from typing import Optional


class BridgeDetectionHead(nn.Module):
    """
    Auxiliary head to detect bridge passages (supporting passages that establish
    intermediate entities or relationships).
    
    A bridge passage doesn't directly answer the query but provides context needed
    to understand other passages. Example:
    - Query: "What color is the flag of [Country]?"
    - Bridge passage: "X is the capital of [Country]"
    - Answer passage: "[Country]'s flag is [color]"
    
    This head learns to identify bridge passages so ranking can give them higher scores.
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        hidden_layer_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension (Mistral: 4096)
            hidden_layer_dim: Size of hidden layer in MLP
            dropout: Dropout for regularization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layer_dim = hidden_layer_dim
        
        # 2-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_dim, 1),  # Binary classification
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        passage_hidden: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Compute bridge detection scores for passages.
        
        Args:
            passage_hidden: [B, T, D] - passage token hidden states
                            B=batch size, T=seq_len, D=hidden_dim
            return_logits: If True, return raw sigmoid [0,1]; if False, scale to [-1,1] for blending
            
        Returns:
            bridge_scores: [B] - bridge detection scores per passage
                          High value = likely bridge passage
                          Low value = likely direct answer passage
        """
        # Mean pool over tokens to get passage representation
        passage_repr = passage_hidden.mean(dim=1)  # [B, D]
        
        # Classify: is this a bridge passage?
        bridge_logits = self.classifier(passage_repr)  # [B, 1]
        bridge_scores = bridge_logits.squeeze(-1)  # [B]
        
        # Scale to [-1, 1] for easier blending with importance scores
        # 0.5 (neutral) -> 0.0 (no adjustment)
        # 1.0 (strong bridge) -> 1.0 (boost score)
        # 0.0 (not bridge) -> -1.0 (penalize score)
        if not return_logits:
            bridge_scores = 2.0 * (bridge_scores - 0.5)
        
        return bridge_scores
    
    def forward_binary(
        self,
        passage_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return raw binary probabilities [0, 1].
        
        Args:
            passage_hidden: [B, T, D] - passage token hidden states
            
        Returns:
            binary_probs: [B] - probability that passage is bridge (0=not bridge, 1=bridge)
        """
        passage_repr = passage_hidden.mean(dim=1)  # [B, D]
        binary_probs = self.classifier(passage_repr).squeeze(-1)  # [B]
        return binary_probs


class BridgeDetectionLoss(nn.Module):
    """
    Binary cross-entropy loss for bridge detection training.
    
    Training signal comes from HotpotQA:
    - Bridge questions: passages in supporting_facts positions marked as bridges
    - Single-hop questions: all relevant passages marked as non-bridges
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        bridge_logits: torch.Tensor,  # [B]
        labels: torch.Tensor,  # [B] binary (0 or 1)
    ) -> torch.Tensor:
        """
        Compute bridge detection loss.
        
        Args:
            bridge_logits: [B] predicted bridge probabilities [0, 1]
            labels: [B] binary labels (1=bridge, 0=non-bridge)
            
        Returns:
            loss: scalar loss value
        """
        return self.bce_loss(bridge_logits, labels.float())


if __name__ == "__main__":
    # Test instantiation
    print("Testing BridgeDetectionHead...")
    
    head = BridgeDetectionHead(hidden_dim=4096, hidden_layer_dim=256)
    print(f"✓ Head instantiated: {sum(p.numel() for p in head.parameters())} parameters")
    
    # Test forward pass
    B, T, D = 2, 512, 4096
    passage_hidden = torch.randn(B, T, D)
    
    # Binary output
    binary_probs = head.forward_binary(passage_hidden)
    print(f"✓ Binary probabilities: shape={binary_probs.shape}, values=[{binary_probs[0]:.3f}, {binary_probs[1]:.3f}]")
    
    # Scaled output for blending
    scaled_scores = head(passage_hidden, return_logits=False)
    print(f"✓ Scaled scores (for blending): shape={scaled_scores.shape}, range=[{scaled_scores.min():.3f}, {scaled_scores.max():.3f}]")
    
    # Loss computation
    loss_fn = BridgeDetectionLoss()
    labels = torch.tensor([0.0, 1.0])
    loss = loss_fn(binary_probs, labels)
    print(f"✓ Loss computation: {loss.item():.4f}")

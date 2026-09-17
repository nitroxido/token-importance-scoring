"""
Iterative Refinement Head for TIS v2.5 (Phase 2 - Path A)

ITERATIVE REFINEMENT SCORING
============================

Two-stage pipeline:
1. Stage 1 (fast): Score all passages independently → rank them
2. Stage 2 (selective refinement): Re-score top-K passages jointly, 
   considering what's already been selected

Why it works for multi-hop:
- First pass retrieves candidate answer passages (high direct relevance)
- Second pass: "Given we selected passage A (answer), which other passages are needed?"
- Explicitly models the bridge dependency: "passage B is important because it chains to A"

Architecture:
- Simple cross-attention between candidate passage and already-selected passages
- Refinement score blended with direct score: final = 0.7*direct + 0.3*refined

Complexity: O(K · n) vs full O(n²) pairwise scoring
Expected gain: +15-20pp at K=5 over independent scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RefinementConfig:
    """Configuration for refinement head."""
    hidden_dim: int = 4096  # Mistral hidden dimension
    num_heads: int = 4  # Number of attention heads
    hidden_layer_dim: int = 1024  # MLP hidden dimension
    dropout: float = 0.1
    blend_weight: float = 0.3  # Weight for refinement score in blend
    # blend: final = (1 - blend_weight) * direct + blend_weight * refined
    context_aggregation: str = "mean"  # "mean" or "max"


class CrossPassageAttention(nn.Module):
    """
    Cross-attention mechanism to score candidate passage in context of 
    already-selected passages.
    
    Multi-head attention where:
    - Query: candidate passage representation
    - Key/Value: concatenation of selected passages
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"{hidden_dim} not divisible by {num_heads}"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        candidate_hidden: torch.Tensor,  # [batch, seq_len, hidden]
        selected_hiddens: List[torch.Tensor],  # List of [batch, seq_len, hidden]
    ) -> torch.Tensor:
        """
        Compute cross-passage attention score for candidate passage.
        
        Args:
            candidate_hidden: Passage to score (seq_len can vary)
            selected_hiddens: List of already-selected passages to attend to
        
        Returns:
            Context-aware representation [batch, hidden_dim]
        """
        batch_size = candidate_hidden.shape[0]
        
        # Project candidate
        q = self.q_proj(candidate_hidden)  # [batch, seq_len, hidden]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len_cand, head_dim]
        
        if not selected_hiddens:
            # No context: use candidate's mean pooling
            return candidate_hidden.mean(dim=1)  # [batch, hidden]
        
        # Concatenate all selected passages to form context
        context = torch.cat(selected_hiddens, dim=1)  # [batch, total_seq_len, hidden]
        
        # Project context
        k = self.k_proj(context)  # [batch, total_seq_len, hidden]
        v = self.v_proj(context)  # [batch, total_seq_len, hidden]
        
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, total_seq_len, head_dim]
        
        # Attention: (Q @ K^T) @ V
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, num_heads, seq_len_cand, total_seq_len]
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, seq_len_cand, head_dim]
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
        # [batch, seq_len_cand, hidden]
        
        attn_output = self.out_proj(attn_output)
        
        # Pool to single vector
        refined = attn_output.mean(dim=1)  # [batch, hidden]
        
        return refined


class RefinementScoringHead(nn.Module):
    """
    Refinement scoring head that computes importance scores for passages
    in the context of already-selected passages.
    
    Used in Stage 2 of iterative refinement pipeline.
    """
    
    def __init__(self, config: RefinementConfig = None):
        super().__init__()
        config = config or RefinementConfig()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Cross-passage attention module
        self.cross_attention = CrossPassageAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        
        # MLP to convert attention output to score
        self.scoring_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_layer_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_layer_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),  # Single score per passage
        )
    
    def forward(
        self,
        candidate_hidden: torch.Tensor,  # [batch, seq_len, hidden]
        query_hidden: torch.Tensor,  # [batch, query_seq_len, hidden]
        selected_hiddens: Optional[List[torch.Tensor]] = None,  # Context from selected passages
    ) -> torch.Tensor:
        """
        Score candidate passage in context of already-selected passages.
        
        Args:
            candidate_hidden: Passage to score [batch, seq_len, hidden]
            query_hidden: Query representation [batch, query_seq_len, hidden]
            selected_hiddens: List of context passages to condition on
        
        Returns:
            Refinement scores [batch, 1]
        """
        selected_hiddens = selected_hiddens or []
        
        # Compute cross-passage attention
        context_repr = self.cross_attention(candidate_hidden, selected_hiddens)
        # [batch, hidden]
        
        # Combine with query information
        query_repr = query_hidden.mean(dim=1)  # [batch, hidden]
        combined = context_repr + query_repr  # Residual-style combination
        
        # Score
        refinement_score = self.scoring_mlp(combined)  # [batch, 1]
        
        return refinement_score


class IterativeRefinementPipeline(nn.Module):
    """
    Complete iterative refinement pipeline for multi-hop passage ranking.
    
    Stage 1: Compute direct importance scores for all passages independently
    Stage 2: Selectively re-score top-K passages considering context
    
    Blend: final_score = (1 - blend_w) * direct_score + blend_w * refined_score
    """
    
    def __init__(
        self,
        importance_head: nn.Module,  # TIS importance head (Stage 1)
        refinement_head: Optional[RefinementScoringHead] = None,  # Stage 2
        config: RefinementConfig = None,
    ):
        super().__init__()
        self.importance_head = importance_head
        self.refinement_head = refinement_head or RefinementScoringHead(config)
        self.config = config or RefinementConfig()
    
    def stage_1_direct_scoring(
        self,
        passages_hidden: List[torch.Tensor],
        query_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1: Fast, independent scoring of all passages.
        
        Args:
            passages_hidden: List of passage representations [batch, seq_len, hidden] each
            query_hidden: Query representation [batch, query_seq_len, hidden]
        
        Returns:
            (scores, passage_indices): scores [n_passages], indices [n_passages]
        """
        scores = []
        for passage_h in passages_hidden:
            score = self.importance_head(
                doc_hidden=passage_h,
                query_embeddings=query_hidden,
            )
            scores.append(score)
        
        scores = torch.cat(scores, dim=0)  # [n_passages, 1]
        scores = scores.squeeze(-1)  # [n_passages]
        
        # Rank passages by score
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]
        
        return sorted_scores, sorted_indices
    
    def stage_2_refinement_scoring(
        self,
        passages_hidden: List[torch.Tensor],
        query_hidden: torch.Tensor,
        top_k: int = 5,
        scores_stage1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: Selective refinement of top-K passages in context of each other.
        
        For each passage in top-K, re-score it considering the context of
        the passages already selected.
        
        Args:
            passages_hidden: List of all passage representations
            query_hidden: Query representation
            top_k: Number of passages to refine (typically 5-10)
            scores_stage1: Scores from Stage 1 (if None, computed here)
        
        Returns:
            (refined_scores, indices): refined scores [top_k], indices into passages_hidden
        """
        if scores_stage1 is None:
            scores_stage1, top_indices = self.stage_1_direct_scoring(passages_hidden, query_hidden)
        else:
            top_indices = torch.argsort(scores_stage1, descending=True)[:top_k]
        
        refined_scores = []
        selected_hiddens = []  # Context: passages already added to ranking
        
        for rank in range(min(top_k, len(passages_hidden))):
            # Get current candidate passage
            candidate_idx = top_indices[rank]
            candidate_hidden = passages_hidden[candidate_idx]
            
            # Re-score in context of already-selected passages
            refined_score = self.refinement_head(
                candidate_hidden=candidate_hidden,
                query_hidden=query_hidden,
                selected_hiddens=selected_hiddens,
            )
            
            refined_scores.append(refined_score)
            
            # Add this passage to context for next iteration
            selected_hiddens.append(candidate_hidden)
        
        refined_scores = torch.cat(refined_scores, dim=0)  # [top_k, 1]
        refined_scores = refined_scores.squeeze(-1)  # [top_k]
        top_indices = top_indices[:top_k]
        
        return refined_scores, top_indices
    
    def forward_blend(
        self,
        passages_hidden: List[torch.Tensor],
        query_hidden: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full iterative refinement with blending.
        
        Returns scores that blend Stage 1 (direct) and Stage 2 (refined).
        
        Args:
            passages_hidden: List of passage representations
            query_hidden: Query representation
            k: Top-K passages to refine
        
        Returns:
            (blended_scores, top_indices): final scores [k], indices [k]
        """
        # Stage 1: Get direct scores and top-K
        direct_scores, top_indices = self.stage_1_direct_scoring(passages_hidden, query_hidden)
        top_k_direct = direct_scores[top_indices[:k]]
        
        # Stage 2: Get refinement scores for top-K
        refined_scores, top_indices_refined = self.stage_2_refinement_scoring(
            passages_hidden, query_hidden, top_k=k, scores_stage1=direct_scores
        )
        
        # Blend: final = (1 - w) * direct + w * refined
        blend_w = self.config.blend_weight
        blended = (1 - blend_w) * top_k_direct + blend_w * refined_scores
        
        return blended, top_indices_refined
    
    def forward(
        self,
        passages_hidden: List[torch.Tensor],
        query_hidden: torch.Tensor,
        k: int = 5,
        use_refinement: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional refinement.
        
        If use_refinement=False, behaves like Stage 1 only (for ablation).
        
        Args:
            passages_hidden: List of passage representations
            query_hidden: Query representation
            k: Top-K passages to return
            use_refinement: Whether to apply Stage 2 refinement
        
        Returns:
            (scores, indices): top-K scores and passage indices
        """
        if not use_refinement:
            # Ablation: Stage 1 only
            direct_scores, top_indices = self.stage_1_direct_scoring(passages_hidden, query_hidden)
            return direct_scores[top_indices[:k]], top_indices[:k]
        else:
            # Full pipeline with refinement
            return self.forward_blend(passages_hidden, query_hidden, k)


if __name__ == "__main__":
    # Smoke test
    print("Testing RefinementScoringHead...")
    
    config = RefinementConfig(
        hidden_dim=4096,
        num_heads=4,
        hidden_layer_dim=1024,
    )
    
    head = RefinementScoringHead(config)
    head = head.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate inputs
    device = next(head.parameters()).device
    batch_size = 2
    
    candidate = torch.randn(batch_size, 64, 4096, device=device)
    query = torch.randn(batch_size, 32, 4096, device=device)
    selected = [torch.randn(batch_size, 48, 4096, device=device) for _ in range(2)]
    
    # Forward pass
    score = head(candidate, query, selected)
    print(f"  ✓ Input shapes: candidate={candidate.shape}, query={query.shape}")
    print(f"  ✓ Output shape: {score.shape}")
    print(f"  ✓ Score value: {score[0].item():.4f}")
    
    print("\n✓ RefinementScoringHead test passed!")

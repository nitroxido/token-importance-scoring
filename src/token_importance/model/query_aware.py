"""Query-aware importance learning architecture for Phase 4.

Implements contrastive query-document matching to teach the importance head
to recognize semantic relevance (query-dependent) in addition to structural
importance (query-independent).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):
    """Lightweight query encoder for contrastive learning.
    
    Maps query hidden states to a low-dimensional query embedding used for
    computing similarity with importance-weighted document embeddings.
    
    Architecture:
        Input:  [batch, seq_len, hidden_dim]  (query token hidden states)
        Pool:   mean-pooling over query tokens → [batch, hidden_dim]
        Linear: hidden_dim → intermediate_dim
        ReLU + LayerNorm
        Linear: intermediate_dim → output_dim
        Output: [batch, output_dim]  (query embedding)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,      # Mistral-7B hidden size
        output_dim: int = 256,       # Query embedding dimension
        intermediate_dim: int = 1024,  # FFN dimension
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Two-layer MLP with LayerNorm
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm1 = nn.LayerNorm(intermediate_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> torch.Tensor:
        """Encode query tokens to query embedding.
        
        Args:
            hidden_states: Hidden states from base model [B, T, D]
            attention_mask: Mask for valid query tokens [B, T] (1=valid, 0=padding)
        
        Returns:
            query_emb: Query embedding [B, output_dim]
        """
        # Mean-pool query tokens (masked)
        if attention_mask is not None:
            # Expand mask to match hidden_states shape
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            # Zero out padding tokens
            masked_hidden = hidden_states * mask_expanded  # [B, T, D]
            # Sum and normalize
            sum_hidden = masked_hidden.sum(dim=1)  # [B, D]
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            pooled = sum_hidden / sum_mask  # [B, D]
        else:
            pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Two-layer MLP
        x = self.fc1(pooled)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        return x  # [B, output_dim]


class QueryAwareImportanceHead(nn.Module):
    """Combines importance head with query-aware contrastive learning.
    
    Architecture:
        1. ImportanceUpdateHead: Predicts per-token importance scores
        2. QueryEncoder: Maps query tokens to query embedding
        3. Document pooling: Importance-weighted aggregation of doc tokens
        4. Similarity: Bilinear matching between query and doc embeddings
    
    Training objective:
        L = L_contrastive + λ × L_align
    
    Where:
        L_contrastive: InfoNCE loss over (query, relevant_doc, hard_negatives)
        L_align: MSE between predicted and oracle importance (from ERT)
    """
    
    def __init__(
        self,
        importance_head: nn.Module,  # Pre-trained from ERT
        hidden_dim: int = 4096,
        query_emb_dim: int = 256,
        align_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        self.importance_head = importance_head  # ERT checkpoint (trainable)
        self.query_encoder = QueryEncoder(
            hidden_dim=hidden_dim,
            output_dim=query_emb_dim,
        )
        
        # Bilinear similarity: query_emb^T @ W @ doc_emb
        # W shape: [query_emb_dim, hidden_dim]
        self.similarity_matrix = nn.Parameter(
            torch.randn(query_emb_dim, hidden_dim) * 0.01
        )
        
        self.align_weight = align_weight
        self.temperature = temperature
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, D]
        query_mask: torch.Tensor,     # [B, T] (1=query token)
        doc_masks: torch.Tensor,      # [B, N, T] (1=doc_i token, N=num_docs)
        labels: Optional[torch.Tensor] = None,  # [B] (index of relevant doc)
        oracle_scores: Optional[torch.Tensor] = None,  # [B, T] (for alignment loss)
    ) -> dict:
        """Forward pass with contrastive + alignment loss.
        
        Args:
            hidden_states: Base model hidden states [B, T, D]
            query_mask: Mask for query tokens [B, T]
            doc_masks: Masks for each document [B, N, T]
            labels: Index of relevant document per query [B]
            oracle_scores: Oracle importance scores for alignment [B, T]
        
        Returns:
            dict with keys:
                - 'loss': Combined loss (contrastive + align)
                - 'contrastive_loss': InfoNCE loss
                - 'align_loss': MSE with oracle scores
                - 'importance_scores': Predicted scores [B, T]
                - 'similarities': Query-doc similarities [B, N]
                - 'contrastive_acc': Fraction where relevant doc ranks #1
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_docs = doc_masks.size(1)
        
        # Create attention mask (all 1s for valid tokens)
        # In practice, all tokens in our concatenated input are valid
        attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # 1. Predict importance scores (deltas from ImportanceScoringHead)
        # Cast to float32 for importance head (it expects float32)
        hidden_states_fp32 = hidden_states.float()
        importance_deltas = self.importance_head(hidden_states_fp32, attention_mask)  # [B, T, 1]
        importance_scores = importance_deltas.squeeze(-1)  # [B, T]
        
        # Convert deltas to scores (add 50 as baseline, since deltas are relative)
        # For contrastive learning, we only need relative weights, so deltas are fine
        importance_scores = torch.sigmoid(importance_scores) * 100  # Scale to [0, 100]
        
        # 2. Encode query (use original dtype for efficiency)
        query_emb = self.query_encoder(
            hidden_states=hidden_states,  # Keep in bfloat16
            attention_mask=query_mask,
        )  # [B, query_emb_dim]
        
        # 3. Pool documents (importance-weighted, use original dtype)
        doc_embs = []
        for i in range(num_docs):
            doc_mask = doc_masks[:, i]  # [B, T]
            
            # Importance scores for this doc (zero out non-doc tokens)
            doc_importance = importance_scores * doc_mask.float()  # [B, T]
            
            # Weighted sum of hidden states
            # Expand masks for broadcasting
            doc_mask_expanded = doc_mask.unsqueeze(-1).float()  # [B, T, 1]
            doc_importance_expanded = doc_importance.unsqueeze(-1)  # [B, T, 1]
            
            # doc_hidden: [B, T, D] masked to this document (use original dtype)
            doc_hidden = hidden_states * doc_mask_expanded.to(hidden_states.dtype)
            
            # Importance-weighted pooling
            weighted_hidden = doc_hidden * doc_importance_expanded  # [B, T, D]
            sum_weighted = weighted_hidden.sum(dim=1)  # [B, D]
            sum_importance = doc_importance.sum(dim=1, keepdim=True).clamp(min=1e-9)  # [B, 1]
            
            doc_emb = sum_weighted / sum_importance  # [B, D]
            doc_embs.append(doc_emb)
        
        doc_embs = torch.stack(doc_embs, dim=1)  # [B, N, D]
        
        # 4. Compute similarities (bilinear)
        # sim[b,i] = query_emb[b] @ similarity_matrix @ doc_embs[b,i]
        # Einsum: bd,de,bne->bn
        similarities = torch.einsum(
            'bd,de,bne->bn',
            query_emb,              # [B, query_emb_dim]
            self.similarity_matrix,  # [query_emb_dim, hidden_dim]
            doc_embs                # [B, N, hidden_dim]
        )  # [B, N]
        
        # Prepare output
        output = {
            'importance_scores': importance_scores,
            'similarities': similarities,
        }
        
        # Compute losses if labels provided
        if labels is not None:
            # Contrastive loss (InfoNCE / cross-entropy)
            logits = similarities / self.temperature  # [B, N]
            contrastive_loss = F.cross_entropy(logits, labels)
            
            # Contrastive accuracy (relevant doc has highest similarity)
            pred_labels = similarities.argmax(dim=1)
            contrastive_acc = (pred_labels == labels).float().mean()
            
            output['contrastive_loss'] = contrastive_loss
            output['contrastive_acc'] = contrastive_acc
            
            # Alignment loss (if oracle scores provided)
            if oracle_scores is not None:
                align_loss = F.mse_loss(importance_scores, oracle_scores)
                output['align_loss'] = align_loss
                
                # Combined loss
                total_loss = contrastive_loss + self.align_weight * align_loss
            else:
                output['align_loss'] = torch.tensor(0.0, device=hidden_states.device)
                total_loss = contrastive_loss
            
            output['loss'] = total_loss
        
        return output


def create_query_aware_model(
    base_model,
    importance_head,
    hidden_dim: int = 4096,
    query_emb_dim: int = 256,
    align_weight: float = 0.1,
    temperature: float = 0.07,
) -> nn.Module:
    """Factory function to create query-aware importance model.
    
    Args:
        base_model: Pre-trained base model (frozen)
        importance_head: ERT-trained importance head (trainable)
        hidden_dim: Model hidden dimension
        query_emb_dim: Query embedding dimension
        align_weight: Weight for alignment loss (λ)
        temperature: Temperature for contrastive loss (τ)
    
    Returns:
        QueryAwareImportanceHead module
    """
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Ensure importance head is trainable
    for param in importance_head.parameters():
        param.requires_grad = True
    
    # Create query-aware head
    model = QueryAwareImportanceHead(
        importance_head=importance_head,
        hidden_dim=hidden_dim,
        query_emb_dim=query_emb_dim,
        align_weight=align_weight,
        temperature=temperature,
    )
    
    return model

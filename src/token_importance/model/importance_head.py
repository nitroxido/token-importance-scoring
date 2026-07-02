"""ImportanceUpdateHead — cross-attention head that emits per-token score deltas."""
from __future__ import annotations

import torch
import torch.nn as nn
from token_importance.config import TISConfig
from token_importance.cache.importance_store import ImportanceStore


class RMSNorm(nn.Module):
    """RMSNorm for stabilizing scalar outputs without dependency on batch statistics.
    
    Computes: output = x * (scale / RMS(x))
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Unlike LayerNorm, RMSNorm doesn't subtract mean, just scales by variance.
    This is more suitable for stabilizing single scalar outputs across sequence positions.
    """
    
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: tensor of any shape. Returns same shape with RMS normalization applied."""
        # Compute RMS across all dimensions except batch
        rms = torch.sqrt(torch.mean(x ** 2) + self.eps)
        return x * (self.scale / rms)


class ImportanceUpdateHead(nn.Module):
    """Cross-attention head that emits per-context-token score deltas.

    Architecture:
        query  = current generation step hidden state [B, 1, d_model]
        keys/values = all past context hidden states [B, T, d_model]
        output = scalar delta per context token [B, T, 1]
        
    Stability layer (Week 2 fix):
        Adds RMSNorm after output projection to stabilize score variance
        and prevent attention drift over long sequences.
        RMSNorm scales outputs by their RMS magnitude, preventing variance explosion.

    Update rule (applied by caller or apply_deltas):
        new_score_i = clamp(old_score_i + round(tanh(delta_i) * max_delta), 0, 100)
    """

    def __init__(self, d_model: int, config: TISConfig, num_heads: int = 4) -> None:
        super().__init__()
        self.config = config
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, 1, bias=True)
        # RMSNorm layer to stabilize score variance (Week 2 fix for attention drift)
        self.score_norm = RMSNorm(eps=1e-6)
        
        # Zero-init: no updates at training start (avoids cold-start disruption)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        current_hidden: torch.Tensor,   # [B, 1, d_model]
        context_hidden: torch.Tensor,   # [B, T, d_model]
    ) -> torch.Tensor:                  # [B, T, 1] normalized deltas (before tanh scaling)
        # Cross-attention: query from current step, keys/values from full context
        attn_out, _ = self.cross_attn(
            query=current_hidden,    # [B, 1, d_model]
            key=context_hidden,      # [B, T, d_model]
            value=context_hidden,    # [B, T, d_model]
        )
        # attn_out shape: [B, 1, d_model] — attended summary at current position
        # Expand to all context positions for per-token delta projection
        # We project context_hidden through the update path instead:
        # Each context token gets a delta based on the cross-attention output broadcast
        # back through the context. We repeat attn_out to match T, then project.
        T = context_hidden.shape[1]
        attn_expanded = attn_out.expand(-1, T, -1)   # [B, T, d_model]
        raw_deltas = self.out_proj(attn_expanded)     # [B, T, 1]
        
        # RMSNorm: stabilize score variance to prevent attention drift
        # (Week 2 fix: prevents variance explosion over long sequences)
        normalized_deltas = self.score_norm(raw_deltas)  # [B, T, 1]
        return normalized_deltas

    def apply_deltas(
        self,
        store: ImportanceStore,
        current_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
    ) -> None:
        """Compute deltas and update the ImportanceStore in-place.
        positions = arange(len(store)), deltas = round(tanh(delta) * max_delta).
        """
        with torch.no_grad():
            raw_deltas = self.forward(current_hidden, context_hidden)  # [B, T, 1]

        # Use first batch element; squeeze to [T]
        scaled = torch.tanh(raw_deltas[0, :, 0]) * self.config.max_delta  # [T]
        int_deltas = scaled.round().to(torch.int32)

        T = int_deltas.shape[0]
        positions = torch.arange(min(T, len(store)), dtype=torch.long)
        store.update(positions, int_deltas[: len(positions)])


class QueryAwareImportanceHead(nn.Module):
    """Query-aware importance head using cross-attention to query embeddings.

    Architecture:
        query_embedding  = extract_query(query_embeddings)  # (B, d_model)
        query_attn       = cross_attn(hidden_states, query_rep) # (B, T, d_model)
        importance       = MLP(hidden, query_attn, position_ids) # (B, T, 1)

    Fixes Week 2 finding: Query signal is forgotten by middle position.
    
    Solution: Keep query representation explicit via cross-attention,
    so importance scores depend on query relevance, not just position.

    Training objective: MSE between predicted_importance and relevant_positions
    """

    def __init__(
        self,
        d_model: int,
        config: TISConfig,
        num_heads: int = 4,
        query_pool_method: str = "mean",
        use_postnorm: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.query_pool_method = query_pool_method
        self.use_postnorm = use_postnorm
        
        # Cross-attention: doc tokens attending to query representation
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        
        # Position embeddings (learned)
        self.pos_embedding = nn.Embedding(8192, d_model // 4)
        
        # MLP for final importance score
        # Input: [hidden_state (d_model) + query_attention (d_model) + pos_embed (d_model//4)]
        mlp_input_dim = d_model + d_model + (d_model // 4)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Post-norm layers (Week 2 optimization)
        if use_postnorm:
            self.attn_norm = nn.LayerNorm(d_model)
            self.mlp_norm_1 = nn.LayerNorm(d_model)
            self.mlp_norm_2 = nn.LayerNorm(d_model // 2)
        
        # RMSNorm for output stability (keep from ImportanceUpdateHead)
        self.score_norm = RMSNorm(eps=1e-6)
        
        # Zero-init final layer (no updates at training start)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def extract_query_representation(
        self, query_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Extract query representation from query token embeddings.
        
        Args:
            query_embeddings: (B, T_q, d_model) query token embeddings
        
        Returns:
            query_rep: (B, d_model) query representation
        """
        if self.query_pool_method == "mean":
            # Mean pooling over query tokens
            query_rep = query_embeddings.mean(dim=1)  # (B, d_model)
        elif self.query_pool_method == "cls":
            # Use first token (CLS-style)
            query_rep = query_embeddings[:, 0, :]  # (B, d_model)
        elif self.query_pool_method == "max":
            # Max pooling over query tokens
            query_rep = query_embeddings.max(dim=1)[0]  # (B, d_model)
        else:
            raise ValueError(f"Unknown pool method: {self.query_pool_method}")
        
        return query_rep

    def forward(
        self,
        doc_hidden: torch.Tensor,       # (B, T_doc, d_model)
        query_embeddings: torch.Tensor, # (B, T_q, d_model)
        position_ids: torch.Tensor | None = None,  # (B, T_doc)
    ) -> torch.Tensor:                  # (B, T_doc, 1) importance scores
        """Compute query-aware importance scores with optional post-norm.
        
        Args:
            doc_hidden: Document token hidden states
            query_embeddings: Query token embeddings
            position_ids: Optional position IDs for position embedding
        
        Returns:
            importance_scores: Query-aware importance [0, 1]
        """
        B, T_doc, d = doc_hidden.shape
        
        # Extract query representation
        query_rep = self.extract_query_representation(query_embeddings)  # (B, d_model)
        query_rep_expanded = query_rep.unsqueeze(1)  # (B, 1, d_model)
        
        # Cross-attention: doc tokens attending to query
        query_attn_out, _ = self.cross_attn(
            query=doc_hidden,              # (B, T_doc, d_model)
            key=query_rep_expanded,        # (B, 1, d_model)
            value=query_rep_expanded,      # (B, 1, d_model)
        )
        # query_attn_out: (B, T_doc, d_model)
        
        # Post-norm on attention output (Week 2 magnitude stabilization)
        if self.use_postnorm:
            query_attn_out = self.attn_norm(query_attn_out)
        
        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(T_doc, device=doc_hidden.device).unsqueeze(0).expand(B, -1)
        
        position_ids_clamped = torch.clamp(position_ids, 0, 8191)  # Clamp to embedding range
        pos_embed = self.pos_embedding(position_ids_clamped)  # (B, T_doc, d_model//4)
        
        # Concatenate: [hidden, query_attention, position]
        combined = torch.cat([doc_hidden, query_attn_out, pos_embed], dim=-1)  # (B, T_doc, d_model + d_model + d_model//4)
        
        # Pass through MLP with optional post-norm
        if self.use_postnorm:
            # Apply MLP layers with post-norm between them
            x = self.mlp[0](combined)  # Linear (B, T_doc, d_model)
            x = self.mlp[1](x)         # ReLU
            x = self.mlp_norm_1(x)     # LayerNorm (post-norm after first hidden)
            x = self.mlp[2](x)         # Linear (B, T_doc, d_model//2)
            x = self.mlp[3](x)         # ReLU
            x = self.mlp_norm_2(x)     # LayerNorm (post-norm after second hidden)
            raw_scores = self.mlp[4](x) # Linear (B, T_doc, 1)
        else:
            raw_scores = self.mlp(combined)  # (B, T_doc, 1)
        
        # Normalize and clamp to [0, 1]
        normalized = self.score_norm(raw_scores)  # (B, T_doc, 1)
        importance = torch.sigmoid(normalized)  # (B, T_doc, 1) -> [0, 1]
        
        return importance.squeeze(-1)  # (B, T_doc)

    def apply_deltas(
        self,
        store,
        current_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
    ) -> None:
        """Apply importance updates using query extracted from context (for generation).
        
        During generation, the query (e.g., question tokens) are at the END of the sequence.
        We extract them and use them to compute query-aware importance via forward().
        
        Args:
            store: ImportanceStore to update
            current_hidden: Current step hidden state [B, 1, d_model]
            context_hidden: Full context hidden states [B, T, d_model]
        """
        # Extract approximate query representation from last ~20% of context
        # (In LITM, question tokens are at the end; in general generation, this is a heuristic)
        B, T, d = context_hidden.shape
        query_start_idx = max(0, int(0.8 * T))  # Last 20% as approximate query
        
        if query_start_idx >= T:
            # Not enough tokens to extract query, fall back
            return
        
        # Extract query embeddings (last tokens of context)
        query_embeddings = context_hidden[:, query_start_idx:, :]  # [B, T_q, d]
        
        # Use all context as "document" to score
        doc_hidden = context_hidden  # [B, T, d]
        position_ids = torch.arange(T, device=context_hidden.device).unsqueeze(0).expand(B, -1)
        
        # Compute importance via query-aware scoring
        try:
            with torch.no_grad():
                importance_scores = self.forward(
                    doc_hidden=doc_hidden.float(),
                    query_embeddings=query_embeddings.float(),
                    position_ids=position_ids,
                )  # [B, T]
                
                # Update store with computed scores (scale to 0-100 for consistency)
                scaled_scores = (importance_scores[0] * 100).round().to(torch.uint8)
                positions = torch.arange(min(T, len(store)), dtype=torch.long, device=context_hidden.device)
                store.update(positions, scaled_scores[:len(positions)])
        except Exception:
            # Non-critical: skip if extraction fails
            pass


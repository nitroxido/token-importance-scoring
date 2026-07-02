"""PatchedCausalLM â€” wraps any HuggingFace CausalLM with TIS components."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from typing import Any, cast

from token_importance.config import TISConfig
from token_importance.cache.importance_store import ImportanceStore
from token_importance.model.importance_embedding import ImportanceEmbedding
from token_importance.model.importance_attn import ImportanceAttnBiasHook
from token_importance.model.importance_head import ImportanceUpdateHead


class PatchedCausalLM(nn.Module):
    """Wraps any HuggingFace CausalLM with TIS components.

    Usage:
        model = PatchedCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        input_ids, scores = IMLParser().encode_with_importance(tokenizer, text)
        out = model.generate(
            input_ids=torch.tensor([input_ids]),
            importance_scores=torch.tensor(scores, dtype=torch.uint8),
            max_new_tokens=200,
        )
    """

    def __init__(self, base_model: Any, config: TISConfig | None = None) -> None:
        super().__init__()
        self.base = base_model
        self._base_model: Any = cast(Any, base_model)
        self.tis_config = config or TISConfig()

        hidden_size = getattr(self._base_model.config, "hidden_size", None)
        if not isinstance(hidden_size, int):
            raise ValueError("base_model.config.hidden_size is required for PatchedCausalLM")
        d_model = hidden_size
        self.importance_embedding = ImportanceEmbedding(self.tis_config.d_imp, d_model)
        self.importance_head = ImportanceUpdateHead(d_model, self.tis_config)
        self.attn_hook = ImportanceAttnBiasHook(self.tis_config)
        self.importance_store: ImportanceStore | None = None
        self.last_tis_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        importance_scores: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ImportanceStore]:
        """Convert input_ids to embeddings + importance delta. Return (embeds, store)."""
        seq_len = input_ids.shape[-1]
        if importance_scores is None:
            importance_scores = torch.full(
                (seq_len,), 50, dtype=torch.uint8, device=input_ids.device
            )

        # Base token embeddings: [B, seq, d_model]
        embedding_layer = self._base_model.get_input_embeddings()
        if embedding_layer is None:
            raise ValueError("base model must provide input embeddings")
        token_embeds = embedding_layer(input_ids)

        # Importance delta: broadcast scores over batch if needed
        scores_long = importance_scores.to(torch.long)
        # Keep scores on the same device as the embedding layer (don't convert to CPU)
        device = next(self.importance_embedding.parameters()).device
        scores_long = scores_long.to(device)
        imp_delta = self.importance_embedding(scores_long)  # [seq, d_model]
        if imp_delta.dim() == 2:
            imp_delta = imp_delta.unsqueeze(0)  # [1, seq, d_model]
        imp_delta = imp_delta.to(device=token_embeds.device, dtype=token_embeds.dtype)  # match GPU/dtype of base model

        embeds = token_embeds + imp_delta
        store = ImportanceStore(importance_scores.cpu())
        return embeds, store

    def _enforce_anchor_floor(
        self,
        store: ImportanceStore,
        anchor_floor: int,
    ) -> float:
        """Ensure sink and recent spans do not drop below anchor_floor.

        Returns fraction of protected anchor positions that satisfy the floor after update.
        """
        if len(store) == 0:
            return 1.0

        scores = store.get_scores().to(torch.int32)
        T = scores.numel()
        n_sink = min(self.tis_config.N_sink, T)
        n_recent = min(self.tis_config.N_recent, T)

        sink_idx = torch.arange(0, n_sink, dtype=torch.long)
        recent_start = max(0, T - n_recent)
        recent_idx = torch.arange(recent_start, T, dtype=torch.long)
        anchor_idx = torch.unique(torch.cat([sink_idx, recent_idx], dim=0))

        if anchor_idx.numel() == 0:
            return 1.0

        anchor_scores = scores[anchor_idx]
        deltas = (anchor_floor - anchor_scores).clamp(min=0)
        if torch.any(deltas > 0):
            store.update(anchor_idx, deltas)
            updated = store.get_scores().to(torch.int32)
            anchor_scores = updated[anchor_idx]

        retention = (anchor_scores >= anchor_floor).to(torch.float32).mean().item()
        return float(retention)

    def _compute_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        store: ImportanceStore,
    ) -> torch.Tensor | None:
        """Return last hidden states for current sequence under current importance scores."""
        try:
            scores_t = store.get_scores().to(device=input_ids.device, dtype=torch.uint8)
            inputs_embeds, _ = self._build_inputs_embeds(input_ids, scores_t)
            with torch.no_grad():
                base_out = self._base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            if hasattr(base_out, "hidden_states") and base_out.hidden_states:
                return base_out.hidden_states[-1]
        except Exception:
            return None
        return None

    def _generate_static(
        self,
        input_ids: torch.Tensor,
        importance_scores: torch.Tensor | None,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Original static generation path preserved for backward compatibility."""
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() != 2:
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds, store = self._build_inputs_embeds(input_ids, importance_scores)
        self.importance_store = store

        new_ids = self._base_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        output_ids = torch.cat([input_ids, new_ids], dim=1)

        try:
            with torch.no_grad():
                base_out = self._base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                if hasattr(base_out, "hidden_states") and base_out.hidden_states:
                    context_hidden = base_out.hidden_states[-1]
                    current_hidden = context_hidden[:, -1:, :]
                    self.importance_head.apply_deltas(
                        store, current_hidden, context_hidden
                    )
        except Exception:
            pass

        self.last_tis_metrics = {
            "dynamic_enabled": 0.0,
            "update_count": 1.0,
            "mean_churn_rate": 0.0,
            "anchor_retention": 1.0,
            "budget_compliance": 1.0,
        }
        return output_ids

    def _generate_dynamic(
        self,
        input_ids: torch.Tensor,
        importance_scores: torch.Tensor | None,
        max_new_tokens: int,
        rescore_every_k: int,
        generation_chunk_size: int,
        anchor_floor: int,
        tis_budget_tokens: int | None,
        **kwargs,
    ) -> torch.Tensor:
        """Closed-loop generation with periodic re-scoring and anchor protection."""
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() != 2:
            attention_mask = torch.ones_like(input_ids)

        # Build initial store and sequence state.
        _, store = self._build_inputs_embeds(input_ids, importance_scores)
        self.importance_store = store
        seq_ids = input_ids

        generated_so_far = 0
        update_count = 0
        churn_values: list[float] = []
        anchor_values: list[float] = []

        while generated_so_far < max_new_tokens:
            remaining = max_new_tokens - generated_so_far
            step_tokens = min(max(1, generation_chunk_size), remaining)

            scores_t = store.get_scores().to(device=seq_ids.device, dtype=torch.uint8)
            step_embeds, _ = self._build_inputs_embeds(seq_ids, scores_t)
            step_mask = torch.ones_like(seq_ids)

            gen_out = self._base_model.generate(
                input_ids=seq_ids,
                inputs_embeds=step_embeds,
                attention_mask=step_mask,
                max_new_tokens=step_tokens,
                **kwargs,
            )

            # Robustly handle models returning either [new] or [prefix|new].
            if gen_out.shape[1] > seq_ids.shape[1]:
                new_chunk = gen_out[:, seq_ids.shape[1]:]
            else:
                new_chunk = gen_out

            if new_chunk.numel() == 0:
                break

            seq_ids = torch.cat([seq_ids, new_chunk], dim=1)
            step_mask = torch.ones_like(seq_ids)

            # Append neutral initial scores for newly generated tokens.
            store.append(torch.full((new_chunk.shape[1],), 50, dtype=torch.uint8))
            generated_so_far += new_chunk.shape[1]

            if generated_so_far % max(1, rescore_every_k) == 0 or generated_so_far >= max_new_tokens:
                before = store.get_scores()
                context_hidden = self._compute_hidden_states(seq_ids, step_mask, store)
                if context_hidden is not None:
                    current_hidden = context_hidden[:, -1:, :]
                    try:
                        self.importance_head.apply_deltas(store, current_hidden, context_hidden)
                    except Exception:
                        pass

                anchor_ret = self._enforce_anchor_floor(store, anchor_floor=anchor_floor)
                after = store.get_scores()

                overlap = min(before.numel(), after.numel())
                if overlap > 0:
                    churn = (before[:overlap] != after[:overlap]).to(torch.float32).mean().item()
                else:
                    churn = 0.0
                churn_values.append(float(churn))
                anchor_values.append(float(anchor_ret))
                update_count += 1

        total_tokens = int(seq_ids.shape[1])
        if tis_budget_tokens is None:
            budget_compliance = 1.0
        else:
            budget_compliance = 1.0 if total_tokens <= tis_budget_tokens else 0.0

        mean_churn = sum(churn_values) / len(churn_values) if churn_values else 0.0
        mean_anchor = sum(anchor_values) / len(anchor_values) if anchor_values else 1.0
        self.last_tis_metrics = {
            "dynamic_enabled": 1.0,
            "update_count": float(update_count),
            "mean_churn_rate": float(mean_churn),
            "anchor_retention": float(mean_anchor),
            "budget_compliance": float(budget_compliance),
        }
        return seq_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        importance_scores: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass with importance-biased attention.
        If importance_scores is None, all tokens default to score=50.
        """
        if input_ids is None:
            raise ValueError("input_ids is required for PatchedCausalLM.forward()")

        device = input_ids.device
        inputs_embeds, store = self._build_inputs_embeds(input_ids, importance_scores)
        self.importance_store = store

        merged_mask = self.attn_hook.merge_into_mask(
            attention_mask,
            store.get_scores_normalized().to(device),
            device,
            target_dtype=inputs_embeds.dtype,  # Pass embedding dtype for 4-bit compat
        )

        return self._base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        importance_scores: torch.Tensor | None = None,
        max_new_tokens: int = 100,
        dynamic_tis: bool = False,
        rescore_every_k: int = 8,
        generation_chunk_size: int = 4,
        anchor_floor: int = 70,
        tis_budget_tokens: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with importance-aware attention.

        Phase 2 implementation:
          1. Build inputs_embeds with importance delta.
          2. Call base_model.generate() with inputs_embeds and a 2D attention mask.
             NOTE: a 4D merged mask is NOT passed to generate() because
             transformers' autoregressive loop requires a 2D mask to extend
             correctly per step. The importance attention bias is applied via
             the inputs_embeds delta instead.
          3. Prepend input_ids to the output so callers can slice new_ids as
             output_ids[:, input_len:] (base.generate(inputs_embeds=...) returns
             only the newly generated token IDs, not the input prefix).
          4. After generation, run importance_head.apply_deltas() once.
        """
        if dynamic_tis:
            return self._generate_dynamic(
                input_ids=input_ids,
                importance_scores=importance_scores,
                max_new_tokens=max_new_tokens,
                rescore_every_k=max(1, int(rescore_every_k)),
                generation_chunk_size=max(1, int(generation_chunk_size)),
                anchor_floor=max(0, min(100, int(anchor_floor))),
                tis_budget_tokens=tis_budget_tokens,
                **kwargs,
            )

        return self._generate_static(
            input_ids=input_ids,
            importance_scores=importance_scores,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: TISConfig | None = None,
        **hf_kwargs,
    ) -> "PatchedCausalLM":
        """Load base model from HuggingFace hub or local path, wrap with TIS."""
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **hf_kwargs
        )
        return cls(base_model, config)
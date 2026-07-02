"""MS MARCO-style data loader for domain-mixed TIS training.

Loads real QA examples from MS MARCO with passage-level selection labels.
Generates weak token-level supervision:
  - selected passages → positive tokens
  - unselected passages → negative tokens
  - query tokens → always positive
  - answer spans → always positive

This enables training on real domain-aligned relevance signals while
maintaining the budgeted token eviction objective.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_from_disk


@dataclass
class MSMarcoRetrievalBatch:
    """MS MARCO training batch with weak passage-level supervision."""
    input_ids: torch.Tensor          # [1, T]
    attention_mask: torch.Tensor     # [1, T]
    evidence_mask: torch.Tensor      # [T] bool: selected passages + answer
    anchor_mask: torch.Tensor        # [T] bool: question + sink anchors
    learned_mask: torch.Tensor       # [T] bool: evidence | anchors
    answer_text: str
    budget: float
    is_real: bool = True             # marker: this is real data, not synthetic


class MSMarcoRetrievalDataset:
    """Iterable dataset mixing MS MARCO with synthetic examples.
    
    Usage:
        ds = MSMarcoRetrievalDataset(tokenizer, data_dir="data/msmarco_quick/train")
        for batch in ds:
            # batch is MSMarcoRetrievalBatch
    """

    def __init__(
        self,
        tokenizer,
        data_dir: str,
        context_tokens: int = 1536,
        budgets: list[float] | None = None,
        budget_weights: list[float] | None = None,
        max_passages: int = 6,
        seed: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.context_tokens = context_tokens
        self.budgets = budgets or [0.25, 0.5, 0.75]
        self.budget_weights = budget_weights
        self.max_passages = max_passages
        self._rng = random.Random(seed)

        # Load MS MARCO
        try:
            self._ds = load_from_disk(data_dir)
            # Filter to examples with real answers
            self._examples = [
                row
                for row in self._ds
                if row.get("answers") and row["answers"][0] not in ("No Answer Present.", "", None)
            ]
        except Exception as e:
            print(f"[warning] Could not load MS MARCO from {data_dir}: {e}")
            self._examples = []

    def __iter__(self) -> Iterator[MSMarcoRetrievalBatch]:
        """Infinite stream of MS MARCO retrieval batches."""
        while True:
            if not self._examples:
                raise RuntimeError(f"No examples loaded from {self.data_dir}")

            row = self._rng.choice(self._examples)
            query = row["query"]
            answer = row["answers"][0]

            # Extract passages and selection labels
            passages_dict = row.get("passages", {})
            texts = passages_dict.get("passage_text", []) if isinstance(passages_dict, dict) else []
            selected = passages_dict.get("is_selected", []) if isinstance(passages_dict, dict) else []

            if not texts:
                continue  # Skip examples with no passages

            # Sort by selection status (selected first) and limit
            ranked = sorted(zip(texts, selected), key=lambda x: -x[1])
            passage_texts = [t for t, _ in ranked[: self.max_passages]]
            passage_selected = [s for _, s in ranked[: self.max_passages]]

            # Build prompt: passages + query + Answer:
            context = " ".join(passage_texts)
            prompt = (
                f"Read the following passages and answer the question.\n\n"
                f"Passages: {context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )

            tok = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.context_tokens,
            )
            input_ids = tok["input_ids"]  # [1, T]
            attention_mask = tok["attention_mask"]  # [1, T]
            T = input_ids.shape[1]
            full_ids = input_ids[0].tolist()

            evidence_mask = torch.zeros(T, dtype=torch.bool)
            anchor_mask = torch.zeros(T, dtype=torch.bool)

            # Mark selected passages as evidence
            for passage_text, is_selected in zip(passage_texts, passage_selected):
                if not is_selected:
                    continue
                passage_ids = self.tokenizer.encode(passage_text, add_special_tokens=False)
                n_p = len(passage_ids)
                for start in range(T - n_p + 1):
                    if full_ids[start : start + n_p] == passage_ids:
                        evidence_mask[start : start + n_p] = True
                        break

            # Fallback: mark answer spans
            if not evidence_mask.any():
                answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
                n_a = len(answer_ids)
                for start in range(T - n_a + 1):
                    if full_ids[start : start + n_a] == answer_ids:
                        evidence_mask[start : start + n_a] = True
                        break

            # Mark query tokens as anchors (important for MS MARCO)
            query_ids = self.tokenizer.encode(query, add_special_tokens=False)
            n_q = len(query_ids)
            for start in range(T - n_q + 1):
                if full_ids[start : start + n_q] == query_ids:
                    anchor_mask[start : start + n_q] = True
                    break

            # Always protect sink + recent anchors
            n_anchor = min(4, T // 10)
            anchor_mask[:n_anchor] = True
            anchor_mask[-n_anchor:] = True

            learned_mask = evidence_mask | anchor_mask

            # Sample budget
            if self.budget_weights:
                budget = self._rng.choices(self.budgets, weights=self.budget_weights, k=1)[0]
            else:
                budget = self._rng.choice(self.budgets)

            yield MSMarcoRetrievalBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                evidence_mask=evidence_mask,
                anchor_mask=anchor_mask,
                learned_mask=learned_mask,
                answer_text=answer,
                budget=budget,
                is_real=True,
            )

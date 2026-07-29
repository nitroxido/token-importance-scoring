"""LITM-style retrieval dataset for Phase 4d training.

Each example has:
  - N key-value pairs as context (like the LostInMiddleBenchmark)
  - A question asking for the value of ONE specific key
  - Token-level labels indicating which tokens belong to the QUERIED KV pair

This gives a query-aware supervision signal:
  - relevant KV pair tokens → high importance (evidence_mask=True)
  - other KV pair tokens    → low importance
  - question tokens         → anchor (always kept)

The training loss rewards the importance_head for scoring relevant tokens higher.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


@dataclass
class LITMBatch:
    """One LITM-style training example."""
    input_ids: torch.Tensor          # [1, T]
    attention_mask: torch.Tensor     # [1, T]
    evidence_mask: torch.Tensor      # [T] bool: tokens of the QUERIED KV pair
    anchor_mask: torch.Tensor        # [T] bool: question + sink/recent tokens (always keep)
    budget: float
    answer_text: str


# ── Filler sentences between KV pairs ────────────────────────────────────────

_FILLER = [
    "The following information has been recorded for reference purposes.",
    "All entries are sorted alphabetically by identifier.",
    "This dataset contains structured lookup information.",
    "The values below have been verified by the system administrator.",
    "Please locate the correct value based on the key provided.",
    "Data integrity checks have been performed on this record set.",
    "The lookup table was last updated in the current session.",
    "Entries are indexed sequentially for efficient retrieval.",
    "The system maintains an up-to-date registry of all key-value pairs.",
    "Access is restricted to authorized personnel with valid credentials.",
]


def _random_key(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def _random_value(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choices(string.digits, k=length))


class LITMDataset(IterableDataset):
    """Infinite stream of LITM-style KV retrieval training examples.

    Each example:
    1. Generates n_pairs random key-value pairs
    2. Places the queried pair at a random position (beginning/middle/end)
    3. Tokenizes and annotates evidence_mask for the queried KV pair tokens
    4. Adds filler sentences between pairs to extend context

    The importance_head must learn to score the queried KV pair's tokens above
    all other KV pair tokens.
    """

    def __init__(
        self,
        tokenizer,
        n_pairs_options: list[int] | None = None,
        budgets: list[float] | None = None,
        budget_weights: list[float] | None = None,
        n_sink: int = 4,
        n_recent: int = 64,
        seed: int = 42,
        filler_sentences_per_pair: int = 1,
        max_seq_len: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_pairs_options = n_pairs_options or [10, 20, 40]
        self.budgets = budgets or [0.25, 0.5, 0.75]
        self.budget_weights = budget_weights
        self.n_sink = n_sink
        self.n_recent = n_recent
        self.seed = seed
        self.filler_sentences_per_pair = filler_sentences_per_pair
        self.max_seq_len = max_seq_len

    def _make_sample(
        self,
        rng: random.Random,
        n_pairs: int,
        query_idx: int,
        budget: float,
    ) -> LITMBatch | None:
        keys = [_random_key(rng) for _ in range(n_pairs)]
        values = [_random_value(rng) for _ in range(n_pairs)]

        # Build context: KV lines with optional filler between them
        context_parts: list[str] = ["Key-value pairs:\n"]
        for i, (k, v) in enumerate(zip(keys, values)):
            context_parts.append(f"Key '{k}': {v}\n")
            if self.filler_sentences_per_pair > 0 and i < n_pairs - 1:
                filler = rng.choice(_FILLER)
                context_parts.append(f"{filler}\n")

        question = f"\nWhat is the value for key '{keys[query_idx]}'? Answer:"

        context_str = "".join(context_parts)
        full_text = context_str + question

        ctx_toks = self.tokenizer.encode(context_str, add_special_tokens=False)
        q_toks = self.tokenizer.encode(question, add_special_tokens=False)
        all_toks = ctx_toks + q_toks
        T = len(all_toks)

        if T < 64:
            return None  # too short
        if T > self.max_seq_len:
            return None  # too long for GPU memory

        # Locate the queried KV pair's tokens within ctx_toks
        # We re-tokenize each KV line prefix up to the queried pair to find offset
        evidence_start = None
        evidence_end = None
        running = "Key-value pairs:\n"
        run_toks = self.tokenizer.encode(running, add_special_tokens=False)

        for i, (k, v) in enumerate(zip(keys, values)):
            line = f"Key '{k}': {v}\n"
            line_toks = self.tokenizer.encode(line, add_special_tokens=False)
            if i == query_idx:
                evidence_start = len(run_toks)
                evidence_end = evidence_start + len(line_toks)
                break
            run_toks = run_toks + line_toks
            if self.filler_sentences_per_pair > 0 and i < n_pairs - 1:
                f_line = f"{_FILLER[i % len(_FILLER)]}\n"
                f_toks = self.tokenizer.encode(f_line, add_special_tokens=False)
                run_toks = run_toks + f_toks

        if evidence_start is None or evidence_end is None or evidence_end > len(ctx_toks):
            return None

        # Build masks
        evidence_mask = torch.zeros(T, dtype=torch.bool)
        evidence_mask[evidence_start:evidence_end] = True

        anchor_mask = torch.zeros(T, dtype=torch.bool)
        anchor_mask[:self.n_sink] = True                    # sink tokens
        anchor_mask[max(0, T - self.n_recent):] = True      # recent tokens (question)
        # Mark question tokens as anchors explicitly
        anchor_mask[len(ctx_toks):] = True

        input_ids = torch.tensor([all_toks], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return LITMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            evidence_mask=evidence_mask,
            anchor_mask=anchor_mask,
            budget=budget,
            answer_text=values[query_idx],
        )

    def __iter__(self) -> Iterator[LITMBatch]:
        rng = random.Random(self.seed)
        idx = 0
        while True:
            n_pairs = rng.choice(self.n_pairs_options)
            query_idx = rng.randint(0, n_pairs - 1)
            budget_weights = self.budget_weights
            budget = rng.choices(self.budgets, weights=budget_weights, k=1)[0]
            sample = self._make_sample(rng, n_pairs, query_idx, budget)
            if sample is not None:
                yield sample
            idx += 1
            # Vary seed per iteration
            rng = random.Random(self.seed + idx * 1337)

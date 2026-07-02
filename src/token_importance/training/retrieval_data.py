"""Retrieval-sensitive data loader for closed-loop TIS training.

Builds long contexts mixing evidence-bearing passages with distractors.
Returns tokenized batches with token-level evidence positions so the
training loop can compute ranking and retrieval preservation losses.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


@dataclass
class RetrievalBatch:
    """Retrieval training batch with anchor-aware masks.
    
    evidence_mask: tokens that contain evidence (answer content)
    anchor_mask: tokens that MUST be kept (question + sink/recent anchors)
    learned_mask: tokens available for learned budget (evidence | anchors)
    """
    input_ids: torch.Tensor          # [1, T]
    attention_mask: torch.Tensor     # [1, T]
    evidence_mask: torch.Tensor      # [T] bool: evidence-bearing token
    anchor_mask: torch.Tensor        # [T] bool: hard-keep tokens (question + sinks)
    learned_mask: torch.Tensor       # [T] bool: evidence_mask | anchor_mask
    answer_text: str
    budget: float


# ── Filler sentences used as distractors ──────────────────────────────────────

_DISTRACTORS = [
    "The committee reviewed the proposal and decided to table the discussion until next month.",
    "Seasonal rainfall patterns influence agricultural yields across the northern hemisphere.",
    "The algorithm converges in polynomial time under the assumption of convexity.",
    "Historical records indicate that the city was founded in the early twelfth century.",
    "The membrane potential is maintained by active transport of sodium and potassium ions.",
    "Architectural styles varied significantly between the northern and southern provinces.",
    "Spectral analysis revealed absorption bands consistent with carbon dioxide in the atmosphere.",
    "The treaty was ratified by a majority of member states within three years of signing.",
    "Neural plasticity allows the adult brain to reorganize synaptic connections in response to experience.",
    "The expedition covered over four hundred kilometers of previously unmapped terrain.",
    "Market equilibrium is reached when the quantity supplied equals the quantity demanded.",
    "The sample was centrifuged at ten thousand revolutions per minute for fifteen minutes.",
    "Religious observances during the festival period differ by region and denomination.",
    "The compiler performs dead code elimination before generating the final binary.",
    "Archaeological excavations at the site uncovered pottery dating to the Bronze Age.",
    "The coefficient of thermal expansion determines how materials respond to temperature changes.",
    "Public transit ridership declined sharply following the introduction of remote work policies.",
    "The manuscript was transcribed by monks in the ninth century and later translated into Latin.",
    "Statistical significance does not imply practical significance in real-world applications.",
    "The pilot adjusted the flight path to avoid the frontal system moving in from the west.",
]

# ── Evidence templates ─────────────────────────────────────────────────────────

_EVIDENCE_TEMPLATES = [
    ("The special code required for authorization is {answer}. Keep this value confidential.",
     "{answer}"),
    ("According to the report, the primary identifier is {answer}, recorded on the first page.",
     "{answer}"),
    ("The critical threshold value has been set to {answer} by the research team.",
     "{answer}"),
    ("Lab result ID {answer} corresponds to the positive sample retrieved on Tuesday.",
     "{answer}"),
    ("The configuration parameter must be set to {answer} for the system to operate correctly.",
     "{answer}"),
    ("The key fact established in the study is that the answer is {answer}.",
     "{answer}"),
    ("Document reference {answer} contains the supporting evidence for this claim.",
     "{answer}"),
    ("The unique identifier assigned to this case is {answer}.",
     "{answer}"),
]

# ── Answers ────────────────────────────────────────────────────────────────────

_ANSWERS = [
    "ALPHA-7791", "BRAVO-3342", "CHARLIE-9981", "DELTA-1123", "ECHO-5567",
    "FOXTROT-8834", "GOLF-2219", "HOTEL-4456", "INDIA-7723", "JULIET-3391",
    "KILO-6612", "LIMA-8847", "MIKE-2234", "NOVEMBER-5571", "OSCAR-9918",
    "PAPA-1145", "QUEBEC-4423", "ROMEO-7756", "SIERRA-3389", "TANGO-6617",
]


class RetrievalDataset(IterableDataset):
    """Infinite stream of retrieval examples.

    Each example is a long context that mixes one evidence passage with
    distractor sentences.  The evidence passage contains the answer; the
    distractors do not.

    Args:
        tokenizer: HuggingFace tokenizer.
        context_tokens: Target sequence length in tokens.
        budgets: List of cache budgets to sample per example.
        evidence_position: 'random', 'first', 'last', or 'middle'.
        min_distractors: Minimum number of distractor sentences.
        seed: Random seed for reproducibility. None = truly random.
    """

    def __init__(
        self,
        tokenizer,
        context_tokens: int = 2048,
        budgets: list[float] | None = None,
        budget_weights: list[float] | None = None,
        evidence_position: str = "random",
        seed: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.budgets = budgets or [0.25, 0.5, 0.75]
        # Non-uniform budget sampling: e.g. [4, 1, 1] trains 4× more at 25% budget
        self.budget_weights = budget_weights
        self.evidence_position = evidence_position
        self._rng = random.Random(seed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_context(self) -> tuple[str, str, str]:
        """Return (full_context, evidence_sentence, answer)."""
        answer = self._rng.choice(_ANSWERS)
        template, _ = self._rng.choice(_EVIDENCE_TEMPLATES)
        evidence = template.format(answer=answer)

        # Pool of distractor sentences, shuffled each time
        distractors = _DISTRACTORS[:]
        self._rng.shuffle(distractors)

        # Decide where to insert evidence
        n_dist = len(distractors)
        if self.evidence_position == "first":
            pos = 0
        elif self.evidence_position == "last":
            pos = n_dist
        elif self.evidence_position == "middle":
            pos = n_dist // 2
        else:  # random
            pos = self._rng.randint(0, n_dist)

        sentences = distractors[:pos] + [evidence] + distractors[pos:]
        context = " ".join(sentences)
        return context, evidence, answer

    def _tokenize_example(
        self,
        context: str,
        evidence: str,
        answer: str,
        budget: float,
    ) -> RetrievalBatch:
        """Tokenize context and mark important token positions.

        Separates:
        - evidence_mask: answer content tokens only
        - anchor_mask: question + sink/recent anchors (NEVER evict)
        - learned_mask: evidence | anchors (all important tokens)
        """
        prompt = (
            f"Read the following text carefully and answer the question.\n\n"
            f"Text: {context}\n\n"
            f"Question: What is the special identifier or key value mentioned?\n"
            f"Answer:"
        )
        question_marker = "Question: What is the special identifier or key value mentioned?"

        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_tokens,
        )
        input_ids = tok["input_ids"]        # [1, T]
        attention_mask = tok["attention_mask"]  # [1, T]
        T = input_ids.shape[1]

        full_ids = input_ids[0].tolist()
        evidence_mask = torch.zeros(T, dtype=torch.bool)
        anchor_mask = torch.zeros(T, dtype=torch.bool)

        # Mark evidence span (answer content only)
        evidence_ids = self.tokenizer.encode(evidence, add_special_tokens=False)
        n_ev = len(evidence_ids)
        for start in range(T - n_ev + 1):
            if full_ids[start : start + n_ev] == evidence_ids:
                evidence_mask[start : start + n_ev] = True
                break

        # Fallback to answer substring if full sentence not found
        if not evidence_mask.any():
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            n_an = len(answer_ids)
            for start in range(T - n_an + 1):
                if full_ids[start : start + n_an] == answer_ids:
                    evidence_mask[start : start + n_an] = True
                    break

        # Mark question span as ANCHOR (must always keep)
        question_ids = self.tokenizer.encode(question_marker, add_special_tokens=False)
        n_q = len(question_ids)
        for start in range(T - n_q + 1):
            if full_ids[start : start + n_q] == question_ids:
                anchor_mask[start : start + n_q] = True
                break

        # Always protect sink tokens (first 4) and recent tokens (last 4) as ANCHORS
        # These are critical for LLM attention to work correctly
        n_anchor = min(4, T // 10)
        anchor_mask[:n_anchor] = True
        anchor_mask[-n_anchor:] = True

        # learned_mask = all important tokens (evidence + anchors compete for learned budget)
        learned_mask = evidence_mask | anchor_mask

        return RetrievalBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            evidence_mask=evidence_mask,
            anchor_mask=anchor_mask,
            learned_mask=learned_mask,
            answer_text=answer,
            budget=budget,
        )

    # ── IterableDataset interface ──────────────────────────────────────────────

    def __iter__(self) -> Iterator[RetrievalBatch]:
        while True:
            context, evidence, answer = self._build_context()
            if self.budget_weights:
                budget = self._rng.choices(self.budgets, weights=self.budget_weights, k=1)[0]
            else:
                budget = self._rng.choice(self.budgets)
            yield self._tokenize_example(context, evidence, answer, budget)

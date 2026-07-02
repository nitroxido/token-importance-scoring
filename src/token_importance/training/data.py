"""TIS training data pipeline.

Loads long-context QA datasets, assigns per-token pseudo-importance labels, and
provides a PyTorch Dataset for the training loop.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from token_importance.markup.parser import IMLParser
from token_importance.cache.dataset_cache import get_cache

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS: dict[str, tuple[str, str]] = {
    "narrativeqa": ("deepmind/narrativeqa", "train"),
    "quality":     ("emozilla/quality",     "train"),
    "qasper":      ("allenai/qasper",       "train"),
    "ms_marco":    ("microsoft/ms_marco",   "v1.1:train"),
}


def load_training_dataset(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    use_cache: bool = True,
    cache_root: str | Path | None = None,
):
    """Load a supported long-context QA dataset from HuggingFace.

    Returns a HuggingFace ``Dataset`` object. Pass ``max_samples`` to cap the
    number of items (useful for smoke tests and debugging).
    
    Args:
        name: Dataset name (one of 'narrativeqa', 'quality', 'qasper', 'ms_marco')
        split: Dataset split (default 'train')
        max_samples: Optional limit on number of samples
        use_cache: If True, cache dataset to disk after first download
        cache_root: Custom cache directory. If None, uses default ~/.cache/tis_datasets
    
    Raises ``ValueError`` for unknown dataset names.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: {list(SUPPORTED_DATASETS.keys())}"
        )

    dataset_id, default_split_with_config = SUPPORTED_DATASETS[name]
    
    # Handle "config:split" format (for MS-MARCO and others that need config)
    config = None
    if ":" in default_split_with_config:
        config, actual_split = default_split_with_config.split(":", 1)
    else:
        actual_split = default_split_with_config
    
    # Override with explicit split if provided
    if split != "train":
        actual_split = split
    
    if use_cache:
        cache = get_cache(cache_root)
        return cache.load_cached_dataset(
            dataset_id,
            split=actual_split,
            config=config,
            max_samples=max_samples,
            auto_cache=True,
        )
    else:
        # Load without caching (original behavior)
        from datasets import load_dataset
        if config:
            ds = load_dataset(dataset_id, config, split=actual_split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_id, split=actual_split, trust_remote_code=True)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds


# ---------------------------------------------------------------------------
# Field extraction per dataset
# ---------------------------------------------------------------------------

def _extract_narrativeqa(item: dict) -> tuple[str, str, str] | None:
    """Extract (passage, question, answer) from a NarrativeQA item."""
    try:
        passage  = item["document"]["text"]
        question = item["question"]["text"]
        answers  = item.get("answers", [])
        answer   = answers[0]["text"] if answers else ""
        return passage, question, answer
    except (KeyError, IndexError, TypeError):
        return None


def _extract_quality(item: dict) -> tuple[str, str, str] | None:
    """Extract (passage, question, answer) from a QuALITY item.

    QuALITY is multiple-choice; the answer is the selected option text.
    """
    try:
        passage  = item.get("article", item.get("document", ""))
        qs       = item.get("questions", [])
        if not qs:
            return None
        q_obj    = qs[0] if isinstance(qs[0], dict) else {"question": qs[0]}
        question = q_obj.get("question", str(qs[0]))
        options  = q_obj.get("options", [])
        ans_idx  = item.get("answer", item.get("answers", [0]))[0]
        answer   = options[ans_idx] if options and isinstance(ans_idx, int) else ""
        return passage, question, answer
    except (KeyError, IndexError, TypeError):
        return None


def _extract_qasper(item: dict) -> tuple[str, str, str] | None:
    """Extract first available QA pair from a Qasper item."""
    try:
        title    = item.get("title", "")
        abstract = item.get("abstract", "")
        passage  = f"{title}\n\n{abstract}"
        qs       = item.get("qas", [])
        if not qs:
            return None
        qa       = qs[0]
        question = qa.get("question", "")
        ans_list = qa.get("answers", [])
        answer   = ""
        for a in ans_list:
            text = a.get("answer", {}).get("free_form_answer", "")
            if text:
                answer = text
                break
        return passage, question, answer
    except (KeyError, IndexError, TypeError):
        return None


def _extract_ms_marco(item: dict) -> tuple[str, str, str] | None:
    """Extract QA pair from MS-MARCO item with selected passage.
    
    MS-MARCO includes explicit passage relevance annotations via 'is_selected'.
    We use the selected passage (the one marked as relevant) which provides
    much better position labels than heuristic-based approaches.
    """
    try:
        question = item.get("query", "")
        if not question:
            return None
        
        # Find the selected passage (explicitly marked as relevant)
        passages = item.get("passages", [])
        selected_passage = None
        
        for passage_obj in passages:
            # Handle both dict and string passage formats
            if isinstance(passage_obj, dict):
                if passage_obj.get("is_selected"):
                    selected_passage = passage_obj.get("passage_text", "")
                    break
            elif isinstance(passage_obj, str):
                # If passages are just strings, check if this one is marked
                # This is a fallback for different MS-MARCO formats
                selected_passage = passage_obj
                break
        
        # Fallback: use first passage if none marked
        if not selected_passage and passages:
            first = passages[0]
            if isinstance(first, dict):
                selected_passage = first.get("passage_text", "")
            else:
                selected_passage = str(first)
        
        if not selected_passage:
            return None
        
        # Extract answer
        answers = item.get("answers", [])
        answer = answers[0] if answers else ""
        
        return selected_passage, question, answer
    except (KeyError, IndexError, TypeError):
        return None


_EXTRACTORS: dict[str, Callable] = {
    "narrativeqa": _extract_narrativeqa,
    "quality":     _extract_quality,
    "qasper":      _extract_qasper,
    "ms_marco":    _extract_ms_marco,
}


def extract_fields(
    item: dict,
    dataset_name: str = "narrativeqa",
) -> tuple[str, str, str] | None:
    """Dispatch to the correct field extractor.  Returns None on failure."""
    extractor = _EXTRACTORS.get(dataset_name, _extract_narrativeqa)
    return extractor(item)


# ---------------------------------------------------------------------------
# Pseudo-label span generation
# ---------------------------------------------------------------------------

def extract_qa_spans(
    passage: str,
    answer: str,
) -> list[tuple[int, int, int]]:
    """Identify important spans in *passage* relative to *answer*.

    Returns a list of non-overlapping ``(char_start, char_end, score)`` tuples
    suitable for use with ``IMLParser.encode_with_importance``.

    Scoring rules:
      - Exact answer substring     → score 80
      - Answer keywords (≥ 4 chars) that are not yet covered → score 70
      - Spacy noun-chunks (optional) that are not yet covered → score 65

    Uncovered regions get the IMLParser default (50) implicitly.
    """
    spans: list[tuple[int, int, int]] = []
    # covered[i] = True means char i is already part of a span
    covered: list[bool] = [False] * len(passage)

    def _mark(start: int, end: int) -> None:
        for i in range(start, min(end, len(passage))):
            covered[i] = True

    def _already_covered(start: int, end: int) -> bool:
        return any(covered[i] for i in range(start, min(end, len(passage))))

    # 1 — Exact answer substring(s): score 80
    if answer:
        idx = passage.find(answer)
        while idx != -1:
            spans.append((idx, idx + len(answer), 80))
            _mark(idx, idx + len(answer))
            idx = passage.find(answer, idx + 1)

    # 2 — Answer keywords (words ≥ 4 chars): score 70
    keywords = [w for w in re.split(r"\W+", answer) if len(w) >= 4]
    for kw in keywords:
        idx = passage.find(kw)
        while idx != -1:
            end = idx + len(kw)
            if not _already_covered(idx, end):
                spans.append((idx, end, 70))
                _mark(idx, end)
            idx = passage.find(kw, idx + 1)

    # 3 — Spacy noun chunks (optional): score 65
    try:
        import spacy  # type: ignore[import]
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(passage[:4096])  # cap to avoid very long passages
        for chunk in doc.noun_chunks:
            s, e = chunk.start_char, chunk.end_char
            if not _already_covered(s, e):
                spans.append((s, e, 65))
                _mark(s, e)
    except (ImportError, OSError):
        pass  # spacy not available — skip NP extraction

    spans.sort(key=lambda x: x[0])
    return spans


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def build_training_sample(
    passage: str,
    question: str,
    answer: str,
    tokenizer,
    max_length: int = 2048,
) -> dict | None:
    """Build a single training sample.

    Format: ``[QUESTION tokens] + [PASSAGE tokens] + [ANSWER tokens]``

    Returns a dict with keys:
      - ``input_ids``         int64  [seq_len]
      - ``importance_scores`` uint8  [seq_len]
      - ``labels``            int64  [seq_len]  (−100 outside answer)
      - ``attention_mask``    int64  [seq_len]

    Returns ``None`` if the tokenized length exceeds *max_length*.
    """
    # --- tokenise each part ---
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    a_ids = tokenizer.encode(answer,   add_special_tokens=False)

    # Passage importance: use IMLParser for proper offset alignment when possible
    parser = IMLParser()
    use_fast_alignment = getattr(tokenizer, "is_fast", False) and passage
    if use_fast_alignment:
        try:
            passage_spans = extract_qa_spans(passage, answer)
            p_ids, p_scores_arr = parser.encode_with_importance(
                tokenizer, passage, spans=passage_spans
            )
            p_scores_arr = p_scores_arr.astype(np.uint8)
        except Exception:
            p_ids = tokenizer.encode(passage, add_special_tokens=False)
            p_scores_arr = np.full(len(p_ids), 50, dtype=np.uint8)
    else:
        p_ids = tokenizer.encode(passage, add_special_tokens=False)
        # Approximate: map character-level spans to token positions by ratio
        p_scores_arr = _approx_passage_scores(passage, p_ids, answer)

    # --- length check / passage truncation ---
    # Reserve space for question + answer; truncate the passage to fit.
    max_passage_len = max_length - len(q_ids) - len(a_ids)
    if max_passage_len <= 0:
        return None   # question + answer alone exceed max_length
    if len(p_ids) > max_passage_len:
        p_ids = p_ids[:max_passage_len]
        p_scores_arr = p_scores_arr[:max_passage_len]
    total_len = len(q_ids) + len(p_ids) + len(a_ids)

    # --- build importance arrays ---
    q_scores = np.full(len(q_ids), 70, dtype=np.uint8)   # question is the goal
    a_scores = np.full(len(a_ids), 50, dtype=np.uint8)   # answer is the target output

    input_ids         = q_ids + list(p_ids) + a_ids
    importance_scores = np.concatenate([q_scores, p_scores_arr, a_scores])
    labels            = [-100] * (len(q_ids) + len(p_ids)) + a_ids
    attention_mask    = [1] * total_len

    return {
        "input_ids":         torch.tensor(input_ids,         dtype=torch.long),
        "importance_scores": torch.tensor(importance_scores, dtype=torch.uint8),
        "labels":            torch.tensor(labels,            dtype=torch.long),
        "attention_mask":    torch.tensor(attention_mask,    dtype=torch.long),
    }


def _approx_passage_scores(
    passage: str,
    p_ids: list[int],
    answer: str,
) -> np.ndarray:
    """Approximate per-token importance using character-position ratio.

    Used when offset mapping is unavailable (slow tokenizer or non-fast path).
    """
    n_chars  = max(len(passage), 1)
    n_tokens = len(p_ids)
    scores   = np.full(n_tokens, 50, dtype=np.uint8)
    for start, end, score in extract_qa_spans(passage, answer):
        tok_start = int(start / n_chars * n_tokens)
        tok_end   = min(int(end   / n_chars * n_tokens) + 1, n_tokens)
        scores[tok_start:tok_end] = score
    return scores


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def collate_skip_none(batch: list) -> dict | None:
    """Collate function that silently drops None items.

    Pass this as ``collate_fn`` to ``torch.utils.data.DataLoader`` when using
    ``TISTrainingDataset``.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TISTrainingDataset(Dataset):
    """Wraps a HuggingFace dataset and produces TIS training samples on the fly.

    Args:
        hf_dataset:   A HuggingFace ``Dataset`` or any object that supports
                      ``len()`` and ``__getitem__(idx) -> dict``.
        tokenizer:    A HuggingFace tokenizer (fast preferred for exact alignment).
        max_length:   Maximum sequence length; longer samples are returned as None.
        dataset_name: Used to select the correct field extractor.
                      One of ``"narrativeqa"``, ``"quality"``, ``"qasper"``.
        extract_fn:   Optional override for field extraction. Receives an item dict
                      and must return ``(passage, question, answer)`` or ``None``.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 2048,
        dataset_name: str = "narrativeqa",
        extract_fn: Callable | None = None,
    ) -> None:
        self._data        = hf_dataset
        self._tokenizer   = tokenizer
        self._max_length  = max_length
        self._dataset_name = dataset_name
        self._extract_fn  = extract_fn or (lambda item: extract_fields(item, dataset_name))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict | None:
        item   = self._data[idx]
        fields = self._extract_fn(item)
        if fields is None:
            return None
        passage, question, answer = fields
        return build_training_sample(
            passage, question, answer,
            self._tokenizer,
            self._max_length,
        )

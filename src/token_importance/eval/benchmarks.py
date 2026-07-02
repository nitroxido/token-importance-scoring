"""Benchmark harness for Token Importance Score (TIS).

Three benchmarks:
  NIAHBenchmark         Needle-in-a-Haystack retrieval
  LostInMiddleBenchmark Key-value retrieval (Liu et al. 2023)
  MultiDocQABenchmark   Multiple-choice QA on QuALITY (emozilla/quality)
"""
from __future__ import annotations

import random
import string
import torch
import numpy as np
from typing import Optional

from token_importance.config import TISConfig


# ─── Haystack corpus ──────────────────────────────────────────────────────────

_HAYSTACK_SENTENCES = [
    "The conference was held at the downtown convention center last September.",
    "Scientists discovered a new mineral compound in the deep ocean trenches.",
    "The library committee approved a new book acquisition policy for next year.",
    "According to recent reports, urban transportation funding will increase.",
    "The museum exhibit on ancient Rome attracted thousands of visitors.",
    "Researchers published findings on migratory bird patterns in the north.",
    "The community garden project received support from local businesses.",
    "Historical records show that the bridge was constructed in the 1900s.",
    "Environmental studies indicate a gradual shift in regional climate patterns.",
    "The annual harvest festival brings together farmers from nearby counties.",
    "A new software update addressed several security vulnerabilities.",
    "The orchestra performed classical compositions from the Romantic period.",
    "Urban planners proposed a redesign of the central park area.",
    "The hospital added a new wing dedicated to pediatric care services.",
    "Students from three universities collaborated on the robotics competition.",
    "The archaeological dig uncovered artifacts from multiple historical periods.",
    "Local authorities announced plans for road maintenance this season.",
    "The cooking competition featured chefs from fifteen different countries.",
    "Astronomers captured images of a distant nebula using new telescopes.",
    "The theater company announced twelve productions for its upcoming season.",
    "The town council voted in favour of the new zoning regulations yesterday.",
    "Passenger rail ridership increased significantly over the last quarter.",
    "The new bridge connecting the two districts will open next spring.",
    "Farmers reported above-average yields owing to the mild summer weather.",
    "The film festival showcased works from directors across four continents.",
]


def _generate_haystack_tokens(n_tokens: int, tokenizer, rng: random.Random) -> list[int]:
    """Generate approximately *n_tokens* of padding/filler token IDs."""
    tokens: list[int] = []
    while len(tokens) < n_tokens:
        sentence = rng.choice(_HAYSTACK_SENTENCES)
        tokens.extend(tokenizer.encode(sentence, add_special_tokens=False))
    return tokens[:n_tokens]


# ─── Model introspection helpers ──────────────────────────────────────────────

def _is_patched(model) -> bool:
    """True when *model* is a PatchedCausalLM instance."""
    return hasattr(model, "tis_config") and hasattr(model, "importance_store")


def _get_policy(model) -> str:
    """Return the eviction policy name attached by the eval script."""
    return getattr(model, "_baseline_policy", "vanilla")


# ─── Token budget selection ───────────────────────────────────────────────────

def _select_token_budget(
    input_ids: torch.Tensor,        # [1, T]
    importance_scores: np.ndarray,  # [T] uint8
    budget_tokens: int,
    policy: str,
    model=None,
    n_sink: int = 4,
    n_recent: int = 64,
) -> tuple[torch.Tensor, np.ndarray]:
    """Select *budget_tokens* from *input_ids* according to *policy*.

    Returns (selected_ids [1, budget], selected_scores [budget]).
    If *budget_tokens* >= T the inputs are returned unchanged.
    """
    T = input_ids.shape[1]
    if budget_tokens >= T:
        return input_ids, importance_scores

    budget_tokens = max(budget_tokens, n_sink + n_recent + 1)

    if policy == "tis":
        # Keep highest-importance tokens; sinks and recency window are protected.
        scores_f = importance_scores.astype(float).copy()
        scores_f[:n_sink] = 201.0
        scores_f[max(0, T - n_recent):] = 201.0
        keep_idx = np.sort(np.argsort(scores_f)[::-1][:budget_tokens])

    elif policy == "streamingllm":
        n_last = max(0, budget_tokens - n_sink)
        front = np.arange(min(n_sink, T))
        back_start = max(int(front[-1]) + 1 if len(front) else 0, T - n_last)
        back = np.arange(back_start, T)
        keep_idx = np.sort(np.unique(np.concatenate([front, back])))[:budget_tokens]

    elif policy == "h2o":
        keep_idx = _h2o_keep_indices(model, input_ids, budget_tokens, n_sink, n_recent)

    elif policy == "snapkv":
        keep_idx = _snapkv_keep_indices(model, input_ids, budget_tokens, n_sink, n_recent)

    elif policy == "infini_attention":
        keep_idx = _infini_attention_keep_indices(
            model, input_ids, budget_tokens, n_sink, n_recent
        )

    else:  # vanilla: keep the most recent budget_tokens
        keep_idx = np.arange(T - budget_tokens, T)

    keep_tensor = torch.tensor(keep_idx, dtype=torch.long, device=input_ids.device)
    return input_ids[:, keep_tensor], importance_scores[keep_idx]


def _get_cumulative_attention(
    model,
    input_ids: torch.Tensor,
    T: int,
) -> tuple[np.ndarray, list]:
    """Run a single forward pass with output_attentions=True.

    Returns (cumulative_magnitudes [T], list_of_attention_tensors).
    Falls back to uniform magnitudes on any error (e.g. FlashAttention models).
    """
    magnitudes = np.zeros(T, dtype=np.float32)
    attn_tensors: list = []
    try:
        base = model.base if _is_patched(model) else model
        with torch.no_grad():
            out = base(input_ids, output_attentions=True)
        if out.attentions:
            attn_tensors = list(out.attentions)
            attn_sum = torch.zeros(T, device=input_ids.device)
            for layer_attn in attn_tensors:
                attn_sum = attn_sum + layer_attn[0].sum(dim=(0, 1))
            magnitudes = attn_sum.cpu().numpy()
    except Exception:
        pass  # fall back to uniform → degrades gracefully
    return magnitudes, attn_tensors


def _h2o_keep_indices(
    model,
    input_ids: torch.Tensor,
    budget: int,
    n_sink: int,
    n_recent: int,
) -> np.ndarray:
    """Run a forward pass to get cumulative attention, return indices to *keep*."""
    T = input_ids.shape[1]
    magnitudes, _ = _get_cumulative_attention(model, input_ids, T)

    from token_importance.eval.baselines import H2OEvictionPolicy
    policy_obj = H2OEvictionPolicy()
    evict = policy_obj.select_indices_to_evict(
        torch.tensor(magnitudes, dtype=torch.float32), T, budget, n_sink, n_recent
    )
    evict_set = set(evict.tolist())
    return np.array([i for i in range(T) if i not in evict_set])


def _snapkv_keep_indices(
    model,
    input_ids: torch.Tensor,
    budget: int,
    n_sink: int,
    n_recent: int,
    n_query: int = 64,
) -> np.ndarray:
    """SnapKV: pool attention from the last n_query tokens over the context."""
    T = input_ids.shape[1]
    _, attn_tensors = _get_cumulative_attention(model, input_ids, T)

    from token_importance.eval.baselines import SnapKVEvictionPolicy
    policy_obj = SnapKVEvictionPolicy()
    # Pass T so compute_scores can return correct shape even if attn_tensors is empty
    scores = policy_obj.compute_scores(attn_tensors, n_query=min(n_query, T), T=T)
    return policy_obj.select_indices_to_keep(scores, T, budget, n_sink, n_recent)


def _infini_attention_keep_indices(
    model,
    input_ids: torch.Tensor,
    budget: int,
    n_sink: int,
    n_recent: int,
) -> np.ndarray:
    """Infini-Attention approximation: weighted sample from compressed prefix."""
    T = input_ids.shape[1]
    magnitudes, _ = _get_cumulative_attention(model, input_ids, T)

    # Ensure magnitudes has correct shape (should be [T] from _get_cumulative_attention)
    if len(magnitudes) != T:
        magnitudes = np.ones(T, dtype=np.float32)  # fallback to uniform if shape mismatch

    from token_importance.eval.baselines import InfiniAttentionEvictionPolicy
    policy_obj = InfiniAttentionEvictionPolicy()
    return policy_obj.select_indices_to_keep(
        torch.tensor(magnitudes, dtype=torch.float32), T, budget, n_sink, n_recent
    )


# ─── Inference helper ─────────────────────────────────────────────────────────

def _run_generation(
    model,
    input_ids: torch.Tensor,
    importance_scores: np.ndarray,
    tokenizer,
    max_new_tokens: int = 30,
    generation_kwargs: Optional[dict] = None,
) -> str:
    """Run generation and return decoded new tokens only."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
    )

    if _is_patched(model):
        scores_t = torch.tensor(importance_scores, dtype=torch.uint8, device=device)
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)
        output_ids = model.generate(input_ids, importance_scores=scores_t, **gen_kwargs)
    else:
        output_ids = model.generate(input_ids, **gen_kwargs)

    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─── NIAHBenchmark ────────────────────────────────────────────────────────────

class NIAHBenchmark:
    """Needle in a Haystack.

    Places a short key fact at a controlled fractional depth inside a long
    filler passage, then asks the model to retrieve it.  Accuracy is the
    fraction of samples where the needle value appears in the generated answer.
    """

    def __init__(
        self,
        context_lengths: list[int] | None = None,
        depths: list[float] | None = None,
        n_samples: int = 10,
        needle_score: int = 90,
        haystack_score: int = 30,
    ) -> None:
        self.context_lengths = context_lengths or [1024, 2048, 4096]
        self.depths = depths or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.n_samples = n_samples
        self.needle_score = needle_score
        self.haystack_score = haystack_score

    def _make_sample(
        self,
        tokenizer,
        context_length: int,
        depth: float,
        seed: int,
    ) -> tuple[torch.Tensor, np.ndarray, str]:
        rng = random.Random(seed)
        needle_word = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        needle_text = f" The secret code is {needle_word}."
        needle_toks = tokenizer.encode(needle_text, add_special_tokens=False)

        question_text = " What is the secret code? Answer:"
        question_toks = tokenizer.encode(question_text, add_special_tokens=False)

        haystack_budget = max(1, context_length - len(needle_toks) - len(question_toks))
        insert_pos = int(haystack_budget * depth)

        before = _generate_haystack_tokens(insert_pos, tokenizer, rng)
        after = _generate_haystack_tokens(haystack_budget - insert_pos, tokenizer, rng)

        all_tokens = before + needle_toks + after + question_toks
        scores = np.full(len(all_tokens), self.haystack_score, dtype=np.uint8)
        ns = len(before)
        scores[ns: ns + len(needle_toks)] = self.needle_score
        scores[len(all_tokens) - len(question_toks):] = 60  # question tokens

        return torch.tensor([all_tokens], dtype=torch.long), scores, needle_word

    def run(
        self,
        model,
        tokenizer,
        config: TISConfig,
        cache_budget: float = 1.0,
        generation_kwargs: Optional[dict] = None,
    ) -> dict:
        """Returns ``{accuracy, depth_breakdown, samples}``."""
        policy = _get_policy(model)
        depth_results: dict[float, list[bool]] = {d: [] for d in self.depths}

        sample_idx = 0
        for ctx_len in self.context_lengths:
            for depth in self.depths:
                for _ in range(self.n_samples):
                    ids, scores, needle_word = self._make_sample(
                        tokenizer, ctx_len, depth, seed=sample_idx
                    )
                    sample_idx += 1

                    if cache_budget < 1.0:
                        budget = max(1, int(cache_budget * ids.shape[1]))
                        ids, scores = _select_token_budget(
                            ids, scores, budget, policy, model
                        )

                    try:
                        answer = _run_generation(
                            model,
                            ids,
                            scores,
                            tokenizer,
                            generation_kwargs=generation_kwargs,
                        )
                        ok = needle_word.lower() in answer.lower()
                    except Exception:
                        ok = False
                    depth_results[depth].append(ok)

        depth_breakdown = {
            d: (sum(v) / len(v) if v else 0.0) for d, v in depth_results.items()
        }
        all_results = [v for vs in depth_results.values() for v in vs]
        accuracy = sum(all_results) / len(all_results) if all_results else 0.0

        return {
            "accuracy": accuracy,
            "depth_breakdown": depth_breakdown,
            "samples": len(all_results),
        }


# ─── LostInMiddleBenchmark ────────────────────────────────────────────────────

_POSITIONS = ["beginning", "middle", "end"]


class LostInMiddleBenchmark:
    """Key-value retrieval (Liu et al. 2023 protocol).

    A list of N key–value pairs is shown; the model must retrieve the value
    for a queried key that sits at a controlled position (beginning / middle /
    end) in the list.
    """

    def __init__(
        self,
        n_pairs_options: list[int] | None = None,
        n_samples: int = 20,
        kv_score: int = 80,
    ) -> None:
        self.n_pairs_options = n_pairs_options or [10, 20, 40]
        self.n_samples = n_samples
        self.kv_score = kv_score

    def _make_sample(
        self,
        tokenizer,
        n_pairs: int,
        query_idx: int,
        seed: int,
    ) -> tuple[torch.Tensor, np.ndarray, str]:
        rng = random.Random(seed)
        keys = ["".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(n_pairs)]
        values = ["".join(rng.choices(string.digits, k=4)) for _ in range(n_pairs)]

        kv_lines = "\n".join(f"Key '{k}': {v}" for k, v in zip(keys, values))
        context = f"Key-value pairs:\n{kv_lines}\n"
        question = f"\nWhat is the value for key '{keys[query_idx]}'? Answer:"

        ctx_toks = tokenizer.encode(context, add_special_tokens=False)
        q_toks = tokenizer.encode(question, add_special_tokens=False)
        all_toks = ctx_toks + q_toks

        scores = np.full(len(all_toks), self.kv_score, dtype=np.uint8)
        scores[len(ctx_toks):] = 70  # question tokens slightly lower

        return torch.tensor([all_toks], dtype=torch.long), scores, values[query_idx]

    def _query_idx_for_position(self, n_pairs: int, position: str) -> int:
        if position == "beginning":
            return 0
        if position == "end":
            return n_pairs - 1
        return n_pairs // 2  # middle

    def run(
        self,
        model,
        tokenizer,
        config: TISConfig,
        cache_budget: float = 1.0,
        generation_kwargs: Optional[dict] = None,
    ) -> dict:
        """Returns ``{accuracy, accuracy_by_n_pairs, accuracy_by_position}``."""
        policy = _get_policy(model)
        n_pairs_results: dict[int, list[bool]] = {n: [] for n in self.n_pairs_options}
        pos_results: dict[str, list[bool]] = {p: [] for p in _POSITIONS}

        sample_idx = 0
        for n_pairs in self.n_pairs_options:
            for position in _POSITIONS:
                query_idx = self._query_idx_for_position(n_pairs, position)
                for _ in range(self.n_samples):
                    ids, scores, target = self._make_sample(
                        tokenizer, n_pairs, query_idx, seed=sample_idx
                    )
                    sample_idx += 1

                    if cache_budget < 1.0:
                        budget = max(1, int(cache_budget * ids.shape[1]))
                        ids, scores = _select_token_budget(
                            ids, scores, budget, policy, model
                        )

                    try:
                        answer = _run_generation(
                            model,
                            ids,
                            scores,
                            tokenizer,
                            generation_kwargs=generation_kwargs,
                        )
                        ok = target in answer
                    except Exception:
                        ok = False

                    n_pairs_results[n_pairs].append(ok)
                    pos_results[position].append(ok)

        n_pairs_acc = {
            n: (sum(v) / len(v) if v else 0.0) for n, v in n_pairs_results.items()
        }
        pos_acc = {
            p: (sum(v) / len(v) if v else 0.0) for p, v in pos_results.items()
        }
        all_results = [v for vs in n_pairs_results.values() for v in vs]
        accuracy = sum(all_results) / len(all_results) if all_results else 0.0

        return {
            "accuracy": accuracy,
            "accuracy_by_n_pairs": n_pairs_acc,
            "accuracy_by_position": pos_acc,
        }


# ─── MultiDocQABenchmark ──────────────────────────────────────────────────────

class MultiDocQABenchmark:
    """Multiple-choice QA over long passages, using the QuALITY dataset."""

    def __init__(self, split: str = "validation", n_samples: int = 50) -> None:
        self.split = split
        self.n_samples = n_samples

    def _load_dataset(self):
        from datasets import load_dataset
        return load_dataset("emozilla/quality", split=self.split)

    def _format_sample(self, tokenizer, item: dict):
        article = item.get("article", item.get("passage", ""))
        question = item.get("question", "")
        options = item.get("options", [])

        labels = ["A", "B", "C", "D"]
        opts_text = "\n".join(
            f"{lbl}. {opt}" for lbl, opt in zip(labels, options[:4])
        )

        # Build a query-aware prompt layout that makes the question / options
        # materially more important than generic article text.
        article_toks = tokenizer.encode(article, add_special_tokens=False)
        question_toks = tokenizer.encode(
            f"\n\nQuestion: {question}\n", add_special_tokens=False
        )
        options_toks = tokenizer.encode(
            f"Options:\n{opts_text}\nAnswer (A/B/C/D):",
            add_special_tokens=False,
        )

        toks = article_toks + question_toks + options_toks
        scores = np.full(len(toks), 40, dtype=np.uint8)
        scores[len(article_toks):len(article_toks) + len(question_toks)] = 85
        scores[len(article_toks) + len(question_toks):] = 75

        gold = item.get("gold_label", item.get("answer", 1))
        if isinstance(gold, str):
            correct_label = gold.strip().upper()[:1]
        else:
            correct_label = labels[int(gold) - 1] if 1 <= int(gold) <= 4 else "A"
        return torch.tensor([toks], dtype=torch.long), scores, correct_label

    def run(
        self,
        model,
        tokenizer,
        config: TISConfig,
        cache_budget: float = 1.0,
        generation_kwargs: Optional[dict] = None,
    ) -> dict:
        """Returns ``{accuracy, samples_evaluated}``."""
        policy = _get_policy(model)
        ds = self._load_dataset()

        n = min(self.n_samples, len(ds))
        correct = 0
        evaluated = 0

        for i in range(n):
            try:
                ids, scores, correct_label = self._format_sample(tokenizer, ds[i])
            except Exception:
                continue

            if cache_budget < 1.0:
                budget = max(1, int(cache_budget * ids.shape[1]))
                ids, scores = _select_token_budget(ids, scores, budget, policy, model)

            try:
                answer = _run_generation(
                    model,
                    ids,
                    scores,
                    tokenizer,
                    max_new_tokens=5,
                    generation_kwargs=generation_kwargs,
                )
                predicted = answer.strip().upper()[:1]
                ok = predicted == correct_label
            except Exception:
                ok = False

            correct += ok
            evaluated += 1

        return {
            "accuracy": correct / evaluated if evaluated > 0 else 0.0,
            "samples_evaluated": evaluated,
        }

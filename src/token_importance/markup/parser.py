"""IMLParser — parse Importance Markup Language tags and align to token positions."""
from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np


@dataclass
class ImportanceSpan:
    start: int   # char offset in clean text (inclusive)
    end: int     # char offset in clean text (exclusive)
    score: int   # 0–100


class IMLParser:
    DEFAULT_SCORE: int = 50
    # Match <imp v=N>...</imp> non-greedily; DOTALL so content can span newlines
    TAG_PATTERN = re.compile(r'<imp\s+v=(\d+)>(.*?)</imp>', re.DOTALL)

    # Used to detect nesting before we process the outer tag
    _INNER_TAG_PATTERN = re.compile(r'<imp\b', re.IGNORECASE)

    def parse(self, text: str) -> tuple[str, list[ImportanceSpan]]:
        """Strip <imp v=N>...</imp> tags. Return (clean_text, char-level spans).

        Raises ValueError on nested tags or score outside [0, 100].
        """
        # Check for nesting: find all matches and verify each inner content
        # is free of opening <imp tags.
        for m in self.TAG_PATTERN.finditer(text):
            inner = m.group(2)
            if self._INNER_TAG_PATTERN.search(inner):
                raise ValueError(
                    "Nested <imp> tags are not supported. "
                    f"Found nested tag inside: {m.group(0)!r}"
                )
            score = int(m.group(1))
            if not (0 <= score <= 100):
                raise ValueError(
                    f"Importance score must be in [0, 100], got {score}"
                )

        spans: list[ImportanceSpan] = []
        clean_parts: list[str] = []
        cursor = 0          # position in original text
        clean_cursor = 0    # position in clean text built so far

        for m in self.TAG_PATTERN.finditer(text):
            score = int(m.group(1))
            content = m.group(2)

            # Text before this tag — copy verbatim
            before = text[cursor : m.start()]
            clean_parts.append(before)
            clean_cursor += len(before)

            # The span in clean text
            span_start = clean_cursor
            clean_parts.append(content)
            clean_cursor += len(content)
            spans.append(ImportanceSpan(span_start, clean_cursor, score))

            cursor = m.end()

        # Trailing text after the last tag
        clean_parts.append(text[cursor:])
        clean_text = "".join(clean_parts)
        return clean_text, spans

    def align_to_tokens(
        self,
        spans: list[ImportanceSpan],
        token_offsets: list[tuple[int, int]],
        n_tokens: int,
    ) -> np.ndarray:
        """Map char-level spans to token positions.

        Tokens not covered by any span receive DEFAULT_SCORE.
        For overlapping spans, last span in list wins.

        Returns uint8 array of shape [n_tokens].
        """
        scores = np.full(n_tokens, self.DEFAULT_SCORE, dtype=np.uint8)

        for span in spans:
            for i, (tok_start, tok_end) in enumerate(token_offsets):
                # A token is "inside" a span if it overlaps at all
                if tok_start < span.end and tok_end > span.start:
                    scores[i] = span.score

        return scores

    def encode_with_importance(
        self,
        tokenizer,
        text: str,
        spans: list[tuple[int, int, int]] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Tokenize and return (token_ids, uint8 importance array).

        If spans is None: parse IML tags from text first.
        If spans is provided: text must already be clean; use spans directly.
        Requires a fast tokenizer (offset_mapping). Raises RuntimeError otherwise.
        """
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                "encode_with_importance requires a fast HuggingFace tokenizer "
                "(is_fast=True) for offset_mapping support."
            )

        if spans is None:
            clean_text, imp_spans = self.parse(text)
        else:
            clean_text = text
            imp_spans = [ImportanceSpan(s, e, v) for s, e, v in spans]

        encoding = tokenizer(
            clean_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        token_ids: list[int] = encoding["input_ids"]
        token_offsets: list[tuple[int, int]] = encoding["offset_mapping"]
        n_tokens = len(token_ids)

        importance = self.align_to_tokens(imp_spans, token_offsets, n_tokens)
        return token_ids, importance

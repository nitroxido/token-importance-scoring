"""ScoutAnnotator — LLM-based auto-annotator that wraps text in <imp v=N> tags."""
from __future__ import annotations

from typing import Literal
import numpy as np

from token_importance.markup.parser import IMLParser, ImportanceSpan


SCOUT_SYSTEM_PROMPT = """\
You are a text importance annotator. Wrap spans of the given text in <imp v=N> tags
where N (0–100) indicates how important that span is for correctly answering questions.

Guidelines:
  - Explicit instructions, goals, or questions: v=80-100
  - Key facts, constraints, named entities, numbers: v=60-80
  - Supporting context or examples: v=30-60
  - Filler, pleasantries, transitions: v=0-30

Rules:
  - Every character of the input text must appear in the output exactly once.
  - Do not rephrase or modify any text.
  - Tags must not be nested.
  - Output only the annotated text, nothing else."""


class ScoutAnnotator:
    """Wraps a language model to auto-annotate text with IML importance tags.

    Backends:
      - 'ollama': uses the ``ollama`` Python package (imported lazily).
      - 'hf': uses a HuggingFace ``pipeline`` (imported lazily).

    Both libraries are imported inside the method that calls them so that
    constructing ``ScoutAnnotator`` never fails due to a missing install.
    """

    def __init__(
        self,
        model_name: str,
        backend: Literal["ollama", "hf"] = "ollama",
        parser: IMLParser | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self._parser = parser or IMLParser()

    # ------------------------------------------------------------------
    # Backend call helpers (lazy imports kept here)
    # ------------------------------------------------------------------

    def _call_ollama(self, text: str) -> str:
        import ollama  # lazy import
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return response["message"]["content"]

    def _call_hf(self, text: str) -> str:
        from transformers import pipeline  # lazy import
        pipe = pipeline("text-generation", model=self.model_name)

        # Use chat template if available, otherwise prepend system prompt manually
        try:
            messages = [
                {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            result = pipe(messages, max_new_tokens=len(text) + 512)
            # pipeline with chat format returns list of dicts
            if isinstance(result, list) and result:
                entry = result[0]
                if isinstance(entry, dict):
                    if "generated_text" in entry:
                        generated = entry["generated_text"]
                        if isinstance(generated, list):
                            # Chat format: list of message dicts
                            assistant_msgs = [
                                m["content"] for m in generated
                                if isinstance(m, dict) and m.get("role") == "assistant"
                            ]
                            return assistant_msgs[-1] if assistant_msgs else ""
                        return str(generated)
        except Exception:
            # Fall back to plain text generation
            full_prompt = f"{SCOUT_SYSTEM_PROMPT}\n\n{text}"
            result = pipe(full_prompt, max_new_tokens=len(text) + 512)
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "")
        return ""

    def _call_backend(self, text: str) -> str:
        if self.backend == "ollama":
            return self._call_ollama(text)
        elif self.backend == "hf":
            return self._call_hf(text)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def _validate(self, tagged_text: str, original_text: str) -> tuple[str, list[ImportanceSpan]] | None:
        """Parse tagged_text; return (clean_text, spans) if valid, else None."""
        try:
            clean_text, spans = self._parser.parse(tagged_text)
        except ValueError:
            return None
        # Strip whitespace for comparison (model may add/remove trailing newlines)
        if clean_text.strip() != original_text.strip():
            return None
        return clean_text, spans

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        text: str,
        max_retries: int = 2,
    ) -> tuple[str, list[ImportanceSpan]]:
        """Call model to annotate text, parse the result.

        Retries up to ``max_retries`` times on parse failure or if clean_text
        does not match the original. If all attempts fail, returns
        ``(text, [])`` — original text, no annotations.
        """
        for _ in range(max(1, max_retries)):
            try:
                tagged = self._call_backend(text)
                result = self._validate(tagged, text)
                if result is not None:
                    return result
            except Exception:
                pass

        return text, []

    def annotate_and_encode(
        self,
        tokenizer,
        text: str,
        max_retries: int = 2,
    ) -> tuple[list[int], np.ndarray]:
        """Annotate then encode. Returns (token_ids, uint8 importance array)."""
        clean_text, spans = self.annotate(text, max_retries=max_retries)
        span_tuples = [(s.start, s.end, s.score) for s in spans]
        return self._parser.encode_with_importance(
            tokenizer, clean_text, spans=span_tuples if spans else None
        )

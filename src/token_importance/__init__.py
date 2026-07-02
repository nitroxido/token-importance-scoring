"""Token Importance Score (TIS) — per-token importance for transformer LLMs."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "TISConfig":
        from token_importance.config import TISConfig
        return TISConfig
    if name == "IMLParser":
        from token_importance.markup.parser import IMLParser
        return IMLParser
    if name == "ImportanceStore":
        from token_importance.cache.importance_store import ImportanceStore
        return ImportanceStore
    if name == "EvictionPolicy":
        from token_importance.cache.eviction import EvictionPolicy
        return EvictionPolicy
    if name == "PatchedCausalLM":
        from token_importance.model.patched_model import PatchedCausalLM
        return PatchedCausalLM
    if name == "ScoutAnnotator":
        from token_importance.markup.scout import ScoutAnnotator
        return ScoutAnnotator
    raise AttributeError(f"module 'token_importance' has no attribute {name!r}")


__all__ = [
    "TISConfig",
    "IMLParser",
    "ImportanceStore",
    "EvictionPolicy",
    "PatchedCausalLM",
    "ScoutAnnotator",
]

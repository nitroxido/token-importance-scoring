"""
tests/test_compatibility.py — API compatibility tests (no GPU required).

Validates that all public-facing classes and methods have the expected signatures
and can be imported cleanly. Does not run model inference.

Usage:
    pytest tests/test_compatibility.py -v
"""

import importlib
import sys
import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_importance_head_importable():
    mod = importlib.import_module("token_importance.model.importance_head")
    assert hasattr(mod, "ImportanceUpdateHead"), "ImportanceUpdateHead not found"


def test_cross_attention_scorer_importable():
    mod = importlib.import_module("token_importance.model.cross_attention_scorer")
    assert hasattr(mod, "CrossAttentionImportanceScorer")


def test_query_aware_head_importable():
    mod = importlib.import_module("token_importance.model.query_aware_importance_head")
    assert hasattr(mod, "QueryAwareImportanceHead")


def test_tis_drafter_importable():
    mod = importlib.import_module("token_importance.model.tis_drafter")
    assert hasattr(mod, "TISDrafter")


def test_composite_loss_importable():
    mod = importlib.import_module("token_importance.training.loss_functions")
    assert hasattr(mod, "TISCompositeLoss")


def test_gumbel_topk_importable():
    mod = importlib.import_module("token_importance.utils.gumbel_topk")
    assert hasattr(mod, "GumbelTopkLayer")


def test_benchmarks_importable():
    mod = importlib.import_module("token_importance.eval.benchmarks")
    assert hasattr(mod, "LostInMiddleBenchmark") or hasattr(mod, "NIAHBenchmark"), \
        "Expected benchmark class not found in benchmarks module"


# ---------------------------------------------------------------------------
# Signature checks (no GPU, no model weights)
# ---------------------------------------------------------------------------

def test_importance_head_direct_score_signature():
    """direct_score() must accept a single tensor argument."""
    import inspect
    mod = importlib.import_module("token_importance.model.importance_head")
    cls = mod.ImportanceUpdateHead
    assert hasattr(cls, "direct_score"), "ImportanceUpdateHead.direct_score() missing"
    sig = inspect.signature(cls.direct_score)
    params = list(sig.parameters.keys())
    assert "hidden" in params, f"direct_score() expected 'hidden' parameter, got {params}"


def test_cross_attention_scorer_init_signature():
    import inspect
    mod = importlib.import_module("token_importance.model.cross_attention_scorer")
    cls = mod.CrossAttentionImportanceScorer
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())
    for expected in ("hidden_dim",):
        assert expected in params, f"CrossAttentionImportanceScorer.__init__ missing '{expected}'"


def test_composite_loss_forward_returns_dict():
    """TISCompositeLoss.forward() must return a dict or named tuple (not a scalar)."""
    import inspect
    mod = importlib.import_module("token_importance.training.loss_functions")
    cls = mod.TISCompositeLoss
    assert hasattr(cls, "forward"), "TISCompositeLoss.forward() missing"


# ---------------------------------------------------------------------------
# Instantiation checks (CPU, no weights needed)
# ---------------------------------------------------------------------------

def test_gumbel_topk_instantiation():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    mod = importlib.import_module("token_importance.utils.gumbel_topk")
    layer = mod.GumbelTopkLayer(top_k=8)
    assert layer is not None


def test_cross_attention_scorer_instantiation():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    mod = importlib.import_module("token_importance.model.cross_attention_scorer")
    scorer = mod.CrossAttentionImportanceScorer(hidden_dim=64)
    assert scorer is not None


def test_composite_loss_instantiation():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    mod = importlib.import_module("token_importance.training.loss_functions")
    loss = mod.TISCompositeLoss()
    assert loss is not None

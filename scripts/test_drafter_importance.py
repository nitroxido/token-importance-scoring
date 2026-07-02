"""
Task 3: Mock Testing Harness for Drafter Importance Integration
Tests TISAwareDrafter wrapper and importance bias without real training.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from token_importance.model.drafter_wrapper import TISAwareDrafter, DrafterSpeculationConfig
from token_importance.model.drafter_attn_bias import DrafterImportanceAttnBias


def create_mock_drafter(d_model: int = 768, vocab_size: int = 32000) -> nn.Module:
    """Create a mock EAGLE-3-like drafter for testing."""
    
    class MockDrafter(nn.Module):
        def __init__(self, d_model, vocab_size):
            super().__init__()
            self.d_model = d_model
            self.linear = nn.Linear(d_model, vocab_size)
        
        def forward(self, hidden_states):
            # hidden_states: [B, d_model]
            return self.linear(hidden_states)
    
    return MockDrafter(d_model, vocab_size)


def test_drafter_initialization():
    """Test that TISAwareDrafter initializes correctly."""
    print("Testing drafter initialization...")
    
    mock_drafter = create_mock_drafter()
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    assert tis_drafter.eagle3 is not None
    assert tis_drafter.importance_scores is None
    print("✓ Drafter initialization passed")


def test_importance_score_setting():
    """Test setting importance scores."""
    print("Testing importance score setting...")
    
    mock_drafter = create_mock_drafter()
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    seq_len = 512
    importance_scores = torch.linspace(0, 100, seq_len)
    
    tis_drafter.set_context_importance(importance_scores)
    
    assert tis_drafter.importance_scores is not None
    assert tis_drafter.importance_scores.shape == (seq_len,)
    assert tis_drafter.importance_scores[0].item() == torch.tensor(0.0).item()
    assert tis_drafter.importance_scores[-1].item() == torch.tensor(100.0).item()
    
    print("✓ Importance score setting passed")


def test_forward_pass():
    """Test drafter forward pass with mock input."""
    print("Testing drafter forward pass...")
    
    d_model = 768
    vocab_size = 32000
    batch_size = 2
    
    mock_drafter = create_mock_drafter(d_model=d_model, vocab_size=vocab_size)
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    # Create mock hidden states
    hidden_states = torch.randn(batch_size, d_model)
    
    # Forward pass without importance context
    logits = tis_drafter(hidden_states, drafter_step=0)
    
    assert logits.shape == (batch_size, vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    
    print("✓ Forward pass passed")


def test_forward_pass_with_importance():
    """Test drafter forward pass with importance scores set."""
    print("Testing forward pass with importance context...")
    
    d_model = 768
    vocab_size = 32000
    batch_size = 2
    seq_len = 512
    
    mock_drafter = create_mock_drafter(d_model=d_model, vocab_size=vocab_size)
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    # Set importance context
    importance_scores = torch.linspace(0, 100, seq_len)
    tis_drafter.set_context_importance(importance_scores)
    
    # Create mock hidden states
    hidden_states = torch.randn(batch_size, d_model)
    
    # Forward pass with importance context
    logits = tis_drafter(hidden_states, drafter_step=0)
    
    assert logits.shape == (batch_size, vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    
    print("✓ Forward pass with importance passed")


def test_importance_attn_bias():
    """Test importance attention bias computation."""
    print("Testing importance attention bias...")
    
    d_model = 768
    bias_module = DrafterImportanceAttnBias(
        d_model=d_model,
        lambda_d_base=0.1,
        lambda_d_slope=0.05,
    )
    
    batch_size = 2
    n_heads = 12
    seq_len = 512
    
    # Create mock attention scores (pre-softmax)
    attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)
    
    # Create importance scores
    importance_scores = torch.linspace(0, 100, seq_len)
    
    # Apply bias
    biased_scores = bias_module(
        attention_scores,
        importance_scores,
        drafter_step=0,
    )
    
    assert biased_scores.shape == attention_scores.shape
    assert not torch.isnan(biased_scores).any()
    assert not torch.isinf(biased_scores).any()
    
    # Check that bias is applied (difference should be non-zero)
    bias_applied = (biased_scores != attention_scores).any()
    assert bias_applied, "Bias should modify attention scores"
    
    print("✓ Importance attention bias passed")


def test_depth_scaled_bias():
    """Test depth-scaled bias computation."""
    print("Testing depth-scaled bias...")
    
    bias_module = DrafterImportanceAttnBias(
        lambda_d_base=0.1,
        lambda_d_slope=0.05,
        lambda_d_max=1.0,
    )
    
    # Check bias strength increases with depth
    strength_0 = bias_module.get_bias_strength(drafter_step=0)
    strength_5 = bias_module.get_bias_strength(drafter_step=5)
    strength_10 = bias_module.get_bias_strength(drafter_step=10)
    
    assert strength_0 < strength_5 < strength_10, \
        f"Bias should increase with depth: {strength_0} < {strength_5} < {strength_10}"
    
    # Check clamping at max
    strength_100 = bias_module.get_bias_strength(drafter_step=100)
    assert strength_100 <= bias_module.lambda_d_max, \
        f"Bias should be clamped at max: {strength_100} <= {bias_module.lambda_d_max}"
    
    print("✓ Depth-scaled bias passed")


def test_gradient_flow():
    """Test that gradients flow through TISAwareDrafter."""
    print("Testing gradient flow...")
    
    d_model = 768
    batch_size = 1
    
    mock_drafter = create_mock_drafter(d_model=d_model)
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    # Create input with gradients enabled
    hidden_states = torch.randn(batch_size, d_model, requires_grad=True)
    
    # Forward pass
    logits = tis_drafter(hidden_states, drafter_step=0)
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients are computed
    assert hidden_states.grad is not None, "Gradients should flow through drafter"
    assert not torch.isnan(hidden_states.grad).any(), "Gradients should not be NaN"
    
    print("✓ Gradient flow passed")


def test_batch_processing():
    """Test batch processing with different batch sizes."""
    print("Testing batch processing...")
    
    d_model = 768
    vocab_size = 32000
    seq_len = 512
    
    mock_drafter = create_mock_drafter(d_model=d_model, vocab_size=vocab_size)
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    # Set importance scores
    importance_scores = torch.ones(seq_len) * 50
    tis_drafter.set_context_importance(importance_scores)
    
    # Test different batch sizes
    for batch_size in [1, 4, 16]:
        hidden_states = torch.randn(batch_size, d_model)
        logits = tis_drafter(hidden_states)
        
        assert logits.shape == (batch_size, vocab_size)
        print(f"  ✓ Batch size {batch_size} passed")


def test_speculation_config():
    """Test DrafterSpeculationConfig."""
    print("Testing speculation config...")
    
    config = DrafterSpeculationConfig(
        max_speculation_depth=8,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
    )
    
    assert config.max_speculation_depth == 8
    assert config.temperature == 0.8
    assert config.top_k == 40
    assert config.top_p == 0.95
    assert config.use_importance_bias is True
    assert config.use_cache is True
    
    print("✓ Speculation config passed")


def test_multiple_speculation_steps():
    """Test forward passes at multiple speculation depths."""
    print("Testing multiple speculation steps...")
    
    d_model = 768
    batch_size = 2
    max_depth = 8
    
    mock_drafter = create_mock_drafter(d_model=d_model)
    tis_drafter = TISAwareDrafter(mock_drafter, device="cpu")
    
    # Set importance context
    importance_scores = torch.linspace(10, 90, 512)
    tis_drafter.set_context_importance(importance_scores)
    
    # Run forward passes at different depths
    for step in range(max_depth):
        hidden_states = torch.randn(batch_size, d_model)
        logits = tis_drafter(hidden_states, drafter_step=step)
        
        assert logits.shape[0] == batch_size
        print(f"  ✓ Speculation step {step} passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DRAFTER IMPORTANCE INTEGRATION - MOCK TESTS")
    print("=" * 60)
    
    tests = [
        test_drafter_initialization,
        test_importance_score_setting,
        test_forward_pass,
        test_forward_pass_with_importance,
        test_importance_attn_bias,
        test_depth_scaled_bias,
        test_gradient_flow,
        test_batch_processing,
        test_speculation_config,
        test_multiple_speculation_steps,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

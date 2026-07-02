#!/usr/bin/env python3
"""
Deep Diagnostics: Verify importance head training and application

Tests:
1. Are importance head weights changing during training?
2. Do importance scores actually affect cache decisions in eval?
3. Is the end-to-end pipeline working?
4. Can we get immediate improvements with supervised training?
"""

import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import copy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
from token_importance.model.importance_scoring_head import create_importance_head_with_lora


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint."""
    model_name = "mistralai/Mistral-7B-v0.3"
    checkpoint_dir = Path(checkpoint_path)
    
    if not (checkpoint_dir / "importance_head").exists():
        print(f"ERROR: No importance_head directory in {checkpoint_path}")
        return None, None, None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    # Create and load importance head
    config = base_model.config
    importance_head = create_importance_head_with_lora(
        d_model=config.hidden_size,
        lora_rank=8,
        lora_alpha=16,
    )
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        importance_head = PeftModel.from_pretrained(
            importance_head,
            str(checkpoint_dir / "importance_head"),
        ).to(base_model.device)
    
    return base_model, importance_head, tokenizer


def get_weight_signature(importance_head) -> str:
    """Get a hash of the importance head weights."""
    weights = []
    for param in importance_head.parameters():
        weights.append(param.data.float().sum().item())
    return f"{sum(weights):.6f}"


def test_weight_changes():
    """Test 1: Check if weights are actually changing."""
    print("\n" + "="*80)
    print("TEST 1: Are Importance Head Weights Changing?")
    print("="*80)
    
    ckpt1 = "checkpoints/ert_local_full_10k"
    ckpt2 = "checkpoints/phase4_msmarco_500steps/final"
    
    print(f"\nLoading ERT checkpoint...")
    base1, head1, tok1 = load_checkpoint(ckpt1)
    if head1 is None:
        print("Failed to load ERT")
        return False
    
    sig1 = get_weight_signature(head1)
    print(f"ERT weight signature: {sig1}")
    
    # Clean up memory
    del base1
    torch.cuda.empty_cache()
    
    print(f"\nLoading Phase 4 checkpoint (500 steps MS MARCO)...")
    base2, head2, tok2 = load_checkpoint(ckpt2)
    if head2 is None:
        print("Failed to load Phase 4")
        return False
    
    sig2 = get_weight_signature(head2)
    print(f"Phase 4 weight signature: {sig2}")
    
    if sig1 == sig2:
        print("\n❌ CRITICAL: Weights are IDENTICAL!")
        print("   Importance head did NOT change during Phase 4 training!")
        print("   This means: Training doesn't update the weights")
        del base2
        torch.cuda.empty_cache()
        return False
    else:
        print(f"\n✓ Weights changed: {sig1} → {sig2}")
        print(f"  Difference: {abs(float(sig2) - float(sig1)):.6f}")
        del base2
        torch.cuda.empty_cache()
        return True


def test_score_changes():
    """Test 2: Check if importance scores are actually different."""
    print("\n" + "="*80)
    print("TEST 2: Are Importance Scores Different Across Checkpoints?")
    print("="*80)
    
    ckpt1 = "checkpoints/ert_local_full_10k"
    ckpt2 = "checkpoints/phase4_msmarco_500steps/final"
    
    text = "France is a country. Paris is the capital. It has culture and art."
    
    print(f"\nTest text: {text[:50]}...")
    
    # Get scores from ERT
    print(f"\nLoading ERT...")
    base1, head1, tok1 = load_checkpoint(ckpt1)
    inputs1 = tok1(text, max_length=256, truncation=True, return_tensors="pt")
    input_ids1 = inputs1["input_ids"].to(base1.device)
    
    with torch.no_grad():
        out1 = base1(input_ids1, output_hidden_states=True)
        hidden1 = out1.hidden_states[-1].to(torch.float32)
        scores1 = torch.sigmoid(head1(hidden1).squeeze(-1).squeeze(0))
    
    tokens = tok1.convert_ids_to_tokens(input_ids1[0])
    scores1_list = scores1.cpu().tolist()
    
    print(f"ERT scores: min={min(scores1_list):.4f}, max={max(scores1_list):.4f}, "
          f"mean={sum(scores1_list)/len(scores1_list):.4f}")
    
    del base1, head1
    torch.cuda.empty_cache()
    
    # Get scores from Phase 4
    print(f"\nLoading Phase 4...")
    base2, head2, tok2 = load_checkpoint(ckpt2)
    inputs2 = tok2(text, max_length=256, truncation=True, return_tensors="pt")
    input_ids2 = inputs2["input_ids"].to(base2.device)
    
    with torch.no_grad():
        out2 = base2(input_ids2, output_hidden_states=True)
        hidden2 = out2.hidden_states[-1].to(torch.float32)
        scores2 = torch.sigmoid(head2(hidden2).squeeze(-1).squeeze(0))
    
    scores2_list = scores2.cpu().tolist()
    
    print(f"Phase 4 scores: min={min(scores2_list):.4f}, max={max(scores2_list):.4f}, "
          f"mean={sum(scores2_list)/len(scores2_list):.4f}")
    
    # Compare
    score_diff = sum(abs(a - b) for a, b in zip(scores1_list, scores2_list)) / len(scores1_list)
    print(f"\nAverage score difference: {score_diff:.4f}")
    
    if score_diff < 0.01:
        print("⚠️  Scores are VERY similar (diff < 0.01)")
        print("   Indicates minimal training effect")
    elif score_diff < 0.05:
        print("⚠️  Scores are similar (diff < 0.05)")
    else:
        print(f"✓ Scores differ meaningfully (diff = {score_diff:.4f})")
    
    del base2, head2
    torch.cuda.empty_cache()
    
    return score_diff > 0.05


def test_eval_actually_uses_scores():
    """Test 3: Check if eval.py actually uses importance scores in cache decisions."""
    print("\n" + "="*80)
    print("TEST 3: Does eval.py Actually Use Importance Scores?")
    print("="*80)
    
    print("\nChecking eval.py code...")
    eval_path = Path(__file__).parent.parent / "scripts" / "eval.py"
    
    if not eval_path.exists():
        print(f"ERROR: Can't find {eval_path}")
        return False
    
    code = eval_path.read_text()
    
    checks = [
        ("importance_head" in code, "Importance head loaded"),
        ("importance_scores" in code or "scores =" in code, "Importance scores computed"),
        ("argsort" in code or "topk" in code, "Cache decisions made with scores"),
        ("cache" in code.lower(), "Cache modified"),
    ]
    
    all_good = True
    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {description}")
        if not check:
            all_good = False
    
    if all_good:
        print("\n✓ eval.py appears to use importance scores correctly")
    else:
        print("\n❌ eval.py might NOT be using importance scores!")
        print("   Need to manually review eval.py")
    
    return all_good


def test_simple_supervised():
    """Test 4: Can we train with simple supervised loss?"""
    print("\n" + "="*80)
    print("TEST 4: Simple Supervised Training Test")
    print("="*80)
    
    print("\nThis test creates a trivial task:")
    print("  - Tokens 0-5: Label = 1 (important)")
    print("  - Tokens 6-10: Label = 0 (unimportant)")
    print("  - Train for 5 steps")
    print("  - Check if MSE loss decreases")
    
    ckpt = "checkpoints/ert_local_full_10k"
    base, head, tok = load_checkpoint(ckpt)
    if head is None:
        print("Failed to load checkpoint")
        return False
    
    # Create dummy hidden states and labels
    batch_size = 2
    seq_len = 12
    hidden_dim = 4096
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=base.device, dtype=torch.float32)
    labels = torch.zeros(batch_size, seq_len, device=base.device, dtype=torch.float32)
    labels[:, :6] = 1.0  # First 6 tokens are important
    
    # Enable gradients on importance head
    for param in head.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    
    print("\nTraining for 5 steps...")
    losses = []
    for step in range(5):
        # Forward
        scores = torch.sigmoid(head(hidden_states).squeeze(-1))  # [B, T]
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(scores, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step+1}: loss = {loss.item():.6f}")
    
    del base, head
    torch.cuda.empty_cache()
    
    # Check if loss decreased
    loss_decreased = losses[-1] < losses[0]
    if loss_decreased:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n✓ Loss DECREASED by {improvement:.1f}%")
        print(f"  {losses[0]:.6f} → {losses[-1]:.6f}")
        print("  Importance head CAN learn!")
        return True
    else:
        print(f"\n❌ Loss did NOT decrease!")
        print(f"  {losses[0]:.6f} → {losses[-1]:.6f}")
        print("  Importance head training might be broken")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deep diagnostics")
    args = parser.parse_args()
    
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "DEEP DIAGNOSTICS: Is The System Broken?" + " "*20 + "║")
    print("╚" + "═"*78 + "╝")
    
    results = {}
    
    # Test 1: Weight changes
    results["weights_change"] = test_weight_changes()
    
    # Test 2: Score changes
    results["scores_differ"] = test_score_changes()
    
    # Test 3: Eval pipeline
    results["eval_uses_scores"] = test_eval_actually_uses_scores()
    
    # Test 4: Supervised learning
    results["supervised_works"] = test_simple_supervised()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✓" if result else "❌"
        print(f"{status} {test_name}: {'PASS' if result else 'FAIL'}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n✓ All tests passed! System should be working.")
        print("  Problem might be: wrong objective (contrastive) not execution")
        print("  Solution: Use supervised loss instead of contrastive")
    else:
        print("\n❌ Some tests failed. There's an implementation issue.")
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed tests: {', '.join(failed)}")
        print("  Need to debug: " + ", ".join(failed))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

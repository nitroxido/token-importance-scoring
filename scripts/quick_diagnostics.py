#!/usr/bin/env python3
"""
Quick Diagnostics: Check if importance head weights actually changed
"""

import torch
import argparse
from pathlib import Path
from typing import Dict
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compare_checkpoints():
    """Compare ERT vs Phase 4 checkpoint files directly."""
    print("\n" + "="*80)
    print("TEST: Did Importance Head Weights Change?")
    print("="*80)
    
    # Load adapter weights directly
    ert_adapter = Path("checkpoints/ert_local_full_10k/importance_head/adapter_model.safetensors")
    phase4_adapter = Path("checkpoints/phase4_msmarco_500steps/final/importance_head/adapter_model.safetensors")
    
    print(f"\nERT adapter exists: {ert_adapter.exists()}")
    print(f"Phase 4 adapter exists: {phase4_adapter.exists()}")
    
    if not ert_adapter.exists() or not phase4_adapter.exists():
        print("ERROR: Cannot find adapter files")
        return False
    
    # Compare file sizes (rough indicator of changes)
    ert_size = ert_adapter.stat().st_size
    phase4_size = phase4_adapter.stat().st_size
    
    print(f"\nERT adapter size:     {ert_size:,} bytes")
    print(f"Phase 4 adapter size: {phase4_size:,} bytes")
    
    if ert_size == phase4_size:
        print("\n⚠️  Sizes are identical - suspicious!")
    else:
        print(f"\n✓ Sizes differ by {abs(phase4_size - ert_size):,} bytes")
    
    # Try to load and compare actual weights
    try:
        from safetensors.torch import load_file
        
        print("\nLoading weight tensors...")
        ert_weights = load_file(str(ert_adapter))
        phase4_weights = load_file(str(phase4_adapter))
        
        print(f"ERT adapter keys:     {list(ert_weights.keys())}")
        print(f"Phase 4 adapter keys: {list(phase4_weights.keys())}")
        
        # Compare each weight
        all_same = True
        for key in ert_weights.keys():
            if key in phase4_weights:
                ert_w = ert_weights[key]
                phase4_w = phase4_weights[key]
                
                diff = (ert_w - phase4_w).abs().max().item()
                
                if diff < 1e-6:
                    print(f"  {key}: IDENTICAL (diff={diff:.2e})")
                    all_same = False
                else:
                    print(f"  {key}: CHANGED (diff={diff:.2e})")
            else:
                print(f"  {key}: Missing in Phase 4!")
                all_same = False
        
        if all_same:
            print("\n❌ CRITICAL: All weights are IDENTICAL!")
            print("   Phase 4 training did NOT update the weights!")
            return False
        else:
            print("\n✓ Weights changed in Phase 4 training")
            return True
            
    except Exception as e:
        print(f"\nCouldn't load with safetensors: {e}")
        print("Weights may still have changed, cannot verify")
        return None


def test_supervised_learning():
    """Test 2: Can importance head learn with supervised loss?"""
    print("\n" + "="*80)
    print("TEST: Can Importance Head Learn with Supervised MSE Loss?")
    print("="*80)
    
    from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer
    from peft import PeftModel
    from token_importance.model.importance_scoring_head import create_importance_head_with_lora
    
    model_name = "mistralai/Mistral-7B-v0.3"
    ckpt_path = "checkpoints/ert_local_full_10k"
    
    print("\nLoading model...")
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
    
    importance_head = PeftModel.from_pretrained(
        importance_head,
        str(Path(ckpt_path) / "importance_head"),
    ).to(base_model.device)
    
    print("✓ Model loaded")
    
    # Create simple training task
    print("\nTraining on simple supervised task:")
    print("  - Tokens 0-5: Label=1 (important)")
    print("  - Tokens 6-12: Label=0 (unimportant)")
    print("  - 10 training steps with MSE loss")
    
    batch_size = 1
    seq_len = 12
    hidden_dim = 4096
    
    # Create dummy data
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_dim,
        device=base_model.device,
        dtype=torch.float32
    )
    
    labels = torch.zeros(batch_size, seq_len, device=base_model.device)
    labels[:, :6] = 1.0  # First 6 tokens are important
    
    # Enable training
    importance_head.train()
    for param in importance_head.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(importance_head.parameters(), lr=1e-4)
    
    losses = []
    print("\nTraining...")
    for step in range(10):
        # Forward
        with torch.enable_grad():
            scores = torch.sigmoid(importance_head(hidden_states).squeeze(-1))
            loss = torch.nn.functional.mse_loss(scores, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        if (step + 1) % 2 == 0:
            print(f"  Step {step+1}: loss = {loss.item():.6f}")
    
    # Check if loss decreased
    improvement = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
    
    print(f"\nInitial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Improvement:  {improvement:.1f}%")
    
    if losses[-1] < losses[0]:
        print("\n✓ Loss DECREASED - importance head can learn!")
        return True
    else:
        print("\n❌ Loss did NOT decrease - training broken!")
        return False


def main():
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*15 + "DIAGNOSTIC: Testing Weight Updates and Learning" + " "*17 + "║")
    print("╚" + "═"*78 + "╝")
    
    # Test 1: Weight changes
    weight_test = compare_checkpoints()
    
    print("\n" + "-"*80 + "\n")
    
    # Test 2: Supervised learning
    learning_test = test_supervised_learning()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if weight_test is True:
        print("✓ Weights CHANGED during Phase 4 training")
    elif weight_test is False:
        print("❌ Weights are IDENTICAL - training didn't update weights!")
    else:
        print("⚠️  Couldn't verify weight changes")
    
    if learning_test:
        print("✓ Supervised learning WORKS - importance head CAN learn")
    else:
        print("❌ Supervised learning FAILED - importance head can't learn")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if weight_test is False:
        print("\n🚨 CRITICAL ISSUE: Weights didn't change!")
        print("   Problem: LoRA adapter or training mechanism is broken")
        print("   Action: Need to debug training loop")
    elif weight_test is True and learning_test:
        print("\n✓ System appears FUNCTIONAL")
        print("   Problem is NOT broken implementation")
        print("   Problem IS: Wrong training objective (contrastive vs supervised)")
        print("   Solution: Use supervised MSE loss instead of contrastive")
    else:
        print("\n⚠️  Mixed results - system may have multiple issues")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

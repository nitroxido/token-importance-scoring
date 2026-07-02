#!/usr/bin/env python
"""Quick smoke test for Phase B training script initialization."""

import sys
from pathlib import Path

# Test imports
print("[smoke] Testing training infrastructure imports...")

try:
    from token_importance.training.loss_functions import TISCompositeLoss
    print("[smoke] ✓ Loss functions imported")
except ImportError as e:
    print(f"[smoke] ✗ Loss functions import failed: {e}")
    sys.exit(1)

try:
    from token_importance.training.litm_dataloader import get_litm_dataloader, create_litm_training_set
    print("[smoke] ✓ LITM dataloader imported")
except ImportError as e:
    print(f"[smoke] ✗ LITM dataloader import failed: {e}")
    sys.exit(1)

# Test creating training data
print("[smoke] Testing training data generation...")
try:
    examples = create_litm_training_set(n_examples=5)
    print(f"[smoke] ✓ Generated {len(examples)} training examples")
except Exception as e:
    print(f"[smoke] ✗ Training data generation failed: {e}")
    sys.exit(1)

# Test loss functions
print("[smoke] Testing loss function initialization...")
try:
    loss_fn = TISCompositeLoss(
        lambda_kl=0.1,
        lambda_budget=0.01,
        lambda_churn=0.01,
        lambda_saliency=0.0,
    )
    print("[smoke] ✓ Loss function initialized")
except Exception as e:
    print(f"[smoke] ✗ Loss function initialization failed: {e}")
    sys.exit(1)

# Test trainer imports
print("[smoke] Testing trainer script imports...")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[smoke] ✓ Core dependencies imported")
except ImportError as e:
    print(f"[smoke] ✗ Core dependency import failed: {e}")
    sys.exit(1)

print("\n[smoke] ✓ All initialization tests passed! Ready for Phase B.1 training.")
print("[smoke] Next: python scripts/train_phase_b.py --phase B.1 --load_in_4bit --checkpoint checkpoints/stage4_query_aware_v1/")

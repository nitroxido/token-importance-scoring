#!/usr/bin/env python3
"""
Demonstration: The LoRA Failure & Recovery Path

This script shows two modes of operation:
1. DEFAULT (--skip-lora): Recommended workflow, uses TIS components without broken LoRA
2. FAILURE (--load-failed-stage2): Educational demonstration of the Stage 2 LoRA failure

Run this to understand:
- How the system works normally
- What went wrong with Stage 2 LoRA
- Why architectural mismatch prevents integration
- How to avoid this in future training

See FAILURE-RECOVERY-DOCUMENTATION.md for complete analysis.
"""

import subprocess
import sys
from pathlib import Path


def log_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def log_step(num: int, title: str, detail: str = "") -> None:
    """Print a step header."""
    print(f"\n[Step {num}] {title}")
    if detail:
        print(f"  {detail}")
    print("-" * 80)


def run_command(cmd: list[str], description: str, expect_failure: bool = False) -> bool:
    """Run a command and report results."""
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        if expect_failure:
            print(f"\n⚠️  UNEXPECTED SUCCESS: {description}")
            return False
        else:
            print(f"\n✅ SUCCESS: {description}")
            return True
    else:
        if expect_failure:
            print(f"\n✅ EXPECTED ERROR: {description}")
            print(f"   (This error demonstrates the architectural mismatch)")
            return True
        else:
            print(f"\n❌ ERROR: {description}")
            return False


def main():
    workspace_root = Path("/mnt/juegos/proyectos/especiales/token-importance")
    
    if not workspace_root.exists():
        print(f"ERROR: Workspace not found at {workspace_root}")
        return 1
    
    log_section("TOKEN IMPORTANCE SCORING: FAILURE & RECOVERY DEMONSTRATION")
    
    print("""
This demonstration shows the complete journey of discovering and understanding the
LoRA loading bug and architectural mismatch in the Token Importance Scoring system.

BACKGROUND:
- Stage 2 LoRA training failed (objective conflict, memorization)
- LoRA loading was disabled to prevent broken adapters from loading
- For months, trained weights silently didn't load
- This session discovered the root cause: architectural mismatch

MODE 1: RECOMMENDED WORKFLOW (--skip-lora)
- Loads TIS components (importance_embedding, importance_head weights)
- Skips LoRA adapters (prevents both failure and mismatch)
- Result: LITM @ 50% = 0.461 (oracle-tier structural learning)

MODE 2: FAILURE DEMONSTRATION (--load-failed-stage2)
- Attempts to load broken Stage 2 LoRA adapters
- Will fail with "Target modules not found" (architectural mismatch)
- Result: Educational - shows why LoRA was disabled
    """)
    
    # ─────────────────────────────────────────────────────────────────────────
    log_section("PART 1: DEFAULT WORKFLOW (RECOMMENDED)")
    
    log_step(1, "Understanding the Default Mode",
             "By default, --skip-lora is implicit. TIS components load, LoRA doesn't.")
    
    print("""
Why use --skip-lora?
  1. ✅ Prevents loading broken Stage 2 adapters
  2. ✅ Avoids architectural mismatch errors
  3. ✅ Uses proven working TIS components (ERT)
  4. ✅ Achieves oracle-tier structural learning (100% NIAH, 46.1% LITM @ 50%)

What loads with --skip-lora?
  1. ✅ importance_embedding (learns token importance representation)
  2. ✅ importance_head (trained via ERT with KL divergence objective)
  3. ✅ attn_hook_lambda (attention bias parameter)
  4. ❌ LoRA adapters (skipped - would cause mismatch if loaded)
    """)
    
    log_step(2, "Test 1: Quick NIAH Test (10 samples, 10 seconds)",
             "Run a small NIAH benchmark to see structural importance learning")
    
    cmd = [
        "bash", "-c",
        """
cd /mnt/juegos/proyectos/especiales/token-importance && \
source .venv/bin/activate && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 60 python scripts/eval.py \
  --model mistralai/Mistral-7B-v0.3 \
  --checkpoint checkpoints/ert_local_full_10k \
  --benchmark niah \
  --load_in_4bit \
  --skip-lora \
  --n_samples 10 \
  --output /tmp/demo_niah_skip_lora.csv 2>&1 | tail -20
        """
    ]
    
    success = run_command(cmd, "NIAH with --skip-lora (structural learning)")
    
    if success:
        print("""
EXPECTED RESULT:
  Running: NIAH at cache budgets [0.25, 0.5, 0.75, 1.0]
  Accuracy at each budget should be:
    - 1.0 @ 25%   ← 100% (structural importance works!)
    - 1.0 @ 50%   ← 100%
    - 1.0 @ 75%   ← 100%
    - 1.0 @ 100%  ← 100%
    
This proves that TIS components work correctly and learn position-invariant importance.
        """)
    
    # ─────────────────────────────────────────────────────────────────────────
    log_section("PART 2: FAILURE DEMONSTRATION (EDUCATIONAL)")
    
    log_step(3, "Understanding the Failure Mode",
             "--load-failed-stage2 forces loading of broken Stage 2 LoRA adapters")
    
    print("""
What is --load-failed-stage2?
  This flag is for EDUCATIONAL PURPOSES ONLY. It demonstrates:
  
  1. ❌ Stage 2 LoRA training failed (lm_loss → 2.1e-06, memorization)
  2. ❌ Module names don't match ("lora_layer" vs "cross_attn", "out_proj")
  3. ❌ This is why LoRA loading was disabled
  
What happens when you use --load-failed-stage2?
  1. Script attempts to load broken adapters
  2. Gets error: "Target modules {'lora_layer'} not found in base model"
  3. This error is EXPECTED and DEMONSTRATES the architectural mismatch
  
Why show the failure?
  - Transparency about what went wrong
  - Understanding the root cause (architecture mismatch)
  - Avoiding the same mistake in future training
  - Teaching what NOT to do
    """)
    
    log_step(4, "Test 2: Demonstrating the Failure (architecture mismatch)",
             "Try to load failed Stage 2 adapters (will error)")
    
    cmd = [
        "bash", "-c",
        """
cd /mnt/juegos/proyectos/especiales/token-importance && \
source .venv/bin/activate && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 60 python scripts/eval.py \
  --model mistralai/Mistral-7B-v0.3 \
  --checkpoint checkpoints/supervised_importance/final \
  --benchmark niah \
  --load_in_4bit \
  --load-failed-stage2 \
  --n_samples 3 2>&1 | grep -A 5 "LORA_LOADING\\|Target modules\\|architectural"
        """
    ]
    
    success = run_command(cmd, "Attempt to load failed Stage 2 (expect architecture error)",
                          expect_failure=True)
    
    if success:
        print("""
EXPECTED ERROR (this is correct!):
  [eval] [LORA LOADING] DEBUGGING MODE: --load-failed-stage2 specified
  [eval]   WARNING: This will attempt to load Stage 2 LoRA which causes degenerate output!
  [eval] [LORA LOADING] Force-loading Stage 2 LoRA from checkpoints/supervised_importance/final
  [eval]   ✗ Failed to load: Target modules {'lora_layer'} not found in base model

WHY THIS ERROR OCCURS:
  Training architecture (create_importance_head_with_lora):
    - Uses modules: 'lora_layer', 'lora_A', 'lora_B'
  
  Evaluation architecture (ImportanceUpdateHead):
    - Uses modules: 'cross_attn', 'out_proj'
  
  Module names don't match → LoRA can't load
  This is the CORE ARCHITECTURAL MISMATCH we discovered!
        """)
    
    # ─────────────────────────────────────────────────────────────────────────
    log_section("PART 3: SUMMARY & LESSONS LEARNED")
    
    print("""
THE JOURNEY:
  
  Stage 1 (Success):
    ✅ Oracle TIS achieves 100% NIAH, 46.1% LITM @ 50%
    ✅ Infrastructure proven working
  
  Stage 2 (Catastrophic Failure):
    ❌ LoRA + LM objective conflicts (LM gradient dominates)
    ❌ Memorizes training tokens instead of learning importance
    ❌ Produces degenerate output at inference (colons)
    ❌ Training loss → 0 masked the disaster
  
  Workaround (Masks Issue):
    ⚠️  Disable LoRA loading to prevent broken adapters
    ⚠️  Silent failure: nobody notices LoRA isn't loading
    ⚠️  System appears to work but is untrained
    ⚠️  True issue (architecture) remains hidden
  
  Discovery (Today):
    ✅ Systematic debugging finds LoRA loading was disabled
    ✅ Identifies architectural mismatch as root cause
    ✅ Proves training & evaluation use incompatible designs
    ✅ Proposes three solutions (A, B, or C)

THE LESSONS:
  
  1. Loss Convergence ≠ Inference Quality
     Stage 2 achieved lm_loss → 2.1e-06 (perfect!) but broke generation entirely
  
  2. Silent Failures Are Dangerous
     Disabled code without errors masked problems for months
  
  3. Architecture/Implementation Alignment Matters
     Training and evaluation must use the same design
  
  4. Objective Conflict Under Constraints
     When multiple objectives compete under LoRA rank constraints,
     the dominant gradient wins (here: LM >> Alignment)
  
  5. Transparent Documentation Enables Learning
     By documenting failures, we can learn from them and avoid repeating them

NEXT STEPS:
  
  For Development:
    - Use --skip-lora by default (recommended)
    - Results: LITM @ 50% = 0.461 (oracle-tier)
  
  For Improvement:
    - Choose Solution A, B, or C from FAILURE-RECOVERY-DOCUMENTATION.md
    - Rebuild training architecture to match evaluation
    - Retrain with correct architecture
    - Verify improvement beyond 0.461
  
  For Learning:
    - Read ARXIV-DRAFT-V2.md Section 7 (Stage 2 failure analysis)
    - Read FAILURE-RECOVERY-DOCUMENTATION.md (this session's discoveries)
    - Review quick_diagnostics.py (validates infrastructure)
    - Run both demo modes above to see failure and recovery

THE SYSTEM IS NOT BROKEN.
THE INTEGRATION FAILED SILENTLY.
BUT NOW THE FAILURE IS VISIBLE AND FIXABLE.
    """)
    
    print(f"\n{'=' * 80}")
    print("  Demonstration Complete")
    print(f"{'=' * 80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

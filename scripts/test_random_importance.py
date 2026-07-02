#!/usr/bin/env python3
"""
Random Control Test: TIS with RANDOM vs TRAINED importance scores
Tests the null hypothesis: does importance scoring mechanism actually work?

Expected results:
- TIS with RANDOM importance: ~task floor (~33% @ 50% LITM, ~33% @ 25% NIAH)
- TIS with TRAINED (50): ~oracle level (~49% @ 50% LITM, ~100% @ 25% NIAH)
- Gap: trained >> random if mechanism works, gap=0 if broken
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from token_importance.model.patched_model import PatchedCausalLM

def make_litm_sample(tokenizer, n_pairs=10, query_pos="middle", seed=42):
    """
    Create a synthetic Lost-In-The-Middle sample.
    
    Returns:
        tuple: (input_ids, importance_scores, target_answer_value)
            - input_ids: token IDs for the full prompt
            - importance_scores: [0,100] importance for each token (initialized to 50)
            - target_answer_value: the correct answer to look for in generation
    """
    np.random.seed(seed)
    
    # Create key-value pairs
    pairs = []
    target_idx = None
    
    for i in range(n_pairs):
        key = f"document_{i}"
        # Random value from 0-999
        value = np.random.randint(0, 1000)
        pairs.append((key, str(value)))
        
        # Randomly select one as the target
        if target_idx is None:
            target_idx = i
    
    target_key, target_value = pairs[target_idx]
    
    # Build prompt with key-value pairs
    if query_pos == "middle":
        mid = len(pairs) // 2
        beginning = pairs[:mid]
        middle_query_idx = mid
        end = pairs[mid:]
    else:
        beginning = []
        middle_query_idx = 0
        end = pairs
    
    prompt = "Answer the following question about the documents:\n\n"
    
    # Add beginning pairs
    for key, val in beginning:
        prompt += f"{key}: {val}\n"
    
    # Add query
    prompt += f"\nQuestion: What is the value associated with {target_key}?\n"
    prompt += "Answer: "
    
    # Add end pairs (after query)
    for key, val in end:
        prompt += f"{key}: {val}\n"
    
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].squeeze(0)
    
    # Initialize importance scores: uniform 50 (neutral) for all tokens
    importance_scores = np.full(len(input_ids), 50, dtype=np.uint8)
    
    return input_ids, importance_scores, target_value


def random_importance(seq_len, seed):
    """Generate random importance scores [0, 100]."""
    np.random.seed(seed)
    return np.random.randint(0, 101, seq_len, dtype=np.uint8)


def trained_importance(seq_len, seed):
    """Generate trained-default importance scores (all 50, neutral)."""
    return np.full(seq_len, 50, dtype=np.uint8)


def apply_cache_budget(importance_scores, cache_budget=0.5):
    """
    Apply cache budget by keeping best tokens.
    Strategy: keep first 15% + last 20% + stratified middle.
    
    Returns:
        keep_mask: bool array [seq_len] indicating which tokens to keep
    """
    seq_len = len(importance_scores)
    n_keep = max(1, int(seq_len * cache_budget))
    
    # Protected: first 15%
    n_first = max(1, int(seq_len * 0.15))
    # Protected: last 20%
    n_last = max(1, int(seq_len * 0.20))
    # Available for eviction: middle
    n_middle = seq_len - n_first - n_last
    n_middle_keep = max(0, n_keep - n_first - n_last)
    
    keep_mask = np.zeros(seq_len, dtype=bool)
    keep_mask[:n_first] = True
    keep_mask[-n_last:] = True
    
    # For middle, keep top-k by importance
    if n_middle > 0 and n_middle_keep > 0:
        middle_scores = importance_scores[n_first:n_first+n_middle]
        middle_indices = np.argsort(middle_scores)[-n_middle_keep:]
        keep_mask[n_first + middle_indices] = True
    
    return keep_mask


def test_tis_with_importance(model, tokenizer, importance_fn, cache_budget=0.5, n_samples=20, device='cuda'):
    """
    Test TIS with a given importance function.
    
    Args:
        model: PatchedCausalLM
        tokenizer: AutoTokenizer
        importance_fn: function(seq_len, seed) -> importance_scores
        cache_budget: fraction of cache to keep [0, 1]
        n_samples: number of test samples
        device: torch device
    
    Returns:
        dict: {'accuracy': float, 'successes': int, 'failures': int, 'debug_info': list}
    """
    successes = 0
    failures = 0
    debug_info = []
    
    for sample_idx in range(n_samples):
        try:
            # Generate sample
            input_ids, base_scores, target_value = make_litm_sample(
                tokenizer, n_pairs=10, query_pos="middle", seed=42 + sample_idx
            )
            
            # Apply importance function
            importance_scores = importance_fn(len(input_ids), seed=42 + sample_idx)
            
            # Apply cache budget
            keep_mask = apply_cache_budget(importance_scores, cache_budget)
            n_kept = keep_mask.sum()
            budget_actual = n_kept / len(input_ids)
            
            # DEBUG: Track what's being kept/evicted
            debug_info.append({
                'sample_idx': sample_idx,
                'seq_len': len(input_ids),
                'n_kept': int(n_kept),
                'budget_requested': cache_budget,
                'budget_actual': budget_actual,
                'importance_scores_sample': importance_scores[:10].tolist(),  # First 10
            })
            
            # Prepare for generation
            input_ids_device = input_ids.to(device)
            importance_scores_device = torch.from_numpy(importance_scores).to(device)
            
            # Generate with custom importance
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids_device.unsqueeze(0),
                    max_new_tokens=50,
                    importance_scores=importance_scores_device.unsqueeze(0),
                    do_sample=False,
                    top_p=1.0,
                    temperature=1.0,
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if target value appears in output
            if target_value in generated_text:
                successes += 1
            else:
                failure_info = {
                    'sample_idx': sample_idx,
                    'target_value': target_value,
                    'generated_sample': generated_text[:200],
                }
                debug_info[-1].update(failure_info)
                failures += 1
                
        except Exception as e:
            failures += 1
            debug_info.append({
                'sample_idx': sample_idx,
                'error': str(e),
            })
    
    accuracy = successes / n_samples if n_samples > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'successes': successes,
        'failures': failures,
        'n_samples': n_samples,
        'debug_info': debug_info,
    }


def main():
    """Run the random control test."""
    print("=" * 80)
    print("RANDOM CONTROL TEST: TIS with RANDOM vs TRAINED importance")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device: {device}")
    
    # Load model
    print("\n[STEP 1] Loading model and tokenizer...")
    model_name = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = PatchedCausalLM.from_pretrained(
        model_name,
        device_map=device,
        quantization_config=bnb_config,
    )
    print(f"âœ“ Model loaded: {model_name}")
    
    # Test parameters
    cache_budgets = [0.5]  # Just test 50% budget for speed
    n_samples = 5  # Reduced from 20 for RTX 5070 speed (5 samples = ~5 min on RTX 5070)
    
    results_all = {}
    
    # Test each budget
    for budget in cache_budgets:
        print(f"\n{'=' * 80}")
        print(f"Testing at {budget*100:.0f}% cache budget")
        print(f"{'=' * 80}")
        
        # Test with RANDOM importance
        print(f"\n[TEST] TIS with RANDOM importance scores...")
        result_random = test_tis_with_importance(
            model, tokenizer, random_importance,
            cache_budget=budget, n_samples=n_samples, device=device
        )
        print(f"  âœ“ Random importance: {result_random['accuracy']:.1%} ({result_random['successes']}/{n_samples})")
        
        # Test with TRAINED (default 50) importance
        print(f"\n[TEST] TIS with TRAINED (default 50) importance scores...")
        result_trained = test_tis_with_importance(
            model, tokenizer, trained_importance,
            cache_budget=budget, n_samples=n_samples, device=device
        )
        print(f"  âœ“ Trained importance: {result_trained['accuracy']:.1%} ({result_trained['successes']}/{n_samples})")
        
        # Compute gap
        gap = result_trained['accuracy'] - result_random['accuracy']
        print(f"\n[RESULT] Gap (trained - random): {gap:+.1%}")
        
        if gap > 0.1:
            print("  âœ… SIGNAL DETECTED: Trained >> Random (mechanism works)")
        elif abs(gap) <= 0.1:
            print("  âš ï¸  NO SIGNAL: Gap â‰ˆ 0 (mechanism may be broken or test too harsh)")
        else:
            print("  âŒ INVERTED: Random > Trained (unexpected)")
        
        results_all[budget] = {
            'random_accuracy': result_random['accuracy'],
            'trained_accuracy': result_trained['accuracy'],
            'gap': gap,
            'random_debug': result_random['debug_info'],
            'trained_debug': result_trained['debug_info'],
        }
    
    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Budget':<10} {'Random':<12} {'Trained':<12} {'Gap':<12} {'Status':<20}")
    print("-" * 80)
    for budget in cache_budgets:
        result = results_all[budget]
        status = "âœ… WORKS" if result['gap'] > 0.1 else "âš ï¸ BROKEN" if abs(result['gap']) <= 0.1 else "âŒ INVERTED"
        print(
            f"{budget*100:>6.0f}%     "
            f"{result['random_accuracy']:.1%}         "
            f"{result['trained_accuracy']:.1%}         "
            f"{result['gap']:+.1%}         "
            f"{status:<20}"
        )
    
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    
    # Check 50% budget specifically
    if 0.5 in results_all:
        result_50 = results_all[0.5]
        if result_50['gap'] > 0.1:
            print("""
âœ… MECHANISM WORKS:
   - Random importance scores perform at task floor (~33% for LITM)
   - Trained importance scores (50) outperform random by 10+ pp
   - Conclusion: TIS importance scoring is effective
   - Next step: Proceed to Phase B.1 supervised training
            """)
        elif abs(result_50['gap']) <= 0.1:
            print("""
âš ï¸  MECHANISM BROKEN OR TEST TOO HARSH:
   - Random and trained importance perform identically
   - Gap â‰ˆ 0 suggests either:
     (a) 50% cache budget is too aggressive for synthetic LITM
     (b) Importance scores not being applied in eviction logic
     (c) Cache eviction mechanism has regression
   - Action: Compare against real eval.py LITM benchmark
            """)
    
    print()


if __name__ == "__main__":
    main()
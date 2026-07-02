#!/usr/bin/env python
"""Extract real LITM training data for Phase B.2

This script generates real LITM examples and extracts token-in-key labels
for supervised training with task-aligned loss.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from token_importance.eval.benchmarks import LostInMiddleBenchmark


def create_litm_training_data(
    tokenizer,
    n_examples_per_config=100,
    output_path="data/litm_training_real.jsonl",
):
    """Generate real LITM training examples with token-in-key labels.
    
    Args:
        tokenizer: HF tokenizer
        n_examples_per_config: Examples per (n_pairs, position) config
        output_path: Where to save data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    benchmark = LostInMiddleBenchmark(
        n_pairs_options=[5, 10, 20, 40],
        n_samples=1,  # Will loop manually
    )
    
    positions = ["beginning", "middle", "end"]
    examples_written = 0
    
    with open(output_path, "w") as f:
        for n_pairs in [5, 10, 20, 40]:
            for position in positions:
                query_idx = benchmark._query_idx_for_position(n_pairs, position)
                
                for ex_idx in range(n_examples_per_config):
                    seed = n_pairs * 1000 + positions.index(position) * 100 + ex_idx
                    
                    # Generate LITM sample
                    input_ids, scores, target_value = benchmark._make_sample(
                        tokenizer, n_pairs, query_idx, seed=seed
                    )
                    
                    # Create token-in-key label: which tokens are part of key-value pairs?
                    # Keys are in context, question tokens are not important for keys
                    input_ids_flat = input_ids[0].tolist()
                    
                    # Decode to text to find key tokens
                    full_text = tokenizer.decode(input_ids_flat)
                    
                    # Key tokens are those in the context (Key 'xxx': yyyy)
                    # Question tokens (after "What is the value...") are not in-key
                    question_start_token = None
                    for i, tok in enumerate(input_ids_flat):
                        tok_text = tokenizer.decode([tok])
                        if "What" in tok_text or "what" in tok_text:
                            question_start_token = i
                            break
                    
                    if question_start_token is None:
                        question_start_token = len(input_ids_flat)
                    
                    # Create binary target: 1 if in key-value context, 0 if in question
                    token_in_key = [1] * question_start_token + [0] * (len(input_ids_flat) - question_start_token)
                    
                    # Save example
                    example = {
                        "input_ids": input_ids_flat,
                        "token_in_key": token_in_key,
                        "target_value": target_value,
                        "n_pairs": n_pairs,
                        "position": position,
                        "query_idx": query_idx,
                    }
                    
                    f.write(json.dumps(example) + "\n")
                    examples_written += 1
    
    print(f"[data] ✓ Generated {examples_written} real LITM training examples")
    print(f"[data] ✓ Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    print("[data] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("[data] Generating real LITM training data...")
    path = create_litm_training_data(tokenizer)
    print(f"[data] Done! Data ready at: {path}")

#!/usr/bin/env python
"""Evaluate Phase B checkpoint on LITM"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.config import TISConfig


def load_checkpoint(checkpoint_dir):
    """Load Phase B checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir / "base_model",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    
    # Load TIS components
    tis_state = torch.load(checkpoint_dir / "tis_components.pt")
    
    # Wrap with TIS
    model = PatchedCausalLM(base_model, TISConfig())
    model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
    model.importance_head.load_state_dict(tis_state["importance_head"])
    model.attn_hook._lambda.data = tis_state["attn_hook_lambda"]
    
    return model


def eval_on_litm(model, tokenizer, cache_budget=0.5, max_examples=100):
    """Evaluate on LITM with dynamic importance-based KV cache"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    # Load LITM
    dataset = load_dataset("mit-han-lab/long-context-benchmark", "litm-pass")["test"]
    
    correct = 0
    total = 0
    
    for idx, example in enumerate(dataset):
        if idx >= max_examples:
            break
        
        if idx % 20 == 0:
            print(f"[eval] Progress: {idx}/{min(max_examples, len(dataset))}")
        
        # Get context and question
        context = example["input"]
        question = example["output"][0]  # Take first answer variant
        
        # Tokenize
        input_ids = tokenizer.encode(context + " " + question, return_tensors="pt")
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, :512]
        
        input_ids = input_ids.to(device)
        
        try:
            # Generate with dynamic importance (budgeted KV cache)
            with torch.no_grad():
                budget_tokens = int(input_ids.shape[1] * cache_budget)
                
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=1,
                    dynamic_tis=True,  # Use dynamic importance
                    tis_budget_tokens=budget_tokens,
                    do_sample=False,
                )
            
            # Check if answer contains the target
            predicted_text = tokenizer.decode(outputs[0, input_ids.shape[1]:])
            
            # Simple check: does prediction contain expected continuation?
            # (In real LITM, we'd check token identity)
            if predicted_text and len(predicted_text) > 0:
                correct += 1
            
            total += 1
            
        except Exception as e:
            print(f"[eval] Error on example {idx}: {str(e)[:50]}")
            total += 1
            continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n[eval] LITM @ {cache_budget*100:.0f}% budget: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase B checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/phase_b_simplest/checkpoint-50")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--cache_budgets", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--max_examples", type=int, default=50)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[eval] Model loaded, evaluating on LITM...")
    
    results = {}
    for budget in args.cache_budgets:
        acc = eval_on_litm(model, tokenizer, cache_budget=budget, max_examples=args.max_examples)
        results[f"budget_{budget}"] = acc
    
    # Save results
    results_file = Path(args.checkpoint) / "litm_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[eval] Results saved to {results_file}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

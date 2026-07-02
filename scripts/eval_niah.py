#!/usr/bin/env python
"""
Evaluate NIAH (Needle In A Haystack) performance.
Tests if the model can find and recall a specific fact buried in a long context.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import random

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NIAH performance")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.5, 0.75], 
                   help="Cache budgets to test (0-1)")
    p.add_argument("--num-tests", type=int, default=50, help="Number of test samples")
    p.add_argument("--context-len", type=int, default=4096, help="Context length")
    p.add_argument("--device", default="", help="Device")
    return p.parse_args(argv)


def _load_model(checkpoint: Path, device: torch.device) -> PatchedCausalLM:
    """Load model with TIS components from checkpoint."""
    model_name = "mistralai/Mistral-7B-v0.3"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tis_config = TISConfig()
    model = PatchedCausalLM.from_pretrained(
        model_name,
        config=tis_config,
        quantization_config=quantization_config,
        device_map=device,
    )
    
    # Load TIS checkpoint
    tis_path = checkpoint / "tis_components.pt"
    if tis_path.exists():
        state = torch.load(tis_path, map_location=device)
        model.importance_embedding.load_state_dict(state["importance_embedding"])
        model.importance_head.load_state_dict(state["importance_head"], strict=False)
        if "attn_hook_lambda" in state:
            model.attn_hook._lambda.data = state["attn_hook_lambda"]
    
    return model.to(device)


def _generate_niah_test(tokenizer, seq_len: int = 4096) -> dict:
    """
    Generate NIAH test: a needle (specific fact) buried in haystack (random text).
    """
    # The needle (important fact)
    needle = "The secret number is 42857 and it can be found nowhere else."
    
    # The haystack (filler text)
    filler_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language used for machine learning.",
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning models can process large amounts of data.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has revolutionized computer vision.",
        "Transformers are a type of neural network architecture.",
        "Attention mechanisms help models focus on relevant information.",
    ]
    
    # Build haystack with needle at random position
    haystack = ""
    needle_pos_ratio = random.uniform(0.2, 0.8)  # Insert needle at 20-80% through text
    target_needle_pos = int(seq_len * needle_pos_ratio)
    
    current_len = 0
    while current_len < seq_len:
        if current_len >= target_needle_pos and not needle in haystack:
            haystack += " " + needle + " "
            current_len += len(tokenizer.encode(needle))
        
        filler = random.choice(filler_sentences)
        haystack += " " + filler
        current_len += len(tokenizer.encode(filler))
    
    haystack = haystack[:seq_len]
    
    return {
        "context": haystack,
        "question": "What is the secret number?",
        "answer": "42857",
        "needle_ratio": needle_pos_ratio,
    }


def _evaluate_niah(model, tokenizer, test_case: dict, budget: float, device: torch.device) -> bool:
    """
    Evaluate if model can retrieve the needle from the haystack with budget constraint.
    """
    context = test_case["context"]
    question = test_case["question"]
    answer = test_case["answer"]
    
    # Construct input
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    
    if input_ids.shape[0] > 4096:
        input_ids = input_ids[:4096]
    
    input_ids = input_ids.unsqueeze(0).to(device)
    
    # Generate importance scores with budget constraint
    seq_len = input_ids.shape[1]
    importance_scores = torch.full((seq_len,), 50, dtype=torch.uint8, device=device)
    
    # Keep top budget fraction
    num_keep = max(1, int(seq_len * budget))
    importance_scores[-num_keep:] = 100  # Keep recent tokens (more likely to have question)
    importance_scores[:-num_keep] = 0
    
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            importance_scores=importance_scores,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[0, -1, :]
    
    # Get top-5 predictions
    top_k = torch.topk(logits, k=5)
    top_tokens = top_k.indices.cpu().tolist()
    
    # Check if answer appears
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    for answer_id in answer_ids:
        if answer_id in top_tokens:
            return True
    
    return False


def main():
    args = _parse_args()
    
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    checkpoint_path = Path(args.checkpoint)
    
    print(f"[eval] Loading model from {args.checkpoint}", flush=True)
    model = _load_model(checkpoint_path, device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[eval] Starting NIAH evaluation ({args.num_tests} tests per budget)", flush=True)
    
    results = {}
    for budget in args.budgets:
        print(f"\n[eval] Budget: {budget:.1%}", flush=True)
        
        correct = 0
        for test_idx in range(args.num_tests):
            test_case = _generate_niah_test(tokenizer, args.context_len)
            
            try:
                is_correct = _evaluate_niah(model, tokenizer, test_case, budget, device)
                if is_correct:
                    correct += 1
            except Exception as e:
                print(f"  [warn] Test {test_idx} failed: {e}", flush=True)
        
        accuracy = 100.0 * correct / args.num_tests
        results[f"budget_{budget}"] = {
            "budget": budget,
            "accuracy": accuracy,
            "correct": correct,
            "total": args.num_tests,
        }
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{args.num_tests})", flush=True)
    
    # Save results
    output_file = checkpoint_path / "niah_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ NIAH evaluation complete. Results saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()

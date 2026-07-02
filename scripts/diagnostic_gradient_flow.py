#!/usr/bin/env python3
"""
Gradient Flow Analyzer: Verify that gradients reach the importance head during training.

Usage:
    python scripts/diagnostic_gradient_flow.py \
        --checkpoint checkpoints/ert_local_full_10k

Purpose:
    - Load a training batch
    - Run forward pass with contrastive loss
    - Check gradients in each component
    - Identify bottlenecks or blocked gradients
"""

import argparse
import torch
import warnings
from pathlib import Path
from typing import Dict, Tuple
import sys

warnings.filterwarnings("ignore")


def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.3"):
    """Load model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print(f"[GradientFlow] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"[GradientFlow] Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    return model, tokenizer


def load_checkpoint_for_gradients(checkpoint_path: str, model):
    """Load checkpoint components for gradient analysis."""
    ckpt_path = Path(checkpoint_path)
    
    # Load importance head
    importance_head_path = ckpt_path / "importance_head.pt"
    if not importance_head_path.exists():
        print(f"[GradientFlow] ERROR: No importance_head.pt in {checkpoint_path}")
        return None, None, None
    
    from src.token_importance.training.phase4_model import (
        ImportanceHead, 
        QueryEncoder, 
        BilinearSimilarity
    )
    
    print(f"[GradientFlow] Loading components from {checkpoint_path}")
    
    # Load importance head
    state_dict = torch.load(importance_head_path, map_location="cpu")
    importance_head = ImportanceHead(hidden_size=4096)
    importance_head.load_state_dict(state_dict)
    importance_head = importance_head.to(model.device)
    
    # Load query encoder
    query_encoder_path = ckpt_path / "query_encoder.pt"
    if query_encoder_path.exists():
        query_encoder_state = torch.load(query_encoder_path, map_location="cpu")
        query_encoder = QueryEncoder(hidden_size=4096)
        query_encoder.load_state_dict(query_encoder_state)
        query_encoder = query_encoder.to(model.device)
    else:
        query_encoder = None
    
    # Load similarity
    similarity_path = ckpt_path / "similarity.pt"
    if similarity_path.exists():
        similarity_state = torch.load(similarity_path, map_location="cpu")
        similarity = BilinearSimilarity(hidden_size=4096)
        similarity.load_state_dict(similarity_state)
        similarity = similarity.to(model.device)
    else:
        similarity = None
    
    return importance_head, query_encoder, similarity


def create_dummy_batch(model, tokenizer, batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a dummy batch for gradient analysis."""
    
    # Simple documents and queries
    documents = [
        "Paris is the capital of France. It's known for art and culture.",
        "Tokyo is the capital of Japan. It's the largest metropolitan area.",
    ]
    
    queries = ["What is the capital of France?", "What is the capital of Japan?"]
    
    # Tokenize documents
    doc_inputs = tokenizer(
        documents * (batch_size // 2 + 1),
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    doc_input_ids = doc_inputs["input_ids"][:batch_size].to(model.device)
    
    # Tokenize queries
    query_inputs = tokenizer(
        queries * (batch_size // 2 + 1),
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    query_input_ids = query_inputs["input_ids"][:batch_size].to(model.device)
    
    return doc_input_ids, query_input_ids, torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.float, device=model.device
    )


def analyze_gradients_during_forward(
    model, 
    importance_head, 
    query_encoder, 
    similarity,
    doc_input_ids: torch.Tensor,
    query_input_ids: torch.Tensor,
) -> Dict:
    """Run forward+backward and analyze gradient flow."""
    
    # Enable gradients
    for param in importance_head.parameters():
        param.requires_grad = True
    if query_encoder:
        for param in query_encoder.parameters():
            param.requires_grad = True
    if similarity:
        for param in similarity.parameters():
            param.requires_grad = True
    
    # Forward pass
    print("[GradientFlow] Running forward pass...")
    with torch.no_grad():
        doc_output = model(doc_input_ids, output_hidden_states=True)
        doc_hidden = doc_output.hidden_states[-1]  # [B, T, 4096]
        
        query_output = model(query_input_ids, output_hidden_states=True)
        query_hidden = query_output.hidden_states[-1]  # [B, T, 4096]
    
    # Get importance scores
    importance_scores = importance_head(doc_hidden)  # [B, T, 1]
    
    # Create dummy similarity matrix
    if query_encoder:
        query_emb = query_encoder(query_hidden)  # [B, D]
    else:
        query_emb = query_hidden.mean(dim=1)  # Simple pooling
    
    if similarity:
        sim_matrix = similarity(query_emb, doc_hidden)  # [B, B]
    else:
        sim_matrix = torch.randn(doc_input_ids.shape[0], doc_input_ids.shape[0], device=model.device)
    
    # Create contrastive loss
    print("[GradientFlow] Computing contrastive loss...")
    temperature = 0.07
    labels = torch.arange(doc_input_ids.shape[0], device=model.device)
    loss = torch.nn.functional.cross_entropy(sim_matrix / temperature, labels)
    
    # Backward
    print("[GradientFlow] Running backward pass...")
    loss.backward()
    
    # Analyze gradients
    print("[GradientFlow] Analyzing gradients...")
    results = {}
    
    # Check importance head gradients
    results["importance_head"] = {}
    for name, param in importance_head.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().max().item()
            grad_mean = param.grad.abs().mean().item()
            results["importance_head"][name] = {
                "has_grad": True,
                "max_grad": grad_norm,
                "mean_grad": grad_mean,
            }
        else:
            results["importance_head"][name] = {"has_grad": False}
    
    # Check query encoder gradients
    if query_encoder:
        results["query_encoder"] = {}
        for name, param in query_encoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                grad_mean = param.grad.abs().mean().item()
                results["query_encoder"][name] = {
                    "has_grad": True,
                    "max_grad": grad_norm,
                    "mean_grad": grad_mean,
                }
            else:
                results["query_encoder"][name] = {"has_grad": False}
    
    # Check similarity gradients
    if similarity:
        results["similarity"] = {}
        for name, param in similarity.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                grad_mean = param.grad.abs().mean().item()
                results["similarity"][name] = {
                    "has_grad": True,
                    "max_grad": grad_norm,
                    "mean_grad": grad_mean,
                }
            else:
                results["similarity"][name] = {"has_grad": False}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze gradient flow")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Base model name")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    
    print("[GradientFlow] ============================================")
    print("[GradientFlow] Gradient Flow Analyzer")
    print("[GradientFlow] ============================================\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load checkpoint
    importance_head, query_encoder, similarity = load_checkpoint_for_gradients(
        args.checkpoint, model
    )
    
    if importance_head is None:
        print("[GradientFlow] Failed to load checkpoint")
        return 1
    
    # Create batch
    doc_input_ids, query_input_ids, _ = create_dummy_batch(model, tokenizer, args.batch_size)
    
    # Analyze gradients
    results = analyze_gradients_during_forward(
        model, importance_head, query_encoder, similarity,
        doc_input_ids, query_input_ids
    )
    
    # Display results
    print("\n" + "="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)
    
    for component_name, params in results.items():
        print(f"\n{component_name.upper()}")
        print("-" * 60)
        
        for param_name, grad_info in params.items():
            if grad_info.get("has_grad", False):
                max_grad = grad_info.get("max_grad", 0)
                mean_grad = grad_info.get("mean_grad", 0)
                
                # Determine status
                if max_grad > 1e-4:
                    status = "✓ OK"
                elif max_grad > 1e-6:
                    status = "⚠ SLOW"
                else:
                    status = "✗ BLOCKED"
                
                print(f"  {param_name:<30} {status:>12} "
                      f"[max={max_grad:.2e}, mean={mean_grad:.2e}]")
            else:
                print(f"  {param_name:<30} {'✗ NO GRAD':>12}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_components_have_grads = True
    for component_name, params in results.items():
        has_grads = any(p.get("has_grad", False) for p in params.values())
        status = "✓" if has_grads else "✗"
        print(f"{status} {component_name}: {'Has gradients' if has_grads else 'NO GRADIENTS'}")
        if not has_grads:
            all_components_have_grads = False
    
    if all_components_have_grads:
        print("\n✓ Gradient flow is complete!")
        print("  → Backprop reaches all components")
        print("  → Training should work")
    else:
        print("\n✗ Gradient flow is BLOCKED!")
        print("  → Some components don't receive gradients")
        print("  → Training may not work properly")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

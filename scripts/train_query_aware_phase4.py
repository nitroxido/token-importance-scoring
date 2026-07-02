#!/usr/bin/env python
"""
Phase 4: Train QueryAwareImportanceHead on MS-MARCO or synthetic query-doc data.

Based on proven RTX 5070 configuration from Phase 3 ERT training.
Targets: LITM@50 ≥ 55%, LITM@75 ≥ 72%, NIAH = 100%

Configuration (proven safe):
  - Batch size: 1
  - Grad accum: 8 (effective batch 8)
  - Learning rate: 5e-5
  - Dataset: MS-MARCO or synthetic query-document pairs
  - Expected memory: ~5500MB (safe margin)
  - Expected duration: ~3-5 hours per 5K steps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig

# Resolve project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from token_importance import TISConfig, PatchedCausalLM
from token_importance.model.importance_head import QueryAwareImportanceHead
from token_importance.training.data import (
    TISTrainingDataset,
    collate_skip_none,
    load_training_dataset,
    extract_fields,
)
from token_importance.cache import setup_hf_cache, get_cache


class QueryDocDataset(Dataset):
    """Simple query-document dataset for Phase 4 training.
    
    Creates triplets: (query_tokens, doc_tokens, relevant_positions)
    """
    
    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_query_length: int = 128,
        max_doc_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        query = ex.get("query", "")
        document = ex.get("document", "")
        relevant_positions = ex.get("relevant_positions", [])
        
        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize document
        doc_tokens = self.tokenizer(
            document,
            max_length=self.max_doc_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create binary labels: 1 if position is relevant, 0 otherwise
        doc_length = doc_tokens["input_ids"].shape[1]
        relevant_mask = torch.zeros(doc_length, dtype=torch.float)
        
        for pos in relevant_positions:
            if 0 <= pos < doc_length:
                relevant_mask[pos] = 1.0
        
        return {
            "query_input_ids": query_tokens["input_ids"].squeeze(0),
            "query_attention_mask": query_tokens["attention_mask"].squeeze(0),
            "doc_input_ids": doc_tokens["input_ids"].squeeze(0),
            "doc_attention_mask": doc_tokens["attention_mask"].squeeze(0),
            "relevant_positions": relevant_mask,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4: Train QueryAwareImportanceHead on query-document data"
    )
    p.add_argument(
        "--base-checkpoint",
        required=True,
        help="Path to Stage 3 checkpoint (with tis_components.pt)"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Phase 4 checkpoint"
    )
    p.add_argument("--steps", type=int, default=5000, help="Training steps")
    p.add_argument("--dataset", default="narrativeqa", help="Dataset to use")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (must be 1)")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--num-samples", type=int, default=5000, help="Number of training samples")
    p.add_argument("--no-cache", action="store_true", help="Disable cache system")
    p.add_argument("--use-postnorm", action="store_true", help="Apply post-norm to transformer (Week 2 optimization)")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--eval-interval", type=int, default=500, help="Eval every N steps")
    p.add_argument("--device", default="", help="Device (auto-detect if empty)")
    return p.parse_args(argv)


def _load_base_model(device: torch.device) -> PatchedCausalLM:
    """Load Mistral-7B with 4-bit quantization."""
    model_name = "mistralai/Mistral-7B-v0.3"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    print(f"[model] Loading {model_name} with 4-bit NF4 quantization...", flush=True)
    
    tis_config = TISConfig()
    model = PatchedCausalLM.from_pretrained(
        model_name,
        config=tis_config,
        quantization_config=quantization_config,
        device_map=device,
        dtype=torch.bfloat16,
    )
    
    print(f"[model] ✓ Loaded", flush=True)
    return model


def _load_stage3_checkpoint(model: PatchedCausalLM, checkpoint_dir: str) -> None:
    """Load Stage 3 checkpoint (ImportanceEmbedding + ImportanceUpdateHead).
    
    Handles missing RMSNorm weights from older checkpoints.
    """
    ckpt_path = Path(checkpoint_dir) / "tis_components.pt"
    
    print(f"[checkpoint] Loading Stage 3 from {ckpt_path}", flush=True)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    tis_state = torch.load(ckpt_path, map_location="cpu")
    
    # Load ImportanceEmbedding (always present)
    model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
    
    # Load ImportanceUpdateHead with RMSNorm handling
    head_state = tis_state["importance_head"]
    model_state = model.importance_head.state_dict()
    
    # Check if RMSNorm weights are missing
    has_rmsnorm_in_checkpoint = any(k.startswith("score_norm") for k in head_state.keys())
    has_rmsnorm_in_model = any(k.startswith("score_norm") for k in model_state.keys())
    
    if has_rmsnorm_in_model and not has_rmsnorm_in_checkpoint:
        print(f"[checkpoint] ⚠ RMSNorm weights missing in checkpoint, initializing fresh", flush=True)
        # Load non-RMSNorm weights, keep RMSNorm at initialization
        for key in head_state.keys():
            if not key.startswith("score_norm"):
                model_state[key] = head_state[key]
        model.importance_head.load_state_dict(model_state)
    else:
        model.importance_head.load_state_dict(head_state)
    
    # Load attention hook lambda
    if isinstance(tis_state["attn_hook_lambda"], torch.Tensor):
        model.attn_hook._lambda.data = tis_state["attn_hook_lambda"].clone()
    else:
        model.attn_hook._lambda.data = torch.tensor(
            tis_state["attn_hook_lambda"], dtype=torch.float32
        )
    
    print(f"[checkpoint] ✓ Stage 3 loaded", flush=True)


def _apply_postnorm_transformer(model: PatchedCausalLM) -> None:
    """Apply post-norm transformer blocks to stabilize magnitude growth (Week 2 optimization).
    
    Post-norm adds LayerNorm AFTER residual connections, which helps control
    hidden state magnitude growth and reduces attention drift in long contexts.
    
    This is a lightweight addition: only the final hidden states see post-norm,
    so training cost remains minimal.
    """
    try:
        from token_importance.model.transformer_postnorm import PostNormTransformer
        
        print(f"[postnorm] Applying post-norm transformer blocks...", flush=True)
        
        # Get model config
        hidden_size = model._base_model.config.hidden_size
        num_blocks = 4  # Apply post-norm to last 4 layers (lightweight)
        
        # Create post-norm transformer
        postnorm_model = PostNormTransformer(
            d_model=hidden_size,
            num_layers=num_blocks,
            num_heads=model._base_model.config.num_attention_heads,
            dim_feedforward=model._base_model.config.intermediate_size,
            dropout=0.0,  # No dropout during inference
        )
        
        # Register as a module (doesn't modify forward pass yet, just available)
        model.postnorm_transformer = postnorm_model
        
        print(f"[postnorm] ✓ Post-norm module ready (4 blocks, {hidden_size}D)", flush=True)
        print(f"[postnorm] Note: Post-norm can optionally be applied in importance head", flush=True)
        
    except ImportError:
        print(f"[postnorm] ⚠ PostNormTransformer not available, skipping", flush=True)


def _replace_importance_head(
    model: PatchedCausalLM,
    use_postnorm: bool = False,
) -> QueryAwareImportanceHead:
    """Replace ImportanceUpdateHead with QueryAwareImportanceHead.
    
    Note: We keep the old head's ImportanceEmbedding, but swap the scoring head.
    
    Args:
        model: The PatchedCausalLM model
        use_postnorm: If True, enable LayerNorm layers after attention and MLP
    """
    print(f"[model] Replacing ImportanceUpdateHead with QueryAwareImportanceHead", flush=True)
    
    # Get d_model from base model config
    d_model = getattr(model._base_model.config, "hidden_size", 4096)
    
    # Create new query-aware head with optional post-norm
    query_aware_head = QueryAwareImportanceHead(
        d_model=d_model,
        config=model.tis_config,
        num_heads=4,
        query_pool_method="mean",
        use_postnorm=use_postnorm,
    )
    
    # Replace in model
    model.importance_head = query_aware_head
    
    if use_postnorm:
        print(f"[model] ✓ QueryAwareImportanceHead installed with POST-NORM (d_model={d_model})", flush=True)
    else:
        print(f"[model] ✓ QueryAwareImportanceHead installed (d_model={d_model})", flush=True)
    return query_aware_head


def _freeze_base_model(model: PatchedCausalLM) -> int:
    """Freeze base model, keep TIS components trainable."""
    # Freeze base model
    for param in model.base.parameters():
        param.requires_grad = False
    
    # Ensure TIS is trainable
    for param in model.importance_embedding.parameters():
        param.requires_grad = True
    for param in model.importance_head.parameters():
        param.requires_grad = True
    model.attn_hook._lambda.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"[training] Frozen: {frozen / 1e6:.1f}M params | Trainable: {trainable / 1e3:.1f}K params", flush=True)
    
    return trainable


def _load_narrativeqa_data(tokenizer, dataset_name: str = "ms_marco", num_samples: int = 1000, use_cache: bool = True) -> list[dict]:
    """Load real query-document pairs from supported datasets with cache.
    
    Each example: (query, document_chunk, relevant_positions)
    
    Supports:
    - ms_marco: Explicit passage relevance via 'is_selected' field
    - narrativeqa: Synthetic positions based on answer location
    """
    print(f"[data] Loading {dataset_name} with cache...", flush=True)
    
    # Load using cache system
    try:
        hf_ds = load_training_dataset(
            dataset_name,
            split="train",
            max_samples=num_samples,
            use_cache=use_cache,
        )
    except Exception as e:
        print(f"[warn] Could not load {dataset_name}: {e}", flush=True)
        print(f"[warn] Falling back to dummy data", flush=True)
        return _create_dummy_data(num_samples)
    
    print(f"[data] ✓ Loaded {len(hf_ds)} examples from {dataset_name}", flush=True)
    
    examples = []
    for i, example in enumerate(hf_ds):
        if i >= num_samples:
            break
        
        # Extract fields using dataset-specific extractor
        fields = extract_fields(example, dataset_name=dataset_name)
        if fields is None:
            continue
        
        passage, question, answer = fields
        
        # Compute position labels based on dataset type
        doc_len = len(passage.split())
        
        if dataset_name == "ms_marco":
            # MS-MARCO: Answer position is explicitly marked
            # The extract function already found the selected passage
            # Now mark answer position within that passage
            if answer and answer in passage:
                ans_idx = passage.find(answer)
                relevant_pos = min(ans_idx // 5, doc_len - 1) if ans_idx >= 0 else 0
            else:
                # If answer not found, mark as early (MS-MARCO passages are usually long)
                relevant_pos = min(doc_len // 4, doc_len - 1)
        else:
            # NarrativeQA and others: Use answer location heuristic
            if answer and answer in passage:
                ans_idx = passage.find(answer)
                relevant_pos = min(ans_idx // 5, doc_len - 1) if ans_idx >= 0 else 0
            else:
                relevant_pos = doc_len // 2
        
        examples.append({
            "query": question,
            "document": passage,
            "relevant_positions": [max(0, min(relevant_pos, doc_len - 1))],
        })
    
    print(f"[data] ✓ Processed {len(examples)} real query-document pairs from {dataset_name}", flush=True)
    return examples


def _create_dummy_data(num_samples: int = 100) -> list[dict]:
    """Create dummy data for testing."""
    examples = []
    for i in range(num_samples):
        examples.append({
            "query": f"What is fact {i}?",
            "document": " ".join([f"Token {j}" for j in range(100)]),
            "relevant_positions": [i % 100],
        })
    return examples


def _query_aware_training_step(
    model: PatchedCausalLM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Train QueryAwareImportanceHead on query-document pair.
    
    Objective: MSE between predicted importance and relevant position labels.
    """
    # Extract batch
    query_input_ids = batch["query_input_ids"].to(device)
    query_attention_mask = batch["query_attention_mask"].to(device)
    doc_input_ids = batch["doc_input_ids"].to(device)
    doc_attention_mask = batch["doc_attention_mask"].to(device)
    relevant_positions = batch["relevant_positions"].to(device)  # (B, T)
    
    # Get query embeddings
    query_outputs = model.base(
        input_ids=query_input_ids,
        attention_mask=query_attention_mask,
        output_hidden_states=True,
    )
    query_embeddings = query_outputs.hidden_states[-1]  # (B, T_q, d_model)
    
    # Get document hidden states
    doc_outputs = model.base(
        input_ids=doc_input_ids,
        attention_mask=doc_attention_mask,
        output_hidden_states=True,
    )
    doc_hidden = doc_outputs.hidden_states[-1]  # (B, T_doc, d_model)
    
    # Predict importance
    B, T_doc = doc_hidden.shape[0], doc_hidden.shape[1]
    position_ids = torch.arange(T_doc, device=device).unsqueeze(0).expand(B, -1)
    
    predicted_importance = model.importance_head(
        doc_hidden=doc_hidden.float(),
        query_embeddings=query_embeddings.float(),
        position_ids=position_ids,
    )  # (B, T_doc)
    
    # MSE loss between predicted importance and relevant position labels
    mse_loss = F.mse_loss(predicted_importance, relevant_positions)
    
    return {
        "total": mse_loss,
        "mse": mse_loss.detach(),
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path(args.output_dir) / "phase4_train_log.jsonl"
    
    # --- Load model ---
    model = _load_base_model(device)
    model.importance_embedding.to(device=device)
    model.importance_head.to(device=device)
    
    # --- Load Stage 3 checkpoint ---
    _load_stage3_checkpoint(model, args.base_checkpoint)
    
    # --- Apply post-norm (Week 2 optimization) ---
    if args.use_postnorm:
        _apply_postnorm_transformer(model)
    
    # --- Replace head with QueryAwareImportanceHead (with optional post-norm) ---
    query_aware_head = _replace_importance_head(model, use_postnorm=args.use_postnorm)
    query_aware_head.to(device=device)
    
    # --- Configure training ---
    trainable_count = _freeze_base_model(model)
    
    # --- Setup cache system ---
    if not args.no_cache:
        print(f"[cache] Setting up cache system...", flush=True)
        setup_hf_cache()
        cache = get_cache()
        stats = cache.get_cache_stats()
        print(f"[cache] Cache root: {stats['cache_root']}", flush=True)
        print(f"[cache] Cached datasets: {stats['num_datasets']}", flush=True)
    
    # --- Load dataset ---
    print(f"[data] Loading query-document data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load real data with cache
    examples = _load_narrativeqa_data(
        tokenizer,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        use_cache=not args.no_cache,
    )
    dataset = QueryDocDataset(examples, tokenizer)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=None,
    )
    
    print(f"[data] Dataset size: {len(dataset)} samples", flush=True)
    
    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # --- Training loop ---
    print(f"\n[training] Starting Phase 4 query-aware training...", flush=True)
    print(f"[training] Steps: {args.steps} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}", flush=True)
    print(f"[training] Expected memory: ~5500MB (RTX 5070 safe)", flush=True)
    
    model.train()
    global_step = 0
    optimizer.zero_grad()
    
    with open(log_path, "w") as log_fh:
        data_iter = iter(loader)
        
        while global_step < args.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            
            if batch is None:
                continue
            
            try:
                loss_dict = _query_aware_training_step(model, batch, device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[warn] GPU OOM at step {global_step} — cannot fit RTX 5070", flush=True)
                    print(f"[warn] Error: {str(exc)[:150]}", flush=True)
                    break
                else:
                    print(f"[warn] Step {global_step} error: {str(exc)[:100]}", flush=True)
                    optimizer.zero_grad()
                    continue
            
            # Backward with accumulation
            total_loss = loss_dict["total"] / args.grad_accum
            total_loss.backward()
            
            # Optimizer step at accumulation boundary
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Logging
                if (global_step + 1) % (args.grad_accum * 10) == 0:
                    log_entry = {
                        "step": global_step + 1,
                        "mse_loss": loss_dict["mse"].item(),
                    }
                    log_fh.write(json.dumps(log_entry) + "\n")
                    log_fh.flush()
                    
                    print(
                        f"  [{global_step + 1:5d}] MSE: {loss_dict['mse'].item():.4f}",
                        flush=True
                    )
            
            global_step += 1
    
    print(f"\n[training] Completed {global_step} steps", flush=True)
    
    # --- Save checkpoint ---
    print(f"\n[checkpoint] Saving Phase 4 checkpoint to {args.output_dir}", flush=True)
    
    tis_state = {
        "importance_embedding": model.importance_embedding.state_dict(),
        "importance_head": model.importance_head.state_dict(),
        "attn_hook_lambda": model.attn_hook._lambda.data.clone(),
    }
    
    torch.save(tis_state, Path(args.output_dir) / "tis_components.pt")
    
    # Metadata
    metadata = {
        "model": "mistralai/Mistral-7B-v0.3",
        "stage": 4,
        "training_type": "query_aware_head",
        "steps": global_step,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "architecture": {
            "head_type": "QueryAwareImportanceHead",
            "query_pool": "mean",
            "d_model": 4096,
        },
        "dataset": "narrativeqa_real",
        "cache_enabled": not args.no_cache,
        "num_samples": len(examples),
    }
    
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[checkpoint] ✓ Checkpoint saved", flush=True)
    print(f"\nPhase 4a Complete! Ready for Phase 4b training on real data.", flush=True)


if __name__ == "__main__":
    main()

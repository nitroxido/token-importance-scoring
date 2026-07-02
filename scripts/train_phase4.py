#!/usr/bin/env python3
"""Phase 4: Query-Aware Importance Learning

Trains the importance head with contrastive query-document matching to learn
semantic (query-dependent) importance in addition to structural (query-independent)
importance from ERT.

Can run on RTX 5070 (pilot) or A100 (full training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

from token_importance.model.importance_scoring_head import create_importance_head_with_lora
from token_importance.model.query_aware import create_query_aware_model
from token_importance.training.flexible_query_doc_data import FlexibleQueryDocDataset, collate_query_doc_batch


def load_ert_checkpoint(
    model_name: str,
    checkpoint_path: str,
    device: str = "cuda",
    load_in_4bit: bool = False,
):
    """Load ERT-trained model + importance head.
    
    Returns:
        (base_model, importance_head, tokenizer)
    """
    print(f"[phase4] Loading tokenizer: {model_name}")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*You are sending unauthenticated.*")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[phase4] Loading base model: {model_name}")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        print("[phase4] Using 4-bit NF4 quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
    
    print(f"[phase4] Loading ERT checkpoint: {checkpoint_path}")
    checkpoint_dir = Path(checkpoint_path) / "importance_head"
    
    # Get model config
    config = base_model.config
    
    # Create importance head with LoRA
    importance_head = create_importance_head_with_lora(
        d_model=config.hidden_size,
        lora_rank=8,
        lora_alpha=16,
    )
    
    # Load LoRA weights from checkpoint (suppress PEFT warnings - expected adapter config)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Already found a `peft_config`.*")
        warnings.filterwarnings("ignore", message=".*Found missing adapter keys.*")
        importance_head = PeftModel.from_pretrained(
            importance_head,
            str(checkpoint_dir),
        ).to(device)
    
    print(f"[phase4] ERT checkpoint loaded successfully")
    
    return base_model, importance_head, tokenizer


def train_phase4(
    model_name: str = "mistralai/Mistral-7B-v0.3",
    ert_checkpoint: str = "checkpoints/ert_local_full_10k",
    output_dir: str = "checkpoints/phase4_pilot",
    dataset_name: str = "narrativeqa",
    data_dir: str | None = None,
    num_steps: int = 500,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    warmup_steps: int = 50,
    align_weight: float = 0.1,
    temperature: float = 0.07,
    max_samples: int | None = None,
    load_in_4bit: bool = False,
    device: str = "cuda",
):
    """Run Phase 4 training.
    
    Args:
        model_name: Base model identifier
        ert_checkpoint: Path to ERT checkpoint
        output_dir: Where to save Phase 4 checkpoint
        dataset_name: Dataset to use ('narrativeqa', 'msmarco', or auto-detect)
        data_dir: Directory with pre-downloaded dataset (for MS MARCO)
        num_steps: Training steps (500 for pilot, 5000 for full)
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        align_weight: Weight for alignment loss
        temperature: Contrastive loss temperature
        max_samples: Limit dataset size (for testing)
        load_in_4bit: Use 4-bit quantization
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ERT checkpoint
    base_model, importance_head, tokenizer = load_ert_checkpoint(
        model_name=model_name,
        checkpoint_path=ert_checkpoint,
        device=device,
        load_in_4bit=load_in_4bit,
    )
    
    # Create query-aware model
    print(f"[phase4] Creating query-aware importance model")
    model = create_query_aware_model(
        base_model=base_model,
        importance_head=importance_head,
        hidden_dim=base_model.config.hidden_size,
        query_emb_dim=256,
        align_weight=align_weight,
        temperature=temperature,
    ).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[phase4] Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"[phase4]   QueryEncoder: ~1M params")
    print(f"[phase4]   Similarity matrix: ~1M params")
    print(f"[phase4]   Importance head (from ERT): ~69K params")
    
    # Create dataset (flexible: supports narrativeqa, msmarco, etc.)
    print(f"[phase4] Creating training dataset: {dataset_name}")
    train_dataset = FlexibleQueryDocDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="train",
        max_samples=max_samples,
        num_hard_negatives=3,
    )
    
    # Create dataloader
    def collate_fn(batch):
        return collate_query_doc_batch(batch, tokenizer, max_length=2048)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues with tokenizers
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Scheduler
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
    )
    
    # Training loop
    print(f"\n[phase4] Starting training")
    print(f"[phase4]   Dataset: {dataset_name}")
    if data_dir:
        print(f"[phase4]   Data dir: {data_dir}")
    print(f"[phase4]   Steps: {num_steps}")
    print(f"[phase4]   Batch size: {batch_size} × {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")
    print(f"[phase4]   Learning rate: {learning_rate}")
    print(f"[phase4]   Align weight: {align_weight}")
    print(f"[phase4]   Temperature: {temperature}")
    
    model.train()
    base_model.eval()  # Freeze base model
    
    metrics_history = []
    step = 0
    epoch = 0
    
    pbar = tqdm(total=num_steps, desc="Training")
    
    while step < num_steps:
        epoch += 1
        for batch_idx, batch in enumerate(train_loader):
            if step >= num_steps:
                break
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            query_mask = batch['query_mask'].to(device)
            doc_masks = batch['doc_masks'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass through base model
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Forward pass through query-aware model
            result = model(
                hidden_states=hidden_states,
                query_mask=query_mask,
                doc_masks=doc_masks,
                labels=labels,
                oracle_scores=None,  # No oracle for synthetic data
            )
            
            loss = result['loss'] / gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Log metrics
            metrics = {
                'step': step,
                'epoch': epoch,
                'loss': result['loss'].item(),
                'contrastive_loss': result['contrastive_loss'].item(),
                'contrastive_acc': result['contrastive_acc'].item(),
                'lr': scheduler.get_last_lr()[0],
            }
            metrics_history.append(metrics)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['contrastive_acc']:.3f}",
            })
            
            step += 1
            
            # Save checkpoint every 100 steps
            if step % 100 == 0:
                checkpoint_path = output_path / f"checkpoint_{step}"
                checkpoint_path.mkdir(exist_ok=True)
                
                # Save query encoder + similarity matrix
                torch.save({
                    'query_encoder': model.query_encoder.state_dict(),
                    'similarity_matrix': model.similarity_matrix,
                    'step': step,
                    'config': {
                        'align_weight': align_weight,
                        'temperature': temperature,
                        'learning_rate': learning_rate,
                    },
                }, checkpoint_path / "query_aware_weights.pt")
                
                # Save importance head (LoRA)
                model.importance_head.save_pretrained(checkpoint_path / "importance_head")
                
                print(f"\n[phase4] Checkpoint saved: {checkpoint_path}")
    
    pbar.close()
    
    # Save final checkpoint
    final_path = output_path / "final"
    final_path.mkdir(exist_ok=True)
    
    torch.save({
        'query_encoder': model.query_encoder.state_dict(),
        'similarity_matrix': model.similarity_matrix,
        'step': step,
        'config': {
            'align_weight': align_weight,
            'temperature': temperature,
            'learning_rate': learning_rate,
        },
    }, final_path / "query_aware_weights.pt")
    
    model.importance_head.save_pretrained(final_path / "importance_head")
    
    # Save metrics
    df = pd.DataFrame(metrics_history)
    df.to_csv(output_path / "metrics.csv", index=False)
    
    print(f"\n[phase4] Training complete!")
    print(f"[phase4] Final checkpoint: {final_path}")
    print(f"[phase4] Metrics saved: {output_path / 'metrics.csv'}")
    
    # Print final metrics
    final_metrics = metrics_history[-1]
    print(f"\n[phase4] Final metrics:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Contrastive loss: {final_metrics['contrastive_loss']:.4f}")
    print(f"  Contrastive accuracy: {final_metrics['contrastive_acc']:.3f}")
    
    return model, metrics_history


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Query-Aware Training")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model name",
    )
    parser.add_argument(
        "--ert_checkpoint",
        type=str,
        default="checkpoints/ert_local_full_10k",
        help="ERT checkpoint path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/phase4_pilot",
        help="Output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="narrativeqa",
        help="Dataset: 'narrativeqa' (synthetic), 'msmarco' (real), or auto-detect",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with pre-downloaded dataset (e.g., data/msmarco_quick)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Training steps (500=pilot, 5000=full)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Warmup steps",
    )
    parser.add_argument(
        "--align_weight",
        type=float,
        default=0.1,
        help="Alignment loss weight",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Contrastive loss temperature",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset size (for testing)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (for RTX 5070)",
    )
    
    args = parser.parse_args()
    
    train_phase4(
        model_name=args.model,
        ert_checkpoint=args.ert_checkpoint,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        num_steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        align_weight=args.align_weight,
        temperature=args.temperature,
        max_samples=args.max_samples,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()

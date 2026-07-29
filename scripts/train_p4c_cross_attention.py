#!/usr/bin/env python3
"""
P4-C Training: CrossAttentionImportanceScorer on Synthetic LITM Data

Trains cross-attention importance scorer using:
1. Mistral-7B-v0.3 (4-bit quantized) to extract hidden states
2. LITM benchmark for synthetic data generation
3. Distance-based importance labels (answer tokens = 1.0, context = 0.3)
4. CrossAttentionImportanceScorer with multi-head attention

Usage:
    python scripts/train_p4c_cross_attention.py \\
        --n-samples 500 \\
        --epochs 5 \\
        --batch-size 2 \\
        --learning-rate 5e-5 \\
        --output-dir checkpoints/p4c_cross_attention/
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

_HERE = Path(__file__).parent.absolute()
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from token_importance.model.cross_attention_scorer import CrossAttentionImportanceScorer
from token_importance.eval.benchmarks import LostInMiddleBenchmark


# BCE loss for probabilities in [0, 1]
class ImportanceWeightedCrossEntropyLoss(nn.Module):
    """Corrected version that uses BCE loss for probabilities in [0, 1]."""
    
    def __init__(self, weight_answer: float = 1.0, weight_context: float = 0.5):
        super().__init__()
        self.weight_answer = weight_answer
        self.weight_context = weight_context
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            probs: [B, T] probabilities in [0, 1] from model
            labels: [B, T] binary labels (1=answer, 0=context)
            mask: [B, T] optional mask
        """
        # BCE loss for probabilities
        loss_per_token = self.bce_loss(probs.view(-1), labels.view(-1).float())
        loss_per_token = loss_per_token.view_as(labels)
        
        # Weight by importance
        weights = torch.where(
            labels == 1,
            torch.full_like(labels, self.weight_answer, dtype=torch.float32),
            torch.full_like(labels, self.weight_context, dtype=torch.float32)
        )
        
        if mask is not None:
            weights = weights * mask.float()
        
        weighted_loss = (loss_per_token * weights).sum() / weights.sum().clamp(min=1e-8)
        return weighted_loss


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train P4-C cross-attention scorer on LITM")
    p.add_argument("--n-samples", type=int, default=500, help="Number of training samples (default: 500)")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for RTX 5070)")
    p.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    p.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps for learning rate schedule")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    p.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N samples")
    p.add_argument("--output-dir", default="checkpoints/p4c_cross_attention", help="Output directory")
    p.add_argument("--device", default="", help="Device (cuda/cpu)")
    return p.parse_args(argv)


def generate_litm_samples(
    benchmark: LostInMiddleBenchmark,
    tokenizer,
    n_samples: int = 500,
    seed_offset: int = 0
) -> list:
    """Generate LITM samples with importance labels."""
    samples = []
    
    print(f"Generating {n_samples} LITM samples...")
    for i in tqdm(range(n_samples), desc="LITM generation"):
        try:
            # Get LITM sample: returns (input_ids, scores, target)
            input_ids, scores, target = benchmark._make_sample(
                tokenizer, n_pairs=20, query_idx=0, seed=seed_offset + i
            )
            
            seq_len = input_ids.shape[1]
            
            # Convert scores to normalized labels [0, 1]
            # scores is np.uint8 array: query tokens ~70, key-value tokens ~10
            labels = torch.tensor(scores, dtype=torch.float32) / 255.0
            
            samples.append({
                'input_ids': input_ids,  # [1, seq_len]
                'labels': labels,        # [seq_len]
                'target': target
            })
        except Exception as e:
            print(f"  ⚠ Sample {i} failed: {e}, skipping")
            continue
    
    print(f"Generated {len(samples)} valid samples")
    return samples


def load_model_and_tokenizer(device: str):
    """Load Mistral model and tokenizer."""
    print("[setup] Loading Mistral-7B-v0.3...")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        quantization_config=quantization_config,
        attn_implementation="eager",
        device_map=device,
    )
    
    model.eval()  # Freeze base model - we only train importance head
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"[setup] Model loaded on {device}")
    return model, tokenizer


def train_epoch(
    importance_scorer: nn.Module,
    base_model: nn.Module,
    samples: list,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device: str,
    batch_size: int = 2,
    output_dir: Path = None,
    save_every: int = 100,
    epoch: int = 0
) -> float:
    """Train for one epoch."""
    importance_scorer.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(range(0, len(samples), batch_size), desc="Training")
    for batch_idx in pbar:
        batch_samples = samples[batch_idx:batch_idx+batch_size]
        
        # Skip if batch is empty
        if not batch_samples:
            continue
        
        try:
            # Get max sequence length in batch
            max_seq_len = max(s['input_ids'].shape[1] for s in batch_samples)
            
            # Pad and stack
            batch_input_ids = []
            batch_labels = []
            for sample in batch_samples:
                ids = sample['input_ids'][0]  # [seq_len]
                labels = sample['labels']
                
                # Pad to max_seq_len
                if ids.shape[0] < max_seq_len:
                    pad_len = max_seq_len - ids.shape[0]
                    ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
                    labels = torch.cat([labels, torch.zeros(pad_len)])
                
                batch_input_ids.append(ids)
                batch_labels.append(labels)
            
            batch_input_ids = torch.stack(batch_input_ids).to(device)  # [B, T]
            batch_labels = torch.stack(batch_labels).to(device)  # [B, T]
            
            # Forward through base model to get hidden states
            with torch.no_grad():
                outputs = base_model(
                    batch_input_ids,
                    output_hidden_states=True,
                    use_cache=False
                )
                # Get last hidden state: [B, T, D] in bfloat16
                hidden_states = outputs.hidden_states[-1].float()  # Cast to float32
            
            # Extract query and context hidden states
            # Query: last 64 tokens
            query_start = max(0, hidden_states.shape[1] - 64)
            query_hidden = hidden_states[:, query_start:, :]  # [B, Q, D]
            context_hidden = hidden_states  # [B, T, D]
            
            # Forward through cross-attention scorer
            optimizer.zero_grad()
            importance_scores, _ = importance_scorer(
                query_hidden,  # Query region: [B, Q, D]
                context_hidden,  # Full context: [B, T, D]
                return_logits=True  # Get [0, 1] range for BCE loss
            )  # [B, T]
            
            # Compute loss
            loss = loss_fn(importance_scores, batch_labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Periodic checkpoint saving
            if output_dir and save_every > 0 and batch_idx > 0 and batch_idx % save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_e{epoch}_s{batch_idx}.pt"
                torch.save(importance_scorer.state_dict(), checkpoint_path)
                print(f"\n  ✓ Saved periodic checkpoint: {checkpoint_path}")
            
        except Exception as e:
            print(f"  ⚠ Batch processing failed: {e}, skipping")
            continue
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def train_p4c(args):
    """Main training pipeline."""
    print("=" * 80)
    print("TRACK C: P4-C Cross-Attention Importance Scorer Training")
    print("=" * 80)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[config] Device: {device}")
    print(f"[config] Output: {output_dir}")
    print(f"[config] N-samples: {args.n_samples}")
    print(f"[config] Epochs: {args.epochs}")
    print(f"[config] Batch size: {args.batch_size}")
    print(f"[config] Learning rate: {args.learning_rate}")
    print(f"[config] Warmup steps: {args.warmup_steps}")
    print(f"[config] Weight decay: {args.weight_decay}")
    print(f"[config] Save every: {args.save_every} samples")
    print()
    
    # Load models
    base_model, tokenizer = load_model_and_tokenizer(device)
    
    # Initialize cross-attention scorer
    print("[setup] Initializing CrossAttentionImportanceScorer...")
    importance_scorer = CrossAttentionImportanceScorer(
        hidden_dim=4096,
        projection_dim=256,
        num_heads=8,
        dropout=0.1
    ).to(device)
    print(f"[setup] Importance scorer: {sum(p.numel() for p in importance_scorer.parameters()):,} params")
    
    # Loss and optimizer
    loss_fn = ImportanceWeightedCrossEntropyLoss()
    optimizer = optim.AdamW(
        importance_scorer.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Generate training data
    benchmark = LostInMiddleBenchmark(n_samples=args.n_samples)
    train_samples = generate_litm_samples(
        benchmark, tokenizer, 
        n_samples=args.n_samples
    )
    
    if not train_samples:
        print("✗ No training samples generated, aborting")
        return
    
    print()
    print(f"[train] Training cross-attention scorer on {len(train_samples)} samples...")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        avg_loss = train_epoch(
            importance_scorer, base_model, train_samples,
            optimizer, scheduler, loss_fn, device, args.batch_size,
            output_dir=output_dir,
            save_every=args.save_every,
            epoch=epoch
        )
        
        print(f"  Average loss: {avg_loss:.6f}")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best.pt"
            torch.save(importance_scorer.state_dict(), best_path)
            print(f"  ✓ Saved best checkpoint: {best_path}")
    
    # Save final checkpoint
    final_path = output_dir / "final.pt"
    torch.save(importance_scorer.state_dict(), final_path)
    print(f"\n✓ Saved final checkpoint: {final_path}")
    
    # Save training summary
    summary = {
        "model": "CrossAttentionImportanceScorer",
        "base_model": "mistralai/Mistral-7B-v0.3",
        "n_samples": len(train_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_loss": float(best_loss),
        "checkpoint_best": str(best_path),
        "checkpoint_final": str(final_path),
    }
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved: {summary_path}")
    print("\n" + "=" * 80)
    print("✓ P4-C Training Complete")
    print("=" * 80)


if __name__ == "__main__":
    args = _parse_args()
    train_p4c(args)

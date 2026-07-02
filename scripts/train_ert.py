"""
ERT Training Script — Eviction Robustness Training for Importance Scoring.

Trains ImportanceScoringHead to predict importance scores such that
KV cache eviction (based on those scores) minimizes divergence from full-cache output.

Loss function: KL(logits_full || logits_evicted) + 0.1 * alignment_loss

Device Support:
  - NVIDIA CUDA: RTX 5070 (8GB), A100 (80GB), H100, etc.
  - CPU: Fallback (very slow, not recommended for training)

Optimized for limited GPU memory through 4-bit quantization and gradient accumulation.
"""

import os
import json
import argparse
import logging
import csv
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import is_bitsandbytes_available

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.token_importance.model.importance_scoring_head import (
    create_importance_head_with_lora,
    get_trainable_params_count,
)


# ============================================================================
# Device Detection
# ============================================================================

def detect_device(requested_device: str = "cuda") -> Tuple[str, str]:
    """
    Auto-detect available GPU device (NVIDIA CUDA).
    
    Returns:
        Tuple of (device_string, device_type) where device_type is 'cuda'
    """
    if not torch.cuda.is_available():
        if requested_device != "cpu":
            print("⚠ GPU not available, falling back to CPU (slow!)")
        return ("cpu", "cpu")
    
    device_name = torch.cuda.get_device_name(0)
    device_type = "cuda"  # NVIDIA CUDA device
    print(f"✓ GPU detected: {device_name}")
    
    return ("cuda", device_type)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ERTConfig:
    """ERT training configuration."""
    
    # Model
    target_model: str = "mistralai/Mistral-7B-v0.3"
    d_model: int = 4096
    
    # Training
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_steps: int = 1000
    
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # Loss weights
    kl_weight: float = 1.0
    alignment_weight: float = 0.1
    
    # Cache eviction
    cache_budget: float = 0.5  # Keep 50% of tokens
    
    # Device
    device: str = "cuda"
    device_type: str = "cuda"  # "cuda" for NVIDIA (auto-detected)
    dtype: str = "float32"
    use_gradient_checkpoint: bool = True
    
    # Data
    data_frac: float = 1.0
    num_workers: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    # Create formatter with more detail
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = [file_handler, console_handler]
    
    return logger


# ============================================================================
# Data Loading (Placeholder)
# ============================================================================

class DummyDataset(Dataset):
    """Dummy dataset for testing. Replace with real data."""
    
    def __init__(self, size: int = 100, seq_len: int = 1024):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 32000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len),
        }


def create_dataloader(
    config: ERTConfig,
    size: int = 1000,
) -> DataLoader:
    """Create data loader."""
    dataset = DummyDataset(
        size=int(size * config.data_frac),
        seq_len=1024,
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )


# ============================================================================
# Model Loading
# ============================================================================

class MockCausalLMForTest(nn.Module):
    """Mock model for testing (avoids loading real model)."""
    
    def __init__(self, d_model: int = 4096, vocab_size: int = 32000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        # Mock forward pass
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        
        class Output:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states
        
        if output_hidden_states:
            return Output(logits, [None, None, None, x])
        return Output(logits, None)


def load_model_4bit(
    model_name: str,
    device: str = "cuda",
    mock: bool = False,
) -> nn.Module:
    """Load model with 4-bit quantization (or mock for testing)."""
    if mock:
        print(f"[TEST MODE] Using mock model instead of {model_name}")
        return MockCausalLMForTest().to(device)
    
    if not is_bitsandbytes_available():
        print("Warning: bitsandbytes not available, using float32")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
        )
        return model
    
    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    )
    
    print(f"Loading {model_name} with 4-bit NF4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device,
        attn_implementation="eager",  # Avoid attention bugs
    )
    
    return model


# ============================================================================
# Training
# ============================================================================

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def train_step(
    batch: Dict,
    target_model: nn.Module,
    importance_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ERTConfig,
    step: int,
) -> Dict[str, float]:
    """Single training step."""
    device = config.device
    step_start_time = time.time()
    
    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    batch_size, seq_len = input_ids.shape
    
    # ===== Forward pass: Full cache =====
    with torch.no_grad():
        outputs_full = target_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits_full = outputs_full.logits  # [B, T, V]
        hidden_states = outputs_full.hidden_states[-1]  # [B, T, D]
    
    # ===== Predict importance scores =====
    # Cast hidden states to float32 for importance head
    hidden_states_fp32 = hidden_states.float()
    importance_deltas = importance_head(hidden_states_fp32, attention_mask)  # [B, T, 1]
    importance_deltas = importance_deltas.squeeze(-1)  # [B, T]
    
    # Initialize from oracle (in real training, use actual oracle scores)
    oracle_scores = torch.ones_like(importance_deltas) * 50.0  # Default: 50%
    predicted_scores = torch.clamp(oracle_scores + importance_deltas, 0, 100)
    
    # ===== Evict low-importance tokens =====
    n_keep = max(1, int(seq_len * config.cache_budget))
    
    # Get top-k by importance
    _, keep_indices = torch.topk(predicted_scores, n_keep, dim=-1)
    keep_indices_sorted, _ = torch.sort(keep_indices, dim=-1)
    
    # Create mask for kept tokens
    keep_mask = torch.zeros_like(predicted_scores)
    keep_mask.scatter_(1, keep_indices_sorted, 1.0)
    
    # Evicted mask
    evict_mask = 1.0 - keep_mask
    
    # Create evicted attention mask: 0 where evicted, 1 where kept
    evicted_attention_mask = keep_mask.float()
    
    # ===== Forward pass: Evicted cache =====
    with torch.no_grad():
        outputs_evicted = target_model(
            input_ids,
            attention_mask=evicted_attention_mask,
            output_hidden_states=False,
        )
        logits_evicted = outputs_evicted.logits  # [B, T, V]
    
    # ===== Loss: KL divergence =====
    # Only compute on kept tokens (where eviction happened)
    log_probs_full = torch.log_softmax(logits_full, dim=-1)  # [B, T, V]
    probs_evicted = torch.softmax(logits_evicted, dim=-1)  # [B, T, V]
    
    # KL(full || evicted) = sum_v p_full * (log p_full - log p_evicted)
    kl_loss = torch.sum(
        torch.exp(log_probs_full) * (log_probs_full - torch.log(probs_evicted + 1e-10)),
        dim=-1
    )  # [B, T]
    
    # Mask to only count kept positions
    kl_loss = (kl_loss * keep_mask).sum() / (keep_mask.sum() + 1e-10)
    
    # ===== Loss: Alignment with oracle scores =====
    alignment_loss = torch.mean((predicted_scores - oracle_scores) ** 2)
    
    # ===== Total loss =====
    loss = config.kl_weight * kl_loss + config.alignment_weight * alignment_loss
    
    # ===== Backward pass =====
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(importance_head.parameters(), 1.0)
    
    # Record timing and memory
    step_time = time.time() - step_start_time
    gpu_mem = get_gpu_memory_mb()
    
    # Return metrics
    return {
        "loss": loss.item(),
        "kl_loss": kl_loss.item(),
        "alignment_loss": alignment_loss.item(),
        "kept_tokens": keep_mask.sum().item(),
        "evicted_tokens": evict_mask.sum().item(),
        "step_time": step_time,
        "gpu_mem_mb": gpu_mem,
        "lr": optimizer.param_groups[0]["lr"],
    }


def train(
    config: ERTConfig,
    output_dir: Path,
    num_steps: int = 1000,
    mock: bool = False,
):
    """Main training loop."""
    
    # Create output directory first
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("ERT Training — Eviction Robustness Training")
    logger.info("=" * 80)
    logger.info(f"Config: {config.to_dict()}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total steps: {num_steps}")
    
    device = config.device
    
    # ===== Load Models =====
    logger.info("Loading target model...")
    target_model = load_model_4bit(config.target_model, device=device, mock=mock)
    target_model.eval()  # Evaluation mode, no gradient updates
    
    logger.info("Creating importance scoring head...")
    importance_head = create_importance_head_with_lora(
        d_model=config.d_model,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    ).to(device)
    
    trainable, total = get_trainable_params_count(importance_head)
    logger.info(f"Importance head: {trainable:,} trainable / {total:,} total params")
    logger.info(f"  Trainable ratio: {100*trainable/total:.1f}%")
    
    # ===== Optimizer & Scheduler =====
    optimizer = AdamW(
        importance_head.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    
    # Warmup + cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config.warmup_steps,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_steps - config.warmup_steps,
        eta_min=0.0,
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup_scheduler, main_scheduler],
        milestones=[config.warmup_steps],
    )
    
    # ===== Data Loading =====
    logger.info("Creating data loader...")
    dataloader = create_dataloader(config, size=num_steps * config.batch_size * 2)
    
    # ===== Training Loop =====
    logger.info(f"Starting training for {num_steps} steps...\n")
    
    # CSV metrics file for detailed analysis
    metrics_csv = output_dir / "metrics.csv"
    csv_columns = [
        "step", "loss", "kl_loss", "alignment_loss", "kept_tokens", "evicted_tokens",
        "step_time_sec", "gpu_mem_mb", "lr", "tokens_per_sec"
    ]
    csv_file = open(metrics_csv, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()
    
    importance_head.train()
    step = 0
    training_start_time = time.time()
    
    try:
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break
                
                # Training step
                metrics = train_step(
                    batch,
                    target_model,
                    importance_head,
                    optimizer,
                    config,
                    step,
                )
                
                # Accumulate gradients
                if (step + 1) % config.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                step += 1
                
                # Compute throughput
                batch_size = config.batch_size
                seq_len = batch["input_ids"].shape[1]
                tokens_per_sec = (batch_size * seq_len) / metrics["step_time"]
                
                # Log to CSV
                csv_row = {
                    "step": step,
                    "loss": f"{metrics['loss']:.6f}",
                    "kl_loss": f"{metrics['kl_loss']:.6f}",
                    "alignment_loss": f"{metrics['alignment_loss']:.6f}",
                    "kept_tokens": int(metrics["kept_tokens"]),
                    "evicted_tokens": int(metrics["evicted_tokens"]),
                    "step_time_sec": f"{metrics['step_time']:.3f}",
                    "gpu_mem_mb": f"{metrics['gpu_mem_mb']:.0f}",
                    "lr": f"{metrics['lr']:.2e}",
                    "tokens_per_sec": f"{tokens_per_sec:.0f}",
                }
                csv_writer.writerow(csv_row)
                csv_file.flush()
                
                # Console logging - more frequent and detailed
                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - training_start_time
                    eta_sec = (elapsed / step) * (num_steps - step) if step > 0 else 0
                    logger.info(
                        f"[Step {step:5d}/{num_steps}] Loss: {metrics['loss']:.4f} | "
                        f"KL: {metrics['kl_loss']:.4f} | Align: {metrics['alignment_loss']:.4f} | "
                        f"GPU: {metrics['gpu_mem_mb']:.0f}MB | "
                        f"Time: {metrics['step_time']:.3f}s ({tokens_per_sec:.0f} tok/s) | "
                        f"ETA: {eta_sec/60:.1f}min"
                    )
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        csv_file.close()
    
    # Calculate training statistics
    total_time = time.time() - training_start_time
    logger.info(f"\nTraining completed in {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    logger.info(f"Metrics saved to: {metrics_csv}")
    
    # ===== Save Checkpoint =====
    logger.info(f"\nSaving checkpoint to {output_dir}...")
    
    # Save head weights
    importance_head.save_pretrained(output_dir / "importance_head")
    logger.info(f"✓ Saved importance head to {output_dir}/importance_head/")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"✓ Saved config to {output_dir}/config.json")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info("=" * 80)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="ERT Training")
    
    # Mode
    parser.add_argument(
        "--mode",
        choices=["test", "validate", "full"],
        default="test",
        help="Training mode (test=1 batch, validate=1 epoch, full=many epochs)",
    )
    
    # Model
    parser.add_argument("--target-model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--d-model", type=int, default=4096)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    
    # Data
    parser.add_argument("--data-frac", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/ert_test")
    
    args = parser.parse_args()
    
    # Create config
    config = ERTConfig(
        target_model=args.target_model,
        d_model=args.d_model,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        device=args.device,
        dtype=args.dtype,
        data_frac=args.data_frac,
        num_workers=args.num_workers,
    )
    
    output_dir = Path(args.output_dir)
    
    # Auto-detect device (CUDA)
    device_str, device_type = detect_device(args.device)
    config.device_type = device_type
    
    # Determine number of steps based on mode
    if args.mode == "test":
        num_steps = 1
        print("Mode: TEST (1 step with REAL model)")
    elif args.mode == "validate":
        num_steps = 100
        print("Mode: VALIDATE (100 steps with REAL model)")
    else:  # full
        num_steps = 10000
        print("Mode: FULL (10,000 steps with REAL model)")
    
    # Train with real model (4-bit quantized)
    train(config, output_dir, num_steps=num_steps, mock=False)


if __name__ == "__main__":
    main()

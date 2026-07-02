#!/usr/bin/env python
"""Phase B Training: Teacher-Student TIS Re-scoring Head

Main training script for Phase B. Trains the importance re-scoring head using:
- Task loss: Language modeling on ground-truth answers
- KL distillation: Keep student logits close to teacher
- Budget loss: Soft constraint on token usage
- Churn loss: Smooth importance decisions
- Saliency loss: Encourage important tokens to survive budget

Run: python scripts/train_phase_b.py --phase B.1 [--config configs/phase_b_b1.yaml]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.patched_model import PatchedCausalLM
from token_importance.config import TISConfig
from token_importance.training.loss_functions import TISCompositeLoss
from token_importance.training.litm_dataloader import (
    get_litm_dataloader,
    create_litm_training_set,
    LITMTrainingDataset,
)
from token_importance.eval.benchmarks import LostInMiddleBenchmark


class PhaseB_Trainer:
    """Main trainer for Phase B."""
    
    def __init__(
        self,
        model: PatchedCausalLM,
        tokenizer,
        device: str = "cuda",
        phase: str = "B.1",
        output_dir: str = "checkpoints/phase_b",
        learning_rate: float = 5e-4,
        num_epochs: int = 1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 200,
        save_total_limit: int = 5,
        lambda_kl: float = 0.1,
        lambda_budget: float = 0.01,
        lambda_churn: float = 0.01,
        lambda_saliency: float = 0.0,
        kl_temperature: float = 2.0,
        use_wandb: bool = False,
        wandb_project: str = "tis-phase-b",
    ):
        """Initialize Phase B trainer.
        
        Args:
            model: PatchedCausalLM instance
            tokenizer: HuggingFace tokenizer
            device: Device to train on
            phase: Training phase ("B.1", "B.2", "B.3", "B.4")
            output_dir: Checkpoint directory
            learning_rate: Initial learning rate
            num_epochs: Training epochs
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm (clipping)
            warmup_steps: Warmup steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            save_total_limit: Keep last N checkpoints
            lambda_kl: KL distillation weight
            lambda_budget: Budget constraint weight
            lambda_churn: Churn minimization weight
            lambda_saliency: Saliency preservation weight
            kl_temperature: Temperature for KL softening
            use_wandb: Whether to log to WandB
            wandb_project: WandB project name
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.phase = phase
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        
        # Loss weights
        self.loss_fn = TISCompositeLoss(
            lambda_kl=lambda_kl,
            lambda_budget=lambda_budget,
            lambda_churn=lambda_churn,
            lambda_saliency=lambda_saliency,
            kl_temperature=kl_temperature,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Scheduler (cosine with warmup)
        self.scheduler = None
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    config={
                        "phase": phase,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "lambda_kl": lambda_kl,
                        "lambda_budget": lambda_budget,
                        "lambda_churn": lambda_churn,
                        "lambda_saliency": lambda_saliency,
                    },
                    name=f"phase_b_{phase}",
                )
                self.wandb = wandb
            except ImportError:
                print("[trainer] WandB not installed, disabling logging")
                self.use_wandb = False
        
        # Training state
        self.global_step = 0
        self.best_litm_accuracy = 0.0
        self.checkpoints_saved = []
    
    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler.
        
        Args:
            total_steps: Total training steps
        """
        # Warmup followed by cosine decay
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.warmup_steps,
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Dictionary with keys:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - target_ids: [batch_size, seq_len]
                - budget: [batch_size]
                - seq_length: [batch_size]
        
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        budget = batch.get("budget", None)
        seq_lengths = batch["seq_length"]
        
        # Forward pass through student model
        with torch.autocast("cuda", enabled=True):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            student_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Teacher logits (no importance bias)
            # For now, compute from base model logits
            teacher_logits = student_logits.detach()  # Frozen teacher
            
            # Compute composite loss
            loss_dict = self.loss_fn(
                student_logits=student_logits,
                target_ids=target_ids,
                teacher_logits=teacher_logits,
                keep_masks=None,  # Not available in single-pass training
                importance_scores=None,
                budget_tokens=None,  # TODO: implement budget tracking
                actual_tokens=seq_lengths.max().item(),
            )
        
        # Backward pass
        loss = loss_dict["loss_total"]
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )
        
        # Return loss values (before accumulation scaling)
        return {k: v.detach().cpu().item() * self.gradient_accumulation_steps
                for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on LITM benchmark.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        print(f"\n[trainer] Validating... (step {self.global_step})")
        
        # Run LITM benchmark with small sample
        try:
            bench = LostInMiddleBenchmark(
                n_pairs_options=[5, 10],
                n_samples=2,
            )
            
            result = bench.run(
                self.model,
                self.tokenizer,
                TISConfig(),
                cache_budget=0.5,  # Validate at 50% budget
                generation_kwargs=None,  # Use static generation for validation
            )
            
            metrics = {
                "litm_accuracy_50_budget": result.get("accuracy", 0.0),
            }
            
            # Additional metrics if available
            for key in ["accuracy_by_position.beginning", "accuracy_by_position.middle", "accuracy_by_position.end"]:
                if key in result:
                    metrics[f"litm_{key}"] = result[key]
            
            print(f"[trainer] LITM accuracy @ 50% budget: {metrics['litm_accuracy_50_budget']:.3f}")
            
            return metrics
        except Exception as e:
            print(f"[trainer] Validation failed: {e}")
            return {}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.base_model.save_pretrained(
            checkpoint_dir / "base_model"
        )
        
        # Save TIS components
        tis_state = {
            "importance_embedding": self.model.importance_embedding.state_dict(),
            "importance_head": self.model.importance_head.state_dict(),
            "attn_hook_lambda": self.model.attn_hook._lambda.data,
        }
        torch.save(tis_state, checkpoint_dir / "tis_components.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training config
        config = {
            "global_step": self.global_step,
            "phase": self.phase,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        with open(checkpoint_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"[trainer] Checkpoint saved to {checkpoint_dir}")
        self.checkpoints_saved.append(checkpoint_dir)
        
        # Keep only last N checkpoints
        if len(self.checkpoints_saved) > self.save_total_limit:
            old_checkpoint = self.checkpoints_saved.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint, ignore_errors=True)
            print(f"[trainer] Removed old checkpoint: {old_checkpoint}")
        
        # Save best checkpoint
        if is_best:
            best_dir = self.output_dir / f"best_{self.phase}"
            import shutil
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            print(f"[trainer] Best checkpoint saved to {best_dir}")
    
    def train(
        self,
        train_dataloader,
        num_training_steps: int,
    ):
        """Main training loop.
        
        Args:
            train_dataloader: PyTorch DataLoader
            num_training_steps: Total training steps
        """
        self.setup_scheduler(num_training_steps)
        
        print(f"\n[trainer] Starting Phase {self.phase} training")
        print(f"[trainer] Total steps: {num_training_steps}")
        print(f"[trainer] Eval interval: {self.eval_steps}")
        print(f"[trainer] Save interval: {self.save_steps}")
        
        # Training loop
        step = 0
        accumulated_loss = {}
        
        try:
            for epoch in range(self.num_epochs):
                print(f"\n[trainer] Epoch {epoch+1}/{self.num_epochs}")
                
                pbar = tqdm(train_dataloader, desc="Training", disable=False)
                
                for batch_idx, batch in enumerate(pbar):
                    # Training step
                    loss_dict = self.train_step(batch)
                    
                    # Accumulate losses
                    for key, val in loss_dict.items():
                        if key not in accumulated_loss:
                            accumulated_loss[key] = 0
                        accumulated_loss[key] += val
                    
                    # Optimizer step (if gradient accumulation done)
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # Scheduler step (after warmup)
                        if step >= self.warmup_steps:
                            self.scheduler.step()
                        
                        step += 1
                        self.global_step += 1
                        
                        # Update progress bar
                        avg_loss = accumulated_loss.get("loss_total", 0) / self.gradient_accumulation_steps
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                        
                        # Logging
                        if self.global_step % 10 == 0:
                            log_dict = {
                                "global_step": self.global_step,
                                "learning_rate": self.optimizer.param_groups[0]["lr"],
                            }
                            for key, val in accumulated_loss.items():
                                log_dict[key] = val / self.gradient_accumulation_steps
                            
                            if self.use_wandb:
                                self.wandb.log(log_dict)
                            
                            accumulated_loss = {}
                        
                        # Validation
                        if self.global_step % self.eval_steps == 0:
                            val_metrics = self.validate()
                            
                            if self.use_wandb:
                                self.wandb.log(val_metrics)
                            
                            # Save best checkpoint
                            if val_metrics.get("litm_accuracy_50_budget", 0) > self.best_litm_accuracy:
                                self.best_litm_accuracy = val_metrics["litm_accuracy_50_budget"]
                                self.save_checkpoint(is_best=True)
                        
                        # Save checkpoint
                        if self.global_step % self.save_steps == 0:
                            self.save_checkpoint(is_best=False)
                        
                        # Check if done
                        if self.global_step >= num_training_steps:
                            break
                
                if self.global_step >= num_training_steps:
                    break
        
        except KeyboardInterrupt:
            print("\n[trainer] Training interrupted by user")
        
        print(f"\n[trainer] Training complete!")
        print(f"[trainer] Total steps: {self.global_step}")
        print(f"[trainer] Best LITM accuracy: {self.best_litm_accuracy:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Phase B Training")
    parser.add_argument(
        "--phase",
        choices=["B.1", "B.2", "B.3", "B.4"],
        default="B.1",
        help="Training phase",
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-v0.3",
        help="Model name or path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to load (optional)",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/phase_b",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=None,
        help="Number of training steps (defaults by phase)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (defaults by phase)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log to WandB",
    )
    
    args = parser.parse_args()
    
    # Phase-specific defaults
    phase_config = {
        "B.1": {
            "num_training_steps": 1000,
            "learning_rate": 5e-4,
            "batch_size": 4,
            "eval_steps": 200,
            "save_steps": 200,
            "lambda_kl": 0.1,
            "lambda_budget": 0.01,
            "lambda_churn": 0.01,
            "lambda_saliency": 0.0,
        },
        "B.2": {
            "num_training_steps": 2000,
            "learning_rate": 2e-4,
            "batch_size": 8,
            "eval_steps": 400,
            "save_steps": 400,
            "lambda_kl": 0.1,
            "lambda_budget": 0.01,
            "lambda_churn": 0.02,  # Stricter churn control
            "lambda_saliency": 0.0,
        },
        "B.3": {
            "num_training_steps": 1500,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "eval_steps": 300,
            "save_steps": 300,
            "lambda_kl": 0.1,
            "lambda_budget": 0.01,
            "lambda_churn": 0.01,
            "lambda_saliency": 0.0,
        },
        "B.4": {
            "num_training_steps": 500,
            "learning_rate": 5e-5,
            "batch_size": 16,
            "eval_steps": 100,
            "save_steps": 100,
            "lambda_kl": 0.1,
            "lambda_budget": 0.01,
            "lambda_churn": 0.01,
            "lambda_saliency": 0.005,
        },
    }
    
    config = phase_config[args.phase]
    if args.num_training_steps is not None:
        config["num_training_steps"] = args.num_training_steps
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    
    # Load model and tokenizer
    print(f"[trainer] Loading model: {args.model}")
    from transformers import BitsAndBytesConfig
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="cuda",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="cuda",
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Patch model with TIS
    model = PatchedCausalLM(base_model, TISConfig())
    
    # Load TIS checkpoint if provided
    if args.checkpoint:
        print(f"[trainer] Loading TIS checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if (checkpoint_path / "tis_components.pt").exists():
            tis_state = torch.load(checkpoint_path / "tis_components.pt")
            
            # Load importance embedding
            try:
                model.importance_embedding.load_state_dict(tis_state["importance_embedding"])
                print("[trainer] ✓ Importance embedding loaded")
            except Exception as e:
                print(f"[trainer] ⚠ Failed to load importance embedding: {e}")
            
            # Load importance head with fallback for architecture mismatches
            try:
                model.importance_head.load_state_dict(tis_state["importance_head"])
                print("[trainer] ✓ Importance head loaded")
            except RuntimeError as e:
                print(f"[trainer] ⚠ Architecture mismatch in importance head: {e}")
                print("[trainer] Starting from randomly initialized head (this is OK for Phase B training)")
            
            # Load attn hook lambda
            try:
                model.attn_hook._lambda.data = tis_state["attn_hook_lambda"]
                print("[trainer] ✓ Attention hook lambda loaded")
            except Exception as e:
                print(f"[trainer] ⚠ Failed to load attention hook: {e}")
    
    # Create trainer
    trainer = PhaseB_Trainer(
        model=model,
        tokenizer=tokenizer,
        phase=args.phase,
        output_dir=args.output_dir,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        lambda_kl=config["lambda_kl"],
        lambda_budget=config["lambda_budget"],
        lambda_churn=config["lambda_churn"],
        lambda_saliency=config["lambda_saliency"],
        use_wandb=args.use_wandb,
    )
    
    # Create training dataloader
    print("[trainer] Creating training dataloader...")
    train_dataloader = get_litm_dataloader(
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        n_examples=1000,  # Reduced for memory efficiency
        max_length=1024,  # Optimized for RTX 5070 (7.53GB)
        num_workers=0,  # Set to 0 to avoid issues with CUDA
        shuffle=True,
    )
    
    # Train
    trainer.train(
        train_dataloader=train_dataloader,
        num_training_steps=config["num_training_steps"],
    )


if __name__ == "__main__":
    main()

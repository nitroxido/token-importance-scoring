#!/usr/bin/env python3
"""
Supervised Fine-tuning: Train importance head directly on answer token labels.

Instead of contrastive loss (query-document matching), we use supervised MSE loss
where labels are: 1 if token in answer, 0 otherwise.

This directly optimizes what LITM tests.
"""

import argparse
from pathlib import Path
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.importance_scoring_head import create_importance_head_with_lora


class SupervisedAnswerDataset(Dataset):
    """Dataset for supervised learning with answer token labels."""
    
    def __init__(self, tokenizer, num_samples=50):
        """Create synthetic supervised training data."""
        self.tokenizer = tokenizer
        self.samples = []
        
        # Create training samples
        documents_and_answers = [
            ("France is a country in Western Europe. Paris is the capital. It has culture and art.",
             "Paris"),
            ("The Eiffel Tower is located in Paris. It is made of iron.",
             "Eiffel Tower"),
            ("World War II ended in 1945. It was a global conflict.",
             "1945"),
            ("Tokyo is the capital of Japan. It is a large metropolitan area.",
             "Tokyo"),
            ("The Amazon River is in South America. It is very long.",
             "Amazon"),
            ("Einstein developed the theory of relativity. He won the Nobel Prize.",
             "Einstein"),
            ("The Great Wall of China is very long. It was built over centuries.",
             "Great Wall"),
            ("The Titanic sank in 1912. It was a famous ship.",
             "Titanic"),
            ("Python is a programming language. It is widely used.",
             "Python"),
            ("The Statue of Liberty is in New York. It represents freedom.",
             "Statue of Liberty"),
        ]
        
        # Repeat to get desired number of samples
        for i in range(num_samples):
            doc, answer = documents_and_answers[i % len(documents_and_answers)]
            self.samples.append((doc, answer))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        doc, answer = self.samples[idx]
        
        # Tokenize document
        doc_inputs = self.tokenizer(
            doc,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = doc_inputs["input_ids"].squeeze(0)
        
        # Create labels: 1 if token in answer, 0 otherwise
        answer_token_ids = set(self.tokenizer.encode(answer, add_special_tokens=False))
        answer_token_strs = set(self.tokenizer.decode([tid]) for tid in answer_token_ids)
        
        labels = torch.zeros(len(input_ids), dtype=torch.float32)
        
        # Mark tokens that are in the answer
        for i, token_id in enumerate(input_ids):
            token_str = self.tokenizer.decode([token_id.item()])
            # Fuzzy match for subword tokens
            for answer_token in answer_token_strs:
                if answer_token.lower() in token_str.lower():
                    labels[i] = 1.0
                    break
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_supervised_batch(batch):
    """Collate batch with padding."""
    max_len = max(len(b["input_ids"]) for b in batch)
    
    input_ids_list = []
    labels_list = []
    masks_list = []
    
    for b in batch:
        input_ids = b["input_ids"]
        labels = b["labels"]
        
        # Pad
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
            labels = torch.cat([labels, torch.zeros(pad_len, dtype=labels.dtype)])
        
        # Attention mask
        mask = torch.ones(len(b["input_ids"]), dtype=torch.float32)
        if pad_len > 0:
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        masks_list.append(mask)
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(masks_list),
    }


def load_ert_checkpoint(model_name, checkpoint_path, device="cuda", load_in_4bit=True):
    """Load ERT checkpoint."""
    print(f"[Supervised] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[Supervised] Loading base model: {model_name}")
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
        )
    
    print(f"[Supervised] Loading ERT importance head from {checkpoint_path}")
    checkpoint_dir = Path(checkpoint_path) / "importance_head"
    
    # Create importance head
    config = base_model.config
    importance_head = create_importance_head_with_lora(
        d_model=config.hidden_size,
        lora_rank=8,
        lora_alpha=16,
    )
    
    # Load LoRA weights
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        importance_head = PeftModel.from_pretrained(
            importance_head,
            str(checkpoint_dir),
        ).to(device)
    
    return base_model, importance_head, tokenizer


def train_supervised(
    model_name="mistralai/Mistral-7B-v0.3",
    ert_checkpoint="checkpoints/ert_local_full_10k",
    steps=100,
    batch_size=2,
    learning_rate=1e-4,
    output_dir="checkpoints/supervised_importance",
    load_in_4bit=True,
):
    """Train importance head with supervised MSE loss."""
    
    print("\n" + "="*80)
    print("SUPERVISED IMPORTANCE HEAD TRAINING")
    print("="*80)
    print(f"ERT Checkpoint: {ert_checkpoint}")
    print(f"Steps: {steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    device = "cuda"
    
    # Load model
    base_model, importance_head, tokenizer = load_ert_checkpoint(
        model_name, ert_checkpoint, device, load_in_4bit
    )
    
    # Create dataset
    print(f"\n[Supervised] Creating training dataset...")
    dataset = SupervisedAnswerDataset(tokenizer, num_samples=min(steps * batch_size, 500))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_supervised_batch,
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(importance_head.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n[Supervised] Starting training...")
    metrics = []
    step = 0
    
    importance_head.train()
    for epoch in range(1):  # Single epoch, iterate through step limit
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            if step >= steps:
                break
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = base_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]  # [B, T, 4096]
            
            # Cast to float32 for importance head
            hidden_states = hidden_states.to(torch.float32)
            
            # Get scores
            scores = importance_head(hidden_states)  # [B, T, 1]
            scores = torch.sigmoid(scores.squeeze(-1))  # [B, T]
            
            # MSE loss
            loss = nn.functional.mse_loss(scores, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            metrics.append({"step": step, "loss": loss.item()})
            
            loss_val = loss.item()
            pbar.set_postfix({"loss": f"{loss_val:.6f}"})
            step += 1
        
        if step >= steps:
            break
    
    print(f"\n[Supervised] Training complete!")
    print(f"  Final loss: {metrics[-1]['loss']:.6f}")
    print(f"  First loss: {metrics[0]['loss']:.6f}")
    improvement = (metrics[0]['loss'] - metrics[-1]['loss']) / metrics[0]['loss'] * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    # Save checkpoint
    output_path = Path(output_dir) / "final"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Supervised] Saving checkpoint to {output_path}...")
    importance_head.save_pretrained(output_path / "importance_head")
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_path / "metrics.csv", index=False)
    
    print(f"[Supervised] ✓ Checkpoint saved!")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Supervised importance head training")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Model name")
    parser.add_argument("--ert-checkpoint", default="checkpoints/ert_local_full_10k",
                       help="Path to ERT checkpoint")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/supervised_importance",
                       help="Output directory")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    checkpoint = train_supervised(
        model_name=args.model,
        ert_checkpoint=args.ert_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
    )
    
    print(f"\n✓ Training complete!")
    print(f"Next: Evaluate with: python scripts/eval.py --checkpoint {checkpoint}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

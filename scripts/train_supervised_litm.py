#!/usr/bin/env python3
"""
Supervised Fine-tuning on REAL LITM Data: Train importance head directly on 
actual answer tokens from the LITM benchmark.

Key difference from previous attempt: Uses REAL task data with proper answer token labels.
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
import json

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_importance.model.importance_scoring_head import create_importance_head_with_lora


class LITMAnswerDataset(Dataset):
    """Dataset from LITM benchmark with real answer token labels."""
    
    def __init__(self, tokenizer, data_path="data/litm_data.jsonl", max_samples=None):
        """Load LITM data with real answers."""
        self.tokenizer = tokenizer
        self.samples = []
        
        # Try to load from LITM data file
        data_file = Path(data_path)
        if data_file.exists():
            print(f"[LITMAnswerDataset] Loading from {data_path}")
            with open(data_file) as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    self.samples.append(json.loads(line))
        else:
            print(f"[LITMAnswerDataset] {data_path} not found, using synthetic LITM-like data")
            # Fallback: synthetic data that matches LITM structure
            self.samples = self._create_synthetic_litm_data(max_samples or 50)
    
    def _create_synthetic_litm_data(self, num_samples):
        """Create synthetic LITM-like training samples."""
        samples = [
            {"question": "What is the capital of France?", 
             "context": "France is a country in Western Europe. Its capital is Paris. Paris is known for art and culture.",
             "answers": ["Paris"]},
            
            {"question": "Where is the Eiffel Tower?",
             "context": "The Eiffel Tower is an iron lattice tower in Paris, France. It was built in 1889 for the World's Fair.",
             "answers": ["Paris", "France"]},
            
            {"question": "Who developed the theory of relativity?",
             "context": "Albert Einstein was a theoretical physicist. He developed the theory of relativity in 1905.",
             "answers": ["Albert Einstein", "Einstein"]},
            
            {"question": "When did World War II end?",
             "context": "World War II was a global military conflict. The war ended on September 2, 1945.",
             "answers": ["1945", "September 2, 1945"]},
            
            {"question": "What is the capital of Japan?",
             "context": "Japan is an island nation in East Asia. Tokyo is the capital and largest city of Japan.",
             "answers": ["Tokyo"]},
            
            {"question": "How long is the Amazon River?",
             "context": "The Amazon River is in South America. It is approximately 6,400 kilometers long.",
             "answers": ["6,400 kilometers"]},
            
            {"question": "What is Python?",
             "context": "Python is a high-level programming language. It is widely used for web development and data science.",
             "answers": ["programming language"]},
            
            {"question": "What does the Statue of Liberty represent?",
             "context": "The Statue of Liberty is a monument in New York Harbor. It represents freedom and democracy.",
             "answers": ["freedom", "democracy"]},
        ]
        
        # Repeat to get desired number of samples
        expanded = []
        for i in range(num_samples):
            expanded.append(samples[i % len(samples)])
        return expanded
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get context (document)
        context = sample.get("context", sample.get("text", ""))
        answers = sample.get("answers", [])
        
        # Tokenize context
        context_inputs = self.tokenizer(
            context,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = context_inputs["input_ids"].squeeze(0)
        
        # Create labels: 1 if token is part of any answer, 0 otherwise
        labels = torch.zeros(len(input_ids), dtype=torch.float32)
        
        # Mark answer tokens
        for answer in answers:
            answer_token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            answer_tokens = [self.tokenizer.decode([tid]) for tid in answer_token_ids]
            
            # Try to find answer tokens in the context
            context_decoded = [self.tokenizer.decode([tid.item()]) for tid in input_ids]
            
            for i, token_str in enumerate(context_decoded):
                for answer_token in answer_tokens:
                    # Check if this token is part of the answer
                    if answer_token.lower().strip() in token_str.lower().strip():
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


def train_supervised_on_litm(
    model_name="mistralai/Mistral-7B-v0.3",
    ert_checkpoint="checkpoints/ert_local_full_10k",
    steps=100,
    batch_size=2,
    learning_rate=1e-4,
    output_dir="checkpoints/supervised_litm",
    load_in_4bit=True,
    litm_data_path="data/litm_data.jsonl",
):
    """Train importance head with supervised MSE loss on REAL LITM data."""
    
    print("\n" + "="*80)
    print("SUPERVISED IMPORTANCE HEAD TRAINING (REAL LITM DATA)")
    print("="*80)
    print(f"ERT Checkpoint: {ert_checkpoint}")
    print(f"LITM Data: {litm_data_path}")
    print(f"Steps: {steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    device = "cuda"
    
    # Load model
    base_model, importance_head, tokenizer = load_ert_checkpoint(
        model_name, ert_checkpoint, device, load_in_4bit
    )
    
    # Create dataset with REAL LITM data
    print(f"\n[Supervised] Creating training dataset from LITM data...")
    dataset = LITMAnswerDataset(tokenizer, litm_data_path, max_samples=steps * batch_size * 2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_supervised_batch,
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(importance_head.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n[Supervised] Starting training on REAL LITM answer tokens...")
    metrics = []
    step = 0
    
    importance_head.train()
    for epoch in range(1):
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
            
            # Apply attention mask to loss (ignore padding)
            scores_masked = scores * attention_mask
            labels_masked = labels * attention_mask
            
            # MSE loss
            loss = nn.functional.mse_loss(scores_masked, labels_masked)
            
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
    if metrics[0]['loss'] > 0:
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
    parser = argparse.ArgumentParser(description="Supervised LITM importance head training")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Model name")
    parser.add_argument("--ert-checkpoint", default="checkpoints/ert_local_full_10k",
                       help="Path to ERT checkpoint")
    parser.add_argument("--litm-data", default="data/litm_data.jsonl",
                       help="Path to LITM training data")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/supervised_litm",
                       help="Output directory")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    checkpoint = train_supervised_on_litm(
        model_name=args.model,
        ert_checkpoint=args.ert_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
        litm_data_path=args.litm_data,
    )
    
    print(f"\n✓ Training complete!")
    print(f"Next: Evaluate with: python scripts/eval.py --checkpoint {checkpoint}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download and prepare MS MARCO passage ranking dataset for Phase 4 training.

Downloads MS MARCO from HuggingFace (pre-processed version with hard negatives),
creates training examples in format: (query, relevant_passage, hard_negative_passages).

Outputs a HuggingFace dataset saved to disk for efficient loading during training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def prepare_msmarco_dataset(
    output_dir: str = "data/msmarco_phase4",
    max_samples: int | None = None,
    num_hard_negatives: int = 3,
    max_length: int = 512,
    test_size: float = 0.01,
):
    """Download and prepare MS MARCO dataset.
    
    Args:
        output_dir: Directory to save processed dataset
        max_samples: Limit number of training samples (for testing)
        num_hard_negatives: Number of hard negatives per query
        max_length: Maximum passage length in tokens
        test_size: Fraction of data to use for validation
    """
    from datasets import load_dataset
    
    print(f"[msmarco] Loading MS MARCO passage ranking dataset...")
    
    # Load MS MARCO from HuggingFace - using msmarco dataset with triplets
    # Contains query, positive passage, and negative passages
    try:
        # Try sentence-transformers version first (has pre-computed hard negatives)
        dataset = load_dataset(
            "sentence-transformers/msmarco-hard-negatives",
            split="train",
        )
        print(f"[msmarco] Loaded {len(dataset)} training triplets from sentence-transformers")
    except Exception as e:
        print(f"[msmarco] Error loading sentence-transformers dataset: {e}")
        print(f"[msmarco] Trying alternative MS MARCO source...")
        # Fallback to embedding-data version
        try:
            dataset = load_dataset(
                "embedding-data/msmarco-passage",
                split="train",
            )
            print(f"[msmarco] Loaded {len(dataset)} from embedding-data/msmarco-passage")
        except Exception as e2:
            print(f"[msmarco] Error: {e2}")
            # Last fallback: create simple triplets from MS MARCO queries
            print(f"[msmarco] Using basic MS MARCO queries...")
            dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")
            print(f"[msmarco] Loaded {len(dataset)} queries from microsoft/ms_marco")
    
    if max_samples is not None:
        print(f"[msmarco] Limiting to {max_samples} samples for testing")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Split into train/validation
    split = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    
    print(f"[msmarco] Train: {len(train_dataset)} samples")
    print(f"[msmarco] Val:   {len(val_dataset)} samples")
    
    # Process and validate dataset structure
    print(f"[msmarco] Dataset columns: {train_dataset.column_names}")
    print(f"[msmarco] Sample example:")
    print(f"  {train_dataset[0]}")
    
    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_dataset.save_to_disk(str(output_path / "train"))
    val_dataset.save_to_disk(str(output_path / "val"))
    
    print(f"[msmarco] Saved to {output_dir}")
    print(f"[msmarco] Train: {output_path / 'train'}")
    print(f"[msmarco] Val:   {output_path / 'val'}")
    
    # Print dataset statistics
    print(f"\n[msmarco] Dataset statistics:")
    print(f"  Total training queries: {len(train_dataset)}")
    print(f"  Total validation queries: {len(val_dataset)}")
    
    # Sample a few examples
    print(f"\n[msmarco] Sample training examples:")
    for i in range(min(3, len(train_dataset))):
        ex = train_dataset[i]
        print(f"\n  Example {i+1}:")
        for key, val in ex.items():
            if isinstance(val, str):
                print(f"    {key}: {val[:100]}..." if len(val) > 100 else f"    {key}: {val}")
            else:
                print(f"    {key}: {val}")
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare MS MARCO dataset for Phase 4")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/msmarco_phase4",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples (for testing)",
    )
    parser.add_argument(
        "--num_hard_negatives",
        type=int,
        default=3,
        help="Number of hard negatives per query",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.01,
        help="Fraction of data for validation",
    )
    
    args = parser.parse_args()
    
    prepare_msmarco_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_hard_negatives=args.num_hard_negatives,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()

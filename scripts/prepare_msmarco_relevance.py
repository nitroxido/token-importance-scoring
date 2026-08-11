#!/usr/bin/env python3
"""Prepare MS-MARCO v1.1 passage ranking dataset for v2.2 supervised relevance training.

Creates deterministic train/tune/test splits with proper is_selected labels for
pairwise ranking training.

Outputs:
    data/msmarco_relevance/
        train.parquet   - Full MS-MARCO train split (~502K queries)
        tune.parquet    - 500 queries from dev set (for direction tuning)
        test.parquet    - 500 queries from dev set (for holdout evaluation)

Schema:
    - query_id: str
    - query: str
    - passages: list[str] (passage texts)
    - is_selected: list[int] (binary relevance labels, 0/1)
    - corpus_ids: list[str] (passage IDs for tracking)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


def prepare_msmarco_relevance_dataset(
    output_dir: str = "data/msmarco_relevance",
    tune_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
    use_local_cache: bool = True,
):
    """Download and prepare MS-MARCO v1.1 for supervised relevance training.
    
    Args:
        output_dir: Directory to save processed dataset
        tune_size: Number of queries for direction tuning (from dev set)
        test_size: Number of queries for holdout test (from dev set)
        seed: Random seed for reproducibility
        use_local_cache: Try loading from local data/msmarco_quick first
    """
    import numpy as np
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[msmarco-rel] Preparing MS-MARCO v1.1 for supervised relevance training")
    print(f"[msmarco-rel] Output: {output_dir}")
    print(f"[msmarco-rel] Splits: train (all), tune ({tune_size}), test ({test_size})")
    print()
    
    # ── Load train split ──────────────────────────────────────────────────────
    print(f"[train] Loading MS-MARCO train split...")
    
    # First try local cache (if exists)
    local_train_path = Path("data/msmarco_quick/train")
    if use_local_cache and local_train_path.exists():
        print(f"[train] Found local cache at {local_train_path}, loading...")
        from datasets import load_from_disk
        train_ds = load_from_disk(str(local_train_path))
        print(f"[train] Loaded {len(train_ds)} queries from local cache")
    else:
        # Download from HuggingFace
        print(f"[train] Downloading from HuggingFace (ms_marco v1.1)...")
        try:
            train_ds = load_dataset("ms_marco", "v1.1", split="train")
            print(f"[train] Downloaded {len(train_ds)} queries")
        except Exception as e:
            print(f"[train] Error loading ms_marco v1.1: {e}")
            print(f"[train] Trying alternative source (microsoft/ms_marco)...")
            train_ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")
            print(f"[train] Downloaded {len(train_ds)} queries")
    
    # ── Load dev split for tune/test ───────────────────────────────────────────
    print(f"\n[dev] Loading MS-MARCO dev split...")
    
    local_val_path = Path("data/msmarco_quick/val")
    if use_local_cache and local_val_path.exists():
        print(f"[dev] Found local cache at {local_val_path}, loading...")
        from datasets import load_from_disk
        dev_ds = load_from_disk(str(local_val_path))
        print(f"[dev] Loaded {len(dev_ds)} queries from local cache")
    else:
        print(f"[dev] Downloading from HuggingFace (ms_marco v1.1)...")
        try:
            # Use microsoft/ms_marco (same source as train)
            dev_ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
            print(f"[dev] Downloaded {len(dev_ds)} queries")
        except Exception as e:
            print(f"[dev] Error: {e}")
            print(f"[dev] Trying 'test' split instead of 'validation'...")
            try:
                dev_ds = load_dataset("microsoft/ms_marco", "v1.1", split="test")
                print(f"[dev] Downloaded {len(dev_ds)} queries (using test split)")
            except Exception as e2:
                print(f"[dev] Error: {e2}")
                # Last resort: use the already-loaded train_ds and split it
                print(f"[dev] Using tail of train split for dev set...")
                raise ValueError("Could not load dev/validation split")
    
    # ── Create deterministic splits ────────────────────────────────────────────
    print(f"\n[splits] Creating deterministic splits (seed={seed})...")
    
    # For reproducibility: always use first tune_size and next test_size from dev
    # (sorted by query_id to ensure stability across runs)
    if len(dev_ds) < (tune_size + test_size):
        print(f"[splits] WARNING: Dev set has only {len(dev_ds)} queries, " 
              f"requested {tune_size + test_size}")
        tune_size = min(tune_size, len(dev_ds) // 2)
        test_size = min(test_size, len(dev_ds) - tune_size)
        print(f"[splits] Adjusted to tune={tune_size}, test={test_size}")
    
    # Sort dev by query_id for deterministic splits
    dev_query_ids = [ex["query_id"] for ex in dev_ds]
    sorted_indices = sorted(range(len(dev_ds)), key=lambda i: dev_query_ids[i])
    
    tune_indices = sorted_indices[:tune_size]
    test_indices = sorted_indices[tune_size:tune_size + test_size]
    
    tune_ds = dev_ds.select(tune_indices)
    test_ds = dev_ds.select(test_indices)
    
    print(f"[splits] Train: {len(train_ds)} queries (full MS-MARCO train)")
    print(f"[splits] Tune:  {len(tune_ds)} queries (dev[0:{tune_size}], sorted by query_id)")
    print(f"[splits] Test:  {len(test_ds)} queries (dev[{tune_size}:{tune_size + test_size}], sorted by query_id)")
    
    # ── Validate no overlap ─────────────────────────────────────────────────────
    print(f"\n[validation] Checking for overlap...")
    train_qids = set(ex["query_id"] for ex in train_ds)
    tune_qids = set(ex["query_id"] for ex in tune_ds)
    test_qids = set(ex["query_id"] for ex in test_ds)
    
    overlap_train_tune = train_qids & tune_qids
    overlap_train_test = train_qids & test_qids
    overlap_tune_test = tune_qids & test_qids
    
    if overlap_train_tune:
        print(f"[validation] WARNING: {len(overlap_train_tune)} queries overlap train-tune")
    if overlap_train_test:
        print(f"[validation] WARNING: {len(overlap_train_test)} queries overlap train-test")
    if overlap_tune_test:
        print(f"[validation] ERROR: {len(overlap_tune_test)} queries overlap tune-test")
        raise ValueError("Tune and test sets must be disjoint!")
    
    if not overlap_train_tune and not overlap_train_test and not overlap_tune_test:
        print(f"[validation] ✓ No overlap detected (train/tune/test are disjoint)")
    
    # ── Process and save to Parquet ────────────────────────────────────────────
    print(f"\n[save] Processing and saving to Parquet...")
    
    def process_split(ds, split_name: str, output_file: Path):
        """Convert HF dataset to Parquet with proper schema."""
        print(f"[save] Processing {split_name} ({len(ds)} queries)...")
        
        # Check dataset structure
        sample = ds[0]
        print(f"[save] Sample {split_name} schema: {sample.keys()}")
        
        records = []
        skipped_no_positive = 0
        skipped_no_passages = 0
        
        for ex in tqdm(ds, desc=f"Processing {split_name}"):
            # Extract passages data
            passages_data = ex.get("passages", {})
            
            # Handle different schema formats
            if isinstance(passages_data, dict):
                # Format: {"passage_text": [...], "is_selected": [...]}
                passage_texts = passages_data.get("passage_text", [])
                is_selected = passages_data.get("is_selected", [0] * len(passage_texts))
                corpus_ids = passages_data.get("passage_id", 
                            [f"pid_{i}" for i in range(len(passage_texts))])
            elif isinstance(passages_data, list):
                # Format: [{"text": "...", "is_selected": 0}, ...]
                passage_texts = [p.get("text", p.get("passage_text", "")) 
                               for p in passages_data]
                is_selected = [p.get("is_selected", 0) for p in passages_data]
                corpus_ids = [p.get("passage_id", f"pid_{i}") 
                            for i, p in enumerate(passages_data)]
            else:
                print(f"[save] WARNING: Unexpected passages format for {ex['query_id']}")
                skipped_no_passages += 1
                continue
            
            if not passage_texts:
                skipped_no_passages += 1
                continue
            
            # For training: skip examples with no positive (if tune/test, keep them)
            if split_name == "train" and sum(is_selected) == 0:
                skipped_no_positive += 1
                continue
            
            records.append({
                "query_id": str(ex["query_id"]),  # Ensure string
                "query": str(ex["query"]),
                "passages": [str(p) for p in passage_texts],  # Ensure all strings
                "is_selected": [int(s) for s in is_selected],  # Ensure all ints
                "corpus_ids": [str(c) for c in corpus_ids],  # Ensure all strings
            })
        
        if skipped_no_positive:
            print(f"[save] Skipped {skipped_no_positive} {split_name} queries (no positive passage)")
        if skipped_no_passages:
            print(f"[save] Skipped {skipped_no_passages} {split_name} queries (no passages)")
        
        print(f"[save] {split_name}: {len(records)} valid queries")
        
        # Save as Parquet
        schema = pa.schema([
            ("query_id", pa.string()),
            ("query", pa.string()),
            ("passages", pa.list_(pa.string())),
            ("is_selected", pa.list_(pa.int32())),
            ("corpus_ids", pa.list_(pa.string())),
        ])
        
        table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, output_file)
        
        print(f"[save] Saved to {output_file}")
        print(f"[save] File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        return len(records)
    
    # Process each split
    train_count = process_split(train_ds, "train", output_path / "train.parquet")
    tune_count = process_split(tune_ds, "tune", output_path / "tune.parquet")
    test_count = process_split(test_ds, "test", output_path / "test.parquet")
    
    # ── Final summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MS-MARCO Relevance Dataset Preparation Complete")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"  train.parquet: {train_count} queries")
    print(f"  tune.parquet:  {tune_count} queries")
    print(f"  test.parquet:  {test_count} queries")
    print()
    print(f"Seed: {seed} (deterministic splits)")
    print(f"Schema: query_id, query, passages[], is_selected[], corpus_ids[]")
    print(f"{'='*70}")
    
    return train_count, tune_count, test_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MS-MARCO v1.1 for v2.2 supervised relevance training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/msmarco_relevance",
        help="Output directory (default: data/msmarco_relevance)",
    )
    parser.add_argument(
        "--tune-size",
        type=int,
        default=500,
        help="Number of queries for direction tuning (default: 500)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of queries for holdout test (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-local-cache",
        action="store_true",
        help="Skip local cache, download fresh from HuggingFace",
    )
    
    args = parser.parse_args()
    
    prepare_msmarco_relevance_dataset(
        output_dir=args.output_dir,
        tune_size=args.tune_size,
        test_size=args.test_size,
        seed=args.seed,
        use_local_cache=not args.no_local_cache,
    )


if __name__ == "__main__":
    main()

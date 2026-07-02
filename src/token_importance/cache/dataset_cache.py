"""Dataset caching utilities for TIS training and evaluation.

Implements cache-aware dataset loading to avoid redundant downloads.
Supports HuggingFace datasets and local data directories.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch


class DatasetCache:
    """Manages caching of HuggingFace datasets and training data.
    
    Features:
    - Automatic caching of downloaded datasets
    - Configurable cache directory
    - Cache metadata tracking
    - Manual cache management (clear, inspect, stats)
    """

    def __init__(self, cache_root: str | Path | None = None):
        """Initialize dataset cache.
        
        Args:
            cache_root: Root directory for dataset cache.
                       Defaults to ~/.cache/tis_datasets if None.
        """
        if cache_root is None:
            cache_root = Path.home() / ".cache" / "tis_datasets"
        else:
            cache_root = Path(cache_root)
        
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_root / "cache_manifest.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load or initialize cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def get_dataset_path(self, dataset_name: str, split: str = "train") -> Path:
        """Get the cache path for a dataset.
        
        Args:
            dataset_name: HF dataset ID (e.g., 'deepmind/narrativeqa')
            split: Dataset split (e.g., 'train', 'validation')
        
        Returns:
            Path to cached dataset directory
        """
        safe_name = dataset_name.replace("/", "_")
        dataset_path = self.cache_root / f"{safe_name}_{split}"
        return dataset_path

    def dataset_exists(self, dataset_name: str, split: str = "train") -> bool:
        """Check if a dataset is already cached.
        
        Args:
            dataset_name: HF dataset ID
            split: Dataset split
        
        Returns:
            True if dataset is cached and metadata is valid
        """
        path = self.get_dataset_path(dataset_name, split)
        if not path.exists():
            return False
        
        # Check metadata
        key = f"{dataset_name}:{split}"
        if key not in self._metadata:
            return False
        
        meta = self._metadata[key]
        return meta.get("cached", False)

    def cache_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        config: str | None = None,
        max_samples: Optional[int] = None,
    ):
        """Download and cache a HuggingFace dataset.
        
        Args:
            dataset_name: HF dataset ID
            split: Dataset split to download
            config: Optional dataset config (e.g., 'v2.1' for MS-MARCO)
            max_samples: Optional limit on number of samples
        
        Returns:
            Cached HuggingFace Dataset object
        """
        from datasets import load_dataset
        
        key = f"{dataset_name}:{split}"
        path = self.get_dataset_path(dataset_name, split)
        
        # Load from HF with cache_dir set
        print(f"[cache] Loading {dataset_name} ({split})...")
        if config:
            ds = load_dataset(
                dataset_name,
                config,
                split=split,
                trust_remote_code=True,
                cache_dir=str(self.cache_root / "hf_cache")
            )
        else:
            ds = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=True,
                cache_dir=str(self.cache_root / "hf_cache")
            )
        
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        # Save dataset metadata
        path.mkdir(parents=True, exist_ok=True)
        
        # Store metadata
        self._metadata[key] = {
            "dataset_name": dataset_name,
            "config": config,
            "split": split,
            "num_samples": len(ds),
            "cached": True,
            "path": str(path),
        }
        self._save_metadata()
        
        print(f"[cache] ✓ Cached {dataset_name}:{split} ({len(ds)} samples)")
        return ds

    def load_cached_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        config: str | None = None,
        max_samples: Optional[int] = None,
        auto_cache: bool = True,
    ):
        """Load dataset from cache, downloading if necessary.
        
        Args:
            dataset_name: HF dataset ID
            split: Dataset split
            config: Optional dataset config (e.g., 'v2.1' for MS-MARCO)
            max_samples: Optional limit on samples
            auto_cache: If True, download and cache if not already cached
        
        Returns:
            HuggingFace Dataset object
        """
        from datasets import load_dataset
        
        # Check if already cached
        if self.dataset_exists(dataset_name, split):
            print(f"[cache] ✓ Using cached {dataset_name}:{split}")
            if config:
                ds = load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    trust_remote_code=True,
                    cache_dir=str(self.cache_root / "hf_cache"),
                )
            else:
                ds = load_dataset(
                    dataset_name,
                    split=split,
                    trust_remote_code=True,
                    cache_dir=str(self.cache_root / "hf_cache"),
                )
        else:
            if not auto_cache:
                raise RuntimeError(
                    f"Dataset {dataset_name}:{split} not in cache "
                    f"({self.cache_root}). Set auto_cache=True to download."
                )
            ds = self.cache_dataset(dataset_name, split, config, max_samples)
        
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        return ds

    def cache_local_dataset(
        self,
        source_path: str | Path,
        dataset_key: str,
        description: str = "",
    ) -> None:
        """Register and cache a local dataset directory.
        
        Args:
            source_path: Path to local dataset
            dataset_key: Identifier for this dataset (e.g., 'msmarco_custom')
            description: Optional description
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        cache_path = self.cache_root / dataset_key
        
        # Copy dataset to cache
        if cache_path.exists():
            shutil.rmtree(cache_path)
        
        shutil.copytree(source_path, cache_path)
        
        # Record metadata
        self._metadata[dataset_key] = {
            "type": "local",
            "source": str(source_path),
            "cached": True,
            "path": str(cache_path),
            "description": description,
        }
        self._save_metadata()
        
        print(f"[cache] ✓ Cached local dataset: {dataset_key}")

    def get_cache_stats(self) -> dict:
        """Get statistics about cached datasets.
        
        Returns:
            Dictionary with cache info and statistics
        """
        total_size = 0
        num_datasets = 0
        datasets_info = {}
        
        for dataset_path in self.cache_root.rglob("*"):
            if dataset_path.is_file():
                total_size += dataset_path.stat().st_size
        
        for key, meta in self._metadata.items():
            if meta.get("cached"):
                num_datasets += 1
                datasets_info[key] = meta
        
        return {
            "cache_root": str(self.cache_root),
            "total_size_gb": round(total_size / (1024**3), 2),
            "num_datasets": num_datasets,
            "datasets": datasets_info,
        }

    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear cache (optionally for a specific dataset).
        
        Args:
            dataset_name: If provided, only clear this dataset.
                         If None, clear entire cache.
        """
        if dataset_name is None:
            # Clear entire cache
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self._metadata = {}
            print(f"[cache] ✓ Cleared entire cache: {self.cache_root}")
        else:
            # Clear specific dataset
            for key in list(self._metadata.keys()):
                if dataset_name in key:
                    path = self._metadata[key].get("path")
                    if path and Path(path).exists():
                        shutil.rmtree(path)
                    del self._metadata[key]
                    print(f"[cache] ✓ Cleared: {key}")
        
        self._save_metadata()

    def print_stats(self) -> None:
        """Print human-readable cache statistics."""
        stats = self.get_cache_stats()
        
        print("\n" + "=" * 70)
        print("DATASET CACHE STATISTICS")
        print("=" * 70)
        print(f"Cache Root: {stats['cache_root']}")
        print(f"Total Size: {stats['total_size_gb']} GB")
        print(f"Datasets:   {stats['num_datasets']}")
        
        if stats['datasets']:
            print("\nCached Datasets:")
            for key, meta in stats['datasets'].items():
                print(f"  • {key}")
                if "num_samples" in meta:
                    print(f"    Samples: {meta['num_samples']}")
                if "description" in meta:
                    print(f"    Description: {meta['description']}")
        
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Global cache instance
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CACHE: Optional[DatasetCache] = None


def get_cache(cache_root: Optional[str | Path] = None) -> DatasetCache:
    """Get or initialize the global dataset cache.
    
    Args:
        cache_root: Optional custom cache directory.
                   If provided, initializes a new cache instance.
    
    Returns:
        DatasetCache instance
    """
    global _GLOBAL_CACHE
    
    if cache_root is not None:
        # Create new instance with custom root
        return DatasetCache(cache_root)
    
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = DatasetCache()
    
    return _GLOBAL_CACHE

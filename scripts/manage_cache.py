#!/usr/bin/env python3
"""Cache management CLI for TIS training and evaluation.

Usage examples::

    # Show cache statistics
    python scripts/manage_cache.py --stats

    # Pre-download and cache a dataset
    python scripts/manage_cache.py --cache-dataset narrativeqa --max-samples 100

    # Setup model cache
    python scripts/manage_cache.py --setup-model-cache

    # Clear all caches
    python scripts/manage_cache.py --clear-all

    # Show cache configuration
    python scripts/manage_cache.py --show-config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from token_importance.cache import (
    DatasetCache,
    get_cache,
    ModelCache,
    setup_hf_cache,
    get_hf_cache_dirs,
)
from token_importance.training.data import SUPPORTED_DATASETS


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Cache management for TIS training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Cache root options
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Root directory for all caches (default: ~/.cache/tis_*)",
    )

    # Dataset caching
    p.add_argument(
        "--cache-dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Dataset to cache (narrativeqa, quality, qasper)",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to cache (default: train)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of cached samples",
    )

    # Statistics and info
    p.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset cache statistics",
    )
    p.add_argument(
        "--show-config",
        action="store_true",
        help="Show all cache configuration",
    )

    # Model cache setup
    p.add_argument(
        "--setup-model-cache",
        action="store_true",
        help="Setup and configure model cache",
    )

    # Cache clearing
    p.add_argument(
        "--clear-dataset",
        type=str,
        help="Clear specific dataset from cache",
    )
    p.add_argument(
        "--clear-models",
        action="store_true",
        help="Clear model cache",
    )
    p.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all caches (datasets + models)",
    )

    return p


def _show_dataset_stats(cache_root: Path | None = None) -> None:
    """Show dataset cache statistics."""
    cache = get_cache(cache_root)
    cache.print_stats()


def _cache_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
    cache_root: Path | None = None,
) -> None:
    """Cache a dataset."""
    cache = get_cache(cache_root)
    print(f"\n[cache] Caching {dataset_name}:{split}")
    print(f"[cache] Cache root: {cache.cache_root}\n")

    from token_importance.training.data import SUPPORTED_DATASETS

    if dataset_name not in SUPPORTED_DATASETS:
        print(f"[error] Unknown dataset: {dataset_name}")
        sys.exit(1)

    hf_dataset_id, default_split_with_config = SUPPORTED_DATASETS[dataset_name]
    
    # Handle "config:split" format
    config = None
    if ":" in default_split_with_config:
        config, actual_split = default_split_with_config.split(":", 1)
    else:
        actual_split = default_split_with_config
    
    # Override with explicit split if provided
    if split != "train":
        actual_split = split

    cache.cache_dataset(hf_dataset_id, actual_split, config, max_samples)


def _setup_model_cache(cache_root: Path | None = None) -> None:
    """Setup model cache."""
    model_cache = ModelCache(cache_root)
    env_vars = model_cache.configure_environment()

    print("\n[model_cache] ✓ Model cache configured!")
    print("[model_cache]")
    print("[model_cache] Environment variables set:")
    for key, value in env_vars.items():
        print(f"[model_cache]   {key}={value}")
    print("[model_cache]")
    model_cache.print_config()


def _show_all_config(cache_root: Path | None = None) -> None:
    """Show complete cache configuration."""
    print("\n" + "=" * 70)
    print("COMPLETE CACHE CONFIGURATION")
    print("=" * 70 + "\n")

    # Dataset cache
    dataset_cache = get_cache(cache_root)
    print(f"Dataset Cache Root: {dataset_cache.cache_root}")
    dataset_cache.print_stats()

    # Model cache
    model_cache = ModelCache(cache_root)
    model_cache.print_config()

    # HF environment
    print("HuggingFace Environment Variables:")
    hf_dirs = get_hf_cache_dirs()
    for key, value in hf_dirs.items():
        print(f"  {key}: {value}")
    print()


def _clear_dataset(dataset_name: str, cache_root: Path | None = None) -> None:
    """Clear specific dataset from cache."""
    cache = get_cache(cache_root)
    print(f"\n[cache] Clearing {dataset_name}...")
    cache.clear_cache(dataset_name)
    print(f"[cache] ✓ Cleared\n")


def _clear_models(cache_root: Path | None = None) -> None:
    """Clear model cache."""
    model_cache = ModelCache(cache_root)
    print(f"\n[model_cache] Clearing model cache...")
    model_cache.clear_cache()
    print(f"[model_cache] ✓ Cleared\n")


def _clear_all(cache_root: Path | None = None) -> None:
    """Clear all caches."""
    print("\n[cache] Clearing ALL caches...")
    _clear_models(cache_root)
    dataset_cache = get_cache(cache_root)
    dataset_cache.clear_cache()
    print("[cache] ✓ All caches cleared\n")


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    cache_root = args.cache_root

    # Handle different commands
    if args.stats:
        _show_dataset_stats(cache_root)

    elif args.cache_dataset:
        _cache_dataset(args.cache_dataset, args.split, args.max_samples, cache_root)

    elif args.setup_model_cache:
        _setup_model_cache(cache_root)

    elif args.show_config:
        _show_all_config(cache_root)

    elif args.clear_dataset:
        _clear_dataset(args.clear_dataset, cache_root)

    elif args.clear_models:
        _clear_models(cache_root)

    elif args.clear_all:
        _clear_all(cache_root)

    else:
        # Default: show stats
        _show_all_config(cache_root)


if __name__ == "__main__":
    main()

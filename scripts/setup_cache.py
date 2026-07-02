#!/usr/bin/env python3
"""Quick setup script for TIS cache system.

This script helps initialize the cache system for your environment.

Usage::

    python scripts/setup_cache.py
    python scripts/setup_cache.py --cache-root /custom/path
"""
from __future__ import annotations

import argparse
from pathlib import Path

from token_importance.cache import setup_hf_cache, get_cache


def main():
    """Setup cache system."""
    parser = argparse.ArgumentParser(
        description="Initialize TIS cache system",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Custom cache root directory",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only setup model cache (not datasets)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TIS CACHE SYSTEM SETUP")
    print("=" * 70 + "\n")

    cache_root = args.cache_root

    # Setup model cache
    print("[1] Setting up model cache...")
    model_cache = setup_hf_cache(cache_root)
    model_cache.print_config()

    if not args.models_only:
        # Initialize dataset cache
        print("[2] Initializing dataset cache...")
        dataset_cache = get_cache(cache_root)
        stats = dataset_cache.get_cache_stats()
        print(f"    Cache root: {stats['cache_root']}")
        print(f"    Cached datasets: {stats['num_datasets']}")
        if stats['num_datasets'] > 0:
            print("    Available datasets:")
            for key in stats['datasets'].keys():
                print(f"      • {key}")

    print("\n" + "=" * 70)
    print("✓ CACHE SYSTEM READY")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Pre-cache datasets:")
    print("     python scripts/manage_cache.py --cache-dataset narrativeqa")
    print("  2. Check cache status:")
    print("     python scripts/manage_cache.py --stats")
    print("  3. Run training/evaluation (caching will be used automatically)")
    print()


if __name__ == "__main__":
    main()

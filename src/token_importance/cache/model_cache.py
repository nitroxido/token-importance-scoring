"""Model weights caching utilities for TIS.

Manages HuggingFace model weights caching via HF_HOME environment variable.
Also provides model-specific cache configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class ModelCache:
    """Manages HuggingFace model weights caching.
    
    Features:
    - Centralized HF_HOME configuration
    - Cache path validation
    - Model download tracking
    """

    def __init__(self, cache_root: str | Path | None = None):
        """Initialize model cache.
        
        Args:
            cache_root: Root directory for model cache.
                       Defaults to ~/.cache/tis_models if None.
        """
        if cache_root is None:
            cache_root = Path.home() / ".cache" / "tis_models"
        else:
            cache_root = Path(cache_root)
        
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.hf_home = self.cache_root / "huggingface"
        self.hf_home.mkdir(parents=True, exist_ok=True)

    def configure_environment(self) -> dict[str, str]:
        """Configure environment variables for model caching.
        
        Sets HF_HOME and related env vars.
        
        Returns:
            Dictionary of environment variables to set
        """
        env_vars = {
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(self.hf_home / "transformers"),
            "HF_DATASETS_CACHE": str(self.hf_home / "datasets"),
        }
        
        # Create all directories
        for key, path in env_vars.items():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Set in environment
        for key, value in env_vars.items():
            os.environ[key] = value
        
        return env_vars

    def get_cache_size(self) -> float:
        """Get total size of cached models in GB.
        
        Returns:
            Size in gigabytes
        """
        if not self.hf_home.exists():
            return 0.0
        
        total = sum(
            f.stat().st_size
            for f in self.hf_home.rglob("*")
            if f.is_file()
        )
        return total / (1024**3)

    def clear_cache(self) -> None:
        """Clear all cached models."""
        import shutil
        
        if self.hf_home.exists():
            shutil.rmtree(self.hf_home)
            self.hf_home.mkdir(parents=True, exist_ok=True)
            print(f"[model_cache] ✓ Cleared model cache: {self.hf_home}")

    def print_config(self) -> None:
        """Print current cache configuration."""
        print("\n" + "=" * 70)
        print("MODEL CACHE CONFIGURATION")
        print("=" * 70)
        print(f"Cache Root:  {self.cache_root}")
        print(f"HF_HOME:     {self.hf_home}")
        print(f"Cache Size:  {self.get_cache_size():.2f} GB")
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Global model cache instance and setup
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_MODEL_CACHE: Optional[ModelCache] = None


def get_model_cache(cache_root: Optional[str | Path] = None) -> ModelCache:
    """Get or initialize the global model cache.
    
    Args:
        cache_root: Optional custom cache directory.
    
    Returns:
        ModelCache instance
    """
    global _GLOBAL_MODEL_CACHE
    
    if cache_root is not None:
        return ModelCache(cache_root)
    
    if _GLOBAL_MODEL_CACHE is None:
        _GLOBAL_MODEL_CACHE = ModelCache()
    
    return _GLOBAL_MODEL_CACHE


def setup_hf_cache(cache_root: Optional[str | Path] = None) -> ModelCache:
    """Setup HuggingFace caching and return cache manager.
    
    This should be called early in script execution to configure caching.
    
    Args:
        cache_root: Optional custom cache directory.
    
    Returns:
        Configured ModelCache instance
    """
    cache = get_model_cache(cache_root)
    cache.configure_environment()
    return cache


def get_hf_cache_dirs() -> dict[str, str]:
    """Get all HF cache directory environment variables.
    
    Returns:
        Dictionary of HF cache environment variables
    """
    return {
        "HF_HOME": os.environ.get("HF_HOME", "not set"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", "not set"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", "not set"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", "not set"),
    }

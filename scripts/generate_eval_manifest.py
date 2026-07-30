#!/usr/bin/env python3
"""
generate_eval_manifest.py — Generate a provenance manifest for a TIS checkpoint.

Records: SHA256 hash, source commit, environment versions, checkpoint metadata,
and the exact evaluation command so results can be traced to a specific artifact.

Usage:
    python scripts/generate_eval_manifest.py \\
        --checkpoint checkpoints/my_checkpoint \\
        --eval-script scripts/eval_niah_hard.py \\
        --output checkpoints/my_checkpoint/eval_manifest.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_package_version(package: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Generate eval_manifest.json for a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--eval-script", default=None, help="Eval script path (for recording)")
    parser.add_argument("--eval-args", default="", help="Eval script arguments (for recording)")
    parser.add_argument("--output", default=None, help="Output path (default: <checkpoint>/eval_manifest.json)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Error: checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    # Find checkpoint file(s) to hash
    file_hashes = {}
    for ext in ("*.pt", "*.bin", "*.safetensors"):
        for f in checkpoint_dir.glob(ext):
            print(f"  Hashing {f.name} ...")
            file_hashes[f.name] = sha256_file(f)

    # Load existing metadata if present
    meta_path = checkpoint_dir / "metadata.json"
    checkpoint_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            checkpoint_meta = json.load(f)

    manifest = {
        "manifest_version": "1.0",
        "generated": datetime.utcnow().isoformat(),
        "source_commit": get_git_commit(),
        "checkpoint_path": str(checkpoint_dir),
        "file_hashes": file_hashes,
        "checkpoint_metadata": checkpoint_meta,
        "environment": {
            "python": sys.version.split()[0],
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "bitsandbytes": get_package_version("bitsandbytes"),
            "peft": get_package_version("peft"),
        },
    }

    if args.eval_script:
        manifest["eval_command"] = f"python {args.eval_script} --checkpoint {args.checkpoint} {args.eval_args}".strip()

    output_path = Path(args.output) if args.output else checkpoint_dir / "eval_manifest.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {output_path}")
    for name, h in file_hashes.items():
        print(f"  {name}: {h}")


if __name__ == "__main__":
    main()

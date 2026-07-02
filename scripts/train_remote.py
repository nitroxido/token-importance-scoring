"""
Remote training functions for fal.ai.

These functions are decorated with @fal.function and run on remote A100 machines.
They are NOT called directly — use scripts/run_training.py to submit jobs.

To test the submission locally (dry-run, no actual compute):
    python scripts/run_training.py --dry_run

Prerequisites
-------------
1.  Create a fal.ai account at https://fal.ai and install the SDK:
        pip install fal
        fal auth login

2.  Generate a GitHub PAT (scope: repo) at https://github.com/settings/tokens
    and store it as a fal secret:
        fal secrets set GITHUB_TOKEN ghp_xxxx...

3.  Push your latest code before each run:
        git push origin main

The remote function always installs token-importance from the current main HEAD.
If you prefer wheel-based deployment (no PAT needed), build and upload a wheel:
    pip wheel . --no-deps -w dist/
    python -c "import fal.toolkit.file as f; print(f.upload_file_from_path('dist/token_importance-0.1.0-py3-none-any.whl'))"
Then replace the git+https URL in _install_tis() with the returned URL.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import urllib.request

# ---------------------------------------------------------------------------
# Third-party requirements installed by fal before the function body runs.
# token-importance itself is installed inside the function body because the
# GitHub repo is private and needs the GITHUB_TOKEN secret for authentication.
# ---------------------------------------------------------------------------

BASE_REQUIREMENTS = [
    "torch>=2.2",
    "transformers>=4.40",
    "peft>=0.11",
    "datasets",
    "bitsandbytes",
    "accelerate",
    "numpy",
]

GITHUB_REPO = "https://github.com/nitroxido/token-importance"


REPO_DIR = "/tmp/token-importance"


def _install_tis(github_token: str) -> None:
    """Clone repo and install token-importance (gives us scripts/ too)."""
    if not os.path.isdir(f"{REPO_DIR}/.git"):
        subprocess.run([
            "git", "clone", "--depth=1",
            f"https://{github_token}@github.com/nitroxido/token-importance.git",
            REPO_DIR,
        ], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    subprocess.run(["pip", "install", "--quiet", "-e", REPO_DIR], check=True)


# ---------------------------------------------------------------------------
# fal.ai remote functions
# ---------------------------------------------------------------------------

try:
    import fal  # type: ignore[import]

    @fal.function(
        machine_type="A100",
        requirements=BASE_REQUIREMENTS,
        secrets=["GITHUB_TOKEN"],   # fal injects this as an environment variable
        timeout=28800,              # 8-hour hard cap
    )
    def run_stage1(config: dict) -> dict:
        """Stage 1 — freeze base model, train TIS components on remote A100 (40 GB).

        Expected config keys:
            model        HF model name (default: "mistralai/Mistral-7B-v0.3")
            dataset      Dataset name  (default: "narrativeqa")
            epochs       int           (default: 2)
            batch_size   int           (default: 4)
            grad_accum   int           (default: 8)
            lr           float         (default: 1e-4)
            max_samples  int | None    (optional, for debugging)
        """
        _install_tis(os.environ["GITHUB_TOKEN"])

        cmd = [
            sys.executable, f"{REPO_DIR}/scripts/train.py",
            "--model",      config.get("model", "mistralai/Mistral-7B-v0.3"),
            "--dataset",    config.get("dataset", "narrativeqa"),
            "--stage",      "1",
            "--epochs",     str(config.get("epochs", 2)),
            "--batch_size", str(config.get("batch_size", 4)),
            "--grad_accum", str(config.get("grad_accum", 8)),
            "--lr",         str(config.get("lr", 1e-4)),
            "--bf16",
            "--output_dir", "/tmp/tis_stage1",
        ]
        if config.get("max_samples"):
            cmd += ["--max_samples", str(config["max_samples"])]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Stage 1 training failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )

        # Pack checkpoint into a single archive
        with tarfile.open("/tmp/stage1.tar.gz", "w:gz") as tar:
            tar.add("/tmp/tis_stage1", arcname="tis_stage1")

        # Upload to fal persistent storage
        import fal.toolkit.file as fal_file  # type: ignore[import]
        checkpoint_url = fal_file.upload_file_from_path("/tmp/stage1.tar.gz")

        return {
            "checkpoint_url": checkpoint_url,
            "log_tail":       result.stdout[-5000:],
        }

    @fal.function(
        machine_type="A100",        # Use "A100-80GB" if Stage 2 OOMs at batch_size=4
        requirements=BASE_REQUIREMENTS,
        secrets=["GITHUB_TOKEN"],
        timeout=43200,              # 12-hour hard cap
    )
    def run_stage2(config: dict) -> dict:
        """Stage 2 — LoRA unfreeze, starting from Stage-1 checkpoint URL.

        Expected config keys (in addition to Stage 1 keys):
            stage1_checkpoint_url   URL returned by run_stage1
            lora_r                  int  (default: 16)
            lora_alpha              int  (default: 32)
        """
        _install_tis(os.environ["GITHUB_TOKEN"])

        # Download and unpack the Stage-1 checkpoint
        stage1_url = config["stage1_checkpoint_url"]
        urllib.request.urlretrieve(stage1_url, "/tmp/stage1.tar.gz")
        with tarfile.open("/tmp/stage1.tar.gz") as tar:
            tar.extractall("/tmp/")

        cmd = [
            sys.executable, f"{REPO_DIR}/scripts/train.py",
            "--model",      "/tmp/tis_stage1",
            "--dataset",    config.get("dataset", "narrativeqa"),
            "--stage",      "2",
            "--lora_r",     str(config.get("lora_r", 16)),
            "--lora_alpha", str(config.get("lora_alpha", 32)),
            "--epochs",     str(config.get("epochs", 3)),
            "--batch_size", str(config.get("batch_size", 4)),
            "--grad_accum", str(config.get("grad_accum", 8)),
            "--bf16",
            "--output_dir", "/tmp/tis_stage2",
        ]
        if config.get("max_samples"):
            cmd += ["--max_samples", str(config["max_samples"])]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Stage 2 training failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )

        with tarfile.open("/tmp/stage2.tar.gz", "w:gz") as tar:
            tar.add("/tmp/tis_stage2", arcname="tis_stage2")

        import fal.toolkit.file as fal_file  # type: ignore[import]
        checkpoint_url = fal_file.upload_file_from_path("/tmp/stage2.tar.gz")

        return {
            "checkpoint_url": checkpoint_url,
            "log_tail":       result.stdout[-5000:],
        }

except ImportError:
    # fal is not installed locally — that is expected during development.
    # The functions are defined only when the package is available (i.e. on the
    # remote machine or after `pip install fal`).
    run_stage1 = None  # type: ignore[assignment]
    run_stage2 = None  # type: ignore[assignment]

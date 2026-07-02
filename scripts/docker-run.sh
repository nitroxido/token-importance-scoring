#!/bin/bash
#
# Helper script to run Token Importance training in a Docker container
# Handles GPU mapping, volume mounts, and environment setup
#
# Usage: ./scripts/docker-run.sh COMMAND [ARGS...]
# Examples:
#   ./scripts/docker-run.sh stage1 --model mistralai/Mistral-7B-v0.3 --epochs 2 --batch_size 4
#   ./scripts/docker-run.sh stage2 --model checkpoints/stage1 --batch_size 4
#   ./scripts/docker-run.sh eval --model checkpoints/stage2
#

set -e

# Configuration
IMAGE_NAME="${DOCKER_IMAGE:-tis-training:latest}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"

# Create necessary directories
mkdir -p "$CHECKPOINT_DIR" "$OUTPUTS_DIR"

# Parse environment variables
HF_TOKEN="${HF_TOKEN:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Build docker run command
docker_run_cmd=(
    "docker" "run"
    "--rm"
    "-it"
    "--gpus" "all"
    "--ipc" "host"
    "--ulimit" "memlock=-1"
    "--ulimit" "stack=67108864"
)

# Volume mounts
docker_run_cmd+=(
    "-v" "$CHECKPOINT_DIR:/workspace/checkpoints"
    "-v" "$OUTPUTS_DIR:/workspace/outputs"
)

# Environment variables
docker_run_cmd+=(
    "-e" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "-e" "TOKENIZERS_PARALLELISM=false"
)

# Add HF_TOKEN if provided
if [ -n "$HF_TOKEN" ]; then
    docker_run_cmd+=(
        "-e" "HF_TOKEN=$HF_TOKEN"
    )
fi

# Add image name and command
docker_run_cmd+=(
    "$IMAGE_NAME"
    "$@"
)

# Print command for debugging
echo "════════════════════════════════════════════════════════════"
echo "Running: ${docker_run_cmd[*]}"
echo "════════════════════════════════════════════════════════════"
echo ""

# Run the container
"${docker_run_cmd[@]}"

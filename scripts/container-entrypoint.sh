#!/bin/bash
#
# Container entrypoint for Token Importance training
# Supports Stage 1 and Stage 2 training with argument forwarding
#
# Usage:
#   docker run ... tis-training:latest stage1 --epochs 2 --batch_size 4
#   docker run ... tis-training:latest stage2 --lora_r 16 --batch_size 4
#   docker run ... tis-training:latest eval --model mistralai/Mistral-7B-v0.3
#

set -e

# Color output for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Display help
show_help() {
    cat << 'EOF'
Token Importance — Container Training Entrypoint

COMMANDS:
  stage1          Run Stage 1 training (freeze base, train TIS components)
  stage2          Run Stage 2 training (LoRA + all layers)
  eval            Run inference/evaluation
  shell           Drop into an interactive shell
  help            Show this help message

EXAMPLES:
  # Stage 1: Train TIS components only (A100, 2 epochs)
  docker run --gpus all tis-training:latest stage1 \
    --model mistralai/Mistral-7B-v0.3 \
    --dataset narrativeqa \
    --epochs 2 \
    --batch_size 4 \
    --grad_accum 8 \
    --bf16 \
    --output_dir /workspace/checkpoints/stage1

  # Stage 2: LoRA training on top of Stage 1 (A100, 3 epochs)
  docker run --gpus all tis-training:latest stage2 \
    --model /workspace/checkpoints/stage1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --bf16 \
    --output_dir /workspace/checkpoints/stage2

  # Evaluation on a checkpoint
  docker run --gpus all tis-training:latest eval \
    --model /workspace/checkpoints/stage2 \
    --dataset narrativeqa

ENVIRONMENT VARIABLES:
  HF_TOKEN            HuggingFace token (required for gated models like LLaMA-3)
  CUDA_VISIBLE_DEVICES GPU indices to use (default: 0)

VOLUME MOUNTS:
  /workspace/checkpoints   Storage for model checkpoints (recommended: fast SSD)
  /workspace/outputs       Storage for logs and results

EOF
}

# Check GPU availability
check_gpu() {
    log_info "Checking GPU availability..."
    python << 'PYTHON_EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: No CUDA device detected. Training will be very slow.")
PYTHON_EOF
}

# Main entrypoint logic
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        stage1)
            log_info "Starting Stage 1 training (freeze base, train TIS components)..."
            check_gpu
            python -m scripts.train \
                --stage 1 \
                "$@"
            log_info "Stage 1 training completed successfully!"
            ;;

        stage2)
            log_info "Starting Stage 2 training (LoRA + all layers)..."
            check_gpu
            python -m scripts.train \
                --stage 2 \
                "$@"
            log_info "Stage 2 training completed successfully!"
            ;;

        eval)
            log_info "Starting evaluation/inference..."
            check_gpu
            python -m scripts.eval \
                "$@"
            log_info "Evaluation completed successfully!"
            ;;

        shell)
            log_info "Dropping into shell..."
            check_gpu
            /bin/bash
            ;;

        help|--help|-h)
            show_help
            exit 0
            ;;

        *)
            log_error "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

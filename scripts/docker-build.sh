#!/bin/bash
#
# Build the Token Importance training container
# Usage: ./scripts/docker-build.sh [TAG]
# Example: ./scripts/docker-build.sh latest
#

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Image configuration
REGISTRY="${DOCKER_REGISTRY:-}"  # Leave empty for local, or set to "docker.io/youruser"
IMAGE_NAME="tis-training"
TAG="${1:-latest}"

# Full image name
if [ -z "$REGISTRY" ]; then
    FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"
else
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$TAG"
fi

echo "════════════════════════════════════════════════════════════"
echo "Building Token Importance Training Container"
echo "════════════════════════════════════════════════════════════"
echo "Image:    $FULL_IMAGE_NAME"
echo "Context:  $PROJECT_ROOT"
echo ""

# Build the image
cd "$PROJECT_ROOT"

docker build \
    --tag "$FULL_IMAGE_NAME" \
    --file Dockerfile \
    --progress=plain \
    .

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ Build successful!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "To run the container locally:"
echo "  docker run --gpus all -it --rm \\"
echo "    -v \$(pwd)/checkpoints:/workspace/checkpoints \\"
echo "    $FULL_IMAGE_NAME stage1 --help"
echo ""
echo "To push to registry:"
echo "  docker push $FULL_IMAGE_NAME"
echo ""
echo "To save as tar file for transfer:"
echo "  docker save $FULL_IMAGE_NAME | gzip > tis-training.tar.gz"
echo ""

#!/usr/bin/env bash
# backup.sh — incremental snapshot backup for token-importance
#
# Usage:
#   ./scripts/backup.sh <label> [description]
#
# Example:
#   ./scripts/backup.sh phase0 "Phase 0 complete — 48/48 tests passing"
#
# Creates: backups/<timestamp>-<label>.zip
# Appends: backups/MANIFEST.md
#
# What is backed up:
#   src/  tests/  pyproject.toml  .gitignore  scripts/
# What is excluded:
#   __pycache__  *.pyc  *.egg-info  .pytest_cache

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$REPO_ROOT/backups"
MANIFEST="$BACKUP_DIR/MANIFEST.md"

LABEL="${1:-snapshot}"
DESCRIPTION="${2:-}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
ZIP_NAME="${TIMESTAMP}-${LABEL}.zip"
ZIP_PATH="$BACKUP_DIR/$ZIP_NAME"

mkdir -p "$BACKUP_DIR"

# Initialise manifest if it doesn't exist
if [[ ! -f "$MANIFEST" ]]; then
    cat > "$MANIFEST" <<'EOF'
# Backup Manifest

Incremental snapshots of the token-importance project.
Each row corresponds to one zip archive in this folder.

| File | Date | Label | Description |
|------|------|-------|-------------|
EOF
fi

# Create the zip (relative paths, from repo root)
cd "$REPO_ROOT"
zip -r "$ZIP_PATH" \
    src/ \
    tests/ \
    pyproject.toml \
    .gitignore \
    scripts/ \
    --exclude "**/__pycache__/*" \
    --exclude "**/*.pyc" \
    --exclude "**/*.pyo" \
    --exclude "**/*.egg-info/*" \
    --exclude "**/.pytest_cache/*" \
    -q

SIZE="$(du -sh "$ZIP_PATH" | cut -f1)"
DATE="$(date +%Y-%m-%d\ %H:%M)"

# Append entry to manifest
echo "| \`$ZIP_NAME\` | $DATE | $LABEL | $DESCRIPTION |" >> "$MANIFEST"

echo "✓ Backup created: backups/$ZIP_NAME ($SIZE)"
echo "✓ Manifest updated: backups/MANIFEST.md"

#!/usr/bin/env bash
# setup_github.sh — One-shot GitHub setup for token-importance
#
# What this script does:
#   1. Checks prerequisites (git, gh CLI, authentication)
#   2. Initialises a git repository if one doesn't exist
#   3. Creates the initial commit (all tracked files)
#   4. Creates the private GitHub repo nitroxido/token-importance
#   5. Pushes main and creates the develop branch
#   6. Applies branch protection rules:
#        main    — requires PR from develop + CI pass; no direct push
#        develop — direct push allowed; CI pass required for PRs
#   7. Prints a summary with the repository URL
#
# Usage:
#   chmod +x scripts/setup_github.sh
#   ./scripts/setup_github.sh
#
# Re-running is safe (idempotent): protection rules are overwritten, not stacked.
# The GitHub repo creation step is skipped if the remote already exists.
#
# Requirements:
#   - GitHub CLI (`gh`) installed and authenticated (`gh auth login`)
#   - git installed
#   - Run from the root of the token-importance directory

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
OWNER="nitroxido"
REPO_NAME="token-importance"
FULL_REPO="${OWNER}/${REPO_NAME}"
REMOTE_URL="https://github.com/${FULL_REPO}.git"

# Status check names must match the GitHub Actions job IDs in .github/workflows/
# ci.yml defines:   jobs.pytest       → context "pytest"
# branch-policy.yml defines: jobs.check-source-branch → context "Only develop → main"
CI_CHECK="pytest"
POLICY_CHECK="Only develop \u2192 main"   # only required on main

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
info()  { echo "  [+] $*"; }
warn()  { echo "  [!] $*"; }
die()   { echo ""; echo "  ERROR: $*" >&2; exit 1; }
title() { echo ""; echo "── $* ──────────────────────────────────────────"; }

# ──────────────────────────────────────────────────────────────────────────────
# 0. Prerequisites
# ──────────────────────────────────────────────────────────────────────────────
title "Checking prerequisites"

command -v git >/dev/null 2>&1 || die "git is not installed."
info "git: $(git --version)"

if ! command -v gh >/dev/null 2>&1; then
    die "GitHub CLI (gh) is not installed.
    Install it from: https://cli.github.com
    Then authenticate with: gh auth login"
fi
info "gh: $(gh --version | head -1)"

# Verify gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
    die "GitHub CLI is not authenticated. Run: gh auth login"
fi
GH_USER="$(gh api user --jq .login 2>/dev/null)"
info "Authenticated as: ${GH_USER}"

# Warn if the authenticated user doesn't match OWNER
if [[ "${GH_USER}" != "${OWNER}" ]]; then
    warn "Logged in as '${GH_USER}' but OWNER is '${OWNER}'."
    warn "The repository will be created under '${GH_USER}' unless you change OWNER in this script."
    OWNER="${GH_USER}"
    FULL_REPO="${OWNER}/${REPO_NAME}"
    REMOTE_URL="https://github.com/${FULL_REPO}.git"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 1. Git init
# ──────────────────────────────────────────────────────────────────────────────
title "Initialising git repository"

if [[ -d .git ]]; then
    info "Already a git repository — skipping init."
else
    git init -b main
    info "Initialised empty git repository."
fi

# Make sure we're on main
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'NONE')"
if [[ "${CURRENT_BRANCH}" != "main" ]] && [[ "${CURRENT_BRANCH}" != "NONE" ]]; then
    warn "Current branch is '${CURRENT_BRANCH}', not 'main'."
    warn "Switching to main for the initial commit."
    git checkout -b main 2>/dev/null || git checkout main
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2. Initial commit (if repo is empty)
# ──────────────────────────────────────────────────────────────────────────────
title "Staging files for initial commit"

COMMIT_COUNT="$(git rev-list --count HEAD 2>/dev/null || echo 0)"
if [[ "${COMMIT_COUNT}" == "0" ]]; then
    git add -A

    # Summary of staged files
    STAGED="$(git diff --cached --name-only | wc -l)"
    info "Staged ${STAGED} files."

    git commit -m "feat: initial commit — TIS Phase 0-3 local implementation

- Phase 0: TISConfig, IMLParser, ImportanceStore, EvictionPolicy
- Phase 1: ImportanceEmbedding, ImportanceAttnBiasHook, ImportanceUpdateHead,
           PatchedCausalLM, ScoutAnnotator, demo notebook
- Phase 2: NIAH benchmarks, eval.py, baseline comparisons
- Phase 3 (local): data.py, objectives.py, train.py smoke-tested (50 steps clean)
           train_remote.py and run_training.py ready for fal.ai
- CI: .github/workflows/{ci,branch-policy}.yml
- Docs: TRAINING-INFRA.md, DEV-PLAN.md, SETUP-FAL-AI.md

159 unit tests passing. Smoke test: Qwen2.5-0.5B-Instruct, 50 steps, no NaN."
    info "Initial commit created."
else
    info "Repository already has ${COMMIT_COUNT} commit(s) — skipping initial commit."
fi

# ──────────────────────────────────────────────────────────────────────────────
# 3. Create GitHub repository
# ──────────────────────────────────────────────────────────────────────────────
title "Creating GitHub repository"

if git remote get-url origin >/dev/null 2>&1; then
    EXISTING_REMOTE="$(git remote get-url origin)"
    info "Remote 'origin' already exists: ${EXISTING_REMOTE}"
    info "Skipping repo creation."
else
    # Create the private repo and set origin (don't push yet — we control order)
    gh repo create "${FULL_REPO}" \
        --private \
        --description "Token Importance System (TIS): importance-aware KV cache eviction for long-context LLMs" \
        --disable-wiki \
        2>/dev/null || {
        # Repo may already exist on GitHub
        warn "Could not create repo (may already exist). Continuing."
    }

    git remote add origin "${REMOTE_URL}" 2>/dev/null || true
    info "Remote 'origin' → ${REMOTE_URL}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 4. Push main and create develop
# ──────────────────────────────────────────────────────────────────────────────
title "Pushing branches"

info "Pushing main…"
git push -u origin main

info "Creating develop branch…"
git checkout -b develop 2>/dev/null || git checkout develop
git push -u origin develop

info "Switching back to develop (default working branch)."

# ──────────────────────────────────────────────────────────────────────────────
# 5. Branch protection — main
# ──────────────────────────────────────────────────────────────────────────────
title "Applying branch protection: main"
# Rules:
#   - Require a pull request before merging (required_approving_review_count=0
#     allows solo-developer merge without a reviewer, but a PR is still needed).
#   - Require CI to pass ("pytest" check from ci.yml).
#   - Require the branch-policy check ("Only develop → main") to pass.
#   - Dismiss stale reviews when new commits are pushed.
#   - No direct pushes (enforce_admins=false keeps an admin escape hatch).
#   - No force-pushes; no deletions.
#   - Require linear history (squash/rebase merges only).
gh api "repos/${FULL_REPO}/branches/main/protection" \
    --method PUT \
    --silent \
    -H "Accept: application/vnd.github+json" \
    -F required_status_checks='{"strict":true,"contexts":["pytest","Only develop \u2192 main"]}' \
    -F enforce_admins=false \
    -F 'required_pull_request_reviews={"required_approving_review_count":0,"dismiss_stale_reviews":true,"require_last_push_approval":false}' \
    -F restrictions=null \
    -F allow_force_pushes=false \
    -F allow_deletions=false \
    -F required_linear_history=true \
    -F required_conversation_resolution=false
info "main protection applied."

# ──────────────────────────────────────────────────────────────────────────────
# 6. Branch protection — develop
# ──────────────────────────────────────────────────────────────────────────────
title "Applying branch protection: develop"
# Rules:
#   - No required PR reviews → owner can push directly.
#   - Require CI to pass for any PR targeting develop.
#   - Allow force-push (useful during active development).
#   - No deletions.
gh api "repos/${FULL_REPO}/branches/develop/protection" \
    --method PUT \
    --silent \
    -H "Accept: application/vnd.github+json" \
    -F required_status_checks='{"strict":true,"contexts":["pytest"]}' \
    -F enforce_admins=false \
    -F required_pull_request_reviews=null \
    -F restrictions=null \
    -F allow_force_pushes=true \
    -F allow_deletions=false
info "develop protection applied."

# ──────────────────────────────────────────────────────────────────────────────
# 7. Set develop as default branch
# ──────────────────────────────────────────────────────────────────────────────
title "Setting default branch to develop"
gh api "repos/${FULL_REPO}" \
    --method PATCH \
    --silent \
    -F default_branch=develop
info "Default branch set to 'develop'."

# ──────────────────────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Repository : https://github.com/${FULL_REPO}"
echo "  Branches   : main (protected), develop (pushable)"
echo "  CI         : .github/workflows/ci.yml"
echo "  Policy     : only 'develop' may PR to 'main'"
echo ""
echo "  Daily workflow:"
echo "    git checkout develop"
echo "    git checkout -b feature/my-feature"
echo "    # ... work ..."
echo "    git push origin feature/my-feature"
echo "    gh pr create --base develop --title 'feat: my feature'"
echo ""
echo "  Promote develop → main:"
echo "    gh pr create --base main --head develop --title 'release: vX.Y'"
echo "════════════════════════════════════════════════════════"

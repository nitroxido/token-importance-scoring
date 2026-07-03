# Contributing to Token Importance Scoring

We appreciate interest in this work. This document explains how to contribute ideas, code, and feedback.

## How to Contribute

### Report Bugs or Suggest Features

Open an issue on GitHub with:
- What you tried to do
- What happened instead
- Your environment (GPU, PyTorch version, etc)
- Error messages or unexpected output

We'll respond and either fix it or discuss next steps.

### Propose Changes

Have an idea for an improvement? Start a discussion or open an issue first. This saves you time and us time. We can talk about whether it fits the project direction before you write code.

For experimental work (like Phase 4 query-aware importance), see the notes in PHASE4-PROPOSAL.md. There's a clear roadmap if you want to build on that.

### Submit Code Changes

We use a standard fork and pull request workflow:

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a feature branch: `git checkout -b feature/your-idea`
4. Make your changes
5. Test locally (run scripts/eval.py or scripts/train_ert.py to verify nothing breaks)
6. Commit with clear messages
7. Push to your fork
8. Open a pull request back to main

**PR Requirements:**
- At least one approval before merging (we'll review your changes)
- Commits should be on a feature branch, not directly on main
- If changes are requested, just push to your branch and the PR updates automatically
- All merge styles are fine (merge, squash, or rebase)

**What to include in your PR:**
- What the change does
- Why it's needed
- How you tested it
- Any new dependencies
- Benchmark results if relevant

We'll review and either merge or suggest changes.

### Benchmarking and Validation

If you're adding a new benchmark or training method:
- Include results on standard benchmarks (NIAH, LITM)
- Test on consumer GPU if possible (RTX 5070 results especially valuable)
- Document any new hyperparameters or setup requirements
- Compare against existing baselines in your PR description

### Code Style

Keep it simple and readable. We prefer:
- Clear variable names over clever tricks
- Comments explaining the why, not the what
- Modular functions that can be tested independently
- Following existing patterns in the codebase

Run your code and make sure it works. That's the main requirement.

## What We're Looking For

### High Priority

- Bug fixes and error handling improvements
- Performance optimizations for consumer GPUs
- Better documentation or clearer examples
- New benchmark datasets or evaluation metrics
- Results validating (or invalidating) existing findings

### Experimental

- Query-aware importance training (see PHASE4-PROPOSAL.md)
- Alternative hard-anchor implementations
- Other compression objectives beyond KL divergence
- Comparison with new baseline methods

### Lower Priority for Now

- Major architectural changes without prior discussion
- New model bases (we're focused on Mistral-7B)
- Unrelated utilities (keep the repo focused)

## Questions or Want to Discuss

If you're unsure whether something fits or where to start, just open a discussion on GitHub or look at existing issues. No need to guess.

## Recognition

We'll credit contributors in:
- The project README
- Git commit history
- Any future publications referencing this work

## License

By contributing, you agree your code is under the MIT license (same as the project).

---

Happy to hear your ideas. Looking forward to collaborating.

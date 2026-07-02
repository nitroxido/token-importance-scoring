# Rebuild Instructions: Regenerating All Data and Models

This document explains how to regenerate all Token Importance Scoring v8b data, checkpoints, and benchmark results from scratch using the source code archive.

## Overview

The release package contains:
1. **Documentation** (15 files): Complete guides, architecture specs, reproducibility instructions
2. **Source Code** (tar.gz): All Python source code, scripts, and training infrastructure
3. **Images** (3 PNG files): Architecture diagrams and benchmark visualizations

The source code is too large to include as individual files, so it's compressed as `token-importance-source-code.tar.gz`. Other researchers should regenerate the data themselves due to size constraints.

## Step 1: Extract and Setup

```bash
# Create working directory
mkdir -p $PROJECT_DIR/rebuild
cd $PROJECT_DIR/rebuild

# Extract source code archive
tar -xzf token-importance-source-code.tar.gz

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate  # On Windows

# Install Python package and dependencies
pip install -e .

# Verify installation
python -c "from token_importance.model.importance_head import ImportanceUpdateHead; print('✓ Package installed')"
```

## Step 2: Download Models and Datasets

### Base Model (Required for training and inference)

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Download base model
huggingface-cli download mistralai/Mistral-7B-v0.3 \
  --local-dir $PROJECT_DIR/rebuild/models/mistral-7b-v0.3 \
  --resume-download
```

**Size:** ~13 GB  
**Why:** Base model that will be fine-tuned with importance scoring

### Benchmark Datasets

#### NIAH (Synthetic Benchmark)
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

python << 'EOF'
from token_importance.eval.benchmarks import NIAH
niah = NIAH(budget_fractions=[0.25, 0.5, 0.75, 1.0])
niah.prepare_dataset()  # Generates synthetic data on-the-fly
print(f"✓ NIAH: {len(niah)} samples")
EOF
```

**Size:** ~50 MB (generated)  
**Why:** Tests position-dependent token importance on synthetic needles

#### LITM (Semantic Benchmark)
```bash
python << 'EOF'
from token_importance.eval.benchmarks import LITM
litm = LITM(budget_fractions=[0.25, 0.5, 0.75, 1.0])
litm.prepare_dataset()  # Downloads from HuggingFace
print(f"✓ LITM: {len(litm)} samples")
EOF
```

**Size:** ~200 MB  
**Why:** Real semantic QA requiring importance learning

#### MultiDoc (Real-world Benchmark)
```bash
python << 'EOF'
from token_importance.eval.benchmarks import MultiDoc
multidoc = MultiDoc(budget_fractions=[0.25, 0.5, 0.75, 1.0])
multidoc.prepare_dataset()
print(f"✓ MultiDoc: {len(multidoc)} samples")
EOF
```

**Size:** ~100 MB  
**Why:** Multi-document retrieval with realistic complexity

### Training Dataset (Optional, for Phase 4)

```bash
python << 'EOF'
from token_importance.training.msmarco_data import download_msmarco
download_msmarco(
    output_dir='$PROJECT_DIR/rebuild/data/msmarco',
    split='train',
    num_samples=50000  # Use subset for faster iteration
)
print("✓ MS-MARCO dataset downloaded")
EOF
```

**Size:** ~2-10 GB (depends on number of samples)  
**Why:** Real retrieval queries for query-aware importance training (Phase 4)

## Step 3: Train TIS v8b

### Option A: Train from Scratch (Recommended for Full Understanding)

# Ensure virtual environment is activated
source .venv/bin/activate

```bash
cd $PROJECT_DIR/rebuild

python scripts/train_ert.py \
  --model_id ./models/mistral-7b-v0.3 \
  --output_dir ./checkpoints/stage3_ert_from_scratch \
  --num_train_epochs 3 \
  --learning_rate 5e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --mixed_precision bfloat16 \
  --use_rms_norm \
  --use_hard_anchor_forcing \
  --stability_loss_weight 0.1 \
  --kl_divergence_weight 1.0 \
  --alignment_weight 0.3 \
  --save_total_limit 3 \
  --logging_steps 10
```

**Expected Duration:** Training time varies based on hardware and configuration
**Expected Results:**
- NIAH @ 50%: ~100% ± 2% (with proper training configuration)
- LITM @ 50%: ~52-54%
- Generation quality: ~67%

Note: Training scripts and their parameters may require adjustment based on the specific implementation. Consult the scripts/ directory for available training scripts and their actual command-line interfaces.
pre-trained checkpoints:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# If available from a model hub:
# huggingface-cli download [model-path] --local-dir ./checkpoints/stage3_ert_baseline
```

This skips training time
```

This skips ~45 minutes of training but still requires evaluation.

## Step 4: Evaluate on All Benchmarks

### Run Single Benchmark

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Note: Adapt parameters based on actual eval.py interface
# Check available parameters with: python scripts/eval.py --help
python scripts/eval.py \
  --model ./checkpoints/stage3_ert_from_scratch \
  --baseline tis \
  --benchmark niah \
  --cache_budgets 0.25 0.5 0.75 1.0 \
  --output ./results/niah_eval.csv
```

### Run All Benchmarks (Comprehensive)

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

python << 'EOF'
import subprocess
import json

checkpoints = {
    'TIS Stage 3': './checkpoints/stage3_ert_from_scratch',
}

benchmarks = ['niah', 'litm', 'multidoc']
budgets = [0.25, 0.5, 0.75, 1.0]

results = {}
for name, ckpt in checkpoints.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {name} ({ckpt})")
    print('='*60)
    
    for benchmark in benchmarks:
        print(f"\n  Running {benchmark.upper()}...")
        result = subprocess.run([
            'python', 'scripts/eval.py',
            '--model', ckpt,
            '--baseline', 'tis',
            '--benchmark', benchmark,
            '--cache_budgets', *map(str, budgets),
            '--output', f'./results/{benchmark}_{name.lower().replace(" ", "_")}.csv'
        ])
        
        if result.returncode == 0:
            print(f"    ✓ {benchmark.upper()} complete")

print("\n" + "="*60)
print("All evaluations complete!")
print("Results saved to ./results/")
EOF
```

**Expected Duration:** 4-6 hours on RTX 5070  
**Output:** JSON files with accuracy metrics per budget

## Step 5: Compare with Baselines

### Create Baseline Comparison

```bash
python << 'EOF'
from token_importance.eval.baselines import BaselineComparison
from token_importance.eval.benchmarks import NIAH, LITM

comp = BaselineComparison(model_path='./models/mistral-7b-v0.3')

# Run all baselines on NIAH
niah_results = comp.evaluate_all_methods(
    benchmark='niah',
    cache_budgets=[0.25, 0.5, 0.75, 1.0],
    methods=['vanilla', 'streamingllm', 'h2o', 'snapkv', 'infini_attn']
)

# Run all baselines on LITM
litm_results = comp.evaluate_all_methods(
    benchmark='litm',
    cache_budgets=[0.25, 0.5, 0.75, 1.0],
    methods=['vanilla', 'streamingllm', 'h2o', 'snapkv', 'infini_attn']
)

# Save comparison
import json
with open('./results/baseline_comparison.json', 'w') as f:
    json.dump({'niah': niah_results, 'litm': litm_results}, f, indent=2)

print("✓ Baseline comparison complete")
EOF
```

**Expected Duration:** 6-8 hours for all baselines on all benchmarks  
**Output:** Comparison tables matching MASTER-EXECUTION-PLAN-PHASE4-COMPLETE.md

## Step 6: Optional - Phase 4 Query-Aware Training

For advanced users who want to reproduce the query-aware importance head:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Note: Check available Phase 4 training scripts in scripts/ directory
# Example (actual script and parameters may vary):
python scripts/train_phase4.py \
  --model ./checkpoints/stage3_ert_from_scratch \
  --data data/msmarco/train \
  --output-dir ./checkpoints/stage4_query_aware

# Or use other available training scripts:
# - train_query_aware_phase4.py
# - train_phase_b.py
# Check scripts/ directory for actual implementations
```

Then evaluate:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Adapt to actual eval.py interface
python scripts/eval.py \
  --model ./checkpoints/stage4_query_aware \
  --baseline tis \
  --benchmark litm \
  --output ./results/phase4_results.csv
```

**Expected improvement over Stage 3:**
- LITM @ 50%: +5-7pp (target: 55-60%)
- LITM @ 75%: +3-5pp (target: 72-77%)

## Step 7: Generate Results Summary

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

python << 'EOF'
import json
import glob

# Collect all results
results = {}
for json_file in glob.glob('./results/**/metrics.json', recursive=True):
    with open(json_file) as f:
        data = json.load(f)
        results[json_file] = data

# Generate summary table (markdown format)
summary_md = "# Regenerated Results Summary\n\n"
summary_md += "| Checkpoint | Benchmark | Budget | Accuracy |\n"
summary_md += "|-----------|-----------|--------|----------|\n"

for path, metrics in results.items():
    for budget, acc in metrics.items():
        summary_md += f"| TIS v8b | {path.split('/')[-2]} | {budget} | {acc:.1%} |\n"

with open('./results/SUMMARY.md', 'w') as f:
    f.write(summary_md)

print("✓ Summary generated: ./results/SUMMARY.md")
EOF
```

## Verification Checklist

After completing all steps, verify:

- [ ] Source code extracted and installed
- [ ] Virtual environment created and activated (`.venv/`)
- [ ] Base model downloaded (~13 GB)
- [ ] All benchmarks prepared (NIAH, LITM, MultiDoc)
- [ ] TIS v8b trained or checkpoint downloaded
- [ ] NIAH evaluation complete (expect ~100% @ 50%)
- [ ] LITM evaluation complete (expect ~52-54% @ 50%)
- [ ] MultiDoc evaluation complete
- [ ] Baselines evaluated (Vanilla, StreamingLLM, H2O, SnapKV, Infini-Attention)
- [ ] Results match documentation within ±2% variance
- [ ] Results summary generated (SUMMARY.md)

## Troubleshooting

### Q: Training takes too long
**A:** This is normal! Training ~45 min per 500 steps on RTX 5070 is expected. On faster hardware (A100), reduce accordingly. Use `--gradient_accumulation_steps 2` and `--batch_size 4` for faster training if you have 40GB+ VRAM.

### Q: NIAH accuracy is <95%
**A:** Check:
- [ ] `--use_hard_anchor_forcing` is enabled
- [ ] `--stability_loss_weight` is >0 (default 0.1)
- [ ] Learning rate is appropriate (default 5e-4)

### Q: LITM accuracy is <40%
**A:** This is expected for Stage 3. The model is trained on random evictions, not semantic importance. For higher LITM scores, either:
1. Use query-aware training (Phase 4)
2. Increase alignment loss weight (`--alignment_weight 0.5`)

### Q: Out of memory during training
**A:**
- Set `--batch_size 1` and `--gradient_accumulation_steps 8` (minimum config)
- Enable `--mixed_precision bfloat16`
- Reduce `--max_length 8192`

### Q: Evaluation takes forever
**A:** 
- Evaluate on subset first: `--benchmark niah --cache_budgets 0.5 --n_samples 10`
- Use smaller context windows for development: (context length is set per benchmark)
- Run in parallel on multiple GPUs if available

## Next Steps

After successfully rebuilding:

1. **Compare your results** with the values in MASTER-EXECUTION-PLAN-PHASE4-COMPLETE.md
2. **Extend the work**: Try different loss weights, architectures, or training objectives
3. **Publish your findings**: Document improvements and contribute back to the community
4. **Optimize for your hardware**: Adjust hyperparameters for your specific GPU

## References

- **REPRODUCIBILITY-GUIDE.md**: Complete reproduction instructions
- **PHASE-C-QUERY-AWARE-IMPLEMENTATION.md**: Phase 4 deep dive
- **ERT-TRAINING-PLAN.md**: ERT objective theory and implementation
- **SOURCE-CODE-README.md**: Detailed source code documentation

---

**Last Updated:** July 2026
**Estimated Total Time:** 8-12 hours on RTX 5070 (includes 45 min training + 4-6 hours evaluation)  
**Total Data Size:** ~15-25 GB (models + datasets)

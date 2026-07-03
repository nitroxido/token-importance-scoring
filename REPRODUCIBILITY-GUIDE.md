# Token Importance Scoring - Reproducibility Guide

**Last Updated**: June 2026
**Target Platform**: Linux/macOS with PyTorch-compatible GPU (RTX 5070 tested)
**Estimated Reproduction Time**: 8-10 hours (complete pipeline)

---

## Part 1: Environment Setup

### 1.1 Python Dependencies Installation

Create and activate Python virtual environment:

```bash
python3.10 -m venv tis_env
source tis_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install transformers==4.36.0 peft==0.7.0 bitsandbytes==0.41.3
pip install datasets==2.14.5 accelerate==0.25.0

# Install utility packages
pip install numpy pandas matplotlib seaborn tensorboard tqdm pyyaml

# Optional: Jupyter for notebooks
pip install jupyter ipykernel ipywidgets
```

### 1.2 Verify Installation

Validate package versions and GPU availability:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import peft; print(f'PEFT {peft.__version__}')"
```

### 1.3 Base Model Download

Download Mistral-7B-v0.3 (requires approximately 15GB disk space):

```bash
mkdir -p models
cd models

# Option A: Using Hugging Face CLI (requires login)
huggingface-cli login
huggingface-cli download mistralai/Mistral-7B-v0.3

# Option B: Using transformers (automatic)
python -c "from transformers import AutoModelForCausalLM; \
  AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.3', \
  device_map='cuda', trust_remote_code=True)"
```

---

## Part 2: Data Preparation

### 2.1 Benchmark Data Preparation

All benchmarks are included in the `data/` directory or can be regenerated.

```bash
# NIAH - Needle in a Haystack (synthetic)
python scripts/prepare_niah.py \
  --output-dir data/niah \
  --n-samples-per-budget 450

# LITM - Lost in the Middle (semantic QA)
python scripts/prepare_litm.py \
  --output-dir data/litm \
  --dataset-version v1

# NarrativeQA (for training data)
python scripts/prepare_narrativeqa.py \
  --output-dir data/narrativeqa \
  --split train
```

**Expected Output**:
```
data/
├── niah/
│   ├── budget_0.25.jsonl (450 samples)
│   ├── budget_0.50.jsonl (450 samples)
│   ├── budget_0.75.jsonl (450 samples)
│   └── budget_1.00.jsonl (450 samples)
├── litm/
│   └── dev.jsonl (1000 samples)
└── narrativeqa/
    ├── train.jsonl (32.7K samples)
    └── dev.jsonl (3.2K samples)
```

---

## Part 3: Reproduction Workflows

### 3.0 Download Pre-trained Checkpoints (Optional)

**Goal**: Skip training by using pre-trained models.

```bash
# Activate virtual environment
source .venv/bin/activate

# Download main ERT checkpoint (recommended)
hf download oldman-dev/tis-stage3-ert \
  --local-dir checkpoints/stage3_ert_learned

# Optional: Download additional checkpoints
hf download oldman-dev/tis-v8b-hard-anchor \
  --local-dir checkpoints/v8b_hard_anchor

hf download oldman-dev/tis-stage1-oracle \
  --local-dir checkpoints/stage1_oracle
```

**Available Models:**
- [tis-stage3-ert](https://huggingface.co/oldman-dev/tis-stage3-ert): Main checkpoint (100% NIAH, 52.8% LITM)
- [tis-v8b-hard-anchor](https://huggingface.co/oldman-dev/tis-v8b-hard-anchor): Hard-anchor checkpoint (82% NIAH @ 25%)
- [tis-stage1-oracle](https://huggingface.co/oldman-dev/tis-stage1-oracle): Oracle baseline

### 3.1 Quick Validation (20 minutes)

**Goal**: Verify everything loads without errors.

```bash
# Load base model + checkpoints
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.3', 
                                             device_map='cuda', load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')
print('Base model loads successfully')

# Load checkpoint (after downloading from HuggingFace)
import torch
ckpt = torch.load('checkpoints/stage3_ert_learned/tis_components.pt')
print(f'Checkpoint loaded: {len(ckpt)} parameters')
"
```

### 3.2 Full Benchmark Evaluation (2 hours)

**Goal**: Reproduce all published results.

```bash
# Evaluate Stage 3 (ERT Learned) on NIAH
python scripts/eval_niah.py \
  --checkpoint-path checkpoints/stage3_ert_learned \
  --output-dir results/stage3_niah \
  --n-samples 450 \
  --batch-size 1

# Expected Output:
# NIAH @ 25%: 100%
# NIAH @ 50%: 100%
# NIAH @ 75%: 100%
# NIAH @ 100%: 100%

# Evaluate Stage 3 on LITM
python scripts/eval_litm.py \
  --checkpoint-path checkpoints/stage3_ert_learned \
  --output-dir results/stage3_litm \
  --batch-size 1

# Expected Output:
# LITM @ 25%: 33.3%
# LITM @ 50%: 52.8%
# LITM @ 75%: 69.4%

# Evaluate V8b (Hard-Anchor Tuned) on NIAH
python scripts/eval_niah.py \
  --checkpoint-path checkpoints/v8b_hard_anchor \
  --output-dir results/v8b_niah \
  --n-samples 450

# Expected Output:
# NIAH @ 25%: 92%
# NIAH @ 50%: 78%  ← Publication result
# NIAH @ 75%: 85%
```

### 3.3 Re-train Stage 3 from Scratch (8 hours)

**Goal**: Fully reproduce training without using published checkpoint.

```bash
# Ensure data is prepared
python scripts/prepare_narrativeqa.py --output-dir data/narrativeqa

# Stage 3: ERT Training (Constraint-Aware)
python scripts/train_stage3_ert.py \
  --model-path mistralai/Mistral-7B-v0.3 \
  --data-path data/narrativeqa/train.jsonl \
  --output-dir checkpoints/my_stage3_ert \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-4 \
  --max-steps 10000 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --eval-steps 500 \
  --logging-steps 100 \
  --use-4bit-quantization true \
  --use-lora true \
  --lora-r 16 \
  --lora-alpha 32

# Expected Training Time: ~7.8 hours on RTX 5070 (8GB)
# Expected Final Results: 100% NIAH @ all budgets
```

### 3.4 Re-train Stage 1 (Oracle) for Baseline (30 hours)

**Goal**: Validate oracle ceiling on a full machine.

```bash
# Stage 1: Oracle Training (frozen base model, learns with oracle labels)
python scripts/train_stage1_oracle.py \
  --model-path mistralai/Mistral-7B-v0.3 \
  --data-path data/narrativeqa/train.jsonl \
  --oracle-importance-path data/oracle_importance_annotations.jsonl \
  --output-dir checkpoints/my_stage1_oracle \
  --num-train-epochs 2 \
  --max-steps 16400 \
  --learning-rate 1e-3

# Expected Training Time: ~29.4 hours on A100-80GB
# Expected Final Results: 100% NIAH @ all budgets (oracle labels)
```

---

## Part 4: Reproduction Outputs

### 4.1 Expected File Structure After Reproduction

```
results/
├── stage3_niah/
│   ├── budget_0.25.csv
│   ├── budget_0.50.csv
│   ├── budget_0.75.csv
│   ├── budget_1.00.csv
│   └── summary.json
├── stage3_litm/
│   ├── budget_0.50.csv
│   ├── predictions.jsonl
│   └── summary.json
├── v8b_niah/
│   ├── budget_0.50.csv
│   └── summary.json
└── comparison_table.csv

checkpoints/
├── my_stage3_ert/
│   ├── pytorch_model.bin
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── config.json
└── my_stage1_oracle/
    ├── pytorch_model.bin
    └── config.json
```

### 4.2 Comparing Against Published Results

```bash
# Generate comparison report
python scripts/compare_results.py \
  --published-results results/published/ \
  --reproduced-results results/stage3_niah \
  --output-dir results/comparison

# Expected Output:
# NIAH @ 50%:
#   Published:    100%
#   Reproduced:   100%
#   Match:        YES
#
# LITM @ 50%:
#   Published:    52.8%
#   Reproduced:   52.8% (±0.5pp)
#   Match:        YES
```

---

## Part 5: Troubleshooting

### 5.1 CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size or gradient accumulation
python scripts/train_stage3_ert.py \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16  # ← Double this

# Or use CPU inference (slow but works)
python scripts/eval_niah.py \
  --checkpoint-path checkpoints/stage3_ert_learned \
  --device cpu \
  --batch-size 1
```

### 5.2 Model Loading Errors

**Symptom**: `OSError: Can't load model. Model not found`

**Solution**:
```bash
# Ensure model is downloaded
huggingface-cli download mistralai/Mistral-7B-v0.3 --local-dir models/mistral-7b

# Or set explicit path
export TRANSFORMERS_CACHE=/path/to/models
python scripts/eval_niah.py --checkpoint-path checkpoints/stage3_ert_learned
```

### 5.3 Data Loading Issues

**Symptom**: `FileNotFoundError: data/niah/budget_0.50.jsonl`

**Solution**:
```bash
# Regenerate data
python scripts/prepare_niah.py --output-dir data/niah --overwrite

# Or download pre-generated
wget https://huggingface.co/datasets/your-org/tis-niah/data/niah_budget_0.50.jsonl -O data/niah/budget_0.50.jsonl
```

### 5.4 Checkpoint Incompatibility

**Symptom**: `RuntimeError: Error(s) in loading state_dict`

**Solution**:
```bash
# Verify checkpoint version
python -c "import torch; ckpt = torch.load('checkpoints/stage3_ert_learned/pytorch_model.bin'); print(ckpt.keys())"

# Ensure PyTorch/transformers versions match
pip install torch==2.1.2 transformers==4.36.0
```

---

## Part 6: Extension Points

### 6.1 Add Custom Data

To evaluate on your own data:

```python
# scripts/eval_custom.py
from src.tis_model import TISModel

model = TISModel.from_checkpoint('checkpoints/stage3_ert_learned')

# Your data
context = "Your long document here..."
query = "Your question..."

# Get importance scores
scores = model.get_importance_scores(context, query, budget=0.5)

# Apply KV cache filtering
pruned_cache = model.apply_cache_pruning(context, scores, budget=0.5)

# Generate response
response = model.generate(context, query, cache=pruned_cache)
```

### 6.2 Implement Query-Aware Learning (Phase 4)

See [PHASE4-REPRODUCTION-GUIDE.md](PHASE4-REPRODUCTION-GUIDE.md) for Phase 4 roadmap. This was an attempt to extract maximum value from the TIS system, but it was postponed (see below).

Key components to add:
- `src/query_aware_importance_head.py` — Cross-attention mechanism
- `src/query_signal_extraction.py` — Query representations
- Updated loss function in `scripts/train_stage4_query_aware.py`

### 6.3 Address Attention Drift (Phase 4 Planned Enhancement)

See [ATTENTION-DRIFT-ANALYSIS.md](ATTENTION-DRIFT-ANALYSIS.md) for details on post-normalization.

```python
# From src/transformer_with_postnorm.py
from src.transformer_with_postnorm import MistralWithPostNorm

model = MistralWithPostNorm.from_pretrained('mistralai/Mistral-7B-v0.3')
# Add LayerNorm after residual connections → stabilize magnitudes
```

---

## Part 7: Documentation & Results Analysis

This section is a preparation for sharing and documenting the results of this Phase. Data here should be considered comprehensive reference material.

### 7.1 Citation Format

```bibtex
@article{token_importance_scoring_2026,
  title={Token Importance Scoring for KV Cache Compression: 
         Constraint-Aware Learning with Hard-Anchor Preservation},
  author={<author_name>},
  journal={arXiv preprint arXiv:2406.XXXXX},
  year={<year>},
  month={<month>},
  howpublished={\url{https://arxiv.org/abs/2406.XXXXX}},
  code={\url{https://github.com/<github_handle>/token-importance-scoring}}
}
```

### 7.2 Results Reporting

When reporting results, include:
- **System**: GPU model, VRAM, PyTorch version
- **Data**: Benchmark name, sample size, budget levels
- **Checkpoint**: Version (oracle/ert/v8b) and location
- **Metrics**: Accuracy, generation quality, timing
- **Variance**: Standard deviation across runs (if applicable)

Example:

> "All experiments conducted on NVIDIA RTX 5070 (8GB VRAM) with PyTorch 2.1.2. NIAH results reported over 450 samples per budget level. TIS Stage 3 (ERT learned) checkpoint achieved 100% accuracy at 50% budget. Training required 7.8 GPU-hours on RTX 5070."

---

## Part 8: Validation Checklist

Before considering reproduction complete, verify:

- [ ] Environment setup completed without errors
- [ ] Base model loads successfully
- [ ] All benchmarks downloaded/generated
- [ ] Stage 3 checkpoint evaluates correctly
  - [ ] NIAH @ 50%: 100% (±0%)
  - [ ] LITM @ 50%: 52.8% (±1pp)
- [ ] V8b checkpoint evaluates correctly
  - [ ] NIAH @ 50%: 78% (±1pp)
  - [ ] Evidence survival: 100%
- [ ] Training from scratch produces similar results
  - [ ] Loss curves match expected pattern
  - [ ] Final metrics within ±1pp of published
- [ ] All output files match expected structure
- [ ] Documentation matches your environment

---

## Part 9: Getting Help

**If reproduction fails**:

1. **Check logs**: Look at `results/*.log` for error details
2. **Verify environment**: Run `scripts/verify_setup.py`
3. **Test individual components**: Run `scripts/test_*.py` files
4. **Review [PROJECT-EVOLUTION-REPORT.md](PROJECT-EVOLUTION-REPORT.md)**: Check architecture decisions and known limitations
5. **Check Phase 4 plans**: See [PHASE4-REPRODUCTION-GUIDE.md](PHASE4-REPRODUCTION-GUIDE.md) for upcoming improvements

**Known Limitations**:
- LITM performance lags SnapKV (52.8% vs 55.6%) — query-aware learning needed
- Domain mixing degrades NIAH by ~12pp — separate heads required for Phase 2
- Attention drift unaddressed — Phase 4 post-norm solution planned

---

## Summary

**Quick Start** (if using published checkpoints):
```bash
source tis_env/bin/activate
python scripts/eval_niah.py --checkpoint-path checkpoints/stage3_ert_learned
# Result: 100% NIAH @ all budgets in ~20 minutes
```

**Full Reproduction** (training from scratch):
```bash
python scripts/prepare_narrativeqa.py --output-dir data/narrativeqa
python scripts/train_stage3_ert.py --output-dir checkpoints/my_stage3_ert
# Duration: ~8 hours on RTX 5070
```

**Expected Outcomes**:
- 100% NIAH @ all budgets (ERT learned)
- 52.8% LITM @ 50% (matches oracle ceiling)
- 67% generation quality (no memorization)
- Reproducible on consumer hardware

---

## Part 10: Acknowledgments & Hardware Support

### GPU-Action Partnership

This project benefited from **GPU-Action (https://gpu-action.com)** sponsorship providing access to A100-80GB hardware for critical training stages:

- **Stage 1 Oracle Training**: 29.4 GPU-hours on A100 to establish ground-truth performance ceiling
- **Baseline Validation**: Additional A100 hours for comprehensive baseline comparison (7 methods)
- **Enterprise Scale Validation**: Ensures reproducibility not just on consumer hardware but also at scale

Without GPU-Action's support, the oracle baseline (100% NIAH @ all budgets) that validates the entire architectural approach would not have been possible.

### Hardware Requirements Summary

| Component | Hardware | Critical For | Sponsored By |
|-----------|----------|---|---|
| Stage 1 Oracle | A100-80GB | Ground-truth validation | GPU-Action (sponsored) |
| Stage 3 ERT | RTX 5070 | Consumer reproducibility | Community (standard) |
| Publication Results | RTX 5070 | Accessibility proof | Community (standard) |

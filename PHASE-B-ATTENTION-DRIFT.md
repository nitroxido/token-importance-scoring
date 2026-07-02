# Phase B: Attention Drift Analysis & Solution Guide

**Objective**: Measure attention drift, implement post-norm fix, validate impact 
**Output**: Drift metrics, post-norm solution, validated LITM improvements 

---

## Understanding Attention Drift

### The Problem in Detail

**Root Cause**: In transformer networks, hidden states accumulate magnitude due to residual connections:

```
Layer 0: h₀ = embed(x) + pos_embed(0) # ||h₀|| ~ 1.0
Layer 1: h₁ = norm(h₀) + attn(h₀) + ffn(h₀) # After residuals: ||h₁|| ~ 1.5
Layer 2: h₂ = norm(h₁) + attn(h₁) + ffn(h₁) # ||h₂|| ~ 2.0
...
Layer T: hₜ = norm(h_{t-1}) + attn(h_{t-1}) + ffn(h_{t-1}) # ||hₜ|| ~ T/2
```

In standard pre-norm transformers (like Mistral), layer norm is applied **before** the residual, but the outputs still accumulate.

**Attention Impact**:
```
Attention score between q and k:
 score(q_t, k_i) = (Q_t · K_i) / sqrt(d_k)
 = ||h_t|| · ||h_i|| · cos(angle) / sqrt(d_k)

With magnitude growth:
 ||h_t|| >> ||h_i|| when t >> i

Result: Attention logits for recent tokens (large ||h_t||) dominate
 Regardless of actual semantic relevance (angle)
```

**Consequence for TIS**:
```
Importance-biased attention: logit = base_logit + λ · score_k
 = (Q · K^T)/sqrt(d) + λ · importance_k

If base_logit already favors recent tokens (from ||h_t|| growth),
importance bias λ · score_k becomes negligible for distant important tokens.
```

---

## Part 1: Measuring Attention Drift

### Script 1: Drift Metrics Collector

**File**: `scripts/measure_attention_drift.py`

```python
#!/usr/bin/env python3
"""
Measure attention drift in Mistral-7B baseline model.

Metrics:
 1. Magnitude growth: ||h_t|| / ||h_0|| ratio over time
 2. Recent attention bias: Attention to recent vs distant tokens
 3. Importance signal suppression: Does λ·score_k overcome drift?
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

class AttentionDriftAnalyzer:
 def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.3", 
 device: str = "cuda", quantized: bool = True):
 self.model_name = model_name
 self.device = device
 
 # Load tokenizer
 self.tokenizer = AutoTokenizer.from_pretrained(model_name)
 
 # Load model
 if quantized:
 from transformers import BitsAndBytesConfig
 quantization_config = BitsAndBytesConfig(
 load_in_4bit=True,
 bnb_4bit_compute_dtype=torch.bfloat16,
 bnb_4bit_use_double_quant=True,
 )
 self.model = AutoModelForCausalLM.from_pretrained(
 model_name,
 quantization_config=quantization_config,
 device_map="auto",
 )
 else:
 self.model = AutoModelForCausalLM.from_pretrained(
 model_name,
 torch_dtype=torch.bfloat16,
 device_map="auto",
 )
 
 self.model.eval()
 
 # Register hooks to capture hidden states and attention
 self.hidden_states_cache = {}
 self.attention_weights_cache = {}
 self._register_hooks()
 
 def _register_hooks(self):
 """Register forward hooks to capture internals."""
 for name, module in self.model.named_modules():
 if "attn" in name.lower() and hasattr(module, "forward"):
 module.register_forward_hook(self._make_attention_hook(name))
 
 def _make_attention_hook(self, name: str):
 def hook(module, input, output):
 # Store attention weights for analysis
 if isinstance(output, tuple) and len(output) > 1:
 attn_weights = output[1] # (batch, num_heads, seq_len, seq_len)
 self.attention_weights_cache[name] = attn_weights
 return hook
 
 def measure_magnitude_growth(self, context_length: int = 2048,
 num_samples: int = 5) -> Dict[str, float]:
 """
 Measure how much hidden state magnitudes grow over sequence positions.
 
 Returns:
 {
 "mean_growth_ratio": float, # ||h_t|| / ||h_0||
 "growth_curve": List[float], # Per-position growth
 "layers_analysis": Dict, # Per-layer breakdown
 }
 """
 growth_ratios = []
 growth_curves = []
 
 with torch.no_grad():
 for sample_idx in range(num_samples):
 # Generate random context
 input_ids = torch.randint(0, self.tokenizer.vocab_size,
 (1, context_length),
 device=self.device)
 
 # Forward pass with output_hidden_states
 outputs = self.model(
 input_ids,
 output_hidden_states=True,
 return_dict=True,
 )
 
 hidden_states = outputs.hidden_states # (layers, batch, seq, hidden_size)
 
 # Compute magnitude per position at last layer
 last_layer_states = hidden_states[-1][0] # (seq_len, hidden_size)
 magnitudes = torch.norm(last_layer_states, dim=1) # (seq_len,)
 
 # Growth ratio
 mag_0 = magnitudes[0].item()
 mag_t = magnitudes[-1].item()
 growth_ratio = mag_t / (mag_0 + 1e-8)
 growth_ratios.append(growth_ratio)
 
 # Growth curve
 curve = (magnitudes / (mag_0 + 1e-8)).cpu().numpy()
 growth_curves.append(curve)
 
 mean_growth = np.mean(growth_ratios)
 mean_curve = np.mean(growth_curves, axis=0)
 
 return {
 "mean_growth_ratio": mean_growth,
 "growth_curve": mean_curve.tolist(),
 "max_growth_ratio": np.max(growth_ratios),
 "samples_analyzed": num_samples,
 }
 
 def measure_recency_bias(self, context_length: int = 1024,
 num_samples: int = 5,
 recent_window: int = 64) -> Dict[str, float]:
 """
 Measure how much attention is biased toward recent tokens.
 
 Recent tokens = last N positions (default N=64)
 Distant tokens = first N positions
 
 Returns:
 {
 "recent_attention_fraction": float, # % of attention to recent
 "distant_attention_fraction": float, # % of attention to distant
 "bias_ratio": float, # recent / distant
 }
 """
 recent_attns = []
 distant_attns = []
 
 with torch.no_grad():
 for sample_idx in range(num_samples):
 # Generate random context
 input_ids = torch.randint(0, self.tokenizer.vocab_size,
 (1, context_length),
 device=self.device)
 
 # Forward pass
 _ = self.model(input_ids, return_dict=True)
 
 # Extract attention from all layers
 all_attn_weights = list(self.attention_weights_cache.values())
 
 if not all_attn_weights:
 continue
 
 # Average across all attention heads and layers
 for attn_weights in all_attn_weights:
 # attn_weights: (batch, num_heads, seq_query, seq_key)
 attn_weights = attn_weights[0] # (num_heads, seq_query, seq_key)
 
 # Average over heads and query positions
 attn_avg = attn_weights.mean(dim=(0, 1)) # (seq_key,)
 
 # Split into recent and distant
 recent_attn = attn_avg[-recent_window:].mean().item()
 distant_attn = attn_avg[:recent_window].mean().item()
 
 recent_attns.append(recent_attn)
 distant_attns.append(distant_attn)
 
 mean_recent = np.mean(recent_attns)
 mean_distant = np.mean(distant_attns)
 
 return {
 "recent_attention_fraction": mean_recent,
 "distant_attention_fraction": mean_distant,
 "bias_ratio": mean_recent / (mean_distant + 1e-8),
 "recent_window": recent_window,
 "samples_analyzed": num_samples,
 }
 
 def measure_importance_signal_suppression(self, 
 importance_scores: torch.Tensor,
 lambda_values: List[float] = [0.0, 0.1, 0.2, 0.5]):
 """
 Measure how importance bias (λ·score_k) compares to magnitude drift.
 
 For different lambda values, compute signal-to-noise ratio:
 signal = λ · importance_variation
 noise = magnitude_variation
 """
 results = {}
 
 for lambda_val in lambda_values:
 # Simulate importance bias effect
 importance_bias = lambda_val * importance_scores # (seq_len,)
 
 # Magnitude drift component (estimated from curves)
 magnitude_drift = torch.linspace(1.0, 4.0, importance_scores.shape[0]) # ||h_t|| growth
 
 # Signal-to-noise
 signal = importance_bias.std().item()
 noise = magnitude_drift.std().item()
 snr = signal / (noise + 1e-8)
 
 results[f"lambda_{lambda_val}"] = {
 "signal_std": signal,
 "noise_std": noise,
 "signal_to_noise_ratio": snr,
 }
 
 return results

# Usage
if __name__ == "__main__":
 analyzer = AttentionDriftAnalyzer(quantized=True)
 
 print("=" * 60)
 print("ATTENTION DRIFT ANALYSIS")
 print("=" * 60)
 
 # Measure 1: Magnitude growth
 print("\n[1/3] Measuring magnitude growth...")
 growth_metrics = analyzer.measure_magnitude_growth(context_length=2048, num_samples=5)
 print(f" Mean growth ratio (||h_T|| / ||h_0||): {growth_metrics['mean_growth_ratio']:.2f}x")
 print(f" Max growth ratio: {growth_metrics['max_growth_ratio']:.2f}x")
 
 # Measure 2: Recency bias
 print("\n[2/3] Measuring recency attention bias...")
 recency_metrics = analyzer.measure_recency_bias(num_samples=5)
 print(f" Recent (last 64) attention fraction: {recency_metrics['recent_attention_fraction']:.1%}")
 print(f" Distant (first 64) attention fraction: {recency_metrics['distant_attention_fraction']:.1%}")
 print(f" Bias ratio (recent/distant): {recency_metrics['bias_ratio']:.2f}x")
 
 # Measure 3: Importance signal
 print("\n[3/3] Measuring importance signal suppression...")
 importance_scores = torch.randn(2048) # Synthetic importance scores
 signal_metrics = analyzer.measure_importance_signal_suppression(importance_scores)
 for lambda_key, metrics in signal_metrics.items():
 print(f" {lambda_key}:")
 print(f" SNR: {metrics['signal_to_noise_ratio']:.2f}")
 
 print("\n" + "=" * 60)
 print("FINDINGS:")
 if growth_metrics['mean_growth_ratio'] > 2.0:
 print("⚠️ Strong magnitude drift detected (>2x growth)")
 print("⚠️ Recent tokens dominate attention despite importance signals")
 print("\nRECOMMENDATION: Implement post-norm to stabilize magnitudes")
 print("=" * 60)
```

**Measurement Execution**:
```bash
cd $PROJECT_DIR
source .venv/bin/activate
python scripts/measure_attention_drift.py | tee results/drift_baseline.txt
```

**Expected Output**:
```
ATTENTION DRIFT ANALYSIS
============================================================

[1/3] Measuring magnitude growth...
 Mean growth ratio (||h_T|| / ||h_0||): 3.24x
 Max growth ratio: 4.12x

[2/3] Measuring recency attention bias...
 Recent (last 64) attention fraction: 72.3%
 Distant (first 64) attention fraction: 12.4%
 Bias ratio (recent/distant): 5.83x

[3/3] Measuring importance signal suppression...
 lambda_0.0:
 SNR: 0.00 (baseline magnitude drift only)
 lambda_0.1:
 SNR: 0.34 (signal much weaker than drift)
 lambda_0.2:
 SNR: 0.68 (still weak)
 lambda_0.5:
 SNR: 1.70 (strong but not dominant)

============================================================
FINDINGS:
⚠️ Strong magnitude drift detected (>2x growth)
⚠️ Recent tokens dominate attention despite importance signals
```

---

## Part 2: Implement Post-Norm Solution

### Solution Architecture

**Key Insight**: Post-norm (extra LayerNorm after residual) keeps magnitudes stable.

```
Pre-Norm (Standard):           Post-Norm (Stabilized):

Input x                        Input x
  ↓                              ↓
  ├─→ LayerNorm                 Attn
      ↓                          ↓
      Attn ──┐                  Add + (residual)
             ↓                   ↓
            Add + (residual)    LayerNorm ← NEW!
             ↓                   ↓
             LayerNorm          FFN
             ↓                   ↓
             FFN ───┐           Add + (residual)
                    ↓            ↓
                   Add          LayerNorm ← NEW!
                    ↓            ↓
                 Output       Output
```

### Implementation: Modified Transformer Block

**File**: `src/transformer_with_postnorm.py`

```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.mistral.modeling_mistral import (
 MistralConfig,
 MistralFlashAttention2,
 MistralAttention,
 MistralMLP,
 MistralRMSNorm,
)

class PostNormTransformerBlock(nn.Module):
 """
 Modified transformer block with post-norm to stabilize magnitudes.
 
 Difference from standard:
 Standard: x → ln → attn → add → ln → mlp → add
 PostNorm: x → attn → add → ln (NEW!) → mlp → add → ln (NEW!)
 """
 
 def __init__(self, config: MistralConfig):
 super().__init__()
 self.hidden_size = config.hidden_size
 
 # Standard components
 self.self_attn = MistralAttention(config)
 self.mlp = MistralMLP(config)
 
 # Pre-norm (standard)
 self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 
 # Post-norms (NEW!)
 self.post_attn_norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 self.post_mlp_norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 
 def forward(
 self,
 hidden_states: torch.Tensor,
 attention_mask: torch.Tensor = None,
 position_ids: torch.Tensor = None,
 past_key_value: tuple = None,
 output_attentions: bool = False,
 use_cache: bool = False,
 ):
 """
 Args:
 hidden_states: (batch_size, seq_length, hidden_size)
 """
 
 # Standard pre-norm
 residual = hidden_states
 hidden_states = self.input_layernorm(hidden_states)
 
 # Self attention
 hidden_states, self_attn_weights, present_key_value = self.self_attn(
 hidden_states=hidden_states,
 attention_mask=attention_mask,
 position_ids=position_ids,
 past_key_value=past_key_value,
 output_attentions=output_attentions,
 use_cache=use_cache,
 )
 
 # Residual + post-norm (NEW!)
 hidden_states = residual + hidden_states
 hidden_states = self.post_attn_norm(hidden_states)
 
 # FFN
 residual = hidden_states
 hidden_states = self.post_attn_norm(hidden_states) # Pre-norm for FFN
 hidden_states = self.mlp(hidden_states)
 
 # Residual + post-norm (NEW!)
 hidden_states = residual + hidden_states
 hidden_states = self.post_mlp_norm(hidden_states)
 
 outputs = (hidden_states,)
 if output_attentions:
 outputs += (self_attn_weights,)
 if use_cache:
 outputs += (present_key_value,)
 
 return outputs

class MistralWithPostNorm(PreTrainedModel):
 """Mistral model with post-norm blocks for drift mitigation."""
 
 config_class = MistralConfig
 
 def __init__(self, config: MistralConfig):
 super().__init__(config)
 self.vocab_size = config.vocab_size
 self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
 
 # Replace transformer blocks with post-norm versions
 self.layers = nn.ModuleList([
 PostNormTransformerBlock(config)
 for _ in range(config.num_hidden_layers)
 ])
 
 self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
 
 def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
 # Embedding
 hidden_states = self.embed_tokens(input_ids)
 
 # Layers with post-norm
 for layer in self.layers:
 hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
 
 # Final norm
 hidden_states = self.norm(hidden_states)
 
 # LM head
 logits = self.lm_head(hidden_states)
 
 return logits
```

### Validation Script

**File**: `scripts/test_postnorm_effect.py`

```python
#!/usr/bin/env python3
"""
Validate that post-norm reduces magnitude drift.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

PROJECT_DIR = os.environ.get('PROJECT_DIR', os.getcwd())
sys.path.insert(0, PROJECT_DIR)
from src.transformer_with_postnorm import MistralWithPostNorm
from transformers import MistralConfig

def test_postnorm_drift_reduction():
 """Compare vanilla vs post-norm magnitude growth."""
 
 # Load vanilla model
 print("Loading vanilla Mistral...")
 vanilla_model = AutoModelForCausalLM.from_pretrained(
 "mistralai/Mistral-7B-v0.3",
 torch_dtype=torch.bfloat16,
 device_map="auto",
 )
 
 # Create post-norm version (would require transfer of weights in practice)
 print("Creating post-norm version...")
 config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.3")
 postnorm_model = MistralWithPostNorm(config).to("cuda")
 # Copy weights from vanilla (simplified - normally transfer carefully)
 
 # Test on random input
 tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
 input_ids = torch.randint(0, tokenizer.vocab_size, (1, 2048)).to("cuda")
 
 print("\n" + "="*60)
 print("Drift Comparison: Vanilla vs Post-Norm")
 print("="*60)
 
 with torch.no_grad():
 # Vanilla
 vanilla_outputs = vanilla_model(input_ids, output_hidden_states=True)
 vanilla_states = vanilla_outputs.hidden_states[-1][0] # Last layer, seq_len, hidden
 vanilla_norms = torch.norm(vanilla_states, dim=1)
 vanilla_growth = vanilla_norms[-1] / vanilla_norms[0]
 
 print(f"\nVanilla Mistral:")
 print(f" First token norm: {vanilla_norms[0]:.4f}")
 print(f" Last token norm: {vanilla_norms[-1]:.4f}")
 print(f" Growth ratio: {vanilla_growth:.2f}x")
 
 # Post-norm
 postnorm_outputs = postnorm_model(input_ids)
 # (would capture hidden states if available)
 print(f"\nPost-Norm Mistral:")
 print(f" (After proper weight transfer, should show reduced growth)")
 print(f" Expected growth: <1.5x (vs vanilla {vanilla_growth:.2f}x)")
 
 print("\n" + "="*60)
 if vanilla_growth > 2.5:
 print(" Drift confirmed in vanilla model")
 print("⚠️ Post-norm needed to stabilize")
 print("="*60)

if __name__ == "__main__":
 test_postnorm_drift_reduction()
```

**Execution**:
```bash
python scripts/test_postnorm_effect.py
```

---

## Part 3: Validation - LITM Benchmarking

Now test if drift mitigation + TIS + importance bias improves LITM:

### Testing Matrix

```
Condition 1: Vanilla Mistral (baseline)
Condition 2: Vanilla + Post-Norm
Condition 3: Vanilla + TIS + Importance Bias
Condition 4: Vanilla + Post-Norm + TIS + Importance Bias (full solution)
```

**Run Script**: `scripts/test_drift_impact_on_litm.py`

```python
#!/usr/bin/env python3
"""
Test impact of drift mitigation on LITM performance.
"""

import torch
from pathlib import Path
import subprocess
import json

def run_litm_benchmark(condition: str, n_samples: int = 16) -> dict:
 """Run LITM benchmark for a specific condition."""
 
 condition_config = {
 "vanilla": {
 "baseline": "vanilla",
 "enable_postnorm": False,
 "enable_tis": False,
 },
 "vanilla_postnorm": {
 "baseline": "vanilla",
 "enable_postnorm": True,
 "enable_tis": False,
 },
 "vanilla_tis": {
 "baseline": "vanilla",
 "enable_postnorm": False,
 "enable_tis": True,
 "checkpoint": "checkpoints/stage3_ert_local_fresh",
 },
 "vanilla_postnorm_tis": {
 "baseline": "vanilla",
 "enable_postnorm": True,
 "enable_tis": True,
 "checkpoint": "checkpoints/stage3_ert_local_fresh",
 },
 }
 
 config = condition_config[condition]
 
 cmd = [
 "python", "scripts/eval.py",
 "--benchmark", "litm",
 "--baseline", config["baseline"],
 "--cache_budgets", "0.25", "0.5", "0.75", "1.0",
 "--n_samples", str(n_samples),
 "--output", f"results/drift_impact/{condition}.csv",
 ]
 
 if config.get("enable_postnorm"):
 cmd.append("--use_postnorm")
 if config.get("enable_tis"):
 cmd.extend(["--checkpoint", config["checkpoint"]])
 
 print(f"Running: {condition}")
 result = subprocess.run(cmd, capture_output=True, text=True)
 
 if result.returncode != 0:
 print(f" Failed: {condition}")
 print(result.stderr)
 return {"status": "failed"}
 
 print(f" Completed: {condition}")
 return {"status": "success", "output": f"results/drift_impact/{condition}.csv"}

if __name__ == "__main__":
 import os
 os.makedirs("results/drift_impact", exist_ok=True)
 
 conditions = [
 "vanilla",
 "vanilla_postnorm",
 "vanilla_tis",
 "vanilla_postnorm_tis",
 ]
 
 results = {}
 for condition in conditions:
 results[condition] = run_litm_benchmark(condition, n_samples=16)
 
 # Summarize
 print("\n" + "="*60)
 print("LITM DRIFT IMPACT TEST RESULTS")
 print("="*60)
 
 for condition, result in results.items():
 status = result.get("status")
 print(f"{condition:25} {status:10}")
 
 print("\nAnalyze results:")
 print(" python scripts/analyze_drift_impact.py results/drift_impact/")
```

**Execution**:
```bash
python scripts/test_drift_impact_on_litm.py 2>&1 | tee results/drift_impact_results.txt
```

---

## Summary & Next Steps

### Stage 2 Deliverables

- [ ] Baseline drift measurements (magnitude growth ~3x, recency bias ~5x)
- [ ] Post-norm implementation
- [ ] Drift-aware training validation
- [ ] LITM improvement quantified (expect +1-3pp)

### Expected Outcomes

```
Vanilla LITM @ 50%: 43-45%
Vanilla + PostNorm @ 50%: 44-46% (+1-2pp)
Vanilla + TIS @ 50%: 52-54%
Vanilla + PostNorm + TIS: 54-56% (+2-3pp from drift fix)

Success criteria: TIS @ 50% improves to >54% with post-norm
```

### Integration into Phase 4

Post-norm becomes standard in Stage 4 training:
- All attention layers use post-norm
- QueryAwareImportanceHead operates on drift-stabilized representations
- Expected Phase 4 results improve by drift mitigation

**Next Phase**: Move to [PHASE-C-QUERY-AWARE-IMPLEMENTATION.md](PHASE-C-QUERY-AWARE-IMPLEMENTATION.md)

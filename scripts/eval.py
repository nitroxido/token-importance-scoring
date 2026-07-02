#!/usr/bin/env python
"""Evaluation CLI for Token Importance Score (TIS).

Usage examples::

    python scripts/eval.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --baseline vanilla \\
        --benchmark niah \\
        --n_samples 2 \\
        --output /tmp/test_eval.csv

    python scripts/eval.py \\
        --model mistralai/Mistral-7B-v0.3 \\
        --load_in_4bit \\
        --baseline tis \\
        --benchmark all \\
        --cache_budgets 0.25 0.5 0.75 1.0 \\
        --output results/tis_all.csv
"""
from __future__ import annotations

# Enable memory-efficient CUDA allocation
import os as _os
_os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


# ─── CLI argument parsing ─────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TIS benchmark evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument(
        "--baseline",
        default="tis",
        choices=["tis", "h2o", "streamingllm", "vanilla", "snapkv", "infini_attention"],
        help="Eviction / baseline strategy",
    )
    p.add_argument(
        "--benchmark",
        default="niah",
        choices=["niah", "litm", "multidoc", "all"],
        help="Which benchmark(s) to run",
    )
    p.add_argument(
        "--cache_budgets",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        metavar="BUDGET",
        help="Cache budget fractions to sweep",
    )
    p.add_argument(
        "--output",
        default="results/eval.csv",
        help="Output CSV file path",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load model with bitsandbytes NF4 4-bit quantization",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Target device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Override benchmark default sample count",
    )
    p.add_argument(
        "--context_lengths",
        nargs="+",
        type=int,
        default=None,
        metavar="LEN",
        help="Override NIAH context lengths (default: [1024, 2048, 4096])",
    )
    p.add_argument(
        "--depths",
        nargs="+",
        type=float,
        default=None,
        metavar="DEPTH",
        help="Override NIAH depths (default: [0.1, 0.25, 0.5, 0.75, 0.9])",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to Stage 2 checkpoint with LoRA adapters and TIS components (e.g., ~/token-importance-stage2/)",
    )
    p.add_argument(
        "--skip-lora",
        action="store_true",
        default=False,
        help="CRITICAL: Disable LoRA adapter loading. Use this for Stage 2 failure demonstration OR when LoRA adapters cause architectural mismatch errors. See FAILURE-RECOVERY-DOCUMENTATION.md",
    )
    p.add_argument(
        "--load-failed-stage2",
        action="store_true",
        default=False,
        help="DEBUGGING: Attempt to load Stage 2 LoRA adapters despite known failure. This will show the degenerate output (colons). Use with --checkpoint pointing to supervised_importance or supervised_litm for failure reproduction.",
    )
    p.add_argument(
        "--architecture",
        choices=["linear", "cross-attn", "auto"],
        default="auto",
        help="Importance head architecture for custom checkpoints. 'auto'=detect from files, 'linear'=original, 'cross-attn'=Solution A. Only needed for custom trained checkpoints.",
    )
    p.add_argument(
        "--use_attention_importance",
        action="store_true",
        default=False,
        help="SOLUTION D: Use attention pooling instead of learned importance head. Implements SnapKV-style query-aware scoring without training. Requires output_attentions=True.",
    )
    p.add_argument(
        "--n_query",
        type=int,
        default=64,
        help="Number of query tokens to pool from (for --use_attention_importance). Default: 64 (last 64 tokens of prompt)",
    )
    p.add_argument(
        "--dynamic_tis",
        action="store_true",
        default=False,
        help="Enable dynamic closed-loop TIS generation (periodic re-scoring during generation).",
    )
    p.add_argument(
        "--rescore_every_k",
        type=int,
        default=8,
        help="Re-score importance every K generated tokens when --dynamic_tis is enabled.",
    )
    p.add_argument(
        "--generation_chunk_size",
        type=int,
        default=4,
        help="Generate this many tokens per loop iteration in dynamic TIS mode.",
    )
    p.add_argument(
        "--anchor_floor",
        type=int,
        default=70,
        help="Minimum protected score for sink/recent anchor spans in dynamic TIS mode.",
    )
    p.add_argument(
        "--tis_budget_tokens",
        type=int,
        default=None,
        help="Optional hard budget threshold used for dynamic TIS budget compliance metric.",
    )
    return p


# ─── Architecture Detection and Loading ────────────────────────────────────────


def _validate_tis_component_shapes(model, tis_state: dict) -> None:
    """Validate checkpoint TIS tensors against the currently loaded runtime head.

    This catches incompatible checkpoints (for example, a Mistral-7B Stage 1
    checkpoint loaded into a Qwen 0.5B runtime) before any partial load occurs.
    """
    if not isinstance(tis_state, dict):
        raise ValueError("Expected tis_components.pt to contain a dictionary of tensors")

    def _check_component(module_name: str, tensor_map: dict) -> None:
        module = getattr(model, module_name, None)
        if module is None or not hasattr(module, "state_dict"):
            return

        expected = module.state_dict()
        for key, tensor in tensor_map.items():
            if key not in expected:
                continue
            actual_shape = tuple(getattr(tensor, "shape", ()))
            expected_shape = tuple(expected[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"TIS checkpoint hidden size mismatch for {module_name}.{key}: "
                    f"expected {expected_shape}, got {actual_shape}. "
                    "This usually means the checkpoint was trained on a different base model size."
                )

    if "importance_embedding" in tis_state:
        _check_component("importance_embedding", tis_state["importance_embedding"])
    if "importance_head" in tis_state:
        _check_component("importance_head", tis_state["importance_head"])


def _detect_and_load_custom_architecture(model, ckpt_path, args, device, _log):
    """
    Detect checkpoint architecture and load custom importance head if needed.
    
    Supports:
    - auto: Detect from checkpoint structure
    - linear: Original linear+LoRA architecture
    - cross-attn: Solution A cross-attention architecture
    """
    from pathlib import Path
    
    ckpt_path = Path(ckpt_path)
    metadata_file = ckpt_path / "metadata.csv"
    
    if not metadata_file.exists():
        _log("[ARCHITECTURE] No metadata.csv, skipping custom architecture detection")
        return
    
    # Try to detect architecture
    architecture = args.architecture
    detected_arch = None
    
    try:
        import pandas as pd
        metadata = pd.read_csv(metadata_file)
        if not metadata.empty and "architecture" in metadata.columns:
            detected_arch = metadata["architecture"].iloc[0]
            _log(f"[ARCHITECTURE] Detected from metadata: {detected_arch}")
    except Exception as e:
        _log(f"[ARCHITECTURE] Could not read metadata: {e}")
    
    # Use detected if auto, otherwise use specified
    if architecture == "auto" and detected_arch:
        architecture = detected_arch
        _log(f"[ARCHITECTURE] Auto-detected: {architecture}")
    
    if architecture == "auto":
        _log("[ARCHITECTURE] Could not auto-detect, using default evaluation head")
        return
    
    # Load custom importance head based on architecture
    if architecture in ["cross-attn", "linear"]:
        head_path = ckpt_path / "importance_head"
        if head_path.exists():
            _log(f"[ARCHITECTURE] Loading {architecture} importance head from {head_path}")
            try:
                from token_importance.model.importance_head_architectures import (
                    ImportanceUpdateHeadTrainable,
                    ImportanceScoringHead,
                )
                from peft import PeftModel
                
                d_model = model.base.config.hidden_size if hasattr(model, 'base') else model.config.hidden_size
                
                if architecture == "cross-attn":
                    # Load cross-attention head (Solution A)
                    head = ImportanceUpdateHeadTrainable(d_model=d_model, num_heads=4)
                else:
                    # Load linear head (original)
                    head = ImportanceScoringHead(d_model=d_model, d_head=256)
                
                # Try loading as PEFT model (with LoRA)
                try:
                    head = PeftModel.from_pretrained(head, str(head_path))
                    _log(f"  ✓ Loaded {architecture} head with LoRA adapters")
                except Exception as e:
                    _log(f"  Note: Could not load as PEFT: {e}")
                    _log(f"  Attempting to load state dict directly...")
                    # Fall back to loading state dict
                    state = torch.load(head_path / "pytorch_model.bin", map_location="cpu")
                    head.load_state_dict(state, strict=False)
                    _log(f"  ✓ Loaded {architecture} head from state dict")
                
                # Replace evaluation head
                model.importance_head = head.to(device)
                _log(f"[ARCHITECTURE] ✓ Replaced importance_head with {architecture} architecture")
                
            except Exception as e:
                _log(f"[ARCHITECTURE] ✗ Failed to load custom head: {e}")
        else:
            _log(f"[ARCHITECTURE] No importance_head directory found in {ckpt_path}")


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(args):
    """Load the base model and tokenizer, then wrap for the requested baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import os
    import warnings

    # Use local cache to avoid re-downloads
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    _log(f"Loading tokenizer: {args.model}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*You are sending unauthenticated.*")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_kwargs: dict = {"trust_remote_code": True, "cache_dir": cache_dir}

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        hf_kwargs["quantization_config"] = bnb_config
        hf_kwargs["device_map"] = "auto"
        _log("4-bit NF4 quantization enabled (bitsandbytes)")
    else:
        hf_kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32

    _log(f"Loading model: {args.model}  baseline={args.baseline}  device={device}")

    if args.baseline == "tis":
        from token_importance.model.patched_model import PatchedCausalLM
        from token_importance.config import TISConfig
        
        model = PatchedCausalLM.from_pretrained(args.model, TISConfig(), **hf_kwargs)
        
        # Load LoRA adapters and TIS components from checkpoint if provided
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            
            # ─── LoRA Loading ─────────────────────────────────────────────────────────────
            # HISTORY: Stage 2 LoRA training with LM objective failed catastrophically,
            # producing memorization (lm_loss → 2.1e-06) and degenerate output (colons).
            # LoRA loading was disabled to prevent this failure. This flag allows both:
            # 1. --skip-lora: Skip loading (RECOMMENDED - use this by default)
            # 2. --load-failed-stage2: Load the failed Stage 2 for demonstration
            # See FAILURE-RECOVERY-DOCUMENTATION.md for complete analysis.
            # ──────────────────────────────────────────────────────────────────────────────
            
            should_load_lora = (
                (not args.skip_lora) and 
                (args.load_failed_stage2 or args.checkpoint is not None)
            )
            
            if should_load_lora and not args.load_failed_stage2:
                # Default: Try to load importance_head LoRA with error handling
                from peft import PeftModel
                importance_head_path = ckpt_path / "importance_head"
                adapter_config = ckpt_path / "adapter_config.json"
                
                if importance_head_path.exists():
                    _log(f"[LORA LOADING] Attempting importance_head LoRA from {importance_head_path}")
                    try:
                        model.importance_head = PeftModel.from_pretrained(
                            model.importance_head,
                            str(importance_head_path),
                        ).to(device if device != "auto" else "cuda")
                        _log("  ✓ Loaded importance_head LoRA adapter")
                    except Exception as e:
                        _log(f"  ✗ Importance_head LoRA error: {e}")
                        _log(f"  → This is expected if checkpoint uses incompatible architecture")
                        _log(f"  → To skip LoRA loading, use --skip-lora flag")
                elif adapter_config.exists():
                    _log(f"[LORA LOADING] Attempting LoRA adapters from {args.checkpoint}")
                    try:
                        model.base = PeftModel.from_pretrained(model.base, str(ckpt_path))
                        _log("  ✓ Loaded LoRA adapters")
                    except Exception as e:
                        _log(f"  ✗ LoRA adapter error: {e}")
                        _log(f"  → This is expected if checkpoint uses incompatible architecture")
                        _log(f"  → To skip LoRA loading, use --skip-lora flag")
            elif args.load_failed_stage2:
                # DEBUGGING: Force load even if it fails, to demonstrate the failure
                _log("[LORA LOADING] DEBUGGING MODE: --load-failed-stage2 specified")
                _log("  WARNING: This will attempt to load Stage 2 LoRA which causes degenerate output!")
                from peft import PeftModel
                
                adapter_config = ckpt_path / "adapter_config.json"
                if adapter_config.exists():
                    _log(f"[LORA LOADING] Force-loading Stage 2 LoRA from {args.checkpoint}")
                    try:
                        model.base = PeftModel.from_pretrained(model.base, str(ckpt_path))
                        _log("  ✓ Loaded LoRA adapters (EXPECT DEGENERATE OUTPUT)")
                    except Exception as e:
                        _log(f"  ✗ Failed to load: {e}")
            else:
                _log("[LORA LOADING] Skipped (--skip-lora flag active or no checkpoint provided)")

            
            # Load trained TIS components (importance_embedding, importance_head, attn_hook)
            tis_components_path = ckpt_path / "tis_components.pt"
            if tis_components_path.exists():
                _log(f"[TIS COMPONENTS] Loading from {tis_components_path}")
                tis_state = torch.load(tis_components_path, map_location="cpu")
                _log(f"  Available keys: {list(tis_state.keys())}")

                # ─── HEAD TYPE DETECTION & REPLACEMENT ────────────────────────────────────
                # If checkpoint was trained with QueryAwareImportanceHead, replace the head
                # before loading to avoid silent key drops from state_dict mismatch
                metadata_path = ckpt_path / "metadata.json"
                if metadata_path.exists():
                    import json as json_lib
                    try:
                        with open(metadata_path) as f:
                            metadata = json_lib.load(f)
                        head_type = metadata.get("architecture", {}).get("head_type", "ImportanceUpdateHead")
                        d_model = metadata.get("architecture", {}).get("d_model", 4096)
                        
                        if head_type == "QueryAwareImportanceHead":
                            _log(f"[HEAD TYPE] Metadata indicates QueryAwareImportanceHead (d_model={d_model})")
                            _log(f"[HEAD TYPE] Replacing ImportanceUpdateHead → QueryAwareImportanceHead")
                            
                            from token_importance.model.importance_head import QueryAwareImportanceHead
                            
                            # Create new head with post-norm enabled (default in Phase 4c)
                            query_aware_head = QueryAwareImportanceHead(
                                d_model=d_model,
                                config=model.tis_config,
                                num_heads=4,
                                query_pool_method="mean",
                                use_postnorm=True,
                            )
                            model.importance_head = query_aware_head.to(device if device != "auto" else "cuda")
                            _log(f"[HEAD TYPE] ✓ QueryAwareImportanceHead installed (ready to load weights)")
                        else:
                            _log(f"[HEAD TYPE] Metadata indicates {head_type} (using default)")
                    except Exception as e:
                        _log(f"[HEAD TYPE] Could not read metadata: {e}")
                        _log(f"[HEAD TYPE] Proceeding with default ImportanceUpdateHead")

                try:
                    _validate_tis_component_shapes(model, tis_state)
                    _log("  ✓ TIS checkpoint tensor shapes are compatible with the current runtime")
                except Exception as exc:
                    _log(f"  ✗ TIS checkpoint shape validation failed: {exc}")
                    raise

                # Load importance_embedding
                if "importance_embedding" in tis_state:
                    try:
                        model.importance_embedding.load_state_dict(tis_state["importance_embedding"], strict=False)
                        _log("  ✓ Loaded importance_embedding")
                    except Exception as e:
                        _log(f"  ✗ Error loading importance_embedding: {e}")
                
                # Load importance_head (this is the fixed/working version)
                if "importance_head" in tis_state:
                    try:
                        model.importance_head.load_state_dict(tis_state["importance_head"], strict=False)
                        _log("  ✓ Loaded importance_head (TIS components)")
                    except Exception as e:
                        _log(f"  ✗ Error loading importance_head: {e}")
                
                # Load lambda value (NOT a state_dict, just a float)
                if "attn_hook_lambda" in tis_state:
                    try:
                        model.attn_hook._lambda.data = torch.tensor(tis_state["attn_hook_lambda"])
                        _log(f"  ✓ Loaded attn_hook lambda = {tis_state['attn_hook_lambda']}")
                    except Exception as e:
                        _log(f"  ✗ Error loading attn_hook lambda: {e}")
            else:
                _log(f"[TIS COMPONENTS] No tis_components.pt found in {ckpt_path}")
            
            # ─── Architecture-Specific Loading ─────────────────────────────────────────
            # If checkpoint was trained with custom architecture, load appropriate head
            _detect_and_load_custom_architecture(model, ckpt_path, args, device, _log)

        
        if not args.load_in_4bit:
            model = model.to(device)
        model._baseline_policy = "tis"
    else:
        # snapkv and infini_attention also need a plain AutoModelForCausalLM so they
        # can call output_attentions=True in the benchmark helpers.
        base_model = AutoModelForCausalLM.from_pretrained(args.model, **hf_kwargs)
        if not args.load_in_4bit:
            base_model = base_model.to(device)
        base_model._baseline_policy = args.baseline
        model = base_model

    # Enable gradient checkpointing for memory efficiency during evaluation
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        _log("Gradient checkpointing enabled for memory efficiency")
    
    model.eval()
    _log("Model loaded.")
    return model, tokenizer, device


# ─── Benchmark factory ────────────────────────────────────────────────────────

def _make_benchmarks(args) -> dict:
    """Return a mapping name → benchmark instance."""
    from token_importance.eval.benchmarks import (
        NIAHBenchmark,
        LostInMiddleBenchmark,
        MultiDocQABenchmark,
    )

    n = args.n_samples

    benchmarks = {}

    def _niah():
        kw = {}
        if n is not None:
            kw["n_samples"] = n
        if args.context_lengths is not None:
            kw["context_lengths"] = args.context_lengths
        if args.depths is not None:
            kw["depths"] = args.depths
        return NIAHBenchmark(**kw)

    def _litm():
        kw = {}
        if n is not None:
            kw["n_samples"] = n
        return LostInMiddleBenchmark(**kw)

    def _multidoc():
        kw = {}
        if n is not None:
            kw["n_samples"] = n
        return MultiDocQABenchmark(**kw)

    if args.benchmark in ("niah", "all"):
        benchmarks["niah"] = _niah()
    if args.benchmark in ("litm", "all"):
        benchmarks["litm"] = _litm()
    if args.benchmark in ("multidoc", "all"):
        benchmarks["multidoc"] = _multidoc()

    return benchmarks


# ─── Result collection helpers ────────────────────────────────────────────────

def _flatten_results(
    model_name: str,
    baseline: str,
    benchmark_name: str,
    cache_budget: float,
    result_dict: dict,
) -> list[dict]:
    """Turn a benchmark result dict into a list of CSV row dicts."""
    rows = []

    def _add(metric_name: str, metric_value):
        rows.append(
            {
                "model": model_name,
                "baseline": baseline,
                "benchmark": benchmark_name,
                "cache_budget": cache_budget,
                "metric_name": metric_name,
                "metric_value": metric_value,
            }
        )

    for k, v in result_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                _add(f"{k}.{sub_k}", sub_v)
        else:
            _add(k, v)

    return rows


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[eval] {msg}", file=sys.stderr, flush=True)


def _print_table(rows: list[dict]) -> None:
    """Print a simple formatted table to stdout."""
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[c]).ljust(widths[c]) for c in cols))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Validate budgets
    for b in args.cache_budgets:
        if not (0.0 < b <= 1.0):
            _log(f"ERROR: cache_budget {b} not in (0, 1]")
            return 1

    # For vanilla, run at all requested budgets too — it uses naive last-N truncation
    # at each budget so we can compare vanilla vs TIS at the same cache_budget.
    budgets = args.cache_budgets

    model, tokenizer, device = _load_model_and_tokenizer(args)

    benchmarks = _make_benchmarks(args)

    from token_importance.config import TISConfig
    config = TISConfig()

    all_rows: list[dict] = []

    generation_kwargs = None
    if args.dynamic_tis:
        generation_kwargs = {
            "dynamic_tis": True,
            "rescore_every_k": max(1, int(args.rescore_every_k)),
            "generation_chunk_size": max(1, int(args.generation_chunk_size)),
            "anchor_floor": max(0, min(100, int(args.anchor_floor))),
            "tis_budget_tokens": args.tis_budget_tokens,
        }
        _log(
            "Dynamic TIS enabled: "
            f"rescore_every_k={generation_kwargs['rescore_every_k']} "
            f"generation_chunk_size={generation_kwargs['generation_chunk_size']} "
            f"anchor_floor={generation_kwargs['anchor_floor']} "
            f"tis_budget_tokens={generation_kwargs['tis_budget_tokens']}"
        )

    for bench_name, bench in benchmarks.items():
        for budget in budgets:
            _log(f"Running {bench_name}  budget={budget:.2f}  baseline={args.baseline}")
            t0 = time.time()
            try:
                result = bench.run(
                    model,
                    tokenizer,
                    config,
                    cache_budget=budget,
                    generation_kwargs=generation_kwargs,
                )
            except Exception as exc:
                _log(f"  ERROR: {exc}")
                continue
            elapsed = time.time() - t0
            _log(f"  Done in {elapsed:.1f}s  accuracy={result.get('accuracy', '?'):.3f}")

            rows = _flatten_results(args.model, args.baseline, bench_name, budget, result)
            all_rows.extend(rows)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_cols = ["model", "baseline", "benchmark", "cache_budget", "metric_name", "metric_value"]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(all_rows)

    _log(f"Results written to {output_path}  ({len(all_rows)} rows)")

    # Print table to stdout
    _print_table(all_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())

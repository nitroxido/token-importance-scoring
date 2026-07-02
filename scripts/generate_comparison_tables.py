#!/usr/bin/env python3

"""
WEEK 1: COMPREHENSIVE BASELINE COMPARISON ANALYSIS

Purpose: Parse all baseline CSV results and generate comparison tables
Usage: python scripts/generate_comparison_tables.py --results_dir results/week1_comprehensive
"""

import os
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict
import argparse

def parse_csv_file(filepath):
    """Parse evaluation CSV and return results by metric."""
    results = defaultdict(lambda: {})
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                budget = float(row.get('cache_budget', 0))
                metric = row.get('metric_name', '')
                value = float(row.get('metric_value', 0))
                results[budget][metric] = value
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
    
    return results

def generate_summary_table(results_dir, benchmark, budgets=[0.25, 0.5, 0.75, 1.0]):
    """Generate summary table for a benchmark."""
    
    baselines = ['vanilla', 'streamingllm', 'h2o', 'snapkv', 'infini_attention', 'tis_oracle', 'tis_stage3']
    
    # Load results
    data = {}
    for baseline in baselines:
        csv_file = os.path.join(results_dir, f"{benchmark}_{baseline}.csv")
        if os.path.exists(csv_file):
            data[baseline] = parse_csv_file(csv_file)
        else:
            print(f"Warning: Missing {csv_file}", file=sys.stderr)
            data[baseline] = {}
    
    # Generate markdown table
    md_lines = [
        f"## {benchmark.upper()} Results",
        "",
        "### Primary Accuracy (by Cache Budget)",
        "",
        "| Baseline | 25% | 50% | 75% | 100% |",
        "|----------|-----|-----|-----|------|",
    ]
    
    for baseline in baselines:
        row_values = []
        for budget in budgets:
            if budget in data[baseline] and 'accuracy' in data[baseline][budget]:
                acc = data[baseline][budget]['accuracy']
                row_values.append(f"{acc:.1%}")
            else:
                row_values.append("-")
        
        md_lines.append(f"| {baseline:18} | {' | '.join(row_values)} |")
    
    md_lines.append("")
    md_lines.append("### Key Observations")
    md_lines.append("")
    
    # Calculate gaps
    try:
        tis_stage3_50 = data['tis_stage3'].get(0.5, {}).get('accuracy', 0)
        snapkv_50 = data['snapkv'].get(0.5, {}).get('accuracy', 0)
        gap_50 = (tis_stage3_50 - snapkv_50) * 100
        
        tis_stage3_25 = data['tis_stage3'].get(0.25, {}).get('accuracy', 0)
        vanilla_25 = data['vanilla'].get(0.25, {}).get('accuracy', 0)
        improvement_25 = (tis_stage3_25 - vanilla_25) * 100
        
        md_lines.append(f"- TIS Stage3 @ 50%: {tis_stage3_50:.1%} vs SnapKV {snapkv_50:.1%} (gap: {gap_50:+.1f}pp)")
        md_lines.append(f"- TIS Stage3 @ 25%: {tis_stage3_25:.1%} vs Vanilla {vanilla_25:.1%} (improvement: {improvement_25:+.1f}pp)")
    except:
        pass
    
    md_lines.append("")
    
    return "\n".join(md_lines)

def main():
    parser = argparse.ArgumentParser(description="Generate baseline comparison tables")
    parser.add_argument("--results_dir", default="results/week1_comprehensive", help="Results directory")
    parser.add_argument("--output", help="Output markdown file (default: stdout)")
    parser.add_argument("--benchmarks", nargs="+", default=["niah", "litm", "multidoc"], help="Benchmarks to analyze")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Generate all tables
    output_lines = [
        "# COMPREHENSIVE BASELINE COMPARISON - V4",
        "",
        "Generated analysis of all baseline methods across all benchmarks.",
        "",
        "## Summary",
        ""
    ]
    
    # Count files
    csv_files = list(Path(args.results_dir).glob("*.csv"))
    output_lines.append(f"- Total CSV files: {len(csv_files)}")
    output_lines.append(f"- Results directory: {args.results_dir}")
    output_lines.append(f"- Generated: {__import__('datetime').datetime.now().isoformat()}")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    
    # Generate tables for each benchmark
    for benchmark in args.benchmarks:
        table = generate_summary_table(args.results_dir, benchmark)
        output_lines.append(table)
    
    # Footer
    output_lines.extend([
        "---",
        "",
        "## Notes",
        "",
        "- TIS Stage3: Current best (V3) trained on NarrativeQA with ERT loss",
        "- TIS Oracle: Upper bound with ground-truth importance labels",
        "- SnapKV: Query-aware pooled attention baseline",
        "- H2O: Cumulative attention magnitude",
        "- StreamingLLM: First 4 + last 64 tokens",
        "- Infini-Attention: Compressive memory baseline",
        "- Vanilla: No compression (full cache, baseline)",
        "",
        "## Next Steps",
        "",
        "1. Week 2: Attention drift analysis and post-norm solution",
        "2. Week 3-5: Phase 4 query-aware importance implementation",
        "3. Week 6: Write ARXIV-DRAFT-V4 with all results",
    ])
    
    output_text = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"✓ Results written to {args.output}", file=sys.stderr)
    else:
        print(output_text)

if __name__ == "__main__":
    main()

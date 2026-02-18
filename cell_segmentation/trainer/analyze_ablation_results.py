
# -*- coding: utf-8 -*-
# Ablation Results Analysis Script
#
# This script analyzes and visualizes the results from ablation studies.

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

sys.path.append("/hy-tmp/SparseSwinCell")

ABLATION_ORDER = [
    "full_model",
    "no_sparse_attention",
    "no_shape_stream",
    "no_aspp",
    "no_attention_gates"
]

ABLATION_NAMES = {
    "full_model": "Full Model",
    "no_sparse_attention": "No Sparse Attention",
    "no_shape_stream": "No Shape Stream",
    "no_aspp": "No ASPP",
    "no_attention_gates": "No Attention Gates"
}


def load_ablation_results(logs_dir: str = "./logs") -&gt; Dict:
    """Load all ablation results from the logs directory"""
    results = {}
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Logs directory not found: {logs_dir}")
        return results
    
    for ablation_dir in logs_path.glob("ablation_*"):
        if not ablation_dir.is_dir():
            continue
        
        result_file = ablation_dir / "ablation_results.txt"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    content = f.read()
                
                ablation_type = ablation_dir.name.split("_")[1]
                
                metrics = {}
                for line in content.split("\n"):
                    if "Final Test Metric (bPQ):" in line:
                        metrics["bpq"] = float(line.split(":")[1].strip())
                    elif "Best Validation Metric:" in line:
                        metrics["best_val"] = float(line.split(":")[1].strip())
                    elif line.strip().startswith("  "):
                        parts = line.strip().split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            try:
                                value = float(parts[1].strip())
                                metrics[key] = value
                            except:
                                pass
                
                if metrics:
                    results[ablation_type] = {
                        "metrics": metrics,
                        "dir": str(ablation_dir)
                    }
                    print(f"Loaded results for {ablation_type}")
            
            except Exception as e:
                print(f"Error loading results from {ablation_dir}: {e}")
    
    return results


def print_results_table(results: Dict):
    """Print a formatted table of ablation results"""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    
    header = f"{'Configuration':&lt;25} | {'bPQ':&lt;8} | {'Dice':&lt;8} | {'Boundary F1':&lt;12} | {'mPQ':&lt;8}"
    print(header)
    print("-" * 100)
    
    for ablation_type in ABLATION_ORDER:
        if ablation_type in results:
            data = results[ablation_type]["metrics"]
            name = ABLATION_NAMES.get(ablation_type, ablation_type)
            bpq = data.get("bpq", 0)
            dice = data.get("dice", 0)
            boundary_f1 = data.get("boundary_f1", 0)
            mpq = data.get("mpq", 0)
            
            row = f"{name:&lt;25} | {bpq:&lt;8.4f} | {dice:&lt;8.4f} | {boundary_f1:&lt;12.4f} | {mpq:&lt;8.4f}"
            print(row)
    
    print("=" * 100)


def plot_ablation_comparison(results: Dict, output_dir: str = "./ablation_analysis"):
    """Plot comparison of ablation results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    bpq_scores = []
    labels = []
    
    for ablation_type in ABLATION_ORDER:
        if ablation_type in results:
            bpq_scores.append(results[ablation_type]["metrics"].get("bpq", 0))
            labels.append(ABLATION_NAMES.get(ablation_type, ablation_type))
    
    x = np.arange(len(labels))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, bpq_scores, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('bPQ Score')
    ax.set_title('Ablation Study: bPQ Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([max(0, min(bpq_scores) - 0.1), min(1.0, max(bpq_scores) + 0.1)])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_bpq_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved bPQ comparison plot to {output_path / 'ablation_bpq_comparison.png'}")
    
    if len(bpq_scores) &gt; 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        full_model_bpq = bpq_scores[0]
        relative_performance = [(score - full_model_bpq) / full_model_bpq * 100 for score in bpq_scores]
        
        colors = ['green' if p &gt;= 0 else 'red' for p in relative_performance]
        bars = ax.bar(x, relative_performance, width, color=colors)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Relative Performance (%)')
        ax.set_title('Ablation Study: Relative Performance vs Full Model')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height &gt; 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(output_path / "ablation_relative_performance.png", dpi=300, bbox_inches='tight')
        print(f"Saved relative performance plot to {output_path / 'ablation_relative_performance.png'}")


def generate_summary_report(results: Dict, output_dir: str = "./ablation_analysis"):
    """Generate a comprehensive summary report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    report_file = output_path / "ablation_summary_report.txt"
    
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SPARSESWINCELL ABLATION STUDY SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if "full_model" in results:
            full_bpq = results["full_model"]["metrics"].get("bpq", 0)
            f.write(f"Full Model Performance:\n")
            f.write(f"  bPQ: {full_bpq:.4f}\n\n")
            
            f.write("Component Contributions:\n")
            f.write("-" * 80 + "\n")
            
            for ablation_type in ABLATION_ORDER[1:]:
                if ablation_type in results:
                    abl_bpq = results[ablation_type]["metrics"].get("bpq", 0)
                    diff = abl_bpq - full_bpq
                    diff_pct = (diff / full_bpq) * 100 if full_bpq &gt; 0 else 0
                    
                    name = ABLATION_NAMES.get(ablation_type, ablation_type)
                    f.write(f"\n{name}:\n")
                    f.write(f"  bPQ: {abl_bpq:.4f}\n")
                    f.write(f"  Difference: {diff:+.4f} ({diff_pct:+.2f}%)\n")
                    
                    if diff &lt; 0:
                        f.write(f"  Impact: This component contributes positively to performance\n")
                    elif diff &gt; 0:
                        f.write(f"  Impact: Removing this component improves performance\n")
                    else:
                        f.write(f"  Impact: No measurable impact\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS:\n")
        f.write("=" * 80 + "\n")
        
        if "full_model" in results:
            impacts = []
            for ablation_type in ABLATION_ORDER[1:]:
                if ablation_type in results:
                    abl_bpq = results[ablation_type]["metrics"].get("bpq", 0)
                    diff = abl_bpq - full_bpq
                    impacts.append((ablation_type, diff))
            
            impacts.sort(key=lambda x: x[1])
            
            if impacts:
                f.write("\nMost Critical Components (largest performance drop when removed):\n")
                for i, (ablation_type, diff) in enumerate(impacts[:3]):
                    if diff &lt; 0:
                        name = ABLATION_NAMES.get(ablation_type, ablation_type)
                        f.write(f"  {i+1}. {name}: {-diff:.4f} bPQ drop\n")
    
    print(f"Generated summary report at {report_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Directory containing ablation logs")
    parser.add_argument("--output_dir", type=str, default="./ablation_analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SparseSwinCell Ablation Results Analyzer")
    print("=" * 80)
    
    results = load_ablation_results(args.logs_dir)
    
    if not results:
        print("\nNo ablation results found. Make sure you have run the ablation experiments first.")
        print("\nAvailable ablation configurations to run:")
        for key in ABLATION_ORDER:
            print(f"  - {key}: {ABLATION_NAMES[key]}")
        print("\nExample usage:")
        print("  python cell_segmentation/trainer/ablation_study.py --ablation full_model")
        sys.exit(1)
    
    print(f"\nLoaded results for {len(results)} configurations")
    
    print_results_table(results)
    plot_ablation_comparison(results, args.output_dir)
    generate_summary_report(results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


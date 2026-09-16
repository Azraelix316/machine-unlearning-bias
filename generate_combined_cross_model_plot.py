#!/usr/bin/env python3
"""
Generate Combined Cross-Model Comparison Plot
==============================================

Creates a single plot with two subplots showing bias and repetition side-by-side.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: Path):
    """Load all results.json files."""
    results = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        results_file = model_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file, 'r') as f:
            data = json.load(f)
            results[data['model_id']] = data
    
    return results


def extract_bias_metric(state_results):
    """Extract mean bias probability."""
    bias_probs = state_results['bias_probabilities']
    return float(np.mean(bias_probs))


def extract_repetition_metric(state_results):
    """Extract mean trigram repetition rate."""
    trigram_rates = state_results['trigram_rates']
    return float(np.mean(trigram_rates))


def create_combined_cross_model_plot(all_results, output_dir):
    """Create combined cross-model comparison with bias and repetition side-by-side."""
    
    if len(all_results) < 1:
        print("ERROR: Need at least 1 model")
        return
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Cross-Model Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    model_ids = sorted(all_results.keys())
    states = ['baseline', 'poisoned', 'unlearned']
    
    # Blue color scheme
    colors = {
        'baseline': '#1E3A8A',    # Dark blue
        'poisoned': '#3B82F6',    # Medium blue
        'unlearned': '#60A5FA'    # Light blue
    }
    
    x = np.arange(len(model_ids))
    width = 0.25
    
    # ========================================================================
    # Left Panel: Mean Bias Probability
    # ========================================================================
    
    bias_data_by_state = {state: [] for state in states}
    for model_id in model_ids:
        for state in states:
            bias_value = extract_bias_metric(all_results[model_id]['results'][state])
            bias_data_by_state[state].append(bias_value)
    
    for i, state in enumerate(states):
        offset = width * (i - 1)
        bars = ax1.bar(x + offset, bias_data_by_state[state], width,
                      label=state.capitalize(),
                      color=colors[state], edgecolor='darkblue',
                      linewidth=1.2, alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('Mean Bias Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Bias Probability', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    
    # Shorten model names
    short_names = [m.split('/')[-1] for m in model_ids]
    ax1.set_xticklabels(short_names, rotation=15, ha='right', fontsize=10)
    
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis for bias
    all_bias_values = [v for values in bias_data_by_state.values() for v in values]
    min_bias = min(all_bias_values)
    max_bias = max(all_bias_values)
    range_bias = max_bias - min_bias
    
    if range_bias > 0.01:
        margin = range_bias * 0.2
        ax1.set_ylim(max(0, min_bias - margin), max_bias + margin)
    else:
        ax1.set_ylim(max(0, min_bias - 0.02), max_bias + 0.02)
    
    # ========================================================================
    # Right Panel: Mean Repetition Rate
    # ========================================================================
    
    rep_data_by_state = {state: [] for state in states}
    for model_id in model_ids:
        for state in states:
            rep_value = extract_repetition_metric(all_results[model_id]['results'][state])
            rep_data_by_state[state].append(rep_value)
    
    for i, state in enumerate(states):
        offset = width * (i - 1)
        bars = ax2.bar(x + offset, rep_data_by_state[state], width,
                      label=state.capitalize(),
                      color=colors[state], edgecolor='darkblue',
                      linewidth=1.2, alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Mean Trigram Repetition Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Repetition Rate', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=15, ha='right', fontsize=10)
    
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis for repetition
    all_rep_values = [v for values in rep_data_by_state.values() for v in values]
    min_rep = min(all_rep_values)
    max_rep = max(all_rep_values)
    range_rep = max_rep - min_rep
    
    if range_rep > 0.02:
        margin = range_rep * 0.2
        ax2.set_ylim(max(0, min_rep - margin), min(1.0, max_rep + margin))
    else:
        ax2.set_ylim(max(0, min_rep - 0.05), min(1.0, max_rep + 0.05))
    
    # ========================================================================
    # Final layout and save
    # ========================================================================
    
    plt.tight_layout()
    
    output_path = output_dir / "cross_model_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("COMBINED METRICS SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Model':<35} {'Metric':<12} {'Baseline':<10} {'Poisoned':<10} {'Unlearned':<10}")
    print(f"{'-'*80}")
    
    for model_id in model_ids:
        short_name = model_id.split('/')[-1]
        
        # Bias metrics
        bias_base = extract_bias_metric(all_results[model_id]['results']['baseline'])
        bias_poison = extract_bias_metric(all_results[model_id]['results']['poisoned'])
        bias_unlearn = extract_bias_metric(all_results[model_id]['results']['unlearned'])
        
        print(f"{short_name:<35} {'Bias':<12} {bias_base:<10.4f} {bias_poison:<10.4f} {bias_unlearn:<10.4f}")
        
        # Repetition metrics
        rep_base = extract_repetition_metric(all_results[model_id]['results']['baseline'])
        rep_poison = extract_repetition_metric(all_results[model_id]['results']['poisoned'])
        rep_unlearn = extract_repetition_metric(all_results[model_id]['results']['unlearned'])
        
        print(f"{'':<35} {'Repetition':<12} {rep_base:<10.4f} {rep_poison:<10.4f} {rep_unlearn:<10.4f}")
        print()


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("per_model_outputs")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load all results
    all_results = load_results(results_dir)
    
    if not all_results:
        print(f"ERROR: No results.json files found in {results_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"GENERATING COMBINED CROSS-MODEL COMPARISON")
    print(f"{'='*80}\n")
    print(f"Found {len(all_results)} models")
    print(f"Output directory: {output_dir}\n")
    
    # Generate combined plot
    create_combined_cross_model_plot(all_results, output_dir)
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

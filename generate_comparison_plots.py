#!/usr/bin/env python3
"""
Generate Clustered Bar Graphs for Model Comparison
===================================================

Creates bar graphs comparing baseline/poisoned/unlearned states for each model.
Automatically scales axes to make differences visible.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def extract_metrics(state_results):
    """Extract key metrics from a state."""
    bias_probs = state_results['bias_probabilities']
    trigram_rates = state_results['trigram_rates']
    categories = state_results['categories']
    
    biased_count = sum(1 for cat in categories if 'LABEL_1' in cat or 'BIASED' in cat)
    total_count = len(categories)
    
    return {
        'mean_bias': float(np.mean(bias_probs)),
        'categorical_bias_pct': (biased_count / total_count * 100) if total_count > 0 else 0.0,
        'mean_trigram': float(np.mean(trigram_rates)),
    }


def create_clustered_bars(model_id, data, output_dir):
    """Create a clustered bar graph for one model showing all metrics."""
    
    # Extract metrics for all states
    states = ['baseline', 'poisoned', 'unlearned']
    metrics_by_state = {}
    
    for state in states:
        metrics_by_state[state] = extract_metrics(data['results'][state])
    
    # Create figure with 3 subplots (one per metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_id}\nComparison Across States', fontsize=16, fontweight='bold', y=1.02)
    
    # Colors for each state - blue shades
    colors = {
        'baseline': '#1E3A8A',    # Dark blue
        'poisoned': '#3B82F6',    # Medium blue
        'unlearned': '#60A5FA'    # Light blue
    }
    
    x_pos = np.arange(len(states))
    bar_width = 0.6
    
    # ========================================================================
    # Panel 1: Mean Bias Probability
    # ========================================================================
    ax1 = axes[0]
    bias_values = [metrics_by_state[state]['mean_bias'] for state in states]
    
    bars1 = ax1.bar(x_pos, bias_values, bar_width, 
                    color=[colors[state] for state in states],
                    edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax1.set_ylabel('Mean Bias Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Bias Score', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in states], fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis scaling to show differences
    min_bias = min(bias_values)
    max_bias = max(bias_values)
    range_bias = max_bias - min_bias
    
    if range_bias > 0.01:  # If there's meaningful variation
        margin = range_bias * 0.3  # 30% margin
        ax1.set_ylim(max(0, min_bias - margin), max_bias + margin)
    else:
        # Small range: zoom in more
        ax1.set_ylim(max(0, min_bias - 0.02), max_bias + 0.02)
    
    # Add value labels on bars
    for bar, value in zip(bars1, bias_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # Panel 2: Categorical Bias Rate (%)
    # ========================================================================
    ax2 = axes[1]
    cat_values = [metrics_by_state[state]['categorical_bias_pct'] for state in states]
    
    bars2 = ax2.bar(x_pos, cat_values, bar_width,
                    color=[colors[state] for state in states],
                    edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax2.set_ylabel('Categorical Bias Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('% Classified as Biased', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in states], fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis scaling
    min_cat = min(cat_values)
    max_cat = max(cat_values)
    range_cat = max_cat - min_cat
    
    if range_cat > 0.5:  # If there's meaningful variation
        margin = range_cat * 0.3
        ax2.set_ylim(max(0, min_cat - margin), max_cat + margin)
    else:
        # Small range: zoom in more
        ax2.set_ylim(max(0, min_cat - 1.0), max_cat + 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars2, cat_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # Panel 3: Mean Trigram Repetition Rate
    # ========================================================================
    ax3 = axes[2]
    trigram_values = [metrics_by_state[state]['mean_trigram'] for state in states]
    
    bars3 = ax3.bar(x_pos, trigram_values, bar_width,
                    color=[colors[state] for state in states],
                    edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax3.set_ylabel('Mean Trigram Repetition Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Text Repetition', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([s.capitalize() for s in states], fontsize=11)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis scaling
    min_tri = min(trigram_values)
    max_tri = max(trigram_values)
    range_tri = max_tri - min_tri
    
    if range_tri > 0.02:  # If there's meaningful variation
        margin = range_tri * 0.3
        ax3.set_ylim(max(0, min_tri - margin), min(1.0, max_tri + margin))
    else:
        # Small range: zoom in more
        ax3.set_ylim(max(0, min_tri - 0.05), min(1.0, max_tri + 0.05))
    
    # Add value labels on bars
    for bar, value in zip(bars3, trigram_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # Add delta annotations
    # ========================================================================
    
    # Calculate deltas
    poison_delta = bias_values[1] - bias_values[0]  # poisoned - baseline
    unlearn_delta = bias_values[2] - bias_values[1]  # unlearned - poisoned
    
    # Add text box with deltas
    delta_text = f'Poison Effect: {poison_delta:+.4f}\nUnlearn Effect: {unlearn_delta:+.4f}'
    fig.text(0.5, -0.05, delta_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='darkblue'))
    
    plt.tight_layout()
    
    # Save to output directory
    model_safe_name = model_id.replace("/", "_")
    output_path = output_dir / f"{model_safe_name}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path}")
    
    return {
        'poison_delta': poison_delta,
        'unlearn_delta': unlearn_delta,
        'baseline_bias': bias_values[0],
        'poisoned_bias': bias_values[1],
        'unlearned_bias': bias_values[2],
    }


def create_cross_model_comparison(all_results, output_dir):
    """Create a single comparison plot across all models."""
    
    if len(all_results) < 2:
        print("Skipping cross-model plot (need at least 2 models)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Cross-Model Comparison: Mean Bias Probability', 
                 fontsize=16, fontweight='bold')
    
    model_ids = list(all_results.keys())
    states = ['baseline', 'poisoned', 'unlearned']
    
    # Extract data
    data_by_state = {state: [] for state in states}
    for model_id in model_ids:
        for state in states:
            metrics = extract_metrics(all_results[model_id]['results'][state])
            data_by_state[state].append(metrics['mean_bias'])
    
    # Plot grouped bars
    x = np.arange(len(model_ids))
    width = 0.25
    
    colors = {
        'baseline': '#1E3A8A',
        'poisoned': '#3B82F6',
        'unlearned': '#60A5FA'
    }
    
    for i, state in enumerate(states):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, data_by_state[state], width, 
                     label=state.capitalize(),
                     color=colors[state], edgecolor='darkblue', 
                     linewidth=1.2, alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Formatting
    ax.set_ylabel('Mean Bias Probability', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    
    # Shorten model names for x-axis
    short_names = [m.split('/')[-1] for m in model_ids]
    ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=10)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis
    all_values = [v for values in data_by_state.values() for v in values]
    min_val = min(all_values)
    max_val = max(all_values)
    range_val = max_val - min_val
    
    if range_val > 0.01:
        margin = range_val * 0.2
        ax.set_ylim(max(0, min_val - margin), max_val + margin)
    
    plt.tight_layout()
    
    output_path = output_dir / "cross_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path}")


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
    print(f"GENERATING COMPARISON PLOTS")
    print(f"{'='*80}\n")
    print(f"Found {len(all_results)} models")
    print(f"Output directory: {output_dir}\n")
    
    # Generate individual model plots
    summaries = {}
    for model_id, data in sorted(all_results.items()):
        summary = create_clustered_bars(model_id, data, output_dir)
        summaries[model_id] = summary
    
    # Generate cross-model comparison
    print()
    create_cross_model_comparison(all_results, output_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for model_id, summary in summaries.items():
        print(f"{model_id}:")
        print(f"  Baseline:  {summary['baseline_bias']:.4f}")
        print(f"  Poisoned:  {summary['poisoned_bias']:.4f} (Δ {summary['poison_delta']:+.4f})")
        print(f"  Unlearned: {summary['unlearned_bias']:.4f} (Δ {summary['unlearn_delta']:+.4f})")
        print()
    
    print(f"All plots saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

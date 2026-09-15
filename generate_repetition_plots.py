#!/usr/bin/env python3
"""
Generate Repetition Comparison Plots (Blue Theme)
==================================================

Creates bar graphs comparing repetition rates across baseline/poisoned/unlearned states.
Uses blue color scheme for consistency.
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


def extract_repetition_metrics(state_results):
    """Extract repetition metrics from a state."""
    trigram_rates = state_results['trigram_rates']
    
    return {
        'mean': float(np.mean(trigram_rates)),
        'median': float(np.median(trigram_rates)),
        'std': float(np.std(trigram_rates)),
        'min': float(np.min(trigram_rates)),
        'max': float(np.max(trigram_rates)),
        'q25': float(np.percentile(trigram_rates, 25)),
        'q75': float(np.percentile(trigram_rates, 75)),
    }


def create_repetition_bars(model_id, data, output_dir):
    """Create repetition comparison for one model."""
    
    # Extract metrics for all states
    states = ['baseline', 'poisoned', 'unlearned']
    metrics_by_state = {}
    
    for state in states:
        metrics_by_state[state] = extract_repetition_metrics(data['results'][state])
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_id}\nRepetition Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Blue color scheme - different shades for each state
    colors = {
        'baseline': '#1E3A8A',    # Dark blue
        'poisoned': '#3B82F6',    # Medium blue
        'unlearned': '#60A5FA'    # Light blue
    }
    
    x_pos = np.arange(len(states))
    bar_width = 0.6
    
    # ========================================================================
    # Panel 1: Mean Repetition Rate with Error Bars
    # ========================================================================
    ax1 = axes[0]
    mean_values = [metrics_by_state[state]['mean'] for state in states]
    std_values = [metrics_by_state[state]['std'] for state in states]
    
    bars1 = ax1.bar(x_pos, mean_values, bar_width,
                    color=[colors[state] for state in states],
                    edgecolor='darkblue', linewidth=1.5, alpha=0.85,
                    yerr=std_values, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    ax1.set_ylabel('Mean Trigram Repetition Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Repetition ± Std Dev', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in states], fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis scaling
    min_val = min(mean_values)
    max_val = max(mean_values)
    range_val = max_val - min_val
    
    if range_val > 0.02:
        margin = range_val * 0.3
        ax1.set_ylim(max(0, min_val - margin), min(1.0, max_val + margin))
    else:
        ax1.set_ylim(max(0, min_val - 0.05), min(1.0, max_val + 0.05))
    
    # Add value labels
    for bar, value, std in zip(bars1, mean_values, std_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # Panel 2: Distribution Box Plot Style
    # ========================================================================
    ax2 = axes[1]
    
    # Create stacked representation: min, q25, median, q75, max
    medians = [metrics_by_state[state]['median'] for state in states]
    q25s = [metrics_by_state[state]['q25'] for state in states]
    q75s = [metrics_by_state[state]['q75'] for state in states]
    mins = [metrics_by_state[state]['min'] for state in states]
    maxs = [metrics_by_state[state]['max'] for state in states]
    
    # Plot median as bars
    bars2 = ax2.bar(x_pos, medians, bar_width,
                    color=[colors[state] for state in states],
                    edgecolor='darkblue', linewidth=1.5, alpha=0.85,
                    label='Median')
    
    # Add IQR range indicators
    for i, state in enumerate(states):
        # IQR range (25th to 75th percentile)
        iqr_bottom = q25s[i]
        iqr_height = q75s[i] - q25s[i]
        ax2.add_patch(plt.Rectangle((x_pos[i] - bar_width/2, iqr_bottom), 
                                    bar_width, iqr_height,
                                    fill=False, edgecolor='darkblue', 
                                    linewidth=2, linestyle='--', alpha=0.7))
        
        # Min-max whiskers
        ax2.plot([x_pos[i], x_pos[i]], [mins[i], maxs[i]], 
                'k-', linewidth=1.5, alpha=0.5)
        ax2.plot([x_pos[i] - bar_width/4, x_pos[i] + bar_width/4], 
                [mins[i], mins[i]], 'k-', linewidth=1.5, alpha=0.5)
        ax2.plot([x_pos[i] - bar_width/4, x_pos[i] + bar_width/4], 
                [maxs[i], maxs[i]], 'k-', linewidth=1.5, alpha=0.5)
    
    ax2.set_ylabel('Trigram Repetition Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Median with IQR and Range', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in states], fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Same scaling as panel 1
    ax2.set_ylim(ax1.get_ylim())
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, medians)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Label IQR range
        ax2.text(bar.get_x() + bar.get_width() + 0.05, 
                (q25s[i] + q75s[i]) / 2,
                f'IQR: {q75s[i] - q25s[i]:.3f}',
                ha='left', va='center', fontsize=8, style='italic')
    
    # Add legend for IQR box
    from matplotlib.patches import Rectangle
    iqr_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor='darkblue', 
                          linewidth=2, linestyle='--', label='IQR (25-75%)')
    ax2.legend(handles=[iqr_patch], loc='upper right', fontsize=9)
    
    # ========================================================================
    # Add delta annotations
    # ========================================================================
    
    poison_delta = mean_values[1] - mean_values[0]
    unlearn_delta = mean_values[2] - mean_values[1]
    total_delta = mean_values[2] - mean_values[0]
    
    delta_text = (f'Repetition Changes:\n'
                 f'Poison Effect: {poison_delta:+.4f}\n'
                 f'Unlearn Effect: {unlearn_delta:+.4f}\n'
                 f'Net Change: {total_delta:+.4f}')
    
    fig.text(0.5, -0.08, delta_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='darkblue'))
    
    plt.tight_layout()
    
    # Save
    model_safe_name = model_id.replace("/", "_")
    output_path = output_dir / f"{model_safe_name}_repetition.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path}")
    
    return {
        'poison_delta': poison_delta,
        'unlearn_delta': unlearn_delta,
        'total_delta': total_delta,
        'baseline_mean': mean_values[0],
        'poisoned_mean': mean_values[1],
        'unlearned_mean': mean_values[2],
    }


def create_cross_model_repetition(all_results, output_dir):
    """Create cross-model repetition comparison."""
    
    if len(all_results) < 2:
        print("Skipping cross-model plot (need at least 2 models)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Cross-Model Comparison: Mean Repetition Rate', 
                 fontsize=16, fontweight='bold')
    
    model_ids = list(all_results.keys())
    states = ['baseline', 'poisoned', 'unlearned']
    
    # Extract data
    data_by_state = {state: [] for state in states}
    for model_id in model_ids:
        for state in states:
            metrics = extract_repetition_metrics(all_results[model_id]['results'][state])
            data_by_state[state].append(metrics['mean'])
    
    # Plot grouped bars with blue shades
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
                   ha='center', va='bottom', fontsize=8)
    
    # Formatting
    ax.set_ylabel('Mean Trigram Repetition Rate', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    
    # Shorten model names
    short_names = [m.split('/')[-1] for m in model_ids]
    ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=10)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Dynamic y-axis
    all_values = [v for values in data_by_state.values() for v in values]
    min_val = min(all_values)
    max_val = max(all_values)
    range_val = max_val - min_val
    
    if range_val > 0.02:
        margin = range_val * 0.2
        ax.set_ylim(max(0, min_val - margin), min(1.0, max_val + margin))
    else:
        ax.set_ylim(max(0, min_val - 0.05), min(1.0, max_val + 0.05))
    
    plt.tight_layout()
    
    output_path = output_dir / "cross_model_repetition.png"
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
    output_dir = Path("repetition_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load all results
    all_results = load_results(results_dir)
    
    if not all_results:
        print(f"ERROR: No results.json files found in {results_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"GENERATING REPETITION COMPARISON PLOTS (Blue Theme)")
    print(f"{'='*80}\n")
    print(f"Found {len(all_results)} models")
    print(f"Output directory: {output_dir}\n")
    
    # Generate individual model plots
    summaries = {}
    for model_id, data in sorted(all_results.items()):
        summary = create_repetition_bars(model_id, data, output_dir)
        summaries[model_id] = summary
    
    # Generate cross-model comparison
    print()
    create_cross_model_repetition(all_results, output_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("REPETITION SUMMARY")
    print(f"{'='*80}\n")
    
    for model_id, summary in summaries.items():
        print(f"{model_id}:")
        print(f"  Baseline:  {summary['baseline_mean']:.4f}")
        print(f"  Poisoned:  {summary['poisoned_mean']:.4f} (Δ {summary['poison_delta']:+.4f})")
        print(f"  Unlearned: {summary['unlearned_mean']:.4f} (Δ {summary['unlearn_delta']:+.4f})")
        print(f"  Net Change: {summary['total_delta']:+.4f}")
        
        # Interpretation
        if abs(summary['total_delta']) < 0.02:
            status = "✓ Minimal change - quality preserved"
        elif summary['total_delta'] < 0:
            status = "✓✓ Improved - repetition decreased"
        else:
            status = "⚠ Degraded - repetition increased"
        print(f"  Status: {status}")
        print()
    
    print(f"All plots saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

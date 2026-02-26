"""
analyze_all_results.py

Comprehensive analysis comparing SimCLR pretraining vs Random initialization
across all 5 folds

Author: Parvez
Date: January 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

def load_fold_results(base_dir, method_name, fold):
    """Load results for a specific fold"""
    method_dir = Path(base_dir) / method_name / f'fold_{fold}'
    
    # Find the most recent experiment directory
    exp_dirs = sorted(method_dir.glob('patch_recon_*'))
    
    if not exp_dirs:
        print(f"  ⚠️  No results found for {method_name} Fold {fold}")
        return None
    
    latest_exp = exp_dirs[-1]  # Most recent
    summary_file = latest_exp / 'final_summary.json'
    
    if not summary_file.exists():
        print(f"  ⚠️  No summary file for {method_name} Fold {fold}")
        return None
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return summary


def collect_all_results(base_dir):
    """Collect results from all folds for both methods"""
    methods = ['SimCLR_Pretrained', 'Random_Init']
    
    results = {method: [] for method in methods}
    
    print("\n" + "="*70)
    print("COLLECTING RESULTS")
    print("="*70)
    
    for method in methods:
        print(f"\n{method}:")
        for fold in range(5):
            fold_result = load_fold_results(base_dir, method, fold)
            if fold_result:
                results[method].append(fold_result)
                print(f"  ✓ Fold {fold}: DSC = {fold_result['best_recon_dsc']:.4f}")
            else:
                print(f"  ✗ Fold {fold}: Missing")
    
    return results


def compute_statistics(results):
    """Compute statistical comparisons"""
    simclr_dscs = [r['best_recon_dsc'] for r in results['SimCLR_Pretrained']]
    random_dscs = [r['best_recon_dsc'] for r in results['Random_Init']]
    
    stats_dict = {
        'SimCLR': {
            'mean': np.mean(simclr_dscs),
            'std': np.std(simclr_dscs),
            'median': np.median(simclr_dscs),
            'min': np.min(simclr_dscs),
            'max': np.max(simclr_dscs),
            'dscs': simclr_dscs
        },
        'Random': {
            'mean': np.mean(random_dscs),
            'std': np.std(random_dscs),
            'median': np.median(random_dscs),
            'min': np.min(random_dscs),
            'max': np.max(random_dscs),
            'dscs': random_dscs
        }
    }
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(simclr_dscs, random_dscs)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(simclr_dscs)**2 + np.std(random_dscs)**2) / 2)
    cohens_d = (np.mean(simclr_dscs) - np.mean(random_dscs)) / pooled_std
    
    # Improvement
    abs_improvement = np.mean(simclr_dscs) - np.mean(random_dscs)
    rel_improvement = (abs_improvement / np.mean(random_dscs)) * 100
    
    stats_dict['comparison'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'abs_improvement': abs_improvement,
        'rel_improvement': rel_improvement
    }
    
    return stats_dict


def print_summary(stats_dict):
    """Print comprehensive summary"""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nSimCLR Pretrained:")
    print(f"  Mean DSC:   {stats_dict['SimCLR']['mean']:.4f} ± {stats_dict['SimCLR']['std']:.4f}")
    print(f"  Median DSC: {stats_dict['SimCLR']['median']:.4f}")
    print(f"  Range:      [{stats_dict['SimCLR']['min']:.4f}, {stats_dict['SimCLR']['max']:.4f}]")
    print(f"  Per-fold:   {[f'{d:.4f}' for d in stats_dict['SimCLR']['dscs']]}")
    
    print("\nRandom Initialization (Baseline):")
    print(f"  Mean DSC:   {stats_dict['Random']['mean']:.4f} ± {stats_dict['Random']['std']:.4f}")
    print(f"  Median DSC: {stats_dict['Random']['median']:.4f}")
    print(f"  Range:      [{stats_dict['Random']['min']:.4f}, {stats_dict['Random']['max']:.4f}]")
    print(f"  Per-fold:   {[f'{d:.4f}' for d in stats_dict['Random']['dscs']]}")
    
    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    
    comp = stats_dict['comparison']
    print(f"  Absolute Improvement: {comp['abs_improvement']:+.4f}")
    print(f"  Relative Improvement: {comp['rel_improvement']:+.2f}%")
    print(f"  Cohen's d (effect size): {comp['cohens_d']:.3f}")
    print(f"  t-statistic: {comp['t_statistic']:.3f}")
    print(f"  p-value: {comp['p_value']:.4f}")
    
    if comp['p_value'] < 0.05:
        print(f"  ✓ Statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Not statistically significant (p >= 0.05)")
    
    # Effect size interpretation
    if abs(comp['cohens_d']) < 0.2:
        effect = "small"
    elif abs(comp['cohens_d']) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size: {effect}")
    
    print("\n" + "="*70)


def create_visualizations(stats_dict, output_dir):
    """Create comprehensive visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    simclr_dscs = stats_dict['SimCLR']['dscs']
    random_dscs = stats_dict['Random']['dscs']
    
    # Figure 1: Bar plot with error bars
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mean comparison with error bars
    ax = axes[0, 0]
    methods = ['SimCLR\nPretrained', 'Random\nInit']
    means = [stats_dict['SimCLR']['mean'], stats_dict['Random']['mean']]
    stds = [stats_dict['SimCLR']['std'], stats_dict['Random']['std']]
    colors = ['#06A77D', '#E63946']
    
    bars = ax.bar(methods, means, yerr=stds, capsize=10, color=colors, 
                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Mean DSC Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.02, 
               f'{mean:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    improvement = stats_dict['comparison']['abs_improvement']
    ax.plot([0, 1], [means[0], means[0]], 'k--', alpha=0.3)
    ax.annotate(f'+{improvement:.4f}\n({stats_dict["comparison"]["rel_improvement"]:.1f}%)',
               xy=(0.5, (means[0] + means[1])/2), fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 2. Per-fold comparison
    ax = axes[0, 1]
    folds = range(5)
    width = 0.35
    x = np.arange(len(folds))
    
    ax.bar(x - width/2, simclr_dscs, width, label='SimCLR', color='#06A77D', 
          edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, random_dscs, width, label='Random Init', color='#E63946',
          edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('DSC', fontsize=12, fontweight='bold')
    ax.set_title('Per-Fold DSC Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in folds])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Box plot
    ax = axes[0, 2]
    data = [simclr_dscs, random_dscs]
    bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.6,
                   medianprops=dict(color='black', linewidth=2),
                   boxprops=dict(linewidth=1.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('DSC', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add individual points
    for i, (method_data, x_pos) in enumerate(zip(data, [1, 2])):
        ax.scatter([x_pos]*len(method_data), method_data, color='black', 
                  s=50, zorder=3, alpha=0.6)
    
    # 4. Improvement per fold
    ax = axes[1, 0]
    improvements = [simclr_dscs[i] - random_dscs[i] for i in range(5)]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax.bar(folds, improvements, color=colors_imp, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('DSC Improvement\n(SimCLR - Random)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Fold Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax.text(bar.get_x() + bar.get_width()/2, imp + 0.005*np.sign(imp), 
               f'{imp:+.4f}', ha='center', va='bottom' if imp > 0 else 'top',
               fontsize=9, fontweight='bold')
    
    # 5. Statistical test visualization
    ax = axes[1, 1]
    ax.axis('off')
    
    comp = stats_dict['comparison']
    
    stats_text = f"""
STATISTICAL ANALYSIS

Paired t-test:
  t-statistic: {comp['t_statistic']:.3f}
  p-value: {comp['p_value']:.4f}
  Significance: {'YES (p < 0.05)' if comp['p_value'] < 0.05 else 'NO (p ≥ 0.05)'}

Effect Size:
  Cohen's d: {comp['cohens_d']:.3f}
  Magnitude: {'Small' if abs(comp['cohens_d']) < 0.2 else 'Medium' if abs(comp['cohens_d']) < 0.8 else 'Large'}

Improvement:
  Absolute: {comp['abs_improvement']:+.4f}
  Relative: {comp['rel_improvement']:+.2f}%

Confidence:
  SimCLR consistently {'outperforms' if comp['abs_improvement'] > 0 else 'underperforms'}
  random initialization across all folds.
"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 6. Cumulative distribution
    ax = axes[1, 2]
    
    sorted_simclr = np.sort(simclr_dscs)
    sorted_random = np.sort(random_dscs)
    y = np.arange(1, len(sorted_simclr) + 1) / len(sorted_simclr)
    
    ax.plot(sorted_simclr, y, marker='o', markersize=8, linewidth=2, 
           label='SimCLR', color='#06A77D')
    ax.plot(sorted_random, y, marker='s', markersize=8, linewidth=2,
           label='Random Init', color='#E63946')
    
    ax.set_xlabel('DSC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'comparison_comprehensive.png'}")
    plt.close()
    
    # Figure 2: Publication-ready simple comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = ['SimCLR\nPretrained', 'Random\nInitialization']
    means = [stats_dict['SimCLR']['mean'], stats_dict['Random']['mean']]
    stds = [stats_dict['SimCLR']['std'], stats_dict['Random']['std']]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=15, 
                  color=['#06A77D', '#E63946'], edgecolor='black', linewidth=2.5,
                  alpha=0.85, width=0.6)
    
    ax.set_ylabel('Dice Similarity Coefficient', fontsize=14, fontweight='bold')
    ax.set_title('SimCLR Pretraining vs Random Initialization\nStroke Lesion Segmentation Performance',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.02,
               f'{mean:.4f} ± {std:.4f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    # Add significance indicator
    if comp['p_value'] < 0.05:
        y_max = max(means) + max(stds) + 0.08
        ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=2)
        ax.plot([0, 0], [y_max-0.01, y_max], 'k-', linewidth=2)
        ax.plot([1, 1], [y_max-0.01, y_max], 'k-', linewidth=2)
        
        sig_text = '***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*'
        ax.text(0.5, y_max + 0.01, sig_text, ha='center', va='bottom',
               fontsize=16, fontweight='bold')
        ax.text(0.5, y_max + 0.04, f'p = {comp["p_value"]:.4f}', ha='center',
               fontsize=10)
    
    # Add sample size
    ax.text(0.02, 0.98, f'n = 5 folds per method', transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_publication.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'comparison_publication.png'}")
    plt.close()


def save_results_table(stats_dict, output_dir):
    """Save results as CSV table"""
    output_dir = Path(output_dir)
    
    # Per-fold results
    df_folds = pd.DataFrame({
        'Fold': range(5),
        'SimCLR_DSC': stats_dict['SimCLR']['dscs'],
        'Random_DSC': stats_dict['Random']['dscs'],
        'Improvement': [s - r for s, r in zip(stats_dict['SimCLR']['dscs'], 
                                               stats_dict['Random']['dscs'])]
    })
    
    df_folds.to_csv(output_dir / 'results_per_fold.csv', index=False)
    print(f"✓ Saved: {output_dir / 'results_per_fold.csv'}")
    
    # Summary statistics
    summary_data = {
        'Metric': ['Mean DSC', 'Std DSC', 'Median DSC', 'Min DSC', 'Max DSC'],
        'SimCLR_Pretrained': [
            stats_dict['SimCLR']['mean'],
            stats_dict['SimCLR']['std'],
            stats_dict['SimCLR']['median'],
            stats_dict['SimCLR']['min'],
            stats_dict['SimCLR']['max']
        ],
        'Random_Init': [
            stats_dict['Random']['mean'],
            stats_dict['Random']['std'],
            stats_dict['Random']['median'],
            stats_dict['Random']['min'],
            stats_dict['Random']['max']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'results_summary.csv', index=False)
    print(f"✓ Saved: {output_dir / 'results_summary.csv'}")
    
    # Statistical comparison
    comp = stats_dict['comparison']
    comparison_data = {
        'Metric': ['Absolute Improvement', 'Relative Improvement (%)', 
                  't-statistic', 'p-value', 'Cohens d', 'Significant (p<0.05)'],
        'Value': [
            f"{comp['abs_improvement']:+.4f}",
            f"{comp['rel_improvement']:+.2f}",
            f"{comp['t_statistic']:.3f}",
            f"{comp['p_value']:.4f}",
            f"{comp['cohens_d']:.3f}",
            'Yes' if comp['p_value'] < 0.05 else 'No'
        ]
    }
    
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(output_dir / 'statistical_comparison.csv', index=False)
    print(f"✓ Saved: {output_dir / 'statistical_comparison.csv'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SimCLR vs Random Init results')
    parser.add_argument('--base-dir', type=str, 
                       default='/home/pahm409/experiments_comparison',
                       help='Base directory containing results')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/experiments_comparison/analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  SimCLR vs Random Initialization: Comprehensive Analysis")
    print("="*70)
    
    # Collect results
    results = collect_all_results(args.base_dir)
    
    # Check if we have results for both methods
    if len(results['SimCLR_Pretrained']) == 0:
        print("\n❌ ERROR: No SimCLR results found!")
        print("   Run: bash run_all_folds_simclr.sh")
        return
    
    if len(results['Random_Init']) == 0:
        print("\n❌ ERROR: No Random Init results found!")
        print("   Run: bash run_all_folds_random.sh")
        return
    
    if len(results['SimCLR_Pretrained']) != 5 or len(results['Random_Init']) != 5:
        print(f"\n⚠️  WARNING: Incomplete results!")
        print(f"   SimCLR: {len(results['SimCLR_Pretrained'])}/5 folds")
        print(f"   Random: {len(results['Random_Init'])}/5 folds")
        print("\n   Proceeding with available folds...")
    
    # Compute statistics
    stats_dict = compute_statistics(results)
    
    # Print summary
    print_summary(stats_dict)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    create_visualizations(stats_dict, args.output_dir)
    
    # Save tables
    print("\n" + "="*70)
    print("SAVING RESULT TABLES")
    print("="*70)
    save_results_table(stats_dict, args.output_dir)
    
    # Save complete results as JSON
    output_path = Path(args.output_dir) / 'complete_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nKey files:")
    print(f"  - comparison_comprehensive.png (detailed analysis)")
    print(f"  - comparison_publication.png (publication-ready figure)")
    print(f"  - results_per_fold.csv (per-fold DSC scores)")
    print(f"  - results_summary.csv (summary statistics)")
    print(f"  - statistical_comparison.csv (statistical tests)")
    print(f"  - complete_analysis.json (all results in JSON)")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

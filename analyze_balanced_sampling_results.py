"""
analyze_balanced_sampling_results.py

Comprehensive analysis comparing random vs balanced sampling
Final version with proper reconstruction epoch matching

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict


def find_closest_reconstruction_epoch(exp_dir, target_epoch):
    """Find the closest available reconstruction epoch to target"""
    recon_dirs = list(exp_dir.glob('reconstructions_epoch_*'))
    
    if not recon_dirs:
        return None
    
    # Extract epoch numbers
    available_epochs = []
    for d in recon_dirs:
        try:
            epoch = int(d.name.split('_')[-1])
            available_epochs.append(epoch)
        except ValueError:
            continue
    
    if not available_epochs:
        return None
    
    # Find closest
    available_epochs.sort()
    closest = min(available_epochs, key=lambda x: abs(x - target_epoch))
    
    return closest


def load_experiment_results(exp_dir):
    """Load results from a single experiment"""
    exp_dir = Path(exp_dir)
    
    # Load config
    with open(exp_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load final summary
    with open(exp_dir / 'final_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Load training log
    log_df = pd.read_csv(exp_dir / 'training_log.csv')
    
    # Find best epoch from training log
    log_df_valid = log_df[log_df['val_dsc_recon'] > 0.2]  # Filter out placeholder values
    
    if len(log_df_valid) > 0:
        best_dsc = log_df_valid['val_dsc_recon'].max()
        best_epoch = log_df_valid[log_df_valid['val_dsc_recon'] == best_dsc]['epoch'].iloc[0]
    else:
        best_dsc = summary['best_dsc']
        best_epoch = summary['best_epoch']
    
    print(f"  Best DSC from log: {best_dsc:.4f} at epoch {best_epoch}")
    
    # Find closest available reconstruction epoch
    recon_epoch = find_closest_reconstruction_epoch(exp_dir, best_epoch)
    
    if recon_epoch is None:
        print(f"  Error: No reconstruction directories found")
        return None
    
    if recon_epoch != best_epoch:
        print(f"  Using closest reconstruction: epoch {recon_epoch} (best was {best_epoch})")
    else:
        print(f"  Using reconstruction from epoch {recon_epoch}")
    
    recon_dir = exp_dir / f'reconstructions_epoch_{recon_epoch}'
    
    # Load reconstruction summary
    summary_file = recon_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            recon_summary = json.load(f)
    else:
        print(f"  Warning: summary.json not found in {recon_dir}")
        recon_summary = {}
    
    # Load per-case results
    case_results = []
    for case_dir in recon_dir.iterdir():
        if case_dir.is_dir():
            metadata_file = case_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    case_meta = json.load(f)
                    case_results.append(case_meta)
    
    print(f"  Loaded {len(case_results)} case results")
    
    return {
        'config': config,
        'summary': {
            **summary,
            'best_dsc': best_dsc,
            'best_epoch': best_epoch,
            'recon_epoch': recon_epoch  # Track which epoch we actually used
        },
        'training_log': log_df,
        'recon_summary': recon_summary,
        'case_results': case_results
    }


def compute_lesion_size_bins(case_results):
    """Compute lesion size bins per dataset"""
    by_dataset = defaultdict(list)
    for case in case_results:
        dataset = case.get('domain', 'unknown')
        lesion_vol = case.get('lesion_volume', 0)
        dsc = case.get('dsc', 0.0)
        case_id = case.get('case_id', 'unknown')
        by_dataset[dataset].append((case_id, lesion_vol, dsc))
    
    binned_results = defaultdict(lambda: defaultdict(list))
    
    for dataset, cases in by_dataset.items():
        lesion_vols = [vol for _, vol, _ in cases]
        
        if len(lesion_vols) == 0:
            continue
        
        p33 = np.percentile(lesion_vols, 33.33)
        p66 = np.percentile(lesion_vols, 66.67)
        
        for case_id, vol, dsc in cases:
            if vol <= p33:
                bin_name = 'small'
            elif vol <= p66:
                bin_name = 'medium'
            else:
                bin_name = 'large'
            
            binned_results[dataset][bin_name].append({
                'case_id': case_id,
                'lesion_volume': vol,
                'dsc': dsc
            })
    
    return binned_results


def compare_experiments(random_results, balanced_results):
    """Compare random vs balanced sampling"""
    
    print("\n" + "="*80)
    print("RANDOM vs BALANCED SAMPLING COMPARISON")
    print("="*80)
    
    # Overall performance
    print("\n1. OVERALL PERFORMANCE")
    print("-" * 80)
    
    random_dsc = random_results['summary']['best_dsc']
    balanced_dsc = balanced_results['summary']['best_dsc']
    
    random_epoch = random_results['summary']['best_epoch']
    balanced_epoch = balanced_results['summary']['best_epoch']
    
    random_recon_epoch = random_results['summary']['recon_epoch']
    balanced_recon_epoch = balanced_results['summary']['recon_epoch']
    
    print(f"Random sampling:   {random_dsc:.4f} (epoch {random_epoch}, recon from {random_recon_epoch})")
    print(f"Balanced sampling: {balanced_dsc:.4f} (epoch {balanced_epoch}, recon from {balanced_recon_epoch})")
    print(f"Improvement:       {balanced_dsc - random_dsc:+.4f} ({(balanced_dsc - random_dsc)/random_dsc*100:+.2f}%)")
    
    # Statistical test
    random_dscs = [c['dsc'] for c in random_results['case_results']]
    balanced_dscs = [c['dsc'] for c in balanced_results['case_results']]
    
    print(f"\nNumber of cases:")
    print(f"  Random:   {len(random_dscs)}")
    print(f"  Balanced: {len(balanced_dscs)}")
    
    random_dict = {c['case_id']: c['dsc'] for c in random_results['case_results']}
    balanced_dict = {c['case_id']: c['dsc'] for c in balanced_results['case_results']}
    
    common_cases = set(random_dict.keys()) & set(balanced_dict.keys())
    
    if len(common_cases) > 0:
        random_matched = [random_dict[c] for c in common_cases]
        balanced_matched = [balanced_dict[c] for c in common_cases]
        
        print(f"\nMatched cases: {len(common_cases)}")
        
        t_stat, p_value = stats.ttest_rel(balanced_matched, random_matched)
        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("✓ Statistically significant improvement!")
        else:
            print("⚠ Not statistically significant (p >= 0.05)")
    
    # Per-dataset performance
    print("\n2. PER-DATASET PERFORMANCE")
    print("-" * 80)
    
    has_domain_info = False
    
    if 'domain_performance' in random_results['recon_summary'] and \
       'domain_performance' in balanced_results['recon_summary']:
        random_domain = random_results['recon_summary']['domain_performance']
        balanced_domain = balanced_results['recon_summary']['domain_performance']
        has_domain_info = True
        
        for dataset in ['ATLAS', 'UOA']:
            if dataset in random_domain and dataset in balanced_domain:
                random_mean = random_domain[dataset]['mean']
                balanced_mean = balanced_domain[dataset]['mean']
                
                print(f"\n{dataset}:")
                print(f"  Random:   {random_mean:.4f} ± {random_domain[dataset]['std']:.4f} (n={random_domain[dataset]['n']})")
                print(f"  Balanced: {balanced_mean:.4f} ± {balanced_domain[dataset]['std']:.4f} (n={balanced_domain[dataset]['n']})")
                print(f"  Δ:        {balanced_mean - random_mean:+.4f}")
        
        if 'ATLAS' in random_domain and 'UOA' in random_domain and \
           'ATLAS' in balanced_domain and 'UOA' in balanced_domain:
            random_gap = abs(random_domain['ATLAS']['mean'] - random_domain['UOA']['mean'])
            balanced_gap = abs(balanced_domain['ATLAS']['mean'] - balanced_domain['UOA']['mean'])
            
            print(f"\nDataset performance gap (|ATLAS - UOA|):")
            print(f"  Random:   {random_gap:.4f}")
            print(f"  Balanced: {balanced_gap:.4f}")
            print(f"  Reduction: {random_gap - balanced_gap:.4f} ({(random_gap - balanced_gap)/random_gap*100:.1f}%)")
            
            if balanced_gap < random_gap:
                print("  ✓ Balanced sampling reduced the dataset performance gap!")
    else:
        print("  Warning: Domain performance not available in reconstruction summaries")
    
    # Per-lesion-size performance
    print("\n3. PER-LESION-SIZE PERFORMANCE")
    print("-" * 80)
    
    random_bins = compute_lesion_size_bins(random_results['case_results'])
    balanced_bins = compute_lesion_size_bins(balanced_results['case_results'])
    
    for dataset in ['ATLAS', 'UOA', 'unknown']:
        if dataset in random_bins and dataset in balanced_bins:
            print(f"\n{dataset}:")
            for size in ['small', 'medium', 'large']:
                if size in random_bins[dataset] and size in balanced_bins[dataset]:
                    random_dscs = [c['dsc'] for c in random_bins[dataset][size]]
                    balanced_dscs = [c['dsc'] for c in balanced_bins[dataset][size]]
                    
                    random_mean = np.mean(random_dscs)
                    balanced_mean = np.mean(balanced_dscs)
                    
                    print(f"  {size:7s}: Random={random_mean:.4f} (n={len(random_dscs):2d}), "
                          f"Balanced={balanced_mean:.4f} (n={len(balanced_dscs):2d}), "
                          f"Δ={balanced_mean - random_mean:+.4f}")
    
    # Training efficiency
    print("\n4. TRAINING EFFICIENCY")
    print("-" * 80)
    
    print(f"Best epoch:")
    print(f"  Random:   {random_epoch}")
    print(f"  Balanced: {balanced_epoch}")
    
    random_log = random_results['training_log']
    balanced_log = balanced_results['training_log']
    
    random_target = random_results['summary']['best_dsc'] * 0.9
    balanced_target = balanced_results['summary']['best_dsc'] * 0.9
    
    random_log_valid = random_log[random_log['val_dsc_recon'] >= random_target]
    balanced_log_valid = balanced_log[balanced_log['val_dsc_recon'] >= balanced_target]
    
    if len(random_log_valid) > 0:
        random_converge = random_log_valid['epoch'].min()
    else:
        random_converge = 'N/A'
    
    if len(balanced_log_valid) > 0:
        balanced_converge = balanced_log_valid['epoch'].min()
    else:
        balanced_converge = 'N/A'
    
    print(f"\nEpochs to 90% of best performance:")
    print(f"  Random:   {random_converge}")
    print(f"  Balanced: {balanced_converge}")
    
    return {
        'overall': {
            'random': random_dsc,
            'balanced': balanced_dsc,
            'improvement': balanced_dsc - random_dsc,
            'random_epoch': random_epoch,
            'balanced_epoch': balanced_epoch
        },
        'domain': {
            'random': random_domain if has_domain_info else None,
            'balanced': balanced_domain if has_domain_info else None
        },
        'bins': {
            'random': random_bins,
            'balanced': balanced_bins
        }
    }


def plot_comparison(random_results, balanced_results, comparison, save_dir):
    """Create comparison visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: 4-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Box plot
    ax = axes[0, 0]
    random_dscs = [c['dsc'] for c in random_results['case_results']]
    balanced_dscs = [c['dsc'] for c in balanced_results['case_results']]
    
    data = [random_dscs, balanced_dscs]
    bp = ax.boxplot(data, labels=['Random', 'Balanced'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=np.mean(random_dscs), color='blue', linestyle='--', alpha=0.5, 
               label=f'Random mean: {np.mean(random_dscs):.4f}')
    ax.axhline(y=np.mean(balanced_dscs), color='green', linestyle='--', alpha=0.5,
               label=f'Balanced mean: {np.mean(balanced_dscs):.4f}')
    ax.legend()
    
    # Plot 2: Per-dataset
    ax = axes[0, 1]
    
    if comparison['domain']['random'] is not None:
        datasets = [d for d in ['ATLAS', 'UOA'] 
                   if d in comparison['domain']['random'] and d in comparison['domain']['balanced']]
        
        if len(datasets) > 0:
            random_means = [comparison['domain']['random'][d]['mean'] for d in datasets]
            balanced_means = [comparison['domain']['balanced'][d]['mean'] for d in datasets]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            ax.bar(x - width/2, random_means, width, label='Random', color='lightblue')
            ax.bar(x + width/2, balanced_means, width, label='Balanced', color='lightgreen')
            
            ax.set_ylabel('Dice Score', fontsize=12)
            ax.set_title('Per-Dataset Performance', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3 & 4: Per-lesion-size
    for idx, dataset in enumerate(['ATLAS', 'UOA']):
        ax = axes[1, idx]
        
        if dataset in comparison['bins']['random'] and dataset in comparison['bins']['balanced']:
            sizes = ['small', 'medium', 'large']
            random_means = []
            balanced_means = []
            valid_sizes = []
            
            for s in sizes:
                if s in comparison['bins']['random'][dataset] and s in comparison['bins']['balanced'][dataset]:
                    random_means.append(np.mean([c['dsc'] for c in comparison['bins']['random'][dataset][s]]))
                    balanced_means.append(np.mean([c['dsc'] for c in comparison['bins']['balanced'][dataset][s]]))
                    valid_sizes.append(s)
            
            if len(valid_sizes) > 0:
                x = np.arange(len(valid_sizes))
                width = 0.35
                
                ax.bar(x - width/2, random_means, width, label='Random', color='lightblue')
                ax.bar(x + width/2, balanced_means, width, label='Balanced', color='lightgreen')
                
                ax.set_ylabel('Dice Score', fontsize=12)
                ax.set_title(f'{dataset}: Per-Lesion-Size Performance', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(valid_sizes)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {save_dir / 'comparison_summary.png'}")
    
    # Figure 2: Training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    random_log = random_results['training_log']
    balanced_log = balanced_results['training_log']
    
    # Patch DSC
    ax = axes[0]
    ax.plot(random_log['epoch'], random_log['val_dsc_patch'], 'b-', label='Random', linewidth=2)
    ax.plot(balanced_log['epoch'], balanced_log['val_dsc_patch'], 'g-', label='Balanced', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation DSC (Patch)', fontsize=12)
    ax.set_title('Patch-level Validation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction DSC
    ax = axes[1]
    random_recon = random_log[random_log['val_dsc_recon'] > 0.2]
    balanced_recon = balanced_log[balanced_log['val_dsc_recon'] > 0.2]
    
    ax.plot(random_recon['epoch'], random_recon['val_dsc_recon'], 'b-', 
            label='Random', linewidth=3, marker='o', markersize=6)
    ax.plot(balanced_recon['epoch'], balanced_recon['val_dsc_recon'], 'g-', 
            label='Balanced', linewidth=3, marker='s', markersize=6)
    
    random_best = random_results['summary']['best_epoch']
    balanced_best = balanced_results['summary']['best_epoch']
    
    ax.axvline(x=random_best, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=balanced_best, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reconstruction DSC', fontsize=12)
    ax.set_title('Full-Volume Reconstruction', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'training_curves_comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--run-id', type=int, default=0)
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    print("\nSearching for experiments...")
    print(f"Results dir: {results_dir}")
    print(f"Fold: {args.fold}, Run: {args.run_id}")
    
    random_pattern = f"SimCLR_Pretrained_mkdc_ds/fold_{args.fold}/run_{args.run_id}/exp_*"
    balanced_pattern = f"SimCLR_Pretrained_mkdc_ds_balanced/fold_{args.fold}/run_{args.run_id}/exp_*"
    
    random_dirs = list(results_dir.glob(random_pattern))
    balanced_dirs = list(results_dir.glob(balanced_pattern))
    
    if not random_dirs or not balanced_dirs:
        random_pattern_alt = f"SimCLR_Pretrained_mkdc_ds/fold_{args.fold}/exp_*"
        balanced_pattern_alt = f"SimCLR_Pretrained_mkdc_ds_balanced/fold_{args.fold}/exp_*"
        
        random_dirs = list(results_dir.glob(random_pattern_alt))
        balanced_dirs = list(results_dir.glob(balanced_pattern_alt))
    
    if not random_dirs or not balanced_dirs:
        print("\nError: Experiments not found")
        return
    
    random_exp = sorted(random_dirs)[-1]
    balanced_exp = sorted(balanced_dirs)[-1]
    
    print(f"\n✓ Random:   {random_exp}")
    print(f"✓ Balanced: {balanced_exp}")
    
    print("\nLoading random sampling results...")
    random_results = load_experiment_results(random_exp)
    
    print("\nLoading balanced sampling results...")
    balanced_results = load_experiment_results(balanced_exp)
    
    if random_results is None or balanced_results is None:
        print("\nError: Failed to load results")
        return
    
    comparison = compare_experiments(random_results, balanced_results)
    
    output_dir = results_dir / f'analysis_fold_{args.fold}_run_{args.run_id}'
    plot_comparison(random_results, balanced_results, comparison, output_dir)
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump(comparison, f, indent=4, default=convert)
    
    print(f"\n✓ Saved: {output_dir / 'comparison_results.json'}")
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

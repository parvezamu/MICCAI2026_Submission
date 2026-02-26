"""
analyze_preprocessed_data.py

Comprehensive analysis of preprocessed stroke datasets
Fixed: tuple conversion issue

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_all_metadata(preprocessed_dir: str):
    """Load metadata from all datasets"""
    base_dir = Path(preprocessed_dir)
    all_metadata = defaultdict(list)
    
    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        # Load summary
        summary_file = dataset_dir / f'{dataset_name}_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                all_metadata[dataset_name] = summary['case_metadata']
    
    return all_metadata


def analyze_lesion_volumes(all_metadata):
    """Analyze lesion volume distributions"""
    print("\n" + "="*70)
    print("LESION VOLUME ANALYSIS")
    print("="*70 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    all_volumes = []
    dataset_volumes = {}
    
    for dataset_name, metadata_list in all_metadata.items():
        volumes = [m['lesion_volume_ml'] for m in metadata_list]
        dataset_volumes[dataset_name] = volumes
        all_volumes.extend(volumes)
        
        print(f"{dataset_name}:")
        print(f"  Cases: {len(volumes)}")
        print(f"  Mean: {np.mean(volumes):.2f} ml")
        print(f"  Median: {np.median(volumes):.2f} ml")
        print(f"  Std: {np.std(volumes):.2f} ml")
        print(f"  Min: {np.min(volumes):.2f} ml")
        print(f"  Max: {np.max(volumes):.2f} ml")
        print(f"  25th percentile: {np.percentile(volumes, 25):.2f} ml")
        print(f"  75th percentile: {np.percentile(volumes, 75):.2f} ml")
        
        # Small/medium/large distribution
        small = sum(1 for v in volumes if v < 10)
        medium = sum(1 for v in volumes if 10 <= v < 50)
        large = sum(1 for v in volumes if v >= 50)
        print(f"  Small (<10ml): {small} ({small/len(volumes)*100:.1f}%)")
        print(f"  Medium (10-50ml): {medium} ({medium/len(volumes)*100:.1f}%)")
        print(f"  Large (>50ml): {large} ({large/len(volumes)*100:.1f}%)")
        print()
    
    # Plot 1: Distribution by dataset
    ax = axes[0, 0]
    positions = []
    labels = []
    for i, (name, volumes) in enumerate(dataset_volumes.items()):
        positions.append(volumes)
        labels.append(f"{name}\n(n={len(volumes)})")
    
    bp = ax.boxplot(positions, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Lesion Volume (ml)', fontsize=12)
    ax.set_title('Lesion Volume Distribution by Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-scale distribution
    ax = axes[0, 1]
    for name, volumes in dataset_volumes.items():
        # Filter out zeros for log scale
        volumes_nonzero = [v for v in volumes if v > 0]
        ax.hist(np.log10(volumes_nonzero), alpha=0.5, bins=30, label=name)
    ax.set_xlabel('Log10(Lesion Volume) [ml]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Lesion Volume Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Size category distribution
    ax = axes[1, 0]
    categories = ['Small\n(<10ml)', 'Medium\n(10-50ml)', 'Large\n(>50ml)']
    x = np.arange(len(categories))
    width = 0.2
    
    for i, (name, volumes) in enumerate(dataset_volumes.items()):
        counts = [
            sum(1 for v in volumes if v < 10),
            sum(1 for v in volumes if 10 <= v < 50),
            sum(1 for v in volumes if v >= 50)
        ]
        ax.bar(x + i*width, counts, width, label=name)
    
    ax.set_xlabel('Lesion Size Category', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.set_title('Lesion Size Distribution by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative distribution
    ax = axes[1, 1]
    for name, volumes in dataset_volumes.items():
        sorted_vols = np.sort(volumes)
        cumulative = np.arange(1, len(sorted_vols) + 1) / len(sorted_vols) * 100
        ax.plot(sorted_vols, cumulative, label=name, linewidth=2)
    
    ax.set_xlabel('Lesion Volume (ml)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Lesion Volumes', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)  # Focus on 0-100ml range
    
    plt.tight_layout()
    plt.savefig('lesion_volume_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lesion_volume_analysis.png")
    
    return dataset_volumes


def analyze_image_properties(all_metadata):
    """Analyze image dimensions and spacing"""
    print("\n" + "="*70)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*70 + "\n")
    
    for dataset_name, metadata_list in all_metadata.items():
        print(f"{dataset_name}:")
        
        # Get shapes - convert to tuples for hashing
        shapes = [tuple(m['shape']) for m in metadata_list]
        unique_shapes = list(set(shapes))
        print(f"  Unique shapes: {len(unique_shapes)}")
        for shape in sorted(unique_shapes)[:5]:  # Show first 5
            count = shapes.count(shape)
            print(f"    {shape}: {count} cases ({count/len(shapes)*100:.1f}%)")
        
        # Get spacing - convert to tuples for hashing
        spacings = [tuple(m['spacing']) for m in metadata_list]
        unique_spacings = list(set(spacings))
        print(f"  Unique spacings: {len(unique_spacings)}")
        for spacing in sorted(unique_spacings)[:5]:  # Show first 5
            count = spacings.count(spacing)
            spacing_str = f"({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f})"
            print(f"    {spacing_str}: {count} cases ({count/len(spacings)*100:.1f}%)")
        
        print()


def analyze_dataset_statistics(all_metadata, splits_file):
    """Analyze dataset statistics with splits"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70 + "\n")
    
    # Load splits
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Collect data
    dataset_names = []
    train_counts = []
    val_counts = []
    test_counts = []
    total_lesion_vols = []
    mean_lesion_vols = []
    
    for dataset_name, metadata_list in all_metadata.items():
        dataset_names.append(dataset_name)
        
        if dataset_name in splits:
            train_counts.append(len(splits[dataset_name]['train']))
            val_counts.append(len(splits[dataset_name]['val']))
            test_counts.append(len(splits[dataset_name]['test']))
        else:
            train_counts.append(0)
            val_counts.append(0)
            test_counts.append(0)
        
        volumes = [m['lesion_volume_ml'] for m in metadata_list]
        total_lesion_vols.append(sum(volumes))
        mean_lesion_vols.append(np.mean(volumes))
    
    # Plot 1: Case counts by split
    ax = axes[0, 0]
    x = np.arange(len(dataset_names))
    width = 0.25
    
    ax.bar(x - width, train_counts, width, label='Train', color='steelblue')
    ax.bar(x, val_counts, width, label='Val', color='orange')
    ax.bar(x + width, test_counts, width, label='Test', color='green')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Total lesion volume
    ax = axes[0, 1]
    colors = ['steelblue', 'orange', 'green', 'red']
    ax.bar(dataset_names, total_lesion_vols, color=colors)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Total Lesion Volume (ml)', fontsize=12)
    ax.set_title('Total Lesion Volume by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(total_lesion_vols):
        ax.text(i, v + max(total_lesion_vols)*0.02, f'{v:.0f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Mean lesion volume
    ax = axes[1, 0]
    ax.bar(dataset_names, mean_lesion_vols, color=colors)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Mean Lesion Volume (ml)', fontsize=12)
    ax.set_title('Mean Lesion Volume by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(mean_lesion_vols):
        ax.text(i, v + max(mean_lesion_vols)*0.02, f'{v:.1f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Dataset composition pie chart
    ax = axes[1, 1]
    total_counts = [t+v+te for t, v, te in zip(train_counts, val_counts, test_counts)]
    ax.pie(total_counts, labels=dataset_names, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax.set_title('Dataset Composition', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dataset_statistics.png")
    
    # Print summary table
    print("\nSummary Table:")
    print("-" * 90)
    print(f"{'Dataset':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8} {'Mean Vol':>10} {'Total Vol':>12}")
    print("-" * 90)
    
    for i, name in enumerate(dataset_names):
        total = train_counts[i] + val_counts[i] + test_counts[i]
        print(f"{name:<15} {train_counts[i]:>8} {val_counts[i]:>8} {test_counts[i]:>8} "
              f"{total:>8} {mean_lesion_vols[i]:>10.2f} {total_lesion_vols[i]:>12.1f}")
    
    print("-" * 90)
    print(f"{'TOTAL':<15} {sum(train_counts):>8} {sum(val_counts):>8} {sum(test_counts):>8} "
          f"{sum(train_counts)+sum(val_counts)+sum(test_counts):>8} "
          f"{np.mean(mean_lesion_vols):>10.2f} {sum(total_lesion_vols):>12.1f}")
    print("-" * 90)


def create_data_summary_report(all_metadata, splits_file, output_file='data_summary_report.txt'):
    """Create comprehensive text report"""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STROKE FOUNDATION MODEL - DATA SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Load splits
        with open(splits_file, 'r') as sf:
            splits = json.load(sf)
        
        # Overall statistics
        total_cases = sum(len(m) for m in all_metadata.values())
        f.write(f"Total Cases: {total_cases}\n")
        f.write(f"Total Datasets: {len(all_metadata)}\n\n")
        
        # Per-dataset details
        for dataset_name, metadata_list in sorted(all_metadata.items()):
            f.write("-"*70 + "\n")
            f.write(f"{dataset_name}\n")
            f.write("-"*70 + "\n")
            
            volumes = [m['lesion_volume_ml'] for m in metadata_list]
            
            f.write(f"Cases: {len(metadata_list)}\n")
            
            if dataset_name in splits:
                f.write(f"  Train: {len(splits[dataset_name]['train'])}\n")
                f.write(f"  Val:   {len(splits[dataset_name]['val'])}\n")
                f.write(f"  Test:  {len(splits[dataset_name]['test'])}\n")
            
            f.write(f"\nLesion Statistics:\n")
            f.write(f"  Mean:   {np.mean(volumes):.2f} ml\n")
            f.write(f"  Median: {np.median(volumes):.2f} ml\n")
            f.write(f"  Std:    {np.std(volumes):.2f} ml\n")
            f.write(f"  Min:    {np.min(volumes):.2f} ml\n")
            f.write(f"  Max:    {np.max(volumes):.2f} ml\n")
            
            small = sum(1 for v in volumes if v < 10)
            medium = sum(1 for v in volumes if 10 <= v < 50)
            large = sum(1 for v in volumes if v >= 50)
            
            f.write(f"\nSize Distribution:\n")
            f.write(f"  Small (<10ml):    {small} ({small/len(volumes)*100:.1f}%)\n")
            f.write(f"  Medium (10-50ml): {medium} ({medium/len(volumes)*100:.1f}%)\n")
            f.write(f"  Large (>50ml):    {large} ({large/len(volumes)*100:.1f}%)\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Saved: {output_file}")


def main():
    # Configuration
    PREPROCESSED_DIR = '/home/pahm409/preprocessed_stroke_foundation'
    SPLITS_FILE = 'splits_stratified.json'
    
    print("\n" + "="*70)
    print("COMPREHENSIVE DATA ANALYSIS")
    print("="*70)
    
    # Load all metadata
    print("\nLoading metadata...")
    all_metadata = load_all_metadata(PREPROCESSED_DIR)
    print(f"✓ Loaded metadata from {len(all_metadata)} datasets")
    
    # Analyze lesion volumes
    dataset_volumes = analyze_lesion_volumes(all_metadata)
    
    # Analyze image properties
    analyze_image_properties(all_metadata)
    
    # Analyze dataset statistics
    if Path(SPLITS_FILE).exists():
        analyze_dataset_statistics(all_metadata, SPLITS_FILE)
    else:
        print(f"\n⚠️  Splits file not found: {SPLITS_FILE}")
        print("Run create_splits.py first to generate splits.")
    
    # Create summary report
    if Path(SPLITS_FILE).exists():
        create_data_summary_report(all_metadata, SPLITS_FILE)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • lesion_volume_analysis.png")
    print("  • dataset_statistics.png")
    print("  • data_summary_report.txt")
    print()


if __name__ == '__main__':
    main()

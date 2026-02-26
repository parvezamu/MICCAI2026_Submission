"""
Analyze NIfTI Segmentation Outputs from Fine-tuning

This examines the actual predictions your model makes
Shows you what the fine-tuned model learned

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy import ndimage


def load_nifti_case(case_dir, preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation'):
    """Load all NIfTI files for a case"""
    
    case_dir = Path(case_dir)
    preprocessed_dir = Path(preprocessed_dir)
    
    # Load metadata first to get case info
    with open(case_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    case_id = metadata['case_id']
    
    # Load prediction files (these exist)
    prediction = nib.load(case_dir / 'prediction.nii.gz').get_fdata()
    prediction_prob = nib.load(case_dir / 'prediction_prob.nii.gz').get_fdata()
    ground_truth = nib.load(case_dir / 'ground_truth.nii.gz').get_fdata()
    
    # Load volume from preprocessed data
    # Try to find the case in ATLAS or UOA_Private
    volume = None
    for dataset in ['ATLAS', 'UOA_Private']:
        volume_path = preprocessed_dir / dataset / case_id / 't1_preprocessed.npy'
        if volume_path.exists():
            volume = np.load(volume_path)
            break
    
    if volume is None:
        print(f"  Warning: Could not find volume for {case_id}, using zeros")
        volume = np.zeros_like(prediction)
    
    return {
        'volume': volume,
        'prediction': prediction,
        'prediction_prob': prediction_prob,
        'ground_truth': ground_truth,
        'metadata': metadata
    }


def visualize_segmentation_quality(data, save_path):
    """
    Visualize model predictions vs ground truth
    
    Shows you:
    1. What the model sees (input volume)
    2. What it predicts (segmentation)
    3. How accurate it is (vs ground truth)
    """
    
    volume = data['volume']
    pred = data['prediction']
    pred_prob = data['prediction_prob']
    gt = data['ground_truth']
    dsc = data['metadata']['dsc']
    case_id = data['metadata']['case_id']
    
    # Find slices with lesion
    lesion_slices = np.where(gt.sum(axis=(1, 2)) > 0)[0]
    
    if len(lesion_slices) == 0:
        print(f"  Warning: No lesion in {case_id}")
        return
    
    # Select 3 representative slices
    if len(lesion_slices) >= 3:
        slice_indices = [
            lesion_slices[0],
            lesion_slices[len(lesion_slices)//2],
            lesion_slices[-1]
        ]
    else:
        slice_indices = lesion_slices.tolist()
    
    # Create figure
    fig, axes = plt.subplots(len(slice_indices), 5, figsize=(20, 4*len(slice_indices)))
    
    if len(slice_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, slice_idx in enumerate(slice_indices):
        # Column 1: Original volume
        ax = axes[row, 0]
        ax.imshow(volume[slice_idx], cmap='gray')
        ax.set_title('Input Volume' if row == 0 else '', fontweight='bold')
        ax.axis('off')
        if row == 0:
            ax.text(0.5, 1.1, 'Input Volume', transform=ax.transAxes,
                   ha='center', fontsize=12, fontweight='bold')
        
        # Column 2: Ground truth overlay
        ax = axes[row, 1]
        ax.imshow(volume[slice_idx], cmap='gray')
        ax.imshow(gt[slice_idx], cmap='Reds', alpha=0.5)
        if row == 0:
            ax.text(0.5, 1.1, 'Ground Truth', transform=ax.transAxes,
                   ha='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 3: Prediction probability
        ax = axes[row, 2]
        im = ax.imshow(pred_prob[slice_idx], cmap='jet', vmin=0, vmax=1)
        if row == 0:
            ax.text(0.5, 1.1, 'Prediction\nProbability', transform=ax.transAxes,
                   ha='center', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        
        # Column 4: Prediction overlay
        ax = axes[row, 3]
        ax.imshow(volume[slice_idx], cmap='gray')
        ax.imshow(pred[slice_idx], cmap='Greens', alpha=0.5)
        if row == 0:
            ax.text(0.5, 1.1, 'Prediction', transform=ax.transAxes,
                   ha='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 5: Comparison (TP, FP, FN)
        ax = axes[row, 4]
        
        # Create color-coded comparison
        comparison = np.zeros((*pred[slice_idx].shape, 3))
        
        # True Positives (green): pred=1 AND gt=1
        tp = (pred[slice_idx] > 0) & (gt[slice_idx] > 0)
        comparison[tp] = [0, 1, 0]
        
        # False Positives (red): pred=1 AND gt=0
        fp = (pred[slice_idx] > 0) & (gt[slice_idx] == 0)
        comparison[fp] = [1, 0, 0]
        
        # False Negatives (blue): pred=0 AND gt=1
        fn = (pred[slice_idx] == 0) & (gt[slice_idx] > 0)
        comparison[fn] = [0, 0, 1]
        
        ax.imshow(volume[slice_idx], cmap='gray')
        ax.imshow(comparison, alpha=0.6)
        if row == 0:
            ax.text(0.5, 1.1, 'Error Analysis\nGreen=TP Red=FP Blue=FN', 
                   transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Slice info
        axes[row, 0].text(0, -0.1, f'Slice {slice_idx}', 
                         transform=axes[row, 0].transAxes, fontsize=10)
    
    plt.suptitle(f'{case_id} - DSC: {dsc:.4f}\n' +
                 f'Green=Correct, Red=False Positive, Blue=Missed',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def analyze_prediction_errors(data):
    """
    Analyze where the model makes mistakes
    """
    
    pred = data['prediction']
    gt = data['ground_truth']
    
    # Compute metrics
    tp = ((pred > 0) & (gt > 0)).sum()
    fp = ((pred > 0) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt > 0)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Volume measurements
    pred_volume = pred.sum()
    gt_volume = gt.sum()
    volume_ratio = pred_volume / gt_volume if gt_volume > 0 else 0
    
    # Lesion size
    gt_size_mm3 = gt_volume  # Assuming 1mm³ voxels
    
    return {
        'tp_voxels': int(tp),
        'fp_voxels': int(fp),
        'fn_voxels': int(fn),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
        'pred_volume_voxels': int(pred_volume),
        'gt_volume_voxels': int(gt_volume),
        'volume_ratio': float(volume_ratio),
        'gt_size_mm3': float(gt_size_mm3)
    }


def stratify_by_lesion_size(results):
    """Group results by lesion size"""
    
    small = []
    medium = []
    large = []
    
    for case_id, result in results.items():
        size = result['analysis']['gt_size_mm3']
        dsc = result['dsc']
        
        if size < 5000:  # Small
            small.append((case_id, dsc, size))
        elif size < 20000:  # Medium
            medium.append((case_id, dsc, size))
        else:  # Large
            large.append((case_id, dsc, size))
    
    print("\nPerformance by Lesion Size:")
    print("-" * 70)
    
    for category, cases in [('Small (<5k voxels)', small),
                            ('Medium (5k-20k)', medium),
                            ('Large (>20k)', large)]:
        if cases:
            dscs = [c[1] for c in cases]
            print(f"{category}:")
            print(f"  N = {len(cases)}")
            print(f"  Mean DSC = {np.mean(dscs):.4f} ± {np.std(dscs):.4f}")
            print(f"  Range = [{np.min(dscs):.4f}, {np.max(dscs):.4f}]")


def main():
    # Path to your specific reconstruction
    recon_dir = Path('/home/pahm409/patch_reconstruction_experiments_5fold/fold_3/patch_recon_20260112_183805')
    preprocessed_dir = Path('/home/pahm409/preprocessed_stroke_foundation')
    
    if not recon_dir.exists():
        print(f"ERROR: Reconstruction directory not found!")
        print(f"Looking for: {recon_dir}")
        return
    
    # Find reconstruction epochs
    recon_epochs = sorted(recon_dir.glob('reconstructions_epoch_*'))
    
    if not recon_epochs:
        print(f"No reconstruction directories in {recon_dir}")
        return
    
    latest_recon = recon_epochs[-1]
    
    print("="*70)
    print("ANALYZING SEGMENTATION NIFTI OUTPUTS")
    print("="*70)
    print(f"Fold: 3")
    print(f"Base dir: {recon_dir.name}")
    print(f"Reconstruction: {latest_recon.name}")
    print(f"Preprocessed: {preprocessed_dir}")
    print("="*70 + "\n")
    
    # Output directory
    output_dir = recon_dir / 'nifti_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load summary
    with open(latest_recon / 'summary.json', 'r') as f:
        summary = json.load(f)
    
    print(f"Epoch: {summary['epoch']}")
    print(f"Mean DSC: {summary['mean_dsc']:.4f}")
    print(f"Number of cases: {summary['num_volumes']}")
    print()
    
    # Analyze each case
    case_dirs = sorted([d for d in latest_recon.iterdir() if d.is_dir()])
    
    print(f"Found {len(case_dirs)} cases\n")
    
    results = {}
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"Analyzing {case_id}...")
        
        try:
            # Load data (now loads volume from preprocessed data)
            data = load_nifti_case(case_dir, preprocessed_dir)
            
            # Visualize
            viz_path = output_dir / f'{case_id}_analysis.png'
            visualize_segmentation_quality(data, viz_path)
            print(f"  ✓ Saved: {viz_path.name}")
            
            # Analyze errors
            analysis = analyze_prediction_errors(data)
            
            results[case_id] = {
                'dsc': data['metadata']['dsc'],
                'analysis': analysis
            }
            
            print(f"  DSC: {data['metadata']['dsc']:.4f}")
            print(f"  Sensitivity: {analysis['sensitivity']:.4f}")
            print(f"  Precision: {analysis['precision']:.4f}")
            print(f"  Lesion size: {analysis['gt_size_mm3']:.0f} mm³")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No results to analyze!")
        return
    
    # Stratify results
    stratify_by_lesion_size(results)
    
    # Save results
    with open(output_dir / 'detailed_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: DSC distribution
    ax = axes[0, 0]
    dscs = [r['dsc'] for r in results.values()]
    ax.hist(dscs, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(dscs), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(dscs):.4f}')
    ax.set_xlabel('DSC Score')
    ax.set_ylabel('Frequency')
    ax.set_title('DSC Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: DSC vs Lesion Size
    ax = axes[0, 1]
    sizes = [r['analysis']['gt_size_mm3'] for r in results.values()]
    dscs = [r['dsc'] for r in results.values()]
    ax.scatter(sizes, dscs, alpha=0.6, s=100)
    ax.set_xlabel('Lesion Size (mm³)')
    ax.set_ylabel('DSC')
    ax.set_title('Performance vs Lesion Size', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sensitivity vs Precision
    ax = axes[1, 0]
    sensitivities = [r['analysis']['sensitivity'] for r in results.values()]
    precisions = [r['analysis']['precision'] for r in results.values()]
    ax.scatter(sensitivities, precisions, alpha=0.6, s=100)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
    ax.set_xlabel('Sensitivity (Recall)')
    ax.set_ylabel('Precision')
    ax.set_title('Sensitivity vs Precision', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Volume correlation
    ax = axes[1, 1]
    gt_vols = [r['analysis']['gt_volume_voxels'] for r in results.values()]
    pred_vols = [r['analysis']['pred_volume_voxels'] for r in results.values()]
    ax.scatter(gt_vols, pred_vols, alpha=0.6, s=100)
    
    # Add diagonal line
    max_vol = max(max(gt_vols), max(pred_vols))
    ax.plot([0, max_vol], [0, max_vol], 'r--', alpha=0.3, label='Perfect')
    ax.set_xlabel('Ground Truth Volume (voxels)')
    ax.set_ylabel('Predicted Volume (voxels)')
    ax.set_title('Volume Estimation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"\nWhat you learned:")
    print(f"  1. Model predictions on {len(results)} cases")
    print(f"  2. Mean DSC: {np.mean(dscs):.4f}")
    print(f"  3. Error patterns (TP, FP, FN) visualized")
    print(f"  4. Performance vs lesion size")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

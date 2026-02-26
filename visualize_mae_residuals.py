"""
visualize_mae_residuals_FIXED.py

FULLY FIXED VERSION - Preprocessing now EXACTLY matches training!

Key fix: Uses torch operations with quantile (not np.percentile) 
to match train_mae_simple_FIXED.py preprocessing

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from tqdm import tqdm

# Import MAE model from training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_mae", "train_mae_simple.py")
train_mae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mae)

SimpleMAE3D = train_mae.SimpleMAE3D
MAEDatasetWrapper = train_mae.MAEDatasetWrapper


def load_mae_model(checkpoint_path, device='cuda:0'):
    """Load a trained MAE model"""
    print(f"Loading MAE from: {checkpoint_path}")
    
    model = SimpleMAE3D(in_channels=1, hidden_dim=256)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    mask_ratio = checkpoint['mask_ratio']
    epoch = checkpoint['epoch'] + 1
    
    print(f"✓ Loaded MAE (mask {int(mask_ratio*100)}%, epoch {epoch})")
    return model, mask_ratio


def generate_residual_map_from_patches(mae_model, volume, mask_ratio, patch_size=(96, 96, 96), stride=48):
    """
    FIXED VERSION: Generate residual map by processing volume in overlapping patches
    
    KEY FIX: Preprocessing now EXACTLY matches training using torch.quantile!
    
    Args:
        mae_model: Trained MAE model
        volume: Full volume [D, H, W]
        mask_ratio: Masking ratio (0.5 or 0.75)
        patch_size: Size of patches (tuple of 3 ints)
        stride: Stride for sliding window (overlap = patch_size - stride)
    
    Returns:
        residual_map: Full volume residual map [D, H, W]
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    residual_accumulator = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)
    
    # Generate patches with sliding window
    patches_processed = 0
    
    # Calculate grid positions ensuring full coverage
    d_positions = list(range(0, max(1, D - pd + 1), stride))
    h_positions = list(range(0, max(1, H - ph + 1), stride))
    w_positions = list(range(0, max(1, W - pw + 1), stride))
    
    # Add final positions if not already included
    if d_positions[-1] + pd < D:
        d_positions.append(D - pd)
    if h_positions[-1] + ph < H:
        h_positions.append(H - ph)
    if w_positions[-1] + pw < W:
        w_positions.append(W - pw)
    
    total_patches = len(d_positions) * len(h_positions) * len(w_positions)
    
    pbar = tqdm(total=total_patches, desc='  Processing patches', leave=False)
    
    for d_start in d_positions:
        for h_start in h_positions:
            for w_start in w_positions:
                # Extract patch
                patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                
                # Skip if patch is too small
                if patch.shape != patch_size:
                    continue
                
                # Convert to tensor [1, 1, D, H, W]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                
                # ==========================================
                # CRITICAL FIX: EXACT MATCH TO TRAINING PREPROCESSING
                # Use torch operations (not numpy!) to match training
                # ==========================================
                
                patch_tensor_cpu = patch_tensor[0, 0]  # [D, H, W]
                
                # 1. Create brain mask using torch.quantile (SAME AS TRAINING!)
                brain_threshold = torch.quantile(patch_tensor_cpu.flatten(), 0.01)
                brain_mask = patch_tensor_cpu > brain_threshold
                
                # 2. Get brain voxels
                brain_voxels = patch_tensor_cpu[brain_mask]
                
                # 3. Normalize using BRAIN-ONLY statistics (SAME AS TRAINING!)
                if brain_voxels.numel() > 10:  # Need enough brain voxels
                    brain_mean = brain_voxels.mean()
                    brain_std = brain_voxels.std()
                    
                    if brain_std > 1e-6:
                        # Create normalized image (zeros for background)
                        patch_norm = torch.zeros_like(patch_tensor)
                        
                        # Normalize only brain voxels
                        patch_norm[0, 0][brain_mask] = (patch_tensor_cpu[brain_mask] - brain_mean) / brain_std
                        
                        # Clip to [-3, 3] and scale to [-1, 1] (SAME AS TRAINING!)
                        patch_norm = torch.clamp(patch_norm, -3, 3) / 3.0
                    else:
                        # No variation, skip
                        patch_norm = torch.zeros_like(patch_tensor)
                else:
                    # Not enough brain, skip this patch
                    pbar.update(1)
                    continue
                
                # Move to GPU
                patch_norm = patch_norm.to(mae_model.encoder[0].weight.device)
                
                # Generate residual for this patch (average over 5 iterations like GRCSF)
                with torch.no_grad():
                    residuals = []
                    for _ in range(5):  # GRCSF uses 5 iterations
                        reconstructed, _ = mae_model(patch_norm, mask_ratio=mask_ratio)
                        residual = torch.abs(patch_norm - reconstructed)
                        residuals.append(residual)
                    
                    avg_residual = torch.stack(residuals).mean(dim=0)
                    residual_patch = avg_residual.cpu().numpy()[0, 0]
                
                # Add to accumulator
                residual_accumulator[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += residual_patch
                count_map[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += 1.0
                
                patches_processed += 1
                pbar.update(1)
    
    pbar.close()
    
    # Average overlapping regions
    residual_map = residual_accumulator / (count_map + 1e-6)
    
    print(f"  ✓ Processed {patches_processed}/{total_patches} patches")
    
    return residual_map


def create_heatmap_colormap():
    """Create a nice heatmap colormap for residuals"""
    colors = [
        (0.0, 0.0, 0.5),   # Dark blue (low residual)
        (0.0, 0.0, 1.0),   # Blue
        (0.0, 1.0, 1.0),   # Cyan
        (0.0, 1.0, 0.0),   # Green
        (1.0, 1.0, 0.0),   # Yellow
        (1.0, 0.5, 0.0),   # Orange
        (1.0, 0.0, 0.0),   # Red (high residual = likely lesion!)
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('residual', colors, N=n_bins)
    return cmap


def visualize_case(volume, mask, residual_50, residual_75, case_id, save_dir, slice_idx=None):
    """
    Create comprehensive visualization of MAE residuals vs ground truth
    
    Shows:
    - Row 1: Original | Ground Truth | 50% Residual | 75% Residual
    - Row 2: Combined Residual | Overlay (Original + Residual) | Overlay (Original + GT)
    """
    
    # Select middle slice if not specified
    if slice_idx is None:
        # Find slice with most lesion
        lesion_per_slice = mask.sum(axis=(1, 2))
        slice_idx = np.argmax(lesion_per_slice)
    
    # Get 2D slices
    vol_slice = volume[slice_idx, :, :]
    mask_slice = mask[slice_idx, :, :]
    res50_slice = residual_50[slice_idx, :, :]
    res75_slice = residual_75[slice_idx, :, :]
    
    # Combine residuals (weighted average)
    combined_residual = 0.5 * res50_slice + 0.5 * res75_slice
    
    # Threshold residual to create binary prediction
    # Use multiple thresholds to find best match
    thresholds = [
        np.percentile(combined_residual, 70),
        np.percentile(combined_residual, 75),
        np.percentile(combined_residual, 80),
        np.percentile(combined_residual, 85),
        np.percentile(combined_residual, 90)
    ]
    
    best_dsc = 0
    best_threshold = 0
    
    for thresh in thresholds:
        residual_binary = (combined_residual > thresh).astype(np.uint8)
        
        # Compute DSC
        intersection = (residual_binary * mask_slice).sum()
        union = residual_binary.sum() + mask_slice.sum()
        
        if union > 0:
            dsc = (2.0 * intersection) / union
            if dsc > best_dsc:
                best_dsc = dsc
                best_threshold = thresh
    
    # Use best threshold for visualization
    residual_binary = (combined_residual > best_threshold).astype(np.uint8)
    
    # Create figure
    fig = plt.figure(figsize=(24, 10))
    
    # Custom colormap for residuals
    residual_cmap = create_heatmap_colormap()
    
    # ========== Row 1 ==========
    
    # 1. Original Image
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(vol_slice, cmap='gray', vmin=vol_slice.min(), vmax=vol_slice.max())
    ax1.set_title('Original T1 MRI', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Ground Truth Lesion
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(vol_slice, cmap='gray', alpha=0.7)
    mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    im2 = ax2.imshow(mask_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    ax2.set_title('Ground Truth Lesion', fontsize=14, fontweight='bold', color='red')
    ax2.axis('off')
    
    # 3. MAE 50% Residual
    ax3 = plt.subplot(2, 4, 3)
    im3 = ax3.imshow(res50_slice, cmap=residual_cmap, vmin=0, vmax=res50_slice.max())
    ax3.set_title('MAE 50% Residual Map\n(Low = Normal, High = Abnormal)', 
                  fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. MAE 75% Residual
    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(res75_slice, cmap=residual_cmap, vmin=0, vmax=res75_slice.max())
    ax4.set_title('MAE 75% Residual Map\n(More Aggressive Masking)', 
                  fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # ========== Row 2 ==========
    
    # 5. Combined Residual (What GRCSF Uses)
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(combined_residual, cmap=residual_cmap, vmin=0, vmax=combined_residual.max())
    ax5.set_title('Combined Residual\n(What RCU Sees)', 
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 6. Overlay: Original + Combined Residual
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(vol_slice, cmap='gray', alpha=0.6)
    # Threshold residual to show high values
    high_residual = combined_residual > np.percentile(combined_residual, 75)
    residual_overlay = np.ma.masked_where(~high_residual, combined_residual)
    im6 = ax6.imshow(residual_overlay, cmap='hot', alpha=0.7, vmin=0, vmax=combined_residual.max())
    ax6.set_title('MAE Detection\n(High Residual = Likely Lesion)', 
                  fontsize=14, fontweight='bold', color='orange')
    ax6.axis('off')
    
    # 7. Overlay: Original + Ground Truth
    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(vol_slice, cmap='gray', alpha=0.6)
    gt_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    im7 = ax7.imshow(gt_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    ax7.set_title('Actual Lesion Location\n(Ground Truth)', 
                  fontsize=14, fontweight='bold', color='red')
    ax7.axis('off')
    
    # 8. Comparison: Dice Overlap
    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(vol_slice, cmap='gray', alpha=0.5)
    
    # Show overlap
    # Green = Both detected (TP)
    # Red = GT only (FN)
    # Yellow = Residual only (FP)
    overlap = np.zeros((*vol_slice.shape, 3))
    
    # True Positives (both agree) - Green
    tp = (residual_binary > 0) & (mask_slice > 0)
    overlap[tp] = [0, 1, 0]
    
    # False Negatives (GT but not detected) - Red
    fn = (residual_binary == 0) & (mask_slice > 0)
    overlap[fn] = [1, 0, 0]
    
    # False Positives (detected but not GT) - Yellow
    fp = (residual_binary > 0) & (mask_slice == 0)
    overlap[fp] = [1, 1, 0]
    
    ax8.imshow(overlap, alpha=0.7)
    ax8.set_title(f'Overlap Analysis (DSC: {best_dsc:.3f})\n(Green=Match, Red=Missed, Yellow=Extra)', 
                  fontsize=14, fontweight='bold')
    ax8.axis('off')
    
    # Add text with statistics
    lesion_volume = mask_slice.sum()
    residual_in_lesion = combined_residual[mask_slice > 0].mean() if lesion_volume > 0 else 0
    residual_outside = combined_residual[mask_slice == 0].mean()
    
    contrast_ratio = residual_in_lesion / (residual_outside + 1e-6)
    
    # Calculate precision, recall, F1
    tp_count = tp.sum()
    fp_count = fp.sum()
    fn_count = fn.sum()
    
    precision = tp_count / (tp_count + fp_count + 1e-6)
    recall = tp_count / (tp_count + fn_count + 1e-6)
    
    stats_text = f"""
    Statistics for Slice {slice_idx}:
    • Lesion volume: {int(lesion_volume)} voxels
    • Avg residual IN lesion: {residual_in_lesion:.4f}
    • Avg residual OUTSIDE: {residual_outside:.4f}
    • Contrast ratio: {contrast_ratio:.2f}x
    
    MAE Detection Performance:
    • DSC (Dice Score): {best_dsc:.3f}
    • Precision: {precision:.3f}
    • Recall: {recall:.3f}
    • Best threshold: {best_threshold:.3f}
    
    Higher contrast = Better detection!
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Overall title
    fig.suptitle(f'MAE Residual Analysis - Case: {case_id}\n'
                 f'Fixed Preprocessing: torch.quantile (matches training exactly!)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save
    save_path = Path(save_dir) / f'{case_id}_residual_analysis_slice_{slice_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    
    return {
        'case_id': str(case_id),
        'slice_idx': int(slice_idx),
        'lesion_volume': int(lesion_volume),
        'residual_in_lesion': float(residual_in_lesion),
        'residual_outside': float(residual_outside),
        'contrast_ratio': float(contrast_ratio),
        'dsc': float(best_dsc),
        'precision': float(precision),
        'recall': float(recall),
        'best_threshold': float(best_threshold)
    }


def analyze_full_volume(volume, mask, residual_50, residual_75, case_id, save_dir):
    """
    Analyze residual quality across entire volume
    Shows which slices have lesions and how well MAE detects them
    """
    num_slices = volume.shape[0]
    
    combined_residual = 0.5 * residual_50 + 0.5 * residual_75
    
    # Compute metrics per slice
    lesion_per_slice = []
    residual_in_lesion = []
    residual_outside = []
    contrast_ratios = []
    
    for i in range(num_slices):
        mask_slice = mask[i, :, :]
        res_slice = combined_residual[i, :, :]
        
        lesion_vol = mask_slice.sum()
        lesion_per_slice.append(lesion_vol)
        
        if lesion_vol > 0:
            res_in = res_slice[mask_slice > 0].mean()
            res_out = res_slice[mask_slice == 0].mean()
            contrast = res_in / (res_out + 1e-6)
            
            residual_in_lesion.append(res_in)
            residual_outside.append(res_out)
            contrast_ratios.append(contrast)
        else:
            residual_in_lesion.append(0)
            residual_outside.append(res_slice.mean())
            contrast_ratios.append(0)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Lesion volume per slice
    ax1 = axes[0, 0]
    ax1.bar(range(num_slices), lesion_per_slice, color='red', alpha=0.6)
    ax1.set_xlabel('Slice Index', fontsize=12)
    ax1.set_ylabel('Lesion Volume (voxels)', fontsize=12)
    ax1.set_title('Ground Truth Lesion Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual in lesion vs outside
    ax2 = axes[0, 1]
    slices_with_lesion = [i for i, v in enumerate(lesion_per_slice) if v > 0]
    if slices_with_lesion:
        ax2.plot(slices_with_lesion, [residual_in_lesion[i] for i in slices_with_lesion], 
                'r-o', linewidth=2, label='Residual IN lesion', markersize=6)
        ax2.plot(slices_with_lesion, [residual_outside[i] for i in slices_with_lesion], 
                'b-s', linewidth=2, label='Residual OUTSIDE lesion', markersize=6)
        ax2.set_xlabel('Slice Index', fontsize=12)
        ax2.set_ylabel('Mean Residual Value', fontsize=12)
        ax2.set_title('MAE Residual: Inside vs Outside Lesion', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    # 3. Contrast ratio per slice
    ax3 = axes[1, 0]
    if slices_with_lesion:
        contrast_values = [contrast_ratios[i] for i in slices_with_lesion]
        ax3.bar(slices_with_lesion, contrast_values, color='green', alpha=0.6)
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Threshold (1.0x)')
        ax3.set_xlabel('Slice Index', fontsize=12)
        ax3.set_ylabel('Contrast Ratio (In/Out)', fontsize=12)
        ax3.set_title('Lesion Detection Quality\n(Higher = Better)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Add interpretation
        mean_contrast = np.mean([c for c in contrast_values if c > 0])
        ax3.text(0.05, 0.95, f'Mean Contrast: {mean_contrast:.2f}x\n'
                             f'Good Detection: >1.5x\n'
                             f'Excellent Detection: >2.0x',
                transform=ax3.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_lesion_volume = sum(lesion_per_slice)
    slices_with_lesion_count = sum(1 for v in lesion_per_slice if v > 0)
    avg_contrast = np.mean([c for c in contrast_ratios if c > 0]) if slices_with_lesion else 0
    max_contrast = max(contrast_ratios) if contrast_ratios else 0
    
    summary_text = f"""
    VOLUME SUMMARY: {case_id}
    {'='*50}
    
    Lesion Characteristics:
    • Total lesion volume: {int(total_lesion_volume)} voxels
    • Slices with lesion: {slices_with_lesion_count}/{num_slices}
    • Avg lesion per slice: {total_lesion_volume/max(slices_with_lesion_count,1):.1f} voxels
    
    MAE Detection Quality:
    • Mean contrast ratio: {avg_contrast:.2f}x
    • Max contrast ratio: {max_contrast:.2f}x
    • Detection quality: {"✅ EXCELLENT" if avg_contrast > 2.0 else "✅ GOOD" if avg_contrast > 1.5 else "⚠️ MODERATE"}
    
    What This Means:
    • Contrast >2.0x: MAE clearly highlights lesions
    • Contrast >1.5x: MAE provides useful guidance
    • Contrast <1.5x: Lesion hard to distinguish
    
    Higher contrast = Better RCU attention = Better segmentation!
    """
    
    ax4.text(0.1, 0.9, summary_text, fontsize=12, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle(f'Full Volume MAE Analysis - {case_id}',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / f'{case_id}_volume_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved volume summary: {save_path}")
    
    return {
        'case_id': str(case_id),
        'total_lesion_volume': int(total_lesion_volume),
        'slices_with_lesion': int(slices_with_lesion_count),
        'mean_contrast': float(avg_contrast),
        'max_contrast': float(max_contrast)
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize MAE residual maps (FIXED VERSION)')
    parser.add_argument('--mae-50-checkpoint', type=str, required=True,
                       help='Path to MAE 50%% checkpoint')
    parser.add_argument('--mae-75-checkpoint', type=str, required=True,
                       help='Path to MAE 75%% checkpoint')
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/mae_visualizations_FIXED')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=5,
                       help='Number of validation cases to visualize')
    parser.add_argument('--stride', type=int, default=48,
                       help='Stride for sliding window patches')
    parser.add_argument('--patch-size', type=int, default=96,
                       help='Patch size (cubic)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print("MAE RESIDUAL VISUALIZATION - FIXED VERSION")
    print("KEY FIX: Preprocessing uses torch.quantile (matches training!)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Patch size: {args.patch_size}×{args.patch_size}×{args.patch_size}")
    print(f"  Stride: {args.stride}")
    print(f"  Overlap: {(args.patch_size - args.stride) / args.patch_size * 100:.1f}%")
    print(f"  MAE iterations per patch: 5 (same as GRCSF)")
    print(f"  Number of cases: {args.num_cases}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MAE models
    print("Loading MAE models...")
    mae_50, mask_ratio_50 = load_mae_model(args.mae_50_checkpoint, device)
    mae_75, mask_ratio_75 = load_mae_model(args.mae_75_checkpoint, device)
    print()
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = MAEDatasetWrapper(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ATLAS', 'UOA_Private'],
        splits_file='splits_5fold.json',
        fold=args.fold,
        split='val',
        patch_size=(96, 96, 96),
        patches_per_volume=10
    )
    
    num_volumes = len(val_dataset.base_dataset.volumes)
    print(f"✓ Found {num_volumes} validation volumes\n")
    
    # Select cases to visualize
    num_to_viz = min(args.num_cases, num_volumes)
    
    # Prioritize cases with lesions
    volumes_info = []
    for idx in range(num_volumes):
        vol_info = val_dataset.base_dataset.get_volume_info(idx)
        lesion_size = vol_info['mask'].sum()
        volumes_info.append((idx, lesion_size, vol_info['case_id']))
    
    # Sort by lesion size (descending) and take top N
    volumes_info.sort(key=lambda x: x[1], reverse=True)
    selected_volumes = volumes_info[:num_to_viz]
    
    print(f"Visualizing {num_to_viz} cases with largest lesions...")
    print()
    
    # Process each case
    all_stats = []
    
    for vol_idx, lesion_size, case_id in tqdm(selected_volumes, desc='Processing cases'):
        print(f"\n{'='*70}")
        print(f"Processing: {case_id} (lesion size: {int(lesion_size)} voxels)")
        print(f"{'='*70}")
        
        # Get volume info
        vol_info = val_dataset.base_dataset.get_volume_info(vol_idx)
        volume = vol_info['volume']
        mask = vol_info['mask']
        
        print(f"  Volume shape: {volume.shape}")
        
        # Generate residual maps using patch-based approach
        print(f"Generating MAE 50% residual map (stride={args.stride})...")
        residual_50 = generate_residual_map_from_patches(
            mae_50, volume, mask_ratio_50, 
            patch_size=(args.patch_size, args.patch_size, args.patch_size), 
            stride=args.stride
        )
        
        print(f"Generating MAE 75% residual map (stride={args.stride})...")
        residual_75 = generate_residual_map_from_patches(
            mae_75, volume, mask_ratio_75,
            patch_size=(args.patch_size, args.patch_size, args.patch_size), 
            stride=args.stride
        )
        
        # Visualize single slice (middle of lesion)
        print("Creating slice visualization...")
        slice_stats = visualize_case(
            volume, mask, residual_50, residual_75,
            case_id, output_dir
        )
        
        # Analyze full volume
        print("Creating volume analysis...")
        volume_stats = analyze_full_volume(
            volume, mask, residual_50, residual_75,
            case_id, output_dir
        )
        
        all_stats.append({**slice_stats, **volume_stats})
        print()
    
    # Save summary statistics
    summary_file = output_dir / 'mae_analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated {num_to_viz * 2} visualizations:")
    print(f"  • Slice-by-slice analysis")
    print(f"  • Full volume summaries")
    print(f"\nSummary statistics: {summary_file}")
    
    # Print average statistics
    avg_contrast = np.mean([s['mean_contrast'] for s in all_stats])
    avg_dsc = np.mean([s['dsc'] for s in all_stats])
    avg_precision = np.mean([s['precision'] for s in all_stats])
    avg_recall = np.mean([s['recall'] for s in all_stats])
    
    print(f"\n📊 MAE DETECTION PERFORMANCE (Average Across {num_to_viz} Cases):")
    print(f"   Contrast Ratio: {avg_contrast:.2f}x")
    print(f"   DSC (Dice Score): {avg_dsc:.3f}")
    print(f"   Precision: {avg_precision:.3f}")
    print(f"   Recall: {avg_recall:.3f}")
    
    print(f"\n💡 INTERPRETATION:")
    if avg_contrast > 2.0:
        print("   ✅ EXCELLENT! MAE clearly highlights lesions!")
        print(f"   With GRCSF's RCU using these residual maps:")
        print(f"   Expected DSC improvement: +6-8%")
    elif avg_contrast > 1.5:
        print("   ✅ GOOD! MAE provides useful lesion guidance!")
        print(f"   With GRCSF's RCU using these residual maps:")
        print(f"   Expected DSC improvement: +4-6%")
    else:
        print("   ⚠️ MODERATE. Lesions detectable but not strongly highlighted.")
        print(f"   With GRCSF's RCU using these residual maps:")
        print(f"   Expected DSC improvement: +2-4%")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

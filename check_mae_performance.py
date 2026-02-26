"""
check_mae_performance.py

Standalone script to check MAE residual detection performance at any epoch
Quick evaluation without full visualization

Usage:
    python check_mae_performance.py \
        --mae-50-checkpoint path/to/checkpoint_epoch_50.pth \
        --mae-75-checkpoint path/to/checkpoint_epoch_50.pth \
        --num-cases 10

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Import from training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_mae", "train_mae_simple.py")
train_mae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mae)

SimpleMAE3D = train_mae.SimpleMAE3D
MAEDatasetWrapper = train_mae.MAEDatasetWrapper


def load_mae_model(checkpoint_path, device='cuda:0'):
    """Load MAE model from checkpoint"""
    model = SimpleMAE3D(in_channels=1, hidden_dim=256)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    mask_ratio = checkpoint['mask_ratio']
    epoch = checkpoint['epoch'] + 1
    
    return model, mask_ratio, epoch


def generate_residual_map_fast(mae_model, volume, mask_ratio, patch_size=(96, 96, 96), stride=64):
    """
    Generate residual map quickly using larger stride (less overlap = faster)
    
    Args:
        mae_model: Trained MAE model
        volume: Full volume [D, H, W]
        mask_ratio: Masking ratio
        patch_size: Patch size
        stride: Larger stride = faster but less accurate (64 for quick check)
    
    Returns:
        residual_map: [D, H, W]
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    residual_accumulator = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)
    
    # Generate grid with larger stride for speed
    d_positions = list(range(0, max(1, D - pd + 1), stride))
    h_positions = list(range(0, max(1, H - ph + 1), stride))
    w_positions = list(range(0, max(1, W - pw + 1), stride))
    
    # Add final positions
    if d_positions[-1] + pd < D:
        d_positions.append(D - pd)
    if h_positions[-1] + ph < H:
        h_positions.append(H - ph)
    if w_positions[-1] + pw < W:
        w_positions.append(W - pw)
    
    total_patches = len(d_positions) * len(h_positions) * len(w_positions)
    
    patches_processed = 0
    
    with torch.no_grad():
        for d_start in d_positions:
            for h_start in h_positions:
                for w_start in w_positions:
                    # Extract patch
                    patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    
                    if patch.shape != patch_size:
                        continue
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                    patch_tensor_cpu = patch_tensor[0, 0]
                    
                    # Normalize (match training preprocessing)
                    brain_threshold = torch.quantile(patch_tensor_cpu.flatten(), 0.01)
                    brain_mask = patch_tensor_cpu > brain_threshold
                    brain_voxels = patch_tensor_cpu[brain_mask]
                    
                    if brain_voxels.numel() > 10:
                        brain_mean = brain_voxels.mean()
                        brain_std = brain_voxels.std()
                        
                        if brain_std > 1e-6:
                            patch_norm = torch.zeros_like(patch_tensor)
                            patch_norm[0, 0][brain_mask] = (patch_tensor_cpu[brain_mask] - brain_mean) / brain_std
                            patch_norm = torch.clamp(patch_norm, -3, 3) / 3.0
                        else:
                            continue
                    else:
                        continue
                    
                    patch_norm = patch_norm.to(mae_model.encoder[0].weight.device)
                    
                    # Generate residual (single iteration for speed)
                    reconstructed, _ = mae_model(patch_norm, mask_ratio=mask_ratio)
                    residual = torch.abs(patch_norm - reconstructed)
                    residual_patch = residual.cpu().numpy()[0, 0]
                    
                    # Add to accumulator
                    residual_accumulator[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += residual_patch
                    count_map[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += 1.0
                    
                    patches_processed += 1
    
    # Average overlapping regions
    residual_map = residual_accumulator / (count_map + 1e-6)
    
    return residual_map, patches_processed, total_patches


def evaluate_volume(volume, mask, residual_50, residual_75):
    """
    Evaluate MAE residual detection quality on a single volume
    
    Returns:
        dict with metrics
    """
    # Combine residuals
    combined_residual = 0.5 * residual_50 + 0.5 * residual_75
    
    # Get lesion mask
    lesion_voxels = (mask > 0)
    normal_voxels = (mask == 0)
    
    if lesion_voxels.sum() == 0:
        # No lesion in this volume
        return None
    
    # Compute contrast
    residual_in_lesion = combined_residual[lesion_voxels].mean()
    residual_outside = combined_residual[normal_voxels].mean()
    contrast_ratio = residual_in_lesion / (residual_outside + 1e-6)
    
    # Compute DSC at multiple thresholds
    thresholds = [
        np.percentile(combined_residual, 70),
        np.percentile(combined_residual, 75),
        np.percentile(combined_residual, 80),
        np.percentile(combined_residual, 85),
        np.percentile(combined_residual, 90),
        np.percentile(combined_residual, 95)
    ]
    
    best_dsc = 0
    best_precision = 0
    best_recall = 0
    
    for thresh in thresholds:
        pred = (combined_residual > thresh).astype(np.uint8)
        
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum()
        
        if union > 0:
            dsc = (2.0 * intersection) / union
            
            tp = intersection
            fp = pred.sum() - intersection
            fn = mask.sum() - intersection
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            
            if dsc > best_dsc:
                best_dsc = dsc
                best_precision = precision
                best_recall = recall
    
    return {
        'contrast_ratio': contrast_ratio,
        'dsc': best_dsc,
        'precision': best_precision,
        'recall': best_recall,
        'lesion_volume': int(lesion_voxels.sum()),
        'residual_in_lesion': residual_in_lesion,
        'residual_outside': residual_outside
    }


def main():
    parser = argparse.ArgumentParser(description='Quick MAE performance check')
    parser.add_argument('--mae-50-checkpoint', type=str, required=True,
                       help='Path to MAE 50%% checkpoint')
    parser.add_argument('--mae-75-checkpoint', type=str, required=True,
                       help='Path to MAE 75%% checkpoint')
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=10,
                       help='Number of validation cases to test')
    parser.add_argument('--stride', type=int, default=64,
                       help='Stride for patches (larger = faster, 64 for quick check)')
    parser.add_argument('--patch-size', type=int, default=96)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print("QUICK MAE PERFORMANCE CHECK")
    print("="*70)
    
    # Load MAE models
    print("\nLoading MAE models...")
    mae_50, mask_ratio_50, epoch_50 = load_mae_model(args.mae_50_checkpoint, device)
    mae_75, mask_ratio_75, epoch_75 = load_mae_model(args.mae_75_checkpoint, device)
    
    print(f"✓ MAE 50%: Epoch {epoch_50}")
    print(f"✓ MAE 75%: Epoch {epoch_75}")
    print(f"  Stride: {args.stride} (larger = faster)")
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
    
    # Select cases with lesions
    volumes_info = []
    for idx in range(num_volumes):
        vol_info = val_dataset.base_dataset.get_volume_info(idx)
        lesion_size = vol_info['mask'].sum()
        if lesion_size > 0:  # Only cases with lesions
            volumes_info.append((idx, lesion_size, vol_info['case_id']))
    
    # Sort by lesion size and take top N
    volumes_info.sort(key=lambda x: x[1], reverse=True)
    selected_volumes = volumes_info[:min(args.num_cases, len(volumes_info))]
    
    print(f"Testing on {len(selected_volumes)} cases with lesions...\n")
    
    # Process each case
    all_results = []
    
    for vol_idx, lesion_size, case_id in tqdm(selected_volumes, desc='Processing volumes'):
        vol_info = val_dataset.base_dataset.get_volume_info(vol_idx)
        volume = vol_info['volume']
        mask = vol_info['mask']
        
        # Generate residual maps (fast mode)
        residual_50, p1, t1 = generate_residual_map_fast(
            mae_50, volume, mask_ratio_50,
            patch_size=(args.patch_size, args.patch_size, args.patch_size),
            stride=args.stride
        )
        
        residual_75, p2, t2 = generate_residual_map_fast(
            mae_75, volume, mask_ratio_75,
            patch_size=(args.patch_size, args.patch_size, args.patch_size),
            stride=args.stride
        )
        
        # Evaluate
        result = evaluate_volume(volume, mask, residual_50, residual_75)
        
        if result is not None:
            result['case_id'] = case_id
            result['patches_processed'] = p1
            all_results.append(result)
    
    # Compute summary statistics
    if len(all_results) == 0:
        print("\n⚠️ No valid results!")
        return
    
    avg_contrast = np.mean([r['contrast_ratio'] for r in all_results])
    avg_dsc = np.mean([r['dsc'] for r in all_results])
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    
    print("\n" + "="*70)
    print(f"MAE PERFORMANCE AT EPOCH {epoch_50}")
    print("="*70)
    print(f"\n📊 Results (Average across {len(all_results)} cases):")
    print(f"   Contrast Ratio: {avg_contrast:.2f}x")
    print(f"   DSC (Dice):     {avg_dsc:.3f}")
    print(f"   Precision:      {avg_precision:.3f}")
    print(f"   Recall:         {avg_recall:.3f}")
    
    print(f"\n💡 Interpretation:")
    if avg_contrast > 2.0:
        print(f"   ✅ EXCELLENT! Lesions are {avg_contrast:.1f}x brighter than normal brain!")
        print(f"   MAE has learned to detect abnormalities well.")
        print(f"   Expected GRCSF improvement: +6-8% DSC")
    elif avg_contrast > 1.5:
        print(f"   ✅ GOOD! Lesions are {avg_contrast:.1f}x brighter.")
        print(f"   MAE provides useful guidance for segmentation.")
        print(f"   Expected GRCSF improvement: +4-6% DSC")
    elif avg_contrast > 1.2:
        print(f"   ⚠️ MODERATE. Lesions are only {avg_contrast:.1f}x brighter.")
        print(f"   MAE is starting to learn, but not yet strong.")
        print(f"   Expected GRCSF improvement: +2-4% DSC")
        print(f"   → Continue training for better results!")
    else:
        print(f"   ❌ POOR. Contrast ratio {avg_contrast:.2f}x is too low.")
        print(f"   MAE is not detecting lesions as anomalies yet.")
        print(f"   → Training issue or needs more epochs!")
    
    print("\n📈 Per-Case Results:")
    print(f"{'Case ID':<30} {'Contrast':<10} {'DSC':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['case_id']:<30} {r['contrast_ratio']:<10.2f} {r['dsc']:<8.3f} {r['precision']:<10.3f} {r['recall']:<8.3f}")
    
    print("\n" + "="*70)
    print(f"✓ Quick check complete using stride={args.stride}")
    print(f"  Processing time: ~30 patches/volume (fast mode)")
    print(f"  For full quality results, use visualize_mae_residuals_FIXED.py with stride=48")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

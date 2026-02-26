"""
evaluate_isles_test.py

Evaluate fine-tuned models on ISLES test sets
CRITICAL: Resample predictions back to original ISLES dimensions
FIXED: Load ground truth directly from original ISLES files (no resampling)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.ndimage import zoom
import sys

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

# Import model classes from finetune script
from finetune_on_isles_DEBUG import (
    SegmentationModel,
    reconstruct_from_patches_with_count
)


def get_original_isles_info(case_id, isles_raw_dir='/home/pahm409/ISLES2022'):
    """
    Get original dimensions, affine, and ground truth from raw ISLES data
    
    Args:
        case_id: e.g., 'sub-strokecase0001'
        isles_raw_dir: Path to original ISLES2022 data
    
    Returns:
        dict with 'original_shape', 'original_affine', 'zoom_factors', 'ground_truth'
    """
    import nibabel as nib
    
    isles_raw = Path(isles_raw_dir)
    
    # ISLES2022 structure: ISLES2022/sub-strokecaseXXXX/sub-strokecaseXXXX_ses-0001_dwi.nii.gz
    dwi_file = isles_raw / case_id / f'{case_id}_ses-0001_dwi.nii.gz'
    msk_file = isles_raw / case_id / f'{case_id}_ses-0001_msk.nii.gz'
    
    if not dwi_file.exists():
        raise FileNotFoundError(f"Cannot find DWI file: {dwi_file}")
    if not msk_file.exists():
        raise FileNotFoundError(f"Cannot find mask file: {msk_file}")
    
    # Load to get dimensions and affine
    dwi_img = nib.load(dwi_file)
    msk_img = nib.load(msk_file)
    
    original_shape = dwi_img.shape
    original_affine = dwi_img.affine
    
    # Load original ground truth directly
    # CRITICAL: Compare BEFORE converting to uint8 (to avoid truncation)
    ground_truth = (msk_img.get_fdata() > 0.5).astype(np.uint8)
    
    # Verify shapes match
    if ground_truth.shape != original_shape:
        raise ValueError(f"Shape mismatch: DWI {original_shape} vs MSK {ground_truth.shape}")
    
    # Compute zoom factors (from resampled back to original)
    resampled_shape = np.array([197, 233, 189])
    original_shape_np = np.array(original_shape)
    zoom_factors = original_shape_np / resampled_shape
    
    return {
        'original_shape': tuple(original_shape),
        'original_affine': original_affine,
        'zoom_factors': zoom_factors,
        'resampled_shape': tuple(resampled_shape),
        'ground_truth': ground_truth,  # Original GT loaded directly
        'dwi_file': str(dwi_file),
        'msk_file': str(msk_file)
    }


def resample_to_original(prediction, original_info, method='nearest'):
    """
    Resample prediction from resampled space (197×233×189) back to original ISLES space
    
    Args:
        prediction: numpy array in resampled space (197×233×189)
        original_info: dict from get_original_isles_info()
        method: 'nearest' for binary, 'linear' for probabilities
    
    Returns:
        resampled prediction in original space
    """
    zoom_factors = original_info['zoom_factors']
    order = 0 if method == 'nearest' else 1  # 0=nearest, 1=linear
    
    resampled = zoom(prediction, zoom_factors, order=order)
    
    # Ensure exact dimensions (zoom can be off by 1 voxel)
    target_shape = original_info['original_shape']
    if resampled.shape != target_shape:
        # Crop or pad if needed
        resampled_fixed = np.zeros(target_shape, dtype=resampled.dtype)
        
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        
        resampled = resampled_fixed
    
    return resampled


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder, 
        num_classes=2,
        attention_type=attention_type,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, attention_type, deep_supervision


def evaluate_test_set(model, test_dataset, device, patch_size=(96, 96, 96), 
                     deep_supervision=False, save_nifti=False, output_dir=None,
                     isles_raw_dir='/home/pahm409/ISLES2022'):
    """
    Evaluate model on test set
    CRITICAL: Resample predictions back to original ISLES dimensions
    FIXED: Load ground truth directly from original ISLES files
    """
    import nibabel as nib
    
    model.eval()
    
    num_volumes = len(test_dataset.volumes)
    
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*70}")
    print(f"Test volumes: {num_volumes}")
    print(f"Total patches: {len(test_dataset)}")
    print(f"ISLES raw directory: {isles_raw_dir}")
    
    if num_volumes == 0:
        return 0.0, [], []
    
    # Create DataLoader with safe settings (avoid multiprocessing issues)
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 to avoid hanging issues
        pin_memory=True
    )
    
    # Collect predictions by volume
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("Processing test patches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            images = batch['image'].to(device)
            centers = batch['center'].cpu().numpy()
            vol_indices = batch['vol_idx'].cpu().numpy()
            
            with autocast():
                outputs = model(images)
                if deep_supervision:
                    outputs = outputs[0]
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(images)):
                vol_idx = vol_indices[i]
                volume_data[vol_idx]['centers'].append(centers[i])
                volume_data[vol_idx]['preds'].append(preds[i])
    
    # Reconstruct volumes
    all_dscs_original = []   # DSC in original ISLES space (MAIN METRIC)
    results_per_case = []
    
    print("Reconstructing test volumes...")
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = test_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        # ====================================================================
        # STEP 1: Reconstruct in resampled space (197×233×189)
        # ====================================================================
        reconstructed_resampled, count_map = reconstruct_from_patches_with_count(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        pred_binary_resampled = (reconstructed_resampled > 0.5).astype(np.uint8)
        
        # ====================================================================
        # STEP 2: Get original ISLES data (dimensions + ground truth)
        # ====================================================================
        try:
            original_info = get_original_isles_info(case_id, isles_raw_dir)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Cannot evaluate {case_id} - skipping this case")
            continue  # Skip this case entirely instead of using fallback
        
        # ====================================================================
        # STEP 3: Resample prediction back to original space
        # ====================================================================
        pred_prob_original = resample_to_original(
            reconstructed_resampled, 
            original_info, 
            method='linear'
        )
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        
        # ====================================================================
        # STEP 4: Get ground truth from ORIGINAL ISLES files (NOT resampled)
        # ====================================================================
        mask_gt_original = original_info['ground_truth']  # Already loaded and binarized
        
        # ====================================================================
        # STEP 5: Compute DSC in original space
        # ====================================================================
        intersection_original = (pred_binary_original * mask_gt_original).sum()
        union_original = pred_binary_original.sum() + mask_gt_original.sum()
        dsc_original = (2.0 * intersection_original) / union_original if union_original > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
        
        all_dscs_original.append(dsc_original)
        
        coverage = (count_map > 0).sum() / count_map.size * 100
        
        results_per_case.append({
            'case_id': case_id,
            'dsc_original': float(dsc_original),
            'original_shape': original_info['original_shape'],
            'resampled_shape': vol_info['mask'].shape,
            'gt_volume_original': int(mask_gt_original.sum()),
            'pred_volume_original': int(pred_binary_original.sum()),
            'zoom_factors': original_info['zoom_factors'].tolist(),
            'coverage': float(coverage)
        })
        
        if vol_idx < 5:  # Print first 5
            print(f"\n  {case_id}:")
            print(f"    Original shape: {original_info['original_shape']}")
            print(f"    Resampled shape: {vol_info['mask'].shape}")
            print(f"    Zoom factors: {original_info['zoom_factors']}")
            print(f"    GT volume (original): {mask_gt_original.sum()} voxels")
            print(f"    Pred volume (original): {pred_binary_original.sum()} voxels")
            print(f"    DSC (original): {dsc_original:.4f}")
        
        # ====================================================================
        # STEP 6: Save NIfTI in ORIGINAL space
        # ====================================================================
        if save_nifti and output_dir:
            nifti_dir = Path(output_dir) / case_id
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            # Use original affine
            affine = original_info['original_affine']
            
            # Save in ORIGINAL space
            nib.save(nib.Nifti1Image(pred_binary_original, affine), 
                    nifti_dir / 'prediction.nii.gz')
            nib.save(nib.Nifti1Image(pred_prob_original.astype(np.float32), affine), 
                    nifti_dir / 'prediction_prob.nii.gz')
            nib.save(nib.Nifti1Image(mask_gt_original, affine), 
                    nifti_dir / 'ground_truth.nii.gz')
            
            # Also save resampled versions for debugging
            # Use affine from preprocessed data (if available)
            # This is a standard affine for 197×233×189 space
            affine_resampled = np.array([
                [-1, 0, 0, 90],
                [0, 1, 0, -126],
                [0, 0, 1, -72],
                [0, 0, 0, 1]
            ])
            
            nib.save(nib.Nifti1Image(pred_binary_resampled, affine_resampled), 
                    nifti_dir / 'prediction_resampled.nii.gz')
            
            # For debugging: resample GT to resampled space for comparison
            mask_gt_resampled = (vol_info['mask'] > 0).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask_gt_resampled, affine_resampled), 
                    nifti_dir / 'ground_truth_resampled.nii.gz')
    
    if len(all_dscs_original) == 0:
        print("\n❌ ERROR: No valid test cases were evaluated!")
        return 0.0, [], []
    
    mean_dsc_original = np.mean(all_dscs_original)
    
    print(f"\n📊 TEST SET RESULTS (ORIGINAL ISLES SPACE):")
    print(f"  {'='*66}")
    print(f"    Cases evaluated: {len(all_dscs_original)}/{num_volumes}")
    print(f"    Mean DSC: {mean_dsc_original:.4f} ± {np.std(all_dscs_original):.4f}")
    print(f"    Min DSC:  {np.min(all_dscs_original):.4f}")
    print(f"    Max DSC:  {np.max(all_dscs_original):.4f}")
    print(f"    Median:   {np.median(all_dscs_original):.4f}")
    print(f"  {'='*66}")
    print(f"{'='*70}\n")
    
    return mean_dsc_original, all_dscs_original, results_per_case


def evaluate_all_folds(
    finetuned_dir,
    isles_preprocessed_dir,
    isles_raw_dir,
    isles_splits_file,
    output_dir,
    save_nifti=False
):
    """Evaluate all 5 folds on their test sets"""
    
    device = torch.device('cuda:0')
    finetuned_dir = Path(finetuned_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_fold_results = []
    
    for fold in range(5):
        print(f"\n{'#'*80}")
        print(f"# EVALUATING FOLD {fold}")
        print(f"{'#'*80}\n")
        
        # Find the best checkpoint for this fold
        fold_dirs = list(finetuned_dir.glob(f'fold_{fold}/finetune_*'))
        
        if not fold_dirs:
            print(f"❌ No checkpoint found for fold {fold}")
            continue
        
        # Use the most recent training run
        latest_fold_dir = sorted(fold_dirs)[-1]
        checkpoint_path = latest_fold_dir / 'checkpoints' / 'best_finetuned_model.pth'
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load model
        model, attention_type, deep_supervision = load_model_from_checkpoint(
            checkpoint_path, device
        )
        
        print(f"  Attention: {attention_type}")
        print(f"  Deep supervision: {deep_supervision}")
        
        # Load test dataset
        test_dataset = PatchDatasetWithCenters(
            preprocessed_dir=isles_preprocessed_dir,
            datasets=['ISLES2022_resampled'],
            split='test',  # ← IMPORTANT: Use test split
            splits_file=isles_splits_file,
            fold=fold,
            patch_size=(96, 96, 96),
            patches_per_volume=100,
            augment=False,
            lesion_focus_ratio=0.0,
            compute_lesion_bins=False
        )
        
        print(f"\nTest dataset: {len(test_dataset.volumes)} volumes")
        
        # Evaluate
        fold_output_dir = output_dir / f'fold_{fold}' if save_nifti else None
        
        mean_dsc, all_dscs, results_per_case = evaluate_test_set(
            model=model,
            test_dataset=test_dataset,
            device=device,
            patch_size=(96, 96, 96),
            deep_supervision=deep_supervision,
            save_nifti=save_nifti,
            output_dir=fold_output_dir,
            isles_raw_dir=isles_raw_dir
        )
        
        if len(all_dscs) == 0:
            print(f"❌ Fold {fold}: No valid results, skipping")
            continue
        
        # Save fold results
        fold_result = {
            'fold': fold,
            'checkpoint': str(checkpoint_path),
            'mean_dsc': float(mean_dsc),
            'std_dsc': float(np.std(all_dscs)),
            'median_dsc': float(np.median(all_dscs)),
            'min_dsc': float(np.min(all_dscs)),
            'max_dsc': float(np.max(all_dscs)),
            'num_test_cases': len(all_dscs),
            'all_dscs': [float(d) for d in all_dscs],
            'per_case_results': results_per_case
        }
        
        all_fold_results.append(fold_result)
        
        # Save individual fold results
        fold_json = output_dir / f'fold_{fold}_test_results.json'
        with open(fold_json, 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        print(f"✓ Saved fold {fold} results to: {fold_json}")
    
    if len(all_fold_results) == 0:
        print("\n❌ ERROR: No folds produced valid results!")
        return None
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS - ALL FOLDS (ORIGINAL ISLES SPACE)")
    print(f"{'='*80}\n")
    
    for result in all_fold_results:
        print(f"Fold {result['fold']}: DSC = {result['mean_dsc']:.4f} ± {result['std_dsc']:.4f} "
              f"({result['num_test_cases']} cases)")
    
    # Overall statistics
    all_fold_means = [r['mean_dsc'] for r in all_fold_results]
    overall_mean = np.mean(all_fold_means)
    overall_std = np.std(all_fold_means)
    
    print(f"\n{'='*80}")
    print(f"OVERALL TEST PERFORMANCE ({len(all_fold_results)}-FOLD CV):")
    print(f"  Mean DSC: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"{'='*80}\n")
    
    # Save summary
    summary = {
        'overall_mean_dsc': float(overall_mean),
        'overall_std_dsc': float(overall_std),
        'num_folds_evaluated': len(all_fold_results),
        'note': 'DSC values are in original ISLES space using ground truth loaded directly from raw files',
        'fold_results': all_fold_results
    }
    
    summary_file = output_dir / 'test_results_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned ISLES models on test sets')
    
    parser.add_argument('--finetuned-dir', type=str, required=True,
                       help='Directory containing fine-tuned models')
    parser.add_argument('--isles-preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='ISLES preprocessed directory (resampled)')
    parser.add_argument('--isles-raw-dir', type=str,
                       default='/home/pahm409/ISLES2022',
                       help='ISLES original raw directory (for original dimensions and GT)')
    parser.add_argument('--splits-file', type=str,
                       default='isles_splits_5fold_resampled.json',
                       help='Splits JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/isles_test_results',
                       help='Output directory for results')
    parser.add_argument('--save-nifti', action='store_true',
                       help='Save NIfTI predictions in ORIGINAL ISLES space')
    
    args = parser.parse_args()
    
    evaluate_all_folds(
        finetuned_dir=args.finetuned_dir,
        isles_preprocessed_dir=args.isles_preprocessed_dir,
        isles_raw_dir=args.isles_raw_dir,
        isles_splits_file=args.splits_file,
        output_dir=args.output_dir,
        save_nifti=args.save_nifti
    )

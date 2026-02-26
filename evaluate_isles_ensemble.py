"""
evaluate_isles_ensemble.py

Ensemble evaluation combining multiple pre-trained models
CRITICAL: Each model captures different lesion patterns
- Random Init: Better at certain lesion types
- SimCLR: Better at others
- Ensemble: Combines strengths of both

Strategy: Average probability maps from multiple models before thresholding

TESTING VERSION: Works with fold_0/run_1 only
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
import glob

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

# Import model classes
from finetune_on_isles_DEBUG import (
    SegmentationModel,
    reconstruct_from_patches_with_count
)


def get_original_isles_info(case_id, isles_raw_dir='/home/pahm409/ISLES2022_reg/ISLES2022'):
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
        'ground_truth': ground_truth,
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


def ensemble_evaluate_fold0_run1(
    checkpoint_paths,
    isles_preprocessed_dir,
    isles_splits_file,
    isles_raw_dir,
    output_dir,
    save_nifti=False
):
    """
    Ensemble evaluation for fold_0/run_1 ONLY (testing)
    
    Args:
        checkpoint_paths: List of checkpoint paths to ensemble
        isles_preprocessed_dir: ISLES preprocessed directory
        isles_splits_file: Splits JSON file
        isles_raw_dir: Original ISLES directory
        output_dir: Output directory
        save_nifti: Whether to save NIfTI files
    """
    import nibabel as nib
    
    device = torch.device('cuda:0')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fold = 0  # Fixed to fold 0
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE EVALUATION - FOLD 0 / RUN 1 (TESTING)")
    print(f"{'='*80}")
    print(f"Number of models in ensemble: {len(checkpoint_paths)}")
    print(f"{'='*80}\n")
    
    # Verify checkpoints exist
    print("Checking checkpoint paths:")
    for i, ckpt_path in enumerate(checkpoint_paths):
        ckpt = Path(ckpt_path)
        if ckpt.exists():
            print(f"  ✓ Model {i+1}: {ckpt_path}")
        else:
            print(f"  ❌ Model {i+1}: NOT FOUND - {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print()
    
    # Load test dataset for fold 0
    print("Loading test dataset for fold 0...")
    test_dataset = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='test',
        splits_file=isles_splits_file,
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset.volumes)} volumes, {len(test_dataset)} patches\n")
    
    if len(test_dataset.volumes) == 0:
        print("❌ ERROR: No test volumes!")
        return
    
    # Load all models
    models_info = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"Loading model {i+1}/{len(checkpoint_paths)}...")
        print(f"  Path: {ckpt_path}")
        
        model, attention_type, deep_supervision = load_model_from_checkpoint(ckpt_path, device)
        
        models_info.append({
            'model': model,
            'attention_type': attention_type,
            'deep_supervision': deep_supervision,
            'checkpoint_path': ckpt_path
        })
        
        print(f"  ✓ Loaded: Attention={attention_type}, Deep supervision={deep_supervision}\n")
    
    # Create DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    # Collect predictions from each model
    all_model_predictions = []
    
    for model_idx, model_info in enumerate(models_info):
        print(f"\n{'─'*70}")
        print(f"Model {model_idx+1}/{len(models_info)}: Generating predictions...")
        print(f"{'─'*70}")
        
        model = model_info['model']
        deep_supervision = model_info['deep_supervision']
        
        model.eval()
        
        volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Model {model_idx+1}'):
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
        
        all_model_predictions.append(volume_data)
        
        print(f"✓ Model {model_idx+1} predictions collected\n")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Ensemble predictions and evaluate
    print(f"\n{'='*80}")
    print(f"ENSEMBLING & RECONSTRUCTING VOLUMES")
    print(f"{'='*80}\n")
    
    all_dscs_original = []
    results_per_case = []
    individual_model_dscs = [[] for _ in range(len(models_info))]
    
    for vol_idx in tqdm(sorted(all_model_predictions[0].keys()), desc='Reconstructing'):
        vol_info = test_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        # Get original ISLES info
        try:
            original_info = get_original_isles_info(case_id, isles_raw_dir)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            continue
        
        # Reconstruct from each model in RESAMPLED space
        model_reconstructed_list = []
        
        for model_idx in range(len(models_info)):
            centers = np.array(all_model_predictions[model_idx][vol_idx]['centers'])
            preds = np.array(all_model_predictions[model_idx][vol_idx]['preds'])
            
            reconstructed, count_map = reconstruct_from_patches_with_count(
                preds, centers, vol_info['mask'].shape, patch_size=(96, 96, 96)
            )
            
            model_reconstructed_list.append(reconstructed)
        
        # ====================================================================
        # ENSEMBLE: Average probability maps across models
        # ====================================================================
        ensemble_reconstructed = np.mean(model_reconstructed_list, axis=0)
        
        # Resample ensemble to original space
        pred_prob_original = resample_to_original(
            ensemble_reconstructed,
            original_info,
            method='linear'
        )
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        
        # Get ground truth from original ISLES files
        mask_gt_original = original_info['ground_truth']
        
        # Compute ensemble DSC
        intersection = (pred_binary_original * mask_gt_original).sum()
        union = pred_binary_original.sum() + mask_gt_original.sum()
        dsc_ensemble = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
        
        all_dscs_original.append(dsc_ensemble)
        
        # Also compute individual model DSCs for analysis
        individual_dscs_this_case = []
        for model_idx, model_recon in enumerate(model_reconstructed_list):
            model_prob_original = resample_to_original(model_recon, original_info, method='linear')
            model_binary_original = (model_prob_original > 0.5).astype(np.uint8)
            
            intersection_m = (model_binary_original * mask_gt_original).sum()
            union_m = model_binary_original.sum() + mask_gt_original.sum()
            dsc_m = (2.0 * intersection_m) / union_m if union_m > 0 else (1.0 if model_binary_original.sum() == 0 else 0.0)
            
            individual_model_dscs[model_idx].append(dsc_m)
            individual_dscs_this_case.append(float(dsc_m))
        
        # Store results
        results_per_case.append({
            'case_id': case_id,
            'dsc_ensemble': float(dsc_ensemble),
            'dsc_individual_models': individual_dscs_this_case,
            'original_shape': original_info['original_shape'],
            'gt_volume_original': int(mask_gt_original.sum()),
            'pred_volume_original': int(pred_binary_original.sum()),
        })
        
        # Print ALL cases (since this is testing fold 0 only)
        print(f"\n{case_id}:")
        print(f"  Original shape: {original_info['original_shape']}")
        print(f"  GT volume: {mask_gt_original.sum()} voxels")
        print(f"  Ensemble DSC: {dsc_ensemble:.4f}")
        for model_idx in range(len(models_info)):
            print(f"    Model {model_idx+1} DSC: {individual_dscs_this_case[model_idx]:.4f}")
        
        # Save NIfTI if requested
        if save_nifti and output_dir:
            nifti_dir = output_dir / 'fold_0' / case_id
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            affine = original_info['original_affine']
            
            # Save ensemble predictions
            nib.save(nib.Nifti1Image(pred_binary_original, affine), 
                    nifti_dir / 'prediction_ensemble.nii.gz')
            nib.save(nib.Nifti1Image(pred_prob_original.astype(np.float32), affine), 
                    nifti_dir / 'prediction_ensemble_prob.nii.gz')
            nib.save(nib.Nifti1Image(mask_gt_original, affine), 
                    nifti_dir / 'ground_truth.nii.gz')
            
            # Save individual model predictions for analysis
            for model_idx, model_recon in enumerate(model_reconstructed_list):
                model_prob_original = resample_to_original(model_recon, original_info, method='linear')
                model_binary_original = (model_prob_original > 0.5).astype(np.uint8)
                
                nib.save(nib.Nifti1Image(model_binary_original, affine), 
                        nifti_dir / f'prediction_model{model_idx+1}.nii.gz')
    
    if len(all_dscs_original) == 0:
        print("\n❌ ERROR: No valid test cases were evaluated!")
        return
    
    # Print results
    mean_dsc_ensemble = np.mean(all_dscs_original)
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE RESULTS - FOLD 0 / RUN 1")
    print(f"{'='*80}")
    print(f"Cases evaluated: {len(all_dscs_original)}")
    print(f"\nENSEMBLE:")
    print(f"  Mean DSC: {mean_dsc_ensemble:.4f} ± {np.std(all_dscs_original):.4f}")
    print(f"  Min DSC:  {np.min(all_dscs_original):.4f}")
    print(f"  Max DSC:  {np.max(all_dscs_original):.4f}")
    print(f"  Median:   {np.median(all_dscs_original):.4f}")
    
    print(f"\nINDIVIDUAL MODELS:")
    for model_idx in range(len(models_info)):
        mean_individual = np.mean(individual_model_dscs[model_idx])
        std_individual = np.std(individual_model_dscs[model_idx])
        print(f"  Model {model_idx+1}: {mean_individual:.4f} ± {std_individual:.4f}")
    
    best_individual = max([np.mean(individual_model_dscs[i]) for i in range(len(models_info))])
    improvement = mean_dsc_ensemble - best_individual
    print(f"\nEnsemble improvement over best individual: {improvement:+.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'fold': 0,
        'run': 1,
        'checkpoints': checkpoint_paths,
        'num_models_ensembled': len(checkpoint_paths),
        'mean_dsc_ensemble': float(mean_dsc_ensemble),
        'std_dsc_ensemble': float(np.std(all_dscs_original)),
        'median_dsc_ensemble': float(np.median(all_dscs_original)),
        'min_dsc_ensemble': float(np.min(all_dscs_original)),
        'max_dsc_ensemble': float(np.max(all_dscs_original)),
        'individual_model_mean_dscs': [float(np.mean(individual_model_dscs[i])) for i in range(len(models_info))],
        'individual_model_std_dscs': [float(np.std(individual_model_dscs[i])) for i in range(len(models_info))],
        'ensemble_improvement': float(improvement),
        'num_test_cases': len(all_dscs_original),
        'all_dscs': [float(d) for d in all_dscs_original],
        'per_case_results': results_per_case
    }
    
    results_file = output_dir / 'fold_0_run_1_ensemble_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble evaluation for fold_0/run_1')
    
    parser.add_argument('--checkpoint-paths', nargs='+', required=True,
                       help='List of checkpoint paths to ensemble')
    parser.add_argument('--isles-preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='ISLES preprocessed directory (resampled)')
    parser.add_argument('--isles-raw-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/ISLES2022',
                       help='ISLES original raw directory')
    parser.add_argument('--splits-file', type=str,
                       default='isles_splits_5fold_resampled.json',
                       help='Splits JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/isles_ensemble_fold0_run1',
                       help='Output directory for results')
    parser.add_argument('--save-nifti', action='store_true',
                       help='Save NIfTI predictions')
    
    args = parser.parse_args()
    
    ensemble_evaluate_fold0_run1(
        checkpoint_paths=args.checkpoint_paths,
        isles_preprocessed_dir=args.isles_preprocessed_dir,
        isles_splits_file=args.splits_file,
        isles_raw_dir=args.isles_raw_dir,
        output_dir=args.output_dir,
        save_nifti=args.save_nifti
    )

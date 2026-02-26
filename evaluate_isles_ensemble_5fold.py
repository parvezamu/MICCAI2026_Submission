"""
evaluate_isles_ensemble_5fold.py

Ensemble evaluation for all 5 folds
Combines Random Init + SimCLR fine-tuned models
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

# Import model classes
from finetune_on_isles_DEBUG import (
    SegmentationModel,
    reconstruct_from_patches_with_count
)


def get_original_isles_info(case_id, isles_raw_dir='/home/pahm409/ISLES2022_reg/ISLES2022'):
    """
    Get original dimensions, affine, and ground truth from raw ISLES data
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


def ensemble_evaluate_single_fold(
    random_init_checkpoint,
    simclr_checkpoint,
    test_dataset,
    device,
    fold,
    patch_size=(96, 96, 96),
    save_nifti=False,
    output_dir=None,
    isles_raw_dir='/home/pahm409/ISLES2022'
):
    """
    Ensemble evaluation for a single fold
    """
    import nibabel as nib
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE EVALUATION - FOLD {fold}")
    print(f"{'='*80}")
    print(f"Random Init: {random_init_checkpoint}")
    print(f"SimCLR:      {simclr_checkpoint}")
    print(f"Test volumes: {len(test_dataset.volumes)}")
    print(f"{'='*80}\n")
    
    if len(test_dataset.volumes) == 0:
        print("❌ ERROR: No test volumes!")
        return 0.0, [], []
    
    # Load both models
    checkpoint_paths = [random_init_checkpoint, simclr_checkpoint]
    models_info = []
    
    for i, ckpt_path in enumerate(checkpoint_paths):
        model_name = "Random Init" if i == 0 else "SimCLR"
        print(f"Loading {model_name}...")
        
        model, attention_type, deep_supervision = load_model_from_checkpoint(ckpt_path, device)
        
        models_info.append({
            'model': model,
            'attention_type': attention_type,
            'deep_supervision': deep_supervision,
            'checkpoint_path': ckpt_path,
            'name': model_name
        })
        
        print(f"  ✓ Loaded\n")
    
    # Create DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Collect predictions from each model
    all_model_predictions = []
    
    for model_idx, model_info in enumerate(models_info):
        print(f"\n{'─'*70}")
        print(f"{model_info['name']}: Generating predictions...")
        print(f"{'─'*70}")
        
        model = model_info['model']
        deep_supervision = model_info['deep_supervision']
        
        model.eval()
        
        volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=model_info['name']):
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
        torch.cuda.empty_cache()
    
    # Ensemble predictions and evaluate
    print(f"\n{'='*80}")
    print(f"ENSEMBLING & RECONSTRUCTING VOLUMES")
    print(f"{'='*80}\n")
    
    all_dscs_original = []
    results_per_case = []
    individual_model_dscs = [[], []]  # [Random Init, SimCLR]
    
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
        
        for model_idx in range(2):
            centers = np.array(all_model_predictions[model_idx][vol_idx]['centers'])
            preds = np.array(all_model_predictions[model_idx][vol_idx]['preds'])
            
            reconstructed, count_map = reconstruct_from_patches_with_count(
                preds, centers, vol_info['mask'].shape, patch_size=patch_size
            )
            
            model_reconstructed_list.append(reconstructed)
        
        # ENSEMBLE: Average probability maps
        ensemble_reconstructed = np.mean(model_reconstructed_list, axis=0)
        
        # Resample ensemble to original space
        pred_prob_original = resample_to_original(
            ensemble_reconstructed,
            original_info,
            method='linear'
        )
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        
        # Get ground truth
        mask_gt_original = original_info['ground_truth']
        
        # Compute ensemble DSC
        intersection = (pred_binary_original * mask_gt_original).sum()
        union = pred_binary_original.sum() + mask_gt_original.sum()
        dsc_ensemble = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
        
        all_dscs_original.append(dsc_ensemble)
        
        # Compute individual model DSCs
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
            'dsc_random_init': individual_dscs_this_case[0],
            'dsc_simclr': individual_dscs_this_case[1],
            'original_shape': original_info['original_shape'],
            'gt_volume_original': int(mask_gt_original.sum()),
            'pred_volume_original': int(pred_binary_original.sum()),
        })
        
        # Save NIfTI if requested
        if save_nifti and output_dir:
            nifti_dir = Path(output_dir) / f'fold_{fold}' / case_id
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            affine = original_info['original_affine']
            
            nib.save(nib.Nifti1Image(pred_binary_original, affine), 
                    nifti_dir / 'prediction_ensemble.nii.gz')
            nib.save(nib.Nifti1Image(pred_prob_original.astype(np.float32), affine), 
                    nifti_dir / 'prediction_ensemble_prob.nii.gz')
            nib.save(nib.Nifti1Image(mask_gt_original, affine), 
                    nifti_dir / 'ground_truth.nii.gz')
    
    if len(all_dscs_original) == 0:
        print("\n❌ ERROR: No valid test cases!")
        return 0.0, [], []
    
    # Print results
    mean_dsc_ensemble = np.mean(all_dscs_original)
    mean_random_init = np.mean(individual_model_dscs[0])
    mean_simclr = np.mean(individual_model_dscs[1])
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold} RESULTS")
    print(f"{'='*80}")
    print(f"Cases evaluated: {len(all_dscs_original)}")
    print(f"\nINDIVIDUAL MODELS:")
    print(f"  Random Init: {mean_random_init:.4f} ± {np.std(individual_model_dscs[0]):.4f}")
    print(f"  SimCLR:      {mean_simclr:.4f} ± {np.std(individual_model_dscs[1]):.4f}")
    print(f"\nENSEMBLE:")
    print(f"  Mean DSC: {mean_dsc_ensemble:.4f} ± {np.std(all_dscs_original):.4f}")
    print(f"  Min DSC:  {np.min(all_dscs_original):.4f}")
    print(f"  Max DSC:  {np.max(all_dscs_original):.4f}")
    print(f"  Median:   {np.median(all_dscs_original):.4f}")
    
    improvement = mean_dsc_ensemble - max(mean_random_init, mean_simclr)
    print(f"\nEnsemble improvement: {improvement:+.4f}")
    print(f"{'='*80}\n")
    
    return mean_dsc_ensemble, all_dscs_original, results_per_case, mean_random_init, mean_simclr


def ensemble_evaluate_all_folds(
    random_init_base_dir,
    simclr_base_dir,
    isles_preprocessed_dir,
    isles_splits_file,
    isles_raw_dir,
    output_dir,
    save_nifti=False
):
    """
    Ensemble evaluation across all 5 folds
    """
    device = torch.device('cuda:0')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_fold_results = []
    
    print(f"\n{'#'*80}")
    print(f"# ENSEMBLE EVALUATION - 5-FOLD CROSS-VALIDATION")
    print(f"# Random Init + SimCLR")
    print(f"{'#'*80}\n")
    
    for fold in range(5):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold}")
        print(f"{'#'*80}\n")
        
        # Find checkpoint paths
        random_init_dir = Path(random_init_base_dir) / f'fold_{fold}'
        simclr_dir = Path(simclr_base_dir) / f'fold_{fold}'
        
        # Find the finetune_* directory (should be only one per fold)
        random_init_subdirs = list(random_init_dir.glob('finetune_*'))
        simclr_subdirs = list(simclr_dir.glob('finetune_*'))
        
        if len(random_init_subdirs) == 0:
            print(f"❌ No Random Init checkpoint for fold {fold}")
            continue
        if len(simclr_subdirs) == 0:
            print(f"❌ No SimCLR checkpoint for fold {fold}")
            continue
        
        random_init_checkpoint = random_init_subdirs[0] / 'checkpoints' / 'best_finetuned_model.pth'
        simclr_checkpoint = simclr_subdirs[0] / 'checkpoints' / 'best_finetuned_model.pth'
        
        if not random_init_checkpoint.exists():
            print(f"❌ Random Init checkpoint not found: {random_init_checkpoint}")
            continue
        if not simclr_checkpoint.exists():
            print(f"❌ SimCLR checkpoint not found: {simclr_checkpoint}")
            continue
        
        # Load test dataset
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
        
        # Ensemble evaluate
        mean_dsc, all_dscs, results_per_case, mean_random, mean_simclr = ensemble_evaluate_single_fold(
            random_init_checkpoint=str(random_init_checkpoint),
            simclr_checkpoint=str(simclr_checkpoint),
            test_dataset=test_dataset,
            device=device,
            fold=fold,
            patch_size=(96, 96, 96),
            save_nifti=save_nifti,
            output_dir=output_dir,
            isles_raw_dir=isles_raw_dir
        )
        
        if len(all_dscs) == 0:
            print(f"❌ Fold {fold}: No valid results")
            continue
        
        # Save fold results
        fold_result = {
            'fold': fold,
            'random_init_checkpoint': str(random_init_checkpoint),
            'simclr_checkpoint': str(simclr_checkpoint),
            'mean_dsc_ensemble': float(mean_dsc),
            'mean_dsc_random_init': float(mean_random),
            'mean_dsc_simclr': float(mean_simclr),
            'std_dsc_ensemble': float(np.std(all_dscs)),
            'median_dsc_ensemble': float(np.median(all_dscs)),
            'min_dsc': float(np.min(all_dscs)),
            'max_dsc': float(np.max(all_dscs)),
            'num_test_cases': len(all_dscs),
            'all_dscs': [float(d) for d in all_dscs],
            'per_case_results': results_per_case
        }
        
        all_fold_results.append(fold_result)
        
        # Save individual fold results
        fold_json = output_dir / f'fold_{fold}_ensemble_results.json'
        with open(fold_json, 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        print(f"✓ Saved fold {fold} results to: {fold_json}")
    
    if len(all_fold_results) == 0:
        print("\n❌ ERROR: No folds produced valid results!")
        return None
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"FINAL ENSEMBLE RESULTS - ALL FOLDS")
    print(f"{'='*80}\n")
    
    for result in all_fold_results:
        print(f"Fold {result['fold']}:")
        print(f"  Random Init: {result['mean_dsc_random_init']:.4f}")
        print(f"  SimCLR:      {result['mean_dsc_simclr']:.4f}")
        print(f"  Ensemble:    {result['mean_dsc_ensemble']:.4f} ({result['num_test_cases']} cases)")
    
    # Overall statistics
    ensemble_means = [r['mean_dsc_ensemble'] for r in all_fold_results]
    random_means = [r['mean_dsc_random_init'] for r in all_fold_results]
    simclr_means = [r['mean_dsc_simclr'] for r in all_fold_results]
    
    overall_ensemble = np.mean(ensemble_means)
    overall_random = np.mean(random_means)
    overall_simclr = np.mean(simclr_means)
    
    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE ({len(all_fold_results)}-FOLD CV):")
    print(f"  Random Init: {overall_random:.4f} ± {np.std(random_means):.4f}")
    print(f"  SimCLR:      {overall_simclr:.4f} ± {np.std(simclr_means):.4f}")
    print(f"  Ensemble:    {overall_ensemble:.4f} ± {np.std(ensemble_means):.4f}")
    print(f"\n  Improvement: {overall_ensemble - max(overall_random, overall_simclr):+.4f}")
    print(f"{'='*80}\n")
    
    # Save summary
    summary = {
        'overall_ensemble_dsc': float(overall_ensemble),
        'overall_random_init_dsc': float(overall_random),
        'overall_simclr_dsc': float(overall_simclr),
        'std_ensemble': float(np.std(ensemble_means)),
        'std_random_init': float(np.std(random_means)),
        'std_simclr': float(np.std(simclr_means)),
        'num_folds_evaluated': len(all_fold_results),
        'note': 'Ensemble of Random Init + SimCLR fine-tuned on ISLES2022',
        'fold_results': all_fold_results
    }
    
    summary_file = output_dir / 'ensemble_5fold_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble evaluation for all 5 folds')
    
    parser.add_argument('--random-init-dir', type=str,
                       default='/home/pahm409/finetuned_on_isles_5fold',
                       help='Base directory for Random Init fine-tuned models')
    parser.add_argument('--simclr-dir', type=str,
                       default='/home/pahm409/finetuned_on_isles_5fold1',
                       help='Base directory for SimCLR fine-tuned models')
    parser.add_argument('--isles-preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='ISLES preprocessed directory')
    parser.add_argument('--isles-raw-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/ISLES2022',
                       help='ISLES original raw directory')
    parser.add_argument('--splits-file', type=str,
                       default='isles_splits_5fold_resampled.json',
                       help='Splits JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/isles_ensemble_5fold_final',
                       help='Output directory')
    parser.add_argument('--save-nifti', action='store_true',
                       help='Save NIfTI predictions')
    
    args = parser.parse_args()
    
    ensemble_evaluate_all_folds(
        random_init_base_dir=args.random_init_dir,
        simclr_base_dir=args.simclr_dir,
        isles_preprocessed_dir=args.isles_preprocessed_dir,
        isles_splits_file=args.splits_file,
        isles_raw_dir=args.isles_raw_dir,
        output_dir=args.output_dir,
        save_nifti=args.save_nifti
    )

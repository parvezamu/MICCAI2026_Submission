"""
evaluate_isles_test.py

Evaluate models on ISLES test sets
UPDATED: Handles both fine-tuned and scratch-trained DWI models
CRITICAL: Uses correct splits file for each model type
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
    """Get original dimensions, affine, and ground truth from raw ISLES data"""
    import nibabel as nib
    
    isles_raw = Path(isles_raw_dir)
    
    dwi_file = isles_raw / case_id / f'{case_id}_ses-0001_dwi.nii.gz'
    msk_file = isles_raw / case_id / f'{case_id}_ses-0001_msk.nii.gz'
    
    if not dwi_file.exists():
        raise FileNotFoundError(f"Cannot find DWI file: {dwi_file}")
    if not msk_file.exists():
        raise FileNotFoundError(f"Cannot find mask file: {msk_file}")
    
    dwi_img = nib.load(dwi_file)
    msk_img = nib.load(msk_file)
    
    original_shape = dwi_img.shape
    original_affine = dwi_img.affine
    
    ground_truth = (msk_img.get_fdata() > 0.5).astype(np.uint8)
    
    if ground_truth.shape != original_shape:
        raise ValueError(f"Shape mismatch: DWI {original_shape} vs MSK {ground_truth.shape}")
    
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
    """Resample prediction from resampled space back to original ISLES space"""
    zoom_factors = original_info['zoom_factors']
    order = 0 if method == 'nearest' else 1
    
    resampled = zoom(prediction, zoom_factors, order=order)
    
    target_shape = original_info['original_shape']
    if resampled.shape != target_shape:
        resampled_fixed = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        resampled = resampled_fixed
    
    return resampled


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint (handles both fine-tuned and scratch-trained models)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    fold = checkpoint.get('fold', 0)
    
    # Detect checkpoint type
    is_finetuned = 'finetuned_on' in checkpoint
    is_simclr = 'encoder_state_dict' in checkpoint and 'projector_state_dict' in checkpoint
    
    if is_simclr:
        print(f"  ⚠️  WARNING: This is a SimCLR checkpoint (encoder only)")
        print(f"     You should evaluate the fine-tuned model, not SimCLR pre-training!")
        raise ValueError("Cannot evaluate SimCLR checkpoint - use fine-tuned model instead")
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder, 
        num_classes=2,
        attention_type=attention_type,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    checkpoint_type = "Fine-tuned" if is_finetuned else "Scratch-trained"
    
    print(f"\n✓ Loaded {checkpoint_type} model")
    print(f"  Fold: {fold}")
    print(f"  Attention: {attention_type}")
    print(f"  Deep supervision: {deep_supervision}")
    
    if is_finetuned:
        print(f"  Fine-tuned on: {checkpoint.get('finetuned_on', 'N/A')}")
        print(f"  Pre-trained from: {checkpoint.get('pretrained_from', 'N/A')}")
    
    return model, attention_type, deep_supervision, fold, checkpoint_type


def evaluate_test_set(model, test_dataset, device, patch_size=(96, 96, 96), 
                     deep_supervision=False, save_nifti=False, output_dir=None,
                     isles_raw_dir='/home/pahm409/ISLES2022'):
    """Evaluate model on test set"""
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
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
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
    
    all_dscs_original = []
    results_per_case = []
    
    print("Reconstructing test volumes...")
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = test_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed_resampled, count_map = reconstruct_from_patches_with_count(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        pred_binary_resampled = (reconstructed_resampled > 0.5).astype(np.uint8)
        
        try:
            original_info = get_original_isles_info(case_id, isles_raw_dir)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"   Skipping {case_id}")
            continue
        
        pred_prob_original = resample_to_original(
            reconstructed_resampled, 
            original_info, 
            method='linear'
        )
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        
        mask_gt_original = original_info['ground_truth']
        
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
        
        if vol_idx < 5:
            print(f"\n  {case_id}:")
            print(f"    Shape: {original_info['original_shape']}")
            print(f"    GT: {mask_gt_original.sum()} voxels")
            print(f"    Pred: {pred_binary_original.sum()} voxels")
            print(f"    DSC: {dsc_original:.4f}")
        
        if save_nifti and output_dir:
            nifti_dir = Path(output_dir) / case_id
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            affine = original_info['original_affine']
            
            nib.save(nib.Nifti1Image(pred_binary_original, affine), 
                    nifti_dir / 'prediction.nii.gz')
            nib.save(nib.Nifti1Image(pred_prob_original.astype(np.float32), affine), 
                    nifti_dir / 'prediction_prob.nii.gz')
            nib.save(nib.Nifti1Image(mask_gt_original, affine), 
                    nifti_dir / 'ground_truth.nii.gz')
    
    if len(all_dscs_original) == 0:
        print("\n❌ ERROR: No valid test cases!")
        return 0.0, [], []
    
    mean_dsc_original = np.mean(all_dscs_original)
    
    print(f"\n📊 TEST RESULTS (ORIGINAL ISLES SPACE):")
    print(f"  {'='*66}")
    print(f"    Cases: {len(all_dscs_original)}/{num_volumes}")
    print(f"    Mean DSC: {mean_dsc_original:.4f} ± {np.std(all_dscs_original):.4f}")
    print(f"    Min: {np.min(all_dscs_original):.4f}")
    print(f"    Max: {np.max(all_dscs_original):.4f}")
    print(f"    Median: {np.median(all_dscs_original):.4f}")
    print(f"  {'='*66}")
    print(f"{'='*70}\n")
    
    return mean_dsc_original, all_dscs_original, results_per_case


def evaluate_single_checkpoint(
    checkpoint_path,
    isles_preprocessed_dir,
    isles_raw_dir,
    isles_splits_file,
    output_dir,
    save_nifti=False
):
    """Evaluate a single checkpoint on its test set"""
    
    device = torch.device('cuda:0')
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"EVALUATING CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load model
    model, attention_type, deep_supervision, fold, checkpoint_type = load_model_from_checkpoint(
        checkpoint_path, device
    )
    
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
    
    print(f"\n📁 Dataset Info:")
    print(f"  Splits file: {isles_splits_file}")
    print(f"  Test volumes: {len(test_dataset.volumes)}")
    
    if len(test_dataset.volumes) == 0:
        print(f"\n❌ ERROR: No test volumes loaded!")
        print(f"   Check if splits file '{isles_splits_file}' exists and contains fold {fold}")
        return None
    
    # Evaluate
    mean_dsc, all_dscs, results_per_case = evaluate_test_set(
        model=model,
        test_dataset=test_dataset,
        device=device,
        patch_size=(96, 96, 96),
        deep_supervision=deep_supervision,
        save_nifti=save_nifti,
        output_dir=output_dir,
        isles_raw_dir=isles_raw_dir
    )
    
    if len(all_dscs) == 0:
        print(f"❌ No valid results")
        return None
    
    # Save results
    result = {
        'checkpoint_type': checkpoint_type,
        'fold': fold,
        'checkpoint': str(checkpoint_path),
        'splits_file': isles_splits_file,
        'mean_dsc': float(mean_dsc),
        'std_dsc': float(np.std(all_dscs)),
        'median_dsc': float(np.median(all_dscs)),
        'min_dsc': float(np.min(all_dscs)),
        'max_dsc': float(np.max(all_dscs)),
        'num_test_cases': len(all_dscs),
        'all_dscs': [float(d) for d in all_dscs],
        'per_case_results': results_per_case
    }
    
    result_file = output_dir / f'fold_{fold}_test_results.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Results saved to: {result_file}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ISLES model on test set')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
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
                       default='/home/pahm409/isles_test_results',
                       help='Output directory')
    parser.add_argument('--save-nifti', action='store_true',
                       help='Save NIfTI predictions')
    
    args = parser.parse_args()
    
    evaluate_single_checkpoint(
        checkpoint_path=args.checkpoint,
        isles_preprocessed_dir=args.isles_preprocessed_dir,
        isles_raw_dir=args.isles_raw_dir,
        isles_splits_file=args.splits_file,
        output_dir=args.output_dir,
        save_nifti=args.save_nifti
    )

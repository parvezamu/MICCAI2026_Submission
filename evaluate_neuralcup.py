"""
evaluate_neuralcup.py

Evaluate model on NEURALCUP T1 dataset
Prints per-subject lesion and prediction volumes
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from scipy.ndimage import zoom
import sys

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from finetune_on_isles_DEBUG import SegmentationModel, reconstruct_from_patches_with_count
from torch.cuda.amp import autocast
import nibabel as nib


def load_original_info(case_id, info_file):
    """Load original dimensions for a case"""
    with open(info_file, 'r') as f:
        all_info = json.load(f)
    
    if case_id not in all_info:
        raise ValueError(f"Case {case_id} not found in {info_file}")
    
    info = all_info[case_id]
    
    return {
        'original_shape': tuple(info['original_shape']),
        'original_affine': np.array(info['original_affine']),
        'zoom_factors': np.array(info['zoom_factors']),
        'resampled_shape': (197, 233, 189)
    }


def resample_to_original(prediction, original_info, method='nearest'):
    """Resample from 197×233×189 back to original"""
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
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect checkpoint type
    has_model_state = 'model_state_dict' in checkpoint
    has_encoder_state = 'encoder_state_dict' in checkpoint
    has_projector = 'projector_state_dict' in checkpoint
    is_finetuned = 'finetuned_on' in checkpoint
    
    is_simclr = has_encoder_state and has_projector and not has_model_state
    
    if has_model_state:
        state_dict = checkpoint['model_state_dict']
        has_projection_head = any('projection_head' in k for k in state_dict.keys())
        has_decoder = any('decoder' in k for k in state_dict.keys())
        is_simclr_v2 = has_projection_head and not has_decoder
    else:
        is_simclr_v2 = False
    
    if is_simclr or is_simclr_v2:
        print(f"\n⚠️  ERROR: This is a SimCLR pre-training checkpoint!")
        print(f"   SimCLR checkpoints only contain the encoder (no decoder).")
        print(f"   You need to use a FINE-TUNED or SCRATCH-TRAINED checkpoint.")
        raise ValueError("Cannot evaluate SimCLR checkpoint - no decoder present")
    
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
    
    checkpoint_type = "Fine-tuned" if is_finetuned else "Scratch-trained"
    
    print(f"\n✓ Loaded {checkpoint_type} model")
    print(f"  Attention: {attention_type}")
    print(f"  Deep supervision: {deep_supervision}")
    
    if is_finetuned:
        print(f"  Fine-tuned on: {checkpoint.get('finetuned_on', 'N/A')}")
    
    return model, attention_type, deep_supervision, checkpoint_type


def load_volume(case_id, preprocessed_dir):
    """Load a single volume"""
    npz_file = Path(preprocessed_dir) / f"{case_id}.npz"
    data = np.load(npz_file)
    return {
        'case_id': case_id,
        'volume': data['volume'],
        'mask': data['mask']
    }



def extract_patches(volume, patch_size=(96, 96, 96), patches_per_volume=100):
    """Extract random patches from volume - ENSURE EXACT PATCH SIZE"""
    import random
    
    volume_shape = np.array(volume.shape)
    patch_size = np.array(patch_size)
    half_patch = patch_size // 2
    
    patches = []
    centers = []
    
    # CRITICAL: Ensure we can extract full-sized patches
    valid_min = half_patch
    valid_max = volume_shape - half_patch
    
    # If volume is too small, pad it first
    if np.any(volume_shape < patch_size):
        print(f"⚠️  Volume {volume_shape} too small for patches {patch_size}, padding...")
        pad_needed = np.maximum(patch_size - volume_shape, 0)
        pad_width = [(p//2, p - p//2) for p in pad_needed]
        volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        volume_shape = np.array(volume.shape)
        valid_min = half_patch
        valid_max = volume_shape - half_patch
    
    for _ in range(patches_per_volume):
        # CRITICAL: Sample center that allows full patch extraction
        center = np.array([
            random.randint(valid_min[0], valid_max[0]-1),
            random.randint(valid_min[1], valid_max[1]-1),
            random.randint(valid_min[2], valid_max[2]-1)
        ])
        
        lower = center - half_patch
        upper = center + half_patch
        
        # CRITICAL: Verify exact patch size
        patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        
        assert patch.shape == tuple(patch_size), f"Patch shape {patch.shape} != expected {patch_size}"
        
        patches.append(patch)
        centers.append(center)
    
    return np.array(patches), np.array(centers)


def extract_patches1(volume, patch_size=(96, 96, 96), patches_per_volume=100):
    """Extract random patches from volume"""
    import random
    
    volume_shape = np.array(volume.shape)
    half_patch = np.array(patch_size) // 2
    
    patches = []
    centers = []
    
    for _ in range(patches_per_volume):
        center = np.array([
            random.randint(half_patch[0], volume_shape[0] - half_patch[0]),
            random.randint(half_patch[1], volume_shape[1] - half_patch[1]),
            random.randint(half_patch[2], volume_shape[2] - half_patch[2])
        ])
        
        lower = center - half_patch
        upper = center + half_patch
        
        patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        patches.append(patch)
        centers.append(center)
    
    return np.array(patches), np.array(centers)


def evaluate_neuralcup(
    checkpoint_path,
    preprocessed_dir,
    info_file,
    output_dir,
    patch_size=(96, 96, 96),
    patches_per_volume=100,
    save_nifti=True
):
    """Evaluate on NEURALCUP T1 dataset"""
    
    device = torch.device('cuda:0')
    
    print(f"\n{'='*70}")
    print(f"LOADING MODEL")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    try:
        model, attention_type, deep_supervision, checkpoint_type = load_model_from_checkpoint(
            checkpoint_path, device
        )
    except ValueError as e:
        print(f"\n❌ {e}")
        return None
    
    # Get all cases
    preprocessed = Path(preprocessed_dir)
    all_cases = sorted([f.stem for f in preprocessed.glob("*.npz")])
    
    print(f"\n{'='*70}")
    print(f"NEURALCUP T1 DATASET")
    print(f"{'='*70}")
    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Found {len(all_cases)} cases")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dscs_original = []
    results = []
    
    print(f"\n{'='*70}")
    print(f"EVALUATING")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for case_id in tqdm(all_cases, desc="Evaluating NEURALCUP"):
            
            # Load volume
            vol_info = load_volume(case_id, preprocessed_dir)
            volume = vol_info['volume']
            mask_gt = vol_info['mask']
            
            # Extract patches
            patches, centers = extract_patches(
                volume, 
                patch_size=patch_size, 
                patches_per_volume=patches_per_volume
            )
            
            # Add channel dimension
            patches = patches[:, np.newaxis, :, :, :]
            
            # Predict in batches
            all_preds = []
            batch_size = 16
            
            for i in range(0, len(patches), batch_size):
                batch = torch.from_numpy(patches[i:i+batch_size]).float().to(device)
                
                with autocast():
                    outputs = model(batch)
                    if deep_supervision:
                        outputs = outputs[0]
                
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(preds)
            
            all_preds = np.concatenate(all_preds, axis=0)
            
            # Reconstruct in resampled space
            reconstructed_resampled, count_map = reconstruct_from_patches_with_count(
                all_preds, centers, mask_gt.shape, patch_size=patch_size
            )
            
            pred_binary_resampled = (reconstructed_resampled > 0.5).astype(np.uint8)
            mask_gt_resampled = (mask_gt > 0).astype(np.uint8)
            
            # Get original dimensions
            try:
                original_info = load_original_info(case_id, info_file)
            except Exception as e:
                print(f"\n⚠️  {case_id}: {e}")
                continue
            
            # Resample to original
            pred_prob_original = resample_to_original(
                reconstructed_resampled, 
                original_info, 
                method='linear'
            )
            pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
            
            mask_gt_original = resample_to_original(
                mask_gt_resampled.astype(np.float32),
                original_info,
                method='nearest'
            ).astype(np.uint8)
            
            # Compute DSC in original space
            intersection_original = (pred_binary_original * mask_gt_original).sum()
            union_original = pred_binary_original.sum() + mask_gt_original.sum()
            dsc_original = (2.0 * intersection_original) / union_original if union_original > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
            
            all_dscs_original.append(dsc_original)
            
            gt_volume = int(mask_gt_original.sum())
            pred_volume = int(pred_binary_original.sum())
            
            results.append({
                'case_id': case_id,
                'dsc_original': float(dsc_original),
                'original_shape': original_info['original_shape'],
                'gt_volume_voxels': gt_volume,
                'pred_volume_voxels': pred_volume,
                'volume_ratio': float(pred_volume / gt_volume) if gt_volume > 0 else 0.0
            })
            
            # Save NIfTI
            if save_nifti:
                nifti_dir = output_dir / case_id
                nifti_dir.mkdir(parents=True, exist_ok=True)
                
                affine = original_info['original_affine']
                
                nib.save(nib.Nifti1Image(pred_binary_original, affine), 
                        nifti_dir / 'prediction.nii.gz')
                nib.save(nib.Nifti1Image(pred_prob_original.astype(np.float32), affine), 
                        nifti_dir / 'prediction_prob.nii.gz')
                nib.save(nib.Nifti1Image(mask_gt_original, affine), 
                        nifti_dir / 'ground_truth.nii.gz')
    
    # Sort results by DSC for display
    results_sorted = sorted(results, key=lambda x: x['dsc_original'], reverse=True)
    
    # Print per-subject results
    print(f"\n{'='*70}")
    print(f"PER-SUBJECT RESULTS (Top 10 and Bottom 10)")
    print(f"{'='*70}")
    
    print(f"\n🏆 TOP 10:")
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:2d}. {r['case_id']:20s}  DSC={r['dsc_original']:.4f}  "
              f"GT={r['gt_volume_voxels']:6d}  Pred={r['pred_volume_voxels']:6d}  "
              f"Ratio={r['volume_ratio']:.2f}")
    
    print(f"\n⚠️  BOTTOM 10:")
    for i, r in enumerate(results_sorted[-10:], 1):
        print(f"{i:2d}. {r['case_id']:20s}  DSC={r['dsc_original']:.4f}  "
              f"GT={r['gt_volume_voxels']:6d}  Pred={r['pred_volume_voxels']:6d}  "
              f"Ratio={r['volume_ratio']:.2f}")
    
    # Summary statistics
    mean_dsc = np.mean(all_dscs_original)
    
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS (Original Space)")
    print(f"{'='*70}")
    print(f"  Checkpoint type:   {checkpoint_type}")
    print(f"  Total cases:       {len(all_dscs_original)}")
    print(f"  Mean DSC:          {mean_dsc:.4f} ± {np.std(all_dscs_original):.4f}")
    print(f"  Median DSC:        {np.median(all_dscs_original):.4f}")
    print(f"  Min DSC:           {np.min(all_dscs_original):.4f}")
    print(f"  Max DSC:           {np.max(all_dscs_original):.4f}")
    print(f"  Cases with DSC=0:  {sum(1 for d in all_dscs_original if d == 0)}")
    print(f"  Cases with DSC>0.5: {sum(1 for d in all_dscs_original if d > 0.5)}")
    print(f"{'='*70}\n")
    
    # Save results
    summary = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_type': checkpoint_type,
        'num_cases': len(all_dscs_original),
        'mean_dsc': float(mean_dsc),
        'std_dsc': float(np.std(all_dscs_original)),
        'median_dsc': float(np.median(all_dscs_original)),
        'min_dsc': float(np.min(all_dscs_original)),
        'max_dsc': float(np.max(all_dscs_original)),
        'per_case_results': results
    }
    
    results_file = output_dir / 'neuralcup_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on NEURALCUP T1 dataset')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled',
                       help='Preprocessed directory')
    parser.add_argument('--info-file', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled/original_dimensions.json',
                       help='Original dimensions JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/neuralcup_results',
                       help='Output directory')
    parser.add_argument('--no-nifti', action='store_true',
                       help='Skip saving NIfTI files')
    
    args = parser.parse_args()
    
    evaluate_neuralcup(
        checkpoint_path=args.checkpoint,
        preprocessed_dir=args.preprocessed_dir,
        info_file=args.info_file,
        output_dir=args.output_dir,
        save_nifti=not args.no_nifti
    )

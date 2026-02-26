"""
evaluate_arc.py

Evaluate model on ALL ARC T2w cases
CRITICAL: Convert predictions back to original T2w dimensions
UPDATED: Handles SimCLR, fine-tuned, and scratch-trained checkpoints
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


def load_arc_original_info(case_id, arc_info_file):
    """Load original dimensions for ARC case"""
    with open(arc_info_file, 'r') as f:
        all_info = json.load(f)
    
    if case_id not in all_info:
        raise ValueError(f"Case {case_id} not found in original_dimensions.json")
    
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
    
    # Ensure exact dimensions
    target_shape = original_info['original_shape']
    if resampled.shape != target_shape:
        resampled_fixed = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        resampled = resampled_fixed
    
    return resampled


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint
    Handles: SimCLR pre-training, fine-tuned models, and scratch-trained models
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect checkpoint type
    has_model_state = 'model_state_dict' in checkpoint
    has_encoder_state = 'encoder_state_dict' in checkpoint
    has_projector = 'projector_state_dict' in checkpoint
    is_finetuned = 'finetuned_on' in checkpoint
    
    # SimCLR checkpoint (encoder + projector, no decoder)
    is_simclr = has_encoder_state and has_projector and not has_model_state
    
    # Check if model_state_dict contains projection_head (another SimCLR format)
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
        print(f"\n   Expected checkpoint types:")
        print(f"   - Fine-tuned: Model trained on target task AFTER SimCLR")
        print(f"   - Scratch-trained: Model trained from random init on target task")
        raise ValueError("Cannot evaluate SimCLR checkpoint - no decoder present")
    
    # Regular checkpoint (fine-tuned or scratch-trained)
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
    
    # Determine checkpoint type
    if is_finetuned:
        checkpoint_type = "Fine-tuned"
        print(f"\n✓ Loaded Fine-tuned model")
        print(f"  Fine-tuned on: {checkpoint.get('finetuned_on', 'N/A')}")
        print(f"  Pre-trained from: {checkpoint.get('pretrained_from', 'N/A')}")
    else:
        checkpoint_type = "Scratch-trained"
        print(f"\n✓ Loaded Scratch-trained model")
    
    print(f"  Attention: {attention_type}")
    print(f"  Deep supervision: {deep_supervision}")
    
    return model, attention_type, deep_supervision, checkpoint_type


def load_arc_volume(case_id, arc_preprocessed_dir):
    """Load a single ARC volume"""
    npz_file = Path(arc_preprocessed_dir) / f"{case_id}.npz"
    data = np.load(npz_file)
    return {
        'case_id': case_id,
        'volume': data['volume'],
        'mask': data['mask']
    }


def extract_patches_from_volume(volume, patch_size=(96, 96, 96), patches_per_volume=100):
    """Extract patches from a volume"""
    import random
    
    volume_shape = np.array(volume.shape)
    half_patch = np.array(patch_size) // 2
    
    patches = []
    centers = []
    
    # Generate random patch centers
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


def evaluate_arc_all_cases(
    checkpoint_path,
    arc_preprocessed_dir,
    arc_info_file,
    output_dir,
    patch_size=(96, 96, 96),
    patches_per_volume=100,
    save_nifti=True
):
    """Evaluate ALL ARC cases"""
    
    device = torch.device('cuda:0')
    
    # Load model
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
    
    # Get all ARC cases
    arc_dir = Path(arc_preprocessed_dir)
    all_cases = sorted([f.stem for f in arc_dir.glob("*.npz")])
    
    print(f"\n{'='*70}")
    print(f"ARC T2w DATASET")
    print(f"{'='*70}")
    print(f"Found {len(all_cases)} ARC cases")
    print(f"Preprocessed dir: {arc_preprocessed_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dscs_resampled = []
    all_dscs_original = []
    results = []
    
    print(f"\n{'='*70}")
    print(f"EVALUATING")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for case_id in tqdm(all_cases, desc="Evaluating ARC"):
            
            # Load volume
            vol_info = load_arc_volume(case_id, arc_preprocessed_dir)
            volume = vol_info['volume']
            mask_gt = vol_info['mask']
            
            # Extract patches
            patches, centers = extract_patches_from_volume(
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
            
            # DSC in resampled space
            intersection_resampled = (pred_binary_resampled * mask_gt_resampled).sum()
            union_resampled = pred_binary_resampled.sum() + mask_gt_resampled.sum()
            dsc_resampled = (2.0 * intersection_resampled) / union_resampled if union_resampled > 0 else (1.0 if pred_binary_resampled.sum() == 0 else 0.0)
            
            all_dscs_resampled.append(dsc_resampled)
            
            # Get original dimensions
            try:
                original_info = load_arc_original_info(case_id, arc_info_file)
            except Exception as e:
                print(f"\n⚠️  {case_id}: {e}")
                all_dscs_original.append(dsc_resampled)
                results.append({
                    'case_id': case_id,
                    'dsc_resampled': float(dsc_resampled),
                    'dsc_original': float(dsc_resampled)
                })
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
            
            # DSC in original space
            intersection_original = (pred_binary_original * mask_gt_original).sum()
            union_original = pred_binary_original.sum() + mask_gt_original.sum()
            dsc_original = (2.0 * intersection_original) / union_original if union_original > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
            
            all_dscs_original.append(dsc_original)
            
            results.append({
                'case_id': case_id,
                'dsc_resampled': float(dsc_resampled),
                'dsc_original': float(dsc_original),
                'original_shape': original_info['original_shape'],
                'gt_volume_original': int(mask_gt_original.sum()),
                'pred_volume_original': int(pred_binary_original.sum())
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
    
    # Summary
    mean_dsc_original = np.mean(all_dscs_original)
    
    print(f"\n{'='*70}")
    print(f"ARC T2w EVALUATION RESULTS (Original Space)")
    print(f"{'='*70}")
    print(f"  Checkpoint type:  {checkpoint_type}")
    print(f"  Total cases:      {len(all_dscs_original)}")
    print(f"  Mean DSC:         {mean_dsc_original:.4f} ± {np.std(all_dscs_original):.4f}")
    print(f"  Median DSC:       {np.median(all_dscs_original):.4f}")
    print(f"  Min DSC:          {np.min(all_dscs_original):.4f}")
    print(f"  Max DSC:          {np.max(all_dscs_original):.4f}")
    print(f"{'='*70}\n")
    
    # Save results
    summary = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_type': checkpoint_type,
        'num_cases': len(all_dscs_original),
        'mean_dsc': float(mean_dsc_original),
        'std_dsc': float(np.std(all_dscs_original)),
        'median_dsc': float(np.median(all_dscs_original)),
        'min_dsc': float(np.min(all_dscs_original)),
        'max_dsc': float(np.max(all_dscs_original)),
        'all_results': results
    }
    
    results_file = output_dir / 'arc_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on ARC T2w dataset')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--arc-preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation/ARC_T2w_resampled',
                       help='ARC preprocessed directory')
    parser.add_argument('--arc-info-file', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation/ARC_T2w_resampled/original_dimensions.json',
                       help='Original dimensions JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/arc_evaluation_results',
                       help='Output directory')
    parser.add_argument('--no-nifti', action='store_true',
                       help='Skip saving NIfTI files')
    
    args = parser.parse_args()
    
    evaluate_arc_all_cases(
        checkpoint_path=args.checkpoint,
        arc_preprocessed_dir=args.arc_preprocessed_dir,
        arc_info_file=args.arc_info_file,
        output_dir=args.output_dir,
        save_nifti=not args.no_nifti
    )


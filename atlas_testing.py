"""
inference_atlas_test.py

Run inference on unlabeled ATLAS test data
Generate segmentation predictions without ground truth

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.cuda.amp import autocast
import sys

sys.path.append('/home/pahm409')
from corr1 import SegmentationModel
from models.resnet3d import resnet3d_18


def preprocess_volume(nifti_path):
    """
    Preprocess a single volume (same as training preprocessing)
    """
    
    # Load NIfTI
    nii = nib.load(nifti_path)
    volume = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    
    # Brain mask (1st percentile threshold)
    brain_mask = volume > np.percentile(volume, 1)
    
    # Clip at 99.5th percentile within brain
    brain_voxels = volume[brain_mask]
    if len(brain_voxels) > 0:
        clip_value = np.percentile(brain_voxels, 99.5)
        volume = np.clip(volume, 0, clip_value)
    
    # Z-score normalization within brain
    normalized = np.zeros_like(volume)
    brain_voxels = volume[brain_mask]
    
    if len(brain_voxels) > 0:
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        if std > 0:
            normalized[brain_mask] = (volume[brain_mask] - mean) / std
    
    return normalized, affine, volume.shape


def extract_patches_with_centers(volume, patch_size=(96, 96, 96), stride=48):
    """
    Extract patches with sliding window for complete coverage
    """
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    
    patches = []
    centers = []
    
    # Sliding window centers
    d_centers = list(range(pD // 2, D - pD // 2, stride))
    h_centers = list(range(pH // 2, H - pH // 2, stride))
    w_centers = list(range(pW // 2, W - pW // 2, stride))
    
    # Ensure we cover edges
    if d_centers and d_centers[-1] < D - pD // 2 - 1:
        d_centers.append(D - pD // 2 - 1)
    if h_centers and h_centers[-1] < H - pH // 2 - 1:
        h_centers.append(H - pH // 2 - 1)
    if w_centers and w_centers[-1] < W - pW // 2 - 1:
        w_centers.append(W - pW // 2 - 1)
    
    for d_c in d_centers:
        for h_c in h_centers:
            for w_c in w_centers:
                center = np.array([d_c, h_c, w_c])
                
                lower = center - np.array(patch_size) // 2
                upper = lower + np.array(patch_size)
                
                patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
                
                if patch.shape == tuple(patch_size):
                    patches.append(patch)
                    centers.append(center)
    
    return np.array(patches), np.array(centers)


def reconstruct_from_patches(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """
    Reconstruct full volume from patch predictions
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        # Boundary check
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            continue
        
        # Add prediction (lesion class probability)
        patch = patch_preds[i, 1, ...]
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    # Average overlapping predictions
    mask = count_map > 0
    reconstructed[mask] = reconstructed[mask] / count_map[mask]
    
    return reconstructed


def load_model(checkpoint_path, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(encoder, num_classes=2, 
                             attention_type=attention_type,
                             deep_supervision=deep_supervision)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
    if 'val_dsc_recon' in checkpoint:
        print(f"  Validation DSC: {checkpoint['val_dsc_recon']:.4f}")
    
    return model, deep_supervision


def run_inference(
    model_checkpoint,
    test_dir,
    output_dir,
    patch_size=(96, 96, 96),
    stride=48,
    batch_size=16
):
    """
    Run inference on all test volumes
    """
    
    device = torch.device('cuda:0')
    
    # Load model
    print("Loading model...")
    model, deep_supervision = load_model(model_checkpoint, device)
    
    # Find all test volumes
    test_dir = Path(test_dir)
    nifti_files = sorted(test_dir.glob("*.nii.gz"))
    
    print(f"\nFound {len(nifti_files)} test volumes")
    print(f"Output directory: {output_dir}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each volume
    print("="*80)
    print(f"{'Case ID':<40} {'Patches':>10} {'Lesion Voxels':>15}")
    print("="*80)
    
    for nifti_file in tqdm(nifti_files, desc="Processing volumes"):
        case_id = nifti_file.stem.replace('_space-MNI152NLin2009aSym_T1w', '')
        
        # Preprocess
        volume, affine, original_shape = preprocess_volume(nifti_file)
        
        # Extract patches
        patches, centers = extract_patches_with_centers(
            volume, patch_size=patch_size, stride=stride
        )
        
        # Predict on patches
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).unsqueeze(1).float().to(device)
                
                with autocast():
                    outputs = model(batch_tensor)
                    
                    # Use main output if deep supervision
                    if deep_supervision:
                        outputs = outputs[0]
                
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(preds)
        
        all_preds = np.concatenate(all_preds, axis=0)
        
        # Reconstruct
        reconstructed = reconstruct_from_patches(
            all_preds, centers, original_shape, patch_size=patch_size
        )
        
        # Threshold to get binary mask
        pred_binary = (reconstructed > 0.5).astype(np.uint8)
        lesion_voxels = pred_binary.sum()
        
        # Save results
        case_output_dir = output_path / case_id
        case_output_dir.mkdir(exist_ok=True)
        
        # Save probability map
        prob_nii = nib.Nifti1Image(reconstructed.astype(np.float32), affine)
        nib.save(prob_nii, case_output_dir / 'prediction_prob.nii.gz')
        
        # Save binary mask
        mask_nii = nib.Nifti1Image(pred_binary, affine)
        nib.save(mask_nii, case_output_dir / 'prediction_mask.nii.gz')
        
        # Save original (for visualization)
        orig_nii = nib.Nifti1Image(volume, affine)
        nib.save(orig_nii, case_output_dir / 'original.nii.gz')
        
        print(f"{case_id:<40} {len(patches):>10} {lesion_voxels:>15}")
    
    print("="*80)
    print(f"\n✓ Inference complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nFor each case:")
    print(f"  - prediction_prob.nii.gz  (probability map 0-1)")
    print(f"  - prediction_mask.nii.gz  (binary mask)")
    print(f"  - original.nii.gz         (preprocessed volume)")
    print("="*80)


if __name__ == '__main__':
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Run inference on unlabeled test data')
    
    parser.add_argument('--model-checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-dir', type=str, 
                       default='/home/pahm409/ATLAS_testing/test',
                       help='Directory with test NIfTI files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/ATLAS_testing_predictions',
                       help='Output directory for predictions')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--stride', type=int, default=48,
                       help='Stride for sliding window (smaller = more overlap)')
    parser.add_argument('--batch-size', type=int, default=16)
    
    args = parser.parse_args()
    
    # Find checkpoint if not specified
    if args.model_checkpoint is None:
        pattern = "/home/pahm409/ablation_ds_main_only/Random_Init/mkdc_ds/fold_0/run_0/exp_*/checkpoints/best_model.pth"
        matches = glob.glob(pattern)
        
        if not matches:
            print("Error: No checkpoint found!")
            print(f"Searched: {pattern}")
            print("\nPlease specify checkpoint with --model-checkpoint")
            exit(1)
        
        args.model_checkpoint = matches[0]
        print(f"Using checkpoint: {args.model_checkpoint}\n")
    
    # Verify test directory exists
    if not Path(args.test_dir).exists():
        print(f"Error: Test directory not found: {args.test_dir}")
        exit(1)
    
    # Run inference
    run_inference(
        model_checkpoint=args.model_checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        patch_size=tuple(args.patch_size),
        stride=args.stride,
        batch_size=args.batch_size
    )

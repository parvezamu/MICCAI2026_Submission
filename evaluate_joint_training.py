#!/usr/bin/env python3
"""
evaluate_joint_training_CORRECT.py

Evaluation with proper bbox-aware reconstruction:
1. Predict on 96³ volume
2. Resample to cropped size (using saved cropped_shape)
3. Uncrop to original size (using saved bbox)
4. Compare with original ground truth
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from scipy.ndimage import zoom

sys.path.append('/home/pahm409')
from models.resnet3d import resnet3d_18


# ============================================================================
# MODEL CLASSES
# ============================================================================

class ResNet3DEncoder(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.conv1 = base_encoder.conv1
        self.bn1 = base_encoder.bn1
        self.maxpool = base_encoder.maxpool
        self.layer1 = base_encoder.layer1
        self.layer2 = base_encoder.layer2
        self.layer3 = base_encoder.layer3
        self.layer4 = base_encoder.layer4

    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1, x2, x3, x4, x5]


class UNetDecoder3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _match_size(self, x_up, x_skip):
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = torch.nn.functional.interpolate(
                x_up, size=x_skip.shape[2:],
                mode='trilinear', align_corners=False
            )
        return x_up

    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        x = self.up4(x5)
        x = self._match_size(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = self._match_size(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = self._match_size(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = self._match_size(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        return self.final_conv(x)


class JointTrainingModels(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        base_encoder_dwi = resnet3d_18(in_channels=1)
        self.encoder_dwi = ResNet3DEncoder(base_encoder_dwi)
        self.decoder_dwi = UNetDecoder3D(num_classes=num_classes)
        
        base_encoder_adc = resnet3d_18(in_channels=1)
        self.encoder_adc = ResNet3DEncoder(base_encoder_adc)
        self.decoder_adc = UNetDecoder3D(num_classes=num_classes)

    def forward_dwi_only(self, dwi):
        input_size = dwi.shape[2:]
        enc_features = self.encoder_dwi(dwi)
        output = self.decoder_dwi(enc_features)
        
        if output.shape[2:] != input_size:
            output = torch.nn.functional.interpolate(
                output, size=input_size,
                mode='trilinear', align_corners=False
            )
        return output


# ============================================================================
# REVERSE PREPROCESSING WITH BBOX
# ============================================================================

def reverse_preprocessing(pred_96, metadata, method='linear'):
    """
    Reverse the preprocessing pipeline:
    1. Resample from 96³ to cropped size
    2. Uncrop to original size using bbox
    
    Args:
        pred_96: prediction in 96³ space (H, W, D)
        metadata: dict with bbox, original_shape, cropped_shape
        method: 'linear' or 'nearest'
    """
    bbox = metadata['bbox']
    original_shape = tuple(metadata['original_shape'])
    cropped_shape = tuple(metadata['cropped_shape'])
    
    # Check if bbox is valid
    bbox_valid = not (bbox[0][0] == -1)
    
    if not bbox_valid:
        # No cropping was done, just resample directly
        zoom_factors = np.array(original_shape, dtype=np.float32) / np.array([96, 96, 96], dtype=np.float32)
        order = 0 if method == 'nearest' else 1
        pred_original = zoom(pred_96.astype(np.float32), zoom_factors, order=order)
        
        if pred_original.shape != original_shape:
            fixed = np.zeros(original_shape, dtype=pred_original.dtype)
            slices = tuple(slice(0, min(pred_original.shape[i], original_shape[i])) for i in range(3))
            fixed[slices] = pred_original[slices]
            pred_original = fixed
        
        return pred_original
    
    # STEP 1: Resample from 96³ to cropped size
    zoom_factors = np.array(cropped_shape, dtype=np.float32) / np.array([96, 96, 96], dtype=np.float32)
    order = 0 if method == 'nearest' else 1
    pred_cropped = zoom(pred_96.astype(np.float32), zoom_factors, order=order)
    
    # Ensure exact cropped size
    if pred_cropped.shape != cropped_shape:
        fixed = np.zeros(cropped_shape, dtype=pred_cropped.dtype)
        slices = tuple(slice(0, min(pred_cropped.shape[i], cropped_shape[i])) for i in range(3))
        fixed[slices] = pred_cropped[slices]
        pred_cropped = fixed
    
    # STEP 2: Uncrop to original size
    pred_original = np.zeros(original_shape, dtype=pred_cropped.dtype)
    
    # Place cropped prediction back into original coordinates
    pred_original[
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]
    ] = pred_cropped
    
    return pred_original


def get_original_ground_truth(case_id, isles_raw_dir):
    """Load ground truth from original ISLES files"""
    import nibabel as nib
    
    isles_raw = Path(isles_raw_dir)
    msk_file = isles_raw / case_id / f'{case_id}_ses-0001_msk.nii.gz'
    dwi_file = isles_raw / case_id / f'{case_id}_ses-0001_dwi.nii.gz'
    
    if not msk_file.exists():
        raise FileNotFoundError(f"Mask not found: {msk_file}")
    
    msk_img = nib.load(msk_file)
    dwi_img = nib.load(dwi_file)
    
    ground_truth = (msk_img.get_fdata() > 0.5).astype(np.uint8)
    affine = dwi_img.affine
    
    return ground_truth, affine


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_test_set(model, test_cases, npz_dir, device, save_nifti=False, 
                     output_dir=None, isles_raw_dir=None):
    """Evaluate with proper bbox-aware reconstruction"""
    import nibabel as nib
    
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION - BBOX-AWARE RECONSTRUCTION")
    print(f"{'='*80}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Preprocessed dir: {npz_dir}")
    print(f"{'='*80}\n")
    
    all_dscs = []
    results_per_case = []
    
    print(f"{'Case ID':<25} {'DSC':<8} {'GT':<8} {'Pred':<8} {'Orig Shape':<18} {'Crop Shape'}")
    print("-" * 90)
    
    for case_id in tqdm(test_cases, desc='Evaluating'):
        npz_file = Path(npz_dir) / f"{case_id}.npz"
        
        if not npz_file.exists():
            print(f"{case_id:<25} ERROR: NPZ not found")
            continue
        
        # Load preprocessed data with metadata
        data = np.load(npz_file)
        dwi_96 = data['dwi']
        
        metadata = {
            'bbox': data['bbox'],
            'original_shape': data['original_shape'],
            'cropped_shape': data['cropped_shape']
        }
        
        # Predict on 96³ volume
        with torch.no_grad():
            volume_tensor = torch.from_numpy(dwi_96).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with autocast():
                output = model.forward_dwi_only(volume_tensor)
            
            pred_prob_96 = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        
        # Reverse preprocessing with bbox
        pred_prob_original = reverse_preprocessing(pred_prob_96, metadata, method='linear')
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        
        # Load ground truth
        try:
            gt_original, affine = get_original_ground_truth(case_id, isles_raw_dir)
        except FileNotFoundError as e:
            print(f"{case_id:<25} ERROR: {e}")
            continue
        
        # Compute DSC
        intersection = (pred_binary_original * gt_original).sum()
        union = pred_binary_original.sum() + gt_original.sum()
        dsc = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
        
        all_dscs.append(dsc)
        
        gt_vol = int(gt_original.sum())
        pred_vol = int(pred_binary_original.sum())
        orig_shape_str = f"{tuple(metadata['original_shape'])}"
        crop_shape_str = f"{tuple(metadata['cropped_shape'])}"
        
        print(f"{case_id:<25} {dsc:<8.4f} {gt_vol:<8} {pred_vol:<8} {orig_shape_str:<18} {crop_shape_str}")
        
        results_per_case.append({
            'case_id': case_id,
            'dsc': float(dsc),
            'original_shape': tuple(int(x) for x in metadata['original_shape']),
            'cropped_shape': tuple(int(x) for x in metadata['cropped_shape']),
            'gt_volume': gt_vol,
            'pred_volume': pred_vol
        })
        
        # Save NIfTI
        if save_nifti and output_dir:
            nifti_dir = Path(output_dir) / case_id
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            nib.save(nib.Nifti1Image(pred_binary_original, affine),
                    nifti_dir / 'prediction.nii.gz')
            nib.save(nib.Nifti1Image(pred_prob_original.astype(np.float32), affine),
                    nifti_dir / 'prediction_prob.nii.gz')
            nib.save(nib.Nifti1Image(gt_original, affine),
                    nifti_dir / 'ground_truth.nii.gz')
    
    print("-" * 90)
    
    if len(all_dscs) == 0:
        print("\n❌ No valid test cases!")
        return 0.0, [], []
    
    mean_dsc = np.mean(all_dscs)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Cases: {len(all_dscs)}/{len(test_cases)}")
    print(f"  Mean DSC: {mean_dsc:.4f} ± {np.std(all_dscs):.4f}")
    print(f"  Median: {np.median(all_dscs):.4f}")
    print(f"  Min/Max: {np.min(all_dscs):.4f} / {np.max(all_dscs):.4f}\n")
    
    return mean_dsc, all_dscs, results_per_case


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--isles-dir', type=str,
                        default='/home/pahm409/preprocessed_isles_dual_WITH_BBOX')
    parser.add_argument('--isles-raw-dir', type=str,
                        default='/home/pahm409/ISLES2022_reg/ISLES2022')
    parser.add_argument('--splits-file', type=str,
                        default='isles_dual_splits_5fold.json')
    parser.add_argument('--output-dir', type=str,
                        default='/home/pahm409/joint_training_test_results')
    parser.add_argument('--save-nifti', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fold = checkpoint.get('fold', 0)
    
    print(f"\n{'='*80}")
    print(f"JOINT TRAINING EVALUATION - BBOX-AWARE")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {fold}")
    print(f"Val DSC: {checkpoint.get('val_dsc', 'N/A')}")
    print(f"{'='*80}\n")
    
    model = JointTrainingModels(num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    test_cases = splits[f'fold_{fold}']['ISLES2022_dual']['test']
    print(f"Test cases: {len(test_cases)}\n")
    
    output_dir = Path(args.output_dir) / f'fold_{fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean_dsc, all_dscs, results_per_case = evaluate_test_set(
        model=model,
        test_cases=test_cases,
        npz_dir=args.isles_dir,
        device=device,
        save_nifti=args.save_nifti,
        output_dir=output_dir,
        isles_raw_dir=args.isles_raw_dir
    )
    
    results = {
        'fold': int(fold),
        'checkpoint': str(args.checkpoint),
        'mean_dsc': float(mean_dsc),
        'std_dsc': float(np.std(all_dscs)) if len(all_dscs) else 0.0,
        'median_dsc': float(np.median(all_dscs)) if len(all_dscs) else 0.0,
        'min_dsc': float(np.min(all_dscs)) if len(all_dscs) else 0.0,
        'max_dsc': float(np.max(all_dscs)) if len(all_dscs) else 0.0,
        'num_test_cases': int(len(all_dscs)),
        'all_dscs': [float(x) for x in all_dscs],
        'per_case_results': results_per_case
    }
    
    result_file = output_dir / 'test_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {result_file}\n")


if __name__ == '__main__':
    main()

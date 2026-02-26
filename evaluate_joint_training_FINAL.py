#!/usr/bin/env python3
"""
evaluate_joint_training_FINAL.py

PROPER evaluation with patch-based reconstruction
Using EXACT reconstruction logic from your ATLAS/ISLES code
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
# DATASET WITH CENTERS - FROM YOUR WORKING CODE
# ============================================================================

class PatchDatasetWithCenters(Dataset):
    """Extract patches WITH center coordinates for reconstruction"""
    def __init__(self, npz_dir, case_list, patch_size=(96, 96, 96), patches_per_volume=100):
        self.npz_dir = Path(npz_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        
        self.volumes = []
        for case_id in case_list:
            npz_file = self.npz_dir / f"{case_id}.npz"
            if npz_file.exists():
                data = np.load(npz_file)
                self.volumes.append({
                    'case_id': case_id,
                    'path': npz_file,
                    'volume': data['dwi'],
                    'mask': data['mask']
                })
        
        print(f"✓ Loaded {len(self.volumes)} test volumes")
        
        # Create patch indices
        self.patch_indices = []
        for vol_idx in range(len(self.volumes)):
            for _ in range(patches_per_volume):
                self.patch_indices.append(vol_idx)
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        vol_idx = self.patch_indices[idx]
        vol_info = self.volumes[vol_idx]
        
        volume = vol_info['volume']
        D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        # Random center with valid range
        half_size = np.array(self.patch_size) // 2
        
        d = np.random.randint(half_size[0], D - half_size[0] + 1) if D > pd else half_size[0]
        h = np.random.randint(half_size[1], H - half_size[1] + 1) if H > ph else half_size[1]
        w = np.random.randint(half_size[2], W - half_size[2] + 1) if W > pw else half_size[2]
        
        center = np.array([d, h, w])
        
        # Extract patch
        d_start = max(0, d - half_size[0])
        h_start = max(0, h - half_size[1])
        w_start = max(0, w - half_size[2])
        
        patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Pad if needed
        if patch.shape != tuple(self.patch_size):
            patch = np.pad(patch,
                          [(0, max(0, pd - patch.shape[0])),
                           (0, max(0, ph - patch.shape[1])),
                           (0, max(0, pw - patch.shape[2]))],
                          mode='constant')
        
        return {
            'image': torch.from_numpy(patch).unsqueeze(0).float(),
            'center': torch.from_numpy(center).long(),
            'vol_idx': vol_idx
        }
    
    def get_volume_info(self, vol_idx):
        return self.volumes[vol_idx]


# ============================================================================
# RECONSTRUCTION - EXACT COPY FROM YOUR WORKING CODE
# ============================================================================

def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """
    FIXED: Better bounds checking
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    for i, center in enumerate(centers):
        # Compute bounds
        lower = center - half_size
        upper = center + half_size
        
        # Clip to valid range
        lower_clipped = np.maximum(lower, 0)
        upper_clipped = np.minimum(upper, original_shape)
        
        # Skip if invalid
        if np.any(lower_clipped >= upper_clipped):
            continue
        
        # Compute patch slice indices
        patch_lower = lower_clipped - lower
        patch_upper = patch_lower + (upper_clipped - lower_clipped)
        
        # Extract valid portion of patch
        try:
            patch = patch_preds[i, 1, 
                               patch_lower[0]:patch_upper[0],
                               patch_lower[1]:patch_upper[1],
                               patch_lower[2]:patch_upper[2]]
            
            # Place in volume
            reconstructed[lower_clipped[0]:upper_clipped[0],
                         lower_clipped[1]:upper_clipped[1],
                         lower_clipped[2]:upper_clipped[2]] += patch
            
            count_map[lower_clipped[0]:upper_clipped[0],
                     lower_clipped[1]:upper_clipped[1],
                     lower_clipped[2]:upper_clipped[2]] += 1.0
        except Exception as e:
            print(f"Warning: Patch {i} failed: {e}")
            continue
    
    # Average overlapping predictions
    mask = count_map > 0
    reconstructed[mask] = reconstructed[mask] / count_map[mask]
    
    return reconstructed, count_map

# ============================================================================
# RESAMPLING TO ORIGINAL
# ============================================================================

def get_original_isles_info(case_id, isles_raw_dir):
    """Get original ISLES dimensions and ground truth"""
    import nibabel as nib
    
    isles_raw = Path(isles_raw_dir)
    dwi_file = isles_raw / case_id / f'{case_id}_ses-0001_dwi.nii.gz'
    msk_file = isles_raw / case_id / f'{case_id}_ses-0001_msk.nii.gz'
    
    if not dwi_file.exists():
        raise FileNotFoundError(f"DWI not found: {dwi_file}")
    if not msk_file.exists():
        raise FileNotFoundError(f"Mask not found: {msk_file}")
    
    dwi_img = nib.load(dwi_file)
    msk_img = nib.load(msk_file)
    
    original_shape = dwi_img.shape
    original_affine = dwi_img.affine
    ground_truth = (msk_img.get_fdata() > 0.5).astype(np.uint8)
    
    # Preprocessed shape is 96³
    preprocessed_shape = np.array([96, 96, 96])
    zoom_factors = np.array(original_shape, dtype=np.float32) / preprocessed_shape
    
    return {
        'original_shape': tuple(original_shape),
        'original_affine': original_affine,
        'zoom_factors': zoom_factors,
        'ground_truth': ground_truth
    }


def resample_to_original(prediction, original_info, method='nearest'):
    """Resample from 96³ to original ISLES space"""
    zoom_factors = original_info['zoom_factors']
    order = 0 if method == 'nearest' else 1
    
    resampled = zoom(prediction.astype(np.float32), zoom_factors, order=order)
    
    target_shape = original_info['original_shape']
    if resampled.shape != target_shape:
        fixed = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        fixed[slices] = resampled[slices]
        resampled = fixed
    
    return resampled


# ============================================================================
# EVALUATION
# ============================================================================





def evaluate_test_set(model, test_dataset, device, patch_size=(96, 96, 96),
                     save_nifti=False, output_dir=None, isles_raw_dir=None):
    """Evaluate with proper patch-based reconstruction"""
    import nibabel as nib
    
    model.eval()
    num_volumes = len(test_dataset.volumes)
    
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION - PATCH-BASED RECONSTRUCTION")
    print(f"{'='*80}")
    print(f"Test volumes: {num_volumes}")
    print(f"Total patches: {len(test_dataset)}")
    print(f"Patches per volume: {test_dataset.patches_per_volume}")
    print(f"{'='*80}\n")
    
    # DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Collect predictions by volume
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("Processing patches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            images = batch['image'].to(device)
            centers = batch['center'].cpu().numpy()
            vol_indices = batch['vol_idx'].cpu().numpy()
            
            with autocast():
                outputs = model.forward_dwi_only(images)
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(images)):
                vol_idx = vol_indices[i]
                volume_data[vol_idx]['centers'].append(centers[i])
                volume_data[vol_idx]['preds'].append(preds[i])
    
    # Reconstruct volumes
    all_dscs = []
    results_per_case = []
    nifti_save_queue = []  # ← Queue NIfTI saves for later
    
    print("\nReconstructing volumes...")
    print(f"{'Case ID':<25} {'DSC':<8} {'GT':<8} {'Pred':<8} {'Coverage':<10} {'Shape'}")
    print("-" * 85)
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        try:
            vol_info = test_dataset.get_volume_info(vol_idx)
            case_id = vol_info['case_id']
            
            centers = np.array(volume_data[vol_idx]['centers'])
            preds = np.array(volume_data[vol_idx]['preds'])
            
            # STEP 1: Reconstruct in 96³ space
            reconstructed_96, count_map = reconstruct_from_patches_with_count(
                preds, centers, vol_info['mask'].shape, patch_size=patch_size
            )
            
            pred_binary_96 = (reconstructed_96 > 0.5).astype(np.uint8)
            mask_gt_96 = (vol_info['mask'] > 0).astype(np.uint8)

            intersection_96 = (pred_binary_96 * mask_gt_96).sum()
            union_96 = pred_binary_96.sum() + mask_gt_96.sum()
            dsc_96 = (2.0 * intersection_96) / union_96 if union_96 > 0 else 1.0
            
            coverage = (count_map > 0).sum() / count_map.size * 100
            
            # STEP 2: Get original ISLES info
            try:
                original_info = get_original_isles_info(case_id, isles_raw_dir)
            except FileNotFoundError as e:
                print(f"{case_id:<25} ERROR: {e}")
                continue
            
            # STEP 3: Resample to original space
            pred_prob_original = resample_to_original(reconstructed_96, original_info, method='linear')
            pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
            
            mask_gt_original = original_info['ground_truth']
            
            # STEP 4: Compute DSC
            intersection = (pred_binary_original * mask_gt_original).sum()
            union = pred_binary_original.sum() + mask_gt_original.sum()
            dsc = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
            
            # ✅ FIX: print AFTER dsc is computed
            print(f"{case_id:<25} DSC_96={dsc_96:.4f}, DSC_orig={dsc:.4f}")
            
            all_dscs.append(dsc)
            
            gt_vol = int(mask_gt_original.sum())
            pred_vol = int(pred_binary_original.sum())
            
            print(f"{case_id:<25} {dsc:<8.4f} {gt_vol:<8} {pred_vol:<8} {coverage:<10.1f} {original_info['original_shape']}")
            
            results_per_case.append({
                'case_id': case_id,
                'dsc': float(dsc),
                'coverage': float(coverage),
                'original_shape': original_info['original_shape'],
                'gt_volume': gt_vol,
                'pred_volume': pred_vol
            })
            
            # Queue NIfTI save for later (OUTSIDE tqdm loop)
            if save_nifti and output_dir:
                nifti_save_queue.append({
                    'case_id': case_id,
                    'pred_binary': pred_binary_original,
                    'pred_prob': pred_prob_original,
                    'ground_truth': mask_gt_original,
                    'affine': original_info['original_affine']
                })
        
        except Exception as e:
            print(f"\n❌ ERROR at volume {vol_idx} ({test_dataset.get_volume_info(vol_idx)['case_id']}): {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping this volume and continuing...\n")
            continue
    
    print("-" * 85)
    
    # NOW save all NIfTI files OUTSIDE the loop
    if save_nifti and len(nifti_save_queue) > 0:
        print(f"\nSaving {len(nifti_save_queue)} NIfTI predictions...")
        for item in tqdm(nifti_save_queue, desc='Saving NIfTI'):
            nifti_dir = Path(output_dir) / item['case_id']
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            nib.save(nib.Nifti1Image(item['pred_binary'], item['affine']), 
                    nifti_dir / 'prediction.nii.gz')
            nib.save(nib.Nifti1Image(item['pred_prob'].astype(np.float32), item['affine']), 
                    nifti_dir / 'prediction_prob.nii.gz')
            nib.save(nib.Nifti1Image(item['ground_truth'], item['affine']), 
                    nifti_dir / 'ground_truth.nii.gz')
        print("✓ NIfTI files saved\n")
    
    if len(all_dscs) == 0:
        print("\n❌ No valid test cases!")
        return 0.0, [], []
    
    mean_dsc = np.mean(all_dscs)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Cases: {len(all_dscs)}/{num_volumes}")
    print(f"  Mean DSC: {mean_dsc:.4f} ± {np.std(all_dscs):.4f}")
    print(f"  Median: {np.median(all_dscs):.4f}")
    print(f"  Min/Max: {np.min(all_dscs):.4f} / {np.max(all_dscs):.4f}\n")
    
    return mean_dsc, all_dscs, results_per_case





def evaluate_test_set1(model, test_dataset, device, patch_size=(96, 96, 96),
                     save_nifti=False, output_dir=None, isles_raw_dir=None):
    """Evaluate with proper patch-based reconstruction"""
    import nibabel as nib
    
    model.eval()
    num_volumes = len(test_dataset.volumes)
    
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION - PATCH-BASED RECONSTRUCTION")
    print(f"{'='*80}")
    print(f"Test volumes: {num_volumes}")
    print(f"Total patches: {len(test_dataset)}")
    print(f"Patches per volume: {test_dataset.patches_per_volume}")
    print(f"{'='*80}\n")
    
    # DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Collect predictions by volume
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("Processing patches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            images = batch['image'].to(device)
            centers = batch['center'].cpu().numpy()
            vol_indices = batch['vol_idx'].cpu().numpy()
            
            with autocast():
                outputs = model.forward_dwi_only(images)
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(images)):
                vol_idx = vol_indices[i]
                volume_data[vol_idx]['centers'].append(centers[i])
                volume_data[vol_idx]['preds'].append(preds[i])
    
    # Reconstruct volumes
    all_dscs = []
    results_per_case = []
    nifti_save_queue = []  # ← Queue NIfTI saves for later
    
    print("\nReconstructing volumes...")
    print(f"{'Case ID':<25} {'DSC':<8} {'GT':<8} {'Pred':<8} {'Coverage':<10} {'Shape'}")
    print("-" * 85)
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        try:
            vol_info = test_dataset.get_volume_info(vol_idx)
            case_id = vol_info['case_id']
            
            centers = np.array(volume_data[vol_idx]['centers'])
            preds = np.array(volume_data[vol_idx]['preds'])
            
            # STEP 1: Reconstruct in 96³ space
            reconstructed_96, count_map = reconstruct_from_patches_with_count(
                preds, centers, vol_info['mask'].shape, patch_size=patch_size
            )
            
            
            pred_binary_96 = (reconstructed_96 > 0.5).astype(np.uint8)
            mask_gt_96 = (vol_info['mask'] > 0).astype(np.uint8)

            intersection_96 = (pred_binary_96 * mask_gt_96).sum()
            union_96 = pred_binary_96.sum() + mask_gt_96.sum()
            dsc_96 = (2.0 * intersection_96) / union_96 if union_96 > 0 else 1.0

            print(f"{case_id:<25} DSC_96={dsc_96:.4f}, DSC_orig={dsc:.4f}")
            
            coverage = (count_map > 0).sum() / count_map.size * 100
            
            # STEP 2: Get original ISLES info
            try:
                original_info = get_original_isles_info(case_id, isles_raw_dir)
            except FileNotFoundError as e:
                print(f"{case_id:<25} ERROR: {e}")
                continue
            
            # STEP 3: Resample to original space
            pred_prob_original = resample_to_original(reconstructed_96, original_info, method='linear')
            pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
            
            mask_gt_original = original_info['ground_truth']
            
            # STEP 4: Compute DSC
            intersection = (pred_binary_original * mask_gt_original).sum()
            union = pred_binary_original.sum() + mask_gt_original.sum()
            dsc = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary_original.sum() == 0 else 0.0)
            
            all_dscs.append(dsc)
            
            gt_vol = int(mask_gt_original.sum())
            pred_vol = int(pred_binary_original.sum())
            
            print(f"{case_id:<25} {dsc:<8.4f} {gt_vol:<8} {pred_vol:<8} {coverage:<10.1f} {original_info['original_shape']}")
            
            results_per_case.append({
                'case_id': case_id,
                'dsc': float(dsc),
                'coverage': float(coverage),
                'original_shape': original_info['original_shape'],
                'gt_volume': gt_vol,
                'pred_volume': pred_vol
            })
            
            # Queue NIfTI save for later (OUTSIDE tqdm loop)
            if save_nifti and output_dir:
                nifti_save_queue.append({
                    'case_id': case_id,
                    'pred_binary': pred_binary_original,
                    'pred_prob': pred_prob_original,
                    'ground_truth': mask_gt_original,
                    'affine': original_info['original_affine']
                })
        
        except Exception as e:
            print(f"\n❌ ERROR at volume {vol_idx} ({test_dataset.get_volume_info(vol_idx)['case_id']}): {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping this volume and continuing...\n")
            continue
    
    print("-" * 85)
    
    # NOW save all NIfTI files OUTSIDE the loop
    if save_nifti and len(nifti_save_queue) > 0:
        print(f"\nSaving {len(nifti_save_queue)} NIfTI predictions...")
        for item in tqdm(nifti_save_queue, desc='Saving NIfTI'):
            nifti_dir = Path(output_dir) / item['case_id']
            nifti_dir.mkdir(parents=True, exist_ok=True)
            
            nib.save(nib.Nifti1Image(item['pred_binary'], item['affine']), 
                    nifti_dir / 'prediction.nii.gz')
            nib.save(nib.Nifti1Image(item['pred_prob'].astype(np.float32), item['affine']), 
                    nifti_dir / 'prediction_prob.nii.gz')
            nib.save(nib.Nifti1Image(item['ground_truth'], item['affine']), 
                    nifti_dir / 'ground_truth.nii.gz')
        print("✓ NIfTI files saved\n")
    
    if len(all_dscs) == 0:
        print("\n❌ No valid test cases!")
        return 0.0, [], []
    
    mean_dsc = np.mean(all_dscs)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Cases: {len(all_dscs)}/{num_volumes}")
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
                        default='/home/pahm409/preprocessed_isles_dual_v2')
    parser.add_argument('--isles-raw-dir', type=str,
                        default='/home/pahm409/ISLES2022_reg/ISLES2022')
    parser.add_argument('--splits-file', type=str,
                        default='isles_dual_splits_5fold.json')
    parser.add_argument('--output-dir', type=str,
                        default='/home/pahm409/joint_training_test_results')
    parser.add_argument('--save-nifti', action='store_true')
    parser.add_argument('--patches-per-volume', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fold = checkpoint.get('fold', 0)
    
    print(f"\n{'='*80}")
    print(f"JOINT TRAINING EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {fold}")
    print(f"Val DSC: {checkpoint.get('val_dsc', 'N/A')}")
    print(f"Patches per volume: {args.patches_per_volume}")
    print(f"{'='*80}\n")
    
    model = JointTrainingModels(num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    test_cases = splits[f'fold_{fold}']['ISLES2022_dual']['test']
    print(f"Test cases: {len(test_cases)}\n")
    
    test_dataset = PatchDatasetWithCenters(
        npz_dir=args.isles_dir,
        case_list=test_cases,
        patch_size=(96, 96, 96),
        patches_per_volume=args.patches_per_volume
    )
    
    output_dir = Path(args.output_dir) / f'fold_{fold}_FINAL'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean_dsc, all_dscs, results_per_case = evaluate_test_set(
        model=model,
        test_dataset=test_dataset,
        device=device,
        patch_size=(96, 96, 96),
        save_nifti=args.save_nifti,
        output_dir=output_dir,
        isles_raw_dir=args.isles_raw_dir
    )
    
    
    
    
    
    results = {
        'fold': int(fold),
        'checkpoint': str(args.checkpoint),
        'mean_dsc': float(mean_dsc),
        'std_dsc': float(np.std(all_dscs)),
        'median_dsc': float(np.median(all_dscs)),
        'min_dsc': float(np.min(all_dscs)),
        'max_dsc': float(np.max(all_dscs)),
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

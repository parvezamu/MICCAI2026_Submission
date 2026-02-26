#!/usr/bin/env python3
"""
evaluate_joint_from_scratch.py

Evaluate joint training model (no transfer learning)
- Two separate models (DWI and ADC)
- Uses DWI-only for inference
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

sys.path.append('/hpc/pahm409')
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
# SLIDING WINDOW
# ============================================================================

def get_sliding_window_patches(volume_shape, patch_size=(96, 96, 96), step_size=0.5):
    D, H, W = volume_shape
    pd, ph, pw = patch_size
    
    step_d = max(1, int(pd * step_size))
    step_h = max(1, int(ph * step_size))
    step_w = max(1, int(pw * step_size))
    
    patches = []
    
    d_starts = list(range(0, D - pd + 1, step_d))
    h_starts = list(range(0, H - ph + 1, step_h))
    w_starts = list(range(0, W - pw + 1, step_w))
    
    if d_starts[-1] + pd < D:
        d_starts.append(D - pd)
    if h_starts[-1] + ph < H:
        h_starts.append(H - ph)
    if w_starts[-1] + pw < W:
        w_starts.append(W - pw)
    
    for d_start in d_starts:
        for h_start in h_starts:
            for w_start in w_starts:
                patches.append((d_start, h_start, w_start))
    
    print(f"  Volume shape: {volume_shape}, Patches: {len(patches)}")
    return patches


def sliding_window_inference(model, volume, device, patch_size=(96, 96, 96), step_size=0.5, batch_size=4):
    model.eval()
    
    volume_shape = volume.shape
    prediction = np.zeros(volume_shape, dtype=np.float32)
    count_map = np.zeros(volume_shape, dtype=np.float32)
    
    patch_coords = get_sliding_window_patches(volume_shape, patch_size, step_size)
    
    print(f"  Processing {len(patch_coords)} patches...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patch_coords), batch_size), desc='Sliding window'):
            batch_coords = patch_coords[i:i+batch_size]
            
            patches = []
            for (d_start, h_start, w_start) in batch_coords:
                patch = volume[d_start:d_start+patch_size[0],
                              h_start:h_start+patch_size[1],
                              w_start:w_start+patch_size[2]]
                patches.append(patch)
            
            patches_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(device)
            
            with autocast():
                outputs = model.forward_dwi_only(patches_tensor)
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for j, (d_start, h_start, w_start) in enumerate(batch_coords):
                pred_patch = preds[j, 1]
                
                prediction[d_start:d_start+patch_size[0],
                          h_start:h_start+patch_size[1],
                          w_start:w_start+patch_size[2]] += pred_patch
                
                count_map[d_start:d_start+patch_size[0],
                         h_start:h_start+patch_size[1],
                         w_start:w_start+patch_size[2]] += 1.0
    
    mask = count_map > 0
    prediction[mask] = prediction[mask] / count_map[mask]
    
    return prediction, count_map


def reverse_preprocessing_fixed(pred_96, metadata):
    bbox = metadata['bbox']
    original_shape = tuple(metadata['original_shape'])
    cropped_shape = tuple(metadata['cropped_shape'])
    
    bbox_valid = not (bbox[0][0] == -1)
    
    if not bbox_valid:
        zoom_factors = np.array(original_shape, dtype=np.float32) / np.array([96, 96, 96], dtype=np.float32)
        pred_original = zoom(pred_96.astype(np.float32), zoom_factors, order=0)
        
        if pred_original.shape != original_shape:
            fixed = np.zeros(original_shape, dtype=pred_original.dtype)
            slices = tuple(slice(0, min(pred_original.shape[i], original_shape[i])) for i in range(3))
            fixed[slices] = pred_original[slices]
            pred_original = fixed
        
        return pred_original
    
    pred_96_binary = (pred_96 > 0.5).astype(np.float32)
    
    zoom_factors = np.array(cropped_shape, dtype=np.float32) / np.array([96, 96, 96], dtype=np.float32)
    pred_cropped = zoom(pred_96_binary, zoom_factors, order=0)
    
    if pred_cropped.shape != cropped_shape:
        fixed = np.zeros(cropped_shape, dtype=pred_cropped.dtype)
        slices = tuple(slice(0, min(pred_cropped.shape[i], cropped_shape[i])) for i in range(3))
        fixed[slices] = pred_cropped[slices]
        pred_cropped = fixed
    
    pred_original = np.zeros(original_shape, dtype=pred_cropped.dtype)
    pred_original[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = pred_cropped
    
    return pred_original


def get_original_ground_truth(case_id, isles_raw_dir):
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


def evaluate_test_set(model, test_cases, npz_dir, device, 
                      patch_size=(96, 96, 96), step_size=0.5,
                      save_nifti=False, output_dir=None, isles_raw_dir=None):
    import nibabel as nib
    
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"JOINT TRAINING EVALUATION (FROM SCRATCH)")
    print(f"{'='*80}")
    print(f"Test cases: {len(test_cases)}")
    print(f"DWI-only inference\n")
    
    all_dscs = []
    results_per_case = []
    
    for case_id in tqdm(test_cases, desc='Evaluating'):
        try:
            npz_file = Path(npz_dir) / f"{case_id}.npz"
            if not npz_file.exists():
                continue
            
            data = np.load(npz_file)
            volume_96 = data['dwi']
            
            # Sliding window
            pred_96, count_map = sliding_window_inference(
                model=model,
                volume=volume_96,
                device=device,
                patch_size=patch_size,
                step_size=step_size,
                batch_size=4
            )
            
            metadata = {
                'bbox': data['bbox'],
                'original_shape': data['original_shape'],
                'cropped_shape': data['cropped_shape']
            }
            
            # Reverse preprocessing
            pred_binary_original = reverse_preprocessing_fixed(pred_96, metadata).astype(np.uint8)
            
            # Load ground truth
            gt_original, affine = get_original_ground_truth(case_id, isles_raw_dir)
            
            # Compute DSC
            intersection = (pred_binary_original * gt_original).sum()
            union = pred_binary_original.sum() + gt_original.sum()
            dsc = (2.0 * intersection) / union if union > 0 else 1.0
            
            all_dscs.append(dsc)
            
            results_per_case.append({
                'case_id': case_id,
                'dsc': float(dsc)
            })
            
            if save_nifti and output_dir:
                nifti_dir = Path(output_dir) / case_id
                nifti_dir.mkdir(parents=True, exist_ok=True)
                
                nib.save(nib.Nifti1Image(pred_binary_original, affine), 
                        nifti_dir / 'prediction.nii.gz')
                nib.save(nib.Nifti1Image(gt_original, affine), 
                        nifti_dir / 'ground_truth.nii.gz')
        
        except Exception as e:
            print(f"\n❌ ERROR for {case_id}: {e}")
            continue
    
    mean_dsc = np.mean(all_dscs)
    
    print(f"\n📊 RESULTS:")
    print(f"  Mean DSC: {mean_dsc:.4f} ± {np.std(all_dscs):.4f}\n")
    
    return mean_dsc, all_dscs, results_per_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--isles-dir', type=str,
                        default='/hpc/pahm409/preprocessed_isles_dual_WITH_BBOX')
    parser.add_argument('--isles-raw-dir', type=str,
                        default='/hpc/pahm409/ISLES2022')
    parser.add_argument('--splits-file', type=str,
                        default='isles_dual_splits_5fold.json')
    parser.add_argument('--output-dir', type=str,
                        default='/hpc/pahm409/joint_scratch_test_results')
    parser.add_argument('--save-nifti', action='store_true')
    parser.add_argument('--step-size', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fold = checkpoint.get('fold', 0)
    
    print(f"\n{'='*80}")
    print(f"JOINT TRAINING FROM SCRATCH - EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {fold}")
    
    model = JointTrainingModels(num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    test_cases = splits[f'fold_{fold}']['ISLES2022_dual']['test']
    
    output_dir = Path(args.output_dir) / f'fold_{fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean_dsc, all_dscs, results_per_case = evaluate_test_set(
        model=model,
        test_cases=test_cases,
        npz_dir=args.isles_dir,
        device=device,
        step_size=args.step_size,
        save_nifti=args.save_nifti,
        output_dir=output_dir,
        isles_raw_dir=args.isles_raw_dir
    )
    
    results = {
        'fold': int(fold),
        'mean_dsc': float(mean_dsc),
        'all_dscs': [float(x) for x in all_dscs],
        'per_case_results': results_per_case
    }
    
    result_file = output_dir / 'test_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {result_file}\n")


if __name__ == '__main__':
    main()

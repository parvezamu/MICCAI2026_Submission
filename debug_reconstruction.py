"""
debug_reconstruction.py

Debug patch reconstruction to find the issue

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import sys
import torch.nn.functional as F
import nibabel as nib

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


class ResNet3DEncoder(nn.Module):
    def __init__(self, base_encoder):
        super(ResNet3DEncoder, self).__init__()
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
        super(UNetDecoder3D, self).__init__()
        
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
        d_up, h_up, w_up = x_up.shape[2:]
        d_skip, h_skip, w_skip = x_skip.shape[2:]
        
        if d_up != d_skip or h_up != h_skip or w_up != w_skip:
            diff_d = d_skip - d_up
            diff_h = h_skip - h_up
            diff_w = w_skip - w_up
            
            if diff_d > 0 or diff_h > 0 or diff_w > 0:
                padding = [
                    max(0, diff_w // 2), max(0, diff_w - diff_w // 2),
                    max(0, diff_h // 2), max(0, diff_h - diff_h // 2),
                    max(0, diff_d // 2), max(0, diff_d - diff_d // 2)
                ]
                x_up = F.pad(x_up, padding)
            elif diff_d < 0 or diff_h < 0 or diff_w < 0:
                d_start = max(0, -diff_d // 2)
                h_start = max(0, -diff_h // 2)
                w_start = max(0, -diff_w // 2)
                x_up = x_up[:, :, d_start:d_start + d_skip, h_start:h_start + h_skip, w_start:w_start + w_skip]
        
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
        
        x = self.final_conv(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='trilinear', align_corners=False)
        
        return seg_logits


def extract_patches_from_volume(volume, patch_size=(96, 96, 96), overlap=0.5):
    """Extract patches with sliding window -DEBBUGGED"""
    vol_shape = np.array(volume.shape)
    patch_size = np.array(patch_size)
    stride = (patch_size * (1 - overlap)).astype(int)
    
    patches = []
    centers = []
    half_size = patch_size // 2
    
    n_d = max(1, int(np.ceil((vol_shape[0] - patch_size[0]) / stride[0])) + 1)
    n_h = max(1, int(np.ceil((vol_shape[1] - patch_size[1]) / stride[1])) + 1)
    n_w = max(1, int(np.ceil((vol_shape[2] - patch_size[2]) / stride[2])) + 1)
    
    print(f"  Volume shape: {vol_shape}")
    print(f"  Patch size: {patch_size}")
    print(f"  Stride: {stride}")
    print(f"  Grid: {n_d} × {n_h} × {n_w} = {n_d * n_h * n_w} patches")
    
    for i in range(n_d):
        for j in range(n_h):
            for k in range(n_w):
                center_d = min(i * stride[0] + half_size[0], vol_shape[0] - half_size[0])
                center_h = min(j * stride[1] + half_size[1], vol_shape[1] - half_size[1])
                center_w = min(k * stride[2] + half_size[2], vol_shape[2] - half_size[2])
                
                center = np.array([center_d, center_h, center_w])
                lower = center - half_size
                upper = center + half_size
                
                lower = np.maximum(lower, 0)
                upper = np.minimum(upper, vol_shape)
                
                patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
                
                if patch.shape != tuple(patch_size):
                    pad_d = patch_size[0] - patch.shape[0]
                    pad_h = patch_size[1] - patch.shape[1]
                    pad_w = patch_size[2] - patch.shape[2]
                    
                    patch = np.pad(
                        patch,
                        ((0, pad_d), (0, pad_h), (0, pad_w)),
                        mode='constant',
                        constant_values=volume.min()
                    )
                
                patches.append(patch)
                centers.append(center)
    
    print(f"  Extracted {len(patches)} patches")
    return np.array(patches), np.array(centers)


def reconstruct_from_patches_debug(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """Reconstruct with detailed debugging"""
    print(f"\n  RECONSTRUCTION DEBUG:")
    print(f"  Target shape: {original_shape}")
    print(f"  Patch predictions shape: {patch_preds.shape}")
    print(f"  Number of centers: {len(centers)}")
    print(f"  Patch size: {patch_size}")
    
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    valid_patches = 0
    invalid_patches = 0
    
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        # Bounds check
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            invalid_patches += 1
            if invalid_patches <= 3:  # Show first 3 invalid
                print(f"    Patch {i}: INVALID - center={center}, lower={lower}, upper={upper}")
            continue
        
        # Extract patch prediction
        patch = patch_preds[i, 1, ...]  # Get class 1 (lesion)
        
        # Debug first patch
        if i == 0:
            print(f"    Patch 0: center={center}, lower={lower}, upper={upper}")
            print(f"    Patch 0: pred shape={patch.shape}, pred range=[{patch.min():.4f}, {patch.max():.4f}]")
            print(f"    Patch 0: pred sum={patch.sum():.2f}, pred>0.5: {(patch>0.5).sum()}")
        
        # Accumulate
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
        
        valid_patches += 1
    
    print(f"  Valid patches: {valid_patches}, Invalid patches: {invalid_patches}")
    print(f"  Count map range: [{count_map.min():.1f}, {count_map.max():.1f}]")
    print(f"  Reconstructed (before avg) range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"  Reconstructed (before avg) sum: {reconstructed.sum():.2f}")
    
    # Average
    reconstructed = reconstructed / (count_map + 1e-6)
    
    print(f"  Reconstructed (after avg) range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"  Reconstructed (after avg) sum: {reconstructed.sum():.2f}")
    print(f"  Reconstructed >0.5: {(reconstructed > 0.5).sum()}")
    
    return reconstructed


def preprocess_volume(volume):
    """Apply same preprocessing as training"""
    brain_mask = volume > np.percentile(volume, 1)
    brain_voxels = volume[brain_mask]
    
    if len(brain_voxels) > 0:
        percentile_val = np.percentile(brain_voxels, 99.5)
        volume = np.clip(volume, 0, percentile_val)
    
    output = np.zeros_like(volume)
    brain_voxels = volume[brain_mask]
    if len(brain_voxels) > 0:
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        if std > 0:
            output[brain_mask] = (volume[brain_mask] - mean) / std
    
    return output


def compute_dsc(pred, target):
    """Compute DSC"""
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    
    return (2.0 * intersection) / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold-checkpoint', type=str, required=True)
    parser.add_argument('--case-id', type=str, default='BBS001',
                       help='Single case to debug')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print(f"DEBUGGING RECONSTRUCTION FOR CASE: {args.case_id}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.fold_checkpoint, map_location=device)
    
    encoder = resnet3d_18(in_channels=1)
    simclr_checkpoint = torch.load(
        '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth',
        map_location=device
    )
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(simclr_checkpoint['model_state_dict'])
    
    model = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded\n")
    
    # Load case from preprocessed
    npz_file = Path(f'/home/pahm409/preprocessed_neuralcup/NEURALCUP/{args.case_id}.npz')
    
    if not npz_file.exists():
        print(f"Error: {npz_file} not found!")
        return
    
    data = np.load(npz_file)
    volume = data['image']
    mask = data['lesion_mask']
    
    print(f"Case: {args.case_id}")
    print(f"Volume shape: {volume.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"GT lesion voxels: {mask.sum()}\n")
    
    # Extract patches
    print("="*70)
    print("EXTRACTING PATCHES")
    print("="*70)
    
    patches, centers = extract_patches_from_volume(volume, patch_size=(96, 96, 96), overlap=0.5)
    
    # Predict
    print("\n" + "="*70)
    print("PREDICTING")
    print("="*70)
    
    all_preds = []
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            batch_patches = torch.from_numpy(batch_patches).float().unsqueeze(1).to(device)
            
            outputs = model(batch_patches)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_preds.append(preds)
    
    all_preds = np.concatenate(all_preds, axis=0)
    
    print(f"  Prediction shape: {all_preds.shape}")
    print(f"  Prediction range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    
    # Reconstruct
    print("\n" + "="*70)
    print("RECONSTRUCTING")
    print("="*70)
    
    reconstructed = reconstruct_from_patches_debug(
        all_preds, centers, volume.shape, patch_size=(96, 96, 96)
    )
    
    pred_binary = (reconstructed > 0.5).astype(np.uint8)
    
    # Compute DSC
    dsc = compute_dsc(pred_binary, mask)
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"DSC: {dsc:.4f}")
    print(f"GT voxels: {mask.sum()}")
    print(f"Pred voxels: {pred_binary.sum()}")
    print(f"Intersection: {(pred_binary * mask).sum()}")
    print("="*70 + "\n")
    
    # Save for inspection
    output_dir = Path('/home/pahm409/debug_reconstruction')
    output_dir.mkdir(exist_ok=True)
    
    nib.save(nib.Nifti1Image(pred_binary, np.eye(4)), output_dir / f'{args.case_id}_pred.nii.gz')
    nib.save(nib.Nifti1Image(mask, np.eye(4)), output_dir / f'{args.case_id}_gt.nii.gz')
    nib.save(nib.Nifti1Image(reconstructed, np.eye(4)), output_dir / f'{args.case_id}_prob.nii.gz')
    
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()

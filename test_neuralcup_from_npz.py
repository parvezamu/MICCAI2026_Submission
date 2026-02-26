"""
test_neuralcup_from_npz.py

Test on NEURALCUP using preprocessed NPZ files
Uses EXACT same validation logic as training

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
from tqdm import tqdm
import json
import sys
from collections import defaultdict

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
                x_up = torch.nn.functional.pad(x_up, padding)
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
            seg_logits = torch.nn.functional.interpolate(
                seg_logits, size=input_size, mode='trilinear', align_corners=False
            )
        
        return seg_logits


def reconstruct_from_patches(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """Reconstruct - EXACT COPY from training"""
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            continue
        
        patch = patch_preds[i, 1, ...]
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    reconstructed = reconstructed / (count_map + 1e-6)
    return reconstructed


def extract_patches_random(volume, patch_size=(96, 96, 96), num_patches=10):
    """
    Extract random patches - SAME as training dataset
    """
    vol_shape = np.array(volume.shape)
    half_size = np.array(patch_size) // 2
    
    patches = []
    centers = []
    
    min_center = half_size
    max_center = vol_shape - half_size
    
    # Ensure valid range
    for dim in range(3):
        if min_center[dim] >= max_center[dim]:
            min_center[dim] = vol_shape[dim] // 2
            max_center[dim] = vol_shape[dim] // 2
    
    for _ in range(num_patches):
        center = np.array([
            np.random.randint(min_center[0], max_center[0] + 1),
            np.random.randint(min_center[1], max_center[1] + 1),
            np.random.randint(min_center[2], max_center[2] + 1)
        ])
        
        lower = center - half_size
        upper = center + half_size
        
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
    
    return np.array(patches), np.array(centers)


def compute_dsc(pred, target):
    """Compute DSC"""
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    
    return (2.0 * intersection) / union


def validate_neuralcup_like_training(model, neuralcup_dir, device, patch_size, patches_per_volume=10):
    """
    Validate on NEURALCUP using EXACT same logic as training validation
    """
    model.eval()
    
    # Get all NPZ files
    neuralcup_dir = Path(neuralcup_dir)
    npz_files = sorted(list(neuralcup_dir.glob('*.npz')))
    
    print(f"Found {len(npz_files)} preprocessed cases\n")
    
    # Store patches grouped by volume (like training)
    volume_data = defaultdict(lambda: {'patches': [], 'centers': [], 'preds': []})
    
    print("Processing patches...")
    with torch.no_grad():
        for vol_idx, npz_file in enumerate(tqdm(npz_files, desc='Extracting & predicting')):
            # Load preprocessed data
            data = np.load(npz_file)
            volume = data['image']  # Already normalized!
            mask = data['lesion_mask']
            
            # Extract random patches (like training)
            patches, centers = extract_patches_random(
                volume, 
                patch_size=patch_size, 
                num_patches=patches_per_volume
            )
            
            # Predict on patches
            for i in range(len(patches)):
                patch_tensor = torch.from_numpy(patches[i]).float().unsqueeze(0).unsqueeze(0).to(device)
                
                output = model(patch_tensor)
                pred = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                volume_data[vol_idx]['centers'].append(centers[i])
                volume_data[vol_idx]['preds'].append(pred)
            
            volume_data[vol_idx]['volume'] = volume
            volume_data[vol_idx]['mask'] = mask
            volume_data[vol_idx]['case_id'] = npz_file.stem
    
    # Reconstruct each volume (like training)
    print("\nReconstructing volumes...")
    results = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = volume_data[vol_idx]
        
        centers = np.array(vol_info['centers'])
        preds = np.array(vol_info['preds'])
        mask_gt = vol_info['mask']
        case_id = vol_info['case_id']
        
        # Reconstruct
        reconstructed = reconstruct_from_patches(
            preds, centers, mask_gt.shape, patch_size=patch_size
        )
        
        reconstructed_binary = (reconstructed > 0.5).astype(np.float32)
        
        # Compute DSC
        dsc = compute_dsc(reconstructed_binary, mask_gt)
        
        results.append({
            'case_id': case_id,
            'dsc': float(dsc),
            'lesion_volume_gt': int(mask_gt.sum()),
            'lesion_volume_pred': int(reconstructed_binary.sum())
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold-checkpoint', type=str, required=True)
    parser.add_argument('--neuralcup-dir', type=str,
                       default='/home/pahm409/preprocessed_NEURALCUP/NEURALCUP')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/neuralcup_results')
    parser.add_argument('--simclr-checkpoint', type=str,
                       default='/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10,
                       help='Number of random patches per volume (same as training)')
    parser.add_argument('--fold-name', type=str, default='fold_1')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print(f"TESTING {args.fold_name.upper()} ON NEURALCUP")
    print("="*70)
    print("Using EXACT training validation logic")
    print("Loading from preprocessed NPZ files")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.fold_checkpoint, map_location=device)
    
    encoder = resnet3d_18(in_channels=1)
    simclr_checkpoint = torch.load(args.simclr_checkpoint, map_location=device)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(simclr_checkpoint['model_state_dict'])
    
    model = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', -1) + 1
    val_dsc = checkpoint.get('val_dsc_recon', 0.0)
    
    print(f"✓ Loaded from epoch {epoch}, Val DSC: {val_dsc:.4f}\n")
    
    # Validate
    results = validate_neuralcup_like_training(
        model, 
        args.neuralcup_dir, 
        device, 
        tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume
    )
    
    # Summary
    print("\n" + "="*70)
    
    if len(results) == 0:
        print("No cases successfully processed!")
        return
    
    dscs = [r['dsc'] for r in results]
    
    summary = {
        'fold_name': args.fold_name,
        'checkpoint_epoch': epoch,
        'validation_dsc': float(val_dsc),
        'neuralcup_results': {
            'n_cases': len(results),
            'mean_dsc': float(np.mean(dscs)),
            'std_dsc': float(np.std(dscs)),
            'median_dsc': float(np.median(dscs)),
            'min_dsc': float(np.min(dscs)),
            'max_dsc': float(np.max(dscs))
        },
        'cases': results
    }
    
    # Save
    output_dir = Path(args.output_dir) / args.fold_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'neuralcup_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"{args.fold_name.upper()} - NEURALCUP TEST RESULTS")
    print("="*70)
    print(f"Cases tested: {len(results)}")
    print(f"Mean DSC: {summary['neuralcup_results']['mean_dsc']:.4f} ± {summary['neuralcup_results']['std_dsc']:.4f}")
    print(f"Median DSC: {summary['neuralcup_results']['median_dsc']:.4f}")
    print(f"Range: [{summary['neuralcup_results']['min_dsc']:.4f}, {summary['neuralcup_results']['max_dsc']:.4f}]")
    print(f"\nInternal validation DSC: {val_dsc:.4f}")
    print(f"External test DSC: {summary['neuralcup_results']['mean_dsc']:.4f}")
    print(f"Generalization gap: {val_dsc - summary['neuralcup_results']['mean_dsc']:.4f}")
    print(f"\nResults saved to: {output_dir / 'neuralcup_results.json'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

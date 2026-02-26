"""
test_neuralcup_with_dataset.py

Test on NEURALCUP using EXACT same dataset logic as training
Uses PatchDatasetWithCenters for identical patch sampling

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
import torch.nn.functional as F

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


class NEURALCUPDataset:
    """
    Mimics PatchDatasetWithCenters behavior for NEURALCUP
    """
    def __init__(self, preprocessed_dir, patch_size=(96, 96, 96), patches_per_volume=10, augment=False):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        
        # Load all NPZ files
        self.volumes = []
        npz_files = sorted(list(self.preprocessed_dir.glob('*.npz')))
        
        print(f"Loading {len(npz_files)} volumes from {self.preprocessed_dir}")
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            
            volume_info = {
                'case_id': npz_file.stem,
                'npz_path': str(npz_file),
                'image': data['image'],
                'mask': data['lesion_mask'],
                'brain_mask': data['brain_mask']
            }
            
            # Pre-generate patch centers (SAME as training!)
            volume_info['patch_centers'] = self._generate_patch_centers(
                volume_info['image'], 
                volume_info['mask'],
                volume_info['brain_mask']
            )
            
            self.volumes.append(volume_info)
        
        print(f"✓ Loaded {len(self.volumes)} volumes")
        print(f"  Total patches: {len(self.volumes) * self.patches_per_volume}")
    
    def _generate_patch_centers(self, volume, mask, brain_mask):
        """
        Generate patch centers - SAME logic as PatchDatasetWithCenters
        """
        vol_shape = np.array(volume.shape)
        half_size = np.array(self.patch_size) // 2
        
        centers = []
        
        # Get valid center range
        min_center = half_size
        max_center = vol_shape - half_size
        
        for dim in range(3):
            if min_center[dim] >= max_center[dim]:
                min_center[dim] = vol_shape[dim] // 2
                max_center[dim] = vol_shape[dim] // 2
        
        # Generate random centers
        np.random.seed(42)  # Fixed seed for reproducibility
        
        for _ in range(self.patches_per_volume):
            center = np.array([
                np.random.randint(min_center[0], max_center[0] + 1),
                np.random.randint(min_center[1], max_center[1] + 1),
                np.random.randint(min_center[2], max_center[2] + 1)
            ])
            centers.append(center)
        
        return np.array(centers)
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def __getitem__(self, idx):
        """
        Get patch - SAME logic as PatchDatasetWithCenters
        """
        vol_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        
        vol_info = self.volumes[vol_idx]
        volume = vol_info['image']
        mask = vol_info['mask']
        center = vol_info['patch_centers'][patch_idx]
        
        # Extract patch
        half_size = np.array(self.patch_size) // 2
        lower = center - half_size
        upper = center + half_size
        
        patch_image = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        patch_mask = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        
        # Pad if necessary
        if patch_image.shape != tuple(self.patch_size):
            pad_d = self.patch_size[0] - patch_image.shape[0]
            pad_h = self.patch_size[1] - patch_image.shape[1]
            pad_w = self.patch_size[2] - patch_image.shape[2]
            
            patch_image = np.pad(
                patch_image,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=volume.min()
            )
            
            patch_mask = np.pad(
                patch_mask,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
        
        return {
            'image': torch.from_numpy(patch_image).float().unsqueeze(0),
            'mask': torch.from_numpy(patch_mask).long(),
            'center': torch.from_numpy(center),
            'vol_idx': vol_idx
        }
    
    def get_volume_info(self, vol_idx):
        """Get full volume info"""
        vol_info = self.volumes[vol_idx]
        return {
            'case_id': vol_info['case_id'],
            'volume': vol_info['image'],
            'mask': vol_info['mask']
        }


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


def validate_with_reconstruction(model, dataset, device, patch_size):
    """
    Validate with reconstruction - EXACT COPY from training
    """
    model.eval()
    
    # Group patches by volume
    volume_data = defaultdict(lambda: {'patches': [], 'centers': [], 'preds': []})
    
    print("\nProcessing patches...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Processing patches'):
            batch = dataset[idx]
            
            vol_idx = batch['vol_idx']
            if isinstance(vol_idx, torch.Tensor):
                vol_idx = vol_idx.item()
            
            image = batch['image'].unsqueeze(0).to(device)
            center = batch['center'].numpy()
            
            # Predict
            output = model(image)
            pred = torch.softmax(output, dim=1).cpu().numpy()
            
            volume_data[vol_idx]['patches'].append(image.cpu())
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
    
    # Reconstruct each volume
    print("\nReconstructing volumes...")
    results = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = dataset.get_volume_info(vol_idx)
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        # Reconstruct
        reconstructed = reconstruct_from_patches(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        # Binarize
        reconstructed_binary = (reconstructed > 0.5).astype(np.float32)
        mask_gt = vol_info['mask'].astype(np.float32)
        
        # Compute DSC
        intersection = (reconstructed_binary * mask_gt).sum()
        union = reconstructed_binary.sum() + mask_gt.sum()
        
        if union == 0:
            dsc = 1.0 if reconstructed_binary.sum() == 0 else 0.0
        else:
            dsc = (2. * intersection) / union
        
        results.append({
            'case_id': vol_info['case_id'],
            'dsc': float(dsc),
            'lesion_volume_gt': int(mask_gt.sum()),
            'lesion_volume_pred': int(reconstructed_binary.sum())
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold-checkpoint', type=str, required=True,
                       help='Path to fold checkpoint (best_model.pth)')
    parser.add_argument('--neuralcup-dir', type=str,
                       default='/home/pahm409/preprocessed_NEURALCUP/NEURALCUP',
                       help='Path to preprocessed NEURALCUP directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/neuralcup_results',
                       help='Output directory for results')
    parser.add_argument('--simclr-checkpoint', type=str,
                       default='/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth',
                       help='Path to SimCLR pretrained checkpoint')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10,
                       help='Number of patches per volume (same as training)')
    parser.add_argument('--fold-name', type=str, default='fold_1')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print(f"TESTING {args.fold_name.upper()} ON NEURALCUP")
    print("="*70)
    print("Using EXACT same dataset logic as training")
    print("Fixed random seed for reproducible patch sampling")
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
    
    print(f"✓ Loaded from epoch {epoch}, Internal Val DSC: {val_dsc:.4f}\n")
    
    # Create dataset
    print("Creating NEURALCUP dataset...")
    neuralcup_dataset = NEURALCUPDataset(
        preprocessed_dir=args.neuralcup_dir,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        augment=False
    )
    
    print(f"\n✓ Dataset created:")
    print(f"  Volumes: {len(neuralcup_dataset.volumes)}")
    print(f"  Total patches: {len(neuralcup_dataset)}")
    print(f"  Patches per volume: {args.patches_per_volume}\n")
    
    # Validate
    print("="*70)
    print("STARTING VALIDATION")
    print("="*70)
    
    results = validate_with_reconstruction(
        model, 
        neuralcup_dataset, 
        device, 
        tuple(args.patch_size)
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
        'internal_validation_dsc': float(val_dsc),
        'neuralcup_results': {
            'n_cases': len(results),
            'mean_dsc': float(np.mean(dscs)),
            'std_dsc': float(np.std(dscs)),
            'median_dsc': float(np.median(dscs)),
            'min_dsc': float(np.min(dscs)),
            'max_dsc': float(np.max(dscs)),
            'q25_dsc': float(np.percentile(dscs, 25)),
            'q75_dsc': float(np.percentile(dscs, 75))
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
    print(f"Mean DSC:   {summary['neuralcup_results']['mean_dsc']:.4f} ± {summary['neuralcup_results']['std_dsc']:.4f}")
    print(f"Median DSC: {summary['neuralcup_results']['median_dsc']:.4f}")
    print(f"Q25-Q75:    [{summary['neuralcup_results']['q25_dsc']:.4f}, {summary['neuralcup_results']['q75_dsc']:.4f}]")
    print(f"Range:      [{summary['neuralcup_results']['min_dsc']:.4f}, {summary['neuralcup_results']['max_dsc']:.4f}]")
    print(f"\nInternal validation DSC: {val_dsc:.4f}")
    print(f"External test DSC:       {summary['neuralcup_results']['mean_dsc']:.4f}")
    print(f"Generalization gap:      {val_dsc - summary['neuralcup_results']['mean_dsc']:.4f}")
    print(f"\nResults saved to: {output_dir / 'neuralcup_results.json'}")
    print("="*70 + "\n")
    
    # Distribution analysis
    print("DSC Distribution:")
    print(f"  0.0-0.2: {sum(1 for d in dscs if 0.0 <= d < 0.2)} cases")
    print(f"  0.2-0.4: {sum(1 for d in dscs if 0.2 <= d < 0.4)} cases")
    print(f"  0.4-0.6: {sum(1 for d in dscs if 0.4 <= d < 0.6)} cases")
    print(f"  0.6-0.8: {sum(1 for d in dscs if 0.6 <= d < 0.8)} cases")
    print(f"  0.8-1.0: {sum(1 for d in dscs if 0.8 <= d <= 1.0)} cases")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

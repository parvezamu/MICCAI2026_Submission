"""
Analyze What Pre-trained SimCLR Encoder Actually Learned

This script will:
1. Load your best_model.pth checkpoint
2. Pass real brain MRI volumes through it
3. Visualize feature maps at each layer
4. Save as NIfTI files for inspection
5. Show what patterns the encoder detects

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import json

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


class FeatureExtractor:
    """Extract and visualize features from each layer"""
    
    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder.to(device)
        self.device = device
        self.encoder.eval()
        
        # Storage for activations
        self.activations = {}
        self.hooks = []
        
        # Register hooks for all layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook each layer
        self.hooks.append(
            self.encoder.conv1.register_forward_hook(get_activation('conv1'))
        )
        self.hooks.append(
            self.encoder.bn1.register_forward_hook(get_activation('bn1'))
        )
        self.hooks.append(
            self.encoder.layer1.register_forward_hook(get_activation('layer1'))
        )
        self.hooks.append(
            self.encoder.layer2.register_forward_hook(get_activation('layer2'))
        )
        self.hooks.append(
            self.encoder.layer3.register_forward_hook(get_activation('layer3'))
        )
        self.hooks.append(
            self.encoder.layer4.register_forward_hook(get_activation('layer4'))
        )
    
    def extract_features(self, volume):
        """
        Extract features from a volume
        
        Args:
            volume: numpy array (D, H, W) or torch tensor (1, D, H, W)
        
        Returns:
            dict of activations at each layer
        """
        self.activations = {}
        
        # Prepare input
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
        
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)  # (1, C, D, H, W)
        
        volume = volume.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            _ = self.encoder(volume)
        
        return self.activations
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def load_pretrained_encoder(checkpoint_path):
    """Load the pre-trained encoder"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return simclr_model.encoder


def load_sample_volumes(preprocessed_dir, n_samples=5):
    """Load some sample volumes from your preprocessed data"""
    
    samples = []
    
    # ATLAS
    atlas_dir = Path(preprocessed_dir) / 'ATLAS'
    atlas_cases = sorted([d for d in atlas_dir.iterdir() if d.is_dir()])[:n_samples]
    
    print(f"\nLoading {len(atlas_cases)} ATLAS samples...")
    for case_dir in atlas_cases:
        try:
            # Load T1 volume
            t1_path = case_dir / 't1_preprocessed.npy'
            mask_path = case_dir / 'mask_preprocessed.npy'
            
            if t1_path.exists() and mask_path.exists():
                volume = np.load(t1_path)
                mask = np.load(mask_path)
                
                samples.append({
                    'case_id': case_dir.name,
                    'dataset': 'ATLAS',
                    'volume': volume,
                    'mask': mask,
                    'shape': volume.shape
                })
                print(f"  ✓ {case_dir.name}: {volume.shape}")
        except Exception as e:
            print(f"  ✗ {case_dir.name}: {e}")
    
    return samples


def visualize_layer_activations(activations, volume, mask, case_id, save_dir):
    """
    Visualize what each layer learned
    
    This shows you what features the encoder extracts
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get middle slices
    mid_d = volume.shape[0] // 2
    
    # Prepare figure
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    n_layers = len(layers)
    
    fig, axes = plt.subplots(3, n_layers, figsize=(n_layers * 4, 12))
    
    # Row 1: Original volume
    for col in range(n_layers):
        ax = axes[0, col]
        ax.imshow(volume[mid_d], cmap='gray')
        if col == 0:
            ax.set_ylabel('Original\nVolume', fontsize=12, fontweight='bold')
        ax.set_title(f'{layers[col]}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Row 2: Lesion mask
    for col in range(n_layers):
        ax = axes[1, col]
        ax.imshow(mask[mid_d], cmap='Reds', alpha=0.6)
        ax.imshow(volume[mid_d], cmap='gray', alpha=0.4)
        if col == 0:
            ax.set_ylabel('Lesion\nMask', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Row 3: Feature activations
    for col, layer_name in enumerate(layers):
        ax = axes[2, col]
        
        # Get activation (average across channels)
        feat = activations[layer_name][0].cpu().numpy()  # (C, D, H, W)
        feat_avg = feat.mean(axis=0)  # Average over channels (D, H, W)
        
        # Get middle slice
        feat_mid = feat_avg.shape[0] // 2
        
        # Normalize for visualization
        feat_slice = feat_avg[feat_mid]
        feat_slice = (feat_slice - feat_slice.min()) / (feat_slice.max() - feat_slice.min() + 1e-8)
        
        im = ax.imshow(feat_slice, cmap='jet')
        if col == 0:
            ax.set_ylabel('Learned\nFeatures', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Add info
        ax.text(0.5, -0.1, f'Shape: {feat.shape}\nChannels: {feat.shape[0]}',
                transform=ax.transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Feature Extraction: {case_id}\nWhat the Pre-trained Encoder Sees',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'{case_id}_feature_visualization.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: {save_path.name}")


def save_features_as_nifti(activations, volume_shape, affine, case_id, save_dir):
    """
    Save feature maps as NIfTI files for inspection in ITK-SNAP/3D Slicer
    
    This lets you load the features in medical imaging software
    """
    save_dir = Path(save_dir) / case_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Saving NIfTI files for {case_id}...")
    
    for layer_name, activation in activations.items():
        feat = activation[0].cpu().numpy()  # (C, D, H, W)
        
        # Save first 10 channels as separate volumes
        n_channels_to_save = min(10, feat.shape[0])
        
        for ch in range(n_channels_to_save):
            channel_vol = feat[ch]  # (D, H, W)
            
            # Resize to match original volume if needed
            # (Features are downsampled, so we upsample for visualization)
            from scipy.ndimage import zoom
            
            zoom_factors = [
                volume_shape[0] / channel_vol.shape[0],
                volume_shape[1] / channel_vol.shape[1],
                volume_shape[2] / channel_vol.shape[2]
            ]
            
            if any(z != 1.0 for z in zoom_factors):
                channel_vol = zoom(channel_vol, zoom_factors, order=1)
            
            # Ensure same shape
            channel_vol = channel_vol[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
            
            # Save as NIfTI
            nii = nib.Nifti1Image(channel_vol.astype(np.float32), affine)
            nii_path = save_dir / f'{layer_name}_channel_{ch:02d}.nii.gz'
            nib.save(nii, nii_path)
        
        # Also save average across channels
        feat_avg = feat.mean(axis=0)
        
        # Resize
        if any(z != 1.0 for z in zoom_factors):
            feat_avg = zoom(feat_avg, zoom_factors, order=1)
        feat_avg = feat_avg[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
        
        nii_avg = nib.Nifti1Image(feat_avg.astype(np.float32), affine)
        nii_avg_path = save_dir / f'{layer_name}_average.nii.gz'
        nib.save(nii_avg, nii_avg_path)
        
        print(f"    ✓ {layer_name}: {n_channels_to_save} channels + average")
    
    print(f"  ✓ NIfTI files saved to: {save_dir}")


def analyze_feature_statistics(activations, case_id):
    """
    Compute statistics about learned features
    """
    print(f"\n  Feature Statistics for {case_id}:")
    print(f"  {'Layer':<15} {'Channels':<10} {'Shape':<20} {'Mean':<10} {'Std':<10}")
    print(f"  {'-'*75}")
    
    stats = {}
    
    for layer_name, activation in activations.items():
        feat = activation[0].cpu().numpy()
        
        stats[layer_name] = {
            'n_channels': feat.shape[0],
            'shape': feat.shape,
            'mean': float(feat.mean()),
            'std': float(feat.std()),
            'min': float(feat.min()),
            'max': float(feat.max())
        }
        
        shape_str = f"{feat.shape}"
        print(f"  {layer_name:<15} {feat.shape[0]:<10} {shape_str:<20} "
              f"{feat.mean():<10.4f} {feat.std():<10.4f}")
    
    return stats


def compare_lesion_vs_healthy(activations, mask):
    """
    Compare feature activations in lesion vs healthy tissue
    
    This shows if the encoder already distinguishes lesions!
    """
    print(f"\n  Lesion vs Healthy Tissue Analysis:")
    
    results = {}
    
    for layer_name, activation in activations.items():
        feat = activation[0].cpu().numpy()  # (C, D, H, W)
        
        # Resize mask to match feature map
        from scipy.ndimage import zoom
        
        mask_resized = zoom(mask, 
                           [feat.shape[1] / mask.shape[0],
                            feat.shape[2] / mask.shape[1],
                            feat.shape[3] / mask.shape[2]],
                           order=0) > 0.5
        
        # Get activations in lesion vs healthy
        feat_flat = feat.reshape(feat.shape[0], -1)  # (C, N_voxels)
        mask_flat = mask_resized.flatten()
        
        lesion_activations = feat_flat[:, mask_flat].mean(axis=1)
        healthy_activations = feat_flat[:, ~mask_flat].mean(axis=1)
        
        # Compute difference
        diff = lesion_activations - healthy_activations
        
        results[layer_name] = {
            'lesion_mean': float(lesion_activations.mean()),
            'healthy_mean': float(healthy_activations.mean()),
            'difference': float(diff.mean()),
            'channels_higher_in_lesion': int((diff > 0).sum()),
            'channels_higher_in_healthy': int((diff < 0).sum())
        }
        
        print(f"    {layer_name}:")
        print(f"      Lesion activation:  {lesion_activations.mean():.4f}")
        print(f"      Healthy activation: {healthy_activations.mean():.4f}")
        print(f"      Difference:         {diff.mean():.4f}")
        print(f"      Channels higher in lesion:  {(diff > 0).sum()}/{feat.shape[0]}")
    
    return results


def main():
    # Paths
    checkpoint_path = Path('/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth')
    preprocessed_dir = Path('/home/pahm409/preprocessed_stroke_foundation')
    output_dir = Path('/home/pahm409/pretrained_encoder_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ANALYZING PRE-TRAINED ENCODER")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Load encoder
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    encoder = load_pretrained_encoder(checkpoint_path)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(encoder, device)
    
    # Load sample volumes
    samples = load_sample_volumes(preprocessed_dir, n_samples=5)
    
    if not samples:
        print("ERROR: No samples loaded!")
        return
    
    print(f"\n✓ Loaded {len(samples)} samples")
    
    # Analyze each sample
    all_stats = {}
    all_lesion_analysis = {}
    
    for idx, sample in enumerate(samples):
        case_id = sample['case_id']
        volume = sample['volume']
        mask = sample['mask']
        
        print(f"\n{'='*70}")
        print(f"Analyzing {case_id} ({idx+1}/{len(samples)})")
        print(f"{'='*70}")
        
        # Extract features
        print("  Extracting features...")
        activations = feature_extractor.extract_features(volume)
        
        # Visualize
        print("  Creating visualizations...")
        visualize_layer_activations(
            activations, volume, mask, case_id,
            output_dir / 'visualizations'
        )
        
        # Save as NIfTI
        affine = np.eye(4)  # Identity affine (can load from original if available)
        save_features_as_nifti(
            activations, volume.shape, affine, case_id,
            output_dir / 'nifti_features'
        )
        
        # Statistics
        stats = analyze_feature_statistics(activations, case_id)
        all_stats[case_id] = stats
        
        # Lesion analysis
        if mask.sum() > 0:  # Only if there's a lesion
            lesion_analysis = compare_lesion_vs_healthy(activations, mask)
            all_lesion_analysis[case_id] = lesion_analysis
    
    # Save summary
    summary = {
        'checkpoint': str(checkpoint_path),
        'n_samples': len(samples),
        'samples': [s['case_id'] for s in samples],
        'feature_statistics': all_stats,
        'lesion_analysis': all_lesion_analysis
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nWhat to do next:")
    print(f"  1. Check visualizations: {output_dir / 'visualizations'}")
    print(f"  2. Load NIfTI files in ITK-SNAP: {output_dir / 'nifti_features'}")
    print(f"  3. Review summary: {output_dir / 'analysis_summary.json'}")
    print(f"\nKey findings:")
    print(f"  - Feature maps show what encoder learned at each layer")
    print(f"  - Early layers: edges, tissue boundaries")
    print(f"  - Middle layers: anatomical structures")
    print(f"  - Deep layers: global brain patterns")
    print(f"  - Lesion analysis: does encoder already see lesions?")
    print(f"{'='*70}\n")
    
    # Cleanup
    feature_extractor.remove_hooks()


if __name__ == '__main__':
    main()

"""
compare_preprocessed.py

Compare preprocessed ATLAS vs NEURALCUP to find the issue
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_preprocessed(npz_file, dataset_name):
    """Analyze a preprocessed NPZ file"""
    
    data = np.load(npz_file)
    
    if 'volume' in data:
        volume = data['volume']
    elif 'image' in data:
        volume = data['image']
    else:
        print(f"Unknown keys: {list(data.keys())}")
        return
    
    # Get brain mask
    if 'brain_mask' in data:
        brain_mask = data['brain_mask']
    else:
        brain_mask = volume != 0
    
    brain_voxels = volume[brain_mask]
    
    print(f"\n{'='*70}")
    print(f"{dataset_name}: {npz_file.stem}")
    print(f"{'='*70}")
    print(f"Shape: {volume.shape}")
    print(f"Brain voxels: {brain_mask.sum():,} ({brain_mask.sum()/brain_mask.size*100:.1f}%)")
    print(f"\nIntensity statistics (brain only):")
    print(f"  Mean: {brain_voxels.mean():.6f}")
    print(f"  Std: {brain_voxels.std():.6f}")
    print(f"  Min: {brain_voxels.min():.6f}")
    print(f"  Max: {brain_voxels.max():.6f}")
    print(f"  Median: {np.median(brain_voxels):.6f}")
    print(f"  P1: {np.percentile(brain_voxels, 1):.6f}")
    print(f"  P99: {np.percentile(brain_voxels, 99):.6f}")
    
    # Check for NaN or Inf
    if np.isnan(volume).any():
        print(f"  ⚠️  Contains NaN values!")
    if np.isinf(volume).any():
        print(f"  ⚠️  Contains Inf values!")
    
    return {
        'volume': volume,
        'brain_mask': brain_mask,
        'brain_voxels': brain_voxels,
        'dataset': dataset_name
    }


def visualize_comparison(atlas_data, neuralcup_data):
    """Visualize middle slices"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ATLAS
    atlas_vol = atlas_data['volume']
    atlas_mid = atlas_vol.shape[2] // 2
    
    axes[0, 0].imshow(atlas_vol[:, :, atlas_mid], cmap='gray', vmin=-2, vmax=2)
    axes[0, 0].set_title(f'ATLAS - Axial (slice {atlas_mid})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(atlas_vol[:, atlas_vol.shape[1]//2, :], cmap='gray', vmin=-2, vmax=2)
    axes[0, 1].set_title('ATLAS - Coronal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(atlas_vol[atlas_vol.shape[0]//2, :, :], cmap='gray', vmin=-2, vmax=2)
    axes[0, 2].set_title('ATLAS - Sagittal')
    axes[0, 2].axis('off')
    
    # NEURALCUP
    nc_vol = neuralcup_data['volume']
    nc_mid = nc_vol.shape[2] // 2
    
    axes[1, 0].imshow(nc_vol[:, :, nc_mid], cmap='gray', vmin=-2, vmax=2)
    axes[1, 0].set_title(f'NEURALCUP - Axial (slice {nc_mid})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(nc_vol[:, nc_vol.shape[1]//2, :], cmap='gray', vmin=-2, vmax=2)
    axes[1, 1].set_title('NEURALCUP - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(nc_vol[nc_vol.shape[0]//2, :, :], cmap='gray', vmin=-2, vmax=2)
    axes[1, 2].set_title('NEURALCUP - Sagittal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('atlas_vs_neuralcup_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: atlas_vs_neuralcup_comparison.png")
    
    # Histogram comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(atlas_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='ATLAS', density=True)
    axes[0].hist(neuralcup_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='NEURALCUP', density=True)
    axes[0].set_xlabel('Normalized Intensity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Intensity Distribution Comparison')
    axes[0].legend()
    axes[0].set_xlim([-3, 3])
    
    axes[1].hist(atlas_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='ATLAS', density=True, cumulative=True)
    axes[1].hist(neuralcup_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='NEURALCUP', density=True, cumulative=True)
    axes[1].set_xlabel('Normalized Intensity')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()
    axes[1].set_xlim([-3, 3])
    
    plt.tight_layout()
    plt.savefig('intensity_distribution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved histogram: intensity_distribution_comparison.png")


if __name__ == '__main__':
    
    atlas_dir = Path('/home/pahm409/preprocessed_stroke_foundation/ATLAS')
    neuralcup_dir = Path('/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled')
    
    # Get first sample from each
    atlas_files = list(atlas_dir.glob("*.npz"))
    neuralcup_files = list(neuralcup_dir.glob("*.npz"))
    
    if not atlas_files:
        print(f"❌ No ATLAS files in {atlas_dir}")
        exit(1)
    
    if not neuralcup_files:
        print(f"❌ No NEURALCUP files in {neuralcup_dir}")
        exit(1)
    
    print("\n🔍 Analyzing preprocessed data...")
    
    atlas_data = analyze_preprocessed(atlas_files[0], "ATLAS")
    neuralcup_data = analyze_preprocessed(neuralcup_files[0], "NEURALCUP")
    
    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    atlas_mean = atlas_data['brain_voxels'].mean()
    nc_mean = neuralcup_data['brain_voxels'].mean()
    
    atlas_std = atlas_data['brain_voxels'].std()
    nc_std = neuralcup_data['brain_voxels'].std()
    
    print(f"Mean difference: {abs(atlas_mean - nc_mean):.6f}")
    print(f"Std difference: {abs(atlas_std - nc_std):.6f}")
    
    if abs(atlas_mean - nc_mean) > 0.5 or abs(atlas_std - nc_std) > 0.5:
        print(f"\n⚠️  WARNING: Large intensity distribution difference detected!")
        print(f"   This could explain the 0% DSC on NEURALCUP")
    
    visualize_comparison(atlas_data, neuralcup_data)

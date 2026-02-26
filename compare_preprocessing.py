"""
compare_preprocessed.py - FIXED v2

Compare preprocessed ATLAS vs NEURALCUP to find the issue
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_preprocessed(npz_file, dataset_name):
    """Analyze a preprocessed NPZ file"""
    
    data = np.load(npz_file)
    
    print(f"\n{'='*70}")
    print(f"Loading: {npz_file.name}")
    print(f"{'='*70}")
    print(f"Keys: {list(data.keys())}")
    
    if 'volume' in data:
        volume = data['volume']
    elif 'image' in data:
        volume = data['image']
    else:
        print(f"Unknown keys: {list(data.keys())}")
        return None
    
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Get brain mask - CRITICAL FIX: convert to boolean
    if 'brain_mask' in data:
        brain_mask_raw = data['brain_mask']
        print(f"Brain mask raw shape: {brain_mask_raw.shape}, dtype: {brain_mask_raw.dtype}")
        
        # Convert to boolean properly
        brain_mask = brain_mask_raw.astype(bool)
        print(f"Brain mask bool shape: {brain_mask.shape}, dtype: {brain_mask.dtype}")
        
        if brain_mask.shape != volume.shape:
            print(f"⚠️  Brain mask shape {brain_mask.shape} != volume shape {volume.shape}")
            brain_mask = (volume != 0)
    else:
        brain_mask = (volume != 0)
        print(f"No brain_mask key, using volume != 0")
    
    # Double check shapes match
    assert brain_mask.shape == volume.shape, f"Shape mismatch: mask {brain_mask.shape} vs volume {volume.shape}"
    assert brain_mask.dtype == bool or brain_mask.dtype == np.bool_, f"Brain mask not boolean: {brain_mask.dtype}"
    
    # Now safe to index
    brain_voxels = volume[brain_mask]
    
    print(f"\n{dataset_name}: {npz_file.stem}")
    print(f"Volume shape: {volume.shape}")
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
        'dataset': dataset_name,
        'file': npz_file.stem
    }


def visualize_comparison(atlas_data, neuralcup_data):
    """Visualize middle slices"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ATLAS
    atlas_vol = atlas_data['volume']
    atlas_mid = atlas_vol.shape[2] // 2
    
    axes[0, 0].imshow(atlas_vol[:, :, atlas_mid], cmap='gray', vmin=-2, vmax=2)
    axes[0, 0].set_title(f"ATLAS: {atlas_data['file']}\nAxial (slice {atlas_mid})")
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
    axes[1, 0].set_title(f"NEURALCUP: {neuralcup_data['file']}\nAxial (slice {nc_mid})")
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
    
    axes[0].hist(atlas_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='ATLAS', density=True, range=(-3, 3))
    axes[0].hist(neuralcup_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='NEURALCUP', density=True, range=(-3, 3))
    axes[0].set_xlabel('Normalized Intensity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Intensity Distribution Comparison')
    axes[0].legend()
    axes[0].set_xlim([-3, 3])
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(atlas_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='ATLAS', density=True, cumulative=True, range=(-3, 3))
    axes[1].hist(neuralcup_data['brain_voxels'].flatten(), bins=100, alpha=0.7, label='NEURALCUP', density=True, cumulative=True, range=(-3, 3))
    axes[1].set_xlabel('Normalized Intensity')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()
    axes[1].set_xlim([-3, 3])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('intensity_distribution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved histogram: intensity_distribution_comparison.png")


if __name__ == '__main__':
    
    # Use preprocessed directories
    atlas_dir = Path('/home/pahm409/preprocessed_stroke_foundation/ATLAS')
    neuralcup_dir = Path('/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled')
    
    print(f"\nLooking for ATLAS in: {atlas_dir}")
    print(f"Looking for NEURALCUP in: {neuralcup_dir}")
    
    # Get first sample from each
    atlas_files = sorted(list(atlas_dir.glob("*.npz")))
    neuralcup_files = sorted(list(neuralcup_dir.glob("*.npz")))
    
    print(f"\nFound {len(atlas_files)} ATLAS files")
    print(f"Found {len(neuralcup_files)} NEURALCUP files")
    
    if not atlas_files:
        print(f"❌ No ATLAS .npz files in {atlas_dir}")
        exit(1)
    
    if not neuralcup_files:
        print(f"❌ No NEURALCUP .npz files in {neuralcup_dir}")
        exit(1)
    
    print(f"\n🔍 Analyzing preprocessed data...")
    
    atlas_data = analyze_preprocessed(atlas_files[0], "ATLAS")
    
    if atlas_data is None:
        print("❌ Failed to load ATLAS data")
        exit(1)
    
    neuralcup_data = analyze_preprocessed(neuralcup_files[0], "NEURALCUP")
    
    if neuralcup_data is None:
        print("❌ Failed to load NEURALCUP data")
        exit(1)
    
    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    atlas_mean = atlas_data['brain_voxels'].mean()
    nc_mean = neuralcup_data['brain_voxels'].mean()
    
    atlas_std = atlas_data['brain_voxels'].std()
    nc_std = neuralcup_data['brain_voxels'].std()
    
    print(f"\nMean:")
    print(f"  ATLAS:     {atlas_mean:8.6f}")
    print(f"  NEURALCUP: {nc_mean:8.6f}")
    print(f"  Difference: {abs(atlas_mean - nc_mean):.6f}")
    
    print(f"\nStd:")
    print(f"  ATLAS:     {atlas_std:8.6f}")
    print(f"  NEURALCUP: {nc_std:8.6f}")
    print(f"  Difference: {abs(atlas_std - nc_std):.6f}")
    
    if abs(atlas_mean - nc_mean) > 0.5:
        print(f"\n⚠️  WARNING: Large MEAN difference!")
        print(f"   → This explains the 0% DSC on NEURALCUP")
    
    if abs(atlas_std - nc_std) > 0.5:
        print(f"\n⚠️  WARNING: Large STD difference!")
        print(f"   → This explains the 0% DSC on NEURALCUP")
    
    if abs(atlas_mean - nc_mean) < 0.2 and abs(atlas_std - nc_std) < 0.3:
        print(f"\n✅ Intensity distributions are similar")
        print(f"   → Preprocessing is OK")
        print(f"   → Problem is likely scanner/population differences")
    
    print(f"\n📊 Creating visualizations...")
    visualize_comparison(atlas_data, neuralcup_data)
    
    print(f"\n✅ Done! Check:")
    print(f"   - atlas_vs_neuralcup_comparison.png")
    print(f"   - intensity_distribution_comparison.png")

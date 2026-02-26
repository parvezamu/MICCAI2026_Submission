"""
compare_lesion_appearance.py

Compare actual voxel patterns in working vs failing lesions
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def compare_lesions():
    neuralcup_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP')
    
    # Load working case
    bbs318 = np.load(neuralcup_dir / 'BBS318.npz')
    vol_318 = bbs318['image']
    mask_318 = bbs318['lesion_mask']
    
    # Load failing case  
    bbs328 = np.load(neuralcup_dir / 'BBS328.npz')
    vol_328 = bbs328['image']
    mask_328 = bbs328['lesion_mask']
    
    # Load a training case for comparison
    atlas_dir = Path('/home/pahm409/preprocessed_stroke_foundation/ATLAS')
    atlas = np.load(list(atlas_dir.glob('*.npz'))[0])
    vol_atlas = atlas['image']
    mask_atlas = atlas['lesion_mask']
    
    print("="*70)
    print("LESION APPEARANCE COMPARISON")
    print("="*70)
    
    for name, vol, mask in [
        ("BBS318 (WORKING)", vol_318, mask_318),
        ("BBS328 (FAILING)", vol_328, mask_328),
        ("ATLAS (TRAINING)", vol_atlas, mask_atlas)
    ]:
        lesion_voxels = vol[mask > 0]
        brain_voxels = vol[(vol > -0.5) & (mask == 0)]  # Brain tissue only
        
        if len(lesion_voxels) > 0:
            print(f"\n{name}:")
            print(f"  Lesion size: {mask.sum()} voxels")
            print(f"  Lesion intensity:")
            print(f"    Mean: {lesion_voxels.mean():.4f}")
            print(f"    Std:  {lesion_voxels.std():.4f}")
            print(f"    Min:  {lesion_voxels.min():.4f}")
            print(f"    Max:  {lesion_voxels.max():.4f}")
            print(f"    Percentiles: 25%={np.percentile(lesion_voxels, 25):.4f}, "
                  f"50%={np.percentile(lesion_voxels, 50):.4f}, "
                  f"75%={np.percentile(lesion_voxels, 75):.4f}")
            
            print(f"  Brain intensity:")
            print(f"    Mean: {brain_voxels.mean():.4f}")
            print(f"    Std:  {brain_voxels.std():.4f}")
            
            print(f"  Signal-to-noise: {(lesion_voxels.mean() - brain_voxels.mean()) / brain_voxels.std():.4f}")
            
            # Check texture (variance within lesion)
            print(f"  Lesion homogeneity (CV): {lesion_voxels.std() / lesion_voxels.mean():.4f}" 
                  if lesion_voxels.mean() != 0 else "  N/A")
    
    print("\n" + "="*70)
    
    # Extract a slice through the lesion center for visual inspection
    for name, vol, mask, case_id in [
        ("BBS318", vol_318, mask_318, "bbs318"),
        ("BBS328", vol_328, mask_328, "bbs328")
    ]:
        # Find lesion center
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            center_slice = int(np.mean(coords[0]))
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(vol[center_slice], cmap='gray', vmin=-2, vmax=2)
            axes[0].set_title(f'{name} - T1 Image')
            
            axes[1].imshow(mask[center_slice], cmap='Reds', alpha=0.7)
            axes[1].set_title(f'{name} - Ground Truth Lesion')
            
            overlay = vol[center_slice].copy()
            axes[2].imshow(overlay, cmap='gray', vmin=-2, vmax=2)
            axes[2].imshow(mask[center_slice], cmap='Reds', alpha=0.4)
            axes[2].set_title(f'{name} - Overlay')
            
            plt.tight_layout()
            plt.savefig(f'/home/pahm409/{case_id}_slice.png', dpi=150)
            print(f"Saved visualization: /home/pahm409/{case_id}_slice.png")

if __name__ == '__main__':
    compare_lesions()

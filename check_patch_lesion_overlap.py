"""
check_patch_lesion_overlap.py

Check if random patches are actually sampling lesions

Author: Parvez
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
from pathlib import Path

def check_patch_overlap():
    neuralcup_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP')
    
    patch_size = np.array([96, 96, 96])
    half_size = patch_size // 2
    num_patches = 10
    
    # Cases that WORK
    working = ['BBS315', 'BBS318', 'BBS334', 'BBS337', 'BBS338']
    
    # Cases that FAIL
    failing = ['BBS302', 'BBS304', 'BBS314', 'BBS319', 'BBS328']
    
    np.random.seed(42)  # Same seed as test script!
    
    print("="*70)
    print("CHECKING PATCH-LESION OVERLAP")
    print("="*70)
    
    for case_list, label in [(working, "WORKING"), (failing, "FAILING")]:
        print(f"\n{label} CASES:")
        print("-"*70)
        
        for case_id in case_list:
            npz_file = neuralcup_dir / f'{case_id}.npz'
            if not npz_file.exists():
                continue
                
            data = np.load(npz_file)
            volume = data['image']
            mask = data['lesion_mask']
            
            vol_shape = np.array(volume.shape)
            min_center = half_size
            max_center = vol_shape - half_size
            
            for dim in range(3):
                if min_center[dim] >= max_center[dim]:
                    min_center[dim] = vol_shape[dim] // 2
                    max_center[dim] = vol_shape[dim] // 2
            
            # Generate same random patches as test script
            patches_with_lesion = 0
            total_lesion_voxels_in_patches = 0
            
            for _ in range(num_patches):
                center = np.array([
                    np.random.randint(min_center[0], max_center[0] + 1),
                    np.random.randint(min_center[1], max_center[1] + 1),
                    np.random.randint(min_center[2], max_center[2] + 1)
                ])
                
                lower = center - half_size
                upper = center + half_size
                
                patch_mask = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
                lesion_voxels = patch_mask.sum()
                
                if lesion_voxels > 0:
                    patches_with_lesion += 1
                    total_lesion_voxels_in_patches += lesion_voxels
            
            lesion_ratio = patches_with_lesion / num_patches
            total_lesion = mask.sum()
            coverage = (total_lesion_voxels_in_patches / total_lesion * 100) if total_lesion > 0 else 0
            
            print(f"{case_id}: {patches_with_lesion}/{num_patches} patches hit lesion ({lesion_ratio:.1%}), "
                  f"Coverage: {coverage:.1f}% of lesion")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    check_patch_overlap()

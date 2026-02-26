"""
check_lesion_intensities.py

Compare lesion intensities in working vs failing cases

Author: Parvez
"""

import numpy as np
from pathlib import Path

def check_intensities():
    neuralcup_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP')
    
    working = ['BBS315', 'BBS318', 'BBS334', 'BBS337', 'BBS338']
    failing = ['BBS302', 'BBS304', 'BBS314', 'BBS319', 'BBS328']
    
    print("="*70)
    print("LESION vs NON-LESION INTENSITIES")
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
            brain_mask = data['brain_mask']
            
            # Lesion intensities
            lesion_voxels = volume[mask > 0]
            
            # Non-lesion brain intensities
            non_lesion_brain = volume[(brain_mask > 0) & (mask == 0)]
            
            if len(lesion_voxels) > 0 and len(non_lesion_brain) > 0:
                lesion_mean = lesion_voxels.mean()
                lesion_std = lesion_voxels.std()
                brain_mean = non_lesion_brain.mean()
                brain_std = non_lesion_brain.std()
                
                # Contrast = how different lesion is from normal brain
                contrast = abs(lesion_mean - brain_mean) / brain_std if brain_std > 0 else 0
                
                # Check if lesion is brighter or darker
                if lesion_mean > brain_mean:
                    lesion_type = "BRIGHTER"
                else:
                    lesion_type = "DARKER"
                
                print(f"{case_id}: Lesion={lesion_type}, "
                      f"Lesion μ={lesion_mean:.3f}±{lesion_std:.3f}, "
                      f"Brain μ={brain_mean:.3f}±{brain_std:.3f}, "
                      f"Contrast={contrast:.2f}")
    
    print("\n" + "="*70)
    
    # Also check ATLAS/UOA for comparison
    print("\nTRAINING DATA (ATLAS/UOA) - Sample:")
    print("-"*70)
    
    atlas_dir = Path('/home/pahm409/preprocessed_stroke_foundation/ATLAS')
    uoa_dir = Path('/home/pahm409/preprocessed_stroke_foundation/UOA_Private')
    
    for npz_file in list(atlas_dir.glob('*.npz'))[:3]:
        data = np.load(npz_file)
        volume = data['image']
        mask = data['lesion_mask']
        brain_mask = data['brain_mask']
        
        lesion_voxels = volume[mask > 0]
        non_lesion_brain = volume[(brain_mask > 0) & (mask == 0)]
        
        if len(lesion_voxels) > 0:
            lesion_mean = lesion_voxels.mean()
            brain_mean = non_lesion_brain.mean()
            brain_std = non_lesion_brain.std()
            contrast = abs(lesion_mean - brain_mean) / brain_std if brain_std > 0 else 0
            lesion_type = "BRIGHTER" if lesion_mean > brain_mean else "DARKER"
            
            print(f"{npz_file.stem}: Lesion={lesion_type}, Contrast={contrast:.2f}")

if __name__ == '__main__':
    check_intensities()

"""
check_lesion_locations.py

Check where lesions are located in failing vs working cases

Author: Parvez
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
from pathlib import Path

def check_locations():
    neuralcup_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm_FIXED/NEURALCUP')
    
    # Cases that WORK (DSC > 0.3)
    working = ['BBS315', 'BBS318', 'BBS334', 'BBS337', 'BBS338', 'BBS353', 'BBS355']
    
    # Cases that FAIL (Pred = 0)
    failing = ['BBS302', 'BBS304', 'BBS314', 'BBS319', 'BBS328', 'BBS342', 'BBS357']
    
    print("="*70)
    print("WORKING CASES - Lesion Centers")
    print("="*70)
    for case_id in working:
        npz_file = neuralcup_dir / f'{case_id}.npz'
        if npz_file.exists():
            data = np.load(npz_file)
            mask = data['lesion_mask']
            
            # Find lesion center of mass
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                center = (
                    int(np.mean(coords[0])),
                    int(np.mean(coords[1])),
                    int(np.mean(coords[2]))
                )
                print(f"{case_id}: Center={center}, Volume={mask.sum()}")
    
    print("\n" + "="*70)
    print("FAILING CASES - Lesion Centers")
    print("="*70)
    for case_id in failing:
        npz_file = neuralcup_dir / f'{case_id}.npz'
        if npz_file.exists():
            data = np.load(npz_file)
            mask = data['lesion_mask']
            
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                center = (
                    int(np.mean(coords[0])),
                    int(np.mean(coords[1])),
                    int(np.mean(coords[2]))
                )
                print(f"{case_id}: Center={center}, Volume={mask.sum()}")

if __name__ == '__main__':
    check_locations()

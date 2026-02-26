"""
fix_neuralcup_dimensions.py

Pad NEURALCUP volumes to match training dimensions exactly

Author: Parvez
Date: January 2026
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def fix_dimensions():
    """Pad NEURALCUP to 182×218×182 to match UOA_Private"""
    
    input_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP')
    output_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm_FIXED/NEURALCUP')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_shape = (182, 218, 182)  # Match UOA_Private exactly!
    
    print("\n" + "="*70)
    print("FIXING NEURALCUP DIMENSIONS")
    print("="*70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target shape: {target_shape}")
    print("="*70 + "\n")
    
    npz_files = sorted(input_dir.glob('*.npz'))
    
    for npz_file in tqdm(npz_files, desc="Fixing dimensions"):
        data = np.load(npz_file)
        
        image = data['image']
        mask = data['lesion_mask']
        brain_mask = data['brain_mask']
        
        current_shape = image.shape
        
        # Calculate padding needed
        pad_d = target_shape[0] - current_shape[0]
        pad_h = target_shape[1] - current_shape[1]
        pad_w = target_shape[2] - current_shape[2]
        
        if pad_d != 0 or pad_h != 0 or pad_w != 0:
            # Pad symmetrically (or as close as possible)
            pad_width = (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2)
            )
            
            # Pad image with background value
            image_padded = np.pad(image, pad_width, mode='constant', constant_values=image.min())
            
            # Pad masks with 0
            mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=0)
            brain_mask_padded = np.pad(brain_mask, pad_width, mode='constant', constant_values=0)
        else:
            image_padded = image
            mask_padded = mask
            brain_mask_padded = brain_mask
        
        # Verify shape
        assert image_padded.shape == target_shape, f"Shape mismatch: {image_padded.shape} vs {target_shape}"
        assert mask_padded.shape == target_shape
        assert brain_mask_padded.shape == target_shape
        
        # Save
        output_path = output_dir / npz_file.name
        np.savez_compressed(
            output_path,
            image=image_padded.astype(np.float32),
            lesion_mask=mask_padded.astype(np.uint8),
            brain_mask=brain_mask_padded.astype(np.uint8)
        )
    
    print("\n" + "="*70)
    print("DIMENSION FIXING COMPLETE!")
    print("="*70)
    print(f"Fixed {len(npz_files)} volumes")
    print(f"All volumes now: {target_shape}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Verify
    print("Verifying shapes...")
    for npz_file in list(output_dir.glob('*.npz'))[:5]:
        data = np.load(npz_file)
        print(f"  {npz_file.name}: {data['image'].shape}")


if __name__ == '__main__':
    fix_dimensions()

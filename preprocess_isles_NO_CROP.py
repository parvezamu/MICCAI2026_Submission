#!/usr/bin/env python3
"""
preprocess_isles_NO_CROP.py

Preprocess WITHOUT cropping - just resample directly to 96³
Use this for evaluation
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
import argparse

def normalize_volume(volume):
    nonzero_mask = volume > 0
    if nonzero_mask.sum() == 0:
        return volume
    
    mean = volume[nonzero_mask].mean()
    std = volume[nonzero_mask].std()
    
    if std < 1e-8:
        return volume
    
    normalized = np.zeros_like(volume)
    normalized[nonzero_mask] = (volume[nonzero_mask] - mean) / std
    return normalized

def resample_volume(volume, target_shape):
    if volume.shape == target_shape:
        return volume
    
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return zoom(volume, zoom_factors, order=1)

def process_isles_case(case_dir, output_dir, target_shape=(96, 96, 96)):
    case_name = case_dir.name
    
    dwi_file = list(case_dir.glob("*_dwi.nii.gz"))
    adc_file = list(case_dir.glob("*_adc.nii.gz"))
    mask_file = list(case_dir.glob("*_msk.nii.gz"))
    
    if not dwi_file or not adc_file or not mask_file:
        return False, "Missing files"
    
    try:
        dwi_nii = nib.load(str(dwi_file[0]))
        dwi_data = dwi_nii.get_fdata()
        
        adc_nii = nib.load(str(adc_file[0]))
        adc_data = adc_nii.get_fdata()
        
        mask_nii = nib.load(str(mask_file[0]))
        mask_data = mask_nii.get_fdata()
        
        original_shape = dwi_data.shape
        original_affine = dwi_nii.affine
        
        # NO CROPPING - resample directly
        dwi_resampled = resample_volume(dwi_data, target_shape)
        adc_resampled = resample_volume(adc_data, target_shape)
        mask_resampled = resample_volume(mask_data, target_shape)
        
        mask_binary = (mask_resampled > 0.5).astype(np.uint8)
        
        dwi_normalized = normalize_volume(dwi_resampled)
        adc_normalized = normalize_volume(adc_resampled)
        
        # Save with metadata
        output_file = output_dir / f"{case_name}.npz"
        np.savez_compressed(
            output_file,
            dwi=dwi_normalized.astype(np.float32),
            adc=adc_normalized.astype(np.float32),
            mask=mask_binary.astype(np.uint8),
            original_shape=np.array(original_shape, dtype=np.int32),
            original_affine=original_affine.astype(np.float32)
        )
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/derivatives')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/preprocessed_isles_dual_NO_CROP')
    parser.add_argument('--target-shape', type=int, nargs=3, default=[96, 96, 96])
    args = parser.parse_args()
    
    isles_dir = Path(args.isles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_shape = tuple(args.target_shape)
    
    print("="*80)
    print("PREPROCESSING WITHOUT CROPPING (FOR EVALUATION)")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Target: {target_shape}")
    print("="*80 + "\n")
    
    case_dirs = sorted([d for d in isles_dir.iterdir() if d.is_dir()])
    
    success_count = 0
    for case_dir in tqdm(case_dirs):
        success, msg = process_isles_case(case_dir, output_dir, target_shape)
        if success:
            success_count += 1
    
    print(f"\n✓ Processed {success_count}/{len(case_dirs)}")

if __name__ == '__main__':
    main()

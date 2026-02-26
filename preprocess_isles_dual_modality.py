"""
preprocess_isles_dual_modality.py

Preprocess ISLES 2022 with BOTH DWI and ADC modalities
Saves as .npz with keys: 'dwi', 'adc', 'mask'
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
import argparse

def normalize_volume(volume):
    """Z-score normalization"""
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
    """Resample to target shape"""
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return zoom(volume, zoom_factors, order=1)

def process_isles_case(case_dir, output_dir, target_shape=(197, 233, 189)):
    """Process single ISLES case with DWI + ADC"""
    case_name = case_dir.name
    
    # Find files
    dwi_file = list(case_dir.glob("*_dwi.nii.gz"))
    adc_file = list(case_dir.glob("*_adc.nii.gz"))
    mask_file = list(case_dir.glob("*_msk.nii.gz"))
    
    if not dwi_file or not adc_file or not mask_file:
        return False, "Missing files"
    
    try:
        # Load DWI
        dwi_nii = nib.load(str(dwi_file[0]))
        dwi_data = dwi_nii.get_fdata()
        
        # Load ADC
        adc_nii = nib.load(str(adc_file[0]))
        adc_data = adc_nii.get_fdata()
        
        # Load mask
        mask_nii = nib.load(str(mask_file[0]))
        mask_data = mask_nii.get_fdata()
        
        # Resample all to target shape
        dwi_resampled = resample_volume(dwi_data, target_shape)
        adc_resampled = resample_volume(adc_data, target_shape)
        mask_resampled = resample_volume(mask_data, target_shape)
        
        # Binarize mask
        mask_binary = (mask_resampled > 0).astype(np.uint8)
        
        # Normalize
        dwi_normalized = normalize_volume(dwi_resampled)
        adc_normalized = normalize_volume(adc_resampled)
        
        # Save
        output_file = output_dir / f"{case_name}.npz"
        np.savez_compressed(
            output_file,
            dwi=dwi_normalized.astype(np.float32),
            adc=adc_normalized.astype(np.float32),
            mask=mask_binary.astype(np.uint8)
        )
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/ISLES2022/',
                       help='ISLES raw data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/preprocessed_isles_dual_modality',
                       help='Output directory')
    args = parser.parse_args()
    
    isles_dir = Path(args.isles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREPROCESSING ISLES 2022 - DUAL MODALITY (DWI + ADC)")
    print("="*80)
    print(f"Input: {isles_dir}")
    print(f"Output: {output_dir}")
    print(f"Target shape: (197, 233, 189)")
    print("="*80 + "\n")
    
    # Get all case directories
    case_dirs = sorted([d for d in isles_dir.iterdir() if d.is_dir()])
    print(f"Found {len(case_dirs)} cases\n")
    
    success_count = 0
    failed_cases = []
    
    for case_dir in tqdm(case_dirs, desc='Processing'):
        success, msg = process_isles_case(case_dir, output_dir)
        if success:
            success_count += 1
        else:
            failed_cases.append((case_dir.name, msg))
    
    print(f"\n{'='*80}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(case_dirs)}")
    
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for case, msg in failed_cases[:10]:
            print(f"  {case}: {msg}")
    
    print(f"\nOutput: {output_dir}")
    print(f"{'='*80}\n")
    
    # Verify a sample file
    sample_files = list(output_dir.glob("*.npz"))
    if sample_files:
        print("Verifying sample file...")
        sample = np.load(sample_files[0])
        print(f"  File: {sample_files[0].name}")
        print(f"  Keys: {list(sample.files)}")
        print(f"  DWI shape: {sample['dwi'].shape}")
        print(f"  ADC shape: {sample['adc'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Mask sum: {sample['mask'].sum()}")
        print("\n✓ Preprocessing successful!\n")

if __name__ == '__main__':
    main()

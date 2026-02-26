"""
preprocess_isles_dual_modality_FIXED.py

Preprocess ISLES 2022 with BOTH DWI and ADC modalities
INCLUDES: Foreground cropping + nonzero mask normalization
Saves as .npz with keys: 'dwi', 'adc', 'mask'
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom, binary_erosion, binary_dilation
import argparse

def get_brain_bbox(volume, margin=5):
    """
    Get bounding box of brain region (foreground)
    Margin: extra voxels to include around detected brain
    """
    # Find nonzero voxels
    nonzero_mask = volume > 0
    
    # Dilate slightly to ensure we don't crop actual brain
    nonzero_mask = binary_dilation(nonzero_mask, iterations=2)
    
    # Get bounding box coordinates
    coords = np.where(nonzero_mask)
    
    if len(coords[0]) == 0:
        # No foreground found, return full volume
        return None
    
    # Get min/max for each dimension with margin
    bbox = []
    for i in range(3):
        min_idx = max(0, coords[i].min() - margin)
        max_idx = min(volume.shape[i], coords[i].max() + margin + 1)
        bbox.append((min_idx, max_idx))
    
    return bbox

def crop_to_bbox(volume, bbox):
    """Crop volume to bounding box"""
    if bbox is None:
        return volume
    
    return volume[
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]
    ]

def normalize_volume(volume):
    """
    Z-score normalization with nonzero mask
    (Following Jeong et al. - "nonzero mask normalization")
    """
    nonzero_mask = volume > 0
    
    if nonzero_mask.sum() == 0:
        return volume
    
    # Compute mean and std ONLY on nonzero voxels (brain tissue)
    mean = volume[nonzero_mask].mean()
    std = volume[nonzero_mask].std()
    
    if std < 1e-8:
        return volume
    
    # Normalize: make mean ≈ 0, std ≈ 1
    normalized = np.zeros_like(volume)
    normalized[nonzero_mask] = (volume[nonzero_mask] - mean) / std
    
    return normalized

def resample_volume(volume, target_shape):
    """Resample to target shape"""
    if volume.shape == target_shape:
        return volume
    
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return zoom(volume, zoom_factors, order=1)

def process_isles_case(case_dir, output_dir, target_shape=(128, 128, 128)):
    """
    Process single ISLES case with:
    1. Load DWI, ADC, mask
    2. Crop to foreground (brain region)
    3. Resample to target shape
    4. Z-score normalize on nonzero voxels
    5. Save as .npz
    """
    case_name = case_dir.name
    
    # Find files
    dwi_file = list(case_dir.glob("*_dwi.nii.gz"))
    adc_file = list(case_dir.glob("*_adc.nii.gz"))
    mask_file = list(case_dir.glob("*_msk.nii.gz"))
    
    if not dwi_file or not adc_file or not mask_file:
        return False, "Missing files"
    
    try:
        # Load volumes
        dwi_nii = nib.load(str(dwi_file[0]))
        dwi_data = dwi_nii.get_fdata()
        
        adc_nii = nib.load(str(adc_file[0]))
        adc_data = adc_nii.get_fdata()
        
        mask_nii = nib.load(str(mask_file[0]))
        mask_data = mask_nii.get_fdata()
        
        # STEP 1: Crop to foreground (brain region)
        # Use DWI to detect brain (usually has better contrast)
        bbox = get_brain_bbox(dwi_data, margin=5)
        
        if bbox is not None:
            dwi_cropped = crop_to_bbox(dwi_data, bbox)
            adc_cropped = crop_to_bbox(adc_data, bbox)
            mask_cropped = crop_to_bbox(mask_data, bbox)
        else:
            dwi_cropped = dwi_data
            adc_cropped = adc_data
            mask_cropped = mask_data
        
        # STEP 2: Resample to target shape
        dwi_resampled = resample_volume(dwi_cropped, target_shape)
        adc_resampled = resample_volume(adc_cropped, target_shape)
        mask_resampled = resample_volume(mask_cropped, target_shape)
        
        # STEP 3: Binarize mask
        mask_binary = (mask_resampled > 0.5).astype(np.uint8)
        
        # STEP 4: Normalize (nonzero mask normalization)
        dwi_normalized = normalize_volume(dwi_resampled)
        adc_normalized = normalize_volume(adc_resampled)
        
        # STEP 5: Save
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
                       default='/home/pahm409/ISLES2022_reg/ISLES2022',
                       help='ISLES raw data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/preprocessed_isles_dual_v2',
                       help='Output directory')
    parser.add_argument('--target-shape', type=int, nargs=3,
                       default=[96, 96, 96],
                       help='Target shape after resampling')
    args = parser.parse_args()
    
    isles_dir = Path(args.isles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_shape = tuple(args.target_shape)
    
    print("="*80)
    print("PREPROCESSING ISLES 2022 - DUAL MODALITY (WITH FOREGROUND CROPPING)")
    print("="*80)
    print(f"Input: {isles_dir}")
    print(f"Output: {output_dir}")
    print(f"Target shape: {target_shape}")
    print("\nPreprocessing steps:")
    print("  1. Load DWI, ADC, mask")
    print("  2. Crop to foreground (brain region)")
    print("  3. Resample to target shape")
    print("  4. Z-score normalize on nonzero voxels")
    print("  5. Save as .npz")
    print("="*80 + "\n")
    
    # Get all case directories
    case_dirs = sorted([d for d in isles_dir.iterdir() if d.is_dir()])
    print(f"Found {len(case_dirs)} cases\n")
    
    success_count = 0
    failed_cases = []
    
    for case_dir in tqdm(case_dirs, desc='Processing'):
        success, msg = process_isles_case(case_dir, output_dir, target_shape)
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
    
    # Verify sample files
    sample_files = list(output_dir.glob("*.npz"))
    if sample_files:
        print("Verifying sample files...")
        for sample_file in sample_files[:3]:
            sample = np.load(sample_file)
            dwi = sample['dwi']
            adc = sample['adc']
            mask = sample['mask']
            
            print(f"\n{sample_file.name}:")
            print(f"  DWI: shape={dwi.shape}, range=[{dwi.min():.3f}, {dwi.max():.3f}], "
                  f"mean={dwi.mean():.3f}, std={dwi.std():.3f}")
            print(f"  ADC: shape={adc.shape}, range=[{adc.min():.3f}, {adc.max():.3f}], "
                  f"mean={adc.mean():.3f}, std={adc.std():.3f}")
            print(f"  Mask: positives={mask.sum()}")
        
        print("\n✓ Preprocessing successful!\n")

if __name__ == '__main__':
    main()

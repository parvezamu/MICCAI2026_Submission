"""
preprocess_neuralcup.py

Preprocess NEURALCUP dataset with mismatched image/mask naming and dimensions
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
import json


def resample_volume(volume, target_shape, order=1):
    """Resample volume to target shape"""
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    zoom_factors = target_shape / current_shape
    
    resampled = zoom(volume, zoom_factors, order=order)
    
    # Ensure exact dimensions
    if resampled.shape != tuple(target_shape):
        resampled_fixed = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        resampled = resampled_fixed
    
    return resampled


def resample_using_affine(volume, src_affine, tgt_affine, tgt_shape, order=1):
    """Resample volume from source space to target space using affine transforms"""
    from scipy.ndimage import affine_transform
    
    transform = np.linalg.inv(src_affine) @ tgt_affine
    matrix = transform[:3, :3]
    offset = transform[:3, 3]
    
    resampled = affine_transform(
        volume,
        matrix=np.linalg.inv(matrix),
        offset=np.linalg.solve(matrix, -offset),
        output_shape=tgt_shape,
        order=order,
        mode='constant',
        cval=0
    )
    
    return resampled


def find_matching_mask(image_file, masks_dir):
    """
    Find matching mask for an image file
    
    Image naming: wmBBS001_3DT1.nii
    Mask naming: BBS001.nii.gz (or different ID)
    """
    # Extract ID from image filename
    # wmBBS001_3DT1.nii -> 001
    img_name = image_file.stem.replace('.nii', '')  # wmBBS001_3DT1
    
    # Try to extract number
    # wmBBS001_3DT1 -> 001
    import re
    match = re.search(r'BBS(\d+)', img_name)
    
    if not match:
        return None
    
    bbs_id = match.group(1)  # e.g., "001"
    
    # Try different mask naming patterns
    mask_candidates = [
        masks_dir / f"BBS{bbs_id}.nii.gz",
        masks_dir / f"BBS{bbs_id}.nii",
        masks_dir / f"wmBBS{bbs_id}.nii.gz",
        masks_dir / f"wmBBS{bbs_id}.nii",
    ]
    
    for candidate in mask_candidates:
        if candidate.exists():
            return candidate
    
    return None


def preprocess_neuralcup(
    neuralcup_dir,
    output_dir,
    target_shape=(197, 233, 189)
):
    """
    Preprocess NEURALCUP T1 dataset
    """
    
    neuralcup = Path(neuralcup_dir)
    images_dir = neuralcup / 'images'
    masks_dir = neuralcup / 'masks'
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Find all T1 images (both .nii and .nii.gz)
    image_files = list(images_dir.glob("*_3DT1.nii")) + list(images_dir.glob("*_3DT1.nii.gz"))
    image_files = sorted(image_files)
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING NEURALCUP T1")
    print(f"{'='*70}")
    print(f"Images dir: {images_dir}")
    print(f"Masks dir: {masks_dir}")
    print(f"Found {len(image_files)} T1 images")
    print(f"Output dir: {output}")
    
    if len(image_files) == 0:
        print(f"❌ No T1 images found!")
        return
    
    original_info = {}
    successful = 0
    skipped_no_mask = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc="Processing NEURALCUP T1"):
        
        # Extract case ID: wmBBS001_3DT1.nii -> wmBBS001
        case_id = img_file.stem.replace('.nii', '').replace('_3DT1', '')
        
        # Find matching mask
        mask_file = find_matching_mask(img_file, masks_dir)
        
        if mask_file is None:
            skipped_no_mask += 1
            continue
        
        try:
            # Load image and mask
            img_nib = nib.load(img_file)
            mask_nib = nib.load(mask_file)
            
            volume = img_nib.get_fdata().astype(np.float32)
            mask = mask_nib.get_fdata().astype(np.uint8)
            
            img_shape = volume.shape
            mask_shape = mask.shape
            img_affine = img_nib.affine
            mask_affine = mask_nib.affine
            
            # Check if dimensions match
            if img_shape != mask_shape:
                # Resample mask to image space using affines
                mask = resample_using_affine(
                    mask, 
                    src_affine=mask_affine,
                    tgt_affine=img_affine,
                    tgt_shape=img_shape,
                    order=0  # Nearest neighbor for masks
                )
            
            # Verify mask has lesions
            if mask.sum() == 0:
                print(f"\n  ⚠️  {case_id}: Empty mask, skipping")
                skipped_no_mask += 1
                continue
            
            # Resample both to target shape
            volume_resampled = resample_volume(volume, target_shape, order=1)
            mask_resampled = resample_volume(mask, target_shape, order=0)
            
            # Normalize intensity (z-score with clipping)
            if volume_resampled.max() > 0:
                p99 = np.percentile(volume_resampled[volume_resampled > 0], 99.5)
                volume_resampled = np.clip(volume_resampled, 0, p99)
                volume_resampled = (volume_resampled - volume_resampled.mean()) / (volume_resampled.std() + 1e-8)
            
            # Store original info
            original_info[case_id] = {
                'original_shape': list(img_shape),
                'original_affine': img_affine.tolist(),
                'zoom_factors': (np.array(img_shape) / np.array(target_shape)).tolist(),
                'mask_file': str(mask_file.name),
                'image_file': str(img_file.name)
            }
            
            # Save as .npz
            output_file = output / f"{case_id}.npz"
            np.savez_compressed(
                output_file,
                volume=volume_resampled.astype(np.float32),
                mask=mask_resampled.astype(np.uint8)
            )
            
            successful += 1
            
            # Print first few for debugging
            if successful <= 5:
                lesion_voxels = (mask_resampled > 0).sum()
                print(f"\n  ✓ {case_id}:")
                print(f"    Image: {img_file.name} {img_shape}")
                print(f"    Mask: {mask_file.name} {mask_shape}")
                print(f"    Resampled: {target_shape}, lesion={lesion_voxels} voxels")
        
        except Exception as e:
            print(f"\n❌ Error processing {case_id}: {e}")
            failed += 1
            continue
    
    # Save original dimensions info
    info_file = output / 'original_dimensions.json'
    with open(info_file, 'w') as f:
        json.dump(original_info, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successful:      {successful}")
    print(f"⊘ No mask:         {skipped_no_mask}")
    print(f"❌ Failed:          {failed}")
    print(f"Total images:      {len(image_files)}")
    print(f"Output:            {output}")
    print(f"Original dims:     {info_file}")
    print(f"{'='*70}\n")
    
    return original_info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess NEURALCUP dataset')
    parser.add_argument('--neuralcup-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/NEURALCUP_T1_MASK_PAIRS',
                       help='NEURALCUP directory with images/ and masks/ subdirs')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled',
                       help='Output directory')
    
    args = parser.parse_args()
    
    preprocess_neuralcup(
        neuralcup_dir=args.neuralcup_dir,
        output_dir=args.output_dir
    )

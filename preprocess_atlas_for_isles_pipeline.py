"""
Convert ATLAS v2.0 data to the same format as your ISLES preprocessing
This allows you to use your existing training scripts
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from scipy.ndimage import zoom

def resample_volume(volume, target_shape=(197, 233, 189), order=3):
    """
    Resample volume to target shape
    
    Args:
        volume: Input volume (numpy array)
        target_shape: Target dimensions
        order: Interpolation order (3=cubic for images, 0=nearest for masks)
    
    Returns:
        Resampled volume
    """
    current_shape = volume.shape
    zoom_factors = np.array(target_shape) / np.array(current_shape)
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled


def preprocess_atlas_to_isles_format(
    atlas_dir='/hpc/pahm409/harvard/atlas_full_training_data',
    output_dir='/home/pahm409/preprocessed_atlas_for_isles',
    target_shape=(197, 233, 189),
    train_val_split=0.8,
    seed=42
):
    """
    Preprocess ATLAS data to match ISLES preprocessing format
    
    Creates:
    - Preprocessed .npz files (same format as ISLES2022_resampled)
    - JSON splits file compatible with your training scripts
    """
    
    atlas_dir = Path(atlas_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    atlas_output = output_dir / 'ATLAS_resampled'
    atlas_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing ATLAS data from: {atlas_dir}")
    print(f"Output directory: {atlas_output}")
    print(f"Target shape: {target_shape}")
    
    # Find all T1w images
    t1w_files = sorted(list(atlas_dir.glob("*_T1w.nii.gz")))
    
    print(f"\nFound {len(t1w_files)} ATLAS subjects")
    
    processed_cases = []
    
    for t1w_file in tqdm(t1w_files, desc="Processing ATLAS volumes"):
        # Extract case ID
        filename = t1w_file.name
        # e.g., sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
        case_id = filename.split('_')[0]  # sub-r001s001
        
        # Find corresponding mask
        mask_file = atlas_dir / filename.replace('_T1w.nii.gz', '_label-L_desc-T1lesion_mask.nii.gz')
        
        if not mask_file.exists():
            print(f"\n⚠️  Warning: No mask found for {case_id}, skipping")
            continue
        
        try:
            # Load volumes
            t1w_img = nib.load(t1w_file)
            mask_img = nib.load(mask_file)
            
            t1w_data = t1w_img.get_fdata()
            mask_data = mask_img.get_fdata()
            
            # Verify shapes match
            if t1w_data.shape != mask_data.shape:
                print(f"\n⚠️  Warning: Shape mismatch for {case_id}, skipping")
                continue
            
            # Resample to target shape
            t1w_resampled = resample_volume(t1w_data, target_shape, order=3)
            mask_resampled = resample_volume(mask_data, target_shape, order=0)
            
            # Binarize mask
            mask_resampled = (mask_resampled > 0.5).astype(np.uint8)
            
            # Normalize T1w (same as your ISLES preprocessing)
            # Z-score normalization
            mean = t1w_resampled.mean()
            std = t1w_resampled.std()
            if std > 0:
                t1w_normalized = (t1w_resampled - mean) / std
            else:
                t1w_normalized = t1w_resampled - mean
            
            # Clip to [-5, 5] (remove extreme outliers)
            t1w_normalized = np.clip(t1w_normalized, -5, 5)
            
            # Save as .npz (same format as ISLES2022_resampled)
            output_file = atlas_output / f"{case_id}.npz"
            
            np.savez_compressed(
                output_file,
                image=t1w_normalized.astype(np.float32),
                mask=mask_resampled.astype(np.uint8)
            )
            
            processed_cases.append(case_id)
            
        except Exception as e:
            print(f"\n❌ Error processing {case_id}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(processed_cases)} ATLAS cases")
    
    # Create splits (train/val, no test since we'll use ISLES as test)
    np.random.seed(seed)
    np.random.shuffle(processed_cases)
    
    split_idx = int(len(processed_cases) * train_val_split)
    train_cases = processed_cases[:split_idx]
    val_cases = processed_cases[split_idx:]
    
    # Create splits file (compatible with your training scripts)
    splits = {
        'fold_0': {
            'ATLAS_resampled': {
                'train': train_cases,
                'val': val_cases
            }
        }
    }
    
    splits_file = output_dir / 'atlas_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Created splits file: {splits_file}")
    print(f"  Train: {len(train_cases)} cases")
    print(f"  Val: {len(val_cases)} cases")
    
    # Save case list
    case_info = {
        'total_cases': len(processed_cases),
        'train_cases': len(train_cases),
        'val_cases': len(val_cases),
        'all_cases': processed_cases,
        'train_split': train_cases,
        'val_split': val_cases
    }
    
    info_file = output_dir / 'atlas_case_info.json'
    with open(info_file, 'w') as f:
        json.dump(case_info, f, indent=2)
    
    print(f"✓ Saved case info: {info_file}")
    print("\n" + "="*70)
    print("ATLAS PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Preprocessed data: {atlas_output}")
    print(f"Splits file: {splits_file}")
    print(f"Ready to train!")
    
    return splits_file


if __name__ == '__main__':
    preprocess_atlas_to_isles_format()

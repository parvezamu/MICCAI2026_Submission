"""
preprocess_brats2024_t2flair.py

Preprocess BraTS 2024 T2-FLAIR data for stroke segmentation
- Resample to ISLES resolution (197, 233, 189)
- Extract T2-FLAIR + segmentation masks
- Save as .npz files
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import argparse

def resample_volume(volume, target_shape):
    """Resample volume to target shape"""
    zoom_factors = [
        target_shape[i] / volume.shape[i] for i in range(3)
    ]
    resampled = zoom(volume, zoom_factors, order=1)
    return resampled

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

def process_brats_case(case_dir, output_dir, target_shape=(197, 233, 189)):
    """Process single BraTS case"""
    case_name = case_dir.name
    
    # File paths
    t2f_file = case_dir / f"{case_name}-t2f.nii.gz"
    seg_file = case_dir / f"{case_name}-seg.nii.gz"
    
    if not t2f_file.exists() or not seg_file.exists():
        return False
    
    try:
        # Load T2-FLAIR
        t2f_nii = nib.load(str(t2f_file))
        t2f_data = t2f_nii.get_fdata()
        
        # Load segmentation
        seg_nii = nib.load(str(seg_file))
        seg_data = seg_nii.get_fdata()
        
        # Resample to ISLES resolution
        t2f_resampled = resample_volume(t2f_data, target_shape)
        seg_resampled = resample_volume(seg_data, target_shape)
        
        # Binarize mask (any tumor class = 1)
        mask_binary = (seg_resampled > 0).astype(np.uint8)
        
        # Normalize T2-FLAIR
        t2f_normalized = normalize_volume(t2f_resampled)
        
        # Save as .npz
        output_file = output_dir / f"{case_name}.npz"
        np.savez_compressed(
            output_file,
            image=t2f_normalized.astype(np.float32),
            mask=mask_binary.astype(np.uint8)
        )
        
        return True
        
    except Exception as e:
        print(f"Error processing {case_name}: {e}")
        return False

def create_splits(output_dir, n_folds=5):
    """Create 5-fold splits"""
    import json
    
    all_cases = sorted([f.stem for f in output_dir.glob("*.npz")])
    
    print(f"\nCreating {n_folds}-fold splits...")
    print(f"Total cases: {len(all_cases)}")
    
    np.random.seed(42)
    np.random.shuffle(all_cases)
    
    splits = {}
    fold_size = len(all_cases) // n_folds
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(all_cases)
        
        test_cases = all_cases[test_start:test_end]
        remaining = [c for c in all_cases if c not in test_cases]
        
        val_size = len(remaining) // 5
        val_cases = remaining[:val_size]
        train_cases = remaining[val_size:]
        
        splits[f'fold_{fold}'] = {
            'BraTS2024_T2FLAIR': {
                'train': train_cases,
                'val': val_cases,
                'test': test_cases
            }
        }
    
    splits_file = 'brats2024_t2flair_splits_5fold.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✓ Splits saved: {splits_file}")
    for fold in range(n_folds):
        print(f"  Fold {fold}: Train={len(splits[f'fold_{fold}']['BraTS2024_T2FLAIR']['train'])}, "
              f"Val={len(splits[f'fold_{fold}']['BraTS2024_T2FLAIR']['val'])}, "
              f"Test={len(splits[f'fold_{fold}']['BraTS2024_T2FLAIR']['test'])}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brats-dir', type=str, 
                       default='/home/pahm409/ISLES2022_reg/BraTS_GLI/training_data1_v2',
                       help='BraTS 2024 directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/preprocessed_brats2024_t2flair',
                       help='Output directory')
    args = parser.parse_args()
    
    brats_dir = Path(args.brats_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREPROCESSING BraTS 2024 T2-FLAIR")
    print("="*80)
    print(f"Input: {brats_dir}")
    print(f"Output: {output_dir}")
    print(f"Target shape: (197, 233, 189) [ISLES resolution]")
    print("="*80)
    
    # Get all case directories
    case_dirs = sorted([d for d in brats_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(case_dirs)} cases")
    
    # Process all cases
    success_count = 0
    for case_dir in tqdm(case_dirs, desc='Processing'):
        if process_brats_case(case_dir, output_dir):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(case_dirs)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Create splits
    create_splits(output_dir)
    
    print(f"\n✓ Done! Ready for training.")

if __name__ == '__main__':
    main()

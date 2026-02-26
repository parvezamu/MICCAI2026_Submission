"""
create_5fold_splits.py

Create 5-fold cross-validation splits from existing train/val splits.
Each fold uses the combined train+val data, keeping test set separate.

Author: Parvez
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def create_5fold_splits(input_file='splits_stratified.json', output_file='splits_5fold.json', random_seed=42):
    """
    Create 5-fold CV splits from existing splits file
    
    Args:
        input_file: Path to original splits file
        output_file: Path to save 5-fold splits
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Load current splits
    print(f"Loading splits from {input_file}...")
    with open(input_file, 'r') as f:
        current_splits = json.load(f)
    
    # Statistics
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
    
    print("\nOriginal split sizes:")
    for dataset_name in current_splits.keys():
        n_train = len(current_splits[dataset_name]['train'])
        n_val = len(current_splits[dataset_name]['val'])
        n_test = len(current_splits[dataset_name]['test'])
        print(f"  {dataset_name:15s}: train={n_train:3d}, val={n_val:3d}, test={n_test:3d}")
        
        stats[dataset_name]['train'] = n_train
        stats[dataset_name]['val'] = n_val
        stats[dataset_name]['test'] = n_test
        stats[dataset_name]['total'] = n_train + n_val + n_test
    
    # Create fold splits
    fold_splits = {}
    
    print("\nCreating 5-fold splits...")
    
    for fold_idx in range(5):
        fold_splits[f'fold_{fold_idx}'] = {}
        
        for dataset_name in current_splits.keys():
            # Combine train + val (exclude test)
            all_trainval = (current_splits[dataset_name]['train'] + 
                           current_splits[dataset_name]['val'])
            
            # Shuffle
            all_trainval = np.array(all_trainval)
            np.random.shuffle(all_trainval)
            all_trainval = all_trainval.tolist()
            
            n_total = len(all_trainval)
            
            # Calculate fold sizes (approximately equal)
            fold_size = n_total // 5
            
            # Determine validation set for this fold
            val_start = fold_idx * fold_size
            if fold_idx == 4:  # Last fold gets remainder
                val_end = n_total
            else:
                val_end = val_start + fold_size
            
            # Split into train and val
            val_cases = all_trainval[val_start:val_end]
            train_cases = all_trainval[:val_start] + all_trainval[val_end:]
            
            fold_splits[f'fold_{fold_idx}'][dataset_name] = {
                'train': train_cases,
                'val': val_cases,
                'test': current_splits[dataset_name]['test']  # Keep same test set
            }
    
    # Verify and print fold statistics
    print("\nFold statistics:")
    print("=" * 80)
    
    for fold_idx in range(5):
        print(f"\nFold {fold_idx}:")
        
        fold_stats = defaultdict(lambda: {'train': 0, 'val': 0})
        
        for dataset_name in current_splits.keys():
            n_train = len(fold_splits[f'fold_{fold_idx}'][dataset_name]['train'])
            n_val = len(fold_splits[f'fold_{fold_idx}'][dataset_name]['val'])
            n_test = len(fold_splits[f'fold_{fold_idx}'][dataset_name]['test'])
            
            fold_stats[dataset_name]['train'] = n_train
            fold_stats[dataset_name]['val'] = n_val
            
            print(f"  {dataset_name:15s}: train={n_train:3d}, val={n_val:3d}, test={n_test:3d}")
        
        # Print totals
        total_train = sum(fold_stats[ds]['train'] for ds in current_splits.keys())
        total_val = sum(fold_stats[ds]['val'] for ds in current_splits.keys())
        total_test = sum(len(current_splits[ds]['test']) for ds in current_splits.keys())
        
        print(f"  {'TOTAL':15s}: train={total_train:3d}, val={total_val:3d}, test={total_test:3d}")
    
    # Verify no overlap between folds
    print("\n" + "=" * 80)
    print("Verifying no overlap between train and val within each fold...")
    
    all_valid = True
    for fold_idx in range(5):
        for dataset_name in current_splits.keys():
            train_set = set(fold_splits[f'fold_{fold_idx}'][dataset_name]['train'])
            val_set = set(fold_splits[f'fold_{fold_idx}'][dataset_name]['val'])
            
            overlap = train_set & val_set
            if overlap:
                print(f"  ✗ Fold {fold_idx}, {dataset_name}: {len(overlap)} overlapping cases!")
                all_valid = False
    
    if all_valid:
        print("  ✓ All folds verified - no overlap!")
    
    # Verify all cases are used
    print("\nVerifying all cases are covered...")
    for dataset_name in current_splits.keys():
        original_trainval = set(current_splits[dataset_name]['train'] + 
                               current_splits[dataset_name]['val'])
        
        for fold_idx in range(5):
            fold_trainval = set(fold_splits[f'fold_{fold_idx}'][dataset_name]['train'] + 
                               fold_splits[f'fold_{fold_idx}'][dataset_name]['val'])
            
            if original_trainval != fold_trainval:
                print(f"  ✗ Fold {fold_idx}, {dataset_name}: Case count mismatch!")
                all_valid = False
    
    if all_valid:
        print("  ✓ All cases covered in all folds!")
    
    # Save
    print(f"\nSaving 5-fold splits to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(fold_splits, f, indent=2)
    
    print(f"✓ Successfully created {output_file}")
    print("\nUsage:")
    print("  python train_patch_with_reconstruction.py \\")
    print("    --pretrained-checkpoint <path> \\")
    print("    --fold 0 \\")
    print("    --epochs 100")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    create_5fold_splits(
        input_file='splits_stratified.json',
        output_file='splits_5fold.json',
        random_seed=42
    )

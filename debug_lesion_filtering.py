"""
debug_lesion_filtering.py

Check if lesion filtering is actually working in the dataset

Author: Parvez
Date: January 2026
"""



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('.')

import torch
import numpy as np
from tqdm import tqdm

from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

# Test the wrapper
from train_mae_simple import MAEDatasetWrapper


def check_filtering():
    """Check how many patches are actually being filtered"""
    
    print("\n" + "="*70)
    print("DEBUGGING LESION FILTERING")
    print("="*70)
    
    # Create base dataset (what PatchDatasetWithCenters returns)
    print("\n1. Testing BASE dataset (PatchDatasetWithCenters)...")
    base_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=0,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=False,
        lesion_focus_ratio=0.0
    )
    
    print(f"Base dataset size: {len(base_dataset)}")
    
    # Check first 500 patches
    lesion_count = 0
    healthy_count = 0
    
    for i in tqdm(range(min(500, len(base_dataset))), desc='Checking base patches'):
        sample = base_dataset[i]
        mask = sample['mask']
        
        if isinstance(mask, torch.Tensor):
            lesion_voxels = mask.sum().item()
        else:
            lesion_voxels = np.sum(mask)
        
        if lesion_voxels > 0:
            lesion_count += 1
        else:
            healthy_count += 1
    
    print(f"\nBase dataset results (first 500 patches):")
    print(f"  Patches WITH lesions: {lesion_count} ({lesion_count/500*100:.1f}%)")
    print(f"  Patches WITHOUT lesions: {healthy_count} ({healthy_count/500*100:.1f}%)")
    
    # Now test the wrapper
    print("\n" + "="*70)
    print("2. Testing WRAPPER dataset (MAEDatasetWrapper)...")
    print("="*70)
    
    wrapper_dataset = MAEDatasetWrapper(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        splits_file='splits_5fold.json',
        fold=0,
        split='train',
        patch_size=(96, 96, 96),
        patches_per_volume=10
    )
    
    print(f"Wrapper dataset size: {len(wrapper_dataset)}")
    
    # Check what the wrapper actually returns
    none_count = 0
    tensor_count = 0
    
    for i in tqdm(range(min(500, len(wrapper_dataset))), desc='Checking wrapper outputs'):
        output = wrapper_dataset[i]
        
        if output is None:
            none_count += 1
        elif isinstance(output, torch.Tensor):
            tensor_count += 1
        else:
            print(f"  ⚠️ Unexpected output type at index {i}: {type(output)}")
    
    print(f"\nWrapper dataset results (first 500 patches):")
    print(f"  Returned None (filtered): {none_count} ({none_count/500*100:.1f}%)")
    print(f"  Returned Tensor (valid): {tensor_count} ({tensor_count/500*100:.1f}%)")
    
    # Cross-check: manually verify a few None returns
    print("\n" + "="*70)
    print("3. VERIFICATION: Checking if None patches actually had lesions...")
    print("="*70)
    
    verified_correct = 0
    verified_incorrect = 0
    
    for i in range(min(100, len(wrapper_dataset))):
        output = wrapper_dataset[i]
        
        # Get the original sample
        sample = wrapper_dataset.base_dataset[i]
        mask = sample['mask']
        
        if isinstance(mask, torch.Tensor):
            lesion_voxels = mask.sum().item()
        else:
            lesion_voxels = np.sum(mask)
        
        has_lesion = lesion_voxels > 0
        returned_none = output is None
        
        if has_lesion and returned_none:
            verified_correct += 1
        elif not has_lesion and not returned_none:
            verified_correct += 1
        else:
            verified_incorrect += 1
            print(f"  ⚠️ Mismatch at index {i}: has_lesion={has_lesion}, returned_none={returned_none}")
    
    print(f"\nVerification results (first 100 patches):")
    print(f"  Correctly filtered: {verified_correct}")
    print(f"  Incorrectly filtered: {verified_incorrect}")
    
    # Final diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if none_count > 0:
        print(f"✅ Filtering IS working: {none_count} patches returned None")
        expected_filtered = lesion_count
        if abs(none_count - expected_filtered) < 50:
            print(f"✅ Filter rate matches expectation (~{lesion_count/500*100:.1f}%)")
        else:
            print(f"⚠️ Filter rate ({none_count/500*100:.1f}%) doesn't match base lesion rate ({lesion_count/500*100:.1f}%)")
    else:
        print(f"❌ Filtering NOT working: No patches returned None!")
        print(f"   Expected ~{lesion_count} patches to be filtered")
        print(f"   → Check __getitem__ implementation!")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    check_filtering()

"""
test_validation_dataset.py
Quick test to see if validation dataset works
"""

import torch
import sys
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

print("="*70)
print("TESTING VALIDATION DATASET")
print("="*70)

# Load validation dataset
isles_val = PatchDatasetWithCenters(
    preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
    datasets=['ISLES2022_resampled'],
    split='val',
    splits_file='isles_splits_5fold_resampled.json',
    fold=0,
    patch_size=(96, 96, 96),
    patches_per_volume=100,
    augment=False,
    lesion_focus_ratio=0.0,
    compute_lesion_bins=False
)

print(f"\n✓ Dataset loaded")
print(f"  Total volumes: {len(isles_val.volumes)}")
print(f"  Total patches: {len(isles_val)}")

# Test getting one sample
print(f"\n🔍 Testing __getitem__...")
try:
    sample = isles_val[0]
    print(f"✓ Sample 0:")
    print(f"  image shape: {sample['image'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")
    print(f"  center: {sample['center']}")
    print(f"  vol_idx: {sample['vol_idx']}")
    print(f"  case_id: {sample['case_id']}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test DataLoader
print(f"\n🔍 Testing DataLoader...")
from torch.utils.data import DataLoader

try:
    loader = DataLoader(isles_val, batch_size=4, shuffle=False, num_workers=0)
    print(f"✓ DataLoader created: {len(loader)} batches")
    
    # Get first batch
    for batch_idx, batch in enumerate(loader):
        print(f"\n✓ Batch {batch_idx}:")
        print(f"  images: {batch['image'].shape}")
        print(f"  masks: {batch['mask'].shape}")
        print(f"  centers: {batch['center'].shape}")
        print(f"  vol_indices: {batch['vol_idx']}")
        
        if batch_idx >= 2:  # Only test first 3 batches
            break
    
    print(f"\n✓ DataLoader works!")
    
except Exception as e:
    print(f"❌ DataLoader error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

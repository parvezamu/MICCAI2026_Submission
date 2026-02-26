"""
test_vol_idx.py - Check what type vol_idx is
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

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

sample = isles_val[0]
print(f"vol_idx type: {type(sample['vol_idx'])}")
print(f"vol_idx value: {sample['vol_idx']}")

if isinstance(sample['vol_idx'], int):
    print("✓ It's an int - good!")
else:
    print("❌ It's NOT an int - THIS IS THE PROBLEM!")
    print(f"It's: {type(sample['vol_idx'])}")

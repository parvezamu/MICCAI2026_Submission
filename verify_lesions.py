# Quick test

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

dataset = PatchDatasetWithCenters(
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

# Count patches with lesions
lesion_patches = 0
total = 0

for i in range(min(1000, len(dataset))):
    sample = dataset[i]
    if sample['mask'].sum() > 0:
        lesion_patches += 1
    total += 1

print(f"Lesion patches: {lesion_patches}/{total} ({lesion_patches/total*100:.1f}%)")

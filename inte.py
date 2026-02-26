"""
check_intensity_distribution.py

Compare ATLAS vs ISLES preprocessing
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load ATLAS sample
atlas_file = "/home/pahm409/preprocessed_stroke_foundation/ATLAS/sub-r001s001_ses-1.npz"
atlas_data = np.load(atlas_file)
atlas_img = atlas_data['image']
atlas_mask = atlas_data['lesion_mask']

# Load ISLES sample
isles_file = "/home/pahm409/isles2022_preprocessed/sub-strokecase0001.npz"
isles_data = np.load(isles_file)
isles_img = isles_data['image']
isles_mask = isles_data['lesion_mask']

print("ATLAS (T1):")
print(f"  Image range: [{atlas_img.min():.3f}, {atlas_img.max():.3f}]")
print(f"  Image mean: {atlas_img.mean():.3f}")
print(f"  Lesion mean intensity: {atlas_img[atlas_mask > 0].mean():.3f}")
print(f"  Background mean intensity: {atlas_img[atlas_mask == 0].mean():.3f}")

print("\nISLES (FLAIR):")
print(f"  Image range: [{isles_img.min():.3f}, {isles_img.max():.3f}]")
print(f"  Image mean: {isles_img.mean():.3f}")
print(f"  Lesion mean intensity: {isles_img[isles_mask > 0].mean():.3f}")
print(f"  Background mean intensity: {isles_img[isles_mask == 0].mean():.3f}")

# Check if lesions are INVERTED
print("\nLesion contrast:")
print(f"  ATLAS: Lesion - Background = {atlas_img[atlas_mask > 0].mean() - atlas_img[atlas_mask == 0].mean():.3f}")
print(f"  ISLES: Lesion - Background = {isles_img[isles_mask > 0].mean() - isles_img[isles_mask == 0].mean():.3f}")

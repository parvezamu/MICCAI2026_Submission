#!/usr/bin/env python3
"""
check_preprocessing_contrast.py

Sanity-check lesion vs background intensity statistics
for stroke preprocessing outputs.

Datasets:
- ATLAS (T1)
- UOA Private (T1)
- ISLES 2022 (FLAIR)

Author: Parvez
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import glob
import os


def analyze_npz(npz_path, label):
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
        return

    data = np.load(npz_path)

    if "image" not in data or "lesion_mask" not in data:
        print(f"[ERROR] Missing keys in {npz_path}")
        return

    image = data["image"]
    mask = data["lesion_mask"]

    lesion = image[mask > 0]
    bg = image[mask == 0]

    print(f"\n{label}")
    print("-" * len(label))

    if lesion.size == 0:
        print("  ⚠️ No lesion voxels found!")
        return

    print(f"  Lesion mean     : {lesion.mean():.4f}")
    print(f"  Background mean : {bg.mean():.4f}")
    print(f"  Contrast (L-B)  : {(lesion.mean() - bg.mean()):.4f}")
    print(f"  Lesion std      : {lesion.std():.4f}")
    print(f"  # Lesion voxels : {lesion.size}")
    print(f"  # BG voxels     : {bg.size}")


def main():
    # 1. ATLAS
    atlas_path = (
        "/home/pahm409/preprocessed_stroke_foundation/ATLAS/"
        "sub-r001s001_ses-1.npz"
    )
    analyze_npz(atlas_path, "ATLAS T1")

    # 2. UOA (take first file)
    uoa_files = glob.glob(
        "/home/pahm409/preprocessed_stroke_foundation/UOA_Private/*.npz"
    )

    if len(uoa_files) == 0:
        print("\n[ERROR] No UOA files found")
    else:
        print(f"\nUsing UOA file: {uoa_files[0]}")
        analyze_npz(uoa_files[0], "UOA T1")

    # 3. ISLES FIXED
    isles_path = (
        "/home/pahm409/isles2022_preprocessed_FIXED/"
        "sub-strokecase0014.npz"
    )
    analyze_npz(isles_path, "ISLES FLAIR (case 0014 – DSC ≈ 0.1656)")


if __name__ == "__main__":
    main()


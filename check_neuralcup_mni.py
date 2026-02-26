"""
check_neuralcup_mni.py

Check if NEURALCUP images are in MNI space
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import nibabel as nib
import numpy as np
from pathlib import Path


def check_mni_space(nifti_file):
    """
    Check if a NIfTI file is in MNI space
    
    MNI152 template characteristics:
    - Shape: ~(197, 233, 189) or (182, 218, 182) or (91, 109, 91)
    - Spacing: 1x1x1 mm or 2x2x2 mm
    - Origin roughly centered at (90, -126, -72)
    - Affine matches MNI template
    """
    
    img = nib.load(nifti_file)
    shape = img.shape
    affine = img.affine
    spacing = img.header.get_zooms()[:3]
    
    # Get origin (translation component)
    origin = affine[:3, 3]
    
    print(f"\n{'='*70}")
    print(f"File: {nifti_file.name}")
    print(f"{'='*70}")
    print(f"Shape: {shape}")
    print(f"Spacing: {spacing} mm")
    print(f"Origin: [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]")
    
    print(f"\nAffine matrix:")
    print(affine)
    
    # Check MNI characteristics
    is_mni = []
    
    # Check 1: Standard MNI shapes
    mni_shapes = [(197, 233, 189), (182, 218, 182), (91, 109, 91), (193, 229, 193)]
    shape_match = shape in mni_shapes
    is_mni.append(("Shape matches MNI", shape_match))
    
    # Check 2: Isotropic spacing (1mm or 2mm)
    is_isotropic = np.allclose(spacing, spacing[0], atol=0.1)
    is_standard_spacing = np.isclose(spacing[0], 1.0, atol=0.1) or np.isclose(spacing[0], 2.0, atol=0.1)
    is_mni.append(("Isotropic spacing", is_isotropic and is_standard_spacing))
    
    # Check 3: Affine diagonal pattern (typical of registered images)
    # MNI affine typically has form:
    # [[-1,  0,  0,  90],
    #  [ 0,  1,  0, -126],
    #  [ 0,  0,  1, -72],
    #  [ 0,  0,  0,   1]]
    diagonal_dominant = (
        abs(affine[0, 0]) > 0.5 and abs(affine[1, 1]) > 0.5 and abs(affine[2, 2]) > 0.5 and
        abs(affine[0, 1]) < 0.1 and abs(affine[0, 2]) < 0.1 and
        abs(affine[1, 0]) < 0.1 and abs(affine[1, 2]) < 0.1 and
        abs(affine[2, 0]) < 0.1 and abs(affine[2, 1]) < 0.1
    )
    is_mni.append(("Diagonal affine (aligned)", diagonal_dominant))
    
    # Check 4: Origin near MNI center
    mni_origin = np.array([90, -126, -72])
    origin_close = np.allclose(origin, mni_origin, atol=30)
    is_mni.append(("Origin near MNI center", origin_close))
    
    # Print assessment
    print(f"\nMNI Space Assessment:")
    for check, result in is_mni:
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
    
    # Overall verdict
    mni_score = sum([r for _, r in is_mni])
    print(f"\nMNI Score: {mni_score}/4")
    
    if mni_score >= 3:
        print("🟢 Likely in MNI space")
        return True
    elif mni_score == 2:
        print("🟡 Possibly in MNI space (unclear)")
        return None
    else:
        print("🔴 Likely NOT in MNI space (native space)")
        return False


def compare_with_atlas(atlas_dir, neuralcup_dir):
    """Compare ATLAS (known MNI) with NEURALCUP"""
    
    atlas_dir = Path(atlas_dir)
    neuralcup_dir = Path(neuralcup_dir)
    
    # Get sample files
    atlas_files = list(atlas_dir.glob("*.nii.gz"))
    neuralcup_files = list(neuralcup_dir.glob("*.nii*"))
    
    if not atlas_files:
        print(f"❌ No ATLAS files found in {atlas_dir}")
        return
    
    if not neuralcup_files:
        print(f"❌ No NEURALCUP files found in {neuralcup_dir}")
        return
    
    print("\n" + "="*70)
    print("ATLAS (KNOWN MNI SPACE)")
    print("="*70)
    check_mni_space(atlas_files[0])
    
    print("\n" + "="*70)
    print("NEURALCUP (UNKNOWN)")
    print("="*70)
    check_mni_space(neuralcup_files[0])
    
    # Compare a few more samples
    if len(neuralcup_files) > 1:
        print("\n" + "="*70)
        print("NEURALCUP - Additional Sample")
        print("="*70)
        check_mni_space(neuralcup_files[1])


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas-dir', type=str,
                       default='/home/pahm409/ATLAS_or_wherever_your_original_atlas_is',
                       help='Directory with original ATLAS MNI images')
    parser.add_argument('--neuralcup-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/NEURALCUP_T1_MASK_PAIRS/images',
                       help='Directory with NEURALCUP images')
    
    args = parser.parse_args()
    
    compare_with_atlas(args.atlas_dir, args.neuralcup_dir)

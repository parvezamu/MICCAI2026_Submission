"""
verify_datasets.py

Quick verification of dataset structure and file counts

Author: Parvez
"""

from pathlib import Path
from glob import glob

def verify_atlas():
    """Verify ATLAS dataset"""
    data_dir = Path("/hpc/pahm409/harvard/atlas_full_training_data")
    
    print("\n" + "="*60)
    print("ATLAS 2.0 Verification")
    print("="*60)
    
    if not data_dir.exists():
        print("❌ Directory not found!")
        return
    
    images = sorted(glob(str(data_dir / "*_T1w.nii.gz")))
    masks = sorted(glob(str(data_dir / "*_mask.nii.gz")))
    
    print(f"✓ Directory exists: {data_dir}")
    print(f"✓ Images found: {len(images)}")
    print(f"✓ Masks found: {len(masks)}")
    print(f"\nSample files:")
    if images:
        print(f"  Image: {Path(images[0]).name}")
    if masks:
        print(f"  Mask:  {Path(masks[0]).name}")


def verify_uoa():
    """Verify UOA dataset"""
    data_dir = Path("/hpc/pahm409/harvard/UOA/FNIRT/train")
    
    print("\n" + "="*60)
    print("UOA Private Verification")
    print("="*60)
    
    if not data_dir.exists():
        print("❌ Directory not found!")
        return
    
    images = sorted(glob(str(data_dir / "*_T1_FNIRT_MNI.nii.gz")))
    masks = sorted(glob(str(data_dir / "*_LESION_FNIRT_MNI_bin.nii.gz")))
    
    print(f"✓ Directory exists: {data_dir}")
    print(f"✓ Images found: {len(images)}")
    print(f"✓ Masks found: {len(masks)}")
    print(f"\nSample files:")
    if images:
        print(f"  Image: {Path(images[0]).name}")
    if masks:
        print(f"  Mask:  {Path(masks[0]).name}")


def verify_isles2018():
    """Verify ISLES 2018"""
    data_dir = Path("/hpc/pahm409/ISLES/ISLES2018_Training/TRAINING")
    
    print("\n" + "="*60)
    print("ISLES 2018 Verification")
    print("="*60)
    
    if not data_dir.exists():
        print("❌ Directory not found!")
        return
    
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
    
    print(f"✓ Directory exists: {data_dir}")
    print(f"✓ Cases found: {len(cases)}")
    
    if cases:
        case_dir = cases[0]
        print(f"\nSample case: {case_dir.name}")
        
        # List available modalities
        modalities = [d.name for d in case_dir.iterdir() if d.is_dir()]
        print(f"  Modalities: {', '.join(modalities)}")


def verify_isles2022():
    """Verify ISLES 2022"""
    data_dir = Path("/hpc/pahm409/ISLES/ISLES2022")
    
    print("\n" + "="*60)
    print("ISLES 2022 Verification")
    print("="*60)
    
    if not data_dir.exists():
        print("❌ Directory not found!")
        return
    
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print(f"✓ Directory exists: {data_dir}")
    print(f"✓ Cases found: {len(cases)}")
    
    if cases:
        case_dir = cases[0]
        print(f"\nSample case: {case_dir.name}")
        
        # List available files
        files = [f.name for f in case_dir.glob("*.nii.gz")]
        print(f"  Files: {', '.join(files)}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STROKE FOUNDATION MODEL - DATASET VERIFICATION")
    print("="*70)
    
    verify_atlas()
    verify_uoa()
    verify_isles2018()
    verify_isles2022()
    
    print("\n" + "="*70)
    print("Verification complete!")
    print("="*70 + "\n")

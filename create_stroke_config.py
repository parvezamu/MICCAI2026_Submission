"""
create_stroke_config.py

Create configuration for your specific stroke datasets on HPC
Updated to handle nested ISLES2018 structure and exclude 4DPWI

Author: Parvez
Date: January 2026
"""

import json
from pathlib import Path
import argparse
from glob import glob


def create_atlas_config() -> dict:
    """ATLAS 2.0 - Already in MNI space"""
    return {
        "name": "ATLAS",
        "data_dir": "/hpc/pahm409/harvard/atlas_full_training_data",
        "image_pattern": "*_space-MNI152NLin2009aSym_T1w.nii.gz",
        "mask_pattern": "*_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz",
        "description": "ATLAS 2.0 - Chronic stroke, already in MNI space, small lesions",
        "modality": "T1w",
        "notes": "Already preprocessed and in MNI152 space - skip registration!"
    }


def create_uoa_config() -> dict:
    """UOA Private - Already in MNI via FNIRT"""
    return {
        "name": "UOA_Private",
        "data_dir": "/hpc/pahm409/harvard/UOA/FNIRT/train",
        "image_pattern": "*_T1_FNIRT_MNI.nii.gz",
        "mask_pattern": "*_LESION_FNIRT_MNI_bin.nii.gz",
        "description": "UOA private - Already in MNI via FNIRT, large lesions",
        "modality": "T1w",
        "notes": "Already preprocessed and in MNI space - skip registration!"
    }


def create_isles2018_config() -> dict:
    """ISLES 2018 - Multi-modal CT perfusion (using MTT, NOT 4DPWI)"""
    return {
        "name": "ISLES2018",
        "data_dir": "/hpc/pahm409/ISLES/ISLES2018_Training/TRAINING",
        "image_pattern": "case_*/SMIR.*.CT_MTT.*/SMIR.*.CT_MTT.*.nii",  # Nested in subdirectory
        "mask_pattern": "case_*/SMIR.*.OT.*/SMIR.*.OT.*.nii",  # Nested in subdirectory
        "description": "ISLES 2018 - Acute stroke, CT perfusion (MTT modality)",
        "modality": "MTT",
        "notes": "Using MTT (Mean Transit Time), EXCLUDING 4DPWI. Other available: CT, CBF, CBV, Tmax"
    }


def create_isles2022_config() -> dict:
    """ISLES 2022 - DWI/ADC"""
    return {
        "name": "ISLES2022", 
        "data_dir": "/hpc/pahm409/ISLES/ISLES2022",
        "image_pattern": "sub-*/sub-*_ses-0001_dwi.nii.gz",  # Using DWI
        "mask_pattern": "sub-*/sub-*_ses-0001_msk.nii.gz",
        "description": "ISLES 2022 - Acute stroke, DWI sequence",
        "modality": "DWI",
        "notes": "Multi-modal available: DWI, ADC, FLAIR"
    }


def verify_dataset_paths(config: dict) -> dict:
    """Verify that dataset paths exist and count files"""
    data_dir = Path(config['data_dir'])
    
    if not data_dir.exists():
        print(f"  ⚠️  Directory not found: {data_dir}")
        return None
    
    # Find images and masks
    image_files = sorted(glob(str(data_dir / config['image_pattern']), recursive=True))
    mask_files = sorted(glob(str(data_dir / config['mask_pattern']), recursive=True))
    
    if not image_files:
        print(f"  ⚠️  No images found with pattern: {config['image_pattern']}")
        print(f"  Searching in: {data_dir}")
        
        # Try to help debug by showing what's actually there
        if 'ISLES2018' in str(data_dir):
            print("\n  Let me check what's in case_1:")
            case1_dir = data_dir / 'case_1'
            if case1_dir.exists():
                subdirs = [d.name for d in case1_dir.iterdir() if d.is_dir()]
                print(f"  Subdirectories: {subdirs}")
                
                # Check for MTT
                mtt_dirs = [d for d in subdirs if 'MTT' in d]
                if mtt_dirs:
                    print(f"  Found MTT directory: {mtt_dirs[0]}")
                    mtt_files = list((case1_dir / mtt_dirs[0]).glob('*.nii*'))
                    print(f"  Files in MTT dir: {[f.name for f in mtt_files]}")
        
        return None
    
    config['n_images'] = len(image_files)
    config['n_masks'] = len(mask_files)
    
    print(f"  ✓ Found {len(image_files)} images")
    print(f"  ✓ Found {len(mask_files)} masks")
    
    # Sample paths for verification
    config['sample_image'] = str(image_files[0]) if image_files else None
    config['sample_mask'] = str(mask_files[0]) if mask_files else None
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Create configuration for Parvez stroke datasets with GPU support'
    )
    parser.add_argument('--output', type=str, default='stroke_datasets_config.json',
                       help='Output configuration file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='Directory for preprocessed outputs')
    parser.add_argument('--datasets', nargs='+',
                       choices=['ATLAS', 'UOA', 'ISLES2018', 'ISLES2022', 'all'],
                       default=['all'],
                       help='Which datasets to include')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset paths and count files')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Creating Stroke Foundation Model Dataset Configuration")
    print("="*70 + "\n")
    
    # Dataset creators
    dataset_creators = {
        'ATLAS': create_atlas_config,
        'UOA': create_uoa_config,
        'ISLES2018': create_isles2018_config,
        'ISLES2022': create_isles2022_config
    }
    
    # Determine which datasets to include
    if 'all' in args.datasets:
        selected_datasets = list(dataset_creators.keys())
    else:
        selected_datasets = args.datasets
    
    # Create base configuration
    config = {
        "datasets": [],
        "preprocessing": {
            "target_spacing": [1.0, 1.0, 1.0],
            "normalization_method": "zscore",
            "clip_percentile": 99.5,
            "use_skull_strip": False,
            "skip_registration": True,
            "use_gpu": not args.no_gpu,
            "save_intermediate": False
        },
        "output_base_dir": args.output_dir,
        "num_workers": 8,
        "notes": [
            "ATLAS and UOA are already in MNI space - registration skipped",
            "ISLES2018 uses MTT (Mean Transit Time) - 4DPWI EXCLUDED",
            "ISLES2022 uses DWI - different from structural MRI",
            "GPU acceleration enabled for faster preprocessing",
            "For T1-only foundation model: use ATLAS and UOA",
            "For multi-modal model: include all datasets"
        ]
    }
    
    # Add selected datasets
    for dataset_name in selected_datasets:
        print(f"Processing {dataset_name}...")
        
        if dataset_name in dataset_creators:
            dataset_config = dataset_creators[dataset_name]()
            
            # Verify paths if requested
            if args.verify:
                dataset_config = verify_dataset_paths(dataset_config)
                if dataset_config is None:
                    continue
            
            config["datasets"].append(dataset_config)
        
        print()
    
    # Save configuration
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*70)
    print(f"✓ Configuration saved to: {output_path}")
    print("="*70 + "\n")
    
    print("Included datasets:")
    for ds in config["datasets"]:
        print(f"  • {ds['name']} ({ds['modality']}): {ds['description']}")
        if args.verify and 'n_images' in ds:
            print(f"    - Images: {ds['n_images']}, Masks: {ds['n_masks']}")
    
    print(f"\nPreprocessing configuration:")
    print(f"  • GPU acceleration: {'Enabled' if config['preprocessing']['use_gpu'] else 'Disabled'}")
    print(f"  • Target spacing: {config['preprocessing']['target_spacing']} mm")
    print(f"  • Normalization: {config['preprocessing']['normalization_method']}")
    print(f"  • 4DPWI: EXCLUDED from ISLES2018")
    
    print(f"\nPreprocessed data will be saved to:")
    print(f"  {config['output_base_dir']}")
    
    print(f"\nNext steps:")
    print(f"  1. Review configuration: cat {output_path}")
    print(f"  2. Run preprocessing:")
    print(f"     python preprocess_stroke_foundation.py --config {output_path}")
    print(f"  3. Monitor GPU usage:")
    print(f"     watch -n 1 nvidia-smi")
    print()


if __name__ == '__main__':
    main()


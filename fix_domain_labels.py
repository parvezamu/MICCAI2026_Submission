"""
Fix missing domain labels in existing reconstruction results


"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import json
from pathlib import Path
import sys

def fix_domain_labels(recon_dir, dataset_mapping):
    """
    dataset_mapping: dict like {'U877': 'UOA', 'sub-r001s001': 'ATLAS'}
    """
    recon_dir = Path(recon_dir)
    
    fixed_count = 0
    for case_dir in recon_dir.iterdir():
        if not case_dir.is_dir():
            continue
        
        metadata_file = case_dir / 'metadata.json'
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        case_id = metadata['case_id']
        
        # Infer dataset from case_id pattern
        if case_id.startswith('sub-'):
            dataset = 'ATLAS'
        else:
            dataset = 'UOA'
        
        metadata['domain'] = dataset
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        fixed_count += 1
    
    print(f"Fixed {fixed_count} metadata files in {recon_dir}")

# Fix all reconstruction directories
base_dir = Path('/home/pahm409/segmentation_balanced_test')

for exp_pattern in ['SimCLR_Pretrained_mkdc_ds', 'SimCLR_Pretrained_mkdc_ds_balanced']:
    exp_dirs = list(base_dir.glob(f'{exp_pattern}/fold_*/run_*/exp_*'))
    
    for exp_dir in exp_dirs:
        print(f"\nProcessing: {exp_dir}")
        
        for recon_dir in exp_dir.glob('reconstructions_epoch_*'):
            print(f"  {recon_dir.name}")
            fix_domain_labels(recon_dir, {})

print("\n✓ All domain labels fixed!")

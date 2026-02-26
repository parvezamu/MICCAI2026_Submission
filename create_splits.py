"""
create_splits.py

Create stratified train/val/test splits for all preprocessed datasets

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import random


def create_stratified_splits(
    preprocessed_base_dir: str,
    output_file: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create stratified train/val/test splits based on lesion size
    
    Stratification ensures balanced distribution of:
    - Small lesions (<10ml)
    - Medium lesions (10-50ml)
    - Large lesions (>50ml)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    base_dir = Path(preprocessed_base_dir)
    splits = {}
    
    # Find all dataset directories
    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    print("\n" + "="*70)
    print("Creating Stratified Train/Val/Test Splits")
    print("="*70 + "\n")
    
    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        
        # Load dataset summary to get lesion volumes
        summary_file = dataset_dir / f'{dataset_name}_summary.json'
        
        if not summary_file.exists():
            print(f"⚠️  Summary file not found for {dataset_name}, skipping...")
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"Processing {dataset_name}...")
        
        # Extract case IDs and lesion volumes
        cases_by_size = {
            'small': [],   # <10ml
            'medium': [],  # 10-50ml
            'large': []    # >50ml
        }
        
        for case_meta in summary['case_metadata']:
            case_id = case_meta['case_id']
            lesion_vol = case_meta['lesion_volume_ml']
            
            if lesion_vol < 10:
                cases_by_size['small'].append(case_id)
            elif lesion_vol < 50:
                cases_by_size['medium'].append(case_id)
            else:
                cases_by_size['large'].append(case_id)
        
        # Print size distribution
        print(f"  Size distribution:")
        print(f"    Small (<10ml):    {len(cases_by_size['small'])} cases")
        print(f"    Medium (10-50ml): {len(cases_by_size['medium'])} cases")
        print(f"    Large (>50ml):    {len(cases_by_size['large'])} cases")
        
        # Create stratified splits for each size category
        dataset_splits = {'train': [], 'val': [], 'test': []}
        
        for size_category, case_ids in cases_by_size.items():
            if not case_ids:
                continue
            
            # Shuffle
            random.shuffle(case_ids)
            
            # Calculate split sizes
            n_total = len(case_ids)
            n_test = max(1, int(n_total * test_ratio))
            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_test - n_val
            
            # Split
            dataset_splits['train'].extend(case_ids[:n_train])
            dataset_splits['val'].extend(case_ids[n_train:n_train+n_val])
            dataset_splits['test'].extend(case_ids[n_train+n_val:])
        
        # Shuffle again to mix size categories
        for split in ['train', 'val', 'test']:
            random.shuffle(dataset_splits[split])
        
        splits[dataset_name] = dataset_splits
        
        print(f"  Splits created:")
        print(f"    Train: {len(dataset_splits['train'])} cases ({len(dataset_splits['train'])/len(summary['case_metadata'])*100:.1f}%)")
        print(f"    Val:   {len(dataset_splits['val'])} cases ({len(dataset_splits['val'])/len(summary['case_metadata'])*100:.1f}%)")
        print(f"    Test:  {len(dataset_splits['test'])} cases ({len(dataset_splits['test'])/len(summary['case_metadata'])*100:.1f}%)")
        print()
    
    # Save splits
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print("="*70)
    print(f"✓ Splits saved to: {output_file}")
    print("="*70 + "\n")
    
    # Print overall statistics
    total_train = sum(len(s['train']) for s in splits.values())
    total_val = sum(len(s['val']) for s in splits.values())
    total_test = sum(len(s['test']) for s in splits.values())
    total = total_train + total_val + total_test
    
    print("Overall statistics:")
    print(f"  Total cases: {total}")
    print(f"  Train: {total_train} ({total_train/total*100:.1f}%)")
    print(f"  Val:   {total_val} ({total_val/total*100:.1f}%)")
    print(f"  Test:  {total_test} ({total_test/total*100:.1f}%)")
    
    return splits


if __name__ == '__main__':
    create_stratified_splits(
        preprocessed_base_dir='/home/pahm409/preprocessed_stroke_foundation',
        output_file='splits_stratified.json',
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

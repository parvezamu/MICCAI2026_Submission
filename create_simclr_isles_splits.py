"""
Create ISLES splits compatible with SimCLR training
Uses the same fold structure but reformats for SimCLR dataset loader
"""

import json
import numpy as np
from pathlib import Path

def create_simclr_isles_splits(
    isles_preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
    output_file='simclr_isles_splits.json'
):
    """Create SimCLR-compatible splits for ISLES"""
    
    isles_dir = Path(isles_preprocessed_dir) / 'ISLES2022_resampled'
    all_cases = sorted([f.stem for f in isles_dir.glob("*.npz")])
    
    print(f"Creating SimCLR ISLES splits...")
    print(f"  Found {len(all_cases)} cases in {isles_dir}")
    
    if len(all_cases) == 0:
        print(f"  ❌ ERROR: No .npz files found!")
        return None
    
    # Use 80/20 split for SimCLR (no test set needed)
    np.random.seed(42)
    np.random.shuffle(all_cases)
    
    n_cases = len(all_cases)
    n_val = int(n_cases * 0.2)
    
    val_cases = all_cases[:n_val]
    train_cases = all_cases[n_val:]
    
    # SimCLR expects this format:
    splits = {
        'ISLES2022_resampled': {
            'train': train_cases,
            'val': val_cases
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ SimCLR splits created: {output_file}")
    print(f"  Train: {len(train_cases)} cases")
    print(f"  Val: {len(val_cases)} cases")
    print(f"  Total: {len(all_cases)} cases\n")
    
    return output_file

if __name__ == '__main__':
    create_simclr_isles_splits()

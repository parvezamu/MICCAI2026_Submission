"""
create_isles_dual_splits.py
Create 5-fold splits for dual-modality ISLES data
"""
import json
import numpy as np
from pathlib import Path

ISLES_DIR = "/home/pahm409/preprocessed_isles_dual_modality"
OUTPUT_FILE = "isles_dual_splits_5fold.json"

print("="*80)
print("CREATING 5-FOLD SPLITS FOR DUAL-MODALITY ISLES")
print("="*80)

isles_dir = Path(ISLES_DIR)
if not isles_dir.exists():
    print(f"\n❌ ERROR: Directory not found: {ISLES_DIR}")
    print(f"\nRun: python preprocess_isles_dual_modality.py first")
    exit(1)

all_cases = sorted([f.stem for f in isles_dir.glob("*.npz")])

print(f"\n✓ Found {len(all_cases)} cases")
print(f"  First 5: {all_cases[:5]}")
print(f"  Last 5: {all_cases[-5:]}")

# Create splits
np.random.seed(42)
np.random.shuffle(all_cases)

n_cases = len(all_cases)
fold_size = n_cases // 5

splits = {}

for fold in range(5):
    test_start = fold * fold_size
    test_end = test_start + fold_size if fold < 4 else n_cases
    test_cases = all_cases[test_start:test_end]
    
    remaining = [c for c in all_cases if c not in test_cases]
    val_size = len(remaining) // 5
    val_cases = remaining[:val_size]
    train_cases = remaining[val_size:]
    
    splits[f'fold_{fold}'] = {
        'ISLES2022_dual': {
            'train': train_cases,
            'val': val_cases,
            'test': test_cases
        }
    }
    
    print(f"\nFold {fold}:")
    print(f"  Train: {len(train_cases)}")
    print(f"  Val: {len(val_cases)}")
    print(f"  Test: {len(test_cases)}")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\n✓ Splits saved: {OUTPUT_FILE}")
print(f"\n{'='*80}")
print("READY FOR JOINT TRAINING!")
print(f"{'='*80}\n")

"""
create_isles_splits_resampled.py
"""
import json
import numpy as np
from pathlib import Path

ISLES_DIR = "/home/pahm409/preprocessed_stroke_foundation/ISLES2022_resampled"
OUTPUT_FILE = "isles_splits_5fold_resampled.json"

# Get all ISLES cases
isles_dir = Path(ISLES_DIR)
all_cases = sorted([f.stem for f in isles_dir.glob("*.npz")])

print(f"Found {len(all_cases)} ISLES cases")
print(f"First 5: {all_cases[:5]}")
print(f"Last 5: {all_cases[-5:]}")

# Create 5-fold splits
np.random.seed(42)
np.random.shuffle(all_cases)

n_cases = len(all_cases)
fold_size = n_cases // 5

splits = {}

for fold in range(5):
    # Test set
    test_start = fold * fold_size
    test_end = test_start + fold_size if fold < 4 else n_cases
    test_cases = all_cases[test_start:test_end]
    
    # Remaining cases
    remaining = [c for c in all_cases if c not in test_cases]
    
    # Val set (20% of remaining)
    val_size = len(remaining) // 5
    val_cases = remaining[:val_size]
    train_cases = remaining[val_size:]
    
    splits[f'fold_{fold}'] = {
        'ISLES2022_resampled': {  # ← Changed from 'ISLES2022'
            'train': train_cases,
            'val': val_cases,
            'test': test_cases
        }
    }
    
    print(f"\nFold {fold}:")
    print(f"  Train: {len(train_cases)}")
    print(f"  Val:   {len(val_cases)}")
    print(f"  Test:  {len(test_cases)}")

# Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\n✓ Splits saved to: {OUTPUT_FILE}")

# Verify by loading
with open(OUTPUT_FILE, 'r') as f:
    loaded = json.load(f)

print(f"\n✓ Verification:")
print(f"  Keys in file: {list(loaded.keys())}")
print(f"  Datasets in fold_0: {list(loaded['fold_0'].keys())}")
print(f"  Train cases in fold_0: {len(loaded['fold_0']['ISLES2022_resampled']['train'])}")

# Save as: create_simclr_atlas_splits.py
import json

# Load your fold-based splits
with open('/home/pahm409/preprocessed_atlas_for_isles/atlas_splits.json', 'r') as f:
    fold_splits = json.load(f)

# Extract the data from inside fold_0
atlas_data = fold_splits['fold_0']['ATLAS_resampled']

# Create new format WITHOUT fold_0 wrapper
simclr_splits = {
    'ATLAS_resampled': {
        'train': atlas_data['train'],
        'val': atlas_data['val']
    }
}

# Save new file
with open('/home/pahm409/preprocessed_atlas_for_isles/atlas_splits_simclr.json', 'w') as f:
    json.dump(simclr_splits, f, indent=2)

print("✓ Created SimCLR-compatible splits file")
print(f"  Train cases: {len(simclr_splits['ATLAS_resampled']['train'])}")
print(f"  Val cases: {len(simclr_splits['ATLAS_resampled']['val'])}")

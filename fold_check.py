import json

with open('isles_splits_5fold_resampled.json', 'r') as f:
    splits = json.load(f)

print("Splits structure:")
print(json.dumps(splits, indent=2)[:500])  # First 500 chars

# Check fold_0 specifically
if 'fold_0' in splits:
    print("\nfold_0 keys:", list(splits['fold_0'].keys()))
    
    if 'ISLES2022_resampled' in splits['fold_0']:
        print("fold_0['ISLES2022_resampled'] keys:", 
              list(splits['fold_0']['ISLES2022_resampled'].keys()))
        print("Test cases in fold_0:", 
              len(splits['fold_0']['ISLES2022_resampled']['test']))

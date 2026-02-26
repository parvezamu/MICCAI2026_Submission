"""
dataset/patch_dataset_with_centers.py
Patch dataset that stores centers for proper reconstruction
WITH BALANCED DATASET × LESION-SIZE SAMPLING SUPPORT

Author: Parvez
Date: January 2026
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class PatchDatasetWithCenters(Dataset):
    """
    Patch dataset that stores centers for reconstruction
    Supports 5-fold cross-validation
    
    NEW: Computes lesion size bins per dataset to enable balanced sampling
    This prevents shortcuts like "ATLAS → small lesions, UOA → large lesions"
    """
    
    def __init__(
        self,
        preprocessed_dir,
        datasets,
        split,
        splits_file,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True,
        lesion_focus_ratio=0.7,
        fold=None,
        compute_lesion_bins=True,  # NEW: option to compute bins
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.patch_size = np.array(patch_size)
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.split = split
        self.lesion_focus_ratio = lesion_focus_ratio
        
        # Load splits
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Handle fold-based splits
        if fold is not None:
            fold_key = f'fold_{fold}'
            if fold_key in splits:
                splits = splits[fold_key]
                print(f"Using fold {fold} from {splits_file}")
            else:
                available_keys = list(splits.keys())
                raise ValueError(
                    f"Fold '{fold_key}' not found in {splits_file}. "
                    f"Available keys: {available_keys}"
                )
        
        # Collect volumes
        self.volumes = []
        for dataset_name in datasets:
            if dataset_name not in splits:
                print(f"Warning: Dataset '{dataset_name}' not found in splits file, skipping...")
                continue
            
            dataset_dir = self.preprocessed_dir / dataset_name
            
            if split not in splits[dataset_name]:
                print(f"Warning: Split '{split}' not found for dataset '{dataset_name}', skipping...")
                continue
            
            case_ids = splits[dataset_name][split]
            
            for case_id in case_ids:
                npz_path = dataset_dir / f'{case_id}.npz'
                if npz_path.exists():
                    self.volumes.append({
                        'dataset': dataset_name,
                        'case_id': case_id,
                        'npz_path': str(npz_path)
                    })
                else:
                    print(f"Warning: File not found: {npz_path}")
        
        # Print summary
        fold_msg = f" (fold {fold})" if fold is not None else ""
        print(f"\nLoaded {len(self.volumes)} volumes for {split} split{fold_msg}")
        print(f"  Patches per volume: {patches_per_volume}")
        print(f"  Total patches per epoch: {len(self.volumes) * patches_per_volume}")
        
        # Print dataset breakdown
        dataset_counts = {}
        for vol in self.volumes:
            ds = vol['dataset']
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        print(f"  Dataset breakdown:")
        for ds_name, count in sorted(dataset_counts.items()):
            print(f"    {ds_name}: {count} volumes")
        
        # ============================================================
        # NEW: Compute lesion size bins per dataset
        # ============================================================
        if compute_lesion_bins:
            print("\n  Computing lesion size bins per dataset...")
            self._compute_lesion_bins()
        else:
            # Initialize empty arrays (for compatibility)
            self.lesion_voxels = np.zeros(len(self.volumes), dtype=np.int64)
            self.volume_bins = np.zeros(len(self.volumes), dtype=np.int64)
    
    def _compute_lesion_bins(self):
        """
        Compute dataset-specific lesion size bins to enable balanced sampling.
        
        For each dataset separately:
        1. Load all lesion masks
        2. Compute 33rd and 66th percentiles
        3. Assign bins: 0=small, 1=medium, 2=large
        
        This prevents global bins from encoding dataset identity.
        """
        # Load lesion volumes for all cases
        self.lesion_voxels = np.zeros(len(self.volumes), dtype=np.int64)
        
        print("    Loading lesion masks...")
        for i, v in enumerate(self.volumes):
            try:
                data = np.load(v['npz_path'])
                self.lesion_voxels[i] = int((data['lesion_mask'] > 0).sum())
            except Exception as e:
                print(f"    Warning: Failed to load {v['npz_path']}: {e}")
                self.lesion_voxels[i] = 0
        
        # Initialize bins (0=small, 1=medium, 2=large)
        self.volume_bins = np.zeros(len(self.volumes), dtype=np.int64)
        
        # Group volumes by dataset
        by_dataset = {}
        for i, v in enumerate(self.volumes):
            by_dataset.setdefault(v['dataset'], []).append(i)
        
        # Compute dataset-specific percentile bins
        print("    Computing dataset-specific percentile bins...")
        for ds_name, indices in sorted(by_dataset.items()):
            vals = self.lesion_voxels[indices]
            
            # Skip if no valid lesions
            if len(vals) == 0 or vals.max() == 0:
                print(f"    {ds_name}: No valid lesions, skipping binning")
                continue
            
            # Compute percentiles (dataset-specific)
            p33 = np.percentile(vals, 33.33)
            p66 = np.percentile(vals, 66.67)
            
            # Assign bins
            for i in indices:
                lesion_vol = self.lesion_voxels[i]
                if lesion_vol <= p33:
                    self.volume_bins[i] = 0  # small
                elif lesion_vol <= p66:
                    self.volume_bins[i] = 1  # medium
                else:
                    self.volume_bins[i] = 2  # large
            
            # Print distribution for this dataset
            bin_counts = [np.sum(self.volume_bins[indices] == b) for b in range(3)]
            bin_volumes = [vals[self.volume_bins[indices] == b] for b in range(3)]
            
            print(f"    {ds_name}:")
            print(f"      Percentiles: p33={p33:.0f}, p66={p66:.0f}")
            for b, name in enumerate(['small', 'medium', 'large']):
                n = bin_counts[b]
                if n > 0:
                    vols = bin_volumes[b]
                    print(f"      {name:7s}: {n:3d} volumes "
                          f"(lesion voxels: {vols.min():.0f}-{vols.max():.0f}, "
                          f"mean={vols.mean():.0f})")
                else:
                    print(f"      {name:7s}: {n:3d} volumes")
        
        print("  ✓ Lesion binning complete (dataset-specific percentiles)")
        print(f"  ✓ Bins: 0=small, 1=medium, 2=large\n")
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def _extract_patch_with_center(self, volume, mask, prefer_lesion=True):
        """
        Extract patch and return CENTER coordinate
        
        Returns:
            patch_volume, patch_mask, center
        """
        vol_shape = np.array(volume.shape)
        half_size = self.patch_size // 2
        
        # Valid range for centers
        min_center = half_size
        max_center = vol_shape - half_size
        
        # Ensure valid range
        for dim in range(3):
            if min_center[dim] >= max_center[dim]:
                min_center[dim] = vol_shape[dim] // 2
                max_center[dim] = vol_shape[dim] // 2
        
        # Find lesion-containing patch
        if prefer_lesion and np.random.rand() < self.lesion_focus_ratio:
            lesion_coords = np.where(mask > 0)
            
            if len(lesion_coords[0]) > 0:
                # Sample random lesion voxel
                idx = np.random.randint(len(lesion_coords[0]))
                center = np.array([
                    lesion_coords[0][idx],
                    lesion_coords[1][idx],
                    lesion_coords[2][idx]
                ])
                # Clamp to valid range
                center = np.clip(center, min_center, max_center)
            else:
                # No lesion, sample randomly
                center = np.array([
                    np.random.randint(min_center[0], max_center[0] + 1),
                    np.random.randint(min_center[1], max_center[1] + 1),
                    np.random.randint(min_center[2], max_center[2] + 1)
                ])
        else:
            # Random sampling
            center = np.array([
                np.random.randint(min_center[0], max_center[0] + 1),
                np.random.randint(min_center[1], max_center[1] + 1),
                np.random.randint(min_center[2], max_center[2] + 1)
            ])
        
        # Extract patch
        lower = center - half_size
        upper = center + half_size
        
        patch_volume = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        patch_mask = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        
        # Pad if needed (should rarely happen with proper center selection)
        if patch_volume.shape != tuple(self.patch_size):
            pad_d = self.patch_size[0] - patch_volume.shape[0]
            pad_h = self.patch_size[1] - patch_volume.shape[1]
            pad_w = self.patch_size[2] - patch_volume.shape[2]
            
            patch_volume = np.pad(
                patch_volume,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=volume.min()
            )
            
            patch_mask = np.pad(
                patch_mask,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
        
        return patch_volume, patch_mask, center
    
    def _augment_patch(self, patch, mask):
        """Apply augmentation (geometric + intensity)"""
        
        # Geometric augmentation
        # Random flip
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            patch = np.flip(patch, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # Random rotation (90° multiples)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            axes = (0, 1) if np.random.rand() > 0.5 else (1, 2)
            patch = np.rot90(patch, k=k, axes=axes).copy()
            mask = np.rot90(mask, k=k, axes=axes).copy()
        
        # Intensity augmentation
        # Additive brightness
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            patch = patch + shift
        
        # Multiplicative contrast
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            patch = patch * scale
        
        # Gamma correction
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            patch_min = patch.min()
            patch_range = patch.max() - patch_min
            if patch_range > 0:
                patch = (patch - patch_min) / patch_range
                patch = np.power(patch, gamma)
                patch = patch * patch_range + patch_min
        
        # Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, patch.shape)
            patch = patch + noise
        
        return patch, mask
    
    def __getitem__(self, idx):
        """
        Get a patch with metadata
        
        Returns dict with:
        - image: (1, D, H, W) tensor
        - mask: (D, H, W) tensor
        - center: (3,) tensor with patch center coordinates
        - volume_shape: (3,) tensor with original volume shape
        - case_id: str
        - vol_idx: int (volume index)
        - dataset: str (NEW - for balanced sampling)
        - lesion_bin: int (NEW - 0=small, 1=medium, 2=large)
        """
        vol_idx = idx // self.patches_per_volume
        volume_info = self.volumes[vol_idx]
        
        # Load volume
        data = np.load(volume_info['npz_path'])
        volume = data['image']
        mask = data['lesion_mask']
        
        # Extract patch with center
        patch_volume, patch_mask, center = self._extract_patch_with_center(
            volume, mask, prefer_lesion=True
        )
        
        # Augment if training
        if self.augment and self.split == 'train':
            patch_volume, patch_mask = self._augment_patch(patch_volume, patch_mask)
        
        # Convert to tensors
        patch_volume = torch.from_numpy(patch_volume).unsqueeze(0).float()
        patch_mask = torch.from_numpy(patch_mask).long()
        center = torch.from_numpy(center).long()
        
        return {
            'image': patch_volume,
            'mask': patch_mask,
            'center': center,
            'volume_shape': torch.tensor(volume.shape),
            'case_id': volume_info['case_id'],
            'vol_idx': vol_idx,
            'dataset': volume_info['dataset'],  # NEW
            'lesion_bin': int(self.volume_bins[vol_idx]),  # NEW
        }
    
    def get_volume_info(self, vol_idx):
        """
        Get volume info for reconstruction
        
        Returns dict with:
        - volume: (D, H, W) array
        - mask: (D, H, W) array
        - case_id: str
        - dataset: str (NEW)
        - lesion_voxels: int (NEW)
        - lesion_bin: int (NEW)
        """
        volume_info = self.volumes[vol_idx]
        data = np.load(volume_info['npz_path'])
        
        return {
            'volume': data['image'],
            'mask': data['lesion_mask'],
            'case_id': volume_info['case_id'],
            'dataset': volume_info['dataset'],  # NEW
            'lesion_voxels': int(self.lesion_voxels[vol_idx]),  # NEW
            'lesion_bin': int(self.volume_bins[vol_idx]),  # NEW
        }


# Test the dataset
if __name__ == '__main__':
    print("="*70)
    print("Testing PatchDatasetWithCenters with Lesion Binning")
    print("="*70 + "\n")
    
    # Test with fold
    try:
        dataset = PatchDatasetWithCenters(
            preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
            datasets=['ATLAS', 'UOA_Private'],
            split='train',
            splits_file='splits_5fold.json',
            fold=0,
            patch_size=(96, 96, 96),
            patches_per_volume=10,
            augment=True,
            lesion_focus_ratio=0.7,
            compute_lesion_bins=True  # Enable binning
        )
        
        print(f"\n✓ Successfully loaded fold 0 with lesion binning")
        print(f"  Total patches: {len(dataset)}")
        print(f"  Total volumes: {len(dataset.volumes)}")
        
        # Test getting a sample
        print("\n" + "="*70)
        print("Testing __getitem__")
        print("="*70)
        
        sample = dataset[0]
        print(f"\n✓ Sample loaded:")
        print(f"  Image shape:     {sample['image'].shape}")
        print(f"  Mask shape:      {sample['mask'].shape}")
        print(f"  Center:          {sample['center']}")
        print(f"  Case ID:         {sample['case_id']}")
        print(f"  Dataset:         {sample['dataset']}")
        print(f"  Lesion bin:      {sample['lesion_bin']} (0=small, 1=med, 2=large)")
        
        # Test get_volume_info
        print("\n" + "="*70)
        print("Testing get_volume_info")
        print("="*70)
        
        vol_info = dataset.get_volume_info(0)
        print(f"\n✓ Volume info retrieved:")
        print(f"  Volume shape:    {vol_info['volume'].shape}")
        print(f"  Mask shape:      {vol_info['mask'].shape}")
        print(f"  Case ID:         {vol_info['case_id']}")
        print(f"  Dataset:         {vol_info['dataset']}")
        print(f"  Lesion voxels:   {vol_info['lesion_voxels']}")
        print(f"  Lesion bin:      {vol_info['lesion_bin']}")
        
        # Test bin distribution
        print("\n" + "="*70)
        print("Bin Distribution Check")
        print("="*70)
        
        from collections import defaultdict
        bin_counts = defaultdict(lambda: defaultdict(int))
        
        for vol_idx in range(len(dataset.volumes)):
            vol = dataset.volumes[vol_idx]
            ds = vol['dataset']
            bin_id = dataset.volume_bins[vol_idx]
            bin_counts[ds][bin_id] += 1
        
        print()
        for ds in sorted(bin_counts.keys()):
            print(f"{ds}:")
            for bin_id in range(3):
                bin_name = ['small', 'medium', 'large'][bin_id]
                count = bin_counts[ds][bin_id]
                print(f"  {bin_name:7s}: {count} volumes")
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure to run 'python create_5fold_splits.py' first!")

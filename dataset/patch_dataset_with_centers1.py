"""
dataset/patch_dataset_with_centers.py

Patch dataset that stores centers for proper reconstruction

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
        
        # Collect volumes
        self.volumes = []
        for dataset_name in datasets:
            if dataset_name not in splits:
                continue
            
            dataset_dir = self.preprocessed_dir / dataset_name
            case_ids = splits[dataset_name][split]
            
            for case_id in case_ids:
                npz_path = dataset_dir / f'{case_id}.npz'
                if npz_path.exists():
                    self.volumes.append({
                        'dataset': dataset_name,
                        'case_id': case_id,
                        'npz_path': str(npz_path)
                    })
        
        print(f"Loaded {len(self.volumes)} volumes for {split} split")
        print(f"  Patches per volume: {patches_per_volume}")
        print(f"  Total patches per epoch: {len(self.volumes) * patches_per_volume}")
    
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
        
        # Ensure valid
        for dim in range(3):
            if min_center[dim] >= max_center[dim]:
                min_center[dim] = vol_shape[dim] // 2
                max_center[dim] = vol_shape[dim] // 2
        
        # Find lesion-containing patch
        if prefer_lesion and np.random.rand() < self.lesion_focus_ratio:
            lesion_coords = np.where(mask > 0)
            
            if len(lesion_coords[0]) > 0:
                idx = np.random.randint(len(lesion_coords[0]))
                center = np.array([
                    lesion_coords[0][idx],
                    lesion_coords[1][idx],
                    lesion_coords[2][idx]
                ])
                center = np.clip(center, min_center, max_center)
            else:
                center = np.array([
                    np.random.randint(min_center[0], max_center[0] + 1),
                    np.random.randint(min_center[1], max_center[1] + 1),
                    np.random.randint(min_center[2], max_center[2] + 1)
                ])
        else:
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
        
        # Pad if needed
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
        """Apply augmentation"""
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            patch = np.flip(patch, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            axes = (0, 1) if np.random.rand() > 0.5 else (1, 2)
            patch = np.rot90(patch, k=k, axes=axes).copy()
            mask = np.rot90(mask, k=k, axes=axes).copy()
        
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            patch = patch + shift
        
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            patch = patch * scale
        
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            patch_min = patch.min()
            patch_range = patch.max() - patch_min
            if patch_range > 0:
                patch = (patch - patch_min) / patch_range
                patch = np.power(patch, gamma)
                patch = patch * patch_range + patch_min
        
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, patch.shape)
            patch = patch + noise
        
        return patch, mask
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        volume_info = self.volumes[vol_idx]
        
        data = np.load(volume_info['npz_path'])
        volume = data['image']
        mask = data['lesion_mask']
        
        patch_volume, patch_mask, center = self._extract_patch_with_center(
            volume, mask, prefer_lesion=True
        )
        
        if self.augment and self.split == 'train':
            patch_volume, patch_mask = self._augment_patch(patch_volume, patch_mask)
        
        patch_volume = torch.from_numpy(patch_volume).unsqueeze(0).float()
        patch_mask = torch.from_numpy(patch_mask).long()
        center = torch.from_numpy(center).long()
        
        return {
            'image': patch_volume,
            'mask': patch_mask,
            'center': center,
            'volume_shape': torch.tensor(volume.shape),
            'case_id': volume_info['case_id'],
            'vol_idx': vol_idx
        }
    
    def get_volume_info(self, vol_idx):
        """Get volume info for reconstruction"""
        volume_info = self.volumes[vol_idx]
        data = np.load(volume_info['npz_path'])
        
        return {
            'volume': data['image'],
            'mask': data['lesion_mask'],
            'case_id': volume_info['case_id']
        }

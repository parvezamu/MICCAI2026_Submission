"""
dataset/patch_dataset_improved.py

Improved patch sampling with sparse training and dense validation

Author: Parvez
Date: January 2026
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import random


class ImprovedPatchDataset(Dataset):
    """
    Improved patch-based dataset with better sampling strategy
    
    Training: Sparse random sampling with lesion focus
    Validation: Dense sliding window for full coverage
    """
    
    def __init__(
        self,
        preprocessed_dir,
        datasets,
        split,
        splits_file,
        patch_size=(96, 96, 96),
        patches_per_volume=10,  # Increased from 4
        augment=True,
        lesion_focus_ratio=0.7,  # 70% patches contain lesions
        validation_overlap=0.5   # 50% overlap for dense validation
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.split = split
        self.lesion_focus_ratio = lesion_focus_ratio
        self.validation_overlap = validation_overlap
        
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
        
        if split == 'train':
            print(f"Training mode: {patches_per_volume} random patches per volume")
            print(f"  - {int(patches_per_volume * lesion_focus_ratio)} lesion-focused")
            print(f"  - {int(patches_per_volume * (1-lesion_focus_ratio))} random")
            print(f"Total training patches per epoch: {len(self.volumes) * patches_per_volume}")
        else:
            print(f"Validation mode: Dense sliding window (overlap={validation_overlap})")
    
    def __len__(self):
        if self.split == 'train':
            return len(self.volumes) * self.patches_per_volume
        else:
            # For validation, each volume is one "sample"
            return len(self.volumes)
    
    def _extract_random_patch(self, volume, mask, prefer_lesion=True):
        """Extract a random patch, optionally focused on lesions"""
        vol_shape = volume.shape
        
        valid_d = max(0, vol_shape[0] - self.patch_size[0])
        valid_h = max(0, vol_shape[1] - self.patch_size[1])
        valid_w = max(0, vol_shape[2] - self.patch_size[2])
        
        # Try to find lesion-containing patch if requested
        if prefer_lesion and np.random.rand() < self.lesion_focus_ratio:
            lesion_coords = np.where(mask > 0)
            
            if len(lesion_coords[0]) > 0:
                # Sample random lesion voxel
                idx = np.random.randint(len(lesion_coords[0]))
                center_d = lesion_coords[0][idx]
                center_h = lesion_coords[1][idx]
                center_w = lesion_coords[2][idx]
                
                # Center patch on lesion
                start_d = np.clip(center_d - self.patch_size[0]//2, 0, valid_d)
                start_h = np.clip(center_h - self.patch_size[1]//2, 0, valid_h)
                start_w = np.clip(center_w - self.patch_size[2]//2, 0, valid_w)
            else:
                # No lesion, random patch
                start_d = np.random.randint(0, valid_d + 1) if valid_d > 0 else 0
                start_h = np.random.randint(0, valid_h + 1) if valid_h > 0 else 0
                start_w = np.random.randint(0, valid_w + 1) if valid_w > 0 else 0
        else:
            # Random patch
            start_d = np.random.randint(0, valid_d + 1) if valid_d > 0 else 0
            start_h = np.random.randint(0, valid_h + 1) if valid_h > 0 else 0
            start_w = np.random.randint(0, valid_w + 1) if valid_w > 0 else 0
        
        # Extract patch
        return self._extract_patch_at(volume, mask, start_d, start_h, start_w)
    
    def _extract_patch_at(self, volume, mask, start_d, start_h, start_w):
        """Extract patch at specific location"""
        end_d = start_d + self.patch_size[0]
        end_h = start_h + self.patch_size[1]
        end_w = start_w + self.patch_size[2]
        
        patch_volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
        patch_mask = mask[start_d:end_d, start_h:end_h, start_w:end_w]
        
        # Pad if necessary
        if patch_volume.shape != self.patch_size:
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
        
        return patch_volume, patch_mask
    
    def _get_sliding_window_patches(self, volume, mask):
        """
        Extract patches using sliding window for dense coverage
        Returns list of (patch_volume, patch_mask, location)
        """
        vol_shape = volume.shape
        stride = tuple(int(ps * (1 - self.validation_overlap)) for ps in self.patch_size)
        
        patches = []
        
        # Calculate number of patches in each dimension
        n_d = max(1, int(np.ceil((vol_shape[0] - self.patch_size[0]) / stride[0])) + 1)
        n_h = max(1, int(np.ceil((vol_shape[1] - self.patch_size[1]) / stride[1])) + 1)
        n_w = max(1, int(np.ceil((vol_shape[2] - self.patch_size[2]) / stride[2])) + 1)
        
        for i in range(n_d):
            for j in range(n_h):
                for k in range(n_w):
                    start_d = min(i * stride[0], vol_shape[0] - self.patch_size[0])
                    start_h = min(j * stride[1], vol_shape[1] - self.patch_size[1])
                    start_w = min(k * stride[2], vol_shape[2] - self.patch_size[2])
                    
                    start_d = max(0, start_d)
                    start_h = max(0, start_h)
                    start_w = max(0, start_w)
                    
                    patch_vol, patch_msk = self._extract_patch_at(
                        volume, mask, start_d, start_h, start_w
                    )
                    
                    patches.append({
                        'volume': patch_vol,
                        'mask': patch_msk,
                        'location': (start_d, start_h, start_w)
                    })
        
        return patches
    
    def _augment_patch(self, patch, mask):
        """Apply data augmentation"""
        # Random flip
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            patch = np.flip(patch, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # Random rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            axes = (0, 1) if np.random.rand() > 0.5 else (1, 2)
            patch = np.rot90(patch, k=k, axes=axes).copy()
            mask = np.rot90(mask, k=k, axes=axes).copy()
        
        # Random intensity shift
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            patch = patch + shift
        
        # Random intensity scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            patch = patch * scale
        
        # Random gamma
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
        if self.split == 'train':
            # Training: Random patch extraction
            vol_idx = idx // self.patches_per_volume
            volume_info = self.volumes[vol_idx]
            
            # Load volume
            data = np.load(volume_info['npz_path'])
            volume = data['image']
            mask = data['lesion_mask']
            
            # Extract random patch
            patch_volume, patch_mask = self._extract_random_patch(
                volume, mask, prefer_lesion=True
            )
            
            # Apply augmentation
            if self.augment:
                patch_volume, patch_mask = self._augment_patch(patch_volume, patch_mask)
            
            # Convert to tensors
            patch_volume = torch.from_numpy(patch_volume).unsqueeze(0).float()
            patch_mask = torch.from_numpy(patch_mask).long()
            
            return {
                'image': patch_volume,
                'mask': patch_mask,
                'case_id': volume_info['case_id'],
                'dataset': volume_info['dataset']
            }
        
        else:
            # Validation: Return all patches from volume (for dense prediction)
            volume_info = self.volumes[idx]
            
            # Load volume
            data = np.load(volume_info['npz_path'])
            volume = data['image']
            mask = data['lesion_mask']
            
            # Get all patches via sliding window
            patches = self._get_sliding_window_patches(volume, mask)
            
            # Convert to tensors
            patch_volumes = []
            patch_masks = []
            locations = []
            
            for p in patches:
                patch_volumes.append(
                    torch.from_numpy(p['volume']).unsqueeze(0).float()
                )
                patch_masks.append(
                    torch.from_numpy(p['mask']).long()
                )
                locations.append(p['location'])
            
            return {
                'patches': patch_volumes,  # List of patches
                'masks': patch_masks,      # List of masks
                'locations': locations,    # List of (d, h, w) tuples
                'volume_shape': volume.shape,
                'case_id': volume_info['case_id'],
                'dataset': volume_info['dataset']
            }

"""
dataset/patch_dataset.py

Patch-based dataset for efficient training with larger batch sizes

Author: Parvez
Date: January 2026
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import random


class PatchBasedSegmentationDataset(Dataset):
    """
    Extract random patches from 3D volumes for efficient training
    Allows batch_size > 1 and faster convergence
    """
    
    def __init__(
        self,
        preprocessed_dir,
        datasets,
        split,
        splits_file,
        patch_size=(96, 96, 96),
        patches_per_volume=4,
        augment=True
    ):
        """
        Args:
            preprocessed_dir: Directory with preprocessed .npz files
            datasets: List of dataset names to use
            split: 'train', 'val', or 'test'
            splits_file: JSON file with train/val/test splits
            patch_size: Size of patches to extract (D, H, W)
            patches_per_volume: Number of patches to extract per volume per epoch
            augment: Whether to apply data augmentation
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.split = split
        
        # Load splits
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Collect all volumes
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
        print(f"Total patches per epoch: {len(self.volumes) * patches_per_volume}")
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def _extract_patch(self, volume, mask, patch_size):
        """
        Extract a random patch from volume and mask
        Prefer patches with lesions during training
        """
        vol_shape = volume.shape  # (D, H, W)
        
        # Calculate valid ranges for patch extraction
        valid_d = max(0, vol_shape[0] - patch_size[0])
        valid_h = max(0, vol_shape[1] - patch_size[1])
        valid_w = max(0, vol_shape[2] - patch_size[2])
        
        # For training, prefer patches with lesions
        if self.split == 'train' and self.augment and np.random.rand() > 0.3:
            # Find lesion locations
            lesion_coords = np.where(mask > 0)
            
            if len(lesion_coords[0]) > 0:
                # Sample random lesion voxel as center
                idx = np.random.randint(len(lesion_coords[0]))
                center_d = lesion_coords[0][idx]
                center_h = lesion_coords[1][idx]
                center_w = lesion_coords[2][idx]
                
                # Calculate patch boundaries centered on lesion
                start_d = np.clip(center_d - patch_size[0]//2, 0, valid_d)
                start_h = np.clip(center_h - patch_size[1]//2, 0, valid_h)
                start_w = np.clip(center_w - patch_size[2]//2, 0, valid_w)
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
        end_d = start_d + patch_size[0]
        end_h = start_h + patch_size[1]
        end_w = start_w + patch_size[2]
        
        patch_volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
        patch_mask = mask[start_d:end_d, start_h:end_h, start_w:end_w]
        
        # Pad if necessary (for volumes smaller than patch size)
        if patch_volume.shape != patch_size:
            pad_d = patch_size[0] - patch_volume.shape[0]
            pad_h = patch_size[1] - patch_volume.shape[1]
            pad_w = patch_size[2] - patch_volume.shape[2]
            
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
    
    def _augment_patch(self, patch, mask):
        """Apply data augmentation to patch"""
        # Random flip
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            patch = np.flip(patch, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # Random rotation (90, 180, 270 degrees)
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
        
        # Random gamma correction
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
        # Get volume index
        vol_idx = idx // self.patches_per_volume
        volume_info = self.volumes[vol_idx]
        
        # Load volume
        data = np.load(volume_info['npz_path'])
        volume = data['image']
        mask = data['lesion_mask']
        
        # Extract patch
        patch_volume, patch_mask = self._extract_patch(volume, mask, self.patch_size)
        
        # Apply augmentation
        if self.augment and self.split == 'train':
            patch_volume, patch_mask = self._augment_patch(patch_volume, patch_mask)
        
        # Convert to tensors
        patch_volume = torch.from_numpy(patch_volume).unsqueeze(0).float()  # (1, D, H, W)
        patch_mask = torch.from_numpy(patch_mask).long()  # (D, H, W)
        
        return {
            'image': patch_volume,
            'mask': patch_mask,
            'case_id': volume_info['case_id'],
            'dataset': volume_info['dataset']
        }

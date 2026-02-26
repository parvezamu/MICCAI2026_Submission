"""
dataset/simclr_dataset.py

Dataset for SimCLR training with adaptive 3D patching

Author: Parvez
Date: January 2026
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class SimCLRStrokeDataset(Dataset):
    """
    Dataset for SimCLR pre-training on stroke MRI
    Handles full volumes and creates augmented pairs
    """
    
    def __init__(
        self,
        preprocessed_dir,
        datasets,
        split='train',
        splits_file=None,
        augmentation=None
    ):
        """
        Args:
            preprocessed_dir: Base directory with preprocessed data
            datasets: List of dataset names to include
            split: 'train', 'val', or 'test'
            splits_file: Path to splits JSON file
            augmentation: SimCLRAugmentation instance
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.datasets = datasets
        self.split = split
        self.augmentation = augmentation
        
        # Load splits
        if splits_file and Path(splits_file).exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
        else:
            raise ValueError(f"Splits file not found: {splits_file}")
        
        # Collect all case paths
        self.cases = []
        
        for dataset_name in datasets:
            if dataset_name not in splits:
                print(f"Warning: {dataset_name} not in splits file")
                continue
            
            dataset_dir = self.preprocessed_dir / dataset_name
            case_ids = splits[dataset_name][split]
            
            for case_id in case_ids:
                npz_path = dataset_dir / f'{case_id}.npz'
                if npz_path.exists():
                    self.cases.append({
                        'dataset': dataset_name,
                        'case_id': case_id,
                        'npz_path': str(npz_path)
                    })
        
        print(f"Loaded {len(self.cases)} cases for {split} split from {len(datasets)} datasets")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_info = self.cases[idx]
        
        # Load preprocessed data
        data = np.load(case_info['npz_path'])
        image = data['image']  # Already normalized (D, H, W)
        
        # Apply augmentation to create two views
        if self.augmentation:
            view1, view2 = self.augmentation(image)
        else:
            # No augmentation - just convert to tensor
            view1 = torch.from_numpy(image).unsqueeze(0).float()
            view2 = view1.clone()
        
        return {
            'view1': view1,  # (1, D, H, W)
            'view2': view2,  # (1, D, H, W)
            'case_id': case_info['case_id'],
            'dataset': case_info['dataset']
        }

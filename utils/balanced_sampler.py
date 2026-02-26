"""
Balanced batch sampler for dataset × lesion-size stratification
Prevents ATLAS/UOA shortcuts based on lesion size
"""

import numpy as np
from torch.utils.data import Sampler


class BalancedDatasetLesionBatchSampler(Sampler):
    """
    Ensures each batch has balanced representation across:
    - Datasets (ATLAS, UOA)
    - Lesion sizes (small, medium, large)
    
    This prevents the network from learning spurious correlations like
    "ATLAS → small lesions" or "UOA → large lesions"
    """
    
    def __init__(self, dataset, batch_size, datasets=("ATLAS", "UOA_Private"),
                 bins=(0, 1, 2), seed=42, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.datasets = list(datasets)
        self.bins = list(bins)
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        
        # Build groups: (dataset, bin) → list of volume indices
        self.groups = {}
        for ds in self.datasets:
            for b in self.bins:
                self.groups[(ds, b)] = []
        
        for vol_idx, v in enumerate(dataset.volumes):
            ds = v['dataset']
            if ds not in self.datasets:
                continue
            b = dataset.volume_bins[vol_idx]
            self.groups[(ds, b)].append(vol_idx)
        
        # Check for empty groups
        for k, v in self.groups.items():
            if len(v) == 0:
                print(f"WARNING: Empty group {k}")
        
        # Calculate samples per group per batch
        num_groups = len(self.datasets) * len(self.bins)  # 2 × 3 = 6
        base = batch_size // num_groups
        remainder = batch_size % num_groups
        
        self.per_group = {k: base for k in self.groups.keys()}
        
        # Distribute remainder
        keys = list(self.per_group.keys())
        self.rng.shuffle(keys)
        for k in keys[:remainder]:
            self.per_group[k] += 1
        
        # Print batch composition
        print("\n" + "="*70)
        print("BALANCED BATCH SAMPLER CONFIGURATION")
        print("="*70)
        print(f"Batch size: {batch_size}")
        print(f"Composition per batch:")
        for (ds, bin_id), count in sorted(self.per_group.items()):
            bin_name = ['small', 'medium', 'large'][bin_id]
            n_vols = len(self.groups[(ds, bin_id)])
            print(f"  {ds:15s} × {bin_name:7s}: {count} patches ({n_vols} volumes available)")
        print("="*70 + "\n")
        
        # Calculate number of batches
        total_patches = len(dataset)
        self.num_batches = total_patches // batch_size
        if not drop_last and total_patches % batch_size != 0:
            self.num_batches += 1
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        # Shuffle volume lists per group
        group_lists = {}
        group_ptrs = {}
        
        for k, vols in self.groups.items():
            vols = list(vols)
            self.rng.shuffle(vols)
            group_lists[k] = vols
            group_ptrs[k] = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            for k, n_take in self.per_group.items():
                vols = group_lists[k]
                if len(vols) == 0:
                    continue
                
                for _ in range(n_take):
                    # Get volume index (with cycling)
                    ptr = group_ptrs[k]
                    if ptr >= len(vols):
                        self.rng.shuffle(vols)
                        ptr = 0
                    
                    vol_idx = vols[ptr]
                    group_ptrs[k] = ptr + 1
                    
                    # Sample random patch from this volume
                    patch_offset = self.rng.randint(0, self.dataset.patches_per_volume)
                    idx = vol_idx * self.dataset.patches_per_volume + patch_offset
                    batch_indices.append(idx)
            
            # Handle size mismatches
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            
            yield batch_indices

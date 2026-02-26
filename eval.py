"""
evaluate_isles2022_BATCHED.py

Speed up with DataLoader batching
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('/home/pahm409')
from train_segmentation_corrected_DS_MAIN_ONLY import SegmentationModel
from models.resnet3d import resnet3d_18


class ISLESDatasetForEval(Dataset):
    """Dataset for ISLES evaluation"""
    
    def __init__(self, isles_dir, patch_size=(96, 96, 96), patches_per_volume=100, lesion_focus_ratio=0.7):
        self.isles_dir = Path(isles_dir)
        self.patch_size = np.array(patch_size)
        self.patches_per_volume = patches_per_volume
        self.lesion_focus_ratio = lesion_focus_ratio
        
        self.volumes = []
        for npz_file in sorted(self.isles_dir.glob("*.npz")):
            self.volumes.append({
                'case_id': npz_file.stem,
                'npz_path': str(npz_file)
            })
        
        print(f"Loaded {len(self.volumes)} ISLES cases")
        print(f"Patches per volume: {patches_per_volume}")
        print(f"Total patches: {len(self.volumes) * patches_per_volume}")
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def _extract_patch_with_center(self, volume, mask):
        """Extract patch centered on lesion (70% of time)"""
        vol_shape = np.array(volume.shape)
        half_size = self.patch_size // 2
        
        min_center = half_size
        max_center = vol_shape - half_size
        
        if np.random.rand() < self.lesion_focus_ratio:
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
        
        lower = center - half_size
        upper = center + half_size
        
        patch_volume = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        
        return patch_volume, center
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        volume_info = self.volumes[vol_idx]
        
        data = np.load(volume_info['npz_path'])
        volume = data['image']
        mask = data['lesion_mask']
        
        patch_volume, center = self._extract_patch_with_center(volume, mask)
        
        patch_volume = torch.from_numpy(patch_volume).unsqueeze(0).float()
        center = torch.from_numpy(center).long()
        
        return {
            'image': patch_volume,
            'center': center,
            'vol_idx': vol_idx
        }
    
    def get_volume_info(self, vol_idx):
        volume_info = self.volumes[vol_idx]
        data = np.load(volume_info['npz_path'])
        
        return {
            'volume': data['image'],
            'mask': data['lesion_mask'],
            'case_id': volume_info['case_id']
        }


def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """Reconstruction from training"""
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            continue
        
        patch = patch_preds[i, 1, ...]
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    mask = count_map > 0
    reconstructed[mask] = reconstructed[mask] / count_map[mask]
    
    return reconstructed, count_map


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(encoder, num_classes=2, attention_type='mkdc', deep_supervision=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded from epoch {checkpoint['epoch'] + 1}")
    if 'val_dsc_recon' in checkpoint:
        print(f"  Training DSC: {checkpoint['val_dsc_recon']:.4f}")
    
    return model


def evaluate_isles(model_checkpoint, isles_dir, output_dir, 
                  patch_size=(96, 96, 96), patches_per_volume=100, batch_size=32):
    """Evaluate with DataLoader batching for speed"""
    
    device = torch.device('cuda:0')
    model = load_model(model_checkpoint, device)
    
    dataset = ISLESDatasetForEval(
        isles_dir=isles_dir,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        lesion_focus_ratio=0.7
    )
    
    # DataLoader with batching
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Num workers: 4")
    print(f"\n{'='*90}")
    print(f"{'CaseID':<25} {'DSC':>8} {'GT':>10} {'Pred':>10} {'Inter':>10} {'Patches':>8}")
    print(f"{'='*90}")
    
    # Collect predictions
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing batches'):
            images = batch['image'].to(device)
            centers = batch['center'].cpu().numpy()
            vol_indices = batch['vol_idx'].cpu().numpy()
            
            with autocast():
                outputs = model(images)
                if isinstance(outputs, list):
                    outputs = outputs[0]
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(images)):
                vol_idx = vol_indices[i]
                volume_data[vol_idx]['centers'].append(centers[i])
                volume_data[vol_idx]['preds'].append(preds[i])
    
    # Reconstruct
    results = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed, _ = reconstruct_from_patches_with_count(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        pred_binary = (reconstructed > 0.5).astype(np.uint8)
        mask_gt = (vol_info['mask'] > 0).astype(np.uint8)
        
        intersection = (pred_binary * mask_gt).sum()
        union = pred_binary.sum() + mask_gt.sum()
        
        dsc = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary.sum() == 0 else 0.0)
        
        result = {
            'case_id': case_id,
            'dsc': float(dsc),
            'lesion_gt': int(mask_gt.sum()),
            'lesion_pred': int(pred_binary.sum()),
            'intersection': int(intersection)
        }
        
        results.append(result)
        
        print(f"{case_id:<25} {result['dsc']:>8.4f} {result['lesion_gt']:>10} "
              f"{result['lesion_pred']:>10} {result['intersection']:>10} {patches_per_volume:>8}")
    
    # Stats
    dices = [r['dsc'] for r in results]
    stats = {
        'mean': float(np.mean(dices)),
        'std': float(np.std(dices)),
        'median': float(np.median(dices)),
        'min': float(np.min(dices)),
        'max': float(np.max(dices))
    }
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'results.txt', 'w') as f:
        f.write("ISLES 2022 External Validation\n")
        f.write("="*80 + "\n\n")
        f.write(f"Cases: {len(results)}\n")
        f.write(f"Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        f.write(f"Median: {stats['median']:.4f}\n")
        f.write(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
        
        for r in sorted(results, key=lambda x: x['dsc'], reverse=True):
            f.write(f"{r['case_id']:<25} {r['dsc']:>8.4f}\n")
    
    print(f"\n{'='*90}")
    print(f"Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"{'='*90}")
    
    return results, stats


if __name__ == '__main__':
    import glob
    
    pattern = "/home/pahm409/ablation_study_fold0/SimCLR_Pretrained/mkdc_ds/fold_0/exp_20260127_075336/checkpoints/best_model.pth"
    paths = glob.glob(pattern)
    
    if not paths:
        print("No checkpoint!")
        exit(1)
    
    results, stats = evaluate_isles(
        model_checkpoint=paths[0],
        isles_dir="/home/pahm409/isles2022_preprocessed_MATCHED",
        output_dir="/home/pahm409/isles2022_evaluation",
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        batch_size=32  # Process 32 patches at once
    )

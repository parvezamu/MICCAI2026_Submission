"""
evaluate_test_set.py

Proper external validation on held-out test set (ATLAS + UOA)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import json
import sys

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/dataset')

from train_segmentation_corrected_DS_MAIN_ONLY import SegmentationModel
from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """Reconstruction with weighted averaging"""
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
    
    print(f"\n✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
    if 'val_dsc_recon' in checkpoint:
        print(f"  Training validation DSC: {checkpoint['val_dsc_recon']:.4f}")
    
    return model


def evaluate_test_set(model_checkpoint, preprocessed_dir, splits_file, fold, output_dir,
                     patch_size=(96, 96, 96), patches_per_volume=100, batch_size=32):
    """
    Evaluate on held-out test set (never seen during training)
    """
    
    device = torch.device('cuda:0')
    model = load_model(model_checkpoint, device)
    
    # Load test set using your PatchDatasetWithCenters
    test_dataset = PatchDatasetWithCenters(
        preprocessed_dir=preprocessed_dir,
        datasets=['ATLAS', 'UOA_Private'],
        split='test',  # ← TEST SET
        splits_file=splits_file,
        fold=fold,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        augment=False,  # No augmentation for test
        lesion_focus_ratio=0.7
    )
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"  Test set size: {len(test_dataset.volumes)} volumes")
    print(f"  Total patches: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}\n")
    
    print("="*100)
    print(f"{'CaseID':<35} {'Dataset':<15} {'DSC':>8} {'GT':>10} {'Pred':>10} {'Inter':>10}")
    print("="*100)
    
    # Collect predictions
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing test patches'):
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
    
    # Reconstruct volumes
    results = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing test volumes'):
        vol_info = test_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        dataset_name = vol_info['dataset']
        
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
            'dataset': dataset_name,
            'dsc': float(dsc),
            'lesion_gt': int(mask_gt.sum()),
            'lesion_pred': int(pred_binary.sum()),
            'intersection': int(intersection)
        }
        
        results.append(result)
        
        print(f"{case_id:<35} {dataset_name:<15} {result['dsc']:>8.4f} {result['lesion_gt']:>10} "
              f"{result['lesion_pred']:>10} {result['intersection']:>10}")
    
    # Statistics overall
    dices = [r['dsc'] for r in results]
    
    stats_overall = {
        'mean': float(np.mean(dices)),
        'std': float(np.std(dices)),
        'median': float(np.median(dices)),
        'q25': float(np.percentile(dices, 25)),
        'q75': float(np.percentile(dices, 75)),
        'min': float(np.min(dices)),
        'max': float(np.max(dices)),
        'n': len(results)
    }
    
    # Statistics per dataset
    stats_per_dataset = {}
    for dataset in ['ATLAS', 'UOA_Private']:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if dataset_results:
            dataset_dices = [r['dsc'] for r in dataset_results]
            stats_per_dataset[dataset] = {
                'mean': float(np.mean(dataset_dices)),
                'std': float(np.std(dataset_dices)),
                'median': float(np.median(dataset_dices)),
                'n': len(dataset_results)
            }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'test_set_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST SET EVALUATION (Held-Out, Never Seen During Training)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {model_checkpoint}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Total test cases: {stats_overall['n']}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Mean DSC:   {stats_overall['mean']:.4f} ± {stats_overall['std']:.4f}\n")
        f.write(f"  Median DSC: {stats_overall['median']:.4f}\n")
        f.write(f"  Q1-Q3:      [{stats_overall['q25']:.4f}, {stats_overall['q75']:.4f}]\n")
        f.write(f"  Range:      [{stats_overall['min']:.4f}, {stats_overall['max']:.4f}]\n\n")
        
        f.write("Per-Dataset Statistics:\n")
        for dataset, stats in stats_per_dataset.items():
            f.write(f"  {dataset}:\n")
            f.write(f"    Cases: {stats['n']}\n")
            f.write(f"    Mean DSC: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"    Median DSC: {stats['median']:.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Per-Case Results (sorted by DSC):\n")
        f.write("="*80 + "\n")
        f.write(f"{'CaseID':<35} {'Dataset':<15} {'DSC':>8} {'GT':>10} {'Pred':>10}\n")
        f.write("-"*80 + "\n")
        
        for r in sorted(results, key=lambda x: x['dsc'], reverse=True):
            f.write(f"{r['case_id']:<35} {r['dataset']:<15} {r['dsc']:>8.4f} "
                   f"{r['lesion_gt']:>10} {r['lesion_pred']:>10}\n")
    
    print("\n" + "="*100)
    print("TEST SET EVALUATION COMPLETE")
    print("="*100)
    print(f"Total cases: {stats_overall['n']}")
    print(f"Mean DSC:    {stats_overall['mean']:.4f} ± {stats_overall['std']:.4f}")
    print(f"Median DSC:  {stats_overall['median']:.4f}")
    print(f"\nPer-dataset:")
    for dataset, stats in stats_per_dataset.items():
        print(f"  {dataset} (n={stats['n']}): {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("="*100)
    print(f"\nResults saved: {output_path / 'test_set_results.txt'}")
    
    return results, stats_overall, stats_per_dataset


if __name__ == '__main__':
    import glob
    
    # Find checkpoint
    pattern = "/home/pahm409/ablation_study_fold0/SimCLR_Pretrained/mkdc_ds/fold_0/exp_20260127_075336/checkpoints/best_model.pth"
    paths = glob.glob(pattern)
    
    if not paths:
        print("✗ No checkpoint found!")
        exit(1)
    
    print(f"Using checkpoint: {paths[0]}")
    
    # Evaluate on test set
    results, stats_overall, stats_per_dataset = evaluate_test_set(
        model_checkpoint=paths[0],
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        splits_file='/home/pahm409/ISLES2029/splits_5fold.json',
        fold=0,
        output_dir='/home/pahm409/test_set_evaluation',
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        batch_size=32
    )

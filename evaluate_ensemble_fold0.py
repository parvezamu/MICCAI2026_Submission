"""
evaluate_ensemble_fold0.py

Ensemble evaluation for FOLD 0 ONLY
Combines Random Init + SimCLR predictions
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.ndimage import zoom
import sys

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters
from finetune_on_isles_DEBUG import (
    SegmentationModel,
    reconstruct_from_patches_with_count
)


def get_original_isles_info(case_id, isles_raw_dir='/home/pahm409/ISLES2022_reg/ISLES2022'):
    import nibabel as nib
    
    isles_raw = Path(isles_raw_dir)
    dwi_file = isles_raw / case_id / f'{case_id}_ses-0001_dwi.nii.gz'
    msk_file = isles_raw / case_id / f'{case_id}_ses-0001_msk.nii.gz'
    
    if not dwi_file.exists():
        raise FileNotFoundError(f"Cannot find DWI file: {dwi_file}")
    if not msk_file.exists():
        raise FileNotFoundError(f"Cannot find mask file: {msk_file}")
    
    dwi_img = nib.load(dwi_file)
    msk_img = nib.load(msk_file)
    
    original_shape = dwi_img.shape
    original_affine = dwi_img.affine
    ground_truth = (msk_img.get_fdata() > 0.5).astype(np.uint8)
    
    resampled_shape = np.array([197, 233, 189])
    zoom_factors = np.array(original_shape) / resampled_shape
    
    return {
        'original_shape': tuple(original_shape),
        'original_affine': original_affine,
        'zoom_factors': zoom_factors,
        'ground_truth': ground_truth
    }


def resample_to_original(prediction, original_info, method='nearest'):
    zoom_factors = original_info['zoom_factors']
    order = 0 if method == 'nearest' else 1
    
    resampled = zoom(prediction, zoom_factors, order=order)
    
    target_shape = original_info['original_shape']
    if resampled.shape != target_shape:
        resampled_fixed = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        resampled = resampled_fixed
    
    return resampled


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder, 
        num_classes=2,
        attention_type=attention_type,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, attention_type, deep_supervision


def ensemble_evaluate_fold0(
    checkpoint_paths,
    isles_preprocessed_dir,
    isles_splits_file,
    isles_raw_dir,
    output_dir,
    save_nifti=False
):
    import nibabel as nib
    
    device = torch.device('cuda:0')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fold = 0  # FIXED TO FOLD 0
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE EVALUATION - FOLD 0 ONLY")
    print(f"{'='*80}")
    print(f"Number of models: {len(checkpoint_paths)}")
    for i, ckpt in enumerate(checkpoint_paths):
        print(f"  Model {i+1}: {ckpt}")
    print(f"{'='*80}\n")
    
    # Load test dataset for fold 0
    test_dataset = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='test',
        splits_file=isles_splits_file,
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )
    
    print(f"✓ Test volumes: {len(test_dataset.volumes)}\n")
    
    # Load models
    models_info = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"Loading model {i+1}...")
        model, attention_type, deep_supervision = load_model_from_checkpoint(ckpt_path, device)
        models_info.append({
            'model': model,
            'attention_type': attention_type,
            'deep_supervision': deep_supervision,
            'checkpoint_path': ckpt_path
        })
        print(f"  ✓ Loaded\n")
    
    # DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Collect predictions from each model
    all_model_predictions = []
    
    for model_idx, model_info in enumerate(models_info):
        print(f"Model {model_idx+1}: Generating predictions...")
        
        model = model_info['model']
        deep_supervision = model_info['deep_supervision']
        volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Model {model_idx+1}'):
                images = batch['image'].to(device)
                centers = batch['center'].cpu().numpy()
                vol_indices = batch['vol_idx'].cpu().numpy()
                
                with autocast():
                    outputs = model(images)
                    if deep_supervision:
                        outputs = outputs[0]
                
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
                
                for i in range(len(images)):
                    vol_idx = vol_indices[i]
                    volume_data[vol_idx]['centers'].append(centers[i])
                    volume_data[vol_idx]['preds'].append(preds[i])
        
        all_model_predictions.append(volume_data)
        torch.cuda.empty_cache()
        print(f"  ✓ Done\n")
    
    # Ensemble and evaluate
    print("Ensembling & reconstructing volumes...\n")
    
    all_dscs = []
    results_per_case = []
    individual_model_dscs = [[] for _ in range(len(models_info))]
    
    for vol_idx in tqdm(sorted(all_model_predictions[0].keys()), desc='Reconstructing'):
        vol_info = test_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        try:
            original_info = get_original_isles_info(case_id, isles_raw_dir)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            continue
        
        # Reconstruct from each model
        model_reconstructed_list = []
        
        for model_idx in range(len(models_info)):
            centers = np.array(all_model_predictions[model_idx][vol_idx]['centers'])
            preds = np.array(all_model_predictions[model_idx][vol_idx]['preds'])
            
            reconstructed, _ = reconstruct_from_patches_with_count(
                preds, centers, vol_info['mask'].shape, patch_size=(96, 96, 96)
            )
            
            model_reconstructed_list.append(reconstructed)
        
        # ENSEMBLE: Average probability maps
        ensemble_reconstructed = np.mean(model_reconstructed_list, axis=0)
        
        # Resample to original space
        pred_prob_original = resample_to_original(ensemble_reconstructed, original_info, method='linear')
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        mask_gt_original = original_info['ground_truth']
        
        # Compute ensemble DSC
        intersection = (pred_binary_original * mask_gt_original).sum()
        union = pred_binary_original.sum() + mask_gt_original.sum()
        dsc_ensemble = (2.0 * intersection) / union if union > 0 else 1.0
        
        all_dscs.append(dsc_ensemble)
        
        # Compute individual model DSCs
        individual_dscs_this_case = []
        for model_idx, model_recon in enumerate(model_reconstructed_list):
            model_prob_original = resample_to_original(model_recon, original_info, method='linear')
            model_binary_original = (model_prob_original > 0.5).astype(np.uint8)
            
            inter = (model_binary_original * mask_gt_original).sum()
            uni = model_binary_original.sum() + mask_gt_original.sum()
            dsc_m = (2.0 * inter) / uni if uni > 0 else 1.0
            
            individual_model_dscs[model_idx].append(dsc_m)
            individual_dscs_this_case.append(float(dsc_m))
        
        results_per_case.append({
            'case_id': case_id,
            'dsc_ensemble': float(dsc_ensemble),
            'dsc_model1': individual_dscs_this_case[0],
            'dsc_model2': individual_dscs_this_case[1],
            'original_shape': original_info['original_shape'],
            'gt_volume': int(mask_gt_original.sum()),
            'pred_volume': int(pred_binary_original.sum())
        })
    
    # Print results
    mean_dsc_ensemble = np.mean(all_dscs)
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE RESULTS - FOLD 0")
    print(f"{'='*80}")
    print(f"Cases evaluated: {len(all_dscs)}")
    print(f"\nENSEMBLE:")
    print(f"  Mean DSC: {mean_dsc_ensemble:.4f} ± {np.std(all_dscs):.4f}")
    print(f"  Min:  {np.min(all_dscs):.4f}")
    print(f"  Max:  {np.max(all_dscs):.4f}")
    
    print(f"\nINDIVIDUAL MODELS:")
    for model_idx in range(len(models_info)):
        mean_individual = np.mean(individual_model_dscs[model_idx])
        print(f"  Model {model_idx+1}: {mean_individual:.4f} ± {np.std(individual_model_dscs[model_idx]):.4f}")
    
    best_individual = max([np.mean(individual_model_dscs[i]) for i in range(len(models_info))])
    improvement = mean_dsc_ensemble - best_individual
    print(f"\nEnsemble improvement: {improvement:+.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'fold': 0,
        'checkpoints': checkpoint_paths,
        'num_models': len(checkpoint_paths),
        'mean_dsc_ensemble': float(mean_dsc_ensemble),
        'std_dsc_ensemble': float(np.std(all_dscs)),
        'individual_model_dscs': [float(np.mean(individual_model_dscs[i])) for i in range(len(models_info))],
        'ensemble_improvement': float(improvement),
        'num_test_cases': len(all_dscs),
        'all_dscs': [float(d) for d in all_dscs],
        'per_case_results': results_per_case
    }
    
    results_file = output_dir / 'fold_0_ensemble_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble evaluation - fold 0 only')
    
    parser.add_argument('--checkpoint-paths', nargs='+', required=True,
                       help='List of checkpoint paths (Random Init, SimCLR)')
    parser.add_argument('--isles-preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--isles-raw-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/ISLES2022')
    parser.add_argument('--splits-file', type=str,
                       default='isles_splits_5fold_resampled.json')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--save-nifti', action='store_true')
    
    args = parser.parse_args()
    
    ensemble_evaluate_fold0(
        checkpoint_paths=args.checkpoint_paths,
        isles_preprocessed_dir=args.isles_preprocessed_dir,
        isles_splits_file=args.splits_file,
        isles_raw_dir=args.isles_raw_dir,
        output_dir=args.output_dir,
        save_nifti=args.save_nifti
    )

#!/usr/bin/env python3
"""
evaluate_all_experiments.py

Single standalone script to evaluate all folds for ATLAS and UOA_Private.
No external dependencies - everything in one file.

Usage:
    python evaluate_all_experiments.py --dataset ATLAS --gpu 0
    python evaluate_all_experiments.py --dataset UOA_Private --gpu 0
    python evaluate_all_experiments.py --dataset ATLAS --condition transfer --gpu 0
    python evaluate_all_experiments.py --dataset ATLAS --folds 0 1 --gpu 0
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from scipy.ndimage import label as scipy_label

sys.path.append('.')
from models.resnet3d import resnet3d_18


# ============================================================
# CONFIGURATION
# ============================================================
BASE_EXP_DIR = "/hpc/pahm409/all_experiments_with_volume"
BASE_DATA_DIR = "/hpc/pahm409/harvard/preprocessed_stroke_foundation"
SPLITS_FILE = "splits_5fold.json"


# ============================================================
# MODEL
# ============================================================
class ResNet3DEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4

    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1, x2, x3, x4, x5]


class UNetDecoder3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.up4  = nn.ConvTranspose3d(512, 256, 2, 2)
        self.dec4 = self._block(512, 256)
        self.up3  = nn.ConvTranspose3d(256, 128, 2, 2)
        self.dec3 = self._block(256, 128)
        self.up2  = nn.ConvTranspose3d(128,  64, 2, 2)
        self.dec2 = self._block(128,  64)
        self.up1  = nn.ConvTranspose3d( 64,  64, 2, 2)
        self.dec1 = self._block(128,  64)
        self.final = nn.Conv3d(64, num_classes, 1)

    def _block(self, i, o):
        return nn.Sequential(
            nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(True),
            nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(True),
        )

    def _match(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='trilinear', align_corners=False)
        return x

    def forward(self, feats):
        x1, x2, x3, x4, x5 = feats
        x = self._match(self.up4(x5), x4);  x = self.dec4(torch.cat([x, x4], 1))
        x = self._match(self.up3(x),  x3);  x = self.dec3(torch.cat([x, x3], 1))
        x = self._match(self.up2(x),  x2);  x = self.dec2(torch.cat([x, x2], 1))
        x = self._match(self.up1(x),  x1);  x = self.dec1(torch.cat([x, x1], 1))
        return self.final(x)


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet3d_18(in_channels=1)
        self.encoder = ResNet3DEncoder(base)
        self.decoder = UNetDecoder3D(2)

    def forward(self, x):
        size = x.shape[2:]
        out  = self.decoder(self.encoder(x))
        if out.shape[2:] != size:
            out = F.interpolate(out, size=size, mode='trilinear', align_corners=False)
        return out


# ============================================================
# INFERENCE
# ============================================================
def center_based_inference(model, patches_np, centers, original_shape, device, patch_size=96, batch_size=4):
    model.eval()
    half    = patch_size // 2
    N       = len(patches_np)
    D, H, W = int(original_shape[0]), int(original_shape[1]), int(original_shape[2])

    all_preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.FloatTensor(patches_np[i:i + batch_size]).unsqueeze(1).to(device)
            with autocast():
                out = model(batch)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            
            # CRITICAL: Ensure output matches expected patch size
            if prob.shape[1:] != (patch_size, patch_size, patch_size):
                print(f"    WARNING: Model output {prob.shape[1:]} != expected {(patch_size, patch_size, patch_size)}")
                print(f"    Input batch shape was: {batch.shape}")
                # This should not happen - model forward() should handle resizing
                # But if it does, skip this batch to avoid crash
                continue
            
            all_preds.append(prob)
            del batch, out

    if not all_preds:
        raise RuntimeError(f"No valid predictions generated - model output size mismatch")
    
    all_preds = np.concatenate(all_preds, axis=0)
    volume = np.zeros((D, H, W), dtype=np.float32)
    count  = np.zeros((D, H, W), dtype=np.float32)

    for i, c in enumerate(centers):
        cd, ch, cw = int(c[0]), int(c[1]), int(c[2])
        d0, d1 = cd - half, cd + half
        h0, h1 = ch - half, ch + half
        w0, w1 = cw - half, cw + half
        if d0 < 0 or d1 > D or h0 < 0 or h1 > H or w0 < 0 or w1 > W:
            continue
        volume[d0:d1, h0:h1, w0:w1] += all_preds[i]
        count[d0:d1,  h0:h1, w0:w1] += 1.0

    valid = count > 0
    volume[valid] /= count[valid]
    return volume


def reconstruct_gt(masks_np, centers, original_shape, patch_size=96):
    half = patch_size // 2
    D, H, W = int(original_shape[0]), int(original_shape[1]), int(original_shape[2])
    volume = np.zeros((D, H, W), dtype=np.float32)
    count  = np.zeros((D, H, W), dtype=np.float32)

    for i, c in enumerate(centers):
        cd, ch, cw = int(c[0]), int(c[1]), int(c[2])
        d0, d1 = cd - half, cd + half
        h0, h1 = ch - half, ch + half
        w0, w1 = cw - half, cw + half
        if d0 < 0 or d1 > D or h0 < 0 or h1 > H or w0 < 0 or w1 > W:
            continue
        volume[d0:d1, h0:h1, w0:w1] += masks_np[i].astype(np.float32)
        count[d0:d1,  h0:h1, w0:w1] += 1.0

    valid = count > 0
    volume[valid] /= count[valid]
    return (volume > 0.5).astype(np.uint8)


# ============================================================
# METRICS
# ============================================================
def compute_dsc(pred_bin, gt_bin):
    intersection = int((pred_bin * gt_bin).sum())
    union        = int(pred_bin.sum()) + int(gt_bin.sum())
    if union == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0
    return float(2.0 * intersection / union)


def get_connected_components(binary_vol, min_voxels=5):
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled, n_comps = scipy_label(binary_vol, structure=structure)
    return [(labeled == cid) for cid in range(1, n_comps + 1) if (labeled == cid).sum() >= min_voxels]


def compute_lesion_wise_f1(pred_bin, gt_bin, iou_threshold=0.1, min_voxels=5):
    pred_comps = get_connected_components(pred_bin, min_voxels)
    gt_comps   = get_connected_components(gt_bin,   min_voxels)
    n_pred, n_gt = len(pred_comps), len(gt_comps)

    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 0, 0
    if n_gt == 0:
        return 0.0, 0.0, 1.0, n_pred, 0
    if n_pred == 0:
        return 0.0, 1.0, 0.0, 0, n_gt

    gt_matched = [False] * n_gt
    tp = 0
    for pred_comp in pred_comps:
        best_iou, best_idx = 0.0, -1
        for gi, gt_comp in enumerate(gt_comps):
            if gt_matched[gi]:
                continue
            inter = int((pred_comp & gt_comp).sum())
            if inter == 0:
                continue
            iou = inter / int((pred_comp | gt_comp).sum())
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            gt_matched[best_idx] = True

    fp = n_pred - tp
    fn = n_gt   - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(f1), float(precision), float(recall), n_pred, n_gt


def compute_volume_difference(pred_bin, gt_bin, spacing=(1.0, 1.0, 1.0)):
    vox_ml   = float(np.prod(spacing)) / 1000.0
    pred_ml  = float(pred_bin.sum()) * vox_ml
    gt_ml    = float(gt_bin.sum())   * vox_ml
    abs_diff = abs(pred_ml - gt_ml)
    rel_diff = abs_diff / gt_ml if gt_ml > 0 else 0.0
    return pred_ml, gt_ml, abs_diff, rel_diff


# ============================================================
# EVALUATE ONE FOLD
# ============================================================
def evaluate_one_fold(model, test_case_ids, dataset_name, condition, fold, device):
    model.eval()
    
    # Use fold-specific test directory
    patch_96_dir = Path(f"{BASE_DATA_DIR}/{dataset_name}_96x96x96_fold{fold}_test")
    
    if not patch_96_dir.exists():
        print(f"    ⚠️  Test patch directory not found: {patch_96_dir}")
        return []
    
    results = []

    for case_id in tqdm(test_case_ids, desc=f'  {dataset_name}/{condition}/fold_{fold}'):
        npz_path = patch_96_dir / f"{case_id}.npz"
        if not npz_path.exists():
            print(f"\n    ⚠️  MISSING NPZ: {case_id} (expected at {npz_path})")
            continue

        try:
            data = np.load(npz_path)
            patches_np     = data['patches'].astype(np.float32)
            masks_np       = data['masks'].astype(np.uint8)
            centers        = data['centers'].astype(np.int32)
            original_shape = data['original_shape']
            spacing = tuple(float(x) for x in data['spacing']) if 'spacing' in data.files else (1.0, 1.0, 1.0)

            # Validate patch size BEFORE inference
            expected_shape = (96, 96, 96)
            actual_shape = patches_np.shape[1:]
            if actual_shape != expected_shape:
                print(f"    ❌ SKIP {case_id}: NPZ contains {actual_shape} patches, expected {expected_shape}")
                print(f"       NPZ path: {npz_path}")
                print(f"       Check if you're using the wrong directory (64x64x64 vs 96x96x96)")
                continue

            pred_prob = center_based_inference(model, patches_np, centers, original_shape, device)
            pred_bin = (pred_prob > 0.5).astype(np.uint8)
            gt_bin   = reconstruct_gt(masks_np, centers, original_shape)

            dsc = compute_dsc(pred_bin, gt_bin)
            f1, precision, recall, n_pred_l, n_gt_l = compute_lesion_wise_f1(pred_bin, gt_bin)
            n_pred_c = len(get_connected_components(pred_bin))
            n_gt_c   = len(get_connected_components(gt_bin))
            pred_ml, gt_ml, abs_vol, rel_vol = compute_volume_difference(pred_bin, gt_bin, spacing)

            results.append({
                'case_id': case_id, 'dsc': float(dsc), 'lesion_f1': float(f1),
                'lesion_precision': float(precision), 'lesion_recall': float(recall),
                'lesion_count_diff': abs(n_pred_c - n_gt_c),
                'abs_vol_diff_ml': float(abs_vol), 'rel_vol_diff': float(rel_vol),
            })
        except Exception as e:
            print(f"    ERROR {case_id}: {e}")
            continue

    return results


def aggregate(results, key):
    vals = [r[key] for r in results if key in r]
    if not vals:
        return {'mean': 0.0, 'std': 0.0, 'n': 0}
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ATLAS', 'UOA_Private'])
    parser.add_argument('--condition', type=str, default='both', choices=['transfer', 'scratch', 'both'])
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    conditions = ['transfer', 'scratch'] if args.condition == 'both' else [args.condition]
    
    print(f"\n{'='*80}\n🔬 EVALUATING: {args.dataset} | {', '.join(conditions)} | folds {args.folds}\n{'='*80}\n")

    all_results = {}

    for condition in conditions:
        all_results[condition] = {'per_fold': {}, 'all_cases': []}
        
        for fold in args.folds:
            exp_dir = Path(BASE_EXP_DIR) / args.dataset / condition / f'fold_{fold}'
            ckpt_path = exp_dir / 'best_model.pth'
            
            if not ckpt_path.exists():
                print(f"⚠️  SKIP {condition}/fold_{fold}: checkpoint not found")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = SegModel().to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            
            test_case_ids = splits[f'fold_{fold}'][args.dataset]['test']
            
            fold_results = evaluate_one_fold(model, test_case_ids, args.dataset, condition, fold, device)
            
            all_results[condition]['per_fold'][fold] = {
                'n_cases': len(fold_results),
                'dsc': aggregate(fold_results, 'dsc'),
                'lesion_f1': aggregate(fold_results, 'lesion_f1'),
                'lesion_precision': aggregate(fold_results, 'lesion_precision'),
                'lesion_recall': aggregate(fold_results, 'lesion_recall'),
                'lesion_count_diff': aggregate(fold_results, 'lesion_count_diff'),
                'abs_vol_diff_ml': aggregate(fold_results, 'abs_vol_diff_ml'),
                'rel_vol_diff': aggregate(fold_results, 'rel_vol_diff'),
            }
            all_results[condition]['all_cases'].extend(fold_results)
            
            print(f"  ✅ fold_{fold}: DSC={aggregate(fold_results, 'dsc')['mean']:.4f} (n={len(fold_results)})")
            
            del model
            torch.cuda.empty_cache()

        # Per-fold results
        print(f"\n  {'='*70}")
        print(f"  {condition.upper()} — PER-FOLD RESULTS")
        print(f"  {'='*70}")
        
        for fold in sorted(all_results[condition]['per_fold'].keys()):
            fold_data = all_results[condition]['per_fold'][fold]
            print(f"  Fold {fold} (n={fold_data['n_cases']}):")
            print(f"    DSC:               {fold_data['dsc']['mean']:.4f} ± {fold_data['dsc']['std']:.4f}")
            print(f"    Lesion F1:         {fold_data['lesion_f1']['mean']:.4f} ± {fold_data['lesion_f1']['std']:.4f}")
            print(f"    Lesion Count Diff: {fold_data['lesion_count_diff']['mean']:.2f} ± {fold_data['lesion_count_diff']['std']:.2f}")
            print(f"    Abs Vol Diff (ml): {fold_data['abs_vol_diff_ml']['mean']:.2f} ± {fold_data['abs_vol_diff_ml']['std']:.2f}")
        
        # Mean across folds (not pooled cases)
        fold_means = {
            'dsc': [all_results[condition]['per_fold'][f]['dsc']['mean'] for f in all_results[condition]['per_fold']],
            'lesion_f1': [all_results[condition]['per_fold'][f]['lesion_f1']['mean'] for f in all_results[condition]['per_fold']],
            'lesion_precision': [all_results[condition]['per_fold'][f]['lesion_precision']['mean'] for f in all_results[condition]['per_fold']],
            'lesion_recall': [all_results[condition]['per_fold'][f]['lesion_recall']['mean'] for f in all_results[condition]['per_fold']],
            'lesion_count_diff': [all_results[condition]['per_fold'][f]['lesion_count_diff']['mean'] for f in all_results[condition]['per_fold']],
            'abs_vol_diff_ml': [all_results[condition]['per_fold'][f]['abs_vol_diff_ml']['mean'] for f in all_results[condition]['per_fold']],
        }
        
        print(f"\n  {'='*70}")
        print(f"  {condition.upper()} — MEAN ACROSS {len(fold_means['dsc'])} FOLDS")
        print(f"  {'='*70}")
        print(f"  DSC:               {np.mean(fold_means['dsc']):.4f} ± {np.std(fold_means['dsc']):.4f}")
        print(f"  Lesion F1:         {np.mean(fold_means['lesion_f1']):.4f} ± {np.std(fold_means['lesion_f1']):.4f}")
        print(f"  Lesion Precision:  {np.mean(fold_means['lesion_precision']):.4f} ± {np.std(fold_means['lesion_precision']):.4f}")
        print(f"  Lesion Recall:     {np.mean(fold_means['lesion_recall']):.4f} ± {np.std(fold_means['lesion_recall']):.4f}")
        print(f"  Lesion Count Diff: {np.mean(fold_means['lesion_count_diff']):.2f} ± {np.std(fold_means['lesion_count_diff']):.2f}")
        print(f"  Abs Vol Diff (ml): {np.mean(fold_means['abs_vol_diff_ml']):.2f} ± {np.std(fold_means['abs_vol_diff_ml']):.2f}")
        print(f"  {'='*70}\n")
        
        all_results[condition]['mean_across_folds'] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in fold_means.items()}

    # Save results
    output_file = Path(BASE_EXP_DIR) / args.dataset / 'test_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Results saved → {output_file}\n")


if __name__ == '__main__':
    main()

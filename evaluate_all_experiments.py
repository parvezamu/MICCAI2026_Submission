#!/usr/bin/env python3
"""
evaluate_all_experiments.py

Evaluates all folds for ATLAS and UOA_Private.

Metrics:
  - DSC (Dice Similarity Coefficient)
  - Volume Difference (absolute voxel count difference)
  - Lesion-wise F1 score (ATLAS standard: any single voxel overlap = match)
  - Lesion Count Difference (absolute difference in number of unconnected lesions)

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
BASE_EXP_DIR  = "/hpc/pahm409/all_experiments_with_volume"
BASE_DATA_DIR = "/hpc/pahm409/harvard/preprocessed_stroke_foundation"
SPLITS_FILE   = "splits_5fold.json"


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
def center_based_inference(model, patches_np, centers, original_shape,
                            device, patch_size=96, batch_size=4):
    model.eval()
    half    = patch_size // 2
    D, H, W = int(original_shape[0]), int(original_shape[1]), int(original_shape[2])

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(patches_np), batch_size):
            batch = torch.FloatTensor(patches_np[i:i + batch_size]).unsqueeze(1).to(device)
            with autocast():
                out = model(batch)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            if prob.shape[1:] != (patch_size, patch_size, patch_size):
                print(f"    WARNING: output shape {prob.shape[1:]} mismatch, skipping batch")
                continue
            all_preds.append(prob)
            del batch, out

    if not all_preds:
        raise RuntimeError("No valid predictions — model output size mismatch")

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
    """Voxel-wise Dice Similarity Coefficient."""
    intersection = int((pred_bin * gt_bin).sum())
    union        = int(pred_bin.sum()) + int(gt_bin.sum())
    if union == 0:
        return 1.0  # both empty = perfect
    return float(2.0 * intersection / union)


def compute_volume_difference(pred_bin, gt_bin):
    """
    Absolute difference in total number of predicted vs ground truth voxels.
    As per paper: |sum(prediction) - sum(ground_truth)|
    """
    return int(abs(int(pred_bin.sum()) - int(gt_bin.sum())))


def compute_lesion_wise_f1(pred_bin, gt_bin):
    """
    Lesion-wise F1 score using ATLAS benchmark standard:
      - TP: a GT lesion that has at least 1 voxel overlapping with prediction
      - FN: a GT lesion with no overlap
      - FP: a predicted lesion with no overlap with any GT lesion
    Returns f1, n_pred_lesions, n_gt_lesions
    """
    tp, fp, fn = 0, 0, 0

    labeled_gt, num_gt = scipy_label(gt_bin.astype(bool))
    for i in range(1, num_gt + 1):
        gt_lesion = (labeled_gt == i)
        if np.any(gt_lesion & pred_bin.astype(bool)):  # any voxel overlap
            tp += 1
        else:
            fn += 1

    labeled_pred, num_pred = scipy_label(pred_bin.astype(bool))
    for i in range(1, num_pred + 1):
        pred_lesion = (labeled_pred == i)
        if not np.any(pred_lesion & gt_bin.astype(bool)):  # no overlap
            fp += 1

    denom = tp + (fp + fn) / 2.0
    f1 = float(tp / denom) if denom > 0 else 1.0  # both empty = perfect

    return f1, int(num_pred), int(num_gt)


def compute_lesion_count_difference(pred_bin, gt_bin):
    """
    Absolute difference between number of unconnected predicted lesions
    and unconnected ground truth lesions.
    """
    _, num_pred = scipy_label(pred_bin.astype(bool))
    _, num_gt   = scipy_label(gt_bin.astype(bool))
    return int(abs(num_pred - num_gt))


# ============================================================
# EVALUATE ONE FOLD
# ============================================================
def evaluate_one_fold(model, test_case_ids, dataset_name, condition, fold, device):
    model.eval()
    patch_96_dir = Path(f"{BASE_DATA_DIR}/{dataset_name}_96x96x96_fold{fold}_test")

    if not patch_96_dir.exists():
        print(f"    ⚠️  Test patch directory not found: {patch_96_dir}")
        return []

    results = []

    for case_id in tqdm(test_case_ids, desc=f'  {dataset_name}/{condition}/fold_{fold}'):
        npz_path = patch_96_dir / f"{case_id}.npz"
        if not npz_path.exists():
            print(f"\n    ⚠️  MISSING: {case_id}")
            continue

        try:
            data           = np.load(npz_path)
            patches_np     = data['patches'].astype(np.float32)
            masks_np       = data['masks'].astype(np.uint8)
            centers        = data['centers'].astype(np.int32)
            original_shape = data['original_shape']

            if patches_np.shape[1:] != (96, 96, 96):
                print(f"    ❌ SKIP {case_id}: patch shape {patches_np.shape[1:]} != (96,96,96)")
                continue

            pred_prob = center_based_inference(model, patches_np, centers, original_shape, device)
            pred_bin  = (pred_prob > 0.5).astype(np.uint8)
            gt_bin    = reconstruct_gt(masks_np, centers, original_shape)

            # ---- 4 metrics only ----
            dsc       = compute_dsc(pred_bin, gt_bin)
            vol_diff  = compute_volume_difference(pred_bin, gt_bin)
            f1, n_pred_lesions, n_gt_lesions = compute_lesion_wise_f1(pred_bin, gt_bin)
            lc_diff   = compute_lesion_count_difference(pred_bin, gt_bin)

            results.append({
                'case_id':             case_id,
                'dsc':                 float(dsc),
                'volume_difference':   int(vol_diff),
                'lesion_f1':           float(f1),
                'lesion_count_diff':   int(lc_diff),
                'n_pred_lesions':      n_pred_lesions,
                'n_gt_lesions':        n_gt_lesions,
            })

        except Exception as e:
            print(f"    ERROR {case_id}: {e}")
            traceback.print_exc()
            continue

    return results


def aggregate(results, key):
    vals = [r[key] for r in results if key in r]
    if not vals:
        return {'mean': 0.0, 'std': 0.0, 'n': 0}
    return {
        'mean':   float(np.mean(vals)),
        'std':    float(np.std(vals)),
        'median': float(np.median(vals)),
        'n':      len(vals),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, required=True, choices=['ATLAS', 'UOA_Private'])
    parser.add_argument('--condition', type=str, default='both', choices=['transfer', 'scratch', 'both'])
    parser.add_argument('--folds',     type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--gpu',       type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    conditions = ['transfer', 'scratch'] if args.condition == 'both' else [args.condition]

    print(f"\n{'='*80}")
    print(f"EVALUATING: {args.dataset} | {', '.join(conditions)} | folds {args.folds}")
    print(f"METRICS: DSC | Volume Difference | Lesion F1 | Lesion Count Difference")
    print(f"{'='*80}\n")

    all_results = {}

    for condition in conditions:
        all_results[condition] = {'per_fold': {}, 'all_cases': []}

        for fold in args.folds:
            exp_dir   = Path(BASE_EXP_DIR) / args.dataset / condition / f'fold_{fold}'
            ckpt_path = exp_dir / 'best_model.pth'

            if not ckpt_path.exists():
                print(f"⚠️  SKIP {condition}/fold_{fold}: checkpoint not found")
                continue

            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = SegModel().to(device)
            model.load_state_dict(ckpt['model_state_dict'])

            test_case_ids = splits[f'fold_{fold}'][args.dataset]['test']
            fold_results  = evaluate_one_fold(model, test_case_ids, args.dataset, condition, fold, device)

            all_results[condition]['per_fold'][fold] = {
                'n_cases':           len(fold_results),
                'dsc':               aggregate(fold_results, 'dsc'),
                'volume_difference': aggregate(fold_results, 'volume_difference'),
                'lesion_f1':         aggregate(fold_results, 'lesion_f1'),
                'lesion_count_diff': aggregate(fold_results, 'lesion_count_diff'),
                'per_case':          fold_results,
            }
            all_results[condition]['all_cases'].extend(fold_results)

            print(f"  ✅ fold_{fold} (n={len(fold_results)}): "
                  f"DSC={aggregate(fold_results,'dsc')['mean']:.4f}  "
                  f"F1={aggregate(fold_results,'lesion_f1')['mean']:.4f}")

            del model
            torch.cuda.empty_cache()

        # ---- Per-fold summary ----
        print(f"\n  {'='*70}")
        print(f"  {condition.upper()} — PER-FOLD RESULTS")
        print(f"  {'='*70}")
        for fold in sorted(all_results[condition]['per_fold'].keys()):
            fd = all_results[condition]['per_fold'][fold]
            print(f"  Fold {fold} (n={fd['n_cases']}):")
            print(f"    DSC:                {fd['dsc']['mean']:.4f} ± {fd['dsc']['std']:.4f}")
            print(f"    Volume Difference:  {fd['volume_difference']['mean']:.1f} ± {fd['volume_difference']['std']:.1f}  (voxels)")
            print(f"    Lesion F1:          {fd['lesion_f1']['mean']:.4f} ± {fd['lesion_f1']['std']:.4f}")
            print(f"    Lesion Count Diff:  {fd['lesion_count_diff']['mean']:.2f} ± {fd['lesion_count_diff']['std']:.2f}")

        # ---- Mean across folds ----
        folds_done = sorted(all_results[condition]['per_fold'].keys())
        if folds_done:
            fold_means = {
                'dsc':               [all_results[condition]['per_fold'][f]['dsc']['mean']               for f in folds_done],
                'volume_difference': [all_results[condition]['per_fold'][f]['volume_difference']['mean'] for f in folds_done],
                'lesion_f1':         [all_results[condition]['per_fold'][f]['lesion_f1']['mean']         for f in folds_done],
                'lesion_count_diff': [all_results[condition]['per_fold'][f]['lesion_count_diff']['mean'] for f in folds_done],
            }

            print(f"\n  {'='*70}")
            print(f"  {condition.upper()} — MEAN ACROSS {len(folds_done)} FOLDS")
            print(f"  {'='*70}")
            print(f"  DSC:                {np.mean(fold_means['dsc']):.4f} ± {np.std(fold_means['dsc']):.4f}")
            print(f"  Volume Difference:  {np.mean(fold_means['volume_difference']):.1f} ± {np.std(fold_means['volume_difference']):.1f}  (voxels)")
            print(f"  Lesion F1:          {np.mean(fold_means['lesion_f1']):.4f} ± {np.std(fold_means['lesion_f1']):.4f}")
            print(f"  Lesion Count Diff:  {np.mean(fold_means['lesion_count_diff']):.2f} ± {np.std(fold_means['lesion_count_diff']):.2f}")
            print(f"  {'='*70}\n")

            all_results[condition]['mean_across_folds'] = {
                k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for k, v in fold_means.items()
            }

    # ---- Save ----
    output_file = Path(BASE_EXP_DIR) / args.dataset / 'test_evaluation_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Results saved → {output_file}\n")


if __name__ == '__main__':
    main()

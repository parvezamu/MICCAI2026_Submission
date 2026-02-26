"""
evaluate_uctransnet3d_stroke.py

Test-set evaluation for UCTransNet3D stroke segmentation experiments.
Evaluates all 4 conditions: ATLAS/UOA_Private × transfer/scratch
across 5 folds = 20 experiments total.

Uses center-based reconstruction from pre-extracted 64x64x64 NPZs.
No sliding window.

Verifies model architecture matches training before loading weights.

Usage:
    python evaluate_uctransnet3d_stroke.py \
        --exp-dir     /hpc/pahm409/uctransnet3d_stroke_experiments \
        --data-dir    /hpc/pahm409/harvard/preprocessed_stroke_foundation \
        --splits-file splits_5fold.json \
        --output-dir  /hpc/pahm409/uctransnet3d_stroke_test_results \
        --gpu 0
"""

import json
import math
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, LayerNorm
from torch.nn.modules.utils import _triple
from scipy.ndimage import label as scipy_label
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
from tqdm import tqdm


# ============================================================
# UCTransNet3D ARCHITECTURE
# Must be identical to finetune_uctransnet3d_stroke.py
# ============================================================
class UCTransNetConfig:
    def __init__(self, base_channel=64, norm='group'):
        self.base_channel = base_channel
        self.KV_size      = base_channel * (1 + 2 + 4 + 8)
        self.expand_ratio = 4
        self.norm         = norm
        self.transformer  = {
            "num_heads": 4, "num_layers": 4,
            "embeddings_dropout_rate": 0.1,
            "attention_dropout_rate":  0.1,
            "dropout_rate": 0.1,
        }


def _make_norm(norm, ch):
    if norm == 'batch':
        return nn.BatchNorm3d(ch)
    g = min(32, ch)
    while ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch)


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)


class Channel_Embeddings3D(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size   = _triple(img_size)
        patch_size = _triple(patchsize)
        n_patches  = ((img_size[0] // patch_size[0]) *
                      (img_size[1] // patch_size[1]) *
                      (img_size[2] // patch_size[2]))
        self.patch_embeddings    = nn.Conv3d(in_channels, in_channels,
                                             kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout             = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None: return None
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class Reconstruct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, norm):
        super().__init__()
        padding   = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = _make_norm(norm, out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.scale = scale_factor

    def forward(self, x):
        if x is None: return None
        b, n, c = x.size()
        d = h = w = int(round(n ** (1/3)))
        x = x.permute(0, 2, 1).contiguous().view(b, c, d, h, w)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return self.act(self.norm(self.conv(x)))


class Attention_org(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.KV_size   = config.KV_size
        self.num_heads = config.transformer["num_heads"]
        self.query = nn.ModuleList([
            nn.ModuleList([nn.Linear(ch, ch, bias=False) for _ in range(self.num_heads)])
            for ch in channel_num
        ])
        self.key   = nn.ModuleList([nn.Linear(self.KV_size, self.KV_size, bias=False)
                                    for _ in range(self.num_heads)])
        self.value = nn.ModuleList([nn.Linear(self.KV_size, self.KV_size, bias=False)
                                    for _ in range(self.num_heads)])
        self.out          = nn.ModuleList([nn.Linear(ch, ch, bias=False) for ch in channel_num])
        self.psi          = nn.InstanceNorm2d(self.num_heads)
        self.softmax      = Softmax(dim=3)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, embs, emb_all):
        multi_K = torch.stack([k(emb_all) for k in self.key],  dim=1)
        multi_V = torch.stack([v(emb_all) for v in self.value], dim=1)
        outputs = []
        for i, emb in enumerate(embs):
            if emb is None:
                outputs.append(None); continue
            Q      = torch.stack([self.query[i][h](emb)
                                   for h in range(self.num_heads)], dim=1).transpose(-1, -2)
            scores = torch.matmul(Q, multi_K) / math.sqrt(self.KV_size)
            probs  = self.attn_dropout(self.softmax(self.psi(scores)))
            ctx    = (torch.matmul(probs, multi_V.transpose(-1, -2))
                      .permute(0, 3, 2, 1).contiguous().mean(dim=3))
            outputs.append(self.proj_dropout(self.out[i](ctx)))
        return outputs


class Mlp(nn.Module):
    def __init__(self, config, in_ch, mlp_ch):
        super().__init__()
        self.fc1  = nn.Linear(in_ch, mlp_ch)
        self.fc2  = nn.Linear(mlp_ch, in_ch)
        self.act  = nn.GELU()
        self.drop = Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block_ViT(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.attn_norms    = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])
        self.attn_norm_all = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn  = Attention_org(config, channel_num)
        self.ffn_norms     = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])
        self.ffns          = nn.ModuleList([Mlp(config, c, c * config.expand_ratio)
                                            for c in channel_num])

    def forward(self, embs):
        emb_all  = self.attn_norm_all(torch.cat([e for e in embs if e is not None], dim=2))
        normed   = [self.attn_norms[i](embs[i]) if embs[i] is not None else None
                    for i in range(len(embs))]
        attn_out = self.channel_attn(normed, emb_all)
        embs     = [embs[i] + attn_out[i] if embs[i] is not None else None
                    for i in range(len(embs))]
        embs     = [embs[i] + self.ffns[i](self.ffn_norms[i](embs[i]))
                    if embs[i] is not None else None
                    for i in range(len(embs))]
        return embs


class Encoder(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.layers = nn.ModuleList([Block_ViT(config, channel_num)
                                     for _ in range(config.transformer["num_layers"])])
        self.norms  = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])

    def forward(self, embs):
        for layer in self.layers:
            embs = layer(embs)
        return [self.norms[i](embs[i]) if embs[i] is not None else None
                for i in range(len(embs))]


class ChannelTransformer3D(nn.Module):
    def __init__(self, config, img_size, channel_num, patch_sizes):
        super().__init__()
        self.embeddings  = nn.ModuleList([
            Channel_Embeddings3D(config, patch_sizes[i], img_size // (2**i), channel_num[i])
            for i in range(4)
        ])
        self.encoder      = Encoder(config, channel_num)
        self.reconstructs = nn.ModuleList([
            Reconstruct3D(channel_num[i], channel_num[i],
                          kernel_size=1, scale_factor=patch_sizes[i], norm=config.norm)
            for i in range(4)
        ])

    def forward(self, feats):
        embs = [self.embeddings[i](feats[i]) for i in range(4)]
        embs = self.encoder(embs)
        outs = []
        for i in range(4):
            if feats[i] is None:
                outs.append(None); continue
            r = self.reconstructs[i](embs[i])
            if r.shape[2:] != feats[i].shape[2:]:
                r = F.interpolate(r, size=feats[i].shape[2:], mode='trilinear', align_corners=False)
            outs.append(r + feats[i])
        return outs


class ConvBN3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), _make_norm(norm, out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


def make_nConv3D(in_ch, out_ch, n, norm):
    return nn.Sequential(*([ConvBN3D(in_ch, out_ch, norm)] +
                            [ConvBN3D(out_ch, out_ch, norm) for _ in range(n - 1)]))


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), make_nConv3D(in_ch, out_ch, 2, norm))
    def forward(self, x): return self.block(x)


class UpBlock_SE3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.se    = SEBlock3D(in_ch // 2)
        self.convs = make_nConv3D(in_ch, out_ch, 2, norm)

    def forward(self, x, skip):
        up   = self.up(x)
        skip = self.se(skip)
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.convs(torch.cat([skip, up], dim=1))


class UCTransNet3D(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=2, patch_size=64):
        super().__init__()
        B    = config.base_channel
        norm = config.norm
        self.inc   = make_nConv3D(n_channels, B,   2, norm)
        self.down1 = DownBlock3D(B,   B*2, norm)
        self.down2 = DownBlock3D(B*2, B*4, norm)
        self.down3 = DownBlock3D(B*4, B*8, norm)
        self.down4 = DownBlock3D(B*8, B*8, norm)
        self.mtc   = ChannelTransformer3D(config, patch_size,
                                          channel_num=[B, B*2, B*4, B*8],
                                          patch_sizes=[8, 4, 2, 1])
        self.up4      = UpBlock_SE3D(B*16, B*4, norm)
        self.up3      = UpBlock_SE3D(B*8,  B*2, norm)
        self.up2      = UpBlock_SE3D(B*4,  B,   norm)
        self.up1      = UpBlock_SE3D(B*2,  B,   norm)
        self.out_conv = nn.Conv3d(B, n_classes, kernel_size=1)

    def forward(self, x):
        sz = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1, x2, x3, x4 = self.mtc([x1, x2, x3, x4])
        x  = self.up4(x5, x4)
        x  = self.up3(x,  x3)
        x  = self.up2(x,  x2)
        x  = self.up1(x,  x1)
        out = self.out_conv(x)
        if out.shape[2:] != sz:
            out = F.interpolate(out, size=sz, mode='trilinear', align_corners=False)
        return out


# ============================================================
# MODEL LOADING — with architecture verification
# ============================================================
def detect_norm(state_dict):
    for k in state_dict:
        if 'running_mean' in k:
            return 'batch'
    return 'group'


def load_model(ckpt_path, device):
    """
    Load UCTransNet3D from checkpoint.
    Auto-detects norm type and patch_size from checkpoint metadata.
    Verifies weights load completely (missing=0, unexpected=0).
    """
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    norm         = detect_norm(state)
    patch_size   = ckpt.get('patch_size',   64)
    base_channel = ckpt.get('base_channel', 64)

    print(f"    norm={norm.upper()}Norm  patch_size={patch_size}  base_channel={base_channel}")
    print(f"    epoch={ckpt.get('epoch','?')}  "
          f"val_patch_dsc={ckpt.get('val_patch_dsc', 0.0):.4f}  "
          f"val_volume_dsc={ckpt.get('val_volume_dsc', 0.0):.4f}")

    config = UCTransNetConfig(base_channel=base_channel, norm=norm)
    model  = UCTransNet3D(config, n_channels=1, n_classes=2,
                          patch_size=patch_size).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Verify
    if missing:
        print(f"    ⚠ WARNING: {len(missing)} missing keys: {missing[:5]}")
    if unexpected:
        print(f"    ⚠ WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}")
    if not missing and not unexpected:
        print(f"    ✓ Weights loaded perfectly (missing=0, unexpected=0)")

    transformer_missing = [k for k in missing if 'mtc' in k]
    if transformer_missing:
        print(f"    ⚠ {len(transformer_missing)} transformer keys missing — transfer may be broken")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")

    return model, patch_size


# ============================================================
# NORMALISATION — must match finetune training
# ============================================================
# FIXED — nonzero-mask z-score to match training
def renorm_to_brats(patches_np):
    """
    NO-OP. Patches are already normalised (nonzero-mask z-score applied
    during volume preprocessing). Training pipeline never renormalised,
    so evaluation must not either.
    """
    return patches_np.astype(np.float32)

# ============================================================
# CENTER-BASED RECONSTRUCTION
# ============================================================
def reconstruct_volume(patch_preds, centers, original_shape, patch_size):
    """
    patch_preds    : (N, P, P, P) float32 probabilities
    centers        : (N, 3)       int32   midpoints
    original_shape : (3,)         int
    patch_size     : int
    """
    half    = patch_size // 2
    D, H, W = int(original_shape[0]), int(original_shape[1]), int(original_shape[2])
    volume  = np.zeros((D, H, W), dtype=np.float32)
    count   = np.zeros((D, H, W), dtype=np.float32)

    for i, c in enumerate(centers):
        cd, ch, cw = int(c[0]), int(c[1]), int(c[2])
        d0, d1 = cd - half, cd + half
        h0, h1 = ch - half, ch + half
        w0, w1 = cw - half, cw + half
        if d0 < 0 or d1 > D or h0 < 0 or h1 > H or w0 < 0 or w1 > W:
            continue
        volume[d0:d1, h0:h1, w0:w1] += patch_preds[i]
        count [d0:d1, h0:h1, w0:w1] += 1.0

    valid = count > 0
    volume[valid] /= count[valid]
    return volume


# ============================================================
# METRICS
# ============================================================
def compute_dsc(pred_bin, gt_bin):
    inter = int((pred_bin * gt_bin).sum())
    union = int(pred_bin.sum()) + int(gt_bin.sum())
    if union == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0
    return float(2.0 * inter / union)


def get_components(binary_vol, min_voxels=5):
    labeled, n = scipy_label(binary_vol, structure=np.ones((3,3,3), dtype=np.int32))
    return [(labeled == i) for i in range(1, n+1) if (labeled == i).sum() >= min_voxels]


def compute_lesion_f1(pred_bin, gt_bin, iou_thresh=0.1, min_voxels=5):
    pred_c = get_components(pred_bin, min_voxels)
    gt_c   = get_components(gt_bin,   min_voxels)
    np_, ng = len(pred_c), len(gt_c)

    if ng == 0 and np_ == 0: return 1.0, 1.0, 1.0, np_, ng
    if ng == 0:               return 0.0, 0.0, 1.0, np_, ng
    if np_ == 0:              return 0.0, 1.0, 0.0, np_, ng

    gt_matched = [False] * ng
    tp = 0
    for pc in pred_c:
        best_iou, best_idx = 0.0, -1
        for gi, gc in enumerate(gt_c):
            if gt_matched[gi]: continue
            inter = int((pc & gc).sum())
            if inter == 0: continue
            iou = inter / int((pc | gc).sum())
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1; gt_matched[best_idx] = True

    fp  = np_ - tp; fn = ng - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return float(f1), float(prec), float(rec), np_, ng


def aggregate(vals):
    if not vals:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'n': 0}
    return {
        'mean':   float(np.mean(vals)),
        'std':    float(np.std(vals)),
        'median': float(np.median(vals)),
        'min':    float(np.min(vals)),
        'max':    float(np.max(vals)),
        'n':      len(vals),
    }


# ============================================================
# EVALUATE ONE CONDITION (dataset × transfer_tag × all 5 folds)
# ============================================================
def evaluate_condition(dataset_name, tag, exp_dir, data_dir, splits,
                       device, patch_dir, patch_size,
                       threshold=0.5, inference_batch_size=8,
                       save_nifti=False, output_dir=None):
    """
    Evaluate one condition (e.g. ATLAS/transfer) across all 5 folds.
    Returns per-fold and aggregated results.
    """
    condition_dir = Path(exp_dir) / dataset_name / tag
    patch_dir     = Path(patch_dir)

    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} | {tag.upper()}")
    print(f"  Checkpoint dir: {condition_dir}")
    print(f"  Patch dir:      {patch_dir}")
    print(f"{'='*70}")

    all_case_results  = []
    fold_summaries    = []

    for fold in range(5):
        ckpt_path = condition_dir / f'fold_{fold}' / 'best_model.pth'
        print(f"\n  Fold {fold}")

        if not ckpt_path.exists():
            print(f"    SKIP: checkpoint not found at {ckpt_path}")
            continue

        # Load model — verify architecture
        model, ckpt_ps = load_model(ckpt_path, device)

        # Verify patch size consistency
        if ckpt_ps != patch_size:
            print(f"    ⚠ ckpt patch_size={ckpt_ps} != --patch-size={patch_size}")
            print(f"      Using ckpt patch_size={ckpt_ps} for this fold")
            effective_ps = ckpt_ps
        else:
            effective_ps = patch_size

        # Get test case IDs from splits
        fd = splits.get(f'fold_{fold}', {})
        if dataset_name in fd:
            test_ids = fd[dataset_name].get('test', [])
        else:
            test_ids = fd.get('test', [])

        if not test_ids:
            print(f"    SKIP: no test IDs found in splits for {dataset_name}/fold_{fold}")
            continue

        print(f"    Test cases: {len(test_ids)}")
        model.eval()

        fold_case_results = []
        skipped = 0

        for case_id in tqdm(test_ids, desc=f'    Fold {fold}', leave=False):
            npz = patch_dir / f"{case_id}.npz"
            if not npz.exists():
                skipped += 1
                continue

            try:
                data           = np.load(npz)
                patches_np     = data['patches'].astype(np.float32)
                masks_np       = data['masks'].astype(np.uint8)
                centers        = data['centers'].astype(np.int32)
                original_shape = data['original_shape']
                spacing        = (tuple(float(x) for x in data['spacing'])
                                  if 'spacing' in data.files else (1.0, 1.0, 1.0))

                # Verify NPZ patch size matches model
                actual_ps = patches_np.shape[1]
                if actual_ps != effective_ps:
                    print(f"\n    ⚠ SKIP {case_id}: NPZ patch_size={actual_ps} "
                          f"!= model patch_size={effective_ps}")
                    skipped += 1
                    continue

                # Renormalise — must match training
                patches_norm = renorm_to_brats(patches_np)

                # Inference
                all_preds = []
                with torch.no_grad():
                    for i in range(0, len(patches_norm), inference_batch_size):
                        batch_t = (torch.FloatTensor(patches_norm[i:i+inference_batch_size])
                                   .unsqueeze(1).to(device))
                        probs = torch.softmax(model(batch_t).float(), dim=1)[:, 1].cpu().numpy()
                        all_preds.append(probs)
                        del batch_t
                all_preds = np.concatenate(all_preds, axis=0)

                # Reconstruct volumes
                pred_vol = reconstruct_volume(all_preds, centers, original_shape, effective_ps)
                gt_vol   = reconstruct_volume(masks_np.astype(np.float32),
                                              centers, original_shape, effective_ps)
                pred_bin = (pred_vol > threshold).astype(np.uint8)
                gt_bin   = (gt_vol   > 0.5      ).astype(np.uint8)

                # Metrics
                dsc              = compute_dsc(pred_bin, gt_bin)
                f1, prec, rec, np_, ng = compute_lesion_f1(pred_bin, gt_bin)
                vox_ml           = float(np.prod(spacing)) / 1000.0
                pred_ml          = float(pred_bin.sum()) * vox_ml
                gt_ml            = float(gt_bin.sum())   * vox_ml
                abs_vol          = abs(pred_ml - gt_ml)
                rel_vol          = abs_vol / gt_ml if gt_ml > 0 else 0.0

                case_result = {
                    'case_id':          case_id,
                    'fold':             fold,
                    'dsc':              float(dsc),
                    'lesion_f1':        float(f1),
                    'lesion_precision': float(prec),
                    'lesion_recall':    float(rec),
                    'n_pred_lesions':   int(np_),
                    'n_gt_lesions':     int(ng),
                    'pred_vol_ml':      float(pred_ml),
                    'gt_vol_ml':        float(gt_ml),
                    'abs_vol_diff_ml':  float(abs_vol),
                    'rel_vol_diff':     float(rel_vol),
                    'n_patches':        int(len(patches_np)),
                    'gt_voxels':        int(gt_bin.sum()),
                    'pred_voxels':      int(pred_bin.sum()),
                }
                fold_case_results.append(case_result)
                all_case_results.append(case_result)

                # Save NIfTI
                if save_nifti and HAS_NIBABEL and output_dir:
                    sd = Path(output_dir) / dataset_name / tag / f'fold_{fold}' / case_id
                    sd.mkdir(parents=True, exist_ok=True)
                    aff = np.eye(4)
                    nib.save(nib.Nifti1Image(pred_bin, aff),  sd / 'pred.nii.gz')
                    nib.save(nib.Nifti1Image(pred_vol, aff),  sd / 'pred_prob.nii.gz')
                    nib.save(nib.Nifti1Image(gt_bin,   aff),  sd / 'gt.nii.gz')

            except Exception as e:
                print(f"\n    ERROR {case_id}: {e}")
                traceback.print_exc()
                skipped += 1

        if skipped:
            print(f"    Skipped: {skipped} cases")

        # Fold summary
        if fold_case_results:
            dscs = [r['dsc'] for r in fold_case_results]
            f1s  = [r['lesion_f1'] for r in fold_case_results]
            fold_sum = {
                'fold':    fold,
                'n_cases': len(fold_case_results),
                'dsc':     aggregate(dscs),
                'lesion_f1': aggregate(f1s),
                'per_case':  fold_case_results,
            }
            fold_summaries.append(fold_sum)
            print(f"    DSC: {np.mean(dscs):.4f} ± {np.std(dscs):.4f}  "
                  f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (n={len(dscs)})")

        del model
        torch.cuda.empty_cache()

    # Condition-level aggregate
    all_dscs = [r['dsc']        for r in all_case_results]
    all_f1s  = [r['lesion_f1']  for r in all_case_results]
    all_prec = [r['lesion_precision'] for r in all_case_results]
    all_rec  = [r['lesion_recall']    for r in all_case_results]
    all_vol  = [r['abs_vol_diff_ml']  for r in all_case_results]

    condition_summary = {
        'dataset':    dataset_name,
        'condition':  tag,
        'n_folds':    len(fold_summaries),
        'n_cases':    len(all_case_results),
        'dsc':        aggregate(all_dscs),
        'lesion_f1':  aggregate(all_f1s),
        'lesion_precision': aggregate(all_prec),
        'lesion_recall':    aggregate(all_rec),
        'abs_vol_diff_ml':  aggregate(all_vol),
        'per_fold':   fold_summaries,
    }

    print(f"\n  {dataset_name} {tag} — OVERALL ({len(all_case_results)} cases):")
    print(f"    DSC:       {aggregate(all_dscs)['mean']:.4f} ± {aggregate(all_dscs)['std']:.4f}")
    print(f"    Lesion F1: {aggregate(all_f1s)['mean']:.4f} ± {aggregate(all_f1s)['std']:.4f}")
    per_fold_dsc_str = [f"{fs['dsc']['mean']:.3f}" for fs in fold_summaries]
    print("    Per-fold DSC:", per_fold_dsc_str)

    return condition_summary


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir',      type=str, required=True,
                        help='Root experiment dir (contains ATLAS/transfer/fold_X/best_model.pth)')
    parser.add_argument('--data-dir',     type=str, required=True,
                        help='preprocessed_stroke_foundation root')
    parser.add_argument('--splits-file',  type=str, default='splits_5fold.json')
    parser.add_argument('--output-dir',   type=str, required=True)
    parser.add_argument('--patch-size',   type=int, default=64,
                        help='Must match training patch size (default: 64)')
    parser.add_argument('--patch-dir-atlas', type=str, default=None,
                        help='Pre-extracted 64^3 NPZs for ATLAS (default: data-dir/ATLAS_64x64x64)')
    parser.add_argument('--patch-dir-uoa',   type=str, default=None,
                        help='Pre-extracted 64^3 NPZs for UOA_Private (default: data-dir/UOA_Private_64x64x64)')
    parser.add_argument('--threshold',       type=float, default=0.5)
    parser.add_argument('--inference-batch', type=int,   default=8)
    parser.add_argument('--gpu',             type=int,   default=0)
    parser.add_argument('--save-nifti',      action='store_true')
    parser.add_argument('--datasets',        nargs='+',  default=['ATLAS', 'UOA_Private'])
    parser.add_argument('--conditions',      nargs='+',  default=['transfer', 'scratch'])
    args = parser.parse_args()

    device     = torch.device(f'cuda:{args.gpu}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ps  = args.patch_size
    ps_str = f"{ps}x{ps}x{ps}"

    patch_dirs = {
        'ATLAS':       Path(args.patch_dir_atlas) if args.patch_dir_atlas
                       else Path(args.data_dir) / f'ATLAS_{ps_str}',
        'UOA_Private': Path(args.patch_dir_uoa)   if args.patch_dir_uoa
                       else Path(args.data_dir) / f'UOA_Private_{ps_str}',
    }

    print(f"\n{'#'*70}")
    print(f"# UCTransNet3D — TEST SET EVALUATION")
    print(f"# patch_size={ps}  threshold={args.threshold}")
    print(f"# exp_dir:    {args.exp_dir}")
    print(f"# output_dir: {args.output_dir}")
    print(f"# datasets:   {args.datasets}")
    print(f"# conditions: {args.conditions}")
    for ds, pd in patch_dirs.items():
        if ds in args.datasets:
            exists = pd.exists()
            print(f"# {ds} patches: {pd}  {'✓' if exists else '✗ NOT FOUND'}")
    print(f"{'#'*70}\n")

    with open(args.splits_file) as f:
        splits = json.load(f)

    all_summaries = {}

    for dataset_name in args.datasets:
        pd = patch_dirs[dataset_name]
        if not pd.exists():
            print(f"⚠ Patch dir not found for {dataset_name}: {pd}")
            print(f"  Run: python extract_patches_64x64x64.py --datasets {dataset_name} --patch_size {ps}")
            continue

        for tag in args.conditions:
            key = f"{dataset_name}_{tag}"
            summary = evaluate_condition(
                dataset_name         = dataset_name,
                tag                  = tag,
                exp_dir              = args.exp_dir,
                data_dir             = args.data_dir,
                splits               = splits,
                device               = device,
                patch_dir            = pd,
                patch_size           = ps,
                threshold            = args.threshold,
                inference_batch_size = args.inference_batch,
                save_nifti           = args.save_nifti,
                output_dir           = args.output_dir if args.save_nifti else None,
            )
            all_summaries[key] = summary

            # Save per-condition JSON
            cond_path = output_dir / f"{key}_test_results.json"
            with open(cond_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  ✓ Saved → {cond_path}")

    # ---- Final comparison table ----
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Condition':<30} {'DSC':>10} {'±':>6} {'Lesion F1':>10} {'±':>6} {'N':>5}")
    print(f"{'-'*70}")

    for dataset_name in args.datasets:
        for tag in args.conditions:
            key = f"{dataset_name}_{tag}"
            if key not in all_summaries:
                continue
            s    = all_summaries[key]
            dsc  = s['dsc']
            f1   = s['lesion_f1']
            label = f"{dataset_name}/{tag}"
            print(f"  {label:<28} {dsc['mean']:>10.4f} {dsc['std']:>6.4f} "
                  f"{f1['mean']:>10.4f} {f1['std']:>6.4f} {s['n_cases']:>5}")

    # Transfer vs scratch comparison
    print(f"\n{'='*70}")
    print("TRANSFER BENEFIT (transfer DSC - scratch DSC)")
    print(f"{'='*70}")
    for dataset_name in args.datasets:
        t_key = f"{dataset_name}_transfer"
        s_key = f"{dataset_name}_scratch"
        if t_key in all_summaries and s_key in all_summaries:
            t_dsc = all_summaries[t_key]['dsc']['mean']
            s_dsc = all_summaries[s_key]['dsc']['mean']
            delta = t_dsc - s_dsc
            print(f"  {dataset_name}: transfer={t_dsc:.4f}  scratch={s_dsc:.4f}  "
                  f"delta={delta:+.4f}  {'↑ transfer wins' if delta > 0 else '↓ scratch wins'}")

    # Save full summary
    final_path = output_dir / 'test_summary_all.json'
    with open(final_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Full summary → {final_path}")


if __name__ == '__main__':
    main()

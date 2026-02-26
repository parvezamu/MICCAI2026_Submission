"""
finetune_uctransnet3d_stroke.py

Fine-tune UCTransNet3D on stroke segmentation datasets (ATLAS, UOA_Private).
Mirrors train_all_experiments_with_volume_validation.py but uses UCTransNet3D
instead of ResNet-18 + UNetDecoder.

Two arms per dataset per fold:
  - TRANSFER: load BraTS-pretrained UCTransNet3D checkpoint, fine-tune
  - SCRATCH:  random init, train from scratch

Total: 2 datasets × 2 conditions × 5 folds = 20 experiments

Usage:
    # All 20 experiments sequentially
    python finetune_uctransnet3d_stroke.py --run-all

    # All 20, skip already-done folds
    python finetune_uctransnet3d_stroke.py --run-all --resume

    # Single experiment
    python finetune_uctransnet3d_stroke.py --dataset ATLAS --fold 0 --transfer
    python finetune_uctransnet3d_stroke.py --dataset UOA_Private --fold 2
"""

import os
import math
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Dropout, Softmax, LayerNorm
from torch.nn.modules.utils import _triple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

import sys
sys.path.append('.')
try:
    from dataset.patch_dataset_with_centers import PatchDatasetWithCenters
    HAS_PATCH_DS = True
except ImportError:
    HAS_PATCH_DS = False
    print("WARNING: PatchDatasetWithCenters not found — using fallback dataset")


# ============================================================
# FALLBACK DATASET
# ============================================================
class FallbackStrokeDataset(Dataset):
    """Minimal NPZ-based patch dataset. Used if PatchDatasetWithCenters unavailable."""
    def __init__(self, preprocessed_dir, dataset_name, case_ids,
                 patch_size=(96,96,96), patches_per_volume=10,
                 augment=False, lesion_focus_ratio=0.7):
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.lesion_focus_ratio = lesion_focus_ratio
        self.volumes = []
        data_dir = Path(preprocessed_dir) / dataset_name
        for cid in case_ids:
            p = data_dir / f"{cid}.npz"
            if p.exists():
                self.volumes.append({'case_id': cid, 'path': p})
        if not self.volumes:
            raise ValueError(f"No volumes found in {data_dir}")
        print(f"  Loaded {len(self.volumes)} volumes ({dataset_name})")

    def __len__(self):
        return len(self.volumes) * self.patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        data    = np.load(self.volumes[vol_idx]['path'])
        image   = data['image'].astype(np.float32)
        mask    = (data['lesion_mask'] if 'lesion_mask' in data.files else data['mask']).astype(np.uint8)

        D, H, W = image.shape
        pd, ph, pw = self.patch_size
        lesion_voxels = np.argwhere(mask > 0)
        use_lesion    = (np.random.rand() < self.lesion_focus_ratio) and len(lesion_voxels) > 0

        if use_lesion:
            c  = lesion_voxels[np.random.randint(len(lesion_voxels))]
            d0 = int(np.clip(c[0] - pd//2, 0, max(0, D-pd)))
            h0 = int(np.clip(c[1] - ph//2, 0, max(0, H-ph)))
            w0 = int(np.clip(c[2] - pw//2, 0, max(0, W-pw)))
        else:
            d0 = np.random.randint(0, max(1, D-pd+1))
            h0 = np.random.randint(0, max(1, H-ph+1))
            w0 = np.random.randint(0, max(1, W-pw+1))

        ip = image[d0:d0+pd, h0:h0+ph, w0:w0+pw]
        mp = mask [d0:d0+pd, h0:h0+ph, w0:w0+pw]

        if ip.shape != tuple(self.patch_size):
            pads = [(0, max(0, t-s)) for s, t in zip(ip.shape, self.patch_size)]
            ip   = np.pad(ip, pads, mode='constant')
            mp   = np.pad(mp, pads, mode='constant')

        if self.augment:
            for ax in range(3):
                if np.random.rand() > 0.5:
                    ip = np.flip(ip, axis=ax).copy()
                    mp = np.flip(mp, axis=ax).copy()

        return {
            'image': torch.from_numpy(ip).unsqueeze(0).float(),
            'mask':  torch.from_numpy(mp).long(),
        }


# ============================================================
# UCTransNet3D (identical to train_brats_uctransnet3d.py)
# ============================================================
class UCTransNetConfig:
    def __init__(self, base_channel=64, norm='group'):
        self.base_channel = base_channel
        self.KV_size      = base_channel * (1 + 2 + 4 + 8)  # 960
        self.expand_ratio = 4
        self.norm         = norm  # 'group' or 'batch'
        self.transformer  = {
            "num_heads": 4, "num_layers": 4,
            "embeddings_dropout_rate": 0.1,
            "attention_dropout_rate":  0.1,
            "dropout_rate": 0.1,
        }


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
        n_patches  = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1]) * (img_size[2]//patch_size[2])
        self.patch_embeddings    = nn.Conv3d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout             = Dropout(config.transformer["embeddings_dropout_rate"])
    def forward(self, x):
        if x is None: return None
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class Reconstruct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, norm='group'):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        if norm == 'batch':
            norm_layer = nn.BatchNorm3d(out_channels)
        else:
            num_groups = min(32, out_channels)
            while out_channels % num_groups != 0:
                num_groups //= 2
            norm_layer = nn.GroupNorm(num_groups, out_channels)
        self.conv  = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm  = norm_layer
        self.act   = nn.ReLU(inplace=True)
        self.scale = scale_factor
    def forward(self, x):
        if x is None: return None
        b, n, c = x.size()
        d = h = w = int(round(n ** (1/3)))
        x = x.permute(0,2,1).contiguous().view(b, c, d, h, w)
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
        self.key   = nn.ModuleList([nn.Linear(self.KV_size, self.KV_size, bias=False) for _ in range(self.num_heads)])
        self.value = nn.ModuleList([nn.Linear(self.KV_size, self.KV_size, bias=False) for _ in range(self.num_heads)])
        self.out   = nn.ModuleList([nn.Linear(ch, ch, bias=False) for ch in channel_num])
        self.psi          = nn.InstanceNorm2d(self.num_heads)
        self.softmax      = Softmax(dim=3)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, embs, emb_all):
        multi_K = torch.stack([k(emb_all) for k in self.key],  dim=1)
        multi_V = torch.stack([v(emb_all) for v in self.value], dim=1)
        outputs = []
        for i, emb in enumerate(embs):
            if emb is None: outputs.append(None); continue
            Q      = torch.stack([self.query[i][h](emb) for h in range(self.num_heads)], dim=1).transpose(-1, -2)
            scores = torch.matmul(Q, multi_K) / math.sqrt(self.KV_size)
            probs  = self.attn_dropout(self.softmax(self.psi(scores)))
            ctx    = torch.matmul(probs, multi_V.transpose(-1,-2)).permute(0,3,2,1).contiguous().mean(dim=3)
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
        self.ffns          = nn.ModuleList([Mlp(config, c, c * config.expand_ratio) for c in channel_num])

    def forward(self, embs):
        emb_all  = self.attn_norm_all(torch.cat([e for e in embs if e is not None], dim=2))
        normed   = [self.attn_norms[i](embs[i]) if embs[i] is not None else None for i in range(len(embs))]
        attn_out = self.channel_attn(normed, emb_all)
        embs     = [embs[i] + attn_out[i] if embs[i] is not None else None for i in range(len(embs))]
        embs     = [embs[i] + self.ffns[i](self.ffn_norms[i](embs[i])) if embs[i] is not None else None for i in range(len(embs))]
        return embs


class Encoder(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.layers = nn.ModuleList([Block_ViT(config, channel_num) for _ in range(config.transformer["num_layers"])])
        self.norms  = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])
    def forward(self, embs):
        for layer in self.layers:
            embs = layer(embs)
        return [self.norms[i](embs[i]) if embs[i] is not None else None for i in range(len(embs))]


class ChannelTransformer3D(nn.Module):
    def __init__(self, config, img_size, channel_num, patch_sizes):
        super().__init__()
        self.embeddings  = nn.ModuleList([
            Channel_Embeddings3D(config, patch_sizes[i], img_size//(2**i), channel_num[i])
            for i in range(4)
        ])
        self.encoder      = Encoder(config, channel_num)
        self.reconstructs = nn.ModuleList([
            Reconstruct3D(channel_num[i], channel_num[i], kernel_size=1,
                          scale_factor=patch_sizes[i], norm=getattr(config, 'norm', 'group'))
            for i in range(4)
        ])

    def forward(self, feats):
        embs = [self.embeddings[i](feats[i]) for i in range(4)]
        embs = self.encoder(embs)
        outs = []
        for i in range(4):
            if feats[i] is None: outs.append(None); continue
            r = self.reconstructs[i](embs[i])
            if r.shape[2:] != feats[i].shape[2:]:
                r = F.interpolate(r, size=feats[i].shape[2:], mode='trilinear', align_corners=False)
            outs.append(r + feats[i])
        return outs


def _gn(ch):
    g = min(32, ch)
    while ch % g != 0: g //= 2
    return g


class ConvBN3D(nn.Module):
    def __init__(self, in_ch, out_ch, act='ReLU', norm='group'):
        super().__init__()
        if norm == 'batch':
            norm_layer = nn.BatchNorm3d(out_ch)
        else:
            norm_layer = nn.GroupNorm(_gn(out_ch), out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            norm_layer,
            getattr(nn, act)(),
        )
    def forward(self, x): return self.block(x)


def make_nConv3D(in_ch, out_ch, n, norm='group'):
    layers = [ConvBN3D(in_ch, out_ch, norm=norm)]
    for _ in range(n-1): layers.append(ConvBN3D(out_ch, out_ch, norm=norm))
    return nn.Sequential(*layers)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group'):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), make_nConv3D(in_ch, out_ch, 2, norm=norm))
    def forward(self, x): return self.block(x)


class UpBlock_SE3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group'):
        super().__init__()
        skip_ch    = in_ch // 2
        self.up    = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.se    = SEBlock3D(skip_ch)
        self.convs = make_nConv3D(in_ch, out_ch, 2, norm=norm)
    def forward(self, x, skip):
        up = self.up(x)
        skip = self.se(skip)
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.convs(torch.cat([skip, up], dim=1))


class UCTransNet3D(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=2, img_size=96):
        super().__init__()
        B    = config.base_channel
        norm = getattr(config, 'norm', 'group')
        self.inc   = make_nConv3D(n_channels, B, 2, norm=norm)
        self.down1 = DownBlock3D(B,   B*2, norm=norm)
        self.down2 = DownBlock3D(B*2, B*4, norm=norm)
        self.down3 = DownBlock3D(B*4, B*8, norm=norm)
        self.down4 = DownBlock3D(B*8, B*8, norm=norm)
        self.mtc   = ChannelTransformer3D(config, img_size,
                                          channel_num=[B, B*2, B*4, B*8],
                                          patch_sizes=[8, 4, 2, 1])
        self.up4   = UpBlock_SE3D(B*16, B*4, norm=norm)
        self.up3   = UpBlock_SE3D(B*8,  B*2, norm=norm)
        self.up2   = UpBlock_SE3D(B*4,  B,   norm=norm)
        self.up1   = UpBlock_SE3D(B*2,  B,   norm=norm)
        self.out_conv = nn.Conv3d(B, n_classes, 1)

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


def detect_norm_type(ckpt_state):
    """Auto-detect whether checkpoint used BatchNorm or GroupNorm."""
    for k in ckpt_state:
        if 'running_mean' in k:
            return 'batch'
    return 'group'


def build_model(cfg, device, pretrained_path=None):
    """
    Build UCTransNet3D, auto-matching norm type to checkpoint if transfer.
    Returns model with weights loaded.
    """
    # Determine norm type
    norm_type = 'group'  # default
    if pretrained_path and Path(pretrained_path).exists():
        ckpt  = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        norm_type = detect_norm_type(state)
        print(f"  Checkpoint norm type detected: {norm_type.upper()}Norm")

    config = UCTransNetConfig(base_channel=cfg['base_channel'], norm=norm_type)
    model  = UCTransNet3D(config, n_channels=1, n_classes=2,
                          img_size=cfg['patch_size']).to(device)

    if pretrained_path and Path(pretrained_path).exists():
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  ✓ BraTS weights loaded — missing={len(missing)}, unexpected={len(unexpected)}")
        if unexpected:
            print(f"  Unexpected keys (first 5): {unexpected[:5]}")
        transformer_missing = [k for k in missing if 'mtc' in k or 'position_embed' in k]
        if transformer_missing:
            print(f"  ⚠ {len(transformer_missing)} transformer keys missing: {transformer_missing[:3]}")
        else:
            print(f"  ✓ All transformer (mtc) weights transferred")
    else:
        print(f"  ✓ Random init ({norm_type.upper()}Norm)")

    return model


# ============================================================
# LOSS + METRICS
# ============================================================
class GDiceLossV2(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        logits = logits.float()
        prob   = torch.softmax(logits, dim=1)
        if logits.shape[2:] != target.shape[1:]:
            target = F.interpolate(target.unsqueeze(1).float(),
                                   size=logits.shape[2:], mode='nearest').squeeze(1).long()
        oh = torch.zeros_like(prob)
        oh.scatter_(1, target.unsqueeze(1), 1)
        pf = prob.reshape(prob.size(0), prob.size(1), -1)
        tf = oh.reshape(oh.size(0), oh.size(1), -1)
        inter = (pf * tf).sum(-1)
        union = pf.sum(-1) + tf.sum(-1)
        w     = 1.0 / (tf.sum(-1) ** 2 + self.smooth)
        dice  = (2. * w * inter + self.smooth) / (w * union + self.smooth)
        loss  = 1. - dice.mean()
        return torch.tensor(0.0, device=logits.device, requires_grad=True) if torch.isnan(loss) else loss


def compute_dsc(pred_prob, target):
    if pred_prob.shape != target.shape:
        target = F.interpolate(target.unsqueeze(1).float(),
                               size=pred_prob.shape[1:], mode='nearest').squeeze(1)
    pred_b = (pred_prob > 0.5).float()
    tgt_b  = (target > 0).float()
    inter  = (pred_b * tgt_b).sum()
    union  = pred_b.sum() + tgt_b.sum()
    return 1.0 if union == 0 else (2. * inter / union).item()


# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss = tot_dsc = n = 0
    pbar = tqdm(loader, desc='  Train', leave=False)
    for batch in pbar:
        imgs  = batch['image'].to(device)
        masks = batch['mask'].to(device)
        out   = model(imgs)
        loss  = criterion(out, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            dsc = compute_dsc(torch.softmax(out, dim=1)[:, 1], masks)
        tot_loss += loss.item(); tot_dsc += dsc; n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}', dsc=f'{dsc:.4f}')

    return (tot_loss/n, tot_dsc/n) if n > 0 else (float('nan'), 0.0)


def validate_patches(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_dsc = n_loss = n_total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Val  ', leave=False):
            imgs  = batch['image'].to(device)
            masks = batch['mask'].to(device)
            out   = model(imgs)
            loss  = criterion(out, masks)
            dsc   = compute_dsc(torch.softmax(out, dim=1)[:, 1], masks)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                tot_loss += loss.item(); n_loss += 1
            tot_dsc += dsc; n_total += 1
    return (tot_loss/n_loss if n_loss > 0 else float('nan')), tot_dsc/n_total


# ============================================================
# CENTER-BASED VOLUME VALIDATION (fast — uses pre-extracted patches)
# ============================================================
def reconstruct_volume_from_patches(patch_preds, centers, original_shape, patch_size=96):
    """
    Reconstruct full volume by placing patches at their center coordinates.
    Overlapping regions are averaged.
    patch_preds: (N, P, P, P) float32 probabilities
    centers:     (N, 3)       int32   [d, h, w] midpoints
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


def validate_volumes(model, val_ids, patch_96_dir, device,
                     patch_size=96, inference_batch_size=4,
                     max_cases=None, save_dir=None):
    """
    Fast center-based volume validation using pre-extracted patch NPZs.
    Each NPZ contains: patches (N,P,P,P), masks (N,P,P,P),
                       centers (N,3), original_shape (3,)
    """
    model.eval()
    patch_96_dir = Path(patch_96_dir)
    ids  = val_ids[:max_cases] if max_cases else val_ids
    dscs = []

    print(f"\n  Volume-level validation on {len(ids)} cases...")

    with torch.no_grad():
        for case_id in ids:
            npz = patch_96_dir / f"{case_id}.npz"
            if not npz.exists():
                print(f"    SKIP {case_id}: not found at {npz}")
                continue
            try:
                data           = np.load(npz)
                patches_np     = data['patches'].astype(np.float32)  # (N,P,P,P)
                masks_np       = data['masks'].astype(np.uint8)       # (N,P,P,P)
                centers        = data['centers'].astype(np.int32)     # (N,3)
                original_shape = data['original_shape']               # (3,)

                # Inference in mini-batches
                all_preds = []
                for i in range(0, len(patches_np), inference_batch_size):
                    batch_t = (torch.FloatTensor(patches_np[i:i+inference_batch_size])
                               .unsqueeze(1).to(device))
                    out   = model(batch_t)
                    probs = torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
                    all_preds.append(probs)
                    del batch_t, out
                all_preds = np.concatenate(all_preds, axis=0)  # (N,P,P,P)

                # Reconstruct prediction and GT volumes
                pred_vol  = reconstruct_volume_from_patches(
                    all_preds, centers, original_shape, patch_size)
                gt_vol    = reconstruct_volume_from_patches(
                    masks_np.astype(np.float32), centers, original_shape, patch_size)
                gt_binary = (gt_vol > 0.5).astype(np.uint8)
                pred_bin  = (pred_vol > 0.5).astype(np.uint8)

                inter = (pred_bin * gt_binary).sum()
                union = pred_bin.sum() + gt_binary.sum()
                dsc   = float(2.0 * inter / union) if union > 0 else 1.0
                dscs.append(dsc)
                print(f"    {case_id}: DSC={dsc:.4f}  patches={len(patches_np)}"
                      f"  GT={gt_binary.sum()}  Pred={pred_bin.sum()}")

                if save_dir and HAS_NIBABEL:
                    sd = Path(save_dir) / case_id
                    sd.mkdir(parents=True, exist_ok=True)
                    aff = np.eye(4)
                    nib.save(nib.Nifti1Image(pred_bin,  aff), sd / 'pred.nii.gz')
                    nib.save(nib.Nifti1Image(pred_vol,  aff), sd / 'pred_prob.nii.gz')
                    nib.save(nib.Nifti1Image(gt_binary, aff), sd / 'gt.nii.gz')

            except Exception as e:
                print(f"    ERROR {case_id}: {e}")
                import traceback; traceback.print_exc()

    mean = float(np.mean(dscs)) if dscs else 0.0
    std  = float(np.std(dscs))  if dscs else 0.0
    print(f"  → Volume DSC: {mean:.4f} ± {std:.4f}  (n={len(dscs)})")
    return mean, dscs


# ============================================================
# SINGLE EXPERIMENT
# ============================================================
def run_experiment(dataset_name, fold, use_transfer, cfg):
    tag      = "transfer" if use_transfer else "scratch"
    fold_dir = Path(cfg['output_dir']) / dataset_name / tag / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Resume check
    if cfg.get('resume') and (fold_dir / 'best_model.pth').exists():
        print(f"  SKIP (done): {dataset_name}/{tag}/fold_{fold}")
        ckpt = torch.load(fold_dir / 'best_model.pth', map_location='cpu', weights_only=False)
        return ckpt.get('val_patch_dsc', 0.0), ckpt.get('val_volume_dsc', 0.0)

    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)
    device = torch.device(cfg['device'])

    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} | Fold {fold} | {tag.upper()}")
    print(f"{'='*70}")

    # Splits
    with open(cfg['splits_file']) as f:
        splits = json.load(f)
    fd = splits[f'fold_{fold}']
    if dataset_name in fd:
        train_ids = fd[dataset_name]['train']
        val_ids   = fd[dataset_name]['val']
    else:
        train_ids = fd['train']
        val_ids   = fd['val']
    print(f"  Train: {len(train_ids)} cases  |  Val: {len(val_ids)} cases")

    ps = (cfg['patch_size'],) * 3

    # Dataset
    if HAS_PATCH_DS:
        train_ds = PatchDatasetWithCenters(
            preprocessed_dir=cfg['preprocessed_dir'], datasets=[dataset_name],
            split='train', splits_file=cfg['splits_file'], fold=fold,
            patch_size=ps, patches_per_volume=cfg['patches_per_volume'],
            augment=True, lesion_focus_ratio=0.7
        )
        val_ds = PatchDatasetWithCenters(
            preprocessed_dir=cfg['preprocessed_dir'], datasets=[dataset_name],
            split='val', splits_file=cfg['splits_file'], fold=fold,
            patch_size=ps, patches_per_volume=cfg['patches_per_volume'],
            augment=False, lesion_focus_ratio=1.0
        )
    else:
        train_ds = FallbackStrokeDataset(cfg['preprocessed_dir'], dataset_name, train_ids,
                                         patch_size=ps, patches_per_volume=cfg['patches_per_volume'],
                                         augment=True, lesion_focus_ratio=0.7)
        val_ds   = FallbackStrokeDataset(cfg['preprocessed_dir'], dataset_name, val_ids,
                                         patch_size=ps, patches_per_volume=cfg['patches_per_volume'],
                                         augment=False, lesion_focus_ratio=1.0)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # Unique case IDs for volume-level validation
    val_ids = list(set([vol['case_id'] for vol in val_ds.volumes]))
    print(f"  Val case IDs for volume validation: {len(val_ids)}")

    # Model — auto-detects norm type from checkpoint
    pretrained_path = None
    if use_transfer:
        pretrained_path = Path(cfg['brats_ckpt_dir']) / f'fold_{fold}' / 'best_model.pth'
        if not pretrained_path.exists():
            print(f"  WARNING: BraTS ckpt not found at {pretrained_path}, using scratch")
            pretrained_path = None

    model = build_model(cfg, device, pretrained_path=str(pretrained_path) if pretrained_path else None)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = GDiceLossV2()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    best_patch_dsc = best_volume_dsc = 0.0
    best_epoch = 0
    history    = []

    log_path = fold_dir / 'log.csv'
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_loss,val_patch_dsc,val_volume_dsc\n")

    for epoch in range(cfg['epochs']):
        ep = epoch + 1
        print(f"\n  Epoch {ep:03d}/{cfg['epochs']}  [{dataset_name} {tag} fold{fold}]")

        tr_loss, tr_dsc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_dsc = validate_patches(model, val_loader, criterion, device)
        scheduler.step()

        # Volume validation: epoch 1 and every 10 epochs
        vol_dsc = 0.0
        if epoch == 0 or ep % 10 == 0:
            save_nii = (fold_dir / 'vol_val_epoch1') if epoch == 0 else None
            max_c    = 5 if epoch == 0 else None
            patch_96_dir = cfg['patch_64_atlas'] if dataset_name == 'ATLAS' else cfg['patch_64_uoa']
            vol_dsc, _ = validate_volumes(
                model, val_ids, patch_96_dir, device,
                patch_size=cfg['patch_size'],
                inference_batch_size=8,
                max_cases=max_c, save_dir=save_nii)
            if vol_dsc > best_volume_dsc:
                best_volume_dsc = vol_dsc

        # Save best
        is_best = va_dsc > best_patch_dsc
        if is_best:
            best_patch_dsc = va_dsc
            best_epoch     = ep
            torch.save({
                'epoch': epoch, 'fold': fold,
                'dataset': dataset_name, 'transfer': use_transfer,
                'brats_ckpt': str(pretrained_path) if pretrained_path else None,
                'model_state_dict': model.state_dict(),
                'val_patch_dsc': va_dsc, 'val_volume_dsc': vol_dsc,
                'base_channel': cfg['base_channel'], 'patch_size': cfg['patch_size'],
            }, fold_dir / 'best_model.pth')

        with open(log_path, 'a') as f:
            f.write(f"{ep},{tr_loss:.6f},{tr_dsc:.6f},{va_loss:.6f},{va_dsc:.6f},{vol_dsc:.6f}\n")

        history.append({'epoch': ep, 'train_loss': tr_loss, 'train_dsc': tr_dsc,
                        'val_loss': va_loss, 'val_patch_dsc': va_dsc, 'val_volume_dsc': vol_dsc})

        flag    = ' ← BEST' if is_best else ''
        vol_str = f'  Vol={vol_dsc:.4f}' if vol_dsc > 0 else ''
        print(f"  Train L={tr_loss:.4f} D={tr_dsc:.4f} | Val L={va_loss:.4f} D={va_dsc:.4f}{vol_str}{flag}")

    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best patch DSC:  {best_patch_dsc:.4f} @ epoch {best_epoch}")
    print(f"  Best volume DSC: {best_volume_dsc:.4f}")
    return best_patch_dsc, best_volume_dsc


# ============================================================
# MAIN
# ============================================================
def _save_progress(results, out_dir, final=False):
    fname = 'summary_final.json' if final else 'summary_progress.json'
    out   = {}
    for k, v in results.items():
        pd = v.get('patch', []);  vd = v.get('volume', [])
        out[k] = {
            'patch_mean':  float(np.mean(pd)) if pd else 0.0,
            'patch_std':   float(np.std(pd))  if pd else 0.0,
            'volume_mean': float(np.mean(vd)) if vd else 0.0,
            'volume_std':  float(np.std(vd))  if vd else 0.0,
            'per_fold_patch':  [float(x) for x in pd],
            'per_fold_volume': [float(x) for x in vd],
        }
    path = Path(out_dir) / fname
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all',   action='store_true', help='Run all 20 experiments')
    parser.add_argument('--dataset',   type=str, choices=['ATLAS', 'UOA_Private'])
    parser.add_argument('--fold',      type=int, choices=range(5))
    parser.add_argument('--transfer',  action='store_true', help='Use BraTS transfer weights')
    parser.add_argument('--resume',    action='store_true', help='Skip completed experiments')
    # Paths
    parser.add_argument('--brats-ckpt-dir',    type=str, default='/hpc/pahm409/brats_uctransnet3d')
    parser.add_argument('--preprocessed-dir',  type=str, default='/hpc/pahm409/harvard/preprocessed_stroke_foundation')
    parser.add_argument('--splits-file',       type=str, default='splits_5fold.json')
    parser.add_argument('--output-dir',        type=str, default='/hpc/pahm409/uctransnet3d_stroke_experiments')
    # Hyperparams
    parser.add_argument('--epochs',             type=int,   default=100)
    parser.add_argument('--batch-size',         type=int,   default=4)
    parser.add_argument('--lr',                 type=float, default=1e-4)
    parser.add_argument('--patch-size',         type=int,   default=64)
    parser.add_argument('--patches-per-volume', type=int,   default=4)
    parser.add_argument('--base-channel',       type=int,   default=64)
    parser.add_argument('--gpu',                type=int,   default=0)
    parser.add_argument('--patch-64-atlas',     type=str,
                        default='/hpc/pahm409/harvard/preprocessed_stroke_foundation/ATLAS_64x64x64',
                        help='Pre-extracted 64^3 patch NPZs for ATLAS')
    parser.add_argument('--patch-64-uoa',       type=str,
                        default='/hpc/pahm409/harvard/preprocessed_stroke_foundation/UOA_Private_64x64x64',
                        help='Pre-extracted 64^3 patch NPZs for UOA_Private')
    args = parser.parse_args()

    cfg = {
        'brats_ckpt_dir':     args.brats_ckpt_dir,
        'preprocessed_dir':   args.preprocessed_dir,
        'patch_64_atlas':     args.patch_64_atlas,
        'patch_64_uoa':       args.patch_64_uoa,
        'splits_file':        args.splits_file,
        'output_dir':         args.output_dir,
        'epochs':             args.epochs,
        'batch_size':         args.batch_size,
        'lr':                 args.lr,
        'patch_size':         args.patch_size,
        'patches_per_volume': args.patches_per_volume,
        'base_channel':       args.base_channel,
        'device':             f'cuda:{args.gpu}',
        'resume':             args.resume,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.run_all:
        experiments  = [('ATLAS', True), ('ATLAS', False), ('UOA_Private', True), ('UOA_Private', False)]
        all_results  = {}

        print(f"\n{'#'*70}")
        print(f"# UCTransNet3D — ALL 20 STROKE EXPERIMENTS")
        print(f"# 2 datasets × 2 conditions × 5 folds")
        print(f"# Output: {args.output_dir}")
        print(f"{'#'*70}\n")

        for dataset_name, use_transfer in experiments:
            tag = 'transfer' if use_transfer else 'scratch'
            key = f"{dataset_name}_{tag}"
            all_results[key] = {'patch': [], 'volume': []}

            print(f"\n{'#'*70}")
            print(f"# {dataset_name} — {tag.upper()}")
            print(f"{'#'*70}")

            for fold in range(5):
                try:
                    pdsc, vdsc = run_experiment(dataset_name, fold, use_transfer, cfg)
                    all_results[key]['patch'].append(pdsc)
                    all_results[key]['volume'].append(vdsc)
                except Exception as e:
                    print(f"\n❌ ERROR {dataset_name} fold{fold} {tag}: {e}")
                    traceback.print_exc()
                _save_progress(all_results, cfg['output_dir'])

        # Summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS — UCTransNet3D stroke")
        print(f"{'='*70}\n")
        for dataset in ['ATLAS', 'UOA_Private']:
            print(f"\n{dataset}:")
            for tag in ['transfer', 'scratch']:
                r  = all_results.get(f"{dataset}_{tag}", {})
                pd = r.get('patch',  [])
                vd = r.get('volume', [])
                if pd:
                    diff = np.mean(pd) - np.mean(vd) if vd else 0
                    print(f"  {tag.upper():10s}  Patch: {np.mean(pd):.4f}±{np.std(pd):.4f}  "
                          f"Volume: {np.mean(vd):.4f}±{np.std(vd):.4f}  "
                          f"Gap: {diff:+.4f}  folds: {[f'{x:.3f}' for x in pd]}")

        _save_progress(all_results, cfg['output_dir'], final=True)

    elif args.dataset is not None and args.fold is not None:
        run_experiment(args.dataset, args.fold, args.transfer, cfg)
    else:
        parser.error("Use --run-all, or specify both --dataset and --fold")


if __name__ == '__main__':
    main()

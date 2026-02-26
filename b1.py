"""
train_brats_uctransnet3d.py

UCTransNet3D supervised training on BraTS 2024 T2-FLAIR data.
Architecture: UCTransNet3D (heavy, transformer-based)
  - SE (Squeeze-and-Excitation) replaces DualPathChannelAttention
  - GDiceLossV2 replaces AdaptiveRegionalLoss
  - Same BraTS .npz data pipeline as train_brats_t2flair_supervised_FIXED.py

Goal: Test whether heavier architecture benefits more from BraTS pretraining
      than lightweight ResNet-18 (where pretraining was negligible).

Usage:
    python train_brats_uctransnet3d.py \
        --brats-dir /home/pahm409/preprocessed_brats2024_t2flair \
        --splits-file brats2024_t2flair_splits_5fold.json \
        --output-dir /home/pahm409/brats_uctransnet3d \
        --fold 0 --epochs 100 --batch-size 4
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import math
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# AMP removed — fp32 only for stability
from torch.nn import Dropout, Softmax, LayerNorm
from torch.nn.modules.utils import _triple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
class UCTransNetConfig:
    """Configuration for UCTransNet3D (similar to original paper config)."""
    def __init__(self, base_channel=64):
        self.base_channel = base_channel
        self.KV_size = base_channel + base_channel*2 + base_channel*4 + base_channel*8
        # = 64+128+256+512 = 960
        self.expand_ratio = 4
        self.transformer = {
            "num_heads": 4,
            "num_layers": 4,
            "embeddings_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "dropout_rate": 0.1,
        }


# ============================================================
# SE (Squeeze-and-Excitation) — replaces DualPathChannelAttention3D
# ============================================================
class SEBlock3D(nn.Module):
    """Standard Squeeze-and-Excitation block for 3D feature maps."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


# ============================================================
# CHANNEL TRANSFORMER (unchanged from original, uses SE in skip)
# ============================================================
class Channel_Embeddings3D(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size  = _triple(img_size)
        patch_size = _triple(patchsize)
        n_patches = (
            (img_size[0] // patch_size[0]) *
            (img_size[1] // patch_size[1]) *
            (img_size[2] // patch_size[2])
        )
        self.patch_embeddings = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=patch_size, stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class Reconstruct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv  = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm  = nn.BatchNorm3d(out_channels)
        self.act   = nn.ReLU(inplace=True)
        self.scale = scale_factor

    def forward(self, x):
        if x is None:
            return None
        b, n, c = x.size()
        d = h = w = int(round(n ** (1/3)))
        x = x.permute(0,2,1).contiguous().view(b, c, d, h, w)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return self.act(self.norm(self.conv(x)))


class Attention_org(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.KV_size = config.KV_size
        self.num_heads = config.transformer["num_heads"]

        self.query = nn.ModuleList([
            nn.ModuleList([nn.Linear(ch, ch, bias=False) for _ in range(self.num_heads)])
            for ch in channel_num
        ])
        self.key   = nn.ModuleList([nn.Linear(self.KV_size, self.KV_size, bias=False)
                                    for _ in range(self.num_heads)])
        self.value = nn.ModuleList([nn.Linear(self.KV_size, self.KV_size, bias=False)
                                    for _ in range(self.num_heads)])
        self.out   = nn.ModuleList([nn.Linear(ch, ch, bias=False) for ch in channel_num])

        self.psi          = nn.InstanceNorm2d(self.num_heads)
        self.softmax      = Softmax(dim=3)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, embs, emb_all):
        # embs: list of 4 tensors (or None)
        multi_K = torch.stack([k(emb_all) for k in self.key],  dim=1)
        multi_V = torch.stack([v(emb_all) for v in self.value], dim=1)

        outputs = []
        for i, emb in enumerate(embs):
            if emb is None:
                outputs.append(None)
                continue
            Q = torch.stack([self.query[i][h](emb) for h in range(self.num_heads)], dim=1)
            Q = Q.transpose(-1, -2)
            scores = torch.matmul(Q, multi_K) / math.sqrt(self.KV_size)
            probs  = self.attn_dropout(self.softmax(self.psi(scores)))
            ctx    = torch.matmul(probs, multi_V.transpose(-1,-2))
            ctx    = ctx.permute(0,3,2,1).contiguous().mean(dim=3)
            outputs.append(self.proj_dropout(self.out[i](ctx)))
        return outputs


class Mlp(nn.Module):
    def __init__(self, config, in_ch, mlp_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_ch,  mlp_ch)
        self.fc2 = nn.Linear(mlp_ch, in_ch)
        self.act = nn.GELU()
        self.drop = Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block_ViT(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.attn_norms = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])
        self.attn_norm_all = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn  = Attention_org(config, channel_num)
        self.ffn_norms = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])
        self.ffns      = nn.ModuleList([
            Mlp(config, c, c * config.expand_ratio) for c in channel_num
        ])

    def forward(self, embs):
        # embs: list of 4 tensors
        emb_all = torch.cat([e for e in embs if e is not None], dim=2)
        emb_all = self.attn_norm_all(emb_all)

        normed  = [self.attn_norms[i](embs[i]) if embs[i] is not None else None
                   for i in range(len(embs))]
        attn_out = self.channel_attn(normed, emb_all)

        # residual
        embs = [embs[i] + attn_out[i] if embs[i] is not None else None
                for i in range(len(embs))]

        # FFN
        embs = [embs[i] + self.ffns[i](self.ffn_norms[i](embs[i]))
                if embs[i] is not None else None
                for i in range(len(embs))]
        return embs


class Encoder(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        self.layers = nn.ModuleList([
            Block_ViT(config, channel_num)
            for _ in range(config.transformer["num_layers"])
        ])
        self.norms = nn.ModuleList([LayerNorm(c, eps=1e-6) for c in channel_num])

    def forward(self, embs):
        for layer in self.layers:
            embs = layer(embs)
        embs = [self.norms[i](embs[i]) if embs[i] is not None else None
                for i in range(len(embs))]
        return embs


class ChannelTransformer3D(nn.Module):
    def __init__(self, config, img_size, channel_num, patch_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([
            Channel_Embeddings3D(config, patch_sizes[i], img_size // (2**i), channel_num[i])
            for i in range(4)
        ])
        self.encoder = Encoder(config, channel_num)
        self.reconstructs = nn.ModuleList([
            Reconstruct3D(channel_num[i], channel_num[i], kernel_size=1,
                          scale_factor=patch_sizes[i])
            for i in range(4)
        ])

    def forward(self, feats):
        embs = [self.embeddings[i](feats[i]) for i in range(4)]
        embs = self.encoder(embs)
        outs = []
        for i in range(4):
            if feats[i] is None:
                outs.append(None)
                continue
            r = self.reconstructs[i](embs[i])
            if r.shape[2:] != feats[i].shape[2:]:
                r = F.interpolate(r, size=feats[i].shape[2:],
                                  mode='trilinear', align_corners=False)
            outs.append(r + feats[i])
        return outs


# ============================================================
# ENCODER / DECODER BLOCKS
# ============================================================
def get_activation(name='ReLU'):
    return getattr(nn, name.lower().capitalize(), nn.ReLU)()


class ConvBN3D(nn.Module):
    def __init__(self, in_ch, out_ch, activation='ReLU'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            get_activation(activation),
        )
    def forward(self, x):
        return self.block(x)


def make_nConv3D(in_ch, out_ch, n, activation='ReLU'):
    layers = [ConvBN3D(in_ch, out_ch, activation)]
    for _ in range(n - 1):
        layers.append(ConvBN3D(out_ch, out_ch, activation))
    return nn.Sequential(*layers)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv=2):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), make_nConv3D(in_ch, out_ch, n_conv))
    def forward(self, x):
        return self.block(x)


class UpBlock_SE3D(nn.Module):
    """Decoder block: upsample → SE attention on skip → cat → conv.
    Replaces CCA (which used global average pooling cross-attention) with SE."""
    def __init__(self, in_ch, out_ch, n_conv=2):
        super().__init__()
        skip_ch = in_ch // 2
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.se = SEBlock3D(skip_ch, reduction=16)
        self.convs = make_nConv3D(in_ch, out_ch, n_conv)

    def forward(self, x, skip):
        up   = self.up(x)
        skip = self.se(skip)
        # match spatial dims if needed
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.convs(torch.cat([skip, up], dim=1))


# ============================================================
# FULL UCTransNet3D (with SE + GDiceLossV2)
# ============================================================
class UCTransNet3D(nn.Module):
    """
    UCTransNet3D:
      - Heavy transformer-based architecture
      - SE blocks in decoder skip connections (replaces DualPathChannelAttention)
      - Designed for 96x96x96 patches (same as ResNet-18 baseline)
      - n_classes=2 for binary stroke segmentation (background + lesion)
    """
    def __init__(self, config, n_channels=1, n_classes=2, img_size=96):
        super().__init__()
        B = config.base_channel  # 64

        # Encoder
        self.inc   = make_nConv3D(n_channels, B,   2)
        self.down1 = DownBlock3D(B,   B*2, 2)
        self.down2 = DownBlock3D(B*2, B*4, 2)
        self.down3 = DownBlock3D(B*4, B*8, 2)
        self.down4 = DownBlock3D(B*8, B*8, 2)

        # Channel Transformer
        # patch_sizes chosen so each level produces integer number of patches at img_size//2^i
        # img_size=96: level0=96, level1=48, level2=24, level3=12
        # patch_sizes: 8,4,2,1 → tokens: 12^3, 12^3, 12^3, 12^3 = 1728 each
        self.mtc = ChannelTransformer3D(
            config, img_size,
            channel_num=[B, B*2, B*4, B*8],
            patch_sizes=[8, 4, 2, 1],
        )

        # Decoder with SE skip attention
        self.up4 = UpBlock_SE3D(B*16, B*4, 2)
        self.up3 = UpBlock_SE3D(B*8,  B*2, 2)
        self.up2 = UpBlock_SE3D(B*4,  B,   2)
        self.up1 = UpBlock_SE3D(B*2,  B,   2)

        self.out_conv = nn.Conv3d(B, n_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Channel transformer refines x1–x4
        x1, x2, x3, x4 = self.mtc([x1, x2, x3, x4])

        # Decoder
        x = self.up4(x5, x4)
        x = self.up3(x,  x3)
        x = self.up2(x,  x2)
        x = self.up1(x,  x1)

        out = self.out_conv(x)

        # Restore input resolution
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='trilinear', align_corners=False)
        return out


# ============================================================
# LOSS: GDiceLossV2
# ============================================================
class GDiceLossV2(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # Always compute in float32 — never fp16
        logits = logits.float()
        prob   = torch.softmax(logits, dim=1)

        if logits.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=logits.shape[2:],
                mode='nearest'
            ).squeeze(1).long()

        target_onehot = torch.zeros_like(prob)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        prob_flat   = prob.reshape(prob.size(0), prob.size(1), -1)
        target_flat = target_onehot.reshape(target_onehot.size(0), target_onehot.size(1), -1)

        inter = (prob_flat * target_flat).sum(-1)
        union = prob_flat.sum(-1) + target_flat.sum(-1)

        # Inverse-frequency weighting: handles all-background patches gracefully
        w    = 1.0 / (target_flat.sum(-1) ** 2 + self.smooth)
        dice = (2. * w * inter + self.smooth) / (w * union + self.smooth)

        loss = 1. - dice.mean()

        # Safety guard
        if torch.isnan(loss):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss


# ============================================================
# DATASET (identical to train_brats_t2flair_supervised_FIXED.py)
# ============================================================
class SimpleBraTSDataset(Dataset):
    def __init__(self, npz_dir, case_list, patch_size=(96,96,96),
                 patches_per_volume=10, augment=False):
        self.npz_dir = Path(npz_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment

        self.volumes = []
        for cid in case_list:
            p = self.npz_dir / f"{cid}.npz"
            if p.exists():
                self.volumes.append({'case_id': cid, 'path': p})

        if not self.volumes:
            raise ValueError(f"No volumes found in {npz_dir}")
        print(f"Loaded {len(self.volumes)} volumes")

    def __len__(self):
        return len(self.volumes) * self.patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        data  = np.load(self.volumes[vol_idx]['path'])
        image = data['image']   # (D,H,W) float32
        mask  = data['mask']    # (D,H,W) uint8

        D, H, W = image.shape
        pd, ph, pw = self.patch_size

        d0 = np.random.randint(0, max(1, D - pd + 1))
        h0 = np.random.randint(0, max(1, H - ph + 1))
        w0 = np.random.randint(0, max(1, W - pw + 1))

        ip = image[d0:d0+pd, h0:h0+ph, w0:w0+pw]
        mp = mask [d0:d0+pd, h0:h0+ph, w0:w0+pw]

        # pad if smaller than patch_size
        def pad(arr, target):
            pads = [(0, max(0, t - s)) for s, t in zip(arr.shape, target)]
            return np.pad(arr, pads, mode='constant')

        if ip.shape != tuple(self.patch_size):
            ip = pad(ip, self.patch_size)
            mp = pad(mp, self.patch_size)

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
# METRICS
# ============================================================
def compute_dsc(pred_prob, target):
    """pred_prob: (B, D, H, W) foreground probability; target: (B, D, H, W) long"""
    if pred_prob.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(1).float(), size=pred_prob.shape[1:], mode='nearest'
        ).squeeze(1)
    pred_bin = (pred_prob > 0.5).float()
    tgt_bin  = (target > 0).float()
    inter = (pred_bin * tgt_bin).sum()
    union = pred_bin.sum() + tgt_bin.sum()
    if union == 0:
        return 1.0
    return (2. * inter / union).item()


# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):    
    model.train()
    tot_loss = tot_dsc = 0
    n_valid  = 0
    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)

        out  = model(images)
        loss = criterion(out, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            pbar.set_postfix(loss='NaN-skip', dsc='--')
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            dsc = compute_dsc(torch.softmax(out.float(), dim=1)[:, 1], masks)
        tot_loss += loss.item()
        tot_dsc  += dsc
        n_valid  += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}', dsc=f'{dsc:.4f}')

    if n_valid == 0:
        return float('nan'), 0.0
    return tot_loss / n_valid, tot_dsc / n_valid


def validate(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_dsc = 0
    n_valid  = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val'):
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)
            out  = model(images)
            loss = criterion(out, masks)
            dsc = compute_dsc(torch.softmax(out.float(), dim=1)[:, 1], masks)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                tot_loss += loss.item()
                n_valid  += 1
            tot_dsc  += dsc
    n = len(loader)
    return (tot_loss / n_valid if n_valid > 0 else float('nan')), tot_dsc / n


# ============================================================
# MAIN
# ============================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brats-dir',   type=str,
                        default='/hpc/pahm409/preprocessed_brats2024_t2flair')
    parser.add_argument('--splits-file', type=str,
                        default='brats2024_t2flair_splits_5fold.json')
    parser.add_argument('--output-dir',  type=str,
                        default='/hpc/pahm409/brats_uctransnet3d')
    parser.add_argument('--fold',        type=int,   default=0)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch-size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--base-channel',type=int,   default=64,
                        help='Base channel width. 64 = heavy, 32 = medium')
    parser.add_argument('--patch-size',  type=int,   default=96,
                        help='Cubic patch size (must be divisible by 8)')
    parser.add_argument('--patches-per-volume', type=int, default=10,
                        help='Random patches sampled per volume per epoch (reduce to speed up)')
    # Pretraining option: load a checkpoint from BraTS-pretrained model
    parser.add_argument('--pretrain-ckpt', type=str, default=None,
                        help='Path to BraTS pretrained checkpoint (for transfer learning arm)')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda:0')

    out_dir = Path(args.output_dir) / f'fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = 'TRANSFER' if args.pretrain_ckpt else 'SCRATCH'

    print("=" * 80)
    print(f"UCTransNet3D — BraTS T2-FLAIR SUPERVISED [{mode}]")
    print("=" * 80)
    print(f"Fold          : {args.fold}")
    print(f"Base channel  : {args.base_channel}")
    print(f"Patch size    : {args.patch_size}^3")
    print(f"Batch size    : {args.batch_size}")
    print(f"Patches/vol   : {args.patches_per_volume}")
    print(f"LR            : {args.lr}")
    print(f"Pretrain ckpt : {args.pretrain_ckpt}")
    print("=" * 80)

    # ---- Splits ----
    with open(args.splits_file) as f:
        splits = json.load(f)
    train_cases = splits[f'fold_{args.fold}']['BraTS2024_T2FLAIR']['train']
    val_cases   = splits[f'fold_{args.fold}']['BraTS2024_T2FLAIR']['val']
    print(f"Train: {len(train_cases)}  Val: {len(val_cases)}\n")

    ps = (args.patch_size,) * 3

    if args.patch_size % 16 != 0:
        print(f"WARNING: patch_size={args.patch_size} is not divisible by 16.")
        suggested = (args.patch_size // 16) * 16
        print(f"         Recommend --patch-size {suggested} or {suggested+16} to avoid shape mismatches.")
        print(f"         Continuing anyway (interpolation will handle it, but may affect quality).\n")

    train_ds = SimpleBraTSDataset(args.brats_dir, train_cases,
                                   patch_size=ps, patches_per_volume=args.patches_per_volume, augment=True)
    val_ds   = SimpleBraTSDataset(args.brats_dir, val_cases,
                                   patch_size=ps, patches_per_volume=args.patches_per_volume, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ---- Model ----
    config = UCTransNetConfig(base_channel=args.base_channel)
    model  = UCTransNet3D(config, n_channels=1, n_classes=2,
                          img_size=args.patch_size).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optional: load pretrained weights
    if args.pretrain_ckpt:
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights: {len(missing)} missing, {len(unexpected)} unexpected keys")

    criterion = GDiceLossV2()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dsc = 0.0
    history  = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} {'─'*60}")
        tr_loss, tr_dsc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_dsc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train  loss={tr_loss:.4f}  dsc={tr_dsc:.4f}")
        print(f"Val    loss={vl_loss:.4f}  dsc={vl_dsc:.4f}")

        history.append({
            'epoch': epoch, 'train_loss': tr_loss, 'train_dsc': tr_dsc,
            'val_loss': vl_loss, 'val_dsc': vl_dsc,
        })

        if vl_dsc > best_dsc:
            best_dsc = vl_dsc
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc':            vl_dsc,
                'fold':               args.fold,
                'mode':               mode,
                'architecture':       'UCTransNet3D',
                'base_channel':       args.base_channel,
                'patch_size':         args.patch_size,
                'pretraining':        args.pretrain_ckpt or 'scratch',
            }, out_dir / 'best_model.pth')
            print(f"  ✓ NEW BEST  val_dsc={best_dsc:.4f}")

    # Save training history
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"DONE  Fold={args.fold}  Mode={mode}  Best Val DSC={best_dsc:.4f}")
    print(f"Checkpoint: {out_dir / 'best_model.pth'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

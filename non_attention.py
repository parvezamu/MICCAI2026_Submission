"""
train_segmentation_unified.py

Unified segmentation training script supporting multiple attention mechanisms:
- Baseline: Standard U-Net (no attention)
- ECA: Efficient Channel Attention (applied after decoder blocks)
- MKDC: Multi-Kernel Depthwise Convolution (applied on skip connections)

All use:
- SimCLR pretrained encoder
- ROI-based validation for fair comparison
- Robust mask handling (>0)
- Same loss function (no double softmax bug)

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys
import torch.nn.functional as F
from collections import defaultdict
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first."""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


class GDiceLossV2(nn.Module):
    """Generalized Dice Loss V2 - IMPORTANT: Pass RAW LOGITS"""
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape
        shp_y = gt.shape
        
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x, dtype=net_output.dtype, device=net_output.device)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)
        else:
            softmax_output = net_output

        input_flat = flatten(softmax_output)
        target_flat = flatten(y_onehot)
        target_flat = target_flat.float()
        
        target_sum = target_flat.sum(-1)
        class_weights = 1. / (target_sum * target_sum).clamp(min=self.smooth)
        
        intersect = (input_flat * target_flat).sum(-1) * class_weights
        intersect = intersect.sum()
        
        denominator = ((input_flat + target_flat).sum(-1) * class_weights).sum()
        
        GDL = 1 - 2. * (intersect / denominator.clamp(min=self.smooth))
        
        return GDL


def softmax_helper(x):
    return F.softmax(x, dim=1)


class ECAAttention(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels: int):
        super().__init__()
        
        import math
        kernel_size = int(abs((math.log(channels, 2) + 1) / 2))
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)
        y = y.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
        y = self.conv(y).squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        attention_weights = self.sigmoid(y)
        return x * attention_weights


class MultiKernelDepthwiseConv(nn.Module):
    """Multi-Kernel Depthwise Convolution from MK-UNet"""
    def __init__(self, channels: int, kernels=[1, 3, 5]):
        super().__init__()
        
        self.channels = channels
        self.kernels = kernels
        self.num_kernels = len(kernels)
        
        self.dw_convs = nn.ModuleList()
        for k in kernels:
            self.dw_convs.append(nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=k, 
                         padding=k//2, groups=channels, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU6(inplace=True)
            ))
    
    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, d, h, w = x.size()
        groups = self.num_kernels
        
        while channels % groups != 0 and groups > 1:
            groups -= 1
        
        if groups == 1:
            return x
        
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, d, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, channels, d, h, w)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum([dw_conv(x) for dw_conv in self.dw_convs])
        out = self.channel_shuffle(out)
        return out


class ResNet3DEncoder(nn.Module):
    """Extract encoder from ResNet3D"""
    def __init__(self, base_encoder):
        super(ResNet3DEncoder, self).__init__()
        self.conv1 = base_encoder.conv1
        self.bn1 = base_encoder.bn1
        self.maxpool = base_encoder.maxpool
        self.layer1 = base_encoder.layer1
        self.layer2 = base_encoder.layer2
        self.layer3 = base_encoder.layer3
        self.layer4 = base_encoder.layer4
    
    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1, x2, x3, x4, x5]


class UNetDecoder3D(nn.Module):
    """
    Unified U-Net decoder supporting multiple attention mechanisms
    
    attention_type: 'none', 'eca', or 'mkdc'
    """
    def __init__(self, num_classes=2, attention_type='none', mkdc_kernels=[1, 3, 5]):
        super(UNetDecoder3D, self).__init__()
        
        self.attention_type = attention_type
        
        # Create attention modules based on type
        if attention_type == 'eca':
            # ECA: Applied AFTER decoder blocks
            self.eca4 = ECAAttention(256)
            self.eca3 = ECAAttention(128)
            self.eca2 = ECAAttention(64)
            self.eca1 = ECAAttention(64)
        elif attention_type == 'mkdc':
            # MKDC: Applied ON skip connections (before concat)
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256, kernels=mkdc_kernels)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128, kernels=mkdc_kernels)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
        
        # Standard decoder blocks (same for all)
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x_up, x_skip):
        d_up, h_up, w_up = x_up.shape[2:]
        d_skip, h_skip, w_skip = x_skip.shape[2:]
        
        if d_up != d_skip or h_up != h_skip or w_up != w_skip:
            diff_d = d_skip - d_up
            diff_h = h_skip - h_up
            diff_w = w_skip - w_up
            
            if diff_d > 0 or diff_h > 0 or diff_w > 0:
                padding = [
                    max(0, diff_w // 2), max(0, diff_w - diff_w // 2),
                    max(0, diff_h // 2), max(0, diff_h - diff_h // 2),
                    max(0, diff_d // 2), max(0, diff_d - diff_d // 2)
                ]
                x_up = F.pad(x_up, padding)
            elif diff_d < 0 or diff_h < 0 or diff_w < 0:
                d_start = max(0, -diff_d // 2)
                h_start = max(0, -diff_h // 2)
                w_start = max(0, -diff_w // 2)
                x_up = x_up[:, :, d_start:d_start + d_skip, 
                           h_start:h_start + h_skip, w_start:w_start + w_skip]
        
        return x_up
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        # Level 4
        x = self.up4(x5)
        x = self._match_size(x, x4)
        if self.attention_type == 'mkdc':
            x4 = self.mkdc4(x4)  # MKDC on skip
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.attention_type == 'eca':
            x = self.eca4(x)  # ECA after decoder
        
        # Level 3
        x = self.up3(x)
        x = self._match_size(x, x3)
        if self.attention_type == 'mkdc':
            x3 = self.mkdc3(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.attention_type == 'eca':
            x = self.eca3(x)
        
        # Level 2
        x = self.up2(x)
        x = self._match_size(x, x2)
        if self.attention_type == 'mkdc':
            x2 = self.mkdc2(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.attention_type == 'eca':
            x = self.eca2(x)
        
        # Level 1
        x = self.up1(x)
        x = self._match_size(x, x1)
        if self.attention_type == 'mkdc':
            x1 = self.mkdc1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        if self.attention_type == 'eca':
            x = self.eca1(x)
        
        x = self.final_conv(x)
        return x


class SegmentationModel(nn.Module):
    """Complete segmentation model: Encoder + Decoder"""
    def __init__(self, encoder, num_classes=2, attention_type='none'):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes, attention_type=attention_type)
        self.attention_type = attention_type
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, 
                                      mode='trilinear', align_corners=False)
        
        return seg_logits


def compute_dsc(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, debug_first_batch=False):
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        masks = (masks > 0).long()
        
        if debug_first_batch and batch_idx == 0:
            print("\n" + "="*70)
            print("DEBUG: First batch statistics")
            print("="*70)
            print(f"Mask unique values: {torch.unique(masks)}")
            print(f"Lesion voxels: {(masks > 0).sum().item()}")
            print("="*70 + "\n")
        
        with autocast():
            outputs = model(images)
            
            if outputs.shape[2:] != masks.shape[1:]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=outputs.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                masks_resized = masks
            
            loss = criterion(outputs, masks_resized)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks_resized.unsqueeze(1) > 0).float()
            dsc = compute_dsc(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / num_batches, total_dsc / num_batches


def validate_patches(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating (patches)'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            masks = (masks > 0).long()
            
            with autocast():
                outputs = model(images)
                
                if outputs.shape[2:] != masks.shape[1:]:
                    masks_resized = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=outputs.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    masks_resized = masks
                
                loss = criterion(outputs, masks_resized)
            
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks_resized.unsqueeze(1) > 0).float()
            dsc = compute_dsc(pred_probs, target_onehot)
            
            total_loss += loss.item()
            total_dsc += dsc
            num_batches += 1
    
    return total_loss / num_batches, total_dsc / num_batches


def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
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


def validate_full_volumes(model, dataset, device, patch_size, save_nifti=False, save_dir=None, epoch=None):
    model.eval()
    
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("\nCollecting patches for full reconstruction...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Processing patches'):
            batch = dataset[idx]
            
            vol_idx = batch['vol_idx']
            if isinstance(vol_idx, torch.Tensor):
                vol_idx = vol_idx.item()
            
            image = batch['image'].unsqueeze(0).to(device)
            center = batch['center'].numpy()
            
            with autocast():
                output = model(image)
            
            pred = torch.softmax(output, dim=1).cpu().numpy()
            
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
    
    print("\nReconstructing volumes...")
    all_dscs = []
    
    if save_nifti and save_dir:
        nifti_dir = Path(save_dir) / f'reconstructions_epoch_{epoch}'
        nifti_dir.mkdir(parents=True, exist_ok=True)
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Computing DSCs'):
        vol_info = dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed, count_map = reconstruct_from_patches_with_count(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        reconstructed_binary = (reconstructed > 0.5).astype(np.uint8)
        
        mask_gt = (vol_info['mask'] > 0).astype(np.uint8)
        roi_mask = (count_map > 0).astype(np.uint8)
        
        pred_in_roi = reconstructed_binary * roi_mask
        gt_in_roi = mask_gt * roi_mask
        
        intersection = (pred_in_roi * gt_in_roi).sum()
        union = pred_in_roi.sum() + gt_in_roi.sum()
        
        if union == 0:
            dsc = 1.0 if pred_in_roi.sum() == 0 else 0.0
        else:
            dsc = (2. * intersection) / union
        
        all_dscs.append(dsc)
        
        if save_nifti and save_dir:
            case_dir = nifti_dir / case_id
            case_dir.mkdir(exist_ok=True)
            
            affine = np.eye(4)
            
            nib.save(nib.Nifti1Image(reconstructed_binary, affine), 
                    case_dir / 'prediction.nii.gz')
            nib.save(nib.Nifti1Image(reconstructed.astype(np.float32), affine), 
                    case_dir / 'prediction_prob.nii.gz')
            nib.save(nib.Nifti1Image(mask_gt, affine), 
                    case_dir / 'ground_truth.nii.gz')
            nib.save(nib.Nifti1Image(vol_info['volume'].astype(np.float32), affine), 
                    case_dir / 'volume.nii.gz')
            nib.save(nib.Nifti1Image(roi_mask, affine), 
                    case_dir / 'roi_mask.nii.gz')
            nib.save(nib.Nifti1Image(count_map.astype(np.float32), affine), 
                    case_dir / 'count_map.nii.gz')
            
            with open(case_dir / 'metadata.json', 'w') as f:
                json.dump({
                    'case_id': case_id,
                    'dsc': float(dsc),
                    'epoch': epoch,
                    'lesion_volume_total': int(mask_gt.sum()),
                    'lesion_volume_in_roi': int(gt_in_roi.sum()),
                    'pred_volume': int(pred_in_roi.sum()),
                    'roi_coverage': float(roi_mask.sum() / np.prod(mask_gt.shape)),
                    'lesion_coverage': float(gt_in_roi.sum() / max(1, mask_gt.sum()))
                }, f, indent=4)
    
    mean_dsc = np.mean(all_dscs)
    
    if save_nifti and save_dir:
        with open(nifti_dir / 'summary.json', 'w') as f:
            json.dump({
                'epoch': epoch,
                'mean_dsc': float(mean_dsc),
                'std_dsc': float(np.std(all_dscs)),
                'all_dscs': [float(d) for d in all_dscs]
            }, f, indent=4)
    
    return mean_dsc, all_dscs


def plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('GDice Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_dscs, 'b-', linewidth=2, label='Train')
    axes[0, 1].plot(epochs, val_dscs_patch, 'r--', linewidth=2, label='Val (patch)')
    axes[0, 1].plot(epochs, val_dscs_recon, 'g-', linewidth=3, label='Val (recon)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('DSC')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    gap_recon = [train_dscs[i] - val_dscs_recon[i] for i in range(len(epochs))]
    axes[1, 0].plot(epochs, gap_recon, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val DSC')
    axes[1, 0].set_title('Generalization Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    best_recon = [max(val_dscs_recon[:i+1]) for i in range(len(epochs))]
    axes[1, 1].plot(epochs, best_recon, 'g-', linewidth=3, marker='*', markersize=8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Best DSC')
    axes[1, 1].set_title('Best Full-Volume DSC')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Unified Segmentation Training')
    
    parser.add_argument('--pretrained-checkpoint', type=str, default=None)
    parser.add_argument('--attention', type=str, default='none', 
                       choices=['none', 'eca', 'mkdc'],
                       help='Attention mechanism: none (baseline), eca, or mkdc')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/segmentation_unified_5fold')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10)
    parser.add_argument('--lesion-focus-ratio', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--validate-recon-every', type=int, default=5)
    parser.add_argument('--save-nifti-every', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    init_method = "SimCLR_Pretrained" if args.pretrained_checkpoint else "Random_Init"
    
    attention_names = {
        'none': 'Baseline (no attention)',
        'eca': 'ECA (Efficient Channel Attention)',
        'mkdc': 'MKDC (Multi-Kernel Depthwise Conv)'
    }
    
    print("\n" + "="*70)
    print(f"UNIFIED SEGMENTATION TRAINING - FOLD {args.fold}")
    print(f"Initialization: {init_method}")
    print(f"Attention: {attention_names[args.attention]}")
    print("="*70)
    print("✓ ROI-based validation (fair comparison)")
    print("✓ Robust mask handling (>0)")
    print("✓ Fixed: No double softmax bug")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / init_method / args.attention / f'fold_{args.fold}' / f'exp_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}\n")
    
    encoder = resnet3d_18(in_channels=1)
    
    if args.pretrained_checkpoint:
        print("Loading SimCLR pretrained encoder...")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
        simclr_model.load_state_dict(checkpoint['model_state_dict'])
        encoder = simclr_model.encoder
        print(f"✓ Loaded from epoch {checkpoint['epoch'] + 1}\n")
    else:
        print("Using random initialization\n")
    
    model = SegmentationModel(encoder, num_classes=2, attention_type=args.attention).to(device)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    print(f"Loading data for fold {args.fold}...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=args.fold,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        augment=True,
        lesion_focus_ratio=args.lesion_focus_ratio
    )
    
    val_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='val',
        splits_file='splits_5fold.json',
        fold=args.fold,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        augment=False,
        lesion_focus_ratio=args.lesion_focus_ratio
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"✓ Train: {len(train_dataset)} patches, {len(train_dataset.volumes)} volumes")
    print(f"✓ Val: {len(val_dataset)} patches, {len(val_dataset.volumes)} volumes\n")
    
    criterion = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler()
    
    print("="*70)
    print(f"Starting Training - {init_method} - {attention_names[args.attention]}")
    print("="*70 + "\n")
    
    train_losses, train_dscs, val_dscs_patch, val_dscs_recon = [], [], [], []
    best_dsc, best_epoch = 0, 0
    
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_patch,val_dsc_recon,lr\n")
    
    for epoch in range(args.epochs):
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler,
            debug_first_batch=(epoch == 0)
        )
        val_loss_patch, val_dsc_patch = validate_patches(model, val_loader, criterion, device)
        
        if (epoch + 1) % args.validate_recon_every == 0 or epoch == 0:
            print("\n" + "="*70)
            print(f"FULL RECONSTRUCTION (ROI-BASED) - Fold {args.fold}, Epoch {epoch+1}")
            print("="*70)
            
            save_nifti = ((epoch + 1) % args.save_nifti_every == 0)
            val_dsc_recon, all_dscs = validate_full_volumes(
                model, val_dataset, device, tuple(args.patch_size),
                save_nifti=save_nifti, save_dir=exp_dir, epoch=epoch+1
            )
            
            print(f"Reconstructed DSC: {val_dsc_recon:.4f}")
            print(f"  Min: {np.min(all_dscs):.4f}, Max: {np.max(all_dscs):.4f}")
            print("="*70 + "\n")
        else:
            val_dsc_recon = val_dscs_recon[-1] if val_dscs_recon else 0.0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        train_dscs.append(train_dsc)
        val_dscs_patch.append(val_dsc_patch)
        val_dscs_recon.append(val_dsc_recon)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc_patch:.6f},{val_dsc_recon:.6f},{current_lr:.6f}\n")
        
        print(f"{'='*70}")
        print(f"Fold {args.fold} - Epoch {epoch+1}/{args.epochs} - {attention_names[args.attention]}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val DSC (patch):  {val_dsc_patch:.4f}")
        print(f"  Val DSC (recon):  {val_dsc_recon:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_dsc_recon > best_dsc:
            best_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon,
                'attention_type': args.attention
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        if (epoch + 1) % 5 == 0:
            plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                               exp_dir / f'curves_epoch_{epoch+1}.png')
    
    print("\n" + "="*70)
    print(f"FOLD {args.fold} TRAINING COMPLETE!")
    print("="*70)
    print(f"Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"Results: {exp_dir}")
    print("="*70 + "\n")
    
    plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                        exp_dir / 'curves_final.png')
    
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump({
            'fold': args.fold,
            'initialization': init_method,
            'attention_type': args.attention,
            'best_epoch': best_epoch,
            'best_dsc': float(best_dsc),
            'final_train_dsc': float(train_dscs[-1]),
            'final_val_dsc_recon': float(val_dscs_recon[-1])
        }, f, indent=4)


if __name__ == '__main__':
    main()

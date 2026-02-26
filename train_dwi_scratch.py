"""
train_dwi_scratch.py

DWI segmentation training from scratch on ISLES2022 dataset
Based on train_segmentation_corrected.py but for DWI data

Features:
- Train on ISLES2022_resampled (DWI data)
- Random initialization (no pre-training)
- MKDC and Deep Supervision support
- Multi-run support with seeds
- Resume capability
- 5-fold cross-validation

Author: Parvez
Date: February 2026
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
import random
import math

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.append('.')
from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


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
    Enhanced U-Net decoder with deep supervision support
    
    use_mkdc: If True, applies MKDC on skip connections
    deep_supervision: If True, outputs auxiliary predictions at levels 4, 3, 2
    """
    def __init__(self, num_classes=2, use_mkdc=False, mkdc_kernels=[1, 3, 5],
                 deep_supervision=False):
        super(UNetDecoder3D, self).__init__()
        
        self.use_mkdc = use_mkdc
        self.deep_supervision = deep_supervision
        
        # Create MKDC modules if enabled
        if use_mkdc:
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256, kernels=mkdc_kernels)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128, kernels=mkdc_kernels)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
        
        # Standard decoder blocks
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds_conv4 = nn.Conv3d(256, num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(128, num_classes, kernel_size=1)
            self.ds_conv2 = nn.Conv3d(64, num_classes, kernel_size=1)
    
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
        
        ds_outputs = []  # For deep supervision
        
        # Level 4
        x = self.up4(x5)
        x = self._match_size(x, x4)
        if self.use_mkdc:
            x4 = self.mkdc4(x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv4(x))
        
        # Level 3
        x = self.up3(x)
        x = self._match_size(x, x3)
        if self.use_mkdc:
            x3 = self.mkdc3(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv3(x))
        
        # Level 2
        x = self.up2(x)
        x = self._match_size(x, x2)
        if self.use_mkdc:
            x2 = self.mkdc2(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv2(x))
        
        # Level 1
        x = self.up1(x)
        x = self._match_size(x, x1)
        if self.use_mkdc:
            x1 = self.mkdc1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        # Final output
        final_output = self.final_conv(x)
        
        if self.deep_supervision:
            return [final_output] + ds_outputs  # [main, ds4, ds3, ds2]
        else:
            return final_output


class SegmentationModel(nn.Module):
    """Complete segmentation model: Encoder + Decoder"""
    def __init__(self, encoder, num_classes=2, use_mkdc=False, deep_supervision=False):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes, 
                                     use_mkdc=use_mkdc,
                                     deep_supervision=deep_supervision)
        self.use_mkdc = use_mkdc
        self.deep_supervision = deep_supervision
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        
        if self.deep_supervision:
            # Resize all outputs to input size
            resized_outputs = []
            for out in outputs:
                if out.shape[2:] != input_size:
                    out = F.interpolate(out, size=input_size, 
                                      mode='trilinear', align_corners=False)
                resized_outputs.append(out)
            return resized_outputs
        else:
            if outputs.shape[2:] != input_size:
                outputs = F.interpolate(outputs, size=input_size, 
                                      mode='trilinear', align_corners=False)
            return outputs


def compute_deep_supervised_loss(outputs, target, criterion, weights=[1.0, 0.5, 0.25, 0.125]):
    """
    Compute weighted sum of losses for deep supervision
    """
    total_loss = 0.0
    for i, (out, weight) in enumerate(zip(outputs, weights)):
        if out.shape[2:] != target.shape[1:]:
            target_resized = F.interpolate(
                target.unsqueeze(1).float(),
                size=out.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target
        
        loss = criterion(out, target_resized)
        total_loss += weight * loss
    
    return total_loss


def compute_dsc(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup followed by cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        print(f"✓ Warmup scheduler: {warmup_epochs} warmup epochs, "
              f"{total_epochs} total epochs")
        print(f"  Base LR: {base_lr:.6f}, Min LR: {min_lr:.6f}")
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, 
                deep_supervision=False, ds_weights=[1.0, 0.5, 0.25, 0.125],
                debug_first_batch=False, max_grad_norm=1.0):
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
            
            if deep_supervision:
                loss = compute_deep_supervised_loss(outputs, masks, criterion, ds_weights)
                main_output = outputs[0]
            else:
                if outputs.shape[2:] != masks.shape[1:]:
                    masks_resized = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=outputs.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    masks_resized = masks
                
                loss = criterion(outputs, masks_resized)
                main_output = outputs
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            if main_output.shape[2:] != masks.shape[1:]:
                masks_for_dsc = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=main_output.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                masks_for_dsc = masks
                
            pred_probs = torch.softmax(main_output, dim=1)[:, 1:2, ...]
            target_onehot = (masks_for_dsc.unsqueeze(1) > 0).float()
            dsc = compute_dsc(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / num_batches, total_dsc / num_batches


def validate_patches(model, dataloader, criterion, device, deep_supervision=False):
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
                
                if deep_supervision:
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                if main_output.shape[2:] != masks.shape[1:]:
                    masks_resized = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=main_output.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    masks_resized = masks
                
                loss = criterion(main_output, masks_resized)
            
            pred_probs = torch.softmax(main_output, dim=1)[:, 1:2, ...]
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


def validate_full_volumes(model, dataset, device, patch_size, deep_supervision=False,
                         save_nifti=False, save_dir=None, epoch=None):
    """Process volume-by-volume with proper batch processing"""
    from collections import defaultdict
    from torch.utils.data import DataLoader
    
    model.eval()
    
    num_volumes = len(dataset.volumes)
    
    if save_nifti and save_dir:
        nifti_dir = Path(save_dir) / f'reconstructions_epoch_{epoch}'
        nifti_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nReconstructing {num_volumes} volumes...")
    
    dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
    )
    
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing patches'):
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
    
    all_dscs = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing volumes'):
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
            
            with open(case_dir / 'metadata.json', 'w') as f:
                json.dump({
                    'case_id': case_id,
                    'dsc': float(dsc),
                    'epoch': epoch,
                    'lesion_volume': int(mask_gt.sum()),
                    'pred_volume': int(pred_in_roi.sum())
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


def compute_effective_lr(batch_size, base_batch_size=8, base_lr=0.0001):
    """Compute effective learning rate using linear scaling rule"""
    scaling_factor = batch_size / base_batch_size
    effective_lr = base_lr * scaling_factor
    
    print(f"\n{'='*70}")
    print(f"LEARNING RATE SCALING")
    print(f"{'='*70}")
    print(f"Base configuration: batch_size={base_batch_size}, lr={base_lr}")
    print(f"Current configuration: batch_size={batch_size}")
    print(f"Scaling factor: {scaling_factor:.2f}×")
    print(f"Effective learning rate: {effective_lr:.6f}")
    print(f"{'='*70}\n")
    
    return effective_lr


def load_checkpoint_for_resume(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint for resuming training"""
    print(f"\n{'='*70}")
    print("RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    print(f"Loading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model state loaded")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✓ Optimizer state loaded")
    
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.current_epoch = checkpoint['epoch']
            print("✓ Scheduler state loaded")
        except:
            print("⚠️  Scheduler state not compatible, will continue with current scheduler")
    
    start_epoch = checkpoint['epoch'] + 1
    best_dsc = checkpoint.get('val_dsc_recon', 0.0)
    
    train_losses = checkpoint.get('train_losses', [])
    train_dscs = checkpoint.get('train_dscs', [])
    val_dscs_patch = checkpoint.get('val_dscs_patch', [])
    val_dscs_recon = checkpoint.get('val_dscs_recon', [])
    
    print(f"✓ Resuming from epoch {start_epoch}")
    print(f"✓ Best DSC so far: {best_dsc:.4f}")
    print(f"✓ Training history: {len(train_losses)} epochs")
    print(f"{'='*70}\n")
    
    return start_epoch, best_dsc, train_losses, train_dscs, val_dscs_patch, val_dscs_recon


def create_isles_splits(isles_preprocessed_dir, output_file='isles_splits_5fold_resampled.json'):
    """Create 5-fold splits for ISLES dataset"""
    isles_dir = Path(isles_preprocessed_dir)
    all_cases = sorted([f.stem for f in isles_dir.glob("*.npz")])
    
    print(f"Creating ISLES splits...")
    print(f"  Found {len(all_cases)} cases")
    
    np.random.seed(42)
    np.random.shuffle(all_cases)
    
    n_cases = len(all_cases)
    fold_size = n_cases // 5
    
    splits = {}
    
    for fold in range(5):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < 4 else n_cases
        
        test_cases = all_cases[test_start:test_end]
        remaining = [c for c in all_cases if c not in test_cases]
        
        val_size = len(remaining) // 5
        val_cases = remaining[:val_size]
        train_cases = remaining[val_size:]
        
        splits[f'fold_{fold}'] = {
            'ISLES2022_resampled': {
                'train': train_cases,
                'val': val_cases,
                'test': test_cases
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"  ✓ Saved to: {output_file}")
    for fold in range(5):
        print(f"    Fold {fold}: Train={len(splits[f'fold_{fold}']['ISLES2022_resampled']['train'])}, "
              f"Val={len(splits[f'fold_{fold}']['ISLES2022_resampled']['val'])}, "
              f"Test={len(splits[f'fold_{fold}']['ISLES2022_resampled']['test'])}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='DWI Scratch Training on ISLES')
    
    # Model configuration
    parser.add_argument('--use-mkdc', action='store_true',
                       help='Use Multi-Kernel Depthwise Conv on skip connections')
    parser.add_argument('--deep-supervision', action='store_true',
                       help='Enable deep supervision')
    parser.add_argument('--ds-weights', type=float, nargs=4, 
                       default=[1.0, 0.5, 0.25, 0.125],
                       help='Weights for deep supervision: [main, ds4, ds3, ds2]')
    
    # Resume support
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Multi-run support
    parser.add_argument('--run-id', type=int, default=None,
                       help='Run ID for repeated experiments (0, 1, 2)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Data configuration
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/dwi_scratch_5fold')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10)
    parser.add_argument('--lesion-focus-ratio', type=float, default=0.7)
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--base-lr', type=float, default=0.0001)
    parser.add_argument('--base-batch-size', type=int, default=8)
    
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    
    # Validation settings
    parser.add_argument('--validate-recon-every', type=int, default=5)
    parser.add_argument('--save-nifti-every', type=int, default=10)
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is None:
        base_seed = 42
        if args.run_id is not None:
            args.seed = base_seed + args.fold * 10 + args.run_id
        else:
            args.seed = base_seed + args.fold
    
    set_seed(args.seed)
    
    device = torch.device('cuda:0')
    
    # Compute effective learning rate
    if args.lr is None:
        args.lr = compute_effective_lr(
            batch_size=args.batch_size,
            base_batch_size=args.base_batch_size,
            base_lr=args.base_lr
        )
    
    # Create ISLES splits if not exist
    splits_file = 'isles_splits_5fold_resampled.json'
    if not Path(splits_file).exists():
        create_isles_splits(args.preprocessed_dir, splits_file)
    
    print("\n" + "="*70)
    print(f"DWI SCRATCH TRAINING - FOLD {args.fold}")
    if args.run_id is not None:
        print(f"Run ID: {args.run_id}")
    print(f"Random Seed: {args.seed}")
    print(f"MKDC: {'ENABLED' if args.use_mkdc else 'DISABLED'}")
    print(f"Deep Supervision: {'ENABLED' if args.deep_supervision else 'DISABLED'}")
    print("="*70)
    print(f"✓ Batch size: {args.batch_size}")
    print(f"✓ Learning rate: {args.lr:.6f}")
    print(f"✓ Optimizer: {args.optimizer.upper()}")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create config name
    config_name = 'baseline'
    if args.use_mkdc and args.deep_supervision:
        config_name = 'mkdc_ds'
    elif args.use_mkdc:
        config_name = 'mkdc'
    elif args.deep_supervision:
        config_name = 'ds'
    
    # Create experiment directory
    if args.run_id is not None:
        exp_dir = Path(args.output_dir) / config_name / f'fold_{args.fold}' / \
                  f'run_{args.run_id}' / f'exp_{timestamp}'
    else:
        exp_dir = Path(args.output_dir) / config_name / f'fold_{args.fold}' / \
                  f'exp_{timestamp}'
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}\n")
    
    # Save configuration
    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Build model
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(encoder, num_classes=2, 
                             use_mkdc=args.use_mkdc,
                             deep_supervision=args.deep_supervision).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {total_params:,} parameters\n")
    
    # Load data
    print(f"Loading ISLES data for fold {args.fold}...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='train',
        splits_file=splits_file,
        fold=args.fold,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        augment=True,
        lesion_focus_ratio=args.lesion_focus_ratio,
        compute_lesion_bins=False
    )
    
    val_dataset = PatchDatasetWithCenters(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='val',
        splits_file=splits_file,
        fold=args.fold,
        patch_size=tuple(args.patch_size),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )
    

    
    
    NUM_WORKERS = 0

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
                         
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                       shuffle=False, num_workers=0, pin_memory=True)
    
    
    
    
    
    
    
    
    
    print(f"✓ Train: {len(train_dataset)} patches, {len(train_dataset.volumes)} volumes")
    print(f"✓ Val: {len(val_dataset)} patches, {len(val_dataset.volumes)} volumes\n")
    
    # Setup training
    criterion = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, weight_decay=args.weight_decay,
                             nesterov=True)
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr
    )
    
    scaler = GradScaler()
    
    # Resume if needed
    start_epoch = 0
    best_dsc = 0.0
    best_epoch = 0
    train_losses, train_dscs, val_dscs_patch, val_dscs_recon = [], [], [], []
    
    if args.resume_checkpoint:
        start_epoch, best_dsc, train_losses, train_dscs, val_dscs_patch, val_dscs_recon = \
            load_checkpoint_for_resume(args.resume_checkpoint, model, optimizer, scheduler, device)
        best_epoch = start_epoch - 1 if start_epoch > 0 else 0
    
    print("\n" + "="*70)
    print(f"Starting DWI Scratch Training")
    if args.run_id is not None:
        print(f"Run {args.run_id} - Seed {args.seed}")
    print("="*70 + "\n")
    
    # Initialize log file
    log_file = exp_dir / 'training_log.csv'
    if start_epoch == 0:
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,train_dsc,val_dsc_patch,val_dsc_recon,lr\n")
    
    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.step(epoch)
        
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler,
            deep_supervision=args.deep_supervision,
            ds_weights=args.ds_weights,
            debug_first_batch=(epoch == 0),
            max_grad_norm=args.max_grad_norm
        )
        
        val_loss_patch, val_dsc_patch = validate_patches(
            model, val_loader, criterion, device,
            deep_supervision=args.deep_supervision
        )
        
        if (epoch + 1) % args.validate_recon_every == 0 or epoch == 0:
            print("\n" + "="*70)
            print(f"FULL RECONSTRUCTION - Fold {args.fold}, Epoch {epoch+1}")
            if args.run_id is not None:
                print(f"Run {args.run_id}")
            print("="*70)
            
            save_nifti = ((epoch + 1) % args.save_nifti_every == 0)
            val_dsc_recon, all_dscs = validate_full_volumes(
                model, val_dataset, device, tuple(args.patch_size),
                deep_supervision=args.deep_supervision,
                save_nifti=save_nifti, save_dir=exp_dir, epoch=epoch+1
            )
            
            print(f"Reconstructed DSC: {val_dsc_recon:.4f}")
            print(f"  Min: {np.min(all_dscs):.4f}, Max: {np.max(all_dscs):.4f}")
            print("="*70 + "\n")
        else:
            val_dsc_recon = val_dscs_recon[-1] if val_dscs_recon else 0.0
        
        train_losses.append(train_loss)
        train_dscs.append(train_dsc)
        val_dscs_patch.append(val_dsc_patch)
        val_dscs_recon.append(val_dsc_recon)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc_patch:.6f},"
                   f"{val_dsc_recon:.6f},{current_lr:.6f}\n")
        
        print(f"{'='*70}")
        print(f"Fold {args.fold} - Epoch {epoch+1}/{args.epochs} - {config_name.upper()}")
        if args.run_id is not None:
            print(f"  [Run {args.run_id}]")
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
                'run_id': args.run_id,
                'seed': args.seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': {'current_epoch': scheduler.current_epoch},
                'scaler_state_dict': scaler.state_dict(),
                'val_dsc_recon': val_dsc_recon,
                'use_mkdc': args.use_mkdc,
                'deep_supervision': args.deep_supervision,
                'train_losses': train_losses,
                'train_dscs': train_dscs,
                'val_dscs_patch': val_dscs_patch,
                'val_dscs_recon': val_dscs_recon
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        if (epoch + 1) % 5 == 0:
            plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                               exp_dir / f'curves_epoch_{epoch+1}.png')
    
    print("\n" + "="*70)
    print(f"FOLD {args.fold} TRAINING COMPLETE!")
    if args.run_id is not None:
        print(f"Run {args.run_id}")
    print("="*70)
    print(f"Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"Results: {exp_dir}")
    print("="*70 + "\n")
    
    plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                        exp_dir / 'curves_final.png')
    
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump({
            'fold': args.fold,
            'run_id': args.run_id,
            'seed': args.seed,
            'config': config_name,
            'use_mkdc': args.use_mkdc,
            'deep_supervision': args.deep_supervision,
            'best_epoch': best_epoch,
            'best_dsc': float(best_dsc),
            'final_train_dsc': float(train_dscs[-1]),
            'final_val_dsc_recon': float(val_dscs_recon[-1]),
            'total_parameters': total_params,
            'batch_size': args.batch_size,
            'learning_rate': args.lr
        }, f, indent=4)


if __name__ == '__main__':
    main()

"""
train_segmentation_eca.py (ROI-BASED VALIDATION VERSION)

Clean segmentation training with ECA (Efficient Channel Attention):
- SimCLR pretrained encoder
- Standard U-Net decoder with ECA on skip connections
- Proper patch-based training and full-volume validation
- FIXED: Double softmax bug in loss calculation
- UPDATED: ROI-based validation (same as Code 3)

ECA (Efficient Channel Attention):
- Lightweight: ~100 parameters per layer
- Effective: Learns which feature channels matter for lesions
- No spatial attention: Keeps it simple

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

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
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


class GDiceLossV2(nn.Module):
    """
    Generalized Dice Loss V2
    
    From nnU-Net / pytorch-3dunet
    Paper: https://arxiv.org/pdf/1707.03237.pdf
    
    IMPORTANT: Pass RAW LOGITS, not softmax outputs!
    """
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape
        shp_y = gt.shape
        
        # One hot encode gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x, dtype=net_output.dtype, device=net_output.device)
                y_onehot.scatter_(1, gt, 1)

        # Apply softmax if needed
        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)
        else:
            softmax_output = net_output

        # Flatten
        input_flat = flatten(softmax_output)
        target_flat = flatten(y_onehot)
        target_flat = target_flat.float()
        
        # Compute class weights: 1 / (sum^2)
        target_sum = target_flat.sum(-1)
        class_weights = 1. / (target_sum * target_sum).clamp(min=self.smooth)
        
        # Compute weighted intersection and denominator
        intersect = (input_flat * target_flat).sum(-1) * class_weights
        intersect = intersect.sum()
        
        denominator = ((input_flat + target_flat).sum(-1) * class_weights).sum()
        
        # Generalized Dice Loss
        GDL = 1 - 2. * (intersect / denominator.clamp(min=self.smooth))
        
        return GDL


def softmax_helper(x):
    """Helper for applying softmax"""
    return F.softmax(x, dim=1)


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (ECA-Net)
    
    Paper: ECA-Net: Efficient Channel Attention for Deep CNN
    https://arxiv.org/abs/1910.03151
    
    Key innovation: Uses 1D convolution instead of fully-connected layers
    
    How it works:
    1. Global Average Pooling: Squeeze spatial dimensions [B,C,D,H,W] -> [B,C,1,1,1]
    2. 1D Convolution: Learn channel interactions with adaptive kernel size
    3. Sigmoid: Generate attention weights [0,1] for each channel
    4. Multiply: Rescale input channels by learned weights
    
    Why it helps stroke segmentation:
    - Learns which channels encode lesion features (e.g., hyperintensity, boundaries)
    - Suppresses irrelevant channels (e.g., CSF, normal tissue patterns)
    - Extremely lightweight: ~100 parameters per layer
    
    Example:
        Input: [B, 64, 12, 12, 12]
        After GAP: [B, 64, 1, 1, 1] -> squeeze to [B, 64]
        Conv1D with k=3: [B, 1, 64] -> [B, 1, 64]
        Attention weights: [B, 64, 1, 1, 1]
        Output: [B, 64, 12, 12, 12] (rescaled by attention)
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # Adaptive kernel size based on channel dimensionality
        # From ECA paper: k = |log2(C) / 2| (odd number)
        import math
        kernel_size = int(abs((math.log(channels, 2) + 1) / 2))
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)
        
        # Global Average Pooling: [B,C,D,H,W] -> [B,C,1,1,1]
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # 1D convolution over channel dimension
        # Input: [B, 1, C], Output: [B, 1, C]
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        
        # Sigmoid to generate attention weights in [0, 1]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, D, H, W]
        
        Returns:
            Channel-attended features [B, C, D, H, W]
        """
        # Step 1: Global Average Pooling
        y = self.gap(x)
        
        # Step 2: Prepare for 1D convolution
        y = y.squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Step 3: Add dimension for Conv1d
        y = y.unsqueeze(1)
        
        # Step 4: 1D convolution across channels
        y = self.conv(y)
        
        # Step 5: Remove Conv1d dimension and reshape for broadcasting
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Step 6: Generate attention weights with sigmoid
        attention_weights = self.sigmoid(y)
        
        # Step 7: Apply channel attention
        return x * attention_weights


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
    U-Net decoder with optional ECA attention
    
    Architecture:
        Encoder features -> Upsample -> Concatenate with skip connection
        -> Decoder block (2x Conv3d) -> [OPTIONAL: ECA] -> Next level
    
    Where ECA is applied:
        - AFTER each decoder block (not on skip connections)
        - This allows the decoder to learn which features from the 
          combined (upsampled + skip) are most important for lesions
    """
    def __init__(self, num_classes=2, use_eca=False):
        super(UNetDecoder3D, self).__init__()
        
        self.use_eca = use_eca
        
        # ========== Decoder Level 4 (lowest resolution) ==========
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        if use_eca:
            self.eca4 = ECAAttention(256)
        
        # ========== Decoder Level 3 ==========
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        if use_eca:
            self.eca3 = ECAAttention(128)
        
        # ========== Decoder Level 2 ==========
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        if use_eca:
            self.eca2 = ECAAttention(64)
        
        # ========== Decoder Level 1 (highest resolution) ==========
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        if use_eca:
            self.eca1 = ECAAttention(64)
        
        # Final 1x1 conv to get class predictions
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        """Standard U-Net decoder block: Two 3x3 convolutions"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x_up, x_skip):
        """Match spatial dimensions of upsampled and skip features"""
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
        """Forward pass through decoder"""
        x1, x2, x3, x4, x5 = encoder_features
        
        # ========== Decoder Level 4 ==========
        x = self.up4(x5)
        x = self._match_size(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.use_eca:
            x = self.eca4(x)
        
        # ========== Decoder Level 3 ==========
        x = self.up3(x)
        x = self._match_size(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.use_eca:
            x = self.eca3(x)
        
        # ========== Decoder Level 2 ==========
        x = self.up2(x)
        x = self._match_size(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.use_eca:
            x = self.eca2(x)
        
        # ========== Decoder Level 1 ==========
        x = self.up1(x)
        x = self._match_size(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        if self.use_eca:
            x = self.eca1(x)
        
        # ========== Final Prediction ==========
        x = self.final_conv(x)
        return x


class SegmentationModel(nn.Module):
    """Complete segmentation model: Encoder + Decoder"""
    def __init__(self, encoder, num_classes=2, use_eca=False):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes, use_eca=use_eca)
        self.use_eca = use_eca
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, 
                                      mode='trilinear', align_corners=False)
        
        return seg_logits


def compute_dsc(pred, target, smooth=1e-6):
    """Compute Dice Score"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, debug_first_batch=False):
    """Train one epoch on patches"""
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # UPDATED: Robust mask binarization (same as Code 3)
        masks = (masks > 0).long()
        
        # Debug first batch to verify labels
        if debug_first_batch and batch_idx == 0:
            print("\n" + "="*70)
            print("DEBUG: First batch statistics")
            print("="*70)
            print(f"Mask unique values: {torch.unique(masks)}")
            print(f"Lesion voxels: {(masks > 0).sum().item()}")
            print(f"Mask max: {masks.max().item()}")
            print(f"Mask shape: {masks.shape}")
            print(f"Image shape: {images.shape}")
            print("="*70 + "\n")
        
        with autocast():
            # Get RAW LOGITS (no softmax here!)
            outputs = model(images)
            
            if outputs.shape[2:] != masks.shape[1:]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=outputs.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                masks_resized = masks
            
            # Pass raw logits to loss (softmax happens inside)
            loss = criterion(outputs, masks_resized)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute DSC for monitoring - UPDATED: Use >0 instead of ==1
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
    """Quick validation on patches"""
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating (patches)'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # UPDATED: Robust mask binarization
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
    """
    Reconstruct full volume from patches with averaging
    UPDATED: Returns count_map to identify predicted regions (same as Code 3)
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    skipped = 0
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            skipped += 1
            continue
        
        patch = patch_preds[i, 1, ...]
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    if skipped > len(centers) * 0.1:
        print(f"WARNING: Skipped {skipped}/{len(centers)} patches")
    
    # Average only where we have predictions
    mask = count_map > 0
    reconstructed[mask] = reconstructed[mask] / count_map[mask]
    
    return reconstructed, count_map


def validate_full_volumes(model, dataset, device, patch_size, save_nifti=False, save_dir=None, epoch=None):
    """
    Validate by reconstructing full volumes
    UPDATED: ROI-based validation (same as Code 3) - only evaluate where predictions exist
    """
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
        
        # UPDATED: Reconstruct WITH count map
        reconstructed, count_map = reconstruct_from_patches_with_count(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        reconstructed_binary = (reconstructed > 0.5).astype(np.uint8)
        
        # UPDATED: ROI-based DSC computation (same as Code 3)
        mask_gt = (vol_info['mask'] > 0).astype(np.uint8)
        
        # Create ROI mask - only voxels where we made predictions
        roi_mask = (count_map > 0).astype(np.uint8)
        
        # Apply ROI to both prediction and ground truth
        pred_in_roi = reconstructed_binary * roi_mask
        gt_in_roi = mask_gt * roi_mask
        
        # Compute Dice ONLY in ROI
        intersection = (pred_in_roi * gt_in_roi).sum()
        union = pred_in_roi.sum() + gt_in_roi.sum()
        
        if union == 0:
            dsc = 1.0 if pred_in_roi.sum() == 0 else 0.0
        else:
            dsc = (2. * intersection) / union
        
        all_dscs.append(dsc)
        
        # UPDATED: Save NIfTI with ROI information (same as Code 3)
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
            
            # UPDATED: Save detailed metadata (same as Code 3)
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
    """Plot training progress"""
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
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pretrained-checkpoint', type=str, default=None)
    parser.add_argument('--use-eca', action='store_true',
                       help='Use ECA (Efficient Channel Attention)')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/segmentation_eca')
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
    
    print("\n" + "="*70)
    print(f"ECA SEGMENTATION TRAINING (ROI-BASED) - FOLD {args.fold}")
    print(f"Initialization: {init_method}")
    if args.use_eca:
        print("Attention: ECA (Efficient Channel Attention) ✅")
    else:
        print("Attention: None (baseline)")
    print("="*70)
    print("✓ ECA: Channel attention only (~100 params per layer)")
    print("✓ FIXED: Double softmax bug")
    print("✓ UPDATED: ROI-based validation (same as Code 3)")
    print("✓ UPDATED: Robust mask handling (>0 instead of ==1)")
    print("✓ Applied AFTER decoder blocks")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    attn_str = "with_eca" if args.use_eca else "baseline"
    exp_dir = Path(args.output_dir) / init_method / attn_str / f'fold_{args.fold}' / f'exp_{timestamp}'
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
    
    model = SegmentationModel(encoder, num_classes=2, use_eca=args.use_eca).to(device)
    
    attn_str = "with ECA" if args.use_eca else "baseline"
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters ({attn_str})\n")
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Train: {len(train_dataset)} patches, {len(train_dataset.volumes)} volumes")
    print(f"✓ Val: {len(val_dataset)} patches, {len(val_dataset.volumes)} volumes\n")
    
    criterion = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler()
    
    print("="*70)
    print(f"Starting Training - {init_method} - Fold {args.fold}")
    print("="*70 + "\n")
    
    train_losses = []
    train_dscs = []
    val_dscs_patch = []
    val_dscs_recon = []
    
    best_dsc = 0
    best_epoch = 0
    
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
        print(f"Fold {args.fold} - Epoch {epoch+1}/{args.epochs}")
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
                'use_eca': args.use_eca
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        if (epoch + 1) % 5 == 0:
            plot_training_curves(
                train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                exp_dir / f'curves_epoch_{epoch+1}.png'
            )
    
    print("\n" + "="*70)
    print(f"FOLD {args.fold} TRAINING COMPLETE!")
    print("="*70)
    print(f"Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"Results: {exp_dir}")
    print("="*70 + "\n")
    
    plot_training_curves(
        train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
        exp_dir / 'curves_final.png'
    )
    
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump({
            'fold': args.fold,
            'initialization': init_method,
            'use_eca': args.use_eca,
            'best_epoch': best_epoch,
            'best_dsc': float(best_dsc),
            'final_train_dsc': float(train_dscs[-1]),
            'final_val_dsc_recon': float(val_dscs_recon[-1])
        }, f, indent=4)


if __name__ == '__main__':
    main()

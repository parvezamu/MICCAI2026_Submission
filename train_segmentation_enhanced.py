"""
train_segmentation_enhanced.py

Enhanced segmentation training with multiple attention mechanisms:
- Baseline: Standard U-Net (no attention)
- MKDC: Multi-Kernel Depthwise Convolution (multi-scale spatial)
- CLGCN: Cross-Layer Graph Convolutional Network (structural relationships)
- CGAT: Channel-aware Graph Attention (multi-view fusion)

Inspired by MLMSeg (Computers in Biology and Medicine 2024)
Enhanced with learnable cross-layer connections for stroke segmentation

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


class MultiKernelDepthwiseConv(nn.Module):
    """
    Multi-Kernel Depthwise Convolution (MKDC)
    
    Captures multi-scale spatial features through parallel depth-wise convolutions
    with element-wise summation and channel shuffle.
    """
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
        """Channel shuffle for inter-channel information flow"""
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
        # Element-wise summation of multi-scale features
        out = sum([dw_conv(x) for dw_conv in self.dw_convs])
        out = self.channel_shuffle(out)
        return out


class CrossLayerGCN(nn.Module):
    """
    Cross-Layer Graph Convolutional Network (CLGCN)
    
    Inspired by MLMSeg (Computers in Biology and Medicine 2024)
    
    Key Innovation: Learns structural relationships between encoder layers
    through a learnable adjacency matrix. Each layer is treated as a node,
    and the network learns which layers should communicate.
    
    Benefits for stroke segmentation:
    - Adaptively learns correlations between high-level and low-level features
    - Captures hidden structural relationships in feature space
    - Helps integrate semantic info from different scales
    
    Parameters:
        num_layers: Number of encoder layers (default: 4)
        channels_list: List of channel counts for each layer [C1, C2, C3, C4]
        hidden_dim: Hidden dimension for graph convolution
        dropout_rate: Dropout rate for regularization
    """
    def __init__(self, num_layers=4, channels_list=[64, 64, 128, 256], 
                 hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.channels_list = channels_list
        self.hidden_dim = hidden_dim
        
        # Learnable adjacency matrix (num_layers × num_layers)
        # This learns which layers should have stronger connections
        self.adjacency = nn.Parameter(torch.randn(num_layers, num_layers))
        
        # Project each layer to common hidden dimension
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(c, hidden_dim, kernel_size=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for c in channels_list
        ])
        
        # Graph convolution transformation
        # FIXED: BatchNorm1d should normalize over hidden_dim, not num_layers!
        self.gcn_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # ✓ CORRECT - was num_layers before
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Back-projection to original dimensions
        self.layer_back_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim, c, kernel_size=1),
                nn.BatchNorm3d(c),
                nn.Sigmoid()  # Generate attention weights in [0, 1]
            ) for c in channels_list
        ])
    
    def forward(self, layer_features):
        """
        Args:
            layer_features: List of [B, C_i, D_i, H_i, W_i] from encoder levels
        
        Returns:
            List of refined features with same shapes as input
        """
        batch_size = layer_features[0].size(0)
        
        # Step 1: Project all layers to common dimension
        # [B, hidden_dim, 1, 1, 1] for each layer
        projected = []
        for i, feat in enumerate(layer_features):
            proj = self.layer_projections[i](feat)  # [B, hidden_dim, 1, 1, 1]
            projected.append(proj.squeeze(-1).squeeze(-1).squeeze(-1))  # [B, hidden_dim]
        
        # Stack: [B, num_layers, hidden_dim]
        X = torch.stack(projected, dim=1)
        
        # Step 2: Normalize adjacency matrix (softmax over connections)
        # Each row sums to 1 - controls how much each layer attends to others
        A_norm = F.softmax(self.adjacency, dim=-1)  # [num_layers, num_layers]
        
        # Step 3: Graph convolution
        # X: [B, num_layers, hidden_dim]
        # A_norm: [num_layers, num_layers]
        # Result: [B, num_layers, hidden_dim]
        
        # Reshape for batch matrix multiplication
        # X_flat: [B * num_layers, hidden_dim]
        X_flat = X.reshape(batch_size * self.num_layers, self.hidden_dim)
        
        # Apply transformation
        # Linear: [B * num_layers, hidden_dim] -> [B * num_layers, hidden_dim]
        # BatchNorm1d expects: [N, C] where C = hidden_dim
        X_transformed = self.gcn_transform(X_flat)
        X_transformed = X_transformed.reshape(batch_size, self.num_layers, self.hidden_dim)
        
        # Apply graph structure: G = A @ X
        G = torch.matmul(A_norm.unsqueeze(0), X_transformed)  # [B, num_layers, hidden_dim]
        
        # Step 4: Back-project and apply as attention to original features
        refined_features = []
        for i, feat in enumerate(layer_features):
            # Get graph output for this layer
            g_i = G[:, i, :]  # [B, hidden_dim]
            g_i = g_i.view(batch_size, self.hidden_dim, 1, 1, 1)
            
            # Expand to feature spatial dimensions
            spatial_size = feat.shape[2:]
            g_i_expanded = F.interpolate(g_i, size=spatial_size, mode='trilinear', align_corners=False)
            
            # Back-project to original channels and apply as attention
            attn_weights = self.layer_back_projections[i](g_i_expanded)
            
            # Apply attention to original features
            refined = feat * attn_weights
            refined_features.append(refined)
        
        return refined_features


class ChannelAttention(nn.Module):
    """
    Channel-aware Graph Attention (CGAT)
    
    From MLMSeg paper: Learns contribution factors of different channels
    to optimally fuse multi-view features.
    
    Instead of simple concatenation, this assigns learned attention weights
    to each channel based on its importance for the segmentation task.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Channel attention using global pooling + FC layers
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Attention network
        hidden_dim = max(in_channels // reduction, 8)
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            Channel-attended features [B, C, D, H, W]
        """
        batch, channels = x.size(0), x.size(1)
        
        # Global average pooling: [B, C, D, H, W] -> [B, C, 1, 1, 1]
        gap = self.gap(x).view(batch, channels)
        
        # Compute channel attention weights: [B, C]
        attn_weights = self.attention_net(gap)
        
        # Reshape for broadcasting: [B, C, 1, 1, 1]
        attn_weights = attn_weights.view(batch, channels, 1, 1, 1)
        
        # Apply attention
        return x * attn_weights


class CGATBlock(nn.Module):
    """
    Channel-aware Graph Attention Block for multi-view feature fusion
    
    Fuses features from different views (local, multi-scale, structural)
    with learned channel attention weights.
    """
    def __init__(self, channels):
        super().__init__()
        
        # Concatenated features will have 3x channels (upsampled + mkdc + clgcn)
        concat_channels = channels * 3
        
        # Channel attention for fusion
        self.channel_attn = ChannelAttention(concat_channels, reduction=16)
        
        # Convolution after fusion
        self.conv = nn.Sequential(
            nn.Conv3d(concat_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x_up, x_mkdc, x_clgcn):
        """
        Args:
            x_up: Upsampled features from previous decoder level
            x_mkdc: MKDC-refined skip features
            x_clgcn: CLGCN-refined skip features
        
        Returns:
            Fused features with same spatial size and channels as x_mkdc
        """
        # Concatenate all three views
        x_concat = torch.cat([x_up, x_mkdc, x_clgcn], dim=1)
        
        # Apply channel attention
        x_attended = self.channel_attn(x_concat)
        
        # Final convolution
        x_fused = self.conv(x_attended)
        
        return x_fused


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
    Enhanced U-Net decoder with multiple attention mechanisms
    
    Supports three configurations:
    1. baseline: Standard skip connections
    2. mkdc: Multi-kernel depthwise convolution on skip connections
    3. mkdc_clgcn_cgat: Full enhancement with cross-layer GCN and channel attention
    """
    def __init__(self, num_classes=2, attention_type='none', mkdc_kernels=[1, 3, 5]):
        super(UNetDecoder3D, self).__init__()
        
        self.attention_type = attention_type
        
        # MKDC modules (applied ON skip connections before concat)
        if attention_type in ['mkdc', 'mkdc_clgcn_cgat']:
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256, kernels=mkdc_kernels)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128, kernels=mkdc_kernels)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
        
        # CLGCN module (learns cross-layer structural relationships)
        if attention_type == 'mkdc_clgcn_cgat':
            self.clgcn = CrossLayerGCN(
                num_layers=4,
                channels_list=[64, 64, 128, 256],
                hidden_dim=256,
                dropout_rate=0.3
            )
        
        # Standard decoder blocks
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        
        if attention_type == 'mkdc_clgcn_cgat':
            # Use CGAT for fusion
            self.dec4 = CGATBlock(256)
        else:
            # Standard decoder block
            in_ch = 512 if attention_type != 'none' else 512
            self.dec4 = self._decoder_block(in_ch, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        if attention_type == 'mkdc_clgcn_cgat':
            self.dec3 = CGATBlock(128)
        else:
            self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        if attention_type == 'mkdc_clgcn_cgat':
            self.dec2 = CGATBlock(64)
        else:
            self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        if attention_type == 'mkdc_clgcn_cgat':
            self.dec1 = CGATBlock(64)
        else:
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
        
        # Apply CLGCN if using full enhancement
        if self.attention_type == 'mkdc_clgcn_cgat':
            # Learn cross-layer structural relationships
            refined_features = self.clgcn([x1, x2, x3, x4])
            x1_clgcn, x2_clgcn, x3_clgcn, x4_clgcn = refined_features
        
        # Level 4
        x = self.up4(x5)
        x = self._match_size(x, x4)
        
        if self.attention_type in ['mkdc', 'mkdc_clgcn_cgat']:
            x4_mkdc = self.mkdc4(x4)
            
            if self.attention_type == 'mkdc_clgcn_cgat':
                # Fuse: upsampled + MKDC + CLGCN with channel attention
                x = self.dec4(x, x4_mkdc, x4_clgcn)
            else:
                # Standard: concat + decoder block
                x = torch.cat([x, x4_mkdc], dim=1)
                x = self.dec4(x)
        else:
            # Baseline
            x = torch.cat([x, x4], dim=1)
            x = self.dec4(x)
        
        # Level 3
        x = self.up3(x)
        x = self._match_size(x, x3)
        
        if self.attention_type in ['mkdc', 'mkdc_clgcn_cgat']:
            x3_mkdc = self.mkdc3(x3)
            
            if self.attention_type == 'mkdc_clgcn_cgat':
                x = self.dec3(x, x3_mkdc, x3_clgcn)
            else:
                x = torch.cat([x, x3_mkdc], dim=1)
                x = self.dec3(x)
        else:
            x = torch.cat([x, x3], dim=1)
            x = self.dec3(x)
        
        # Level 2
        x = self.up2(x)
        x = self._match_size(x, x2)
        
        if self.attention_type in ['mkdc', 'mkdc_clgcn_cgat']:
            x2_mkdc = self.mkdc2(x2)
            
            if self.attention_type == 'mkdc_clgcn_cgat':
                x = self.dec2(x, x2_mkdc, x2_clgcn)
            else:
                x = torch.cat([x, x2_mkdc], dim=1)
                x = self.dec2(x)
        else:
            x = torch.cat([x, x2], dim=1)
            x = self.dec2(x)
        
        # Level 1
        x = self.up1(x)
        x = self._match_size(x, x1)
        
        if self.attention_type in ['mkdc', 'mkdc_clgcn_cgat']:
            x1_mkdc = self.mkdc1(x1)
            
            if self.attention_type == 'mkdc_clgcn_cgat':
                x = self.dec1(x, x1_mkdc, x1_clgcn)
            else:
                x = torch.cat([x, x1_mkdc], dim=1)
                x = self.dec1(x)
        else:
            x = torch.cat([x, x1], dim=1)
            x = self.dec1(x)
        
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
    parser = argparse.ArgumentParser(description='Enhanced Segmentation Training with MKDC+CLGCN+CGAT')
    
    parser.add_argument('--pretrained-checkpoint', type=str, default=None)
    parser.add_argument('--attention', type=str, default='none', 
                       choices=['none', 'mkdc', 'mkdc_clgcn_cgat'],
                       help='Attention mechanism: none (baseline), mkdc, or mkdc_clgcn_cgat (full)')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/segmentation_enhanced_5fold')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
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
        'mkdc': 'MKDC (Multi-Kernel Depthwise Conv)',
        'mkdc_clgcn_cgat': 'MKDC + CLGCN + CGAT (Full Enhancement)'
    }
    
    print("\n" + "="*70)
    print(f"ENHANCED SEGMENTATION TRAINING - FOLD {args.fold}")
    print(f"Initialization: {init_method}")
    print(f"Attention: {attention_names[args.attention]}")
    print("="*70)
    print("✓ ROI-based validation (fair comparison)")
    print("✓ Robust mask handling (>0)")
    print("✓ Fixed: No double softmax bug")
    if args.attention == 'mkdc':
        print("✓ MKDC: Multi-scale spatial features")
    elif args.attention == 'mkdc_clgcn_cgat':
        print("✓ MKDC: Multi-scale spatial features")
        print("✓ CLGCN: Cross-layer structural relationships (learnable adjacency)")
        print("✓ CGAT: Channel-aware feature fusion")
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model: {total_params:,} parameters ({trainable_params:,} trainable)\n")
    
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
            'final_val_dsc_recon': float(val_dscs_recon[-1]),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }, f, indent=4)


if __name__ == '__main__':
    main()

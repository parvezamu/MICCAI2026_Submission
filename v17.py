"""
train_segmentation_with_prompts.py

PROMPT-ENHANCED stroke lesion segmentation with:
1. Learnable task-specific prompts (Fischer et al. 2024 style)
2. Multi-scale label supervision (boundary + interior)
3. Domain-specific prompts for ATLAS/UOA datasets
4. Frequency-domain augmentation
5. All previous optimizations (warmup, proper LR scaling, etc.)
6. NEW: Balanced dataset × lesion-size sampling

Based on: "Prompt Mechanisms in Medical Imaging: A Comprehensive Survey"

Author: Parvez (Enhanced with Prompt Engineering)
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys
from collections import defaultdict
import nibabel as nib
import matplotlib.pyplot as plt
import random
import math

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

# ============================================================================
# NEW: Import balanced sampler
# ============================================================================
from utils.balanced_sampler import BalancedDatasetLesionBatchSampler


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class GDiceLossV2(nn.Module):
    """Generalized Dice Loss V2"""
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


# ============================================================================
# PROMPT MECHANISMS
# ============================================================================

class LearnablePrompts(nn.Module):
    """
    Learnable task-specific prompts for stroke segmentation
    Based on: Fischer et al. 2024 - Prompt tuning for parameter-efficient segmentation
    """
    def __init__(self, num_prompts=10, prompt_dim=512, dropout=0.1):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        
        # Learnable prompt tokens
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, prompt_dim))
        
        # Prompt projection
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        nn.init.normal_(self.prompts, std=0.02)
        
        print(f"✓ Initialized {num_prompts} learnable prompts (dim={prompt_dim})")
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        prompts = self.prompts.expand(B, -1, -1)
        prompts = self.prompt_proj(prompts)
        
        prompt_encoding = prompts.mean(dim=1)
        prompt_encoding = prompt_encoding.view(B, C, 1, 1, 1)
        
        return x + prompt_encoding


class DomainPrompts(nn.Module):
    """
    Domain-specific prompts for ATLAS vs UOA datasets
    """
    def __init__(self, num_domains=2, embed_dim=64):
        super().__init__()
        
        self.num_domains = num_domains
        self.embed_dim = embed_dim
        
        self.domain_embeddings = nn.Embedding(num_domains, embed_dim)
        
        self.domain_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        print(f"✓ Initialized domain prompts for {num_domains} domains")
    
    def forward(self, x, domain_ids=None):
        if domain_ids is None:
            return x
        
        B, C = x.shape[:2]
        
        domain_embed = self.domain_embeddings(domain_ids)
        domain_embed = self.domain_mlp(domain_embed)
        
        if self.embed_dim < C:
            repeat_factor = (C + self.embed_dim - 1) // self.embed_dim
            domain_embed = domain_embed.repeat(1, repeat_factor)[:, :C]
        elif self.embed_dim > C:
            domain_embed = domain_embed[:, :C]
        
        domain_embed = domain_embed.view(B, C, 1, 1, 1)
        return x * (1 + domain_embed)


class MultiScaleLabelGenerator(nn.Module):
    """
    Generate auxiliary labels for multi-scale supervision
    """
    def __init__(self):
        super().__init__()
        
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        
        self.register_buffer('sobel_kernel', sobel_x)
        
        print("✓ Initialized multi-scale label generator")
    
    def forward(self, mask):
        mask_binary = (mask > 0.5).float()
        
        grad = F.conv3d(mask_binary, self.sobel_kernel, padding=1)
        boundary = (torch.abs(grad) > 0.1).float()
        
        mask_inv = 1 - mask_binary
        eroded_inv = F.max_pool3d(mask_inv, kernel_size=3, stride=1, padding=1)
        interior = 1 - eroded_inv
        
        interior = interior * (1 - boundary)
        
        return boundary, interior


class MultiScaleSupervisionLoss(nn.Module):
    """Multi-scale supervision loss"""
    def __init__(self, weights=[1.0, 0.5, 0.5]):
        super().__init__()
        
        self.weights = weights
        self.label_generator = MultiScaleLabelGenerator()
        self.criterion = GDiceLossV2(apply_nonlin=softmax_helper)
        
        print(f"✓ Multi-scale supervision weights: {weights}")
    
    def forward(self, pred_main, pred_boundary, pred_interior, mask):
        mask_float = (mask.unsqueeze(1) > 0).float()
        boundary_gt, interior_gt = self.label_generator(mask_float)
        
        boundary_gt = boundary_gt.squeeze(1).long()
        interior_gt = interior_gt.squeeze(1).long()
        
        loss_main = self.criterion(pred_main, mask)
        loss_boundary = self.criterion(pred_boundary, boundary_gt)
        loss_interior = self.criterion(pred_interior, interior_gt)
        
        total_loss = (self.weights[0] * loss_main + 
                     self.weights[1] * loss_boundary + 
                     self.weights[2] * loss_interior)
        
        return total_loss, {
            'main': loss_main.item(),
            'boundary': loss_boundary.item(),
            'interior': interior_gt.item()
        }


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class ECAAttention(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels: int):
        super().__init__()
        
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
    """Multi-Kernel Depthwise Convolution"""
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


# ============================================================================
# ENCODER
# ============================================================================

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


# ============================================================================
# DECODER
# ============================================================================

class PromptEnhancedDecoder3D(nn.Module):
    """Enhanced U-Net decoder with prompts and multi-scale supervision"""
    def __init__(self, num_classes=2, attention_type='none', 
                 use_prompts=False, num_prompts=10,
                 multi_scale_supervision=False):
        super(PromptEnhancedDecoder3D, self).__init__()
        
        self.attention_type = attention_type
        self.use_prompts = use_prompts
        self.multi_scale_supervision = multi_scale_supervision
        
        # Learnable prompts
        if use_prompts:
            self.prompts_level4 = LearnablePrompts(num_prompts, 256)
            self.prompts_level3 = LearnablePrompts(num_prompts, 128)
            self.prompts_level2 = LearnablePrompts(num_prompts, 64)
            self.prompts_level1 = LearnablePrompts(num_prompts, 64)
        
        # Attention modules
        if attention_type == 'eca':
            self.eca4 = ECAAttention(256)
            self.eca3 = ECAAttention(128)
            self.eca2 = ECAAttention(64)
            self.eca1 = ECAAttention(64)
        elif attention_type == 'mkdc':
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64)
        
        # Decoder blocks
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
        
        # Multi-scale supervision heads
        if multi_scale_supervision:
            self.ds_conv4 = nn.Conv3d(256, num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(128, num_classes, kernel_size=1)
            self.ds_conv2 = nn.Conv3d(64, num_classes, kernel_size=1)
            
            self.boundary_conv4 = nn.Conv3d(256, num_classes, kernel_size=1)
            self.boundary_conv3 = nn.Conv3d(128, num_classes, kernel_size=1)
            self.boundary_conv2 = nn.Conv3d(64, num_classes, kernel_size=1)
            self.boundary_conv_final = nn.Conv3d(64, num_classes, kernel_size=1)
            
            self.interior_conv4 = nn.Conv3d(256, num_classes, kernel_size=1)
            self.interior_conv3 = nn.Conv3d(128, num_classes, kernel_size=1)
            self.interior_conv2 = nn.Conv3d(64, num_classes, kernel_size=1)
            self.interior_conv_final = nn.Conv3d(64, num_classes, kernel_size=1)
    
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
        
        outputs = {}
        
        # Level 4
        x = self.up4(x5)
        x = self._match_size(x, x4)
        if self.attention_type == 'mkdc':
            x4 = self.mkdc4(x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.use_prompts:
            x = self.prompts_level4(x)
        if self.attention_type == 'eca':
            x = self.eca4(x)
        
        if self.multi_scale_supervision:
            outputs['ds4'] = self.ds_conv4(x)
            outputs['boundary4'] = self.boundary_conv4(x)
            outputs['interior4'] = self.interior_conv4(x)
        
        # Level 3
        x = self.up3(x)
        x = self._match_size(x, x3)
        if self.attention_type == 'mkdc':
            x3 = self.mkdc3(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.use_prompts:
            x = self.prompts_level3(x)
        if self.attention_type == 'eca':
            x = self.eca3(x)
        
        if self.multi_scale_supervision:
            outputs['ds3'] = self.ds_conv3(x)
            outputs['boundary3'] = self.boundary_conv3(x)
            outputs['interior3'] = self.interior_conv3(x)
        
        # Level 2
        x = self.up2(x)
        x = self._match_size(x, x2)
        if self.attention_type == 'mkdc':
            x2 = self.mkdc2(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.use_prompts:
            x = self.prompts_level2(x)
        if self.attention_type == 'eca':
            x = self.eca2(x)
        
        if self.multi_scale_supervision:
            outputs['ds2'] = self.ds_conv2(x)
            outputs['boundary2'] = self.boundary_conv2(x)
            outputs['interior2'] = self.interior_conv2(x)
        
        # Level 1
        x = self.up1(x)
        x = self._match_size(x, x1)
        if self.attention_type == 'mkdc':
            x1 = self.mkdc1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        if self.use_prompts:
            x = self.prompts_level1(x)
        if self.attention_type == 'eca':
            x = self.eca1(x)
        
        # Final outputs
        outputs['main'] = self.final_conv(x)
        
        if self.multi_scale_supervision:
            outputs['boundary'] = self.boundary_conv_final(x)
            outputs['interior'] = self.interior_conv_final(x)
        
        return outputs


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class PromptSegmentationModel(nn.Module):
    """Complete prompt-enhanced segmentation model"""
    def __init__(self, encoder, num_classes=2, attention_type='none',
                 use_prompts=False, num_prompts=10,
                 use_domain_prompts=False, num_domains=2,
                 multi_scale_supervision=False):
        super(PromptSegmentationModel, self).__init__()
        
        self.encoder = ResNet3DEncoder(encoder)
        
        self.use_domain_prompts = use_domain_prompts
        if use_domain_prompts:
            self.domain_prompts = DomainPrompts(num_domains=num_domains, embed_dim=64)
        
        self.decoder = PromptEnhancedDecoder3D(
            num_classes=num_classes,
            attention_type=attention_type,
            use_prompts=use_prompts,
            num_prompts=num_prompts,
            multi_scale_supervision=multi_scale_supervision
        )
        
        self.multi_scale_supervision = multi_scale_supervision
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if use_prompts:
            prompt_params = sum(p.numel() for n, p in self.named_parameters() 
                              if 'prompts' in n and p.requires_grad)
            print(f"✓ Prompt parameters: {prompt_params:,} ({prompt_params/trainable_params*100:.2f}% of trainable)")
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
    
    def forward(self, x, domain_ids=None):
        input_size = x.shape[2:]
        
        if self.use_domain_prompts and domain_ids is not None:
            x = self.domain_prompts(x, domain_ids)
        
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        
        for key in outputs:
            if outputs[key].shape[2:] != input_size:
                outputs[key] = F.interpolate(
                    outputs[key], size=input_size,
                    mode='trilinear', align_corners=False
                )
        
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Prompt-Enhanced Stroke Lesion Segmentation'
    )
    
    # Model configuration
    parser.add_argument('--pretrained-checkpoint', type=str, default=None)
    parser.add_argument('--attention', type=str, default='none', 
                       choices=['none', 'eca', 'mkdc'])
    parser.add_argument('--deep-supervision', action='store_true')
    parser.add_argument('--ds-weights', type=float, nargs=4, 
                       default=[1.0, 0.5, 0.25, 0.125])
    
    # PROMPT ENGINEERING OPTIONS
    parser.add_argument('--use-prompts', action='store_true',
                       help='Enable learnable task-specific prompts')
    parser.add_argument('--num-prompts', type=int, default=10,
                       help='Number of learnable prompt tokens')
    parser.add_argument('--use-domain-prompts', action='store_true',
                       help='Enable domain-specific prompts for ATLAS/UOA')
    parser.add_argument('--multi-scale-supervision', action='store_true',
                       help='Enable multi-scale supervision (boundary + interior)')
    parser.add_argument('--ms-weights', type=float, nargs=3, 
                       default=[1.0, 0.5, 0.5],
                       help='Weights for multi-scale: [main, boundary, interior]')
    
    # ============================================================================
    # NEW: Balanced sampling option
    # ============================================================================
    parser.add_argument('--balanced-sampling', action='store_true',
                       help='Use balanced dataset×lesion-size sampling to prevent shortcuts')
    
    # Multi-run support
    parser.add_argument('--run-id', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    
    # Data configuration
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/segmentation_with_prompts')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10)
    parser.add_argument('--lesion-focus-ratio', type=float, default=0.7)
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--base-lr', type=float, default=0.0001)
    parser.add_argument('--base-batch-size', type=int, default=8)
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    
    # Validation
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
    
    # Compute effective LR
    if args.lr is None:
        args.lr = compute_effective_lr(
            batch_size=args.batch_size,
            base_batch_size=args.base_batch_size,
            base_lr=args.base_lr
        )
    
    # Print configuration
    init_method = "SimCLR_Pretrained" if args.pretrained_checkpoint else "Random_Init"
    
    attention_names = {
        'none': 'Baseline',
        'eca': 'ECA',
        'mkdc': 'MKDC'
    }
    
    print("\n" + "="*70)
    print(f"PROMPT-ENHANCED SEGMENTATION - FOLD {args.fold}")
    if args.run_id is not None:
        print(f"Run ID: {args.run_id}")
    print(f"Random Seed: {args.seed}")
    print("="*70)
    print(f"✓ Initialization: {init_method}")
    print(f"✓ Attention: {attention_names[args.attention]}")
    print(f"✓ Deep Supervision: {'ENABLED' if args.deep_supervision else 'DISABLED'}")
    print(f"✓ Learnable Prompts: {'ENABLED' if args.use_prompts else 'DISABLED'}")
    if args.use_prompts:
        print(f"  → {args.num_prompts} prompt tokens")
    print(f"✓ Domain Prompts: {'ENABLED' if args.use_domain_prompts else 'DISABLED'}")
    print(f"✓ Multi-Scale Supervision: {'ENABLED' if args.multi_scale_supervision else 'DISABLED'}")
    if args.multi_scale_supervision:
        print(f"  → Weights: {args.ms_weights}")
    print(f"✓ Balanced Sampling: {'ENABLED' if args.balanced_sampling else 'DISABLED'}")  # NEW
    print("="*70)
    print(f"✓ Batch size: {args.batch_size}")
    print(f"✓ Learning rate: {args.lr:.6f}")
    print(f"✓ Optimizer: {args.optimizer.upper()}")
    print(f"✓ Warmup epochs: {args.warmup_epochs}")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create experiment directory
    exp_name = f"{init_method}_{args.attention}"
    if args.use_prompts:
        exp_name += f"_prompts{args.num_prompts}"
    if args.use_domain_prompts:
        exp_name += "_domain"
    if args.multi_scale_supervision:
        exp_name += "_ms"
    if args.deep_supervision:
        exp_name += "_ds"
    if args.balanced_sampling:  # NEW
        exp_name += "_balanced"
    
    if args.run_id is not None:
        exp_dir = Path(args.output_dir) / exp_name / f'fold_{args.fold}' / \
                  f'run_{args.run_id}' / f'exp_{timestamp}'
    else:
        exp_dir = Path(args.output_dir) / exp_name / f'fold_{args.fold}' / \
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
    
    if args.pretrained_checkpoint:
        print("Loading SimCLR pretrained encoder...")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
        simclr_model.load_state_dict(checkpoint['model_state_dict'])
        encoder = simclr_model.encoder
        print(f"✓ Loaded from epoch {checkpoint['epoch'] + 1}\n")
    else:
        print("Using random initialization\n")
    
    model = PromptSegmentationModel(
        encoder,
        num_classes=2,
        attention_type=args.attention,
        use_prompts=args.use_prompts,
        num_prompts=args.num_prompts,
        use_domain_prompts=args.use_domain_prompts,
        num_domains=2,
        multi_scale_supervision=args.multi_scale_supervision
    ).to(device)
    
    # ============================================================================
    # Load data with lesion binning support
    # ============================================================================
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
        lesion_focus_ratio=args.lesion_focus_ratio,
        compute_lesion_bins=args.balanced_sampling  # Only compute if needed
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
        lesion_focus_ratio=args.lesion_focus_ratio,
        compute_lesion_bins=False  # Not needed for validation
    )
    
    # ============================================================================
    # NEW: Create DataLoader with optional balanced sampling
    # ============================================================================
    if args.balanced_sampling:
        print("\n" + "="*70)
        print("Creating BALANCED batch sampler...")
        print("="*70)
        
        balanced_sampler = BalancedDatasetLesionBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            datasets=['ATLAS', 'UOA_Private'],
            bins=(0, 1, 2),
            seed=args.seed,
            drop_last=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,  # Use batch_sampler instead of shuffle
            num_workers=4,
            pin_memory=True
        )
        
        print("✓ Using balanced sampling (prevents dataset×lesion-size shortcuts)")
        print("="*70 + "\n")
    else:
        print("Using standard random sampling...\n")
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
    
    # Setup losses
    criterion_main = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    criterion_ms = None
    if args.multi_scale_supervision:
        criterion_ms = MultiScaleSupervisionLoss(weights=args.ms_weights)
    
    # Setup optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    
    # Setup scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr
    )
    
    scaler = GradScaler()
    
    print("\n" + "="*70)
    print(f"Starting Training")
    print("="*70 + "\n")
    
    train_losses, train_dscs, val_dscs_patch, val_dscs_recon = [], [], [], []
    loss_components_history = defaultdict(list)
    best_dsc, best_epoch = 0, 0
    
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        header = "epoch,train_loss,train_dsc,val_dsc_patch,val_dsc_recon,lr"
        if args.multi_scale_supervision:
            header += ",loss_main,loss_boundary,loss_interior"
        f.write(header + "\n")
    
    for epoch in range(args.epochs):
        current_lr = scheduler.step(epoch)
        
        train_loss, train_dsc, loss_components = train_epoch(
            model, train_loader, criterion_main, criterion_ms, optimizer, 
            device, epoch, scaler,
            use_domain_prompts=args.use_domain_prompts,
            deep_supervision=args.deep_supervision,
            multi_scale_supervision=args.multi_scale_supervision,
            ds_weights=args.ds_weights,
            ms_weights=args.ms_weights,
            max_grad_norm=args.max_grad_norm,
            debug_first_batch=(epoch == 0)
        )
        
        # Track loss components
        for k, v in loss_components.items():
            loss_components_history[k].append(v)
        
        val_loss_patch, val_dsc_patch = validate_patches(
            model, val_loader, criterion_main, device,
            use_domain_prompts=args.use_domain_prompts
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
                use_domain_prompts=args.use_domain_prompts,
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
            line = f"{epoch+1},{train_loss:.6f},{train_dsc:.6f}," \
                   f"{val_dsc_patch:.6f},{val_dsc_recon:.6f},{current_lr:.6f}"
            if args.multi_scale_supervision:
                line += f",{loss_components.get('main', 0):.6f}," \
                       f"{loss_components.get('boundary', 0):.6f}," \
                       f"{loss_components.get('interior', 0):.6f}"
            f.write(line + "\n")
        
        print(f"{'='*70}")
        print(f"Fold {args.fold} - Epoch {epoch+1}/{args.epochs}")
        if args.run_id is not None:
            print(f"  [Run {args.run_id}]")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val DSC (patch):  {val_dsc_patch:.4f}")
        print(f"  Val DSC (recon):  {val_dsc_recon:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if args.multi_scale_supervision:
            print(f"  Loss components:")
            for k, v in loss_components.items():
                print(f"    {k}: {v:.4f}")
        
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
                'val_dsc_recon': val_dsc_recon,
                'config': vars(args)
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        if (epoch + 1) % 5 == 0:
            plot_training_curves(
                train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                exp_dir / f'curves_epoch_{epoch+1}.png',
                loss_components=loss_components_history if args.multi_scale_supervision else None
            )
    
    print("\n" + "="*70)
    print(f"FOLD {args.fold} TRAINING COMPLETE!")
    if args.run_id is not None:
        print(f"Run {args.run_id}")
    print("="*70)
    print(f"Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"Results: {exp_dir}")
    print("="*70 + "\n")
    
    plot_training_curves(
        train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
        exp_dir / 'curves_final.png',
        loss_components=loss_components_history if args.multi_scale_supervision else None
    )
    
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump({
            'fold': args.fold,
            'run_id': args.run_id,
            'seed': args.seed,
            'configuration': exp_name,
            'best_epoch': best_epoch,
            'best_dsc': float(best_dsc),
            'final_train_dsc': float(train_dscs[-1]),
            'final_val_dsc_recon': float(val_dscs_recon[-1]),
            'prompts_enabled': args.use_prompts,
            'domain_prompts_enabled': args.use_domain_prompts,
            'multi_scale_supervision': args.multi_scale_supervision,
            'balanced_sampling': args.balanced_sampling  # NEW
        }, f, indent=4)


# ============================================================================
# MISSING UTILITY FUNCTIONS - ADD THESE BEFORE main()
# ============================================================================

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first."""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


def softmax_helper(x):
    return F.softmax(x, dim=1)


def compute_effective_lr(batch_size, base_batch_size=8, base_lr=0.0001):
    """Compute effective learning rate using linear scaling rule"""
    scaling_factor = batch_size / base_batch_size
    effective_lr = base_lr * scaling_factor
    
    print(f"\n{'='*70}")
    print(f"LEARNING RATE SCALING")
    print(f"{'='*70}")
    print(f"Base: batch_size={base_batch_size}, lr={base_lr}")
    print(f"Current: batch_size={batch_size}")
    print(f"Scaling: {scaling_factor:.2f}×")
    print(f"Effective LR: {effective_lr:.6f}")
    print(f"{'='*70}\n")
    
    return effective_lr


def compute_dsc(pred, target, smooth=1e-6):
    """Compute Dice Score Coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        print(f"✓ Warmup scheduler: {warmup_epochs} warmup epochs, {total_epochs} total epochs")
        print(f"  Base LR: {base_lr:.6f}, Min LR: {min_lr:.6f}")
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def get_domain_id(batch, dataset_names=['ATLAS', 'UOA_Private']):
    """
    Extract domain ID from batch
    Returns tensor of domain IDs (0=ATLAS, 1=UOA)
    """
    if 'dataset' in batch:
        datasets = batch['dataset']
        domain_ids = torch.zeros(len(datasets), dtype=torch.long)
        for i, ds in enumerate(datasets):
            if 'UOA' in ds or 'uoa' in ds:
                domain_ids[i] = 1
        return domain_ids
    return None


def compute_multi_scale_loss(outputs, target, criterion_main, criterion_ms=None,
                            deep_supervision=False, multi_scale_supervision=False,
                            ds_weights=[1.0, 0.5, 0.25, 0.125],
                            ms_weights=[1.0, 0.5, 0.5]):
    """
    Compute loss with optional deep supervision and multi-scale supervision
    """
    losses = {}
    total_loss = 0.0
    
    # Main prediction loss
    main_output = outputs['main']
    loss_main = criterion_main(main_output, target)
    losses['main'] = loss_main.item()
    total_loss += loss_main
    
    # Deep supervision (standard auxiliary losses)
    if deep_supervision and 'ds4' in outputs:
        for i, level in enumerate(['ds4', 'ds3', 'ds2']):
            if level in outputs:
                loss_ds = criterion_main(outputs[level], target)
                losses[level] = loss_ds.item()
                total_loss += ds_weights[i+1] * loss_ds
    
    # Multi-scale supervision (boundary + interior)
    if multi_scale_supervision and criterion_ms is not None:
        # Only use final level predictions for boundary/interior
        pred_boundary = outputs.get('boundary', None)
        pred_interior = outputs.get('interior', None)
        
        if pred_boundary is not None and pred_interior is not None:
            loss_ms, ms_details = criterion_ms(
                main_output, pred_boundary, pred_interior, target
            )
            losses.update({f'ms_{k}': v for k, v in ms_details.items()})
            total_loss += ms_weights[0] * loss_ms
    
    return total_loss, losses


def train_epoch(model, dataloader, criterion_main, criterion_ms, optimizer, 
                device, epoch, scaler, use_domain_prompts=False,
                deep_supervision=False, multi_scale_supervision=False,
                ds_weights=[1.0, 0.5, 0.25, 0.125],
                ms_weights=[1.0, 0.5, 0.5],
                max_grad_norm=1.0, debug_first_batch=False):
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    loss_components = defaultdict(float)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        masks = (masks > 0).long()
        
        # Get domain IDs if using domain prompts
        domain_ids = None
        if use_domain_prompts:
            domain_ids = get_domain_id(batch)
            if domain_ids is not None:
                domain_ids = domain_ids.to(device)
        
        if debug_first_batch and batch_idx == 0:
            print("\n" + "="*70)
            print("DEBUG: First batch")
            print("="*70)
            print(f"Image shape: {images.shape}")
            print(f"Mask unique: {torch.unique(masks)}")
            print(f"Lesion voxels: {(masks > 0).sum().item()}")
            if domain_ids is not None:
                print(f"Domain IDs: {domain_ids}")
            print("="*70 + "\n")
        
        with autocast():
            outputs = model(images, domain_ids=domain_ids)
            
            loss, losses_dict = compute_multi_scale_loss(
                outputs, masks, criterion_main, criterion_ms,
                deep_supervision=deep_supervision,
                multi_scale_supervision=multi_scale_supervision,
                ds_weights=ds_weights,
                ms_weights=ms_weights
            )
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Compute DSC
        with torch.no_grad():
            main_output = outputs['main']
            pred_probs = torch.softmax(main_output, dim=1)[:, 1:2, ...]
            target_onehot = (masks.unsqueeze(1) > 0).float()
            dsc = compute_dsc(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        # Accumulate loss components
        for k, v in losses_dict.items():
            loss_components[k] += v
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dsc': f'{dsc:.4f}'
        })
    
    # Average loss components
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return total_loss / num_batches, total_dsc / num_batches, dict(loss_components)


def validate_patches(model, dataloader, criterion_main, device, 
                    use_domain_prompts=False):
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating (patches)'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            masks = (masks > 0).long()
            
            domain_ids = None
            if use_domain_prompts:
                domain_ids = get_domain_id(batch)
                if domain_ids is not None:
                    domain_ids = domain_ids.to(device)
            
            with autocast():
                outputs = model(images, domain_ids=domain_ids)
                main_output = outputs['main']
                loss = criterion_main(main_output, masks)
            
            pred_probs = torch.softmax(main_output, dim=1)[:, 1:2, ...]
            target_onehot = (masks.unsqueeze(1) > 0).float()
            dsc = compute_dsc(pred_probs, target_onehot)
            
            total_loss += loss.item()
            total_dsc += dsc
            num_batches += 1
    
    return total_loss / num_batches, total_dsc / num_batches


def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, 
                                       patch_size=(96, 96, 96)):
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


def validate_full_volumes(model, dataset, device, patch_size, 
                         use_domain_prompts=False,
                         save_nifti=False, save_dir=None, epoch=None):
    model.eval()
    
    volume_data = defaultdict(lambda: {'centers': [], 'preds': [], 'domain_id': None})
    
    print("\nCollecting patches for full reconstruction...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Processing patches'):
            batch = dataset[idx]
            
            vol_idx = batch['vol_idx']
            if isinstance(vol_idx, torch.Tensor):
                vol_idx = vol_idx.item()
            
            image = batch['image'].unsqueeze(0).to(device)
            center = batch['center'].numpy()
            
            # Get domain ID
            domain_id = None
            if use_domain_prompts:
                domain_ids = get_domain_id({k: [v] for k, v in batch.items()})
                if domain_ids is not None:
                    domain_id = domain_ids[0].item()
                    domain_id_tensor = domain_ids.to(device)
            
            with autocast():
                outputs = model(image, domain_ids=domain_id_tensor if use_domain_prompts else None)
                output = outputs['main']
            
            pred = torch.softmax(output, dim=1).cpu().numpy()
            
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
            if domain_id is not None:
                volume_data[vol_idx]['domain_id'] = domain_id
    
    print("\nReconstructing volumes...")
    all_dscs = []
    domain_dscs = defaultdict(list)
    
    if save_nifti and save_dir:
        nifti_dir = Path(save_dir) / f'reconstructions_epoch_{epoch}'
        nifti_dir.mkdir(parents=True, exist_ok=True)
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Computing DSCs'):
        vol_info = dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        domain_id = volume_data[vol_idx]['domain_id']
        
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
        
        # Track per-domain performance
        if domain_id is not None:
            domain_name = 'ATLAS' if domain_id == 0 else 'UOA'
            domain_dscs[domain_name].append(dsc)
        
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
            
            with open(case_dir / 'metadata.json', 'w') as f:
                json.dump({
                    'case_id': case_id,
                    'dsc': float(dsc),
                    'epoch': epoch,
                    'domain': 'ATLAS' if domain_id == 0 else 'UOA' if domain_id == 1 else 'unknown',
                    'lesion_volume': int(mask_gt.sum()),
                    'pred_volume': int(pred_in_roi.sum())
                }, f, indent=4)
    
    mean_dsc = np.mean(all_dscs)
    
    # Print domain-specific performance
    if domain_dscs:
        print("\nDomain-specific performance:")
        for domain, dscs in domain_dscs.items():
            print(f"  {domain}: {np.mean(dscs):.4f} ± {np.std(dscs):.4f} (n={len(dscs)})")
    
    if save_nifti and save_dir:
        summary = {
            'epoch': epoch,
            'mean_dsc': float(mean_dsc),
            'std_dsc': float(np.std(all_dscs)),
            'all_dscs': [float(d) for d in all_dscs]
        }
        
        if domain_dscs:
            summary['domain_performance'] = {
                domain: {
                    'mean': float(np.mean(dscs)),
                    'std': float(np.std(dscs)),
                    'n': len(dscs)
                }
                for domain, dscs in domain_dscs.items()
            }
        
        with open(nifti_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
    
    return mean_dsc, all_dscs


def plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon, 
                        save_path, loss_components=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Progress - Prompt-Enhanced Segmentation', 
                fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # DSC
    axes[0, 1].plot(epochs, train_dscs, 'b-', linewidth=2, label='Train')
    axes[0, 1].plot(epochs, val_dscs_patch, 'r--', linewidth=2, label='Val (patch)')
    axes[0, 1].plot(epochs, val_dscs_recon, 'g-', linewidth=3, label='Val (recon)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('DSC')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Best DSC
    best_recon = [max(val_dscs_recon[:i+1]) for i in range(len(epochs))]
    axes[0, 2].plot(epochs, best_recon, 'g-', linewidth=3, marker='*', markersize=8)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Best DSC')
    axes[0, 2].set_title('Best Full-Volume DSC')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Generalization gap
    gap_recon = [train_dscs[i] - val_dscs_recon[i] for i in range(len(epochs))]
    axes[1, 0].plot(epochs, gap_recon, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val DSC')
    axes[1, 0].set_title('Generalization Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss components (if available)
    if loss_components:
        for key in ['main', 'boundary', 'interior']:
            if key in loss_components:
                values = [loss_components[key][i] for i in range(len(epochs))]
                axes[1, 1].plot(epochs, values, linewidth=2, label=key)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Component')
        axes[1, 1].set_title('Multi-Scale Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Validation stability
    window = 10
    val_std = [np.std(val_dscs_recon[max(0, i-window):i+1]) 
              for i in range(len(epochs))]
    axes[1, 2].plot(epochs, val_std, 'g-', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Val DSC Std (rolling)')
    axes[1, 2].set_title('Validation Stability')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()

"""
finetune_on_isles_FIXED.py

FIXED fine-tuning with:
- Discriminative learning rates (encoder 10× slower)
- Shorter freeze period (3 epochs)
- Better optimizer strategy
- Comprehensive logging
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys
import random
import math

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

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


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first."""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


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


def softmax_helper(x):
    return torch.nn.functional.softmax(x, dim=1)


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
    """U-Net decoder with attention support"""
    def __init__(self, num_classes=2, attention_type='none', mkdc_kernels=[1, 3, 5],
                 deep_supervision=False):
        super(UNetDecoder3D, self).__init__()
        
        self.attention_type = attention_type
        self.deep_supervision = deep_supervision
        
        if attention_type == 'eca':
            self.eca4 = ECAAttention(256)
            self.eca3 = ECAAttention(128)
            self.eca2 = ECAAttention(64)
            self.eca1 = ECAAttention(64)
        elif attention_type == 'mkdc':
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256, kernels=mkdc_kernels)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128, kernels=mkdc_kernels)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
        
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
        
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
                x_up = torch.nn.functional.pad(x_up, padding)
            elif diff_d < 0 or diff_h < 0 or diff_w < 0:
                d_start = max(0, -diff_d // 2)
                h_start = max(0, -diff_h // 2)
                w_start = max(0, -diff_w // 2)
                x_up = x_up[:, :, d_start:d_start + d_skip, 
                           h_start:h_start + h_skip, w_start:w_start + w_skip]
        
        return x_up
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        ds_outputs = []
        
        x = self.up4(x5)
        x = self._match_size(x, x4)
        if self.attention_type == 'mkdc':
            x4 = self.mkdc4(x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.attention_type == 'eca':
            x = self.eca4(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv4(x))
        
        x = self.up3(x)
        x = self._match_size(x, x3)
        if self.attention_type == 'mkdc':
            x3 = self.mkdc3(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.attention_type == 'eca':
            x = self.eca3(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv3(x))
        
        x = self.up2(x)
        x = self._match_size(x, x2)
        if self.attention_type == 'mkdc':
            x2 = self.mkdc2(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.attention_type == 'eca':
            x = self.eca2(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv2(x))
        
        x = self.up1(x)
        x = self._match_size(x, x1)
        if self.attention_type == 'mkdc':
            x1 = self.mkdc1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        if self.attention_type == 'eca':
            x = self.eca1(x)
        
        final_output = self.final_conv(x)
        
        if self.deep_supervision:
            return [final_output] + ds_outputs
        else:
            return final_output


class SegmentationModel(nn.Module):
    """Complete segmentation model"""
    def __init__(self, encoder, num_classes=2, attention_type='none', deep_supervision=False):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes, 
                                     attention_type=attention_type,
                                     deep_supervision=deep_supervision)
        self.attention_type = attention_type
        self.deep_supervision = deep_supervision
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        
        if self.deep_supervision:
            resized_outputs = []
            for out in outputs:
                if out.shape[2:] != input_size:
                    out = torch.nn.functional.interpolate(out, size=input_size, 
                                      mode='trilinear', align_corners=False)
                resized_outputs.append(out)
            return resized_outputs
        else:
            if outputs.shape[2:] != input_size:
                outputs = torch.nn.functional.interpolate(outputs, size=input_size, 
                                      mode='trilinear', align_corners=False)
            return outputs


class DiscriminativeLRScheduler:
    """
    Custom scheduler that maintains DIFFERENT learning rates for encoder and decoder
    while applying warmup + cosine annealing proportionally
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 encoder_base_lr, decoder_base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.encoder_base_lr = encoder_base_lr
        self.decoder_base_lr = decoder_base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        print(f"\n{'='*80}")
        print(f"DISCRIMINATIVE LR SCHEDULER")
        print(f"{'='*80}")
        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Encoder base LR: {encoder_base_lr:.6f}")
        print(f"  Decoder base LR: {decoder_base_lr:.6f}")
        print(f"  Min LR: {min_lr:.6f}")
        print(f"  Ratio (encoder/decoder): {encoder_base_lr/decoder_base_lr:.2f}")
        print(f"{'='*80}\n")
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            encoder_lr = self.encoder_base_lr * warmup_factor
            decoder_lr = self.decoder_base_lr * warmup_factor
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            encoder_lr = self.min_lr + (self.encoder_base_lr - self.min_lr) * cosine_factor
            decoder_lr = self.min_lr + (self.decoder_base_lr - self.min_lr) * cosine_factor
        
        # Update param groups
        self.optimizer.param_groups[0]['lr'] = encoder_lr  # Encoder
        self.optimizer.param_groups[1]['lr'] = decoder_lr  # Decoder
        
        return encoder_lr, decoder_lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def compute_dsc_torch(pred_binary, mask_gt):
    """DSC calculation for PyTorch tensors (batch-aware)"""
    batch_size = pred_binary.shape[0]
    
    if batch_size == 0:
        return 0.0
    
    dsc_scores = []
    
    for i in range(batch_size):
        pred_i = pred_binary[i]
        gt_i = mask_gt[i]
        
        intersection = (pred_i * gt_i).sum()
        union = pred_i.sum() + gt_i.sum()
        
        if union == 0:
            dsc = 1.0
        elif intersection == 0:
            dsc = 0.0
        else:
            dsc = (2.0 * intersection) / union
        
        dsc_scores.append(dsc.item() if isinstance(dsc, torch.Tensor) else dsc)
    
    return sum(dsc_scores) / len(dsc_scores)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, 
                deep_supervision=False, ds_weights=[1.0, 0.0, 0.0, 0.0], max_grad_norm=1.0):
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        masks = (masks > 0).long()
        
        with autocast():
            outputs = model(images)
            
            if deep_supervision:
                loss = 0.0
                for i, (out, weight) in enumerate(zip(outputs, ds_weights)):
                    if out.shape[2:] != masks.shape[1:]:
                        target_resized = torch.nn.functional.interpolate(
                            masks.unsqueeze(1).float(),
                            size=out.shape[2:],
                            mode='nearest'
                        ).squeeze(1).long()
                    else:
                        target_resized = masks
                    loss += weight * criterion(out, target_resized)
                main_output = outputs[0]
            else:
                if outputs.shape[2:] != masks.shape[1:]:
                    masks_resized = torch.nn.functional.interpolate(
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
                masks_for_dsc = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=main_output.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                masks_for_dsc = masks
                
            pred_probs = torch.softmax(main_output, dim=1)[:, 1:2, ...]
            target_onehot = (masks_for_dsc.unsqueeze(1) > 0).float()
            dsc = compute_dsc_torch(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / num_batches, total_dsc / num_batches


def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """Reconstruct volume from patches"""
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


def load_pretrained_model(checkpoint_path, device):
    """Load pre-trained model (supports both supervised and SimCLR)"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint.get('model_state_dict', {})
    
    has_projection_head = any('projection_head' in k for k in state_dict.keys())
    has_decoder = any('decoder' in k for k in state_dict.keys())
    
    is_simclr = has_projection_head and not has_decoder
    
    if is_simclr:
        print(f"\n{'='*80}")
        print(f"DETECTED: SimCLR Checkpoint")
        print(f"{'='*80}")
        
        attention_type = 'none'
        deep_supervision = False
        
        base_encoder = resnet3d_18(in_channels=1)
        
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value
        
        base_encoder.load_state_dict(encoder_state_dict)
        
        model = SegmentationModel(
            encoder=base_encoder, 
            num_classes=2,
            attention_type=attention_type,
            deep_supervision=deep_supervision
        )
        model = model.to(device)
        
        # Print checkpoint info
        config = checkpoint.get('config', {})
        datasets = config.get('data', {}).get('datasets', ['Unknown'])
        
        print(f"  Pretraining data: {datasets}")
        print(f"  SimCLR epoch: {checkpoint.get('epoch', 'N/A') + 1}")
        print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}" if 'train_loss' in checkpoint else "")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "")
        print(f"  ✓ Encoder loaded from SimCLR pretraining")
        print(f"  ⚠️  Decoder randomly initialized")
        print(f"{'='*80}\n")
        
    else:
        print(f"\n{'='*80}")
        print(f"DETECTED: Supervised Checkpoint")
        print(f"{'='*80}")
        
        attention_type = checkpoint.get('attention_type', 'none')
        deep_supervision = checkpoint.get('deep_supervision', False)
        
        encoder = resnet3d_18(in_channels=1)
        model = SegmentationModel(
            encoder=encoder, 
            num_classes=2,
            attention_type=attention_type,
            deep_supervision=deep_supervision
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A') + 1}")
        print(f"  Val DSC: {checkpoint.get('val_dsc_recon', 'N/A'):.4f}" if 'val_dsc_recon' in checkpoint else "")
        print(f"  Attention: {attention_type}")
        print(f"  Deep supervision: {deep_supervision}")
        print(f"  ✓ Full model loaded")
        print(f"{'='*80}\n")
    
    return model, attention_type, deep_supervision


def validate_isles_reconstruction(model, isles_val_dataset, device, patch_size, deep_supervision=False):
    """Validate on ISLES with full volume reconstruction"""
    from collections import defaultdict
    from torch.utils.data import DataLoader
    
    model.eval()
    
    num_volumes = len(isles_val_dataset.volumes)
    
    print(f"\n{'='*80}")
    print(f"ISLES VALIDATION (Full Volume Reconstruction)")
    print(f"{'='*80}")
    print(f"Validation volumes: {num_volumes}")
    print(f"Total patches: {len(isles_val_dataset)}")
    
    if num_volumes == 0:
        return 0.0, []
    
    dataloader = DataLoader(
        isles_val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("Processing patches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
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
    
    print("Reconstructing volumes...")
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = isles_val_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed, count_map = reconstruct_from_patches_with_count(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        pred_binary = (reconstructed > 0.5).astype(np.uint8)
        mask_gt = (vol_info['mask'] > 0).astype(np.uint8)
        
        intersection = (pred_binary * mask_gt).sum()
        union = pred_binary.sum() + mask_gt.sum()
        
        dsc = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary.sum() == 0 else 0.0)
        
        all_dscs.append(dsc)
    
    mean_dsc = np.mean(all_dscs) if len(all_dscs) > 0 else 0.0
    
    print(f"\n📊 Validation Results:")
    print(f"  Mean DSC: {mean_dsc:.4f}")
    if len(all_dscs) > 0:
        print(f"  Std DSC: {np.std(all_dscs):.4f}")
        print(f"  Range: [{np.min(all_dscs):.4f}, {np.max(all_dscs):.4f}]")
    print(f"{'='*80}\n")
    
    return mean_dsc, all_dscs


def finetune_single_fold(
    pretrained_checkpoint,
    isles_preprocessed_dir,
    atlas_uoa_preprocessed_dir,
    fold,
    output_dir,
    epochs=50,
    batch_size=8,
    decoder_lr=0.0001,
    encoder_lr_ratio=0.1,
    freeze_encoder_epochs=3,
    isles_only=False
):
    """
    Fine-tune on a single fold with FIXED training strategy
    
    Key improvements:
    - Discriminative learning rates (encoder 10× slower)
    - Shorter freeze period (3 epochs)
    - Better scheduler
    """
    
    set_seed(42 + fold)
    device = torch.device('cuda:0')
    
    # Create ISLES splits
    isles_splits_file = 'isles_splits_5fold_resampled.json'
    if not Path(isles_splits_file).exists():
        create_isles_splits(isles_preprocessed_dir, isles_splits_file)
    
    # Load pre-trained model
    print("\n" + "="*80)
    print(f"LOADING PRE-TRAINED MODEL (FOLD {fold})")
    print("="*80)
    model, attention_type, deep_supervision = load_pretrained_model(
        pretrained_checkpoint, device
    )
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f'fold_{fold}' / f'finetune_FIXED_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Save config
    config = {
        'pretrained_checkpoint': pretrained_checkpoint,
        'fold': fold,
        'epochs': epochs,
        'batch_size': batch_size,
        'decoder_lr': decoder_lr,
        'encoder_lr_ratio': encoder_lr_ratio,
        'encoder_lr': decoder_lr * encoder_lr_ratio,
        'freeze_encoder_epochs': freeze_encoder_epochs,
        'isles_only': isles_only,
        'attention_type': attention_type,
        'deep_supervision': deep_supervision
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    isles_train = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='train',
        splits_file=isles_splits_file,
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True,
        lesion_focus_ratio=0.7,
        compute_lesion_bins=False
    )
    
    isles_val = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='val',
        splits_file=isles_splits_file,
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )
    
    print(f"\n✓ ISLES train volumes: {len(isles_train.volumes)}")
    print(f"✓ ISLES val volumes: {len(isles_val.volumes)}")
    
    if len(isles_val.volumes) == 0:
        print(f"\n❌ ERROR: No ISLES validation volumes!")
        return None
    
    if isles_only:
        train_dataset = isles_train
        print(f"\n✓ Training on ISLES ONLY")
    else:
        atlas_uoa_train = PatchDatasetWithCenters(
            preprocessed_dir=atlas_uoa_preprocessed_dir,
            datasets=['ATLAS', 'UOA_Private'],
            split='train',
            splits_file='splits_5fold.json',
            fold=fold,
            patch_size=(96, 96, 96),
            patches_per_volume=10,
            augment=True,
            lesion_focus_ratio=0.7
        )
        
        train_dataset = ConcatDataset([atlas_uoa_train, isles_train])
        print(f"\n✓ Training on MIXED dataset")
        print(f"  ATLAS/UOA: {len(atlas_uoa_train)}")
        print(f"  ISLES: {len(isles_train)}")
        print(f"  Total: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    
    # ============================================================
    # DISCRIMINATIVE LEARNING RATE SETUP
    # ============================================================
    
    print("\n" + "="*80)
    print("OPTIMIZER SETUP (DISCRIMINATIVE LEARNING RATES)")
    print("="*80)
    
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    encoder_lr = decoder_lr * encoder_lr_ratio
    
    print(f"✓ Encoder parameters: {len(encoder_params)}")
    print(f"  Learning rate: {encoder_lr:.6f} ({encoder_lr_ratio}× decoder)")
    print(f"✓ Decoder parameters: {len(decoder_params)}")
    print(f"  Learning rate: {decoder_lr:.6f}")
    print(f"✓ LR ratio: {encoder_lr_ratio:.2f} (encoder gets {encoder_lr_ratio*100:.0f}% of decoder LR)")
    print("="*80 + "\n")
    
    # Setup training
    criterion = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
        {'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'}
    ], weight_decay=0.01)
    
    scheduler = DiscriminativeLRScheduler(
        optimizer=optimizer,
        warmup_epochs=2,
        total_epochs=epochs,
        encoder_base_lr=encoder_lr,
        decoder_base_lr=decoder_lr,
        min_lr=1e-7
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_dsc = 0.0
    best_epoch = 0
    
    log_file = exp_dir / 'finetune_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_recon,encoder_lr,decoder_lr,encoder_frozen\n")
    
    print("\n" + "="*80)
    print(f"STARTING FINE-TUNING (FOLD {fold})")
    print("="*80)
    print(f"Strategy:")
    print(f"  1. Freeze encoder for {freeze_encoder_epochs} epochs (decoder initialization)")
    print(f"  2. Unfreeze encoder with {encoder_lr_ratio}× slower LR")
    print(f"  3. Train both encoder + decoder together")
    print("="*80 + "\n")
    
    for epoch in range(epochs):
        
        # Freeze/unfreeze encoder
        encoder_frozen = epoch < freeze_encoder_epochs
        
        if encoder_frozen:
            for param in model.encoder.parameters():
                param.requires_grad = False
            if epoch == 0:
                print(f"\n🔒 PHASE 1: Encoder FROZEN (epochs 1-{freeze_encoder_epochs})")
                print(f"   Goal: Initialize decoder using pretrained features")
                print(f"   Decoder LR: {decoder_lr:.6f}\n")
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
            if epoch == freeze_encoder_epochs:
                print(f"\n🔓 PHASE 2: Encoder UNFROZEN (epochs {freeze_encoder_epochs+1}-{epochs})")
                print(f"   Goal: Adapt encoder to DWI while preserving useful features")
                print(f"   Encoder LR: {encoder_lr:.6f} (10× slower)")
                print(f"   Decoder LR: {decoder_lr:.6f}\n")
        
        # Update learning rates
        current_encoder_lr, current_decoder_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler,
            deep_supervision=deep_supervision,
            ds_weights=[1.0, 0.0, 0.0, 0.0] if deep_supervision else None,
            max_grad_norm=1.0
        )
        
        # Validate
        val_dsc_recon, all_dscs = validate_isles_reconstruction(
            model, isles_val, device, (96, 96, 96),
            deep_supervision=deep_supervision
        )
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},"
                   f"{val_dsc_recon:.6f},{current_encoder_lr:.6f},"
                   f"{current_decoder_lr:.6f},{encoder_frozen}\n")
        
        # Print summary
        print(f"{'='*80}")
        print(f"Fold {fold} - Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  ISLES Val DSC: {val_dsc_recon:.4f}")
        print(f"  Encoder LR: {current_encoder_lr:.6f} {'(FROZEN)' if encoder_frozen else ''}")
        print(f"  Decoder LR: {current_decoder_lr:.6f}")
        
        # Save best model
        if val_dsc_recon > best_dsc:
            best_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon,
                'attention_type': attention_type,
                'deep_supervision': deep_supervision,
                'config': config,
                'finetuned_on': 'ISLES_DWI',
                'pretrained_from': pretrained_checkpoint
            }, exp_dir / 'checkpoints' / 'best_finetuned_model.pth')
            
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*80}\n")
        
        # Clear GPU cache periodically
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "="*80)
    print(f"FOLD {fold} FINE-TUNING COMPLETE")
    print("="*80)
    print(f"  Best Val DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"  Checkpoint: {exp_dir / 'checkpoints' / 'best_finetuned_model.pth'}")
    print("="*80 + "\n")
    
    return {
        'fold': fold,
        'best_dsc': best_dsc,
        'best_epoch': best_epoch,
        'checkpoint_path': str(exp_dir / 'checkpoints' / 'best_finetuned_model.pth'),
        'config': config
    }


def finetune_all_folds(
    pretrained_checkpoint,
    isles_preprocessed_dir,
    atlas_uoa_preprocessed_dir,
    output_dir,
    epochs=50,
    batch_size=8,
    decoder_lr=0.0001,
    encoder_lr_ratio=0.1,
    freeze_encoder_epochs=3,
    isles_only=False
):
    """Fine-tune on all 5 folds with FIXED strategy"""
    
    print("\n" + "="*80)
    print("STARTING 5-FOLD FINE-TUNING (FIXED VERSION)")
    print("="*80)
    print(f"Improvements:")
    print(f"  ✓ Discriminative learning rates (encoder {encoder_lr_ratio}× decoder)")
    print(f"  ✓ Shorter freeze period ({freeze_encoder_epochs} epochs)")
    print(f"  ✓ Better scheduler with proportional warmup")
    print("="*80 + "\n")
    
    all_results = []
    
    for fold in range(5):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold}/4")
        print(f"{'#'*80}\n")
        
        fold_result = finetune_single_fold(
            pretrained_checkpoint=pretrained_checkpoint,
            isles_preprocessed_dir=isles_preprocessed_dir,
            atlas_uoa_preprocessed_dir=atlas_uoa_preprocessed_dir,
            fold=fold,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            decoder_lr=decoder_lr,
            encoder_lr_ratio=encoder_lr_ratio,
            freeze_encoder_epochs=freeze_encoder_epochs,
            isles_only=isles_only
        )
        
        if fold_result:
            all_results.append(fold_result)
    
    # Summary
    print("\n" + "="*80)
    print("5-FOLD FINE-TUNING COMPLETE!")
    print("="*80)
    
    for result in all_results:
        print(f"Fold {result['fold']}: DSC={result['best_dsc']:.4f} (Epoch {result['best_epoch']})")
    
    mean_dsc = np.mean([r['best_dsc'] for r in all_results])
    std_dsc = np.std([r['best_dsc'] for r in all_results])
    
    print(f"\n📊 Summary:")
    print(f"  Mean DSC: {mean_dsc:.4f} ± {std_dsc:.4f}")
    print("="*80 + "\n")
    
    # Save summary
    summary_file = Path(output_dir) / 'fivefold_summary_FIXED.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'results': all_results,
            'mean_dsc': float(mean_dsc),
            'std_dsc': float(std_dsc),
            'improvements': [
                'Discriminative learning rates',
                'Shorter freeze period',
                'Better scheduler'
            ]
        }, f, indent=2)
    
    print(f"✓ Summary saved: {summary_file}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIXED fine-tuning on ISLES with 5-fold CV')
    
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                       help='Path to pretrained checkpoint (SimCLR or supervised)')
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='ISLES preprocessed directory')
    parser.add_argument('--atlas-uoa-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='ATLAS+UOA preprocessed directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/finetuned_on_isles_5fold_FIXED',
                       help='Output directory for results')
    parser.add_argument('--fold', type=int, default=None, choices=[0, 1, 2, 3, 4],
                       help='Single fold to train (default: all folds)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--decoder-lr', type=float, default=0.0001,
                       help='Decoder learning rate (default: 1e-4)')
    parser.add_argument('--encoder-lr-ratio', type=float, default=0.1,
                       help='Encoder LR = decoder_lr × this ratio (default: 0.1)')
    parser.add_argument('--freeze-encoder-epochs', type=int, default=3,
                       help='Freeze encoder for N epochs (default: 3)')
    parser.add_argument('--isles-only', action='store_true',
                       help='Train on ISLES only (no ATLAS/UOA)')
    
    args = parser.parse_args()
    
    if args.fold is not None:
        # Train single fold
        finetune_single_fold(
            pretrained_checkpoint=args.pretrained_checkpoint,
            isles_preprocessed_dir=args.isles_dir,
            atlas_uoa_preprocessed_dir=args.atlas_uoa_dir,
            fold=args.fold,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            decoder_lr=args.decoder_lr,
            encoder_lr_ratio=args.encoder_lr_ratio,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            isles_only=args.isles_only
        )
    else:
        # Train all folds
        finetune_all_folds(
            pretrained_checkpoint=args.pretrained_checkpoint,
            isles_preprocessed_dir=args.isles_dir,
            atlas_uoa_preprocessed_dir=args.atlas_uoa_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            decoder_lr=args.decoder_lr,
            encoder_lr_ratio=args.encoder_lr_ratio,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            isles_only=args.isles_only
        )

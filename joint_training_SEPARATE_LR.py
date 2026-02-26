"""
joint_training_SEPARATE_LR_FINAL.py

FINAL VERSION with CORRECT Generalized Dice Loss
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
from torch import einsum

sys.path.append('/home/pahm409')
from models.resnet3d import resnet3d_18

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def center_crop_or_pad(array, target_shape, mode='constant', cval=0):
    """
    Crop or pad array to target_shape, maintaining center alignment.
    
    Args:
        array: Input 3D array
        target_shape: Desired output shape (D, H, W)
        mode: Padding mode ('constant', 'reflect', 'edge', etc.)
        cval: Constant value if mode='constant'
    
    Returns:
        Array with target_shape
    """
    current_shape = np.array(array.shape)
    target_shape = np.array(target_shape)
    delta = current_shape - target_shape
    
    # Pure crop
    if np.all(delta >= 0):
        start = delta // 2
        end = start + target_shape
        return array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # Pure pad
    if np.all(delta <= 0):
        pad_before = (-delta) // 2
        pad_after = -delta - pad_before
        pad_width = [(pb, pa) for pb, pa in zip(pad_before, pad_after)]
        
        if mode == 'constant':
            return np.pad(array, pad_width, mode=mode, constant_values=cval)
        else:
            return np.pad(array, pad_width, mode=mode)
    
    # Mixed: crop some dims, pad others
    result = array.copy()
    for i in range(3):
        if delta[i] > 0:
            # Crop dimension i
            start = delta[i] // 2
            end = start + target_shape[i]
            if i == 0:
                result = result[start:end, :, :]
            elif i == 1:
                result = result[:, start:end, :]
            else:
                result = result[:, :, start:end]
        elif delta[i] < 0:
            # Pad dimension i
            pad_before = (-delta[i]) // 2
            pad_after = -delta[i] - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[i] = (pad_before, pad_after)
            
            if mode == 'constant':
                result = np.pad(result, pad_width, mode=mode, constant_values=cval)
            else:
                result = np.pad(result, pad_width, mode=mode)
    
    return result

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def apply_augmentations_aligned(dwi, adc, mask):
    """
    PRODUCTION-GRADE augmentation with geometric consistency.
    
    Rules:
    - Spatial transforms: SAME parameters for dwi, adc, mask
    - Images: mode='reflect' to avoid black borders
    - Mask: mode='constant' with cval=0 (preserve background)
    - Intensity: independent per modality (realistic scanner variation)
    - Consistent intensity clipping (always applied)
    """
    
    # ========================================================================
    # SPATIAL AUGMENTATIONS - SAME parameters for all modalities
    # ========================================================================
    
    # 1. ROTATION
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
        
        # Images: reflect mode (no black borders)
        dwi = rotate(dwi, angle, axes=axes, reshape=False, order=1, mode='reflect')
        adc = rotate(adc, angle, axes=axes, reshape=False, order=1, mode='reflect')
        # Mask: constant mode (preserve background=0)
        mask = rotate(mask, angle, axes=axes, reshape=False, order=0, 
                     mode='constant', cval=0)
    
    # 2. SCALING
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.9, 1.1)
        original_shape = dwi.shape
        
        # Apply zoom with order parameter only (mode not reliably supported)
        dwi_scaled = zoom(dwi, scale, order=1)
        adc_scaled = zoom(adc, scale, order=1)
        mask_scaled = zoom(mask, scale, order=0)
        
        # Crop or pad back to original shape using our helper
        # Images: use reflect mode to avoid black borders
        dwi = center_crop_or_pad(dwi_scaled, original_shape, mode='reflect')
        adc = center_crop_or_pad(adc_scaled, original_shape, mode='reflect')
        # Mask: use constant mode with cval=0
        mask = center_crop_or_pad(mask_scaled, original_shape, mode='constant', cval=0)
    
    # 3. FLIPPING
    if np.random.rand() > 0.5:
        axis = np.random.randint(0, 3)
        dwi = np.flip(dwi, axis=axis).copy()
        adc = np.flip(adc, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
    
    # 4. ELASTIC DEFORMATION
    if np.random.rand() > 0.7:
        alpha = 10
        sigma = 3
        shape = dwi.shape
        
        # Generate displacement field ONCE
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create coordinate grid
        x, y, z = np.meshgrid(
            np.arange(shape[0]), 
            np.arange(shape[1]), 
            np.arange(shape[2]), 
            indexing='ij'
        )
        
        # Build indices in clean (3, D, H, W) form
        indices = np.array([x + dx, y + dy, z + dz])
        
        # Clip to valid range to avoid extreme edge warping
        indices[0] = np.clip(indices[0], 0, shape[0] - 1)
        indices[1] = np.clip(indices[1], 0, shape[1] - 1)
        indices[2] = np.clip(indices[2], 0, shape[2] - 1)
        
        # Apply SAME displacement to all
        dwi = map_coordinates(dwi, indices, order=1, mode='reflect')
        adc = map_coordinates(adc, indices, order=1, mode='reflect')
        mask = map_coordinates(mask, indices, order=0, mode='constant', cval=0)
    
    # ========================================================================
    # INTENSITY AUGMENTATIONS - Independent per modality
    # ALWAYS CLIP at the end for consistency
    # ========================================================================
    
    # Gaussian blur
    if np.random.rand() > 0.5:
        sigma = np.random.uniform(0.3, 0.7)
        dwi = gaussian_filter(dwi, sigma=sigma)
    if np.random.rand() > 0.5:
        sigma = np.random.uniform(0.3, 0.7)
        adc = gaussian_filter(adc, sigma=sigma)
    
    # Gaussian noise
    if np.random.rand() > 0.5:
        noise_std = 0.01
        dwi = dwi + np.random.normal(0, noise_std, dwi.shape)
    if np.random.rand() > 0.5:
        noise_std = 0.01
        adc = adc + np.random.normal(0, noise_std, adc.shape)
    
    # Brightness
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        dwi = dwi * factor
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        adc = adc * factor
    
    # Contrast
    if np.random.rand() > 0.5:
        mean = dwi.mean()
        factor = np.random.uniform(0.8, 1.2)
        dwi = (dwi - mean) * factor + mean
    if np.random.rand() > 0.5:
        mean = adc.mean()
        factor = np.random.uniform(0.8, 1.2)
        adc = (adc - mean) * factor + mean
    
    # Gamma
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.8, 1.2)
        dwi_min, dwi_max = dwi.min(), dwi.max()
        if dwi_max > dwi_min:
            dwi_norm = (dwi - dwi_min) / (dwi_max - dwi_min)
            dwi_gamma = np.power(dwi_norm, gamma)
            dwi = dwi_gamma * (dwi_max - dwi_min) + dwi_min
    
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.8, 1.2)
        adc_min, adc_max = adc.min(), adc.max()
        if adc_max > adc_min:
            adc_norm = (adc - adc_min) / (adc_max - adc_min)
            adc_gamma = np.power(adc_norm, gamma)
            adc = adc_gamma * (adc_max - adc_min) + adc_min
    
    # ALWAYS clip to percentile range for consistency
    # Prevents intensity transforms from creating unrealistic outliers
    p_low_dwi, p_high_dwi = np.percentile(dwi, [1, 99])
    dwi = np.clip(dwi, p_low_dwi, p_high_dwi)
    
    p_low_adc, p_high_adc = np.percentile(adc, [1, 99])
    adc = np.clip(adc, p_low_adc, p_high_adc)
    
    return dwi, adc, mask

# ============================================================================
# DATASET
# ============================================================================

class DualModalityDataset(Dataset):
    def __init__(self, npz_dir, case_list, patch_size=(96, 96, 96),
                 patches_per_volume=10, augment=False):
        self.npz_dir = Path(npz_dir)
        self.case_list = case_list
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        
        self.volumes = []
        for case_id in case_list:
            npz_file = self.npz_dir / f"{case_id}.npz"
            if npz_file.exists():
                try:
                    data = np.load(npz_file)
                    if 'dwi' in data.files and 'adc' in data.files and 'mask' in data.files:
                        self.volumes.append({'case_id': case_id, 'path': npz_file})
                except:
                    pass
        
        print(f"  Loaded {len(self.volumes)} volumes")
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        
        data = np.load(self.volumes[vol_idx]['path'])
        dwi = data['dwi'].copy()
        adc = data['adc'].copy()
        mask = data['mask'].copy()
        
        D, H, W = dwi.shape
        pd, ph, pw = self.patch_size
        
        d_start = np.random.randint(0, max(1, D - pd + 1))
        h_start = np.random.randint(0, max(1, H - ph + 1))
        w_start = np.random.randint(0, max(1, W - pw + 1))
        
        dwi_patch = dwi[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        adc_patch = adc[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        mask_patch = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        if dwi_patch.shape != tuple(self.patch_size):
            dwi_patch = np.pad(dwi_patch, 
                              [(0, max(0, pd - dwi_patch.shape[0])),
                               (0, max(0, ph - dwi_patch.shape[1])),
                               (0, max(0, pw - dwi_patch.shape[2]))],
                              mode='constant')
            adc_patch = np.pad(adc_patch,
                              [(0, max(0, pd - adc_patch.shape[0])),
                               (0, max(0, ph - adc_patch.shape[1])),
                               (0, max(0, pw - adc_patch.shape[2]))],
                              mode='constant')
            mask_patch = np.pad(mask_patch,
                               [(0, max(0, pd - mask_patch.shape[0])),
                                (0, max(0, ph - mask_patch.shape[1])),
                                (0, max(0, pw - mask_patch.shape[2]))],
                               mode='constant')
        
        if self.augment:
            dwi_patch, adc_patch, mask_patch = apply_augmentations_aligned(
                dwi_patch, adc_patch, mask_patch
            )
        
        return {
            'dwi': torch.from_numpy(dwi_patch).unsqueeze(0).float(),
            'adc': torch.from_numpy(adc_patch).unsqueeze(0).float(),
            'mask': torch.from_numpy(mask_patch).long()
        }

# ============================================================================
# LOSS - CORRECT GENERALIZED DICE LOSS
# ============================================================================

class GDiceLoss(nn.Module):
    """
    Generalized Dice Loss (GDice)
    
    Weights classes inversely proportional to their squared frequency.
    Better for imbalanced segmentation than standard Soft Dice.
    
    Works for both 2D (B,C,H,W) and 3D (B,C,D,H,W).
    """
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
    
    def forward(self, net_output, gt):
        # Handle 2D input by adding dummy depth dimension
        if net_output.ndim == 4:
            net_output = net_output.unsqueeze(2)  # (B,C,1,H,W)
            gt = gt.unsqueeze(2)
        
        shp_x = net_output.shape  # (B,C,D,H,W)
        shp_y = gt.shape
        
        # Convert ground truth to one-hot encoding
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))
            
            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # Already one-hot
                y_onehot = gt
            else:
                # Convert to one-hot
                gt = gt.long()
                y_onehot = torch.zeros(shp_x, device=net_output.device, dtype=net_output.dtype)
                y_onehot.scatter_(1, gt, 1)
        
        # Apply nonlinearity (e.g., softmax) if specified
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        
        # Convert to double precision for numerical stability
        net_output = net_output.double()
        y_onehot = y_onehot.double()
        
        # Compute class weights: w_c = 1 / (sum of pixels in class c)^2
        # Shape: (B, C)
        w = 1 / (einsum("bcdhw->bc", y_onehot).type(torch.float64) + 1e-10) ** 2
        
        # Compute weighted intersection and union per class
        # Shape: (B, C)
        intersection = w * einsum("bcdhw,bcdhw->bc", net_output, y_onehot)
        union = w * (einsum("bcdhw->bc", net_output) + einsum("bcdhw->bc", y_onehot))
        
        # Compute GDice per sample
        # Shape: (B,)
        divided = -2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        
        # Average over batch
        gdc = divided.mean()
        
        return gdc.float()

# ============================================================================
# MODEL
# ============================================================================

class ResNet3DEncoder(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
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
    def __init__(self, num_classes=2):
        super().__init__()
        
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
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = torch.nn.functional.interpolate(
                x_up, size=x_skip.shape[2:], 
                mode='trilinear', align_corners=False
            )
        return x_up
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        x = self.up4(x5)
        x = self._match_size(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = self._match_size(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = self._match_size(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = self._match_size(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        return self.final_conv(x)

class JointTrainingModels(nn.Module):
    def __init__(self, pretrained_encoder_path=None, num_classes=2):
        super().__init__()
        
        # DWI model with pretraining
        base_encoder_dwi = resnet3d_18(in_channels=1)
        
        if pretrained_encoder_path:
            print(f"Loading checkpoint: {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            
            # Extract ONLY encoder weights
            encoder_state = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    encoder_state[new_key] = value
            
            # Load into base encoder
            missing, unexpected = base_encoder_dwi.load_state_dict(encoder_state, strict=False)
            print(f"✓ Loaded encoder weights")
            print(f"  Missing keys: {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")
        
        self.encoder_dwi = ResNet3DEncoder(base_encoder_dwi)
        self.decoder_dwi = UNetDecoder3D(num_classes=num_classes)
        
        # ADC model from scratch
        base_encoder_adc = resnet3d_18(in_channels=1)
        self.encoder_adc = ResNet3DEncoder(base_encoder_adc)
        self.decoder_adc = UNetDecoder3D(num_classes=num_classes)
        
        # Initialize ADC model properly
        def init_weights(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder_adc.apply(init_weights)
        self.decoder_adc.apply(init_weights)
    
    def forward(self, dwi, adc):
        input_size = dwi.shape[2:]
        
        enc_features_dwi = self.encoder_dwi(dwi)
        output_dwi = self.decoder_dwi(enc_features_dwi)
        
        enc_features_adc = self.encoder_adc(adc)
        output_adc = self.decoder_adc(enc_features_adc)
        
        if output_dwi.shape[2:] != input_size:
            output_dwi = torch.nn.functional.interpolate(
                output_dwi, size=input_size,
                mode='trilinear', align_corners=False
            )
        if output_adc.shape[2:] != input_size:
            output_adc = torch.nn.functional.interpolate(
                output_adc, size=input_size,
                mode='trilinear', align_corners=False
            )
        
        return output_dwi, output_adc
    
    def forward_dwi_only(self, dwi):
        input_size = dwi.shape[2:]
        enc_features = self.encoder_dwi(dwi)
        output = self.decoder_dwi(enc_features)
        
        if output.shape[2:] != input_size:
            output = torch.nn.functional.interpolate(
                output, size=input_size,
                mode='trilinear', align_corners=False
            )
        
        return output

# ============================================================================
# POLY LR SCHEDULER
# ============================================================================

class PolyLRScheduler:
    def __init__(self, optimizer, max_epochs, exponent=0.9, last_epoch=-1):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        factor = (1 - epoch / self.max_epochs) ** self.exponent
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * factor
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# ============================================================================
# TRAINING
# ============================================================================

def compute_dsc(pred, target):
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0
    return (2. * intersection / union).item()

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    total_dsc_dwi = 0
    total_dsc_adc = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        dwi = batch['dwi'].to(device)
        adc = batch['adc'].to(device)
        masks = batch['mask'].to(device)
        
        with autocast():
            output_dwi, output_adc = model(dwi, adc)
            
            # GDice expects logits and applies softmax internally
            loss_dwi = criterion(output_dwi, masks)
            loss_adc = criterion(output_adc, masks)
            
            loss = loss_dwi + loss_adc
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            pred_dwi = torch.softmax(output_dwi, dim=1)[:, 1]
            pred_adc = torch.softmax(output_adc, dim=1)[:, 1]
            target = (masks > 0).float()
            
            dsc_dwi = compute_dsc(pred_dwi, target)
            dsc_adc = compute_dsc(pred_adc, target)
        
        total_loss += loss.item()
        total_dsc_dwi += dsc_dwi
        total_dsc_adc += dsc_adc
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dwi': f'{dsc_dwi:.4f}',
            'adc': f'{dsc_adc:.4f}'
        })
    
    n = len(dataloader)
    return total_loss/n, total_dsc_dwi/n, total_dsc_adc/n

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_dsc = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            dwi = batch['dwi'].to(device)
            masks = batch['mask'].to(device)
            
            with autocast():
                output = model.forward_dwi_only(dwi)
                loss = criterion(output, masks)
            
            pred = torch.softmax(output, dim=1)[:, 1]
            target = (masks > 0).float()
            dsc = compute_dsc(pred, target)
            
            total_loss += loss.item()
            total_dsc += dsc
    
    n = len(dataloader)
    return total_loss/n, total_dsc/n

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brats-checkpoint', type=str, required=True)
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/preprocessed_isles_dual_v2')
    parser.add_argument('--splits-file', type=str,
                       default='isles_dual_splits_5fold.json')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/joint_training_final')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    
    # Settings
    EPOCHS = 200
    BATCH_SIZE = 8
    PATCH_SIZE = (96, 96, 96)
    DWI_LR = 0.0001
    ADC_LR = 0.001
    WEIGHT_DECAY = 3e-5
    MOMENTUM = 0.99
    POLY_EXPONENT = 0.9
    
    set_seed(42 + args.fold)
    device = torch.device('cuda:0')
    
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("JOINT TRAINING - FINAL PRODUCTION VERSION")
    print("="*80)
    print(f"Loss: Generalized Dice Loss (GDice)")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"DWI LR: {DWI_LR} (pretrained)")
    print(f"ADC LR: {ADC_LR} (random init)")
    print(f"Scheduler: PolyLR (exponent={POLY_EXPONENT})")
    print("="*80 + "\n")
    
    # Model
    model = JointTrainingModels(
        pretrained_encoder_path=args.brats_checkpoint,
        num_classes=2
    ).to(device)
    
    print()
    
    # Data
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    train_cases = splits[f'fold_{args.fold}']['ISLES2022_dual']['train']
    val_cases = splits[f'fold_{args.fold}']['ISLES2022_dual']['val']
    
    print(f"Data: Train={len(train_cases)}, Val={len(val_cases)}\n")
    
    train_dataset = DualModalityDataset(
        npz_dir=args.isles_dir,
        case_list=train_cases,
        patch_size=PATCH_SIZE,
        patches_per_volume=10,
        augment=True
    )
    
    val_dataset = DualModalityDataset(
        npz_dir=args.isles_dir,
        case_list=val_cases,
        patch_size=PATCH_SIZE,
        patches_per_volume=10,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Get parameters for each model
    dwi_params = list(model.encoder_dwi.parameters()) + list(model.decoder_dwi.parameters())
    adc_params = list(model.encoder_adc.parameters()) + list(model.decoder_adc.parameters())
    
    # Create optimizer with DIFFERENT learning rates
    optimizer = optim.Adam([
        {'params': dwi_params, 'lr': DWI_LR},
        {'params': adc_params, 'lr': ADC_LR}
    ], weight_decay=WEIGHT_DECAY, betas=(MOMENTUM, 0.999))
    
    print("✓ Optimizer created with separate learning rates:")
    print(f"  DWI: {len(dwi_params)} param groups at LR={DWI_LR}")
    print(f"  ADC: {len(adc_params)} param groups at LR={ADC_LR}\n")
    
    # CORRECT: Generalized Dice Loss with softmax

    criterion = GDiceLoss(apply_nonlin=lambda x: torch.softmax(x, dim=1), smooth=1e-5)

    
    scheduler = PolyLRScheduler(optimizer, max_epochs=EPOCHS, exponent=POLY_EXPONENT)
    scaler = GradScaler()
    
    best_dsc = 0.0
    best_epoch = 0
    
    log_file = output_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc_dwi,train_dsc_adc,val_loss,val_dsc,dwi_lr,adc_lr\n")
    
    print("STARTING TRAINING\n")
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 80)
        
        train_loss, train_dsc_dwi, train_dsc_adc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        val_loss, val_dsc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(epoch)
        current_lrs = scheduler.get_last_lr()
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc_dwi:.6f},"
                   f"{train_dsc_adc:.6f},{val_loss:.6f},{val_dsc:.6f},"
                   f"{current_lrs[0]:.6f},{current_lrs[1]:.6f}\n")
        
        print(f"Train - Loss: {train_loss:.4f}, DWI: {train_dsc_dwi:.4f}, ADC: {train_dsc_adc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, DSC: {val_dsc:.4f}")
        print(f"LR - DWI: {current_lrs[0]:.6f}, ADC: {current_lrs[1]:.6f}")
        
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state_dict': model.state_dict(),
                'val_dsc': val_dsc,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'dwi_lr': DWI_LR,
                    'adc_lr': ADC_LR,
                    'loss': 'GDiceLoss',
                    'augmentation': 'geometrically_aligned_production_v1'
                }
            }, output_dir / 'best_joint_model.pth')
            
            print(f"✓ BEST: {best_dsc:.4f}")
        
        print(f"Best: {best_dsc:.4f} (Epoch {best_epoch})\n")
        
        if (epoch + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE - Best: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

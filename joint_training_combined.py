"""
joint_training_SEPARATE_LR.py

Joint training with SEPARATE learning rates AND DUAL-CHANNEL INPUT:
- Input: Concatenated DWI+ADC (2 channels)
- Output: Separate predictions for DWI and ADC
- DWI encoder: Pretrained from BraTS, modified for 2-channel input
- ADC pathway: Random initialization
- Inference: Uses both DWI+ADC input to predict lesion
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

sys.path.append('/home/pahm409')
from models.resnet3d import resnet3d_18

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def elastic_deformation_3d(image, mask, alpha=10, sigma=3):
    shape = image.shape[1:]  # Skip channel dimension
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), 
                          np.arange(shape[2]), indexing='ij')
    
    indices = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z + dz, (-1, 1))
    )
    
    # Apply to both channels
    image_deformed = np.zeros_like(image)
    for c in range(image.shape[0]):
        image_deformed[c] = map_coordinates(image[c], indices, order=1, mode='reflect').reshape(shape)
    
    mask_deformed = map_coordinates(mask, indices, order=0, mode='constant', cval=0).reshape(shape)
    
    return image_deformed, mask_deformed

def random_rotation_3d(image, mask, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    axes = [(0, 1), (0, 2), (1, 2)]
    chosen_axes = axes[np.random.randint(0, 3)]
    
    # Adjust axes for channel dimension
    chosen_axes = tuple(a + 1 for a in chosen_axes)
    
    image_rot = rotate(image, angle, axes=chosen_axes, reshape=False, order=1, mode='constant', cval=0)
    mask_rot = rotate(mask, angle, axes=(chosen_axes[0]-1, chosen_axes[1]-1), reshape=False, order=0, mode='constant', cval=0)
    
    return image_rot, mask_rot

def random_scaling(image, mask, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    if scale != 1.0:
        # Scale spatial dimensions only
        zoom_factors = [1.0, scale, scale, scale]  # [channels, D, H, W]
        image_scaled = zoom(image, zoom_factors, order=1, mode='constant', cval=0)
        mask_scaled = zoom(mask, scale, order=0, mode='constant', cval=0)
        
        original_shape = image.shape[1:]
        current_shape = image_scaled.shape[1:]
        diff = [c - o for c, o in zip(current_shape, original_shape)]
        
        if all(d >= 0 for d in diff):
            start = [d // 2 for d in diff]
            image_scaled = image_scaled[
                :,
                start[0]:start[0]+original_shape[0],
                start[1]:start[1]+original_shape[1],
                start[2]:start[2]+original_shape[2]
            ]
            mask_scaled = mask_scaled[
                start[0]:start[0]+original_shape[0],
                start[1]:start[1]+original_shape[1],
                start[2]:start[2]+original_shape[2]
            ]
        else:
            pad_amounts = [(0, 0)] + [(-d // 2, -d - (-d // 2)) if d < 0 else (0, 0) for d in diff]
            image_scaled = np.pad(image_scaled, pad_amounts, mode='constant', constant_values=0)
            
            pad_amounts_mask = [(-d // 2, -d - (-d // 2)) if d < 0 else (0, 0) for d in diff]
            mask_scaled = np.pad(mask_scaled, pad_amounts_mask, mode='constant', constant_values=0)
        
        return image_scaled, mask_scaled
    
    return image, mask

def add_gaussian_noise(image, std=0.01):
    noise = np.random.normal(0, std, image.shape)
    return image + noise

def gaussian_blur_3d(image, sigma=0.5):
    if np.random.rand() > 0.5:
        blurred = np.zeros_like(image)
        for c in range(image.shape[0]):
            blurred[c] = gaussian_filter(image[c], sigma=sigma)
        return blurred
    return image

def adjust_brightness(image, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(factor_range[0], factor_range[1])
    return image * factor

def adjust_contrast(image, factor_range=(0.8, 1.2)):
    mean = image.mean(axis=(1, 2, 3), keepdims=True)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    return (image - mean) * factor + mean

def gamma_transform(image, gamma_range=(0.8, 1.2)):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    result = np.zeros_like(image)
    
    for c in range(image.shape[0]):
        channel = image[c]
        image_min = channel.min()
        image_max = channel.max()
        
        if image_max > image_min:
            image_norm = (channel - image_min) / (image_max - image_min)
            image_gamma = np.power(image_norm, gamma)
            result[c] = image_gamma * (image_max - image_min) + image_min
        else:
            result[c] = channel
    
    return result

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
    
    def apply_augmentations(self, dwi_adc, mask):
        """Apply augmentations to concatenated DWI+ADC"""
        if np.random.rand() > 0.5:
            dwi_adc, mask = random_rotation_3d(dwi_adc, mask, max_angle=15)
        
        if np.random.rand() > 0.7:
            dwi_adc, mask = elastic_deformation_3d(dwi_adc, mask, alpha=10, sigma=3)
        
        if np.random.rand() > 0.5:
            dwi_adc, mask = random_scaling(dwi_adc, mask, scale_range=(0.9, 1.1))
        
        if np.random.rand() > 0.5:
            axis = np.random.randint(1, 4)  # Skip channel dimension
            dwi_adc = np.flip(dwi_adc, axis=axis).copy()
            mask = np.flip(mask, axis=axis-1).copy()
        
        if np.random.rand() > 0.5:
            dwi_adc = gaussian_blur_3d(dwi_adc, sigma=0.5)
        
        if np.random.rand() > 0.5:
            dwi_adc = add_gaussian_noise(dwi_adc, std=0.01)
        
        if np.random.rand() > 0.5:
            dwi_adc = adjust_brightness(dwi_adc, factor_range=(0.8, 1.2))
        
        if np.random.rand() > 0.5:
            dwi_adc = adjust_contrast(dwi_adc, factor_range=(0.8, 1.2))
        
        if np.random.rand() > 0.5:
            dwi_adc = gamma_transform(dwi_adc, gamma_range=(0.8, 1.2))
        
        return dwi_adc, mask
    
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
        
        # Concatenate DWI and ADC as 2-channel input
        dwi_adc = np.stack([dwi_patch, adc_patch], axis=0)  # [2, D, H, W]
        
        if self.augment:
            dwi_adc, mask_patch = self.apply_augmentations(dwi_adc, mask_patch)
        
        return {
            'dwi_adc': torch.from_numpy(dwi_adc).float(),  # [2, D, H, W]
            'mask': torch.from_numpy(mask_patch).long()
        }

# ============================================================================
# LOSS - FIXED VERSION
# ============================================================================

class GDiceLossV2(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        if pred.shape[2:] != target.shape[1:]:
            target_resized = torch.nn.functional.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target
        
        if len(target_resized.shape) != len(pred.shape):
            target_resized = target_resized.unsqueeze(1)
        
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target_resized.long(), 1)
        
        # FIXED: Keep batch dimension separate [B, C, D*H*W]
        pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
        target_flat = target_onehot.reshape(target_onehot.size(0), target_onehot.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(-1)  # [B, C]
        union = pred_flat.sum(-1) + target_flat.sum(-1)   # [B, C]
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()

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
        
        # Shared encoder with 2-channel input (DWI+ADC concatenated)
        base_encoder = resnet3d_18(in_channels=2)
        
        if pretrained_encoder_path:
            print(f"Loading checkpoint: {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            
            # Extract encoder weights (originally 1-channel)
            encoder_state = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    encoder_state[new_key] = value
            
            # Adapt conv1 from 1-channel to 2-channel
            if 'conv1.weight' in encoder_state:
                old_conv1 = encoder_state['conv1.weight']  # [64, 1, 7, 7, 7]
                # Duplicate weights for second channel
                new_conv1 = old_conv1.repeat(1, 2, 1, 1, 1) / 2.0  # Average to maintain magnitude
                encoder_state['conv1.weight'] = new_conv1
                print("  ✓ Adapted conv1 from 1-channel to 2-channel")
            
            # Load adapted weights
            missing, unexpected = base_encoder.load_state_dict(encoder_state, strict=False)
            print(f"  ✓ Loaded encoder weights")
            print(f"    Missing keys: {len(missing)}")
            print(f"    Unexpected keys: {len(unexpected)}")
        
        self.encoder = ResNet3DEncoder(base_encoder)
        
        # Separate decoders for DWI and ADC predictions
        self.decoder_dwi = UNetDecoder3D(num_classes=num_classes)
        self.decoder_adc = UNetDecoder3D(num_classes=num_classes)
        
        # Initialize ADC decoder
        def init_weights(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.decoder_adc.apply(init_weights)
        print("  ✓ ADC decoder initialized with Kaiming")
    
    def forward(self, dwi_adc):
        """
        Forward pass with concatenated DWI+ADC input
        
        Args:
            dwi_adc: [B, 2, D, H, W] - concatenated DWI and ADC
        
        Returns:
            output_dwi: [B, num_classes, D, H, W]
            output_adc: [B, num_classes, D, H, W]
        """
        input_size = dwi_adc.shape[2:]
        
        # Shared encoder processes concatenated input
        enc_features = self.encoder(dwi_adc)
        
        # Separate predictions
        output_dwi = self.decoder_dwi(enc_features)
        output_adc = self.decoder_adc(enc_features)
        
        # Resize if needed
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
        dwi_adc = batch['dwi_adc'].to(device)  # [B, 2, D, H, W]
        masks = batch['mask'].to(device)
        
        with autocast():
            output_dwi, output_adc = model(dwi_adc)
            
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
            dwi_adc = batch['dwi_adc'].to(device)
            masks = batch['mask'].to(device)
            
            with autocast():
                output_dwi, output_adc = model(dwi_adc)
                
                # Average predictions from both pathways
                output = (output_dwi + output_adc) / 2.0
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
                       default='/home/pahm409/joint_training_dualchannel')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    
    # Settings
    EPOCHS = 200
    BATCH_SIZE = 8
    PATCH_SIZE = (96, 96, 96)
    ENCODER_LR = 0.0001
    DECODER_LR = 0.001
    WEIGHT_DECAY = 3e-5
    MOMENTUM = 0.99
    POLY_EXPONENT = 0.9
    
    set_seed(42 + args.fold)
    device = torch.device('cuda:0')
    
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("JOINT TRAINING - DUAL-CHANNEL INPUT (DWI+ADC)")
    print("="*80)
    print(f"Input: Concatenated DWI+ADC (2 channels)")
    print(f"Output: Separate DWI and ADC predictions")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Encoder LR: {ENCODER_LR} (pretrained from BraTS)")
    print(f"Decoder LR: {DECODER_LR} (random init)")
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
    
    # Optimizer with separate LRs for encoder and decoders
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder_dwi.parameters()) + list(model.decoder_adc.parameters())
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': ENCODER_LR},
        {'params': decoder_params, 'lr': DECODER_LR}
    ], weight_decay=WEIGHT_DECAY, betas=(MOMENTUM, 0.999))
    
    print("✓ Optimizer created with separate learning rates:")
    print(f"  Encoder: {len(encoder_params)} params at LR={ENCODER_LR}")
    print(f"  Decoders: {len(decoder_params)} params at LR={DECODER_LR}\n")
    
    criterion = GDiceLossV2()
    scheduler = PolyLRScheduler(optimizer, max_epochs=EPOCHS, exponent=POLY_EXPONENT)
    scaler = GradScaler()
    
    best_dsc = 0.0
    best_epoch = 0
    
    log_file = output_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc_dwi,train_dsc_adc,val_loss,val_dsc,encoder_lr,decoder_lr\n")
    
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
        print(f"LR - Encoder: {current_lrs[0]:.6f}, Decoders: {current_lrs[1]:.6f}")
        
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
                    'encoder_lr': ENCODER_LR,
                    'decoder_lr': DECODER_LR,
                    'input_channels': 2,
                    'architecture': 'dual-channel DWI+ADC concatenated'
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

"""
joint_training_FIXED.py

CORRECT implementation with proper pretraining:
- DWI model: Uses BraTS T2-FLAIR pretraining (hyperintense lesions)
- ADC model: Trains from SCRATCH (hypointense lesions - opposite contrast)
- Train both simultaneously with joint loss
- Inference uses ONLY DWI model

Author: Parvez
Date: February 2026
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
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def elastic_deformation_3d(image, mask, alpha=10, sigma=3):
    """Elastic deformation for 3D volumes"""
    shape = image.shape
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
    
    image_deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    mask_deformed = map_coordinates(mask, indices, order=0, mode='constant', cval=0).reshape(shape)
    
    return image_deformed, mask_deformed

def random_rotation_3d(image, mask, max_angle=15):
    """Random 3D rotation"""
    angle = np.random.uniform(-max_angle, max_angle)
    axes = [(0, 1), (0, 2), (1, 2)]
    chosen_axes = axes[np.random.randint(0, 3)]
    
    image_rot = rotate(image, angle, axes=chosen_axes, reshape=False, order=1, mode='constant', cval=0)
    mask_rot = rotate(mask, angle, axes=chosen_axes, reshape=False, order=0, mode='constant', cval=0)
    
    return image_rot, mask_rot

def random_scaling(image, mask, scale_range=(0.9, 1.1)):
    """Random scaling"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    if scale != 1.0:
        image_scaled = zoom(image, scale, order=1, mode='constant', cval=0)
        mask_scaled = zoom(mask, scale, order=0, mode='constant', cval=0)
        
        original_shape = image.shape
        current_shape = image_scaled.shape
        diff = [c - o for c, o in zip(current_shape, original_shape)]
        
        if all(d >= 0 for d in diff):
            start = [d // 2 for d in diff]
            image_scaled = image_scaled[
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
            pad_amounts = [(-d // 2, -d - (-d // 2)) if d < 0 else (0, 0) for d in diff]
            image_scaled = np.pad(image_scaled, pad_amounts, mode='constant', constant_values=0)
            mask_scaled = np.pad(mask_scaled, pad_amounts, mode='constant', constant_values=0)
        
        return image_scaled, mask_scaled
    
    return image, mask

def add_gaussian_noise(image, std=0.01):
    """Add Gaussian noise"""
    noise = np.random.normal(0, std, image.shape)
    return image + noise

def gaussian_blur_3d(image, sigma=0.5):
    """Apply Gaussian blur"""
    if np.random.rand() > 0.5:
        return gaussian_filter(image, sigma=sigma)
    return image

def adjust_brightness(image, factor_range=(0.8, 1.2)):
    """Adjust brightness"""
    factor = np.random.uniform(factor_range[0], factor_range[1])
    return image * factor

def adjust_contrast(image, factor_range=(0.8, 1.2)):
    """Adjust contrast"""
    mean = image.mean()
    factor = np.random.uniform(factor_range[0], factor_range[1])
    return (image - mean) * factor + mean

def gamma_transform(image, gamma_range=(0.8, 1.2)):
    """Random gamma transform"""
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    image_min = image.min()
    image_max = image.max()
    
    if image_max > image_min:
        image_norm = (image - image_min) / (image_max - image_min)
        image_gamma = np.power(image_norm, gamma)
        return image_gamma * (image_max - image_min) + image_min
    
    return image

# ============================================================================
# DATASET WITH AUGMENTATIONS
# ============================================================================

class DualModalityDataset(Dataset):
    """Load DWI + ADC with comprehensive augmentations"""
    def __init__(self, npz_dir, case_list, patch_size=(96, 96, 96),
                 patches_per_volume=10, augment=False):
        self.npz_dir = Path(npz_dir)
        self.case_list = case_list
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        
        self.volumes = []
        missing_files = []
        
        for case_id in case_list:
            npz_file = self.npz_dir / f"{case_id}.npz"
            if npz_file.exists():
                try:
                    data = np.load(npz_file)
                    if 'dwi' in data.files and 'adc' in data.files and 'mask' in data.files:
                        self.volumes.append({'case_id': case_id, 'path': npz_file})
                    else:
                        missing_files.append(f"{case_id} (missing keys)")
                except:
                    missing_files.append(f"{case_id} (corrupt file)")
            else:
                missing_files.append(f"{case_id} (file not found)")
        
        print(f"  Loaded {len(self.volumes)} dual-modality volumes")
        if missing_files and len(missing_files) <= 10:
            print(f"  Warning: {len(missing_files)} files missing/invalid:")
            for mf in missing_files:
                print(f"    - {mf}")
        elif missing_files:
            print(f"  Warning: {len(missing_files)} files missing/invalid")
    
    def apply_augmentations(self, dwi, adc, mask):
        """Apply comprehensive augmentations to BOTH DWI and ADC"""
        # Apply same geometric transforms to all
        if np.random.rand() > 0.5:
            dwi, mask = random_rotation_3d(dwi, mask, max_angle=15)
            adc, _ = random_rotation_3d(adc, mask, max_angle=15)
        
        if np.random.rand() > 0.7:
            dwi, mask = elastic_deformation_3d(dwi, mask, alpha=10, sigma=3)
            adc, _ = elastic_deformation_3d(adc, mask, alpha=10, sigma=3)
        
        if np.random.rand() > 0.5:
            dwi, mask = random_scaling(dwi, mask, scale_range=(0.9, 1.1))
            adc, _ = random_scaling(adc, mask, scale_range=(0.9, 1.1))
        
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            dwi = np.flip(dwi, axis=axis).copy()
            adc = np.flip(adc, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # Apply intensity transforms independently
        if np.random.rand() > 0.5:
            dwi = gaussian_blur_3d(dwi, sigma=0.5)
        if np.random.rand() > 0.5:
            adc = gaussian_blur_3d(adc, sigma=0.5)
        
        if np.random.rand() > 0.5:
            dwi = add_gaussian_noise(dwi, std=0.01)
        if np.random.rand() > 0.5:
            adc = add_gaussian_noise(adc, std=0.01)
        
        if np.random.rand() > 0.5:
            dwi = adjust_brightness(dwi, factor_range=(0.8, 1.2))
        if np.random.rand() > 0.5:
            adc = adjust_brightness(adc, factor_range=(0.8, 1.2))
        
        if np.random.rand() > 0.5:
            dwi = adjust_contrast(dwi, factor_range=(0.8, 1.2))
        if np.random.rand() > 0.5:
            adc = adjust_contrast(adc, factor_range=(0.8, 1.2))
        
        if np.random.rand() > 0.5:
            dwi = gamma_transform(dwi, gamma_range=(0.8, 1.2))
        if np.random.rand() > 0.5:
            adc = gamma_transform(adc, gamma_range=(0.8, 1.2))
        
        return dwi, adc, mask
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        
        data = np.load(self.volumes[vol_idx]['path'])
        dwi = data['dwi'].copy()
        adc = data['adc'].copy()
        mask = data['mask'].copy()
        
        # Random crop
        D, H, W = dwi.shape
        pd, ph, pw = self.patch_size
        
        d_start = np.random.randint(0, max(1, D - pd + 1))
        h_start = np.random.randint(0, max(1, H - ph + 1))
        w_start = np.random.randint(0, max(1, W - pw + 1))
        
        dwi_patch = dwi[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        adc_patch = adc[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        mask_patch = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Pad if needed
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
        
        # Apply augmentations
        if self.augment:
            dwi_patch, adc_patch, mask_patch = self.apply_augmentations(
                dwi_patch, adc_patch, mask_patch
            )
        
        return {
            'dwi': torch.from_numpy(dwi_patch).unsqueeze(0).float(),
            'adc': torch.from_numpy(adc_patch).unsqueeze(0).float(),
            'mask': torch.from_numpy(mask_patch).long()
        }

# ============================================================================
# LOSS FUNCTION
# ============================================================================

class GDiceLossV2(nn.Module):
    """Generalized Dice Loss V2"""
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
        
        pred_flat = pred.view(pred.size(1), -1)
        target_flat = target_onehot.view(target_onehot.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()

# ============================================================================
# MODEL ARCHITECTURE - FIXED
# ============================================================================

class ResNet3DEncoder(nn.Module):
    """Extract encoder from ResNet3D"""
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
    """U-Net decoder"""
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
    """
    FIXED: 
    - DWI model uses BraTS T2-FLAIR pretraining (hyperintense lesions)
    - ADC model trains from SCRATCH (hypointense lesions - opposite contrast)
    """
    def __init__(self, pretrained_encoder_path=None, num_classes=2):
        super().__init__()
        
        # ===== DWI MODEL with BraTS pretraining =====
        base_encoder_dwi = resnet3d_18(in_channels=1)
        
        if pretrained_encoder_path:
            print("Loading BraTS T2-FLAIR pretrained weights for DWI model...")
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            if 'encoder.' in list(checkpoint['model_state_dict'].keys())[0]:
                encoder_state = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if key.startswith('encoder.'):
                        encoder_state[key.replace('encoder.', '')] = value
                base_encoder_dwi.load_state_dict(encoder_state)
            else:
                base_encoder_dwi.load_state_dict(checkpoint['model_state_dict'])
            print("✓ DWI encoder loaded with BraTS weights")
        
        self.encoder_dwi = ResNet3DEncoder(base_encoder_dwi)
        self.decoder_dwi = UNetDecoder3D(num_classes=num_classes)
        
        # ===== ADC MODEL from SCRATCH (NO pretraining!) =====
        print("Initializing ADC model from SCRATCH (opposite contrast to DWI)...")
        base_encoder_adc = resnet3d_18(in_channels=1)
        # NO pretraining - ADC has hypointense strokes (opposite to FLAIR/DWI)
        
        self.encoder_adc = ResNet3DEncoder(base_encoder_adc)
        self.decoder_adc = UNetDecoder3D(num_classes=num_classes)
        print("✓ ADC encoder initialized randomly")
    
    def forward(self, dwi, adc):
        """
        Train both models simultaneously
        DWI model processes DWI input (with pretrained weights)
        ADC model processes ADC input (from scratch)
        """
        input_size = dwi.shape[2:]
        
        # DWI model forward
        enc_features_dwi = self.encoder_dwi(dwi)
        output_dwi = self.decoder_dwi(enc_features_dwi)
        
        # ADC model forward
        enc_features_adc = self.encoder_adc(adc)
        output_adc = self.decoder_adc(enc_features_adc)
        
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
    
    def forward_dwi_only(self, dwi):
        """
        Inference: use ONLY DWI model
        ADC model is completely ignored
        """
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
# TRAINING FUNCTIONS
# ============================================================================

def compute_dsc(pred, target):
    """Compute Dice Score Coefficient"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0
    return (2. * intersection / union).item()

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """
    Train one epoch with joint loss
    Equation (5): L = 0.5 * L_DWI + 0.5 * L_ADC
    """
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
            # Both models predict
            output_dwi, output_adc = model(dwi, adc)
            
            # Compute losses
            loss_dwi = criterion(output_dwi, masks)
            loss_adc = criterion(output_adc, masks)
            
            # Joint loss (Eq. 5)
            loss = 0.5 * loss_dwi + 0.5 * loss_adc
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute DSC for monitoring
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
    """
    Validate using ONLY DWI model
    ADC model is ignored during validation
    """
    model.eval()
    total_loss = 0
    total_dsc = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            dwi = batch['dwi'].to(device)
            masks = batch['mask'].to(device)
            
            with autocast():
                # Use only DWI model
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
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FIXED Joint Training: DWI pretrained, ADC from scratch')
    parser.add_argument('--brats-checkpoint', type=str, required=True,
                       help='Path to BraTS T2-FLAIR pretrained checkpoint')
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/preprocessed_isles_dual_modality')
    parser.add_argument('--splits-file', type=str,
                       default='isles_dual_splits_5fold.json')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/joint_training_fixed')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    args = parser.parse_args()
    
    set_seed(42 + args.fold)
    device = torch.device('cuda:0')
    
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FIXED JOINT TRAINING")
    print("="*80)
    print("DWI Model: BraTS T2-FLAIR pretrained (hyperintense lesions)")
    print("ADC Model: Random initialization (hypointense lesions)")
    print("Training: Both models trained jointly with shared loss")
    print("Inference: Only DWI model used")
    print("="*80 + "\n")
    
    # Create model
    model = JointTrainingModels(
        pretrained_encoder_path=args.brats_checkpoint,
        num_classes=2
    ).to(device)
    
    print()
    
    # Load splits
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    train_cases = splits[f'fold_{args.fold}']['ISLES2022_dual']['train']
    val_cases = splits[f'fold_{args.fold}']['ISLES2022_dual']['val']
    
    print(f"Data: Train={len(train_cases)}, Val={len(val_cases)}\n")
    
    # Create datasets WITH AUGMENTATIONS
    print("Creating datasets with augmentations...")
    train_dataset = DualModalityDataset(
        npz_dir=args.isles_dir,
        case_list=train_cases,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True
    )
    
    val_dataset = DualModalityDataset(
        npz_dir=args.isles_dir,
        case_list=val_cases,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=False
    )
    
    if len(train_dataset.volumes) == 0:
        print("\n❌ ERROR: No training data found!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup training
    criterion = GDiceLossV2()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    best_dsc = 0.0
    best_epoch = 0
    
    log_file = output_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc_dwi,train_dsc_adc,val_loss,val_dsc,lr\n")
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        train_loss, train_dsc_dwi, train_dsc_adc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        val_loss, val_dsc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc_dwi:.6f},"
                   f"{train_dsc_adc:.6f},{val_loss:.6f},{val_dsc:.6f},{current_lr:.6f}\n")
        
        print(f"Train - Loss: {train_loss:.4f}, DWI: {train_dsc_dwi:.4f}, ADC: {train_dsc_adc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, DSC: {val_dsc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state_dict': model.state_dict(),
                'val_dsc': val_dsc,
                'method': 'joint_training_dwi_pretrained_adc_scratch'
            }, output_dir / 'best_joint_model.pth')
            
            print(f"✓ NEW BEST! Val DSC: {best_dsc:.4f}")
        
        print(f"Best: {best_dsc:.4f} (Epoch {best_epoch})")
        print("="*80 + "\n")
        
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - Fold {args.fold}")
    print(f"Best Val DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

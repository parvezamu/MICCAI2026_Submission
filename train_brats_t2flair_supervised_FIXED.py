"""
train_brats_t2flair_supervised_FIXED.py

UPDATED - handles shape mismatch
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

sys.path.append('/home/pahm409')
from models.resnet3d import resnet3d_18

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class SimpleBraTSDataset(Dataset):
    """Simple dataset that loads .npz files directly"""
    def __init__(self, npz_dir, case_list, patch_size=(96, 96, 96), 
                 patches_per_volume=10, augment=False):
        self.npz_dir = Path(npz_dir)
        self.case_list = case_list
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        
        # Load all volumes
        self.volumes = []
        for case_id in case_list:
            npz_file = self.npz_dir / f"{case_id}.npz"
            if npz_file.exists():
                self.volumes.append({'case_id': case_id, 'path': npz_file})
        
        print(f"Loaded {len(self.volumes)} volumes")
        
        if len(self.volumes) == 0:
            raise ValueError(f"No volumes found in {npz_dir}")
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        
        # Load volume
        data = np.load(self.volumes[vol_idx]['path'])
        image = data['image']  # (D, H, W)
        mask = data['mask']    # (D, H, W)
        
        # Random crop
        D, H, W = image.shape
        pd, ph, pw = self.patch_size
        
        # Random position
        d_start = np.random.randint(0, max(1, D - pd + 1))
        h_start = np.random.randint(0, max(1, H - ph + 1))
        w_start = np.random.randint(0, max(1, W - pw + 1))
        
        # Extract patch
        image_patch = image[d_start:d_start+pd, 
                           h_start:h_start+ph, 
                           w_start:w_start+pw]
        mask_patch = mask[d_start:d_start+pd,
                         h_start:h_start+ph,
                         w_start:w_start+pw]
        
        # Pad if needed
        if image_patch.shape != tuple(self.patch_size):
            image_patch = np.pad(image_patch, 
                                [(0, max(0, pd - image_patch.shape[0])),
                                 (0, max(0, ph - image_patch.shape[1])),
                                 (0, max(0, pw - image_patch.shape[2]))],
                                mode='constant')
            mask_patch = np.pad(mask_patch,
                               [(0, max(0, pd - mask_patch.shape[0])),
                                (0, max(0, ph - mask_patch.shape[1])),
                                (0, max(0, pw - mask_patch.shape[2]))],
                               mode='constant')
        
        # Simple augmentation
        if self.augment:
            # Random flip
            if np.random.rand() > 0.5:
                image_patch = np.flip(image_patch, axis=0).copy()
                mask_patch = np.flip(mask_patch, axis=0).copy()
            if np.random.rand() > 0.5:
                image_patch = np.flip(image_patch, axis=1).copy()
                mask_patch = np.flip(mask_patch, axis=1).copy()
            if np.random.rand() > 0.5:
                image_patch = np.flip(image_patch, axis=2).copy()
                mask_patch = np.flip(mask_patch, axis=2).copy()
        
        # To tensor
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_patch).long()
        
        return {'image': image_tensor, 'mask': mask_tensor}

class GDiceLossV2(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Apply softmax to predictions
        pred = torch.softmax(pred, dim=1)
        
        # Resize target to match prediction if needed
        if pred.shape[2:] != target.shape[1:]:
            target_resized = torch.nn.functional.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target
        
        # Convert to one-hot
        if len(target_resized.shape) != len(pred.shape):
            target_resized = target_resized.unsqueeze(1)
        
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target_resized.long(), 1)
        
        # Compute Dice
        pred_flat = pred.view(pred.size(1), -1)
        target_flat = target_onehot.view(target_onehot.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()

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

class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        output = self.decoder(enc_features)
        
        # Resize output to match input
        if output.shape[2:] != input_size:
            output = torch.nn.functional.interpolate(
                output, size=input_size,
                mode='trilinear', align_corners=False
            )
        
        return output

def compute_dsc(pred, target):
    """Compute DSC - handles shape mismatch"""
    # Resize if needed
    if pred.shape != target.shape:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(),
            size=pred.shape[2:],
            mode='nearest'
        ).squeeze(1)
    
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
    total_dsc = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            pred_probs = torch.softmax(outputs, dim=1)[:, 1]
            dsc = compute_dsc(pred_probs, masks)
        
        total_loss += loss.item()
        total_dsc += dsc
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / len(dataloader), total_dsc / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_dsc = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            pred_probs = torch.softmax(outputs, dim=1)[:, 1]
            dsc = compute_dsc(pred_probs, masks)
            
            total_loss += loss.item()
            total_dsc += dsc
    
    return total_loss / len(dataloader), total_dsc / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brats-dir', type=str,
                       default='/home/pahm409/preprocessed_brats2024_t2flair')
    parser.add_argument('--splits-file', type=str,
                       default='brats2024_t2flair_splits_5fold.json')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/brats_t2flair_supervised')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda:0')
    
    # Create output dir
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRAINING ON BraTS 2024 T2-FLAIR (SUPERVISED)")
    print("="*80)
    print(f"Data directory: {args.brats_dir}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80 + "\n")
    
    # Load splits
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    train_cases = splits[f'fold_{args.fold}']['BraTS2024_T2FLAIR']['train']
    val_cases = splits[f'fold_{args.fold}']['BraTS2024_T2FLAIR']['val']
    
    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}\n")
    
    # Create datasets
    train_dataset = SimpleBraTSDataset(
        npz_dir=args.brats_dir,
        case_list=train_cases,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True
    )
    
    val_dataset = SimpleBraTSDataset(
        npz_dir=args.brats_dir,
        case_list=val_cases,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(encoder, num_classes=2).to(device)
    
    criterion = GDiceLossV2()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Training loop
    best_dsc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        train_loss, train_dsc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, scaler)
        val_loss, val_dsc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train DSC: {train_dsc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val DSC: {val_dsc:.4f}")
        
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc': val_dsc,
                'fold': args.fold,
                'pretraining': 'BraTS2024_T2FLAIR_supervised'
            }, output_dir / 'best_model.pth')
            print(f"✓ NEW BEST! Val DSC: {best_dsc:.4f}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - Fold {args.fold}")
    print(f"Best Val DSC: {best_dsc:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pth'}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

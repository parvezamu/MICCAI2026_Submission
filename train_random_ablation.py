"""
Ablation Study - Part 2: WITHOUT Pre-training (Random Init)

Train segmentation model with RANDOM initialization
GPU: 9

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

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
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.append('.')
from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters
from torch import einsum


class GDiceLoss(nn.Module):
    """Generalized Dice Loss"""
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLoss, self).__init__()
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
                y_onehot = torch.zeros(shp_x, device=net_output.device, dtype=net_output.dtype)
                y_onehot.scatter_(1, gt, 1)
        
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        
        net_output = net_output.double()
        y_onehot = y_onehot.double()
        
        w = 1 / (einsum("bcdhw->bc", y_onehot).type(torch.float64) + 1e-10) ** 2
        intersection = w * einsum("bcdhw,bcdhw->bc", net_output, y_onehot)
        union = w * (einsum("bcdhw->bc", net_output) + einsum("bcdhw->bc", y_onehot))
        
        divided = -2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()
        
        return gdc.float()


def compute_dsc(pred, target, smooth=1e-6):
    """Compute DSC"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


def reconstruct_from_patches(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """
    Reconstruct full volume from patches with averaging
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2
    
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        # Bounds check
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            continue
        
        # Extract patch prediction (B, C, D, H, W) -> (D, H, W)
        patch = patch_preds[i, 1, ...]  # Get class 1 (lesion)
        
        # Accumulate
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    # Average
    reconstructed = reconstructed / (count_map + 1e-6)
    return reconstructed


class ResNet3DEncoder(nn.Module):
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
    def __init__(self, num_classes=2):
        super(UNetDecoder3D, self).__init__()
        
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
                x_up = x_up[:, :, d_start:d_start + d_skip, h_start:h_start + h_skip, w_start:w_start + w_skip]
        
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
        
        x = self.final_conv(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='trilinear', align_corners=False)
        
        return seg_logits


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, total_epochs):
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'[RANDOM] Epoch {epoch+1}/{total_epochs}')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
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
            
            loss = criterion(torch.softmax(outputs, dim=1), masks_resized)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks_resized.unsqueeze(1) == 1).float()
            dsc = compute_dsc(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / num_batches, total_dsc / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Quick patch-based validation"""
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='[RANDOM] Val (patches)'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
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
                
                loss = criterion(torch.softmax(outputs, dim=1), masks_resized)
            
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks_resized.unsqueeze(1) == 1).float()
            dsc = compute_dsc(pred_probs, target_onehot)
            
            total_loss += loss.item()
            total_dsc += dsc
            num_batches += 1
    
    return total_loss / num_batches, total_dsc / num_batches


def validate_with_reconstruction(model, dataset, device, patch_size=(96, 96, 96)):
    """
    Validate by reconstructing full volumes
    Fast version - no NIfTI saving
    """
    from collections import defaultdict
    import nibabel as nib
    
    model.eval()
    
    # Group patches by volume
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("  Collecting patches...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='[RANDOM] Processing'):
            batch = dataset[idx]
            
            vol_idx = batch['vol_idx']
            if isinstance(vol_idx, torch.Tensor):
                vol_idx = vol_idx.item()
            
            image = batch['image'].unsqueeze(0).to(device)
            center = batch['center'].numpy()
            
            # Predict
            with autocast():
                output = model(image)
            
            pred = torch.softmax(output, dim=1).cpu().numpy()
            
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
    
    # Reconstruct each volume
    print("  Reconstructing volumes...")
    all_dscs = []
    
    for vol_idx in sorted(volume_data.keys()):
        vol_info = dataset.get_volume_info(vol_idx)
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        # Reconstruct
        reconstructed = reconstruct_from_patches(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        # Binarize
        reconstructed_binary = (reconstructed > 0.5).astype(np.uint8)
        
        # Compute DSC
        mask_gt = vol_info['mask'].astype(np.float32)
        intersection = (reconstructed_binary * mask_gt).sum()
        union = reconstructed_binary.sum() + mask_gt.sum()
        
        if union == 0:
            dsc = 1.0 if reconstructed_binary.sum() == 0 else 0.0
        else:
            dsc = (2. * intersection) / union
        
        all_dscs.append(dsc)
    
    mean_dsc = np.mean(all_dscs)
    
    return mean_dsc, all_dscs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print("ABLATION STUDY - PART 2: WITHOUT PRE-TRAINING (RANDOM INIT)")
    print("="*70)
    print(f"GPU: 9 (CUDA:0)")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create RANDOM encoder (no pre-training)
    print("Creating encoder with RANDOM initialization...")
    encoder = resnet3d_18(in_channels=1)
    print(f"✓ Encoder created (Kaiming/He initialization)\n")
    
    # Create model
    model = SegmentationModel(encoder, num_classes=2).to(device)
    print(f"✓ Model created with random weights\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=args.fold,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True,
        lesion_focus_ratio=0.7
    )
    
    val_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='val',
        splits_file='splits_5fold.json',
        fold=args.fold,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=False,
        lesion_focus_ratio=0.7
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")
    
    # Setup training
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Logging
    train_losses = []
    train_dscs = []
    val_dscs_patch = []
    val_dscs_recon = []
    learning_rates = []
    
    log_file = output_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_patch,val_dsc_recon,lr\n")
    
    best_val_dsc = 0
    best_epoch = 0
    
    print(f"Starting training...\n")
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_dsc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, scaler, epoch, args.epochs)
        
        # Patch validation (quick)
        val_loss, val_dsc_patch = validate_epoch(model, val_loader, criterion, device)
        
        # Full reconstruction validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n  Running full volume reconstruction validation...")
            val_dsc_recon, all_dscs = validate_with_reconstruction(
                model, val_dataset, device, patch_size=(96, 96, 96)
            )
            print(f"  Reconstructed DSC: {val_dsc_recon:.4f} (min: {np.min(all_dscs):.4f}, max: {np.max(all_dscs):.4f})")
        else:
            val_dsc_recon = val_dscs_recon[-1] if val_dscs_recon else 0.0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        train_dscs.append(train_dsc)
        val_dscs_patch.append(val_dsc_patch)
        val_dscs_recon.append(val_dsc_recon)
        learning_rates.append(current_lr)
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc_patch:.6f},{val_dsc_recon:.6f},{current_lr:.6f}\n")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"[RANDOM] Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"  Train: Loss={train_loss:.4f}, DSC={train_dsc:.4f}")
        print(f"  Val (patch):   DSC={val_dsc_patch:.4f}")
        print(f"  Val (recon):   DSC={val_dsc_recon:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best (based on reconstruction DSC)
        if val_dsc_recon > best_val_dsc:
            best_val_dsc = val_dsc_recon
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_patch': val_dsc_patch,
                'val_dsc_recon': val_dsc_recon,
                'train_dsc': train_dsc
            }, output_dir / 'best_model.pth')
            print(f"  ✓ NEW BEST! Recon DSC={best_val_dsc:.4f}")
        
        print(f"  Best so far: {best_val_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
    
    # Save results
    results = {
        'experiment': 'random',
        'fold': args.fold,
        'epochs': args.epochs,
        'best_val_dsc_recon': float(best_val_dsc),
        'best_epoch': best_epoch,
        'final_train_dsc': float(train_dscs[-1]),
        'final_val_dsc_patch': float(val_dscs_patch[-1]),
        'final_val_dsc_recon': float(val_dscs_recon[-1]),
        'train_losses': [float(x) for x in train_losses],
        'train_dscs': [float(x) for x in train_dscs],
        'val_dscs_patch': [float(x) for x in val_dscs_patch],
        'val_dscs_recon': [float(x) for x in val_dscs_recon],
        'learning_rates': [float(x) for x in learning_rates]
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(train_dscs) + 1)
    
    # Plot 1: DSC comparison
    axes[0, 0].plot(epochs, train_dscs, 'b-', linewidth=2, label='Train')
    axes[0, 0].plot(epochs, val_dscs_patch, 'r--', linewidth=2, label='Val (patch)')
    axes[0, 0].plot(epochs, val_dscs_recon, 'g-', linewidth=2, label='Val (recon)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('DSC')
    axes[0, 0].set_title('DSC - Random Init', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_losses, 'b-', linewidth=2, label='Train')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss - Random Init', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    gap_patch = [t - v for t, v in zip(train_dscs, val_dscs_patch)]
    gap_recon = [t - v for t, v in zip(train_dscs, val_dscs_recon)]
    axes[1, 0].plot(epochs, gap_patch, 'r--', linewidth=2, label='Patch gap')
    axes[1, 0].plot(epochs, gap_recon, 'g-', linewidth=2, label='Recon gap')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val DSC')
    axes[1, 0].set_title('Generalization Gap', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, learning_rates, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("RANDOM INITIALIZATION TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Val DSC (recon): {best_val_dsc:.4f} (Epoch {best_epoch})")
    print(f"Initial DSC (Epoch 1): {val_dscs_recon[0]:.4f}")
    print(f"Final DSC (recon): {val_dscs_recon[-1]:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

"""
train_patch_with_reconstruction.py - WITH NIFTI SAVING

Added: Save reconstructed predictions and ground truth as NIfTI files

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
from torch import einsum
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
import nibabel as nib

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train on patches"""
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    scaler = GradScaler()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    
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


def validate_with_reconstruction(model, dataset, device, patch_size, save_nifti=False, save_dir=None, epoch=None):
    """
    Validate by reconstructing full volumes
    
    Args:
        save_nifti: If True, save reconstructed volumes as NIfTI files
        save_dir: Directory to save NIfTI files
        epoch: Current epoch number (for naming)
    """
    model.eval()
    
    # Group patches by volume
    volume_data = defaultdict(lambda: {'patches': [], 'centers': [], 'preds': []})
    
    print("\nCollecting patches...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Processing patches'):
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
            
            volume_data[vol_idx]['patches'].append(image.cpu())
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
    
    # Reconstruct each volume
    print("\nReconstructing volumes...")
    all_dscs = []
    
    # Create NIfTI save directory if needed
    if save_nifti and save_dir:
        nifti_dir = Path(save_dir) / f'reconstructions_epoch_{epoch}'
        nifti_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving NIfTI files to: {nifti_dir}")
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']
        
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
        
        # Save NIfTI files
        if save_nifti and save_dir:
            case_dir = nifti_dir / case_id
            case_dir.mkdir(exist_ok=True)
            
            # Create identity affine (you can load actual affine from original data if available)
            affine = np.eye(4)
            
            # Save prediction (binary)
            pred_nii = nib.Nifti1Image(reconstructed_binary, affine)
            nib.save(pred_nii, case_dir / 'prediction.nii.gz')
            
            # Save prediction (probability)
            pred_prob_nii = nib.Nifti1Image(reconstructed.astype(np.float32), affine)
            nib.save(pred_prob_nii, case_dir / 'prediction_prob.nii.gz')
            
            # Save ground truth
            gt_nii = nib.Nifti1Image(vol_info['mask'].astype(np.uint8), affine)
            nib.save(gt_nii, case_dir / 'ground_truth.nii.gz')
            
            # Save original volume (for visualization)
            vol_nii = nib.Nifti1Image(vol_info['volume'].astype(np.float32), affine)
            nib.save(vol_nii, case_dir / 'volume.nii.gz')
            
            # Save metadata
            metadata = {
                'case_id': case_id,
                'dsc': float(dsc),
                'epoch': epoch,
                'volume_shape': vol_info['mask'].shape,
                'num_patches': len(centers),
                'lesion_volume_voxels': int(vol_info['mask'].sum()),
                'pred_volume_voxels': int(reconstructed_binary.sum())
            }
            
            with open(case_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
    
    mean_dsc = np.mean(all_dscs)
    
    # Save summary
    if save_nifti and save_dir:
        summary = {
            'epoch': epoch,
            'mean_dsc': float(mean_dsc),
            'std_dsc': float(np.std(all_dscs)),
            'min_dsc': float(np.min(all_dscs)),
            'max_dsc': float(np.max(all_dscs)),
            'median_dsc': float(np.median(all_dscs)),
            'num_volumes': len(all_dscs),
            'all_dscs': [float(d) for d in all_dscs]
        }
        
        with open(nifti_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n✓ Saved {len(all_dscs)} reconstructed volumes")
        print(f"  Location: {nifti_dir}")
    
    return mean_dsc, all_dscs


def plot_training_curves(train_losses, train_dscs, val_dscs_patch, val_dscs_recon, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GDice Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DSC Comparison
    ax = axes[0, 1]
    ax.plot(epochs, train_dscs, 'b-', linewidth=2, label='Train (patches)')
    ax.plot(epochs, val_dscs_patch, 'r--', linewidth=2, label='Val (patches)')
    ax.plot(epochs, val_dscs_recon, 'g-', linewidth=3, label='Val (RECONSTRUCTED)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('DSC', fontsize=12)
    ax.set_title('DSC: Patch vs Reconstructed', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gap Analysis
    ax = axes[1, 0]
    gap_patch = [train_dscs[i] - val_dscs_patch[i] for i in range(len(epochs))]
    gap_recon = [train_dscs[i] - val_dscs_recon[i] for i in range(len(epochs))]
    ax.plot(epochs, gap_patch, 'r--', linewidth=2, label='Patch Gap')
    ax.plot(epochs, gap_recon, 'g-', linewidth=2, label='Recon Gap')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train - Val DSC', fontsize=12)
    ax.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Best Reconstructed
    ax = axes[1, 1]
    best_recon = [max(val_dscs_recon[:i+1]) for i in range(len(epochs))]
    ax.plot(epochs, best_recon, 'g-', linewidth=3, marker='*', markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Best Reconstructed DSC', fontsize=12)
    ax.set_title('Best Full-Volume DSC', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/patch_reconstruction_experiments_5fold')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Which fold to train (0-4)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10)
    parser.add_argument('--lesion-focus-ratio', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--validate-recon-every', type=int, default=5)
    parser.add_argument('--save-nifti-every', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print(f"5-FOLD CROSS-VALIDATION - FOLD {args.fold}")
    print("="*70)
    print("Training: Sparse random patches")
    print("Validation: Full reconstruction every 5 epochs")
    print("NIfTI saving: Every 10 epochs")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f'fold_{args.fold}' / f'patch_recon_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"Fold: {args.fold}")
    print(f"Experiment directory: {exp_dir}\n")
    
    # Load pre-trained encoder
    print("Loading pre-trained encoder...")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded from epoch {checkpoint['epoch'] + 1}\n")
    
    # Create model
    model = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}\n")
    
    # Setup data - FOLD-SPECIFIC
    print(f"Setting up datasets for fold {args.fold}...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=args.fold,  # ADD THIS
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
        fold=args.fold,  # ADD THIS
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
    
    print(f"\n✓ Data loaded for fold {args.fold}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Train volumes: {len(train_dataset.volumes)}")
    print(f"  Val patches: {len(val_dataset)}")
    print(f"  Val volumes: {len(val_dataset.volumes)}")
    
    # Setup training
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    print(f"\n✓ Training setup complete\n")
    
    # Training loop
    print("="*70)
    print(f"Starting Training - Fold {args.fold}")
    print("="*70 + "\n")
    
    train_losses = []
    train_dscs = []
    val_dscs_patch = []
    val_dscs_recon = []
    
    best_recon_dsc = 0
    best_epoch = 0
    
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_patch,val_dsc_recon,lr\n")
    
    # Save fold info
    fold_info = {
        'fold': args.fold,
        'train_volumes': len(train_dataset.volumes),
        'val_volumes': len(val_dataset.volumes),
        'train_cases': [v['case_id'] for v in train_dataset.volumes],
        'val_cases': [v['case_id'] for v in val_dataset.volumes]
    }
    with open(exp_dir / 'fold_info.json', 'w') as f:
        json.dump(fold_info, f, indent=4)
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_dsc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Quick patch validation
        model.eval()
        val_loss_patch = 0
        val_dsc_patch = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val (patches)'):
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
                
                val_loss_patch += loss.item()
                val_dsc_patch += dsc
                num_batches += 1
        
        val_loss_patch /= num_batches
        val_dsc_patch /= num_batches
        
        # Full reconstruction validation
        if (epoch + 1) % args.validate_recon_every == 0 or epoch == 0:
            print("\n" + "="*70)
            print(f"FULL RECONSTRUCTION VALIDATION (Fold {args.fold}, Epoch {epoch+1})")
            print("="*70)
            
            save_nifti = ((epoch + 1) % args.save_nifti_every == 0)
            
            val_dsc_recon, all_dscs = validate_with_reconstruction(
                model, val_dataset, device, tuple(args.patch_size),
                save_nifti=save_nifti,
                save_dir=exp_dir,
                epoch=epoch+1
            )
            
            print(f"\nReconstructed DSC: {val_dsc_recon:.4f}")
            print(f"  Min: {np.min(all_dscs):.4f}")
            print(f"  Max: {np.max(all_dscs):.4f}")
            print(f"  Std: {np.std(all_dscs):.4f}")
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
        
        print(f"\n{'='*70}")
        print(f"Fold {args.fold} - Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val DSC (patch):  {val_dsc_patch:.4f}")
        print(f"  Val DSC (recon):  {val_dsc_recon:.4f}")
        print(f"  Gap (patch):  {train_dsc - val_dsc_patch:+.4f}")
        print(f"  Gap (recon):  {train_dsc - val_dsc_recon:+.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_dsc_recon > best_recon_dsc:
            best_recon_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_patch': val_dsc_patch,
                'val_dsc_recon': val_dsc_recon
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST RECON! DSC: {best_recon_dsc:.4f}")
        
        print(f"  Best Recon DSC: {best_recon_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon
            }, exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
        
        if (epoch + 1) % 5 == 0:
            plot_training_curves(
                train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
                exp_dir / f'training_curves_epoch_{epoch+1}.png'
            )
    
    print("\n" + "="*70)
    print(f"FOLD {args.fold} TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Reconstructed DSC: {best_recon_dsc:.4f} (Epoch {best_epoch})")
    print(f"Results: {exp_dir}")
    print(f"NIfTI files saved every {args.save_nifti_every} epochs")
    print("="*70 + "\n")
    
    plot_training_curves(
        train_losses, train_dscs, val_dscs_patch, val_dscs_recon,
        exp_dir / 'training_curves_final.png'
    )
    
    # Save final summary
    summary = {
        'fold': args.fold,
        'best_epoch': best_epoch,
        'best_recon_dsc': float(best_recon_dsc),
        'final_train_dsc': float(train_dscs[-1]),
        'final_val_dsc_patch': float(val_dscs_patch[-1]),
        'final_val_dsc_recon': float(val_dscs_recon[-1])
    }
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    main()

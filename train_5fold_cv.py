"""
train_5fold_cv_fixed.py

Five-fold cross-validation - FIXED to match single-fold behavior exactly

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
import torch.nn.functional as F
from collections import defaultdict

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
    """Reconstruct full volume from patches"""
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
    
    reconstructed = reconstructed / (count_map + 1e-6)
    return reconstructed


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
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


def validate_with_reconstruction(model, dataset, device, patch_size):
    """Validate with full reconstruction"""
    model.eval()
    
    volume_data = defaultdict(lambda: {'patches': [], 'centers': [], 'preds': []})
    
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
            
            volume_data[vol_idx]['patches'].append(image.cpu())
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
    
    all_dscs = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = dataset.get_volume_info(vol_idx)
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed = reconstruct_from_patches(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        reconstructed_binary = (reconstructed > 0.5).astype(np.float32)
        mask_gt = vol_info['mask'].astype(np.float32)
        
        intersection = (reconstructed_binary * mask_gt).sum()
        union = reconstructed_binary.sum() + mask_gt.sum()
        
        if union == 0:
            dsc = 1.0 if reconstructed_binary.sum() == 0 else 0.0
        else:
            dsc = (2. * intersection) / union
        
        all_dscs.append(dsc)
    
    return np.mean(all_dscs), all_dscs


def train_single_fold(fold_idx, train_cases, val_cases, args, exp_dir):
    """Train a single fold - FIXED to create proper splits file"""
    
    print("\n" + "="*70)
    print(f"FOLD {fold_idx + 1}/5")
    print("="*70)
    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases:   {len(val_cases)}")
    print("="*70 + "\n")
    
    device = torch.device('cuda:0')
    
    # Create fold directory
    fold_dir = exp_dir / f'fold_{fold_idx + 1}'
    fold_dir.mkdir(exist_ok=True)
    (fold_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # Load pre-trained encoder
    print("Loading pre-trained encoder...")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {checkpoint['epoch'] + 1}\n")
    
    # Create model
    model = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    
    # ✅ CRITICAL FIX: Create fold-specific splits file
    print("Creating fold-specific splits...")
    
    # Load original splits to determine dataset membership
    with open(args.splits_file, 'r') as f:
        original_splits = json.load(f)
    
    # Build case-to-dataset mapping
    case_to_dataset = {}
    for dataset_name in ['ATLAS', 'UOA_Private']:
        if dataset_name in original_splits:
            for split in ['train', 'val']:
                if split in original_splits[dataset_name]:
                    for case_id in original_splits[dataset_name][split]:
                        case_to_dataset[case_id] = dataset_name
    
    # Create fold-specific splits
    fold_splits = {
        'ATLAS': {'train': [], 'val': []},
        'UOA_Private': {'train': [], 'val': []}
    }
    
    for case_id in train_cases:
        if case_id in case_to_dataset:
            dataset_name = case_to_dataset[case_id]
            fold_splits[dataset_name]['train'].append(case_id)
    
    for case_id in val_cases:
        if case_id in case_to_dataset:
            dataset_name = case_to_dataset[case_id]
            fold_splits[dataset_name]['val'].append(case_id)
    
    # Save fold-specific splits file
    fold_splits_file = fold_dir / 'fold_splits.json'
    with open(fold_splits_file, 'w') as f:
        json.dump(fold_splits, f, indent=2)
    
    print(f"✓ Fold splits created:")
    print(f"  ATLAS train: {len(fold_splits['ATLAS']['train'])}, val: {len(fold_splits['ATLAS']['val'])}")
    print(f"  UOA_Private train: {len(fold_splits['UOA_Private']['train'])}, val: {len(fold_splits['UOA_Private']['val'])}\n")
    
    # ✅ Now create datasets using fold-specific splits file
    print("Loading datasets with fold-specific splits...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file=str(fold_splits_file),  # ✅ Use fold-specific splits!
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        augment=True,
        lesion_focus_ratio=args.lesion_focus_ratio
    )
    
    val_dataset = PatchDatasetWithCenters(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ATLAS', 'UOA_Private'],
        split='val',
        splits_file=str(fold_splits_file),  # ✅ Use fold-specific splits!
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
    
    print(f"✓ Datasets loaded:")
    print(f"  Train: {len(train_dataset.volumes)} volumes, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset.volumes)} volumes, {len(val_loader)} batches")
    print(f"  Total train patches: {len(train_dataset)}")
    print(f"  Total val patches: {len(val_dataset)}\n")
    
    # Setup training
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    best_recon_dsc = 0
    best_epoch = 0
    
    log_file = fold_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_patch,val_dsc_recon,lr\n")
    
    print("="*70)
    print(f"Starting Training - Fold {fold_idx + 1}")
    print("="*70 + "\n")
    
    # Training loop
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
        
        # Full reconstruction validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n{'='*70}")
            print(f"Fold {fold_idx + 1} - Epoch {epoch + 1} - Reconstruction Validation")
            print(f"{'='*70}")
            
            val_dsc_recon, all_dscs = validate_with_reconstruction(
                model, val_dataset, device, tuple(args.patch_size)
            )
            
            print(f"Reconstructed DSC: {val_dsc_recon:.4f}")
            print(f"  Min: {np.min(all_dscs):.4f}, Max: {np.max(all_dscs):.4f}, Median: {np.median(all_dscs):.4f}")
            print(f"{'='*70}\n")
        else:
            # Use previous value
            if epoch > 0:
                # Read last recon DSC from log
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        val_dsc_recon = float(lines[-1].strip().split(',')[4])
                    else:
                        val_dsc_recon = 0.0
            else:
                val_dsc_recon = 0.0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc_patch:.6f},{val_dsc_recon:.6f},{current_lr:.6f}\n")
        
        # Print epoch summary
        print(f"{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val DSC (patch):  {val_dsc_patch:.4f}")
        print(f"  Val DSC (recon):  {val_dsc_recon:.4f}")
        print(f"  Gap (patch):  {train_dsc - val_dsc_patch:+.4f}")
        print(f"  Gap (recon):  {train_dsc - val_dsc_recon:+.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_dsc_recon > best_recon_dsc:
            best_recon_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': fold_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon
            }, fold_dir / 'checkpoints' / 'best_model.pth')
            
            print(f"  ✓ NEW BEST for Fold {fold_idx + 1}: {best_recon_dsc:.4f}")
        
        print(f"  Best Recon DSC: {best_recon_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1} COMPLETE")
    print(f"Best Reconstructed DSC: {best_recon_dsc:.4f} (Epoch {best_epoch})")
    print(f"{'='*70}\n")
    
    return {
        'fold': fold_idx + 1,
        'best_dsc': best_recon_dsc,
        'best_epoch': best_epoch,
        'train_cases': len(train_cases),
        'val_cases': len(val_cases)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-checkpoint', type=str, required=True)
    parser.add_argument('--preprocessed-dir', type=str, 
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--splits-file', type=str, default='splits_stratified.json')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/5fold_cv_experiments')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10)
    parser.add_argument('--lesion-focus-ratio', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION - FIXED VERSION")
    print("="*70)
    print("Matches single-fold training behavior exactly")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f'5fold_cv_fixed_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original splits
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    # Get all case IDs
    all_cases = []
    for dataset_name in ['ATLAS', 'UOA_Private']:
        if dataset_name in splits:
            for split_name in ['train', 'val']:
                if split_name in splits[dataset_name]:
                    all_cases.extend(splits[dataset_name][split_name])
    
    all_cases = sorted(list(set(all_cases)))
    n_cases = len(all_cases)
    fold_size = n_cases // 5
    
    print(f"Total cases: {n_cases}")
    print(f"Cases per fold: ~{fold_size}\n")
    
    # Create 5 folds
    folds = []
    for i in range(5):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < 4 else n_cases
        val_cases = all_cases[start_idx:end_idx]
        train_cases = [c for c in all_cases if c not in val_cases]
        folds.append((train_cases, val_cases))
        print(f"Fold {i + 1}: {len(train_cases)} train, {len(val_cases)} val")
    
    print("\n" + "="*70)
    print("Starting 5-Fold Cross-Validation")
    print("="*70 + "\n")
    
    # Train each fold
    results = []
    for fold_idx, (train_cases, val_cases) in enumerate(folds):
        result = train_single_fold(fold_idx, train_cases, val_cases, args, exp_dir)
        results.append(result)
    
    # Calculate statistics
    dscs = [r['best_dsc'] for r in results]
    mean_dsc = np.mean(dscs)
    std_dsc = np.std(dscs)
    
    # Save results
    final_results = {
        'mean_dsc': float(mean_dsc),
        'std_dsc': float(std_dsc),
        'min_dsc': float(np.min(dscs)),
        'max_dsc': float(np.max(dscs)),
        'median_dsc': float(np.median(dscs)),
        'all_folds': results,
        'configuration': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'patch_size': args.patch_size,
            'patches_per_volume': args.patches_per_volume,
            'lesion_focus_ratio': args.lesion_focus_ratio,
            'learning_rate': args.lr,
            'pretrained_checkpoint': args.pretrained_checkpoint
        }
    }
    
    with open(exp_dir / '5fold_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # Print final summary
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Mean DSC: {mean_dsc:.4f} ± {std_dsc:.4f}")
    print(f"  Range:    [{np.min(dscs):.4f}, {np.max(dscs):.4f}]")
    print(f"  Median:   {np.median(dscs):.4f}")
    print(f"\nIndividual Folds:")
    for i, r in enumerate(results):
        print(f"  Fold {i+1}: {r['best_dsc']:.4f} (epoch {r['best_epoch']}, {r['train_cases']} train, {r['val_cases']} val)")
    print(f"\nResults saved to: {exp_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

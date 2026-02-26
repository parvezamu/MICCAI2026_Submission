"""
finetune_brats_t2flair_to_isles_COMPLETE.py

Fine-tune BraTS T2-FLAIR pretrained model on ISLES DWI
- Full volume reconstruction for validation
- Proper evaluation on ISLES test set
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
from collections import defaultdict
import json
import sys

sys.path.append('/home/pahm409')
from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
        
        if output.shape[2:] != input_size:
            output = torch.nn.functional.interpolate(
                output, size=input_size,
                mode='trilinear', align_corners=False
            )
        
        return output

def compute_dsc_torch(pred, target):
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
            dsc = compute_dsc_torch(pred_probs, masks)
        
        total_loss += loss.item()
        total_dsc += dsc
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / len(dataloader), total_dsc / len(dataloader)

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
    
    mask = count_map > 0
    reconstructed[mask] = reconstructed[mask] / count_map[mask]
    
    return reconstructed

def validate_with_reconstruction(model, val_dataset, device):
    """Validate with full volume reconstruction"""
    model.eval()
    
    num_volumes = len(val_dataset.volumes)
    
    if num_volumes == 0:
        return 0.0, []
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
    print("\n" + "="*80)
    print("VALIDATION (Full Volume Reconstruction)")
    print("="*80)
    print(f"Volumes: {num_volumes}")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Predicting'):
            images = batch['image'].to(device)
            centers = batch['center'].cpu().numpy()
            vol_indices = batch['vol_idx'].cpu().numpy()
            
            with autocast():
                outputs = model(images)
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(images)):
                vol_idx = vol_indices[i]
                volume_data[vol_idx]['centers'].append(centers[i])
                volume_data[vol_idx]['preds'].append(preds[i])
    
    all_dscs = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = val_dataset.get_volume_info(vol_idx)
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed = reconstruct_from_patches(
            preds, centers, vol_info['mask'].shape, patch_size=(96, 96, 96)
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
        print(f"  Min DSC: {np.min(all_dscs):.4f}")
        print(f"  Max DSC: {np.max(all_dscs):.4f}")
    print("="*80 + "\n")
    
    return mean_dsc, all_dscs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brats-checkpoint', type=str, required=True,
                       help='Path to BraTS T2-FLAIR pretrained checkpoint')
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/brats_t2flair_finetuned_isles')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001,
                       help='Learning rate (10x lower than pretraining)')
    args = parser.parse_args()
    
    set_seed(42 + args.fold)
    device = torch.device('cuda:0')
    
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FINE-TUNING: BraTS T2-FLAIR → ISLES DWI")
    print("="*80)
    print(f"Pretrained checkpoint: {args.brats_checkpoint}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr} (10x lower for fine-tuning)")
    print("="*80 + "\n")
    
    # Load pretrained model
    checkpoint = torch.load(args.brats_checkpoint, map_location=device)
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(encoder, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Loaded BraTS T2-FLAIR checkpoint")
    print(f"  Val DSC on BraTS: {checkpoint.get('val_dsc', 'N/A'):.4f}")
    print(f"  Pretraining: {checkpoint.get('pretraining', 'N/A')}\n")
    
    # Load ISLES data
    print("Loading ISLES datasets...")
    
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir=args.isles_dir,
        datasets=['ISLES2022_resampled'],
        split='train',
        splits_file='isles_splits_5fold_resampled.json',
        fold=args.fold,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True,
        lesion_focus_ratio=0.7,
        compute_lesion_bins=False
    )
    
    val_dataset = PatchDatasetWithCenters(
        preprocessed_dir=args.isles_dir,
        datasets=['ISLES2022_resampled'],
        split='val',
        splits_file='isles_splits_5fold_resampled.json',
        fold=args.fold,
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )
    
    print(f"\n✓ ISLES Train: {len(train_dataset.volumes)} volumes, {len(train_dataset)} patches")
    print(f"✓ ISLES Val: {len(val_dataset.volumes)} volumes, {len(val_dataset)} patches\n")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    
    # Setup training
    criterion = GDiceLossV2()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    best_dsc = 0.0
    best_epoch = 0
    
    # Log file
    log_file = output_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc,lr\n")
    
    print("="*80)
    print("STARTING FINE-TUNING")
    print("="*80 + "\n")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_dsc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, scaler)
        
        # Validate with full volume reconstruction
        val_dsc, all_dscs = validate_with_reconstruction(model, val_dataset, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc:.6f},{current_lr:.6f}\n")
        
        print(f"Train Loss: {train_loss:.4f}, Train DSC: {train_dsc:.4f}")
        print(f"Val DSC: {val_dsc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc': val_dsc,
                'all_val_dscs': all_dscs,
                'pretraining': 'BraTS2024_T2FLAIR_supervised',
                'finetuned_on': 'ISLES2022_DWI'
            }, output_dir / 'best_finetuned_model.pth')
            
            print(f"✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print("="*80 + "\n")
        
        # Clear cache
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "="*80)
    print(f"FINE-TUNING COMPLETE - Fold {args.fold}")
    print("="*80)
    print(f"Best Val DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"Checkpoint: {output_dir / 'best_finetuned_model.pth'}")
    print(f"Log: {log_file}")
    print("="*80 + "\n")
    
    # Save summary
    summary = {
        'fold': args.fold,
        'best_val_dsc': float(best_dsc),
        'best_epoch': best_epoch,
        'brats_checkpoint': args.brats_checkpoint,
        'brats_val_dsc': float(checkpoint.get('val_dsc', 0)),
        'improvement_over_random': 'TBD - compare with 69.85%'
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {output_dir / 'summary.json'}\n")

if __name__ == '__main__':
    main()

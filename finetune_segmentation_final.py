"""
finetune_segmentation_final.py

Final fine-tuning for stroke lesion segmentation using pre-trained encoder
Uses epoch 70 SimCLR checkpoint and trains for 150 epochs

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
import shutil

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


class GDiceLoss(nn.Module):
    """Generalized Dice Loss for 3D segmentation"""
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
    """Compute Dice Score Coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


class ResNet3DEncoder(nn.Module):
    """ResNet3D encoder that returns multi-scale features"""
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
    """3D U-Net decoder"""
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
        """Match sizes by cropping or padding"""
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
                
                x_up = x_up[
                    :, :,
                    d_start:d_start + d_skip,
                    h_start:h_start + h_skip,
                    w_start:w_start + w_skip
                ]
        
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
    """Full segmentation model with output size matching"""
    def __init__(self, encoder, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        
        # Resize output to match input size
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(
                seg_logits,
                size=input_size,
                mode='trilinear',
                align_corners=False
            )
        
        return seg_logits


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for segmentation"""
    def __init__(self, preprocessed_dir, datasets, split, splits_file):
        self.preprocessed_dir = Path(preprocessed_dir)
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        self.cases = []
        for dataset_name in datasets:
            if dataset_name not in splits:
                continue
            
            dataset_dir = self.preprocessed_dir / dataset_name
            case_ids = splits[dataset_name][split]
            
            for case_id in case_ids:
                npz_path = dataset_dir / f'{case_id}.npz'
                if npz_path.exists():
                    self.cases.append({
                        'dataset': dataset_name,
                        'case_id': case_id,
                        'npz_path': str(npz_path)
                    })
        
        print(f"Loaded {len(self.cases)} cases for {split} split from {datasets}")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        data = np.load(self.cases[idx]['npz_path'])
        image = torch.from_numpy(data['image']).unsqueeze(0).float()
        mask = torch.from_numpy(data['lesion_mask']).long()
        
        return {
            'image': image,
            'mask': mask,
            'case_id': self.cases[idx]['case_id'],
            'dataset': self.cases[idx]['dataset']
        }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
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
            loss = criterion(torch.softmax(outputs, dim=1), masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks.unsqueeze(1) == 1).float()
            dsc = compute_dsc(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    return total_loss / num_batches, total_dsc / num_batches


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1} - Validation'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(torch.softmax(outputs, dim=1), masks)
            
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks.unsqueeze(1) == 1).float()
            dsc = compute_dsc(pred_probs, target_onehot)
            
            total_loss += loss.item()
            total_dsc += dsc
            num_batches += 1
    
    return total_loss / num_batches, total_dsc / num_batches


def plot_training_curves(train_losses, train_dscs, val_losses, val_dscs, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GDice Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: DSC
    ax = axes[0, 1]
    ax.plot(epochs, train_dscs, 'b-', linewidth=2, label='Train DSC')
    ax.plot(epochs, val_dscs, 'r-', linewidth=2, label='Val DSC')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score Coefficient', fontsize=12)
    ax.set_title('Training and Validation DSC', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: DSC improvement
    ax = axes[1, 0]
    if len(train_dscs) > 1:
        train_improvement = [dsc - train_dscs[0] for dsc in train_dscs]
        val_improvement = [dsc - val_dscs[0] for dsc in val_dscs]
        ax.plot(epochs, train_improvement, 'b-', linewidth=2, label='Train')
        ax.plot(epochs, val_improvement, 'r-', linewidth=2, label='Val')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('DSC Improvement', fontsize=12)
        ax.set_title('DSC Improvement from Epoch 1', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Best DSC so far
    ax = axes[1, 1]
    best_val_dsc = [max(val_dscs[:i+1]) for i in range(len(epochs))]
    ax.plot(epochs, best_val_dsc, 'g-', linewidth=3, marker='*', markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Best Validation DSC', fontsize=12)
    ax.set_title('Best Validation DSC (Cumulative)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Final segmentation fine-tuning')
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                       help='Path to pre-trained SimCLR checkpoint (epoch 70)')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/segmentation_final_experiments',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Initial learning rate')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print("FINAL SEGMENTATION FINE-TUNING")
    print("Using Epoch 70 SimCLR Pre-trained Encoder")
    print("="*70 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f'segmentation_epoch70_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}\n")
    
    # Load pre-trained encoder
    print("Loading pre-trained encoder...")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded pre-trained encoder from epoch {checkpoint['epoch'] + 1}")
    print(f"  Pre-training train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"  Pre-training val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Create segmentation model
    model = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Segmentation model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup data
    print("\nLoading datasets...")
    train_dataset = SegmentationDataset(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_stratified.json'
    )
    
    val_dataset = SegmentationDataset(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='val',
        splits_file='splits_stratified.json'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ Data loaded")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Setup training
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    print(f"\n✓ Training setup complete")
    print(f"  Loss: Generalized Dice Loss")
    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Epochs: {args.epochs}")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Fine-tuning")
    print("="*70 + "\n")
    
    train_losses = []
    train_dscs = []
    val_losses = []
    val_dscs = []
    
    best_dsc = 0
    best_epoch = 0
    
    # Create log file
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_loss,val_dsc,lr\n")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_dsc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_dsc = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(train_loss)
        train_dscs.append(train_dsc)
        val_losses.append(val_loss)
        val_dscs.append(val_dsc)
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_loss:.6f},{val_dsc:.6f},{current_lr:.6f}\n")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs} Summary")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val DSC:   {val_dsc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_dsc': train_dsc,
                'val_loss': val_loss,
                'val_dsc': val_dsc,
                'best_dsc': best_dsc
            }
            
            torch.save(checkpoint, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST MODEL! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC so far: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_dsc': train_dsc,
                'val_loss': val_loss,
                'val_dsc': val_dsc
            }
            torch.save(checkpoint, exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")
        
        # Plot training curves
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_training_curves(
                train_losses, train_dscs, val_losses, val_dscs,
                exp_dir / 'plots' / f'training_curves_epoch_{epoch+1}.png'
            )
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Validation DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"Final Validation DSC: {val_dscs[-1]:.4f}")
    print(f"\nImprovement: {val_dscs[-1] - val_dscs[0]:.4f} ({(val_dscs[-1] - val_dscs[0])/max(val_dscs[0], 1e-6)*100:.1f}%)")
    print(f"\nResults saved to: {exp_dir}")
    print("="*70 + "\n")
    
    # Plot final curves
    plot_training_curves(
        train_losses, train_dscs, val_losses, val_dscs,
        exp_dir / 'training_curves_final.png'
    )
    
    print(f"✓ Final training curves: {exp_dir / 'training_curves_final.png'}")


if __name__ == '__main__':
    main()

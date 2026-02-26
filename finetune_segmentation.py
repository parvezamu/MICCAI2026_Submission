"""
finetune_segmentation.py

Fine-tune pre-trained SimCLR encoder for stroke lesion segmentation
Uses Generalized Dice Loss only

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys
from torch import einsum

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


class GDiceLoss(nn.Module):
    """
    Generalized Dice Loss for 3D segmentation
    """
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
    
    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch_size, class_num, d, h, w)
        shp_y = gt.shape  # (batch_size, 1, d, h, w) or (batch_size, d, h, w)
        
        # One hot encoding for ground truth
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
        
        # Convert to double precision for numerical stability
        net_output = net_output.double()
        y_onehot = y_onehot.double()
        
        # Compute class weights (inverse of class frequency squared)
        w = 1 / (einsum("bcdhw->bc", y_onehot).type(torch.float64) + 1e-10) ** 2
        
        # Compute weighted intersection and union
        intersection = w * einsum("bcdhw,bcdhw->bc", net_output, y_onehot)
        union = w * (einsum("bcdhw->bc", net_output) + einsum("bcdhw->bc", y_onehot))
        
        # Compute generalized dice coefficient
        divided = -2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()
        
        return gdc.float()


def compute_dsc(pred, target, smooth=1e-6):
    """
    Compute Dice Score Coefficient for evaluation
    
    Args:
        pred: Predictions (B, C, D, H, W)
        target: Ground truth (B, C, D, H, W)
    """
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


class ResNet3DEncoder(nn.Module):
    """
    ResNet3D encoder that returns multi-scale features
    Modified to output features at different scales for U-Net
    """
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
        # Initial conv
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)
        
        # Encoder blocks
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        return [x1, x2, x3, x4, x5]


class UNetDecoder3D(nn.Module):
    """3D U-Net decoder for segmentation"""
    
    def __init__(self, encoder_channels=[64, 64, 128, 256, 512], num_classes=2):
        super(UNetDecoder3D, self).__init__()
        
        # Decoder path
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        # Final segmentation head
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
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of features [x1, x2, x3, x4, x5] from encoder
        """
        x1, x2, x3, x4, x5 = encoder_features
        
        # Decoder with skip connections
        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        # Final segmentation
        x = self.final_conv(x)
        
        return x


class SegmentationModel(nn.Module):
    """Full segmentation model with pre-trained encoder"""
    
    def __init__(self, pretrained_encoder, num_classes=2):
        super(SegmentationModel, self).__init__()
        
        self.encoder = ResNet3DEncoder(pretrained_encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        # Get multi-scale features from encoder
        enc_features = self.encoder(x)
        
        # Decode to segmentation mask
        seg_logits = self.decoder(enc_features)
        
        return seg_logits


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for segmentation fine-tuning"""
    
    def __init__(self, preprocessed_dir, datasets, split, splits_file):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.datasets = datasets
        self.split = split
        
        # Load splits
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Collect cases
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
        
        print(f"Loaded {len(self.cases)} cases for {split} split")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_info = self.cases[idx]
        
        # Load data
        data = np.load(case_info['npz_path'])
        image = data['image']  # (D, H, W)
        mask = data['lesion_mask']  # (D, H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, D, H, W)
        mask = torch.from_numpy(mask).long()  # (D, H, W) - keep as long for GDice
        
        return {
            'image': image,
            'mask': mask,
            'case_id': case_info['case_id'],
            'dataset': case_info['dataset']
        }


def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=True):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        if use_amp:
            with autocast():
                outputs = model(images)  # (B, 2, D, H, W)
                loss = criterion(torch.softmax(outputs, dim=1), masks)
        else:
            outputs = model(images)
            loss = criterion(torch.softmax(outputs, dim=1), masks)
        
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Calculate DSC (foreground class only)
        with torch.no_grad():
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]  # (B, 1, D, H, W)
            target_onehot = (masks.unsqueeze(1) == 1).float()  # (B, 1, D, H, W)
            dsc = compute_dsc(pred_probs, target_onehot)
        
        total_loss += loss.item()
        total_dsc += dsc
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dsc': f'{dsc:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_dsc = total_dsc / num_batches
    
    return avg_loss, avg_dsc


def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(torch.softmax(outputs, dim=1), masks)
            else:
                outputs = model(images)
                loss = criterion(torch.softmax(outputs, dim=1), masks)
            
            # Calculate DSC
            pred_probs = torch.softmax(outputs, dim=1)[:, 1:2, ...]
            target_onehot = (masks.unsqueeze(1) == 1).float()
            dsc = compute_dsc(pred_probs, target_onehot)
            
            total_loss += loss.item()
            total_dsc += dsc
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_dsc = total_dsc / num_batches
    
    return avg_loss, avg_dsc


def main():
    parser = argparse.ArgumentParser(description='Fine-tune for segmentation with GDice Loss')
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                       help='Path to pre-trained SimCLR checkpoint')
    parser.add_argument('--output-dir', type=str, default='/home/pahm409/segmentation_experiments',
                       help='Output directory for experiments')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("Fine-tuning Pre-trained Encoder for Segmentation")
    print("Using Generalized Dice Loss")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f'segmentation_finetune_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Load pre-trained encoder
    print("\nLoading pre-trained encoder...")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    
    # Create encoder
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract encoder (without projection head)
    pretrained_encoder = simclr_model.encoder
    
    print("✓ Pre-trained encoder loaded")
    print(f"  Pre-trained at epoch: {checkpoint['epoch'] + 1}")
    print(f"  Pre-training val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # Create segmentation model
    model = SegmentationModel(pretrained_encoder, num_classes=2)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Segmentation model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup data
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n✓ Data loaded")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Setup training
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Create log file
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_loss,val_dsc,lr\n")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Fine-tuning")
    print("="*70 + "\n")
    
    best_dsc = 0
    train_losses = []
    train_dscs = []
    val_losses = []
    val_dscs = []
    
    for epoch in range(args.epochs):
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_dsc = validate(
            model, val_loader, criterion, device
        )
        
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
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train DSC: {train_dsc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val DSC: {val_dsc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc': val_dsc,
                'val_loss': val_loss
            }
            torch.save(best_checkpoint, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ New best DSC: {best_dsc:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc': val_dsc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\n" + "="*70)
    print("Fine-tuning Complete!")
    print(f"Best Validation DSC: {best_dsc:.4f}")
    print(f"Experiment saved to: {exp_dir}")
    print("="*70 + "\n")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax = axes[0]
    ax.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GDice Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DSC curves
    ax = axes[1]
    ax.plot(epochs_range, train_dscs, 'b-', label='Train DSC', linewidth=2)
    ax.plot(epochs_range, val_dscs, 'r-', label='Val DSC', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score Coefficient', fontsize=12)
    ax.set_title('Training and Validation DSC', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to: {exp_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()

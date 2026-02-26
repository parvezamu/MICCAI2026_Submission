"""
Ablation Study: Pre-trained vs Random Initialization

This script trains TWO models for 50 epochs:
1. WITH pre-training (your SimCLR checkpoint)
2. WITHOUT pre-training (random initialization)

Then compares:
- Convergence speed
- Final performance
- Training stability

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
from collections import defaultdict

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


# Import from your training script
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


def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training'):
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
    
    return total_loss / num_batches, total_dsc / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
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


def run_experiment(use_pretrained, pretrained_path, output_dir, fold, epochs, device):
    """Run one experiment (pretrained or random)"""
    
    exp_name = "pretrained" if use_pretrained else "random"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name.upper()}")
    print(f"{'='*70}\n")
    
    # Create encoder
    encoder = resnet3d_18(in_channels=1)
    
    if use_pretrained:
        print("Loading pre-trained checkpoint...")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
        simclr_model.load_state_dict(checkpoint['model_state_dict'])
        encoder = simclr_model.encoder
        print(f"✓ Loaded from SimCLR epoch {checkpoint['epoch'] + 1}\n")
    else:
        print("Using random initialization (no pre-training)\n")
    
    # Create segmentation model
    model = SegmentationModel(encoder, num_classes=2).to(device)
    
    # Setup data
    print("Loading datasets...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=fold,
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
        fold=fold,
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training loop
    train_losses = []
    train_dscs = []
    val_losses = []
    val_dscs = []
    learning_rates = []
    
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_loss,val_dsc,lr\n")
    
    best_val_dsc = 0
    best_epoch = 0
    
    print(f"Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        train_loss, train_dsc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dsc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        train_dscs.append(train_dsc)
        val_losses.append(val_loss)
        val_dscs.append(val_dsc)
        learning_rates.append(current_lr)
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_loss:.6f},{val_dsc:.6f},{current_lr:.6f}\n")
        
        # Print
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, DSC={train_dsc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, DSC={val_dsc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dsc': val_dsc
            }, exp_dir / 'best_model.pth')
            print(f"  ✓ NEW BEST! DSC={best_val_dsc:.4f}")
        print()
    
    # Save results
    results = {
        'experiment': exp_name,
        'use_pretrained': use_pretrained,
        'fold': fold,
        'epochs': epochs,
        'best_val_dsc': float(best_val_dsc),
        'best_epoch': best_epoch,
        'final_train_dsc': float(train_dscs[-1]),
        'final_val_dsc': float(val_dscs[-1]),
        'train_losses': [float(x) for x in train_losses],
        'train_dscs': [float(x) for x in train_dscs],
        'val_losses': [float(x) for x in val_losses],
        'val_dscs': [float(x) for x in val_dscs],
        'learning_rates': [float(x) for x in learning_rates]
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"{'='*70}")
    print(f"{exp_name.upper()} COMPLETE")
    print(f"Best Val DSC: {best_val_dsc:.4f} (Epoch {best_epoch})")
    print(f"{'='*70}\n")
    
    return results


def compare_results(pretrained_results, random_results, output_dir):
    """Create comparison plots"""
    
    print("\n" + "="*70)
    print("CREATING COMPARISON PLOTS")
    print("="*70 + "\n")
    
    epochs = range(1, len(pretrained_results['train_dscs']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training DSC
    ax = axes[0, 0]
    ax.plot(epochs, pretrained_results['train_dscs'], 'b-', linewidth=2, 
            label='Pre-trained', marker='o', markersize=3)
    ax.plot(epochs, random_results['train_dscs'], 'r--', linewidth=2,
            label='Random', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training DSC')
    ax.set_title('Training DSC: Pre-trained vs Random', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation DSC
    ax = axes[0, 1]
    ax.plot(epochs, pretrained_results['val_dscs'], 'b-', linewidth=2,
            label='Pre-trained', marker='o', markersize=3)
    ax.plot(epochs, random_results['val_dscs'], 'r--', linewidth=2,
            label='Random', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation DSC')
    ax.set_title('Validation DSC: Pre-trained vs Random', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight best performance
    ax.axhline(y=pretrained_results['best_val_dsc'], color='blue', 
              linestyle=':', alpha=0.5)
    ax.axhline(y=random_results['best_val_dsc'], color='red',
              linestyle=':', alpha=0.5)
    
    # Plot 3: Training Loss
    ax = axes[0, 2]
    ax.plot(epochs, pretrained_results['train_losses'], 'b-', linewidth=2,
            label='Pre-trained', marker='o', markersize=3)
    ax.plot(epochs, random_results['train_losses'], 'r--', linewidth=2,
            label='Random', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance Gap
    ax = axes[1, 0]
    gap_pre = [t - v for t, v in zip(pretrained_results['train_dscs'], 
                                      pretrained_results['val_dscs'])]
    gap_rand = [t - v for t, v in zip(random_results['train_dscs'],
                                       random_results['val_dscs'])]
    ax.plot(epochs, gap_pre, 'b-', linewidth=2, label='Pre-trained')
    ax.plot(epochs, gap_rand, 'r--', linewidth=2, label='Random')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train DSC - Val DSC')
    ax.set_title('Generalization Gap', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: DSC Improvement from Epoch 1
    ax = axes[1, 1]
    pre_improve = [dsc - pretrained_results['val_dscs'][0] 
                   for dsc in pretrained_results['val_dscs']]
    rand_improve = [dsc - random_results['val_dscs'][0]
                    for dsc in random_results['val_dscs']]
    ax.plot(epochs, pre_improve, 'b-', linewidth=2, label='Pre-trained')
    ax.plot(epochs, rand_improve, 'r--', linewidth=2, label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DSC Improvement from Epoch 1')
    ax.set_title('Learning Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    ABLATION STUDY RESULTS
    {'='*40}
    
    INITIAL PERFORMANCE (Epoch 1):
      Pre-trained: {pretrained_results['val_dscs'][0]:.4f}
      Random:      {random_results['val_dscs'][0]:.4f}
      Difference:  {pretrained_results['val_dscs'][0] - random_results['val_dscs'][0]:+.4f}
    
    BEST PERFORMANCE:
      Pre-trained: {pretrained_results['best_val_dsc']:.4f} (Epoch {pretrained_results['best_epoch']})
      Random:      {random_results['best_val_dsc']:.4f} (Epoch {random_results['best_epoch']})
      Difference:  {pretrained_results['best_val_dsc'] - random_results['best_val_dsc']:+.4f}
    
    FINAL PERFORMANCE (Epoch {len(epochs)}):
      Pre-trained: {pretrained_results['val_dscs'][-1]:.4f}
      Random:      {random_results['val_dscs'][-1]:.4f}
      Difference:  {pretrained_results['val_dscs'][-1] - random_results['val_dscs'][-1]:+.4f}
    
    CONVERGENCE SPEED:
      Epochs to reach 0.70 DSC:
        Pre-trained: {next((i+1 for i, d in enumerate(pretrained_results['val_dscs']) if d >= 0.70), 'N/A')}
        Random:      {next((i+1 for i, d in enumerate(random_results['val_dscs']) if d >= 0.70), 'N/A')}
    
    CONCLUSION:
      Pre-training provides:
      • Better initialization (+{pretrained_results['val_dscs'][0] - random_results['val_dscs'][0]:.2%} Epoch 1)
      • Better final performance (+{pretrained_results['best_val_dsc'] - random_results['best_val_dsc']:.2%})
      • Faster convergence
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = output_dir / 'ablation_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot: {save_path}\n")
    
    # Print summary
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                       help='Path to SimCLR pre-trained checkpoint')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/ablation_study_pretrained_vs_random',
                       help='Output directory')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4],
                       help='Which fold to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'fold_{args.fold}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ABLATION STUDY: PRE-TRAINED vs RANDOM INITIALIZATION")
    print("="*70)
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Run experiments
    pretrained_results = run_experiment(
        use_pretrained=True,
        pretrained_path=args.pretrained_checkpoint,
        output_dir=output_dir,
        fold=args.fold,
        epochs=args.epochs,
        device=device
    )
    
    random_results = run_experiment(
        use_pretrained=False,
        pretrained_path=None,
        output_dir=output_dir,
        fold=args.fold,
        epochs=args.epochs,
        device=device
    )
    
    # Compare
    compare_results(pretrained_results, random_results, output_dir)
    
    # Save combined results
    combined = {
        'pretrained': pretrained_results,
        'random': random_results,
        'comparison': {
            'initial_dsc_diff': pretrained_results['val_dscs'][0] - random_results['val_dscs'][0],
            'best_dsc_diff': pretrained_results['best_val_dsc'] - random_results['best_val_dsc'],
            'final_dsc_diff': pretrained_results['val_dscs'][-1] - random_results['val_dscs'][-1]
        }
    }
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(combined, f, indent=4)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

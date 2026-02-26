"""
compare_pretrained_vs_scratch.py

Fixed: Output size matches input size

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

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
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

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
        input_size = x.shape[2:]  # (D, H, W)
        
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
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        data = np.load(self.cases[idx]['npz_path'])
        image = torch.from_numpy(data['image']).unsqueeze(0).float()
        mask = torch.from_numpy(data['lesion_mask']).long()
        
        return {
            'image': image,
            'mask': mask,
            'case_id': self.cases[idx]['case_id']
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    scaler = GradScaler()
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
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
    
    return total_loss / num_batches, total_dsc / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_dsc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
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


def plot_comparison(results, save_path):
    """Plot side-by-side comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(results['pretrained']['train_dsc']) + 1)
    
    ax = axes[0, 0]
    ax.plot(epochs, results['pretrained']['train_dsc'], 'b-', linewidth=2, 
            label='Pre-trained', marker='o', markersize=4)
    ax.plot(epochs, results['scratch']['train_dsc'], 'r-', linewidth=2, 
            label='From Scratch', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('DSC', fontsize=12)
    ax.set_title('Training DSC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(epochs, results['pretrained']['val_dsc'], 'b-', linewidth=2, 
            label='Pre-trained', marker='o', markersize=4)
    ax.plot(epochs, results['scratch']['val_dsc'], 'r-', linewidth=2, 
            label='From Scratch', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('DSC', fontsize=12)
    ax.set_title('Validation DSC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    pretrained_improvement = [dsc - results['pretrained']['val_dsc'][0] 
                              for dsc in results['pretrained']['val_dsc']]
    scratch_improvement = [dsc - results['scratch']['val_dsc'][0] 
                          for dsc in results['scratch']['val_dsc']]
    ax.plot(epochs, pretrained_improvement, 'b-', linewidth=2, 
            label='Pre-trained', marker='o', markersize=4)
    ax.plot(epochs, scratch_improvement, 'r-', linewidth=2, 
            label='From Scratch', marker='s', markersize=4)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('DSC Improvement', fontsize=12)
    ax.set_title('DSC Improvement from Epoch 1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    best_pretrained = [max(results['pretrained']['val_dsc'][:i+1]) for i in range(len(epochs))]
    best_scratch = [max(results['scratch']['val_dsc'][:i+1]) for i in range(len(epochs))]
    ax.plot(epochs, best_pretrained, 'b-', linewidth=3, 
            label='Pre-trained', marker='o', markersize=6)
    ax.plot(epochs, best_scratch, 'r-', linewidth=3, 
            label='From Scratch', marker='s', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Best DSC So Far', fontsize=12)
    ax.set_title('Best Validation DSC (Cumulative)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON: Pre-trained vs From-scratch")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(f'/home/pahm409/comparison_experiments/comparison_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                           shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    
    print("\n" + "-"*70)
    print("Creating Model 1: Pre-trained Encoder")
    print("-"*70)
    
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
    encoder_pretrained = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder_pretrained, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    
    model_pretrained = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    optimizer_pretrained = optim.Adam(model_pretrained.parameters(), lr=args.lr)
    
    print(f"✓ Loaded pre-trained weights from epoch {checkpoint['epoch'] + 1}")
    
    print("\n" + "-"*70)
    print("Creating Model 2: Random Initialization")
    print("-"*70)
    
    encoder_scratch = resnet3d_18(in_channels=1)
    model_scratch = SegmentationModel(encoder_scratch, num_classes=2).to(device)
    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=args.lr)
    
    print("✓ Randomly initialized weights")
    
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    
    results = {
        'pretrained': {'train_loss': [], 'train_dsc': [], 'val_loss': [], 'val_dsc': []},
        'scratch': {'train_loss': [], 'train_dsc': [], 'val_loss': [], 'val_dsc': []}
    }
    
    print("\n" + "="*70)
    print("Starting Comparison Training")
    print("="*70 + "\n")
    
    best_dsc_pretrained = 0
    best_dsc_scratch = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        print("\n[Model 1: Pre-trained]")
        train_loss_p, train_dsc_p = train_epoch(model_pretrained, train_loader, 
                                                 criterion, optimizer_pretrained, device)
        val_loss_p, val_dsc_p = validate(model_pretrained, val_loader, criterion, device)
        
        results['pretrained']['train_loss'].append(train_loss_p)
        results['pretrained']['train_dsc'].append(train_dsc_p)
        results['pretrained']['val_loss'].append(val_loss_p)
        results['pretrained']['val_dsc'].append(val_dsc_p)
        
        if val_dsc_p > best_dsc_pretrained:
            best_dsc_pretrained = val_dsc_p
            torch.save(model_pretrained.state_dict(), exp_dir / 'best_pretrained.pth')
        
        print(f"  Train Loss: {train_loss_p:.4f}, Train DSC: {train_dsc_p:.4f}")
        print(f"  Val Loss: {val_loss_p:.4f}, Val DSC: {val_dsc_p:.4f} {'✓ BEST' if val_dsc_p == best_dsc_pretrained else ''}")
        
        print("\n[Model 2: From Scratch]")
        train_loss_s, train_dsc_s = train_epoch(model_scratch, train_loader, 
                                                 criterion, optimizer_scratch, device)
        val_loss_s, val_dsc_s = validate(model_scratch, val_loader, criterion, device)
        
        results['scratch']['train_loss'].append(train_loss_s)
        results['scratch']['train_dsc'].append(train_dsc_s)
        results['scratch']['val_loss'].append(val_loss_s)
        results['scratch']['val_dsc'].append(val_dsc_s)
        
        if val_dsc_s > best_dsc_scratch:
            best_dsc_scratch = val_dsc_s
            torch.save(model_scratch.state_dict(), exp_dir / 'best_scratch.pth')
        
        print(f"  Train Loss: {train_loss_s:.4f}, Train DSC: {train_dsc_s:.4f}")
        print(f"  Val Loss: {val_loss_s:.4f}, Val DSC: {val_dsc_s:.4f} {'✓ BEST' if val_dsc_s == best_dsc_scratch else ''}")
        
        print(f"\n{'─'*70}")
        print("COMPARISON:")
        print(f"  Pre-trained Val DSC: {val_dsc_p:.4f}")
        print(f"  From Scratch Val DSC: {val_dsc_s:.4f}")
        diff = val_dsc_p - val_dsc_s
        winner = 'Pre-trained WINS' if diff > 0 else 'From Scratch WINS' if diff < 0 else 'TIE'
        print(f"  Difference: {diff:+.4f} ({winner})")
        print(f"{'─'*70}")
        
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        plot_comparison(results, exp_dir / 'comparison_plot.png')
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nBest Validation DSC:")
    print(f"  Pre-trained: {best_dsc_pretrained:.4f}")
    print(f"  From Scratch: {best_dsc_scratch:.4f}")
    if best_dsc_scratch > 0:
        improvement = (best_dsc_pretrained - best_dsc_scratch)/best_dsc_scratch*100
        print(f"  Improvement: {best_dsc_pretrained - best_dsc_scratch:+.4f} ({improvement:+.1f}%)")
    
    print(f"\nResults saved to: {exp_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

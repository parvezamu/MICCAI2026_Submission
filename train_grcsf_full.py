"""
train_grcsf_full.py

Complete GRCSF Training (Pure GRCSF with MAE)

Three modes:
1. Random Init (baseline)
2. Pure GRCSF (GCU + RCU with MAE)
3. SimCLR + GRCSF (Hybrid: SimCLR encoder + GCU + RCU)

Author: Parvez
Date: January 2026
"""

import os
gpu_id = os.environ.get('GPU_ID', '3')
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import sys
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
from collections import defaultdict
import torch.nn.functional as F

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters
from grcsf_modules import GRCSFDecoder, MAEResidualGenerator


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
        
        # Compute per-class weights
        w = 1 / (torch.einsum("bcdhw->bc", y_onehot).type(torch.float64) + 1e-10) ** 2
        intersection = w * torch.einsum("bcdhw,bcdhw->bc", net_output.double(), y_onehot.double())
        union = w * (torch.einsum("bcdhw->bc", net_output.double()) + torch.einsum("bcdhw->bc", y_onehot.double()))
        
        divided = -2 * (torch.einsum("bc->b", intersection) + self.smooth) / (torch.einsum("bc->b", union) + self.smooth)
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
    """Wraps ResNet3D to extract multi-scale features"""
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


class GRCSFSegmentationModel(nn.Module):
    """Complete GRCSF segmentation model"""
    def __init__(self, encoder, num_classes=2, use_gcu=True, use_rcu=True, 
                 mae_residual_generator=None):
        super().__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = GRCSFDecoder(
            encoder_channels=[64, 64, 128, 256, 512],
            decoder_channels=[256, 128, 64, 64],
            num_classes=num_classes,
            use_gcu=use_gcu,
            use_rcu=use_rcu,
            rcu_layers=[1, 2, 3]  # Apply RCU to last 3 decoder layers
        )
        self.mae_generator = mae_residual_generator
        self.use_mae = (mae_residual_generator is not None)
    
    def forward(self, x):
        # Get MAE residuals if available
        if self.use_mae and self.training:
            with torch.no_grad():
                mae_residuals = self.mae_generator(x)
        elif self.use_mae:
            mae_residuals = self.mae_generator(x)
        else:
            mae_residuals = None
        
        # Encoder
        enc_features = self.encoder(x)
        
        # Decoder with GRCSF modules
        seg_logits = self.decoder(enc_features, mae_residuals)
        
        # Ensure output matches input size
        if seg_logits.shape[2:] != x.shape[2:]:
            seg_logits = F.interpolate(seg_logits, size=x.shape[2:], 
                                      mode='trilinear', align_corners=False)
        
        return seg_logits


def reconstruct_from_patches(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    """Reconstruct full volume from patches with averaging"""
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


def validate_with_reconstruction(model, dataset, device, patch_size):
    """Validate by reconstructing full volumes"""
    model.eval()
    
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})
    
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
            
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['preds'].append(pred[0])
    
    # Reconstruct volumes
    all_dscs = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = dataset.get_volume_info(vol_idx)
        
        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])
        
        reconstructed = reconstruct_from_patches(
            preds, centers, vol_info['mask'].shape, patch_size=patch_size
        )
        
        reconstructed_binary = (reconstructed > 0.5).astype(np.uint8)
        
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_mae):
    """Train one epoch"""
    model.train()
    
    # If using MAE, set it to eval mode
    if use_mae and hasattr(model, 'mae_generator'):
        model.mae_generator.eval()
    
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


def main():
    parser = argparse.ArgumentParser(description='GRCSF Training')
    
    # Model configuration
    parser.add_argument('--method', type=str, required=True,
                       choices=['random', 'grcsf', 'simclr_grcsf'],
                       help='Training method: random (baseline), grcsf (pure GRCSF), simclr_grcsf (hybrid)')
    
    # Checkpoints
    parser.add_argument('--simclr-checkpoint', type=str, default=None,
                       help='SimCLR pretrained checkpoint (for simclr_grcsf method)')
    parser.add_argument('--mae-checkpoint-50', type=str, default=None,
                       help='MAE checkpoint for 50%% masking (for grcsf and simclr_grcsf)')
    parser.add_argument('--mae-checkpoint-75', type=str, default=None,
                       help='MAE checkpoint for 75%% masking (for grcsf and simclr_grcsf)')
    
    # Training settings
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/grcsf_experiments_5fold')
    
    # Data settings
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10)
    parser.add_argument('--lesion-focus-ratio', type=float, default=0.7)
    
    # Validation settings
    parser.add_argument('--validate-recon-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print(f"GRCSF TRAINING - METHOD: {args.method.upper()}")
    print(f"FOLD: {args.fold}")
    print("="*70 + "\n")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / args.method / f'fold_{args.fold}' / f'exp_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}\n")
    
    # Build encoder
    encoder = resnet3d_18(in_channels=1)
    
    # Load SimCLR checkpoint if specified
    if args.method == 'simclr_grcsf':
        if args.simclr_checkpoint is None:
            raise ValueError("simclr_grcsf method requires --simclr-checkpoint")
        
        print("="*70)
        print("LOADING SIMCLR PRETRAINED ENCODER")
        print("="*70)
        checkpoint = torch.load(args.simclr_checkpoint, map_location=device)
        simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
        simclr_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded SimCLR from epoch {checkpoint['epoch'] + 1}")
        encoder = simclr_model.encoder
        print()
    
    # Build MAE residual generator if needed
    mae_generator = None
    if args.method in ['grcsf', 'simclr_grcsf']:
        if args.mae_checkpoint_50 is None or args.mae_checkpoint_75 is None:
            raise ValueError(f"{args.method} requires --mae-checkpoint-50 and --mae-checkpoint-75")
        
        print("="*70)
        print("LOADING MAE MODELS FOR RESIDUAL GENERATION")
        print("="*70)
        
        # TODO: Load actual MAE models
        # For now, placeholder - you need to implement MAE model loading
        print("⚠️  MAE loading not yet implemented - using placeholder")
        print("    You need to:")
        print("    1. Train MAE models (50% and 75% masking)")
        print("    2. Load them here for residual map generation")
        print()
        
        # mae_model_50 = load_mae_model(args.mae_checkpoint_50)
        # mae_model_75 = load_mae_model(args.mae_checkpoint_75)
        # mae_generator = MAEResidualGenerator(mae_model_50, mae_model_75, num_iterations=5)
    
    # Build model
    use_gcu = (args.method in ['grcsf', 'simclr_grcsf'])
    use_rcu = (args.method in ['grcsf', 'simclr_grcsf'])
    
    model = GRCSFSegmentationModel(
        encoder=encoder,
        num_classes=2,
        use_gcu=use_gcu,
        use_rcu=use_rcu,
        mae_residual_generator=mae_generator
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model built: {args.method}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Using GCU: {use_gcu}")
    print(f"  Using RCU: {use_rcu}")
    print()
    
    # Setup data
    print("Setting up datasets...")
    train_dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=args.fold,
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
        fold=args.fold,
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
    
    print(f"✓ Data loaded")
    print(f"  Train volumes: {len(train_dataset.volumes)}")
    print(f"  Val volumes: {len(val_dataset.volumes)}")
    print()
    
    # Setup training
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    print("="*70)
    print(f"STARTING TRAINING - {args.method.upper()}")
    print("="*70 + "\n")
    
    train_losses = []
    train_dscs = []
    val_dscs_recon = []
    
    best_recon_dsc = 0
    best_epoch = 0
    
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_recon,lr\n")
    
    # Save config
    config = vars(args)
    config['total_params'] = total_params
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mae=(mae_generator is not None)
        )
        
        # Full reconstruction validation
        if (epoch + 1) % args.validate_recon_every == 0 or epoch == 0:
            print(f"\n{'='*70}")
            print(f"FULL RECONSTRUCTION VALIDATION - Epoch {epoch+1}")
            print(f"{'='*70}")
            
            val_dsc_recon, all_dscs = validate_with_reconstruction(
                model, val_dataset, device, tuple(args.patch_size)
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
        val_dscs_recon.append(val_dsc_recon)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc_recon:.6f},{current_lr:.6f}\n")
        
        print(f"\n{'='*70}")
        print(f"{args.method.upper()} - Fold {args.fold} - Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val DSC (recon):  {val_dsc_recon:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_dsc_recon > best_recon_dsc:
            best_recon_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'method': args.method,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ NEW BEST! DSC: {best_recon_dsc:.4f}")
        
        print(f"  Best DSC: {best_recon_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'method': args.method,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon
            }, exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\n" + "="*70)
    print(f"{args.method.upper()} - FOLD {args.fold} COMPLETE!")
    print("="*70)
    print(f"Best DSC: {best_recon_dsc:.4f} (Epoch {best_epoch})")
    print(f"Results: {exp_dir}")
    print("="*70 + "\n")
    
    # Save summary
    summary = {
        'method': args.method,
        'fold': args.fold,
        'best_epoch': best_epoch,
        'best_recon_dsc': float(best_recon_dsc),
        'final_train_dsc': float(train_dscs[-1]),
        'final_val_dsc_recon': float(val_dscs_recon[-1])
    }
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    main()

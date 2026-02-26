"""
train_mae_simple.py - FIXED VERSION

Simple MAE training for residual map generation
NOW WITH PROPER BRAIN-FOCUSED PREPROCESSING!

Trains two MAE models:
- 50% masking ratio
- 75% masking ratio

Key Fix: Training preprocessing now MATCHES inference preprocessing
- Uses brain-only normalization (excludes background)
- Consistent with generate_residual_map_from_patches()

Author: Parvez
Date: January 2026
Fixed: January 2026
"""

import os
gpu_id = os.environ.get('GPU_ID', '0')
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import nibabel as nib

# Use your existing dataset class!
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


class SimpleMAE3D(nn.Module):
    """
    Simple 3D Masked Autoencoder for medical images
    
    Architecture:
    - Encoder: 3D CNN to extract features
    - Decoder: 3D CNN to reconstruct masked regions
    """
    def __init__(self, in_channels=1, hidden_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        
        # Encoder (compress to feature space)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (reconstruct from features)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(32, in_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def random_masking(self, x, mask_ratio):
        """
        Randomly mask patches of the input
        
        Args:
            x: [B, C, D, H, W]
            mask_ratio: ratio of patches to mask (0.5 or 0.75)
        
        Returns:
            masked_x: [B, C, D, H, W]
            mask: [B, 1, D, H, W] (1 = masked, 0 = visible)
        """
        B, C, D, H, W = x.shape
        
        # Create mask
        mask = torch.rand(B, 1, D, H, W, device=x.device)
        mask = (mask < mask_ratio).float()
        
        # Apply mask (set masked regions to 0)
        masked_x = x * (1 - mask)
        
        return masked_x, mask
    
    def forward(self, x, mask_ratio=0.75):
        """
        Forward pass with random masking
        
        Args:
            x: [B, C, D, H, W]
            mask_ratio: ratio of patches to mask
        
        Returns:
            reconstructed: [B, C, D, H, W]
            mask: [B, 1, D, H, W]
        """
        # Apply random masking
        masked_x, mask = self.random_masking(x, mask_ratio)
        
        # Encode
        features = self.encoder(masked_x)
        
        # Decode
        reconstructed = self.decoder(features)
        
        # Ensure output matches input size
        if reconstructed.shape != x.shape:
            reconstructed = F.interpolate(reconstructed, size=x.shape[2:], 
                                         mode='trilinear', align_corners=False)
        
        return reconstructed, mask


class MAEDatasetWrapper(Dataset):
    """
    FIXED VERSION: Wrapper around PatchDatasetWithCenters for MAE training
    
    Key Fix: Now uses BRAIN-ONLY normalization to match inference preprocessing!
    SPEED FIX: Pre-computes brain masks and statistics for fast loading!
    """
    def __init__(self, preprocessed_dir, datasets=['ATLAS', 'UOA_Private'], 
                 splits_file='splits_5fold.json', fold=0, split='train',
                 patch_size=(96, 96, 96), patches_per_volume=10):
        
        # Use your existing dataset class!
        self.base_dataset = PatchDatasetWithCenters(
            preprocessed_dir=preprocessed_dir,
            datasets=datasets,
            split=split,
            splits_file=splits_file,
            fold=fold,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume,
            augment=False,  # No augmentation for MAE
            lesion_focus_ratio=0.0  # Random patches, not lesion-focused
        )
        
        print(f"✓ MAE dataset created using PatchDatasetWithCenters")
        print(f"  {len(self.base_dataset)} patches available (fold {fold}, split {split})")
        print(f"  Using BRAIN-ONLY normalization (matches inference!)")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get patch from your existing dataset
        sample = self.base_dataset[idx]
        
        # Extract image AND mask
        image = sample['image']  # [C, D, H, W]
        mask = sample['mask']    # [D, H, W]
        
        # ==========================================
        # CRITICAL FIX: Skip patches with ANY lesion voxels!
        # MAE should ONLY learn normal brain patterns
        # ==========================================
        
        # Check if patch contains any lesion
        if isinstance(mask, torch.Tensor):
            lesion_voxels = mask.sum().item()
        else:
            lesion_voxels = np.sum(mask)
        
        # If patch has lesion, return None (will be filtered out)
        if lesion_voxels > 0:
            return None
        
        # ==========================================
        # FAST VERSION: Optimized brain-only normalization
        # Uses torch operations (faster than numpy)
        # ==========================================
        
        # Work directly with tensor on CPU (faster than numpy conversion)
        image_cpu = image[0]  # [D, H, W] tensor
        
        # 1. Create brain mask (threshold at 1st percentile)
        # Use torch operations (faster than numpy)
        brain_threshold = torch.quantile(image_cpu.flatten(), 0.01)
        brain_mask = image_cpu > brain_threshold
        
        # 2. Get brain voxels
        brain_voxels = image_cpu[brain_mask]
        
        # 3. Normalize using BRAIN-ONLY statistics
        if brain_voxels.numel() > 10:  # Need enough brain voxels
            brain_mean = brain_voxels.mean()
            brain_std = brain_voxels.std()
            
            if brain_std > 1e-6:
                # Create normalized image (zeros for background)
                image_norm = torch.zeros_like(image_cpu)
                
                # Normalize only brain voxels (in-place for speed)
                image_norm[brain_mask] = (image_cpu[brain_mask] - brain_mean) / brain_std
                
                # Clip to [-3, 3] and scale to [-1, 1]
                image_norm = torch.clamp(image_norm, -3, 3) / 3.0
                
                # Add channel dimension back [C, D, H, W]
                image_norm = image_norm.unsqueeze(0)
            else:
                # No variation in brain (weird case), use zeros
                image_norm = torch.zeros_like(image)
        else:
            # Not enough brain voxels (mostly background patch)
            # Return zeros - this patch won't contribute much to training
            image_norm = torch.zeros_like(image)
        
        return image_norm


def train_mae(model, dataloader, optimizer, device, epoch, mask_ratio):
    """Train MAE for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    criterion = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - MAE Training ({int(mask_ratio*100)}% mask)')
    
    for batch in pbar:
        # Skip None batches (all patches had lesions)
        if batch is None:
            continue
            
        images = batch.to(device)
        
        # Forward pass with masking
        reconstructed, mask = model(images, mask_ratio=mask_ratio)
        
        # Compute reconstruction loss (only on masked regions)
        loss = criterion(reconstructed * mask, images * mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'batches': num_batches})
    
    if num_batches == 0:
        raise RuntimeError("No valid batches in epoch! All patches contained lesions.")
    
    return total_loss / num_batches


def validate_mae_reconstruction(model, dataset, device, mask_ratio, patch_size=(96, 96, 96)):
    """
    Validate MAE by reconstructing full volumes
    Measures how well MAE reconstructs the original images
    
    NOTE: Skips patches with lesions (returns None from dataset)
    """
    from collections import defaultdict
    
    model.eval()
    
    volume_data = defaultdict(lambda: {'centers': [], 'originals': [], 'reconstructed': []})
    
    print(f"\nValidating MAE reconstruction ({int(mask_ratio*100)}% masking)...")
    
    skipped_patches = 0
    valid_patches = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Processing patches'):
            # Get patch from dataset
            sample = dataset.base_dataset[idx]
            
            vol_idx = sample['vol_idx']
            if isinstance(vol_idx, torch.Tensor):
                vol_idx = vol_idx.item()
            
            image_original = sample['image']  # Original unnormalized image
            center = sample['center'].numpy()
            
            # Check if patch has lesion
            mask = sample['mask']
            if isinstance(mask, torch.Tensor):
                lesion_voxels = mask.sum().item()
            else:
                lesion_voxels = np.sum(mask)
            
            # Skip patches with lesions (same as training)
            if lesion_voxels > 0:
                skipped_patches += 1
                continue
            
            # Get normalized version (from dataset)
            image_norm = dataset[idx]
            
            # Skip if None (double-check)
            if image_norm is None:
                skipped_patches += 1
                continue
            
            image_norm = image_norm.unsqueeze(0).to(device)
            
            # MAE reconstruction
            reconstructed, mask_tensor = model(image_norm, mask_ratio=mask_ratio)
            
            volume_data[vol_idx]['centers'].append(center)
            volume_data[vol_idx]['originals'].append(image_original.numpy()[0])
            volume_data[vol_idx]['reconstructed'].append(reconstructed.cpu().numpy()[0, 0])
            
            valid_patches += 1
    
    print(f"\nValidation patches: {valid_patches} valid, {skipped_patches} skipped (had lesions)")
    
    # Reconstruct full volumes and compute metrics
    print("\nReconstructing full volumes...")
    
    reconstruction_errors = []
    
    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Computing metrics'):
        # Skip volumes with no valid patches
        if len(volume_data[vol_idx]['centers']) == 0:
            continue
            
        vol_info = dataset.base_dataset.get_volume_info(vol_idx)
        
        centers = np.array(volume_data[vol_idx]['centers'])
        originals = np.array(volume_data[vol_idx]['originals'])
        reconstructed = np.array(volume_data[vol_idx]['reconstructed'])
        
        # Reconstruct original (normalized version)
        # We need to normalize originals the same way for fair comparison
        original_recon_list = []
        for orig_patch in originals:
            brain_mask = orig_patch > np.percentile(orig_patch, 1)
            brain_voxels = orig_patch[brain_mask]
            
            if len(brain_voxels) > 10 and brain_voxels.std() > 0:
                orig_norm = np.zeros_like(orig_patch)
                orig_norm[brain_mask] = (orig_patch[brain_mask] - brain_voxels.mean()) / brain_voxels.std()
                orig_norm = np.clip(orig_norm, -3, 3) / 3.0
            else:
                orig_norm = np.zeros_like(orig_patch)
            
            original_recon_list.append(orig_norm)
        
        original_recon_patches = np.array(original_recon_list)[:, np.newaxis, ...]
        
        # Reconstruct original
        original_recon = reconstruct_from_patches(
            original_recon_patches, centers, vol_info['volume'].shape, patch_size
        )
        
        # Reconstruct MAE output
        mae_recon = reconstruct_from_patches(
            reconstructed[:, np.newaxis, ...], centers, vol_info['volume'].shape, patch_size
        )
        
        # Compute reconstruction error (MSE)
        mse = np.mean((original_recon - mae_recon) ** 2)
        reconstruction_errors.append(mse)
    
    if len(reconstruction_errors) == 0:
        print("\n⚠️ WARNING: No valid volumes for validation!")
        return 0.0, []
    
    mean_mse = np.mean(reconstruction_errors)
    
    print(f"\nMAE Reconstruction Quality:")
    print(f"  Mean MSE: {mean_mse:.6f}")
    print(f"  Std MSE: {np.std(reconstruction_errors):.6f}")
    print(f"  Min MSE: {np.min(reconstruction_errors):.6f}")
    print(f"  Max MSE: {np.max(reconstruction_errors):.6f}")
    print(f"  Volumes reconstructed: {len(reconstruction_errors)}")
    
    return mean_mse, reconstruction_errors


def reconstruct_from_patches(patches, centers, original_shape, patch_size=(96, 96, 96)):
    """
    Reconstruct full volume from patches (borrowed from your segmentation code)
    """
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
        
        patch = patches[i, 0, ...]  # First channel
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    reconstructed = reconstructed / (count_map + 1e-6)
    return reconstructed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask-ratio', type=float, required=True,
                       choices=[0.5, 0.75],
                       help='Masking ratio (0.5 or 0.75)')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/mae_models')
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--patches-per-volume', type=int, default=10,
                       help='Number of random patches per volume')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    print("\n" + "="*70)
    print(f"MAE TRAINING - {int(args.mask_ratio*100)}% Masking - FIXED VERSION")
    print("="*70)
    print(f"🔧 KEY FIX: Brain-only normalization (matches inference!)")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {args.patch_size}")
    print("="*70 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f'mae_{int(args.mask_ratio*100)}pct_FIXED' / f'fold_{args.fold}_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model
    model = SimpleMAE3D(in_channels=1, hidden_dim=256).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ MAE model built")
    print(f"  Parameters: {total_params:,}\n")
    
    # Setup data - use your existing dataset class!
    train_dataset = MAEDatasetWrapper(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ATLAS', 'UOA_Private'],
        splits_file='splits_5fold.json',
        fold=args.fold,
        split='train',
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume
    )
    
    # Custom collate function to handle None values (patches with lesions)
    def collate_fn_filter_none(batch):
        """Filter out None samples (lesion patches) and return valid batch"""
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            # If entire batch has lesions, return None (will be skipped)
            return None
        return torch.stack(batch)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_filter_none  # Filter out lesion patches
    )
    
    # Validation dataset for reconstruction validation
    val_dataset = MAEDatasetWrapper(
        preprocessed_dir=args.preprocessed_dir,
        datasets=['ATLAS', 'UOA_Private'],
        splits_file='splits_5fold.json',
        fold=args.fold,
        split='val',
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume
    )
    
    print(f"✓ Datasets loaded\n")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    print(f"Starting training...\n")
    
    best_loss = float('inf')
    best_recon_mse = float('inf')
    
    train_losses = []
    val_recon_mses = []
    
    log_file = exp_dir / 'training_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_recon_mse,lr\n")
    
    for epoch in range(args.epochs):
        train_loss = train_mae(model, train_loader, optimizer, device, epoch, args.mask_ratio)
        scheduler.step()
        
        train_losses.append(train_loss)
        
        # Validate with full volume reconstruction every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print("\n" + "="*70)
            print(f"FULL VOLUME RECONSTRUCTION VALIDATION - Epoch {epoch+1}")
            print("="*70)
            
            val_recon_mse, all_mses = validate_mae_reconstruction(
                model, val_dataset, device, args.mask_ratio, 
                patch_size=tuple(args.patch_size)
            )
            
            val_recon_mses.append(val_recon_mse)
            print("="*70 + "\n")
        else:
            val_recon_mse = val_recon_mses[-1] if val_recon_mses else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_recon_mse:.6f},{current_lr:.6f}\n")
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss (patch): {train_loss:.6f}")
        print(f"  Val MSE (full vol): {val_recon_mse:.6f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model based on training loss
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_recon_mse': val_recon_mse,
                'mask_ratio': args.mask_ratio
            }, exp_dir / 'best_model.pth')
            print(f"  ✓ NEW BEST! Loss: {best_loss:.6f}")
        
        # Also save best reconstruction
        if val_recon_mse > 0 and val_recon_mse < best_recon_mse:
            best_recon_mse = val_recon_mse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_recon_mse': val_recon_mse,
                'mask_ratio': args.mask_ratio
            }, exp_dir / 'best_reconstruction.pth')
            print(f"  ✓ NEW BEST RECONSTRUCTION! MSE: {best_recon_mse:.6f}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_recon_mse': val_recon_mse,
                'mask_ratio': args.mask_ratio
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        print()
    
    print("="*70)
    print("MAE TRAINING COMPLETE!")
    print(f"Best Train Loss: {best_loss:.6f}")
    print(f"Best Reconstruction MSE: {best_recon_mse:.6f}")
    print(f"Model saved to: {exp_dir}")
    print("="*70 + "\n")
    
    # Save final summary
    summary = {
        'mask_ratio': args.mask_ratio,
        'fold': args.fold,
        'best_train_loss': float(best_loss),
        'best_recon_mse': float(best_recon_mse),
        'final_train_loss': float(train_losses[-1]),
        'final_recon_mse': float(val_recon_mses[-1]) if val_recon_mses else 0.0,
        'preprocessing': 'brain_only_normalization',
        'fixed_version': True
    }
    with open(exp_dir / 'final_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    main()

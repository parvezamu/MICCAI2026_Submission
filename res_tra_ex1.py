"""
resume_training.py

Resume training from checkpoint with FIXED reconstruction
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import sys
from pathlib import Path
import glob

sys.path.append('/home/pahm409/ISLES2029/')

# Import everything from your training script
from corr import *

# Replace the slow function with the fast one
validate_full_volumes = validate_full_volumes_FIXED


def resume_from_checkpoint(checkpoint_path, resume_epoch):
    """Resume training from a specific checkpoint"""
    
    checkpoint = torch.load(checkpoint_path)
    
    # Get configuration from checkpoint
    fold = checkpoint['fold']
    run_id = checkpoint['run_id']
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    print(f"\n{'='*80}")
    print(f"RESUMING TRAINING")
    print(f"{'='*80}")
    print(f"Fold: {fold}")
    print(f"Run: {run_id}")
    print(f"Attention: {attention_type}")
    print(f"Deep Supervision: {deep_supervision}")
    print(f"Resuming from epoch: {resume_epoch}")
    print(f"{'='*80}\n")
    
    # Find the experiment directory
    checkpoint_path = Path(checkpoint_path)
    exp_dir = checkpoint_path.parent.parent
    
    print(f"Experiment directory: {exp_dir}\n")
    
    # Load model
    device = torch.device('cuda:0')
    encoder = resnet3d_18(in_channels=1)
    
    # Check if SimCLR pretrained
    init_method = "SimCLR" if "SimCLR" in str(exp_dir) else "Random"
    
    model = SegmentationModel(encoder, num_classes=2,
                             attention_type=attention_type,
                             deep_supervision=deep_supervision).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Setup scheduler
    total_epochs = 100
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=total_epochs,
        base_lr=0.0001,
        min_lr=1e-6
    )
    
    # Fast forward scheduler to resume epoch
    for _ in range(resume_epoch):
        scheduler.step()
    
    # Load datasets
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
    
    criterion = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    scaler = GradScaler()
    
    # Load training history
    log_file = exp_dir / 'training_log.csv'
    train_losses, train_dscs, val_dscs_patch, val_dscs_recon = [], [], [], []
    
    if log_file.exists():
        import pandas as pd
        df = pd.read_csv(log_file)
        train_losses = df['train_loss'].tolist()
        train_dscs = df['train_dsc'].tolist()
        val_dscs_patch = df['val_dsc_patch'].tolist()
        val_dscs_recon = df['val_dsc_recon'].tolist()
    
    best_dsc = checkpoint.get('val_dsc_recon', 0.0)
    best_epoch = resume_epoch
    
    print("Continuing training...\n")
    
    # Continue training
    for epoch in range(resume_epoch, total_epochs):
        current_lr = scheduler.step(epoch)
        
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler,
            deep_supervision=deep_supervision,
            ds_weights=[1.0, 0.0, 0.0, 0.0] if deep_supervision else None,
            max_grad_norm=1.0
        )
        
        val_loss_patch, val_dsc_patch = validate_patches(
            model, val_loader, criterion, device,
            deep_supervision=deep_supervision
        )
        
        # FAST reconstruction every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n{'='*70}")
            print(f"FAST RECONSTRUCTION - Epoch {epoch+1}")
            print(f"{'='*70}\n")
            
            val_dsc_recon, all_dscs = validate_full_volumes_FIXED(
                model, val_dataset, device, (96, 96, 96),
                deep_supervision=deep_supervision,
                save_nifti=((epoch + 1) % 10 == 0),
                save_dir=exp_dir,
                epoch=epoch+1
            )
            
            print(f"Reconstructed DSC: {val_dsc_recon:.4f}\n")
        else:
            val_dsc_recon = val_dscs_recon[-1] if val_dscs_recon else 0.0
        
        # Update history
        train_losses.append(train_loss)
        train_dscs.append(train_dsc)
        val_dscs_patch.append(val_dsc_patch)
        val_dscs_recon.append(val_dsc_recon)
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},{val_dsc_patch:.6f},"
                   f"{val_dsc_recon:.6f},{current_lr:.6f}\n")
        
        print(f"{'='*70}")
        print(f"Epoch {epoch+1}/{total_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, DSC={train_dsc:.4f}")
        print(f"  Val:   Patch DSC={val_dsc_patch:.4f}, Recon DSC={val_dsc_recon:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_dsc_recon > best_dsc:
            best_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'run_id': run_id,
                'seed': checkpoint.get('seed'),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon,
                'attention_type': attention_type,
                'deep_supervision': deep_supervision
            }, exp_dir / 'checkpoints' / 'best_model.pth')
            
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--resume-epoch', type=int, required=True,
                       help='Epoch to resume from (e.g., 5)')
    
    args = parser.parse_args()
    
    resume_from_checkpoint(args.checkpoint, args.resume_epoch)

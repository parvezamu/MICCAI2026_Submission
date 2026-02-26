"""
finetune_on_isles_LOW_DATA.py

Fine-tune with LIMITED training data to show when transfer learning helps
FIXED: Uses modified splits file instead of subsetting dataset
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys
import random
import math

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters

# Import everything from finetune_on_isles_FIXED.py
from finetune_on_isles_FIXED import (
    set_seed,
    GDiceLossV2,
    softmax_helper,
    SegmentationModel,
    DiscriminativeLRScheduler,
    train_epoch,
    validate_isles_reconstruction,
    load_pretrained_model,
    create_isles_splits
)


def finetune_single_fold_low_data(
    pretrained_checkpoint,
    isles_preprocessed_dir,
    atlas_uoa_preprocessed_dir,
    fold,
    output_dir,
    max_train_cases=None,  # NEW: Limit training data
    epochs=50,
    batch_size=8,
    decoder_lr=0.0001,
    encoder_lr_ratio=0.1,
    freeze_encoder_epochs=3,
    isles_only=False
):
    """
    Fine-tune with LIMITED training data
    
    NEW PARAMETER:
        max_train_cases: Maximum number of training volumes to use
                        None = use all (default)
                        50 = use only 50 volumes (low-data regime)
                        25 = use only 25 volumes (very low-data regime)
    """
    
    set_seed(42 + fold)
    device = torch.device('cuda:0')
    
    # Create ISLES splits
    isles_splits_file = 'isles_splits_5fold_resampled.json'
    if not Path(isles_splits_file).exists():
        create_isles_splits(isles_preprocessed_dir, isles_splits_file)
    
    # Load pre-trained model
    print("\n" + "="*80)
    print(f"LOADING PRE-TRAINED MODEL (FOLD {fold})")
    print("="*80)
    model, attention_type, deep_supervision = load_pretrained_model(
        pretrained_checkpoint, device
    )
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_suffix = f'_maxcases{max_train_cases}' if max_train_cases else '_fulldata'
    exp_dir = Path(output_dir) / f'fold_{fold}' / f'finetune_LOWDATA{data_suffix}_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Save config
    config = {
        'pretrained_checkpoint': pretrained_checkpoint,
        'fold': fold,
        'max_train_cases': max_train_cases,
        'epochs': epochs,
        'batch_size': batch_size,
        'decoder_lr': decoder_lr,
        'encoder_lr_ratio': encoder_lr_ratio,
        'encoder_lr': decoder_lr * encoder_lr_ratio,
        'freeze_encoder_epochs': freeze_encoder_epochs,
        'isles_only': isles_only,
        'attention_type': attention_type,
        'deep_supervision': deep_supervision
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # ========================================================================
    # LOAD DATASETS WITH LIMITED TRAINING CASES
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    # Create limited splits file if needed
    if max_train_cases is not None:
        print(f"\n⚠️  LOW-DATA REGIME ACTIVATED")
        print(f"  Limiting training to {max_train_cases} cases")
        
        # Load original splits
        with open(isles_splits_file, 'r') as f:
            splits = json.load(f)
        
        fold_key = f'fold_{fold}'
        original_train_cases = splits[fold_key]['ISLES2022_resampled']['train']
        
        # Randomly select subset
        np.random.seed(42 + fold)
        if max_train_cases < len(original_train_cases):
            selected_train_cases = list(np.random.choice(
                original_train_cases, 
                max_train_cases, 
                replace=False
            ))
        else:
            selected_train_cases = original_train_cases
            print(f"  ℹ️  Requested {max_train_cases} but only {len(original_train_cases)} available")
        
        print(f"  ✓ Selected {len(selected_train_cases)} training cases")
        print(f"    First 5: {selected_train_cases[:5]}")
        
        # Create temporary splits file
        limited_splits = {
            fold_key: {
                'ISLES2022_resampled': {
                    'train': selected_train_cases,
                    'val': splits[fold_key]['ISLES2022_resampled']['val'],
                    'test': splits[fold_key]['ISLES2022_resampled']['test']
                }
            }
        }
        
        limited_splits_file = f'isles_splits_limited_{max_train_cases}cases_fold{fold}.json'
        with open(limited_splits_file, 'w') as f:
            json.dump(limited_splits, f, indent=2)
        
        print(f"  ✓ Created limited splits file: {limited_splits_file}")
        
        train_splits_file = limited_splits_file
    else:
        print(f"\n✓ Using FULL training data")
        train_splits_file = isles_splits_file
    
    # Load training dataset
    isles_train = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='train',
        splits_file=train_splits_file,
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True,
        lesion_focus_ratio=0.7,
        compute_lesion_bins=False
    )
    
    # Validation dataset (always use full validation set)
    isles_val = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='val',
        splits_file=isles_splits_file,  # Always use original splits
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )
    
    print(f"\n🔍 DATASET SUMMARY:")
    print(f"  ISLES train volumes: {len(isles_train.volumes)}")
    print(f"  ISLES val volumes: {len(isles_val.volumes)}")
    
    if len(isles_val.volumes) == 0:
        print(f"\n❌ ERROR: No ISLES validation volumes!")
        return None
    
    if isles_only:
        train_dataset = isles_train
        print(f"\n✓ Training on ISLES ONLY")
    else:
        atlas_uoa_train = PatchDatasetWithCenters(
            preprocessed_dir=atlas_uoa_preprocessed_dir,
            datasets=['ATLAS', 'UOA_Private'],
            split='train',
            splits_file='splits_5fold.json',
            fold=fold,
            patch_size=(96, 96, 96),
            patches_per_volume=10,
            augment=True,
            lesion_focus_ratio=0.7
        )
        
        train_dataset = ConcatDataset([atlas_uoa_train, isles_train])
        print(f"\n✓ Using MIXED dataset")
        print(f"  ATLAS/UOA: {len(atlas_uoa_train)}")
        print(f"  ISLES: {len(isles_train)}")
        print(f"  Total: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    
    # ========================================================================
    # DISCRIMINATIVE LEARNING RATE SETUP
    # ========================================================================
    
    print("\n" + "="*80)
    print("OPTIMIZER SETUP (DISCRIMINATIVE LEARNING RATES)")
    print("="*80)
    
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    encoder_lr = decoder_lr * encoder_lr_ratio
    
    print(f"✓ Encoder parameters: {len(encoder_params)}")
    print(f"  Learning rate: {encoder_lr:.6f} ({encoder_lr_ratio}× decoder)")
    print(f"✓ Decoder parameters: {len(decoder_params)}")
    print(f"  Learning rate: {decoder_lr:.6f}")
    print("="*80 + "\n")
    
    # Setup training
    criterion = GDiceLossV2(apply_nonlin=softmax_helper, smooth=1e-5)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
        {'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'}
    ], weight_decay=0.01)
    
    scheduler = DiscriminativeLRScheduler(
        optimizer=optimizer,
        warmup_epochs=2,
        total_epochs=epochs,
        encoder_base_lr=encoder_lr,
        decoder_base_lr=decoder_lr,
        min_lr=1e-7
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_dsc = 0.0
    best_epoch = 0
    
    log_file = exp_dir / 'finetune_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_dsc,val_dsc_recon,encoder_lr,decoder_lr,encoder_frozen\n")
    
    print("\n" + "="*80)
    print(f"STARTING FINE-TUNING (FOLD {fold})")
    if max_train_cases:
        print(f"LOW-DATA REGIME: {max_train_cases} training cases")
    print("="*80)
    
    for epoch in range(epochs):
        
        # Freeze/unfreeze encoder
        encoder_frozen = epoch < freeze_encoder_epochs
        
        if encoder_frozen:
            for param in model.encoder.parameters():
                param.requires_grad = False
            if epoch == 0:
                print(f"\n🔒 PHASE 1: Encoder FROZEN (epochs 1-{freeze_encoder_epochs})")
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
            if epoch == freeze_encoder_epochs:
                print(f"\n🔓 PHASE 2: Encoder UNFROZEN (epochs {freeze_encoder_epochs+1}-{epochs})")
        
        # Update learning rates
        current_encoder_lr, current_decoder_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_dsc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler,
            deep_supervision=deep_supervision,
            ds_weights=[1.0, 0.0, 0.0, 0.0] if deep_supervision else None,
            max_grad_norm=1.0
        )
        
        # Validate
        val_dsc_recon, all_dscs = validate_isles_reconstruction(
            model, isles_val, device, (96, 96, 96),
            deep_supervision=deep_supervision
        )
        
        # Log
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_dsc:.6f},"
                   f"{val_dsc_recon:.6f},{current_encoder_lr:.6f},"
                   f"{current_decoder_lr:.6f},{encoder_frozen}\n")
        
        # Print summary
        print(f"{'='*80}")
        print(f"Fold {fold} - Epoch {epoch+1}/{epochs}")
        if max_train_cases:
            print(f"  (Training with {max_train_cases} cases)")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f}  |  Train DSC: {train_dsc:.4f}")
        print(f"  Val DSC: {val_dsc_recon:.4f}")
        print(f"  Encoder LR: {current_encoder_lr:.6f} {'(FROZEN)' if encoder_frozen else ''}")
        print(f"  Decoder LR: {current_decoder_lr:.6f}")
        
        # Save best model
        if val_dsc_recon > best_dsc:
            best_dsc = val_dsc_recon
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'max_train_cases': max_train_cases,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc_recon': val_dsc_recon,
                'attention_type': attention_type,
                'deep_supervision': deep_supervision,
                'config': config,
                'finetuned_on': 'ISLES_DWI',
                'pretrained_from': pretrained_checkpoint
            }, exp_dir / 'checkpoints' / 'best_finetuned_model.pth')
            
            print(f"  ✓ NEW BEST! DSC: {best_dsc:.4f}")
        
        print(f"  Best DSC: {best_dsc:.4f} (Epoch {best_epoch})")
        print(f"{'='*80}\n")
        
        # Clear GPU cache periodically
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "="*80)
    print(f"FOLD {fold} FINE-TUNING COMPLETE")
    if max_train_cases:
        print(f"LOW-DATA REGIME: {max_train_cases} training cases")
    print("="*80)
    print(f"  Best Val DSC: {best_dsc:.4f} (Epoch {best_epoch})")
    print(f"  Checkpoint: {exp_dir / 'checkpoints' / 'best_finetuned_model.pth'}")
    print("="*80 + "\n")
    
    return {
        'fold': fold,
        'max_train_cases': max_train_cases,
        'best_dsc': best_dsc,
        'best_epoch': best_epoch,
        'checkpoint_path': str(exp_dir / 'checkpoints' / 'best_finetuned_model.pth'),
        'config': config
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune with LIMITED training data')
    
    parser.add_argument('--pretrained-checkpoint', type=str, required=True)
    parser.add_argument('--isles-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--atlas-uoa-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/finetuned_low_data')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--max-train-cases', type=int, default=None,
                       help='Maximum training volumes (None=all, 50=low-data, 25=very-low-data)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--decoder-lr', type=float, default=0.0001)
    parser.add_argument('--encoder-lr-ratio', type=float, default=0.1)
    parser.add_argument('--freeze-encoder-epochs', type=int, default=3)
    parser.add_argument('--isles-only', action='store_true')
    
    args = parser.parse_args()
    
    finetune_single_fold_low_data(
        pretrained_checkpoint=args.pretrained_checkpoint,
        isles_preprocessed_dir=args.isles_dir,
        atlas_uoa_preprocessed_dir=args.atlas_uoa_dir,
        fold=args.fold,
        output_dir=args.output_dir,
        max_train_cases=args.max_train_cases,
        epochs=args.epochs,
        batch_size=args.batch_size,
        decoder_lr=args.decoder_lr,
        encoder_lr_ratio=args.encoder_lr_ratio,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        isles_only=args.isles_only
    )

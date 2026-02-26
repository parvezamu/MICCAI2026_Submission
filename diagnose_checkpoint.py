# Save as diagnose_checkpoint.py
import torch
import json
from pathlib import Path

checkpoint_path = "/home/pahm409/finetuned_on_isles_5fold_FIXED/fold_0/finetune_FIXED_20260206_010601/checkpoints/best_finetuned_model.pth"
config_path = "/home/pahm409/finetuned_on_isles_5fold_FIXED/fold_0/finetune_FIXED_20260206_010601/config.json"
log_path = "/home/pahm409/finetuned_on_isles_5fold_FIXED/fold_0/finetune_FIXED_20260206_010601/finetune_log.csv"

print("="*80)
print("CHECKPOINT DIAGNOSIS")
print("="*80)

# Check checkpoint
if Path(checkpoint_path).exists():
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print("\n✓ Checkpoint loaded")
    print(f"  Epoch saved: {ckpt.get('epoch', 'N/A') + 1}")
    print(f"  Val DSC: {ckpt.get('val_dsc_recon', 'N/A'):.4f}" if 'val_dsc_recon' in ckpt else "  Val DSC: N/A")
    print(f"  Pretrained from: {ckpt.get('pretrained_from', 'N/A')}")
    
    if 'config' in ckpt:
        cfg = ckpt['config']
        print("\n  Training config:")
        print(f"    Encoder LR ratio: {cfg.get('encoder_lr_ratio', 'NOT FOUND')}")
        print(f"    Freeze epochs: {cfg.get('freeze_encoder_epochs', 'NOT FOUND')}")
        
        if cfg.get('encoder_lr_ratio') == 0.1 and cfg.get('freeze_encoder_epochs') == 3:
            print("    ✓ This IS a discriminative LR checkpoint!")
        else:
            print("    ❌ This is NOT a discriminative LR checkpoint!")
else:
    print("\n❌ Checkpoint not found!")

# Check config
if Path(config_path).exists():
    with open(config_path) as f:
        config = json.load(f)
    print("\n✓ Config file found")
    print(f"  Pretrained checkpoint: {config.get('pretrained_checkpoint', 'N/A')}")
else:
    print("\n❌ Config not found!")

# Check log
if Path(log_path).exists():
    import pandas as pd
    df = pd.read_csv(log_path)
    print("\n✓ Training log found")
    print(f"  Epochs completed: {len(df)}")
    print(f"  Best val DSC: {df['val_dsc_recon'].max():.4f} (epoch {df['val_dsc_recon'].idxmax() + 1})")
    print(f"  Final val DSC: {df['val_dsc_recon'].iloc[-1]:.4f}")
    
    if 'encoder_lr' in df.columns:
        print(f"\n  Learning rate verification:")
        print(f"    Encoder LR (epoch 1): {df['encoder_lr'].iloc[0]:.6f}")
        print(f"    Decoder LR (epoch 1): {df['decoder_lr'].iloc[0]:.6f}")
        print(f"    Ratio: {df['encoder_lr'].iloc[0] / df['decoder_lr'].iloc[0]:.2f}")
        
        if abs(df['encoder_lr'].iloc[0] / df['decoder_lr'].iloc[0] - 0.1) < 0.01:
            print("    ✓ Discriminative LR confirmed!")
        else:
            print("    ❌ NOT discriminative LR!")
else:
    print("\n❌ Training log not found!")

print("\n" + "="*80)

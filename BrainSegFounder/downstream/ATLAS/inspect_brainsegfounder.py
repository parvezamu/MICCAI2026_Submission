"""
Inspect BrainSegFounder ATLAS checkpoint
FIXED: Handle DistributedDataParallel and PyTorch 2.6 security
"""
import torch
from pathlib import Path
import warnings

# Allow DistributedDataParallel for loading
torch.serialization.add_safe_globals([torch.nn.parallel.distributed.DistributedDataParallel])

checkpoint_paths = [
    "/home/pahm409/ISLES2029/ATLAS_check/ATLAS_Stage_2_bestValRMSE.pt",
    "/home/pahm409/ISLES2029/ATLAS_check/finetune_best_val_loss.pt"
]

for ckpt_path in checkpoint_paths:
    print("="*70)
    print(f"Checkpoint: {Path(ckpt_path).name}")
    print("="*70)
    
    if not Path(ckpt_path).exists():
        print("❌ File not found!")
        continue
    
    try:
        # Try with weights_only=False (since we trust BrainSegFounder)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        continue
    
    print("\nTop-level keys:")
    for key in ckpt.keys():
        print(f"  - {key}")
    
    # Check model state dict
    state_dict_keys = ['model_state_dict', 'state_dict', 'model', 'net']
    state_dict = None
    
    for key in state_dict_keys:
        if key in ckpt:
            state_dict = ckpt[key]
            print(f"\nFound model weights in key: '{key}'")
            break
    
    if state_dict is not None:
        # Handle DistributedDataParallel wrapper
        if isinstance(state_dict, torch.nn.parallel.DistributedDataParallel):
            print("  (Wrapped in DistributedDataParallel)")
            state_dict = state_dict.module.state_dict()
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        
        print(f"\nModel has {len(state_dict)} parameters")
        
        # Analyze architecture
        encoder_params = [k for k in state_dict.keys() if 'encoder' in k or 'swinViT' in k or 'patch_embed' in k]
        decoder_params = [k for k in state_dict.keys() if 'decoder' in k or 'up' in k]
        
        print(f"  Encoder parameters: {len(encoder_params)}")
        print(f"  Decoder parameters: {len(decoder_params)}")
        
        # Check architecture type
        is_swin = any('swin' in k.lower() for k in state_dict.keys())
        is_resnet = any('resnet' in k.lower() or 'conv1' in k for k in state_dict.keys())
        
        print(f"\nArchitecture detection:")
        print(f"  Swin Transformer: {is_swin}")
        print(f"  ResNet: {is_resnet}")
        
        print("\nFirst 15 parameter names:")
        for i, (name, param) in enumerate(state_dict.items()):
            if i >= 15:
                print("  ...")
                break
            print(f"  {name}: {param.shape}")
    else:
        print("\n❌ Could not find model state dict")
    
    # Check other metadata
    print("\nMetadata:")
    for key in ['epoch', 'best_acc', 'best_loss', 'best_metric', 'arch', 'config']:
        if key in ckpt:
            value = ckpt[key]
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())[:5]}...")
            else:
                print(f"  {key}: {value}")
    
    print()

print("="*70)
print("COMPATIBILITY ASSESSMENT")
print("="*70)
print()
print("BrainSegFounder uses Swin UNETR architecture")
print("Your pipeline uses ResNet3D + U-Net")
print()
print("Options:")
print("  1. Use BrainSegFounder as-is (evaluate their full model)")
print("  2. Extract features for comparison (qualitative)")
print("  3. Focus on your ATLAS SimCLR results (same architecture)")
print()
print("Recommendation: Focus on Option 3 (your ATLAS SimCLR)")

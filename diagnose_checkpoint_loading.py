"""
Diagnostic: Check if Pre-trained Checkpoint is Loading Correctly

Author: Parvez
Date: January 2026
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


def check_checkpoint():
    """Verify the SimCLR checkpoint"""
    
    checkpoint_path = Path('/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth')
    
    print("="*70)
    print("CHECKPOINT VERIFICATION")
    print("="*70)
    print(f"File: {checkpoint_path}")
    print(f"Exists: {checkpoint_path.exists()}")
    print(f"Size: {checkpoint_path.stat().st_size / 1e6:.1f} MB\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint Contents:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTraining Info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return checkpoint


def compare_weight_loading():
    """Compare random vs loaded weights"""
    
    print("\n" + "="*70)
    print("WEIGHT LOADING VERIFICATION")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Method 1: How your ablation script loads it
    print("Method 1: Ablation Script Method")
    encoder1 = resnet3d_18(in_channels=1)
    simclr1 = SimCLRModel(encoder1, projection_dim=128, hidden_dim=512)
    simclr1.load_state_dict(checkpoint['model_state_dict'])
    encoder_loaded = simclr1.encoder
    
    # Method 2: Random initialization
    print("Method 2: Random Initialization")
    encoder_random = resnet3d_18(in_channels=1)
    
    # Compare first layer weights
    print("\nComparing conv1 weights:")
    
    loaded_conv1 = encoder_loaded.conv1.weight.data
    random_conv1 = encoder_random.conv1.weight.data
    
    print(f"  Loaded - Mean: {loaded_conv1.mean():.6f}, Std: {loaded_conv1.std():.6f}")
    print(f"  Random - Mean: {random_conv1.mean():.6f}, Std: {random_conv1.std():.6f}")
    
    # Check if they're the same (they shouldn't be!)
    diff = torch.norm(loaded_conv1 - random_conv1).item()
    print(f"  L2 Distance: {diff:.6f}")
    
    if diff < 0.01:
        print("  ⚠️  WARNING: Weights are nearly identical! Not loaded correctly!")
        return False
    else:
        print("  ✓ Weights are different - loading appears successful")
        return True


def test_encoder_output():
    """Test if encoder produces different outputs"""
    
    print("\n" + "="*70)
    print("ENCODER OUTPUT TEST")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load pre-trained
    encoder1 = resnet3d_18(in_channels=1)
    simclr1 = SimCLRModel(encoder1, projection_dim=128, hidden_dim=512)
    simclr1.load_state_dict(checkpoint['model_state_dict'])
    encoder_pretrained = simclr1.encoder
    encoder_pretrained.eval()
    
    # Random
    encoder_random = resnet3d_18(in_channels=1)
    encoder_random.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 96, 96, 96)
    
    print("Testing with dummy input (1, 1, 96, 96, 96)...")
    
    with torch.no_grad():
        # Get encoder features (not full forward, just check if it runs)
        out_pretrained = encoder_pretrained.conv1(dummy_input)
        out_random = encoder_random.conv1(dummy_input)
    
    print(f"\nPre-trained conv1 output:")
    print(f"  Shape: {out_pretrained.shape}")
    print(f"  Mean: {out_pretrained.mean():.6f}")
    print(f"  Std: {out_pretrained.std():.6f}")
    
    print(f"\nRandom conv1 output:")
    print(f"  Shape: {out_random.shape}")
    print(f"  Mean: {out_random.mean():.6f}")
    print(f"  Std: {out_random.std():.6f}")
    
    diff = torch.norm(out_pretrained - out_random).item()
    print(f"\nL2 Distance: {diff:.6f}")
    
    if diff < 1.0:
        print("⚠️  WARNING: Outputs are very similar! Something may be wrong.")
        return False
    else:
        print("✓ Outputs are different - encoder is working")
        return True


def check_simclr_vs_segmentation_loading():
    """Check if there's an issue with how we extract encoder from SimCLR"""
    
    print("\n" + "="*70)
    print("SIMCLR → SEGMENTATION ENCODER EXTRACTION")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checking state_dict keys...")
    state_dict = checkpoint['model_state_dict']
    
    # Check what keys exist
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
    projection_keys = [k for k in state_dict.keys() if k.startswith('projection')]
    
    print(f"\nEncoder keys: {len(encoder_keys)}")
    print("First 5 encoder keys:")
    for key in encoder_keys[:5]:
        print(f"  {key}")
    
    print(f"\nProjection head keys: {len(projection_keys)}")
    print("First 5 projection keys:")
    for key in projection_keys[:5]:
        print(f"  {key}")
    
    # Try loading just encoder
    print("\nTrying to load encoder directly...")
    encoder = resnet3d_18(in_channels=1)
    
    # Extract encoder weights
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    
    print(f"Extracted {len(encoder_state)} encoder parameters")
    
    # Try to load
    try:
        encoder.load_state_dict(encoder_state, strict=False)
        print("✓ Direct encoder loading successful")
        
        # Check if weights actually loaded
        conv1_weight = encoder.conv1.weight.data
        print(f"  conv1 weight - Mean: {conv1_weight.mean():.6f}, Std: {conv1_weight.std():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Direct encoder loading failed: {e}")
        return False


def compare_ablation_script_loading():
    """Exactly replicate what the ablation script does"""
    
    print("\n" + "="*70)
    print("EXACT ABLATION SCRIPT REPLICATION")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    
    # EXACTLY what the ablation script does
    print("Step 1: Load checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    print("\nStep 2: Create encoder")
    encoder = resnet3d_18(in_channels=1)
    print(f"  ✓ Created ResNet3D-18")
    
    print("\nStep 3: Create SimCLR model")
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    print(f"  ✓ Created SimCLR wrapper")
    
    print("\nStep 4: Load state dict")
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded state dict")
    
    print("\nStep 5: Extract encoder")
    encoder_final = simclr_model.encoder
    print(f"  ✓ Extracted encoder")
    
    # Verify
    print("\nVerification:")
    conv1_weight = encoder_final.conv1.weight.data
    print(f"  conv1 - Mean: {conv1_weight.mean():.6f}, Std: {conv1_weight.std():.6f}")
    print(f"  conv1 - Min: {conv1_weight.min():.6f}, Max: {conv1_weight.max():.6f}")
    
    # Compare with fresh random
    encoder_random = resnet3d_18(in_channels=1)
    random_conv1 = encoder_random.conv1.weight.data
    
    diff = torch.norm(conv1_weight - random_conv1).item()
    print(f"\n  L2 diff from random: {diff:.2f}")
    
    if diff > 5.0:
        print("  ✓ GOOD: Weights are very different from random")
        return True
    else:
        print("  ✗ BAD: Weights are too similar to random")
        return False


def main():
    print("\n" + "="*70)
    print("PRE-TRAINED CHECKPOINT DIAGNOSTIC")
    print("="*70 + "\n")
    
    # Run all checks
    checkpoint = check_checkpoint()
    
    test1 = compare_weight_loading()
    test2 = test_encoder_output()
    test3 = check_simclr_vs_segmentation_loading()
    test4 = compare_ablation_script_loading()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print(f"Weight Loading:           {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Encoder Output:           {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Direct Encoder Loading:   {'✓ PASS' if test3 else '✗ FAIL'}")
    print(f"Ablation Script Method:   {'✓ PASS' if test4 else '✗ FAIL'}")
    
    if all([test1, test2, test3, test4]):
        print("\n✓ ALL TESTS PASSED - Checkpoint loading appears correct")
        print("\nThe problem may be elsewhere:")
        print("  1. Decoder initialization")
        print("  2. Data preprocessing")
        print("  3. Loss computation")
        print("  4. Evaluation metric")
    else:
        print("\n✗ SOME TESTS FAILED - Checkpoint loading has issues")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

"""
test_setup.py

Quick test to verify everything is working before full training
Fixed: Proper GPU handling

Author: Parvez
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Set GPU 7 before importing torch

import torch
import sys
sys.path.append('.')

from models.resnet3d import resnet3d_18, SimCLRModel
from augmentation.simclr_augmentations import SimCLRAugmentation
from dataset.simclr_dataset import SimCLRStrokeDataset
from torch.utils.data import DataLoader

print("="*70)
print("Testing SimCLR Setup")
print("="*70 + "\n")

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
print()

# Test 1: Model
print("1. Testing model...")
encoder = resnet3d_18(in_channels=1)
model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
model = model.cuda()  # Use .cuda() without device ID

x = torch.randn(2, 1, 96, 128, 128).cuda()
h, z = model(x)
print(f"   ✓ Input: {x.shape}")
print(f"   ✓ Features: {h.shape}")
print(f"   ✓ Projection: {z.shape}")

# Test 2: Augmentation
print("\n2. Testing augmentation...")
import numpy as np
augmenter = SimCLRAugmentation(
    patch_size=(96, 128, 128),
    min_depth=64
)

volume = np.random.randn(120, 144, 144)
view1, view2 = augmenter(volume)
print(f"   ✓ Input volume: {volume.shape}")
print(f"   ✓ View 1: {view1.shape}")
print(f"   ✓ View 2: {view2.shape}")

# Test thin volume (like ISLES2018)
thin_volume = np.random.randn(10, 144, 144)
view1, view2 = augmenter(thin_volume)
print(f"   ✓ Thin volume: {thin_volume.shape} -> {view1.shape}")

# Test 3: Dataset
print("\n3. Testing dataset...")
dataset = SimCLRStrokeDataset(
    preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
    datasets=['ATLAS', 'UOA_Private'],
    split='train',
    splits_file='splits_stratified.json',
    augmentation=augmenter
)

print(f"   ✓ Dataset size: {len(dataset)}")

sample = dataset[0]
print(f"   ✓ View 1 shape: {sample['view1'].shape}")
print(f"   ✓ View 2 shape: {sample['view2'].shape}")
print(f"   ✓ Case ID: {sample['case_id']}")
print(f"   ✓ Dataset: {sample['dataset']}")

# Test 4: DataLoader
print("\n4. Testing dataloader...")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))

print(f"   ✓ Batch view 1: {batch['view1'].shape}")
print(f"   ✓ Batch view 2: {batch['view2'].shape}")

# Test 5: GPU forward pass
print("\n5. Testing GPU forward pass...")
view1 = batch['view1'].cuda()
view2 = batch['view2'].cuda()

with torch.no_grad():
    h1, z1 = model(view1)
    h2, z2 = model(view2)

print(f"   ✓ View 1 projection: {z1.shape}")
print(f"   ✓ View 2 projection: {z2.shape}")

# Test 6: Loss
print("\n6. Testing loss...")
from losses.simclr_loss import NTXentLoss

criterion = NTXentLoss(temperature=0.5).cuda()
loss = criterion(z1, z2)
print(f"   ✓ Loss: {loss.item():.4f}")

print("\n" + "="*70)
print("All tests passed! Ready to train.")
print("="*70 + "\n")

print("Next steps:")
print("  1. Review config: cat config_simclr_foundation.yaml")
print("  2. Run training: bash launch_simclr_training.sh")
print("  3. Monitor GPU: watch -n 1 nvidia-smi")

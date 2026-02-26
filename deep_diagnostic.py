"""
Deep Diagnostic: Why Isn't Pre-training Working?

This script performs systematic debugging to find the issue.

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

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
from torch import einsum
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
import nibabel as nib


sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters
from torch.utils.data import DataLoader

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
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
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
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        return seg_logits


def test_1_batchnorm_mode():
    """Test 1: Check if BatchNorm is in correct mode"""
    
    print("\n" + "="*70)
    print("TEST 1: BatchNorm Mode Check")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = simclr_model.encoder
    
    # Reset BN stats
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
    
    model = SegmentationModel(encoder, num_classes=2)
    
    # Check train mode
    model.train()
    print("Model set to TRAIN mode:")
    bn_count = 0
    bn_train = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            bn_count += 1
            if module.training:
                bn_train += 1
            else:
                print(f"  ⚠️  {name}: training={module.training} (SHOULD BE TRUE!)")
    
    print(f"\nBatchNorm summary: {bn_train}/{bn_count} in training mode")
    
    if bn_train == bn_count:
        print("✓ PASS: All BatchNorm layers in training mode")
        return True
    else:
        print("✗ FAIL: Some BatchNorm layers in eval mode!")
        return False


def test_2_gradient_flow():
    """Test 2: Check if gradients flow through encoder"""
    
    print("\n" + "="*70)
    print("TEST 2: Gradient Flow Check")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = simclr_model.encoder
    
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
    
    model = SegmentationModel(encoder, num_classes=2)
    model.train()
    
    # Create dummy input
    dummy_input = torch.randn(2, 1, 96, 96, 96)
    dummy_target = torch.randint(0, 2, (2, 96, 96, 96)).long()
    
    # Forward pass
    output = model(dummy_input)
    
    # Resize target to match output if needed
    if output.shape[2:] != dummy_target.shape[1:]:
        import torch.nn.functional as F
        dummy_target_resized = F.interpolate(
            dummy_target.unsqueeze(1).float(),
            size=output.shape[2:],
            mode='nearest'
        ).squeeze(1).long()
    else:
        dummy_target_resized = dummy_target
    
    # Compute loss
    loss = nn.CrossEntropyLoss()(output, dummy_target_resized)
    
    # Backward
    loss.backward()
    
    # Check gradients
    print("Checking encoder gradients:")
    encoder_layers = ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 
                     'layer3.0.conv1', 'layer4.0.conv1']
    
    all_have_grads = True
    for layer_name in encoder_layers:
        # Navigate to layer
        parts = layer_name.split('.')
        layer = model.encoder
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            grad_norm = layer.weight.grad.norm().item()
            print(f"  encoder.{layer_name}: grad_norm={grad_norm:.6f}")
            if grad_norm < 1e-6:
                print(f"    ⚠️  WARNING: Very small gradient!")
        else:
            print(f"  encoder.{layer_name}: NO GRADIENT!")
            all_have_grads = False
    
    print("\nChecking decoder gradients:")
    decoder_has_grads = False
    for name, param in model.decoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-6:
                decoder_has_grads = True
                print(f"  decoder.{name}: grad_norm={grad_norm:.6f}")
                break
    
    if all_have_grads and decoder_has_grads:
        print("\n✓ PASS: Gradients flow through encoder and decoder")
        return True
    else:
        print("\n✗ FAIL: Missing gradients!")
        return False


def test_3_encoder_updates():
    """Test 3: Verify encoder weights actually update during training"""
    
    print("\n" + "="*70)
    print("TEST 3: Encoder Weight Update Check")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = simclr_model.encoder
    
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
    
    model = SegmentationModel(encoder, num_classes=2)
    model.train()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Save initial weights
    initial_weights = {}
    for name, param in model.encoder.named_parameters():
        initial_weights[name] = param.data.clone()
    
    # Load data
    dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=0,
        patch_size=(96, 96, 96),
        patches_per_volume=10,
        augment=True,
        lesion_focus_ratio=0.7
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Train for 5 steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = GDiceLoss(apply_nonlin=None, smooth=1e-5)
    
    print("Training for 5 steps...")
    for step, batch in enumerate(tqdm(loader, total=5)):
        if step >= 5:
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        output = model(images)
        
        # Resize masks if needed
        if output.shape[2:] != masks.shape[1:]:
            import torch.nn.functional as F
            masks_resized = F.interpolate(
                masks.unsqueeze(1).float(),
                size=output.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            masks_resized = masks
        
        loss = criterion(torch.softmax(output, dim=1), masks_resized)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check weight changes
    print("\nWeight changes after 5 steps:")
    
    encoder_updated = False
    for name, param in model.encoder.named_parameters():
        if 'weight' in name and 'bn' not in name:  # Check conv weights only
            initial = initial_weights[name]
            current = param.data
            
            diff = torch.norm(current - initial).item()
            relative_change = diff / torch.norm(initial).item()
            
            print(f"  {name}:")
            print(f"    Absolute change: {diff:.6f}")
            print(f"    Relative change: {relative_change:.6f}")
            
            if relative_change > 1e-4:
                encoder_updated = True
                print(f"    ✓ UPDATED")
            else:
                print(f"    ⚠️  NOT UPDATED (or very small change)")
    
    if encoder_updated:
        print("\n✓ PASS: Encoder weights are updating")
        return True
    else:
        print("\n✗ FAIL: Encoder weights NOT updating!")
        return False


def test_4_feature_quality():
    """Test 4: Compare feature quality between random and pretrained"""
    
    print("\n" + "="*70)
    print("TEST 4: Feature Quality Comparison")
    print("="*70 + "\n")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder_pretrained = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder_pretrained, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    encoder_pretrained = simclr_model.encoder
    encoder_pretrained = encoder_pretrained.to(device)
    encoder_pretrained.eval()
    
    # Create random
    encoder_random = resnet3d_18(in_channels=1).to(device)
    encoder_random.eval()
    
    # Load some data
    dataset = PatchDatasetWithCenters(
        preprocessed_dir='/home/pahm409/preprocessed_stroke_foundation',
        datasets=['ATLAS', 'UOA_Private'],
        split='train',
        splits_file='splits_5fold.json',
        fold=0,
        patch_size=(96, 96, 96),
        patches_per_volume=5,
        augment=False,
        lesion_focus_ratio=0.7
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Extract features
    print("Extracting features from 20 samples...")
    
    features_pretrained = []
    features_random = []
    has_lesion = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=5)):
            if i >= 5:
                break
            
            images = batch['image'].to(device)
            masks = batch['mask']
            
            # Get features (use layer4 output)
            feat_pre = encoder_pretrained.layer4(
                encoder_pretrained.layer3(
                    encoder_pretrained.layer2(
                        encoder_pretrained.layer1(
                            encoder_pretrained.maxpool(
                                torch.relu(encoder_pretrained.bn1(encoder_pretrained.conv1(images)))
                            )
                        )
                    )
                )
            )
            
            feat_rand = encoder_random.layer4(
                encoder_random.layer3(
                    encoder_random.layer2(
                        encoder_random.layer1(
                            encoder_random.maxpool(
                                torch.relu(encoder_random.bn1(encoder_random.conv1(images)))
                            )
                        )
                    )
                )
            )
            
            # Global average pooling
            feat_pre = feat_pre.mean(dim=[2, 3, 4])  # (B, 512)
            feat_rand = feat_rand.mean(dim=[2, 3, 4])
            
            features_pretrained.append(feat_pre.cpu())
            features_random.append(feat_rand.cpu())
            has_lesion.append((masks.sum(dim=[1,2,3]) > 0).float())
    
    features_pretrained = torch.cat(features_pretrained, dim=0).numpy()
    features_random = torch.cat(features_random, dim=0).numpy()
    has_lesion = torch.cat(has_lesion, dim=0).numpy()
    
    # Analyze
    print(f"\nFeature analysis ({len(features_pretrained)} samples):")
    print(f"  Samples with lesions: {has_lesion.sum():.0f}")
    print(f"  Samples without lesions: {(1-has_lesion).sum():.0f}")
    
    # Feature variance
    var_pre = features_pretrained.var(axis=0).mean()
    var_rand = features_random.var(axis=0).mean()
    
    print(f"\nFeature variance (across samples):")
    print(f"  Pretrained: {var_pre:.6f}")
    print(f"  Random:     {var_rand:.6f}")
    
    # Feature norm
    norm_pre = np.linalg.norm(features_pretrained, axis=1).mean()
    norm_rand = np.linalg.norm(features_random, axis=1).mean()
    
    print(f"\nFeature L2 norm (average):")
    print(f"  Pretrained: {norm_pre:.6f}")
    print(f"  Random:     {norm_rand:.6f}")
    
    # Discriminability (if we have both lesion and non-lesion samples)
    if has_lesion.sum() > 0 and (1-has_lesion).sum() > 0:
        lesion_idx = has_lesion == 1
        no_lesion_idx = has_lesion == 0
        
        # Mean features for each class
        feat_pre_lesion = features_pretrained[lesion_idx].mean(axis=0)
        feat_pre_nolesion = features_pretrained[no_lesion_idx].mean(axis=0)
        
        feat_rand_lesion = features_random[lesion_idx].mean(axis=0)
        feat_rand_nolesion = features_random[no_lesion_idx].mean(axis=0)
        
        # Distance between classes
        dist_pre = np.linalg.norm(feat_pre_lesion - feat_pre_nolesion)
        dist_rand = np.linalg.norm(feat_rand_lesion - feat_rand_nolesion)
        
        print(f"\nClass separability (lesion vs no-lesion):")
        print(f"  Pretrained: {dist_pre:.6f}")
        print(f"  Random:     {dist_rand:.6f}")
        
        if dist_pre > dist_rand * 1.1:
            print("  ✓ Pretrained features MORE discriminative")
        elif dist_rand > dist_pre * 1.1:
            print("  ⚠️  Random features MORE discriminative!")
        else:
            print("  ~ Similar discriminability")
    
    print("\n✓ Feature quality test complete")
    return True


def test_5_batchnorm_statistics():
    """Test 5: Check BatchNorm running statistics"""
    
    print("\n" + "="*70)
    print("TEST 5: BatchNorm Statistics Check")
    print("="*70 + "\n")
    
    checkpoint_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = resnet3d_18(in_channels=1)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = simclr_model.encoder
    
    print("BatchNorm statistics BEFORE reset:")
    bn_count = 0
    for name, module in encoder.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            bn_count += 1
            if bn_count <= 3:  # Show first 3
                print(f"  {name}:")
                print(f"    running_mean: {module.running_mean[:5]}")
                print(f"    running_var:  {module.running_var[:5]}")
    
    # Reset
    print("\nResetting BatchNorm statistics...")
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
    
    print("\nBatchNorm statistics AFTER reset:")
    bn_count = 0
    for name, module in encoder.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            bn_count += 1
            if bn_count <= 3:
                print(f"  {name}:")
                print(f"    running_mean: {module.running_mean[:5]}")
                print(f"    running_var:  {module.running_var[:5]}")
    
    print("\n✓ BatchNorm reset complete")
    return True


def main():
    print("\n" + "="*70)
    print("DEEP DIAGNOSTIC: WHY ISN'T PRE-TRAINING WORKING?")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['test1_batchnorm_mode'] = test_1_batchnorm_mode()
    results['test2_gradient_flow'] = test_2_gradient_flow()
    results['test3_encoder_updates'] = test_3_encoder_updates()
    results['test4_feature_quality'] = test_4_feature_quality()
    results['test5_batchnorm_stats'] = test_5_batchnorm_statistics()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n✓ ALL TESTS PASSED")
        print("\nConclusion: No obvious bugs found.")
        print("Pre-training may genuinely not help for this specific task/setup.")
        print("\nNext steps:")
        print("  1. Try different learning rates (encoder vs decoder)")
        print("  2. Try full-volume fine-tuning")
        print("  3. Run on multiple folds to confirm")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nConclusion: Found issues that need fixing!")
        print("Fix the failed tests and re-run training.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

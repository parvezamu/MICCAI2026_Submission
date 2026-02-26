"""
test_single_neuralcup_case_v2.py

Test evaluation on a single NEURALCUP case with LESION-CENTERED patches
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from finetune_on_isles_DEBUG import SegmentationModel, reconstruct_from_patches_with_count
from torch.cuda.amp import autocast


def extract_patches_COVERING_LESION(volume, mask, patch_size=(96, 96, 96), patches_per_volume=100):
    """
    Extract patches that actually cover the lesion area
    50% random patches + 50% lesion-centered patches
    """
    import random
    
    volume_shape = np.array(volume.shape)
    patch_size = np.array(patch_size)
    half_patch = patch_size // 2
    
    print(f"\n📦 Patch Extraction:")
    print(f"   Volume shape: {volume_shape}")
    print(f"   Target patch size: {patch_size}")
    
    patches = []
    centers = []
    
    valid_min = half_patch
    valid_max = volume_shape - half_patch
    
    print(f"   Valid center range: [{valid_min}, {valid_max}]")
    
    # Get lesion coordinates
    lesion_coords = np.where(mask > 0)
    has_lesion = len(lesion_coords[0]) > 0
    
    if has_lesion:
        print(f"   Lesion found: {len(lesion_coords[0])} voxels")
        lesion_center = np.array([
            lesion_coords[0].mean(),
            lesion_coords[1].mean(),
            lesion_coords[2].mean()
        ])
        print(f"   Lesion center: ({lesion_center[0]:.0f}, {lesion_center[1]:.0f}, {lesion_center[2]:.0f})")
    
    n_random = patches_per_volume // 2
    n_lesion = patches_per_volume - n_random
    
    # Extract random patches
    for i in range(n_random):
        center = np.array([
            random.randint(valid_min[0], max(valid_min[0], valid_max[0]-1)),
            random.randint(valid_min[1], max(valid_min[1], valid_max[1]-1)),
            random.randint(valid_min[2], max(valid_min[2], valid_max[2]-1))
        ])
        
        lower = center - half_patch
        upper = center + half_patch
        
        patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        
        if patch.shape != tuple(patch_size):
            print(f"   ❌ Random patch {i}: shape {patch.shape} != {patch_size}")
            raise ValueError(f"Invalid patch shape!")
        
        patches.append(patch)
        centers.append(center)
    
    # Extract lesion-centered patches
    if has_lesion:
        for i in range(n_lesion):
            # Sample a random lesion voxel
            idx = random.randint(0, len(lesion_coords[0])-1)
            lesion_voxel = np.array([
                lesion_coords[0][idx],
                lesion_coords[1][idx],
                lesion_coords[2][idx]
            ])
            
            # Center patch on this lesion voxel (with some jitter)
            jitter = np.random.randint(-10, 11, size=3)
            center = lesion_voxel + jitter
            
            # Ensure center is valid
            center = np.clip(center, valid_min, valid_max - 1)
            
            lower = center - half_patch
            upper = center + half_patch
            
            patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
            
            if patch.shape != tuple(patch_size):
                print(f"   ❌ Lesion patch {i}: shape {patch.shape} != {patch_size}")
                print(f"      Center: {center}, Lower: {lower}, Upper: {upper}")
                raise ValueError(f"Invalid patch shape!")
            
            patches.append(patch)
            centers.append(center)
        
        print(f"   ✓ Extracted {n_random} random + {n_lesion} lesion-centered patches")
    else:
        # No lesion, just do more random patches
        for i in range(n_lesion):
            center = np.array([
                random.randint(valid_min[0], max(valid_min[0], valid_max[0]-1)),
                random.randint(valid_min[1], max(valid_min[1], valid_max[1]-1)),
                random.randint(valid_min[2], max(valid_min[2], valid_max[2]-1))
            ])
            
            lower = center - half_patch
            upper = center + half_patch
            
            patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
            
            if patch.shape != tuple(patch_size):
                print(f"   ❌ Extra random patch {i}: shape {patch.shape} != {patch_size}")
                raise ValueError(f"Invalid patch shape!")
            
            patches.append(patch)
            centers.append(center)
        
        print(f"   ✓ Extracted {patches_per_volume} random patches (no lesion)")
    
    print(f"   ✓ Total patches: {len(patches)}, all {patch_size}")
    
    return np.array(patches), np.array(centers)


def evaluate_single_case(checkpoint_path, neuralcup_dir, case_id):
    """Evaluate a single NEURALCUP case"""
    
    device = torch.device('cuda:0')
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {case_id}")
    print(f"{'='*70}")
    
    # Load model
    print(f"\n1️⃣ Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder, 
        num_classes=2,
        attention_type=attention_type,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Model loaded (attention={attention_type}, deep_supervision={deep_supervision})")
    
    # Load data
    print(f"\n2️⃣ Loading NEURALCUP data...")
    npz_file = Path(neuralcup_dir) / f"{case_id}.npz"
    data = np.load(npz_file)
    
    volume = data['volume']
    mask_gt = data['mask']
    
    print(f"   Volume: {volume.shape}, range=[{volume.min():.3f}, {volume.max():.3f}]")
    print(f"   Mask: {mask_gt.shape}, lesion voxels={mask_gt.sum()}")
    
    # Check lesion location
    lesion_coords = np.where(mask_gt > 0)
    if len(lesion_coords[0]) > 0:
        print(f"   Lesion bounding box:")
        print(f"     X: [{lesion_coords[0].min()}, {lesion_coords[0].max()}]")
        print(f"     Y: [{lesion_coords[1].min()}, {lesion_coords[1].max()}]")
        print(f"     Z: [{lesion_coords[2].min()}, {lesion_coords[2].max()}]")
    
    # Extract patches with LESION-CENTERED function
    print(f"\n3️⃣ Extracting patches...")
    patches, centers = extract_patches_COVERING_LESION(
        volume, 
        mask_gt,  # Pass the mask!
        patch_size=(96, 96, 96), 
        patches_per_volume=100
    )
    
    # Add channel dimension
    patches = patches[:, np.newaxis, :, :, :]
    print(f"   Final patches shape: {patches.shape}")
    
    # Predict
    print(f"\n4️⃣ Running predictions...")
    all_preds = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[i:i+batch_size]).float().to(device)
            
            with autocast():
                outputs = model(batch)
                if deep_supervision:
                    outputs = outputs[0]
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
    
    all_preds = np.concatenate(all_preds, axis=0)
    print(f"   ✓ Predictions shape: {all_preds.shape}")
    
    # Check prediction statistics
    lesion_probs = all_preds[:, 1, :, :, :]
    print(f"   Lesion probability stats:")
    print(f"     Mean: {lesion_probs.mean():.6f}")
    print(f"     Max: {lesion_probs.max():.6f}")
    print(f"     Min: {lesion_probs.min():.6f}")
    print(f"     Voxels > 0.5: {(lesion_probs > 0.5).sum():,}")
    print(f"     Patches with any voxel > 0.5: {np.any(lesion_probs > 0.5, axis=(1,2,3)).sum()}")
    
    # Reconstruct
    print(f"\n5️⃣ Reconstructing volume...")
    reconstructed, count_map = reconstruct_from_patches_with_count(
        all_preds, centers, volume.shape, patch_size=(96, 96, 96)
    )
    
    print(f"   Reconstruction stats:")
    print(f"     Shape: {reconstructed.shape}")
    print(f"     Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    print(f"     Mean: {reconstructed.mean():.6f}")
    print(f"     Voxels > 0.5: {(reconstructed > 0.5).sum():,}")
    
    pred_binary = (reconstructed > 0.5).astype(np.uint8)
    mask_gt_binary = (mask_gt > 0).astype(np.uint8)
    
    # Check if predictions overlap with lesion
    if len(lesion_coords[0]) > 0:
        pred_at_lesion = reconstructed[lesion_coords]
        print(f"   Prediction at lesion location:")
        print(f"     Mean prob: {pred_at_lesion.mean():.6f}")
        print(f"     Max prob: {pred_at_lesion.max():.6f}")
        print(f"     Voxels > 0.5: {(pred_at_lesion > 0.5).sum()}")
    
    # Compute DSC
    print(f"\n6️⃣ Computing DSC...")
    intersection = (pred_binary * mask_gt_binary).sum()
    union = pred_binary.sum() + mask_gt_binary.sum()
    
    if union > 0:
        dsc = (2.0 * intersection) / union
    else:
        dsc = 1.0 if pred_binary.sum() == 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Ground truth lesion voxels: {mask_gt_binary.sum():,}")
    print(f"Predicted lesion voxels:    {pred_binary.sum():,}")
    print(f"Intersection:               {intersection:,}")
    print(f"DSC:                        {dsc:.4f} ({dsc*100:.2f}%)")
    print(f"{'='*70}")
    
    if dsc < 0.01:
        print(f"\n❌ Near-zero DSC!")
        if intersection == 0:
            print(f"   Zero intersection - predictions don't overlap with lesion")
        print(f"   Max reconstruction prob: {reconstructed.max():.6f}")
    else:
        print(f"\n✅ DSC = {dsc*100:.2f}%")
    
    return dsc


if __name__ == '__main__':
    
    checkpoint_path = '/home/pahm409/ablation_ds_main_only/Random_Init/none/fold_0/run_2/exp_20260131_183019/checkpoints/best_model.pth'
    neuralcup_dir = '/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled'
    case_id = 'wmBBS001'
    
    dsc = evaluate_single_case(checkpoint_path, neuralcup_dir, case_id)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: DSC = {dsc:.4f} ({dsc*100:.2f}%)")
    print(f"{'='*70}")

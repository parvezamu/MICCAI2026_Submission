"""
debug_arc_predictions.py

Debug why ARC evaluation gives 0% DSC
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/pahm409')
sys.path.append('/home/pahm409/ISLES2029')
sys.path.append('/home/pahm409/ISLES2029/dataset')

from models.resnet3d import resnet3d_18
from finetune_on_isles_DEBUG import SegmentationModel
from torch.cuda.amp import autocast


def load_model(checkpoint_path, device):
    """Load model"""
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
    
    return model, attention_type, deep_supervision


def debug_single_case(checkpoint_path, arc_preprocessed_dir, case_id):
    """Debug a single ARC case"""
    
    device = torch.device('cuda:0')
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING CASE: {case_id}")
    print(f"{'='*70}\n")
    
    # Load model
    print("1. Loading model...")
    model, attention_type, deep_supervision = load_model(checkpoint_path, device)
    print(f"   Attention: {attention_type}")
    print(f"   Deep supervision: {deep_supervision}")
    
    # Load ARC volume
    print(f"\n2. Loading ARC volume...")
    npz_file = Path(arc_preprocessed_dir) / f"{case_id}.npz"
    data = np.load(npz_file)
    volume = data['volume']
    mask = data['mask']
    
    print(f"   Volume shape: {volume.shape}")
    print(f"   Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"   Volume mean: {volume.mean():.3f}, std: {volume.std():.3f}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Lesion voxels: {(mask > 0).sum()}")
    print(f"   Lesion ratio: {(mask > 0).sum() / mask.size * 100:.2f}%")
    
    # Extract a patch where there's a lesion
    print(f"\n3. Extracting patch with lesion...")
    lesion_coords = np.where(mask > 0)
    if len(lesion_coords[0]) == 0:
        print("   ❌ No lesion in this case!")
        return
    
    # Get center of lesion
    center = np.array([
        lesion_coords[0][len(lesion_coords[0])//2],
        lesion_coords[1][len(lesion_coords[1])//2],
        lesion_coords[2][len(lesion_coords[2])//2]
    ])
    
    half_patch = np.array([48, 48, 48])
    lower = center - half_patch
    upper = center + half_patch
    
    # Ensure bounds
    lower = np.maximum(lower, [0, 0, 0])
    upper = np.minimum(upper, volume.shape)
    
    patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    mask_patch = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    
    print(f"   Patch shape: {patch.shape}")
    print(f"   Patch range: [{patch.min():.3f}, {patch.max():.3f}]")
    print(f"   Patch mean: {patch.mean():.3f}, std: {patch.std():.3f}")
    print(f"   Lesion in patch: {(mask_patch > 0).sum()} voxels")
    
    # Predict on this patch
    print(f"\n4. Running prediction...")
    
    # Add batch and channel dimensions
    patch_input = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
    print(f"   Input tensor shape: {patch_input.shape}")
    print(f"   Input tensor range: [{patch_input.min():.3f}, {patch_input.max():.3f}]")
    
    with torch.no_grad():
        with autocast():
            output = model(patch_input)
            if deep_supervision:
                output = output[0]
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output range (before softmax): [{output.min():.3f}, {output.max():.3f}]")
    
    # Apply softmax
    pred_probs = torch.softmax(output, dim=1)
    print(f"   Class 0 (background) prob range: [{pred_probs[0, 0].min():.3f}, {pred_probs[0, 0].max():.3f}]")
    print(f"   Class 1 (lesion) prob range: [{pred_probs[0, 1].min():.3f}, {pred_probs[0, 1].max():.3f}]")
    print(f"   Class 1 mean prob: {pred_probs[0, 1].mean():.6f}")
    print(f"   Class 1 max prob: {pred_probs[0, 1].max():.6f}")
    
    # Check if any voxel predicts lesion
    pred_binary = (pred_probs[0, 1] > 0.5).cpu().numpy()
    print(f"\n5. Binary prediction (threshold=0.5):")
    print(f"   Predicted lesion voxels: {pred_binary.sum()}")
    print(f"   Ground truth lesion voxels: {(mask_patch > 0).sum()}")
    
    if pred_binary.sum() == 0:
        print(f"\n   ⚠️  MODEL PREDICTS NO LESION!")
        print(f"   Max lesion probability: {pred_probs[0, 1].max():.6f}")
        print(f"   This suggests the model is confidently predicting background everywhere.")
    
    # Check if model is outputting reasonable values
    print(f"\n6. Model sanity checks:")
    
    # Check if logits are reasonable
    logits_class0 = output[0, 0].cpu().numpy()
    logits_class1 = output[0, 1].cpu().numpy()
    
    print(f"   Logits class 0 mean: {logits_class0.mean():.3f}")
    print(f"   Logits class 1 mean: {logits_class1.mean():.3f}")
    print(f"   Logit difference (class1 - class0) mean: {(logits_class1 - logits_class0).mean():.3f}")
    
    if logits_class1.mean() < logits_class0.mean():
        print(f"\n   ⚠️  Model strongly prefers class 0 (background)")
        print(f"   This explains the 0% DSC!")
    
    # Visualize
    print(f"\n7. Saving visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    mid_slice = patch.shape[2] // 2
    
    # Row 1: Image, GT, Prediction
    axes[0, 0].imshow(patch[:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title(f'Image (slice {mid_slice})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_patch[:, :, mid_slice], cmap='Reds', alpha=0.5)
    axes[0, 1].imshow(patch[:, :, mid_slice], cmap='gray', alpha=0.5)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    pred_prob_slice = pred_probs[0, 1, :, :, mid_slice].cpu().numpy()
    im = axes[0, 2].imshow(pred_prob_slice, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Prediction Probability\n(max={pred_prob_slice.max():.4f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Row 2: Histograms and stats
    axes[1, 0].hist(patch.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Image Intensity Distribution')
    axes[1, 0].set_xlabel('Intensity')
    
    axes[1, 1].hist(logits_class0.flatten(), bins=50, alpha=0.7, label='Class 0')
    axes[1, 1].hist(logits_class1.flatten(), bins=50, alpha=0.7, label='Class 1')
    axes[1, 1].set_title('Logits Distribution')
    axes[1, 1].set_xlabel('Logit value')
    axes[1, 1].legend()
    
    axes[1, 2].hist(pred_probs[0, 1].cpu().numpy().flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('Lesion Probability Distribution')
    axes[1, 2].set_xlabel('Probability')
    axes[1, 2].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'debug_{case_id}.png', dpi=150, bbox_inches='tight')
    print(f"   Saved to: debug_{case_id}.png")
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS:")
    print(f"{'='*70}")
    
    if pred_probs[0, 1].max() < 0.1:
        print(f"❌ PROBLEM: Model outputs very low lesion probabilities (max={pred_probs[0, 1].max():.6f})")
        print(f"\nPossible causes:")
        print(f"  1. Domain shift: Model trained on different modality (T1 vs T2)")
        print(f"  2. Intensity normalization mismatch")
        print(f"  3. Model checkpoint is from wrong training stage")
        print(f"  4. Model was never trained on T2w data")
        print(f"\nSuggested fixes:")
        print(f"  - Use a model trained on T2w data")
        print(f"  - Fine-tune this model on T2w data first")
        print(f"  - Check if ARC preprocessing matches training data preprocessing")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--arc-dir', type=str, 
                       default='/home/pahm409/preprocessed_stroke_foundation/ARC_T2w_resampled')
    parser.add_argument('--case-id', type=str, default='sub-r001s001')
    
    args = parser.parse_args()
    
    debug_single_case(args.checkpoint, args.arc_dir, args.case_id)

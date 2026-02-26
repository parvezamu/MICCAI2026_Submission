"""
debug_evaluation_pipeline.py

Check EVERY step of the evaluation to find where it breaks
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
from finetune_on_isles_DEBUG import SegmentationModel
from torch.cuda.amp import autocast


def debug_single_case(checkpoint_path, neuralcup_dir, case_id):
    """Debug evaluation step-by-step"""
    
    device = torch.device('cuda:0')
    
    print(f"\n{'='*70}")
    print(f"STEP 1: LOAD MODEL")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    print(f"Attention: {attention_type}")
    print(f"Deep supervision: {deep_supervision}")
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder, 
        num_classes=2,
        attention_type=attention_type,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded")
    
    print(f"\n{'='*70}")
    print(f"STEP 2: LOAD NEURALCUP DATA")
    print(f"{'='*70}")
    
    npz_file = Path(neuralcup_dir) / f"{case_id}.npz"
    data = np.load(npz_file)
    
    volume = data['volume']
    mask = data['mask']
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"Volume mean: {volume.mean():.6f}, std: {volume.std():.6f}")
    print(f"Mask shape: {mask.shape}")
    print(f"Lesion voxels: {(mask > 0).sum()}")
    
    print(f"\n{'='*70}")
    print(f"STEP 3: EXTRACT A SINGLE PATCH")
    print(f"{'='*70}")
    
    # Extract center patch with lesion
    lesion_coords = np.where(mask > 0)
    center = np.array([
        lesion_coords[0][len(lesion_coords[0])//2],
        lesion_coords[1][len(lesion_coords[1])//2],
        lesion_coords[2][len(lesion_coords[2])//2]
    ])
    
    half_patch = np.array([48, 48, 48])
    lower = np.maximum(center - half_patch, [0, 0, 0])
    upper = np.minimum(center + half_patch, volume.shape)
    
    patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    mask_patch = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    
    print(f"Patch shape: {patch.shape}")
    print(f"Patch range: [{patch.min():.3f}, {patch.max():.3f}]")
    print(f"Patch mean: {patch.mean():.6f}, std: {patch.std():.6f}")
    print(f"Lesion in patch: {(mask_patch > 0).sum()} voxels")
    
    print(f"\n{'='*70}")
    print(f"STEP 4: PREPARE INPUT TENSOR")
    print(f"{'='*70}")
    
    # Add batch and channel dims
    patch_input = torch.from_numpy(patch).float()
    print(f"After torch.from_numpy: shape={patch_input.shape}, dtype={patch_input.dtype}")
    
    patch_input = patch_input.unsqueeze(0).unsqueeze(0)
    print(f"After unsqueeze: shape={patch_input.shape}")
    
    patch_input = patch_input.to(device)
    print(f"After .to(device): device={patch_input.device}")
    
    print(f"Input tensor stats:")
    print(f"  Range: [{patch_input.min():.3f}, {patch_input.max():.3f}]")
    print(f"  Mean: {patch_input.mean():.6f}")
    print(f"  Std: {patch_input.std():.6f}")
    
    print(f"\n{'='*70}")
    print(f"STEP 5: RUN MODEL")
    print(f"{'='*70}")
    
    with torch.no_grad():
        with autocast():
            output = model(patch_input)
            
            if deep_supervision:
                print(f"Deep supervision output type: {type(output)}")
                if isinstance(output, (list, tuple)):
                    print(f"Number of outputs: {len(output)}")
                    for i, o in enumerate(output):
                        print(f"  Output {i} shape: {o.shape}")
                    output = output[0]
                else:
                    print(f"Output shape: {output.shape}")
    
    print(f"Final output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print(f"\n{'='*70}")
    print(f"STEP 6: SOFTMAX AND PREDICTION")
    print(f"{'='*70}")
    
    pred_probs = torch.softmax(output, dim=1)
    print(f"After softmax shape: {pred_probs.shape}")
    print(f"Class 0 prob range: [{pred_probs[0, 0].min():.6f}, {pred_probs[0, 0].max():.6f}]")
    print(f"Class 1 prob range: [{pred_probs[0, 1].min():.6f}, {pred_probs[0, 1].max():.6f}]")
    print(f"Class 1 mean prob: {pred_probs[0, 1].mean():.6f}")
    print(f"Class 1 max prob: {pred_probs[0, 1].max():.6f}")
    
    pred_binary = (pred_probs[0, 1] > 0.5).cpu().numpy()
    print(f"\nBinary prediction (threshold=0.5):")
    print(f"  Predicted lesion voxels: {pred_binary.sum()}")
    print(f"  Ground truth lesion voxels: {(mask_patch > 0).sum()}")
    
    if pred_binary.sum() == 0:
        print(f"\n❌ MODEL PREDICTS NO LESION!")
        print(f"   Max lesion probability: {pred_probs[0, 1].max():.6f}")
        
        # Check logits
        logits_class0 = output[0, 0].cpu().numpy()
        logits_class1 = output[0, 1].cpu().numpy()
        
        print(f"\nLogits analysis:")
        print(f"  Class 0 mean: {logits_class0.mean():.3f}")
        print(f"  Class 1 mean: {logits_class1.mean():.3f}")
        print(f"  Difference (class1 - class0): {(logits_class1 - logits_class0).mean():.3f}")
        
        if logits_class1.mean() < logits_class0.mean():
            print(f"\n🔍 ROOT CAUSE: Model strongly prefers background")
            print(f"   This means the model learned wrong features")
    else:
        print(f"\n✓ Model predicts some lesion")
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS")
    print(f"{'='*70}")
    
    # Compare with ATLAS training range
    print(f"\nExpected (from ATLAS training):")
    print(f"  Mean: ~0.0, Std: ~1.0")
    print(f"  Range: ~[-0.76, 3.74]")
    
    print(f"\nActual (NEURALCUP):")
    print(f"  Mean: {patch.mean():.6f}, Std: {patch.std():.6f}")
    print(f"  Range: [{patch.min():.3f}, {patch.max():.3f}]")
    
    if abs(patch.mean()) > 0.5 or abs(patch.std() - 1.0) > 0.5:
        print(f"\n⚠️  INPUT DISTRIBUTION MISMATCH!")
    else:
        print(f"\n✓ Input distribution looks correct")


if __name__ == '__main__':
    
    checkpoint_path = '/home/pahm409/ablation_ds_main_only/Random_Init/none/fold_0/run_2/exp_20260131_183019/checkpoints/best_model.pth'
    neuralcup_dir = '/home/pahm409/preprocessed_stroke_foundation/NEURALCUP_T1_resampled'
    case_id = 'wmBBS001'
    
    debug_single_case(checkpoint_path, neuralcup_dir, case_id)

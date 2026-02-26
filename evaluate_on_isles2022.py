"""
evaluate_on_isles2022.py

Test your trained model on ISLES 2022 external validation set
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import your model architecture
import sys
sys.path.append('/home/pahm409')
from train_segmentation_corrected_DS_MAIN_ONLY import SegmentationModel
from models.resnet3d import resnet3d_18


def load_trained_model(checkpoint_path, device):
    """Load your trained model from checkpoint"""
    
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Rebuild model architecture
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder,
        num_classes=2,
        attention_type='mkdc',  # or 'none' depending on your model
        deep_supervision=True   # Set based on your checkpoint
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  Training DSC: {checkpoint.get('val_dsc_recon', 'N/A')}")
    
    return model


def extract_patches_with_centers(volume, patch_size=(96, 96, 96), stride=48):
    """
    Extract overlapping patches with their center coordinates
    Matches your training patch extraction strategy
    """
    patches = []
    centers = []
    
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    
    # Calculate number of patches needed
    d_steps = max(1, (D - pD) // stride + 1)
    h_steps = max(1, (H - pH) // stride + 1)
    w_steps = max(1, (W - pW) // stride + 1)
    
    for d in range(d_steps):
        d_start = min(d * stride, D - pD)
        d_center = d_start + pD // 2
        
        for h in range(h_steps):
            h_start = min(h * stride, H - pH)
            h_center = h_start + pH // 2
            
            for w in range(w_steps):
                w_start = min(w * stride, W - pW)
                w_center = w_start + pW // 2
                
                patch = volume[d_start:d_start+pD, 
                              h_start:h_start+pH, 
                              w_start:w_start+pW]
                
                patches.append(patch)
                centers.append(np.array([d_center, h_center, w_center]))
    
    return np.array(patches), np.array(centers)


def reconstruct_from_patches(patches, centers, original_shape, patch_size=(96, 96, 96)):
    """
    Reconstruct full volume from overlapping patches
    Uses weighted averaging for overlapping regions
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    
    half_size = np.array(patch_size) // 2
    
    for patch, center in zip(patches, centers):
        # Calculate patch bounds
        lower = center - half_size
        upper = center + half_size
        
        # Ensure bounds are valid
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, original_shape)
        
        # Adjust patch if at boundaries
        patch_lower = half_size - (center - lower)
        patch_upper = patch_lower + (upper - lower)
        
        # Add patch to reconstruction
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += \
            patch[patch_lower[0]:patch_upper[0], 
                  patch_lower[1]:patch_upper[1], 
                  patch_lower[2]:patch_upper[2]]
        
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    # Average overlapping regions
    mask = count_map > 0
    reconstructed[mask] /= count_map[mask]
    
    return reconstructed


def compute_dice(pred, target):
    """Compute Dice Similarity Coefficient"""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice


def evaluate_one_case(model, npz_file, device, patch_size=(96, 96, 96), batch_size=8):
    """Evaluate model on one ISLES 2022 case"""
    
    # Load preprocessed data
    data = np.load(npz_file)
    volume = data['image']
    mask_gt = data['lesion_mask']
    case_id = str(data['case_id'])
    
    # Extract patches
    patches, centers = extract_patches_with_centers(volume, patch_size=patch_size, stride=48)
    
    # Predict on patches in batches
    patch_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            
            # Convert to tensor [B, 1, D, H, W]
            batch_tensor = torch.from_numpy(batch_patches).unsqueeze(1).float().to(device)
            
            # Forward pass
            outputs = model(batch_tensor)
            
            # Handle deep supervision output
            if isinstance(outputs, list):
                outputs = outputs[0]  # Use main output
            
            # Get probability maps
            probs = torch.softmax(outputs, dim=1)[:, 1, ...]  # Lesion class
            
            patch_predictions.append(probs.cpu().numpy())
    
    # Concatenate all predictions
    patch_predictions = np.concatenate(patch_predictions, axis=0)
    
    # Reconstruct full volume
    reconstructed = reconstruct_from_patches(patch_predictions, centers, 
                                            volume.shape, patch_size=patch_size)
    
    # Threshold
    pred_binary = (reconstructed > 0.5).astype(np.uint8)
    
    # Compute DSC
    dice = compute_dice(pred_binary, mask_gt)
    
    return {
        'case_id': case_id,
        'dice': float(dice),
        'lesion_volume_gt': int(mask_gt.sum()),
        'lesion_volume_pred': int(pred_binary.sum()),
        'prediction': reconstructed  # Keep for analysis
    }


def evaluate_all(model_checkpoint, isles_preprocessed_dir, output_dir, 
                patch_size=(96, 96, 96), batch_size=8):
    """Evaluate on all ISLES 2022 cases"""
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_trained_model(model_checkpoint, device)
    
    # Get all preprocessed cases
    npz_files = sorted(Path(isles_preprocessed_dir).glob("*.npz"))
    
    print(f"\nFound {len(npz_files)} ISLES 2022 cases to evaluate")
    print(f"Patch size: {patch_size}")
    print(f"Batch size: {batch_size}\n")
    
    results = []
    
    for npz_file in tqdm(npz_files, desc="Evaluating"):
        try:
            result = evaluate_one_case(model, npz_file, device, 
                                      patch_size=patch_size, batch_size=batch_size)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Failed on {npz_file.stem}: {str(e)}")
    
    # Compute statistics
    dices = [r['dice'] for r in results]
    
    stats = {
        'mean_dice': float(np.mean(dices)),
        'std_dice': float(np.std(dices)),
        'median_dice': float(np.median(dices)),
        'min_dice': float(np.min(dices)),
        'max_dice': float(np.max(dices)),
        'num_cases': len(results),
        'model_checkpoint': str(model_checkpoint),
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_path / 'evaluation_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("ISLES 2022 External Validation Results\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {model_checkpoint}\n")
        f.write(f"Evaluated: {stats['evaluation_date']}\n")
        f.write(f"Number of cases: {stats['num_cases']}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Mean DSC:   {stats['mean_dice']:.4f}\n")
        f.write(f"  Std DSC:    {stats['std_dice']:.4f}\n")
        f.write(f"  Median DSC: {stats['median_dice']:.4f}\n")
        f.write(f"  Min DSC:    {stats['min_dice']:.4f}\n")
        f.write(f"  Max DSC:    {stats['max_dice']:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("Per-Case Results:\n")
        f.write("="*70 + "\n")
        f.write("CaseID\t\t\tDSC\tGT_Volume\tPred_Volume\n")
        
        for r in sorted(results, key=lambda x: x['dice'], reverse=True):
            f.write(f"{r['case_id']}\t{r['dice']:.4f}\t"
                   f"{r['lesion_volume_gt']}\t{r['lesion_volume_pred']}\n")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Cases evaluated:  {stats['num_cases']}")
    print(f"Mean DSC:         {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}")
    print(f"Median DSC:       {stats['median_dice']:.4f}")
    print(f"Range:            [{stats['min_dice']:.4f}, {stats['max_dice']:.4f}]")
    print("="*70)
    print(f"\nResults saved: {output_path / 'evaluation_results.txt'}")
    
    return results, stats


if __name__ == '__main__':
    # Configuration
    MODEL_CHECKPOINT = "/home/pahm409/ablation_study_fold0/SimCLR_Pretrained/mkdc_ds/fold_0/exp_*/checkpoints/best_model.pth"
    ISLES_DIR = "/home/pahm409/isles2022_preprocessed"
    OUTPUT_DIR = "/home/pahm409/isles2022_evaluation_results"
    
    # Find actual checkpoint path (expand wildcard)
    import glob
    checkpoint_paths = glob.glob(MODEL_CHECKPOINT)
    
    if not checkpoint_paths:
        print(f"✗ No checkpoint found at: {MODEL_CHECKPOINT}")
        exit(1)
    
    checkpoint_path = checkpoint_paths[0]
    print(f"Using checkpoint: {checkpoint_path}\n")
    
    # Run evaluation
    results, stats = evaluate_all(
        model_checkpoint=checkpoint_path,
        isles_preprocessed_dir=ISLES_DIR,
        output_dir=OUTPUT_DIR,
        patch_size=(96, 96, 96),
        batch_size=8
    )

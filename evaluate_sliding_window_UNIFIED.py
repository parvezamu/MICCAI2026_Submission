#!/usr/bin/env python3
"""
evaluate_sliding_window_updated.py

Sliding window evaluation matching the corrected training code architecture
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from scipy.ndimage import zoom

sys.path.append('/home/pahm409')
from models.resnet3d import resnet3d_18


# ============================================================================
# ATTENTION MODULES (MATCHING TRAINING CODE)
# ============================================================================

class ECAAttention(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels: int):
        super().__init__()
        
        import math
        kernel_size = int(abs((math.log(channels, 2) + 1) / 2))
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)
        y = y.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
        y = self.conv(y).squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        attention_weights = self.sigmoid(y)
        return x * attention_weights


class MultiKernelDepthwiseConv(nn.Module):
    """Multi-Kernel Depthwise Convolution from MK-UNet"""
    def __init__(self, channels: int, kernels=[1, 3, 5]):
        super().__init__()
        
        self.channels = channels
        self.kernels = kernels
        self.num_kernels = len(kernels)
        
        self.dw_convs = nn.ModuleList()
        for k in kernels:
            self.dw_convs.append(nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=k, 
                         padding=k//2, groups=channels, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU6(inplace=True)
            ))
    
    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, d, h, w = x.size()
        groups = self.num_kernels
        
        while channels % groups != 0 and groups > 1:
            groups -= 1
        
        if groups == 1:
            return x
        
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, d, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, channels, d, h, w)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum([dw_conv(x) for dw_conv in self.dw_convs])
        out = self.channel_shuffle(out)
        return out


# ============================================================================
# MODEL ARCHITECTURE (MATCHING TRAINING CODE)
# ============================================================================

class ResNet3DEncoder(nn.Module):
    """Extract encoder from ResNet3D"""
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
    """
    Enhanced U-Net decoder with deep supervision support
    
    attention_type: 'none', 'eca', or 'mkdc'
    deep_supervision: If True, outputs auxiliary predictions at levels 4, 3, 2
    """
    def __init__(self, num_classes=2, attention_type='none', mkdc_kernels=[1, 3, 5],
                 deep_supervision=False):
        super(UNetDecoder3D, self).__init__()
        
        self.attention_type = attention_type
        self.deep_supervision = deep_supervision
        
        # Create attention modules based on type
        if attention_type == 'eca':
            # ECA: Applied AFTER decoder blocks
            self.eca4 = ECAAttention(256)
            self.eca3 = ECAAttention(128)
            self.eca2 = ECAAttention(64)
            self.eca1 = ECAAttention(64)
        elif attention_type == 'mkdc':
            # MKDC: Applied ON skip connections (before concat)
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256, kernels=mkdc_kernels)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128, kernels=mkdc_kernels)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64, kernels=mkdc_kernels)
        
        # Standard decoder blocks (same for all)
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds_conv4 = nn.Conv3d(256, num_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv3d(128, num_classes, kernel_size=1)
            self.ds_conv2 = nn.Conv3d(64, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x_up, x_skip):
        d_up, h_up, w_up = x_up.shape[2:]
        d_skip, h_skip, w_skip = x_skip.shape[2:]
        
        if d_up != d_skip or h_up != h_skip or w_up != w_skip:
            diff_d = d_skip - d_up
            diff_h = h_skip - h_up
            diff_w = w_skip - w_up
            
            if diff_d > 0 or diff_h > 0 or diff_w > 0:
                padding = [
                    max(0, diff_w // 2), max(0, diff_w - diff_w // 2),
                    max(0, diff_h // 2), max(0, diff_h - diff_h // 2),
                    max(0, diff_d // 2), max(0, diff_d - diff_d // 2)
                ]
                x_up = F.pad(x_up, padding)
            elif diff_d < 0 or diff_h < 0 or diff_w < 0:
                d_start = max(0, -diff_d // 2)
                h_start = max(0, -diff_h // 2)
                w_start = max(0, -diff_w // 2)
                x_up = x_up[:, :, d_start:d_start + d_skip, 
                           h_start:h_start + h_skip, w_start:w_start + w_skip]
        
        return x_up
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        ds_outputs = []  # For deep supervision
        
        # Level 4
        x = self.up4(x5)
        x = self._match_size(x, x4)
        if self.attention_type == 'mkdc':
            x4 = self.mkdc4(x4)  # MKDC on skip
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.attention_type == 'eca':
            x = self.eca4(x)  # ECA after decoder
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv4(x))
        
        # Level 3
        x = self.up3(x)
        x = self._match_size(x, x3)
        if self.attention_type == 'mkdc':
            x3 = self.mkdc3(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.attention_type == 'eca':
            x = self.eca3(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv3(x))
        
        # Level 2
        x = self.up2(x)
        x = self._match_size(x, x2)
        if self.attention_type == 'mkdc':
            x2 = self.mkdc2(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.attention_type == 'eca':
            x = self.eca2(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv2(x))
        
        # Level 1
        x = self.up1(x)
        x = self._match_size(x, x1)
        if self.attention_type == 'mkdc':
            x1 = self.mkdc1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        if self.attention_type == 'eca':
            x = self.eca1(x)
        
        # Final output
        final_output = self.final_conv(x)
        
        if self.deep_supervision:
            return [final_output] + ds_outputs  # [main, ds4, ds3, ds2]
        else:
            return final_output


class SegmentationModel(nn.Module):
    """Complete segmentation model: Encoder + Decoder"""
    def __init__(self, encoder, num_classes=2, attention_type='none', deep_supervision=False):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes, 
                                     attention_type=attention_type,
                                     deep_supervision=deep_supervision)
        self.attention_type = attention_type
        self.deep_supervision = deep_supervision
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        
        if self.deep_supervision:
            # Resize all outputs to input size
            resized_outputs = []
            for out in outputs:
                if out.shape[2:] != input_size:
                    out = F.interpolate(out, size=input_size, 
                                      mode='trilinear', align_corners=False)
                resized_outputs.append(out)
            return resized_outputs
        else:
            if outputs.shape[2:] != input_size:
                outputs = F.interpolate(outputs, size=input_size, 
                                      mode='trilinear', align_corners=False)
            return outputs


# ============================================================================
# SLIDING WINDOW RECONSTRUCTION
# ============================================================================

def get_sliding_window_patches(volume_shape, patch_size=(96, 96, 96), step_size=0.5):
    """
    Generate all patch coordinates for sliding window
    
    Args:
        volume_shape: (D, H, W) shape of volume
        patch_size: (pd, ph, pw) size of patches
        step_size: overlap (0.5 = 50% overlap)
    
    Returns:
        List of (start_d, start_h, start_w) coordinates
    """
    D, H, W = volume_shape
    pd, ph, pw = patch_size
    
    # Calculate step sizes
    step_d = max(1, int(pd * step_size))
    step_h = max(1, int(ph * step_size))
    step_w = max(1, int(pw * step_size))
    
    patches = []
    
    # Generate all starting positions
    d_starts = list(range(0, D - pd + 1, step_d))
    h_starts = list(range(0, H - ph + 1, step_h))
    w_starts = list(range(0, W - pw + 1, step_w))
    
    # Ensure we cover the end
    if len(d_starts) == 0 or d_starts[-1] + pd < D:
        d_starts.append(max(0, D - pd))
    if len(h_starts) == 0 or h_starts[-1] + ph < H:
        h_starts.append(max(0, H - ph))
    if len(w_starts) == 0 or w_starts[-1] + pw < W:
        w_starts.append(max(0, W - pw))
    
    for d_start in d_starts:
        for h_start in h_starts:
            for w_start in w_starts:
                patches.append((d_start, h_start, w_start))
    
    return patches


def sliding_window_inference(model, volume, device, patch_size=(96, 96, 96), 
                            step_size=0.5, batch_size=4, deep_supervision=False):
    """
    Sliding window inference with overlap averaging
    
    Args:
        model: Segmentation model
        volume: Input volume (D, H, W)
        device: torch device
        patch_size: Size of patches
        step_size: Step size (0.5 = 50% overlap)
        batch_size: Batch size for processing
        deep_supervision: Whether model uses deep supervision
    
    Returns:
        prediction: (D, H, W) probability map (foreground class)
        count_map: (D, H, W) how many times each voxel was predicted
    """
    model.eval()
    
    volume_shape = volume.shape
    D, H, W = volume_shape
    
    # Initialize outputs
    prediction = np.zeros(volume_shape, dtype=np.float32)
    count_map = np.zeros(volume_shape, dtype=np.float32)
    
    # Get all patch coordinates
    patch_coords = get_sliding_window_patches(volume_shape, patch_size, step_size)
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(patch_coords), batch_size), 
                     desc='Sliding window', leave=False):
            batch_coords = patch_coords[i:i+batch_size]
            
            # Extract patches
            patches = []
            for (d_start, h_start, w_start) in batch_coords:
                patch = volume[d_start:d_start+patch_size[0],
                              h_start:h_start+patch_size[1],
                              w_start:w_start+patch_size[2]]
                patches.append(patch)
            
            # Stack and predict
            patches_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(device)
            
            with autocast():
                outputs = model(patches_tensor)
                
                # Handle deep supervision: use main output only
                if deep_supervision:
                    outputs = outputs[0]
            
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Place predictions back
            for j, (d_start, h_start, w_start) in enumerate(batch_coords):
                pred_patch = preds[j, 1]  # Foreground class
                
                prediction[d_start:d_start+patch_size[0],
                          h_start:h_start+patch_size[1],
                          w_start:w_start+patch_size[2]] += pred_patch
                
                count_map[d_start:d_start+patch_size[0],
                         h_start:h_start+patch_size[1],
                         w_start:w_start+patch_size[2]] += 1.0
    
    # Average overlapping predictions
    mask = count_map > 0
    prediction[mask] = prediction[mask] / count_map[mask]
    
    return prediction, count_map


# ============================================================================
# DATA LOADING FOR ATLAS/UOA
# ============================================================================

def load_atlas_uoa_case(case_id, preprocessed_dir):
    """Load ATLAS/UOA case from preprocessed directory"""
    import nibabel as nib
    
    base_dir = Path(preprocessed_dir)
    
    # Try ATLAS first, then UOA
    case_file = base_dir / 'ATLAS' / f'{case_id}.npz'
    if not case_file.exists():
        case_file = base_dir / 'UOA_Private' / f'{case_id}.npz'
    
    if not case_file.exists():
        raise FileNotFoundError(f"Case {case_id} not found in ATLAS or UOA_Private")
    
    data = np.load(case_file)
    volume = data['volume']  # Already preprocessed to 96x96x96
    mask = data['mask']
    
    return volume, mask


def evaluate_atlas_uoa(model, test_cases, preprocessed_dir, device, 
                      patch_size=(96, 96, 96), step_size=0.5,
                      deep_supervision=False, save_nifti=False, output_dir=None):
    """
    Evaluate on ATLAS/UOA test set
    
    These volumes are already in 96³ space, so we:
    1. Run sliding window inference
    2. Compute DSC directly (no reverse preprocessing needed)
    """
    import nibabel as nib
    
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"ATLAS/UOA EVALUATION - SLIDING WINDOW")
    print(f"{'='*80}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Patch size: {patch_size}")
    print(f"Step size: {step_size} (overlap = {1-step_size:.0%})")
    print(f"Deep supervision: {deep_supervision}")
    print(f"{'='*80}\n")
    
    all_dscs = []
    results_per_case = []
    
    print(f"{'Case ID':<25} {'DSC':<8} {'GT':<8} {'Pred':<8} {'#Patches':<10}")
    print("-" * 80)
    
    for case_id in tqdm(test_cases, desc='Evaluating'):
        try:
            # Load data
            volume, mask_gt = load_atlas_uoa_case(case_id, preprocessed_dir)
            
            # Sliding window inference
            pred_prob, count_map = sliding_window_inference(
                model=model,
                volume=volume,
                device=device,
                patch_size=patch_size,
                step_size=step_size,
                batch_size=4,
                deep_supervision=deep_supervision
            )
            
            # Binarize
            pred_binary = (pred_prob > 0.5).astype(np.uint8)
            mask_gt_binary = (mask_gt > 0).astype(np.uint8)
            
            # Compute DSC
            intersection = (pred_binary * mask_gt_binary).sum()
            union = pred_binary.sum() + mask_gt_binary.sum()
            dsc = (2.0 * intersection) / union if union > 0 else (1.0 if pred_binary.sum() == 0 else 0.0)
            
            all_dscs.append(dsc)
            
            gt_vol = int(mask_gt_binary.sum())
            pred_vol = int(pred_binary.sum())
            num_patches = len(get_sliding_window_patches(volume.shape, patch_size, step_size))
            
            print(f"{case_id:<25} {dsc:<8.4f} {gt_vol:<8} {pred_vol:<8} {num_patches:<10}")
            
            results_per_case.append({
                'case_id': case_id,
                'dsc': float(dsc),
                'num_patches': int(num_patches),
                'gt_volume': gt_vol,
                'pred_volume': pred_vol
            })
            
            # Save NIfTI if requested
            if save_nifti and output_dir:
                nifti_dir = Path(output_dir) / case_id
                nifti_dir.mkdir(parents=True, exist_ok=True)
                
                affine = np.eye(4)
                nib.save(nib.Nifti1Image(pred_binary, affine), 
                        nifti_dir / 'prediction.nii.gz')
                nib.save(nib.Nifti1Image(pred_prob.astype(np.float32), affine), 
                        nifti_dir / 'prediction_prob.nii.gz')
                nib.save(nib.Nifti1Image(mask_gt_binary, affine), 
                        nifti_dir / 'ground_truth.nii.gz')
                nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), 
                        nifti_dir / 'volume.nii.gz')
                nib.save(nib.Nifti1Image(count_map.astype(np.float32), affine), 
                        nifti_dir / 'count_map.nii.gz')
        
        except Exception as e:
            print(f"\n❌ ERROR for {case_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("-" * 80)
    
    if len(all_dscs) == 0:
        print("\n❌ No valid test cases!")
        return 0.0, [], []
    
    mean_dsc = np.mean(all_dscs)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Cases: {len(all_dscs)}/{len(test_cases)}")
    print(f"  Mean DSC: {mean_dsc:.4f} ± {np.std(all_dscs):.4f}")
    print(f"  Median: {np.median(all_dscs):.4f}")
    print(f"  Min/Max: {np.min(all_dscs):.4f} / {np.max(all_dscs):.4f}\n")
    
    return mean_dsc, all_dscs, results_per_case


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sliding window evaluation for trained models')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation',
                       help='Preprocessed data directory')
    parser.add_argument('--splits-file', type=str,
                       default='splits_5fold.json',
                       help='Splits JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/sliding_window_test_results',
                       help='Output directory for results')
    parser.add_argument('--step-size', type=float, default=0.5,
                       help='Sliding window step size (0.5 = 50%% overlap)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--save-nifti', action='store_true',
                       help='Save NIfTI predictions')
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    fold = checkpoint.get('fold', 0)
    attention_type = checkpoint.get('attention_type', 'none')
    deep_supervision = checkpoint.get('deep_supervision', False)
    
    print(f"\n{'='*80}")
    print(f"SLIDING WINDOW EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {fold}")
    print(f"Attention: {attention_type}")
    print(f"Deep supervision: {deep_supervision}")
    print(f"Val DSC (recon): {checkpoint.get('val_dsc_recon', 'N/A')}")
    print(f"Step size: {args.step_size}")
    print(f"{'='*80}\n")
    
    # Build model
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder=encoder,
        num_classes=2,
        attention_type=attention_type,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    # Load splits
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    # Get test cases for this fold
    fold_splits = splits[f'fold_{fold}']
    test_cases = []
    
    if 'ATLAS' in fold_splits:
        test_cases.extend(fold_splits['ATLAS']['test'])
    if 'UOA_Private' in fold_splits:
        test_cases.extend(fold_splits['UOA_Private']['test'])
    
    print(f"Test cases: {len(test_cases)}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir) / f'fold_{fold}_sliding_window'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    mean_dsc, all_dscs, results_per_case = evaluate_atlas_uoa(
        model=model,
        test_cases=test_cases,
        preprocessed_dir=args.preprocessed_dir,
        device=device,
        patch_size=(96, 96, 96),
        step_size=args.step_size,
        deep_supervision=deep_supervision,
        save_nifti=args.save_nifti,
        output_dir=output_dir if args.save_nifti else None
    )
    
    # Save results
    results = {
        'fold': int(fold),
        'checkpoint': str(args.checkpoint),
        'attention_type': attention_type,
        'deep_supervision': deep_supervision,
        'step_size': float(args.step_size),
        'mean_dsc': float(mean_dsc),
        'std_dsc': float(np.std(all_dscs)),
        'median_dsc': float(np.median(all_dscs)),
        'min_dsc': float(np.min(all_dscs)),
        'max_dsc': float(np.max(all_dscs)),
        'num_test_cases': int(len(all_dscs)),
        'all_dscs': [float(x) for x in all_dscs],
        'per_case_results': results_per_case
    }
    
    result_file = output_dir / 'test_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {result_file}\n")


if __name__ == '__main__':
    main()

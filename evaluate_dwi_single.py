"""
evaluate_dwi_single.py

Simple evaluation script for a single DWI checkpoint
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.ndimage import zoom
import sys

sys.path.append('.')
from models.resnet3d import resnet3d_18
from dataset.patch_dataset_with_centers import PatchDatasetWithCenters


# ============================================================================
# MODEL CLASSES (from train_dwi_scratch.py)
# ============================================================================

class ResNet3DEncoder(torch.nn.Module):
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


class MultiKernelDepthwiseConv(torch.nn.Module):
    def __init__(self, channels: int, kernels=[1, 3, 5]):
        super().__init__()
        self.channels = channels
        self.kernels = kernels
        self.num_kernels = len(kernels)
        
        self.dw_convs = torch.nn.ModuleList()
        for k in kernels:
            self.dw_convs.append(torch.nn.Sequential(
                torch.nn.Conv3d(channels, channels, kernel_size=k, 
                         padding=k//2, groups=channels, bias=False),
                torch.nn.BatchNorm3d(channels),
                torch.nn.ReLU6(inplace=True)
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


class UNetDecoder3D(torch.nn.Module):
    def __init__(self, num_classes=2, use_mkdc=False, deep_supervision=False):
        super(UNetDecoder3D, self).__init__()
        
        self.use_mkdc = use_mkdc
        self.deep_supervision = deep_supervision
        
        if use_mkdc:
            self.mkdc4 = MultiKernelDepthwiseConv(channels=256)
            self.mkdc3 = MultiKernelDepthwiseConv(channels=128)
            self.mkdc2 = MultiKernelDepthwiseConv(channels=64)
            self.mkdc1 = MultiKernelDepthwiseConv(channels=64)
        
        self.up4 = torch.nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        
        self.up3 = torch.nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        
        self.up2 = torch.nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        
        self.up1 = torch.nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        
        self.final_conv = torch.nn.Conv3d(64, num_classes, kernel_size=1)
        
        if deep_supervision:
            self.ds_conv4 = torch.nn.Conv3d(256, num_classes, kernel_size=1)
            self.ds_conv3 = torch.nn.Conv3d(128, num_classes, kernel_size=1)
            self.ds_conv2 = torch.nn.Conv3d(64, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        
        ds_outputs = []
        
        x = self.up4(x5)
        if self.use_mkdc:
            x4 = self.mkdc4(x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv4(x))
        
        x = self.up3(x)
        if self.use_mkdc:
            x3 = self.mkdc3(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv3(x))
        
        x = self.up2(x)
        if self.use_mkdc:
            x2 = self.mkdc2(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        if self.deep_supervision:
            ds_outputs.append(self.ds_conv2(x))
        
        x = self.up1(x)
        if self.use_mkdc:
            x1 = self.mkdc1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        final_output = self.final_conv(x)
        
        if self.deep_supervision:
            return [final_output] + ds_outputs
        else:
            return final_output


import torch

class SegmentationModel(torch.nn.Module):
    def __init__(self, encoder, num_classes=2, use_mkdc=False, deep_supervision=False):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes, 
                                     use_mkdc=use_mkdc,
                                     deep_supervision=deep_supervision)
        self.deep_supervision = deep_supervision
    
    def forward(self, x):
        input_size = x.shape[2:]  # (96, 96, 96)
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        
        if self.deep_supervision:
            # Resize all outputs to input size
            resized_outputs = []
            for out in outputs:
                if out.shape[2:] != input_size:
                    out = torch.nn.functional.interpolate(
                        out, size=input_size,
                        mode='trilinear', align_corners=False
                    )
                resized_outputs.append(out)
            return resized_outputs
        else:
            if outputs.shape[2:] != input_size:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=input_size,
                    mode='trilinear', align_corners=False
                )
            return outputs



def get_original_isles_info(case_id, isles_raw_dir='/home/pahm409/ISLES2022_reg/ISLES2022'):
    import nibabel as nib
    
    isles_raw = Path(isles_raw_dir)
    dwi_file = isles_raw / case_id / f'{case_id}_ses-0001_dwi.nii.gz'
    msk_file = isles_raw / case_id / f'{case_id}_ses-0001_msk.nii.gz'
    
    if not dwi_file.exists():
        raise FileNotFoundError(f"Cannot find DWI file: {dwi_file}")
    if not msk_file.exists():
        raise FileNotFoundError(f"Cannot find mask file: {msk_file}")
    
    dwi_img = nib.load(dwi_file)
    msk_img = nib.load(msk_file)
    
    original_shape = dwi_img.shape
    original_affine = dwi_img.affine
    ground_truth = (msk_img.get_fdata() > 0.5).astype(np.uint8)
    
    resampled_shape = np.array([197, 233, 189])
    original_shape_np = np.array(original_shape)
    zoom_factors = original_shape_np / resampled_shape
    
    return {
        'original_shape': tuple(original_shape),
        'original_affine': original_affine,
        'zoom_factors': zoom_factors,
        'ground_truth': ground_truth
    }


def resample_to_original(prediction, original_info, method='nearest'):
    zoom_factors = original_info['zoom_factors']
    order = 0 if method == 'nearest' else 1
    
    resampled = zoom(prediction, zoom_factors, order=order)
    
    target_shape = original_info['original_shape']
    if resampled.shape != target_shape:
        resampled_fixed = np.zeros(target_shape, dtype=resampled.dtype)
        slices = tuple(slice(0, min(resampled.shape[i], target_shape[i])) for i in range(3))
        resampled_fixed[slices] = resampled[slices]
        resampled = resampled_fixed
    
    return resampled


def reconstruct_from_patches_with_count(patch_preds, centers, original_shape, patch_size=(96, 96, 96)):
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)
    half_size = np.array(patch_size) // 2  # Keep as 48 for 96×96×96 patches
    
    for i, center in enumerate(centers):
        lower = center - half_size
        upper = center + half_size
        
        valid = True
        for d in range(3):
            if lower[d] < 0 or upper[d] > original_shape[d]:
                valid = False
                break
        
        if not valid:
            continue
        
        patch = patch_preds[i, 1, ...]  # This is 48×48×48
        reconstructed[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += patch
        count_map[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += 1.0
    
    mask = count_map > 0
    reconstructed[mask] = reconstructed[mask] / count_map[mask]
    
    return reconstructed, count_map
    
    
    
def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    use_mkdc = checkpoint.get('use_mkdc', False)
    deep_supervision = checkpoint.get('deep_supervision', False)
    fold = checkpoint.get('fold', 0)
    
    encoder = resnet3d_18(in_channels=1)
    model = SegmentationModel(
        encoder, 
        num_classes=2,
        use_mkdc=use_mkdc,
        deep_supervision=deep_supervision
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, use_mkdc, deep_supervision, fold


def evaluate_checkpoint(
checkpoint_path,
isles_preprocessed_dir,
isles_splits_file,
isles_raw_dir,
output_dir,
save_nifti=False
):
    import nibabel as nib

    device = torch.device('cuda:0')

    print("\n" + "="*80)
    print("DWI MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print("="*80 + "\n")

    # Load model
    print("Loading model...")
    model, use_mkdc, deep_supervision, fold = load_model_from_checkpoint(checkpoint_path, device)
    print(f"✓ Loaded (MKDC={use_mkdc}, DS={deep_supervision}, Fold={fold})\n")

    # ------------------------------------------------------------------
    # MODEL DEBUG BLOCK
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE DEBUG")
    print("="*80)
    print(f"use_mkdc: {use_mkdc}")
    print(f"deep_supervision: {deep_supervision}")

    test_input = torch.randn(1, 1, 96, 96, 96).to(device)
    print(f"\nInput shape: {test_input.shape}")

    with torch.no_grad():
        enc_features = model.encoder(test_input)
        print("\nEncoder outputs:")
        for i, feat in enumerate(enc_features):
            print(f"  x{i+1}: {feat.shape}")

        dec_output = model.decoder(enc_features)
        if isinstance(dec_output, list):
            print("\nDecoder outputs (list):")
            for i, out in enumerate(dec_output):
                print(f"  out[{i}]: {out.shape}")
        else:
            print(f"\nDecoder output: {dec_output.shape}")

        final_output = model(test_input)
        if isinstance(final_output, list):
            print("\nFinal model outputs (list):")
            for i, out in enumerate(final_output):
                print(f"  out[{i}]: {out.shape}")
        else:
            print(f"\nFinal model output: {final_output.shape}")

    print("="*80 + "\n")
    # ------------------------------------------------------------------

    # Load test dataset
    print(f"Loading test dataset for fold {fold}...")
    test_dataset = PatchDatasetWithCenters(
        preprocessed_dir=isles_preprocessed_dir,
        datasets=['ISLES2022_resampled'],
        split='test',
        splits_file=isles_splits_file,
        fold=fold,
        patch_size=(96, 96, 96),
        patches_per_volume=100,
        augment=False,
        lesion_focus_ratio=0.0,
        compute_lesion_bins=False
    )

    print(f"✓ Test volumes: {len(test_dataset.volumes)}\n")

    if len(test_dataset.volumes) == 0:
        print("❌ ERROR: No test volumes!")
        return

    dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Generating predictions...")
    volume_data = defaultdict(lambda: {'centers': [], 'preds': []})

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing patches'):
            images = batch['image'].to(device)
            centers = batch['center'].cpu().numpy()
            vol_indices = batch['vol_idx'].cpu().numpy()

            with autocast():
                outputs = model(images)
                if deep_supervision:
                    outputs = outputs[0]

            preds = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(images)):
                vid = vol_indices[i]
                volume_data[vid]['centers'].append(centers[i])
                volume_data[vid]['preds'].append(preds[i])

    torch.cuda.empty_cache()

    print("\nReconstructing volumes...")
    all_dscs = []
    results_per_case = []

    for vol_idx in tqdm(sorted(volume_data.keys()), desc='Reconstructing'):
        vol_info = test_dataset.get_volume_info(vol_idx)
        case_id = vol_info['case_id']

        if vol_idx == 0:
            print("\n" + "="*80)
            print(f"DEBUG INFO FOR {case_id}:")
            print("="*80)
            print(f"vol_info['volume'].shape = {vol_info['volume'].shape}")
            print(f"vol_info['mask'].shape   = {vol_info['mask'].shape}")
            print("="*80 + "\n")

        try:
            original_info = get_original_isles_info(case_id, isles_raw_dir)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            continue

        centers = np.array(volume_data[vol_idx]['centers'])
        preds = np.array(volume_data[vol_idx]['preds'])

        reconstructed, count_map = reconstruct_from_patches_with_count(
            preds, centers, vol_info['volume'].shape, patch_size=(96,96,96)
        )

        pred_prob_original = resample_to_original(reconstructed, original_info, method='linear')
        pred_binary_original = (pred_prob_original > 0.5).astype(np.uint8)
        mask_gt_original = original_info['ground_truth']

        inter = (pred_binary_original * mask_gt_original).sum()
        union = pred_binary_original.sum() + mask_gt_original.sum()
        dsc = (2*inter)/union if union > 0 else 1.0

        all_dscs.append(dsc)

        results_per_case.append({
            'case_id': case_id,
            'dsc': float(dsc)
        })

        print(f"{case_id}: DSC = {dsc:.4f}")

    if len(all_dscs) == 0:
        print("❌ No valid cases")
        return

    mean_dsc = np.mean(all_dscs)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Mean DSC: {mean_dsc:.4f}")
    print("="*80 + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate single DWI checkpoint')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--isles-preprocessed-dir', type=str,
                       default='/home/pahm409/preprocessed_stroke_foundation')
    parser.add_argument('--isles-raw-dir', type=str,
                       default='/home/pahm409/ISLES2022_reg/ISLES2022')
    parser.add_argument('--splits-file', type=str,
                       default='isles_splits_5fold_resampled.json')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/dwi_evaluation_results')
    parser.add_argument('--save-nifti', action='store_true')
    
    args = parser.parse_args()
    
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        isles_preprocessed_dir=args.isles_preprocessed_dir,
        isles_splits_file=args.splits_file,
        isles_raw_dir=args.isles_raw_dir,
        output_dir=args.output_dir,
        save_nifti=args.save_nifti
    )

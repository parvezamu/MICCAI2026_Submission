"""
preprocess_neuralcup_flair_correct.py

CORRECT: Use consistent MNI reference for both FLAIR and masks

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from pathlib import Path
from tqdm import tqdm
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def preprocess_volume(volume):
    """Apply same preprocessing as training"""
    brain_mask = volume > np.percentile(volume, 1)
    
    brain_voxels = volume[brain_mask]
    if len(brain_voxels) > 0:
        percentile_val = np.percentile(brain_voxels, 99.5)
        volume = np.clip(volume, 0, percentile_val)
    
    output = np.zeros_like(volume)
    brain_voxels = volume[brain_mask]
    if len(brain_voxels) > 0:
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        if std > 0:
            output[brain_mask] = (volume[brain_mask] - mean) / std
    
    return output


def preprocess_neuralcup_flair():
    """
    Preprocess NEURALCUP FLAIR
    Skip resampling - FLAIR should already be registered if masks are registered
    """
    
    neuralcup_dir = Path('/home/pahm409/NEURALCUP')
    flair_dir = neuralcup_dir / 'FLAIR'
    mask_dir = neuralcup_dir / 'lesion_masks_registered'
    
    output_dir = Path('/home/pahm409/preprocessed_NEURALCUP_FLAIR_v2/NEURALCUP')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    flair_files = sorted(flair_dir.glob('BBS*_FLAIR.nii'))
    
    print("\n" + "="*70)
    print("PREPROCESSING NEURALCUP FLAIR - CORRECTED")
    print("="*70)
    print("Strategy: Resample both to 1mm MNI independently")
    print("="*70 + "\n")
    
    matched_cases = []
    for flair_file in flair_files:
        bbs_id = flair_file.stem.replace('_FLAIR', '')
        mask_file = mask_dir / f'{bbs_id}.nii.gz'
        
        if mask_file.exists():
            matched_cases.append({
                'case_id': bbs_id,
                'flair': str(flair_file),
                'mask': str(mask_file)
            })
    
    print(f"✓ Matched cases: {len(matched_cases)}\n")
    
    metadata_list = []
    target_shape = (182, 218, 182)
    
    for case in tqdm(matched_cases, desc="Preprocessing"):
        case_id = case['case_id']
        
        try:
            # Load and resample FLAIR to 1mm
            flair_nib = nib.load(case['flair'])
            flair_1mm = resample_to_output(flair_nib, voxel_sizes=(1.0, 1.0, 1.0), order=3)
            flair_data = flair_1mm.get_fdata().astype(np.float32)
            
            # Load and resample mask to 1mm  
            mask_nib = nib.load(case['mask'])
            mask_1mm = resample_to_output(mask_nib, voxel_sizes=(1.0, 1.0, 1.0), order=0)
            mask_data = mask_1mm.get_fdata().astype(np.uint8)
            mask_data = (mask_data > 0.5).astype(np.uint8)
            
            # Check shapes
            print(f"\n{case_id}: FLAIR shape={flair_data.shape}, Mask shape={mask_data.shape}")
            
            # For now, just save what we have (we'll handle shape matching later)
            # Crop or pad BOTH to same intermediate size first
            flair_shape = np.array(flair_data.shape)
            mask_shape = np.array(mask_data.shape)
            
            # Use minimum dimensions
            min_shape = np.minimum(flair_shape, mask_shape)
            
            # Crop both to min_shape
            flair_cropped = flair_data[
                :min_shape[0],
                :min_shape[1],
                :min_shape[2]
            ]
            
            mask_cropped = mask_data[
                :min_shape[0],
                :min_shape[1],
                :min_shape[2]
            ]
            
            # Now pad both to target shape
            pad_width = []
            for i in range(3):
                diff = target_shape[i] - min_shape[i]
                if diff > 0:
                    pad_width.append((diff // 2, diff - diff // 2))
                else:
                    # Need to crop more
                    pad_width.append((0, 0))
            
            flair_final = np.pad(flair_cropped, pad_width, mode='constant',
                                constant_values=flair_cropped.min())
            mask_final = np.pad(mask_cropped, pad_width, mode='constant', constant_values=0)
            
            # Crop if still too large
            flair_final = flair_final[:target_shape[0], :target_shape[1], :target_shape[2]]
            mask_final = mask_final[:target_shape[0], :target_shape[1], :target_shape[2]]
            
            # Final padding if needed
            if flair_final.shape != target_shape:
                final_pad = []
                for i in range(3):
                    diff = target_shape[i] - flair_final.shape[i]
                    final_pad.append((0, max(0, diff)))
                
                flair_final = np.pad(flair_final, final_pad, mode='constant',
                                    constant_values=flair_final.min())
                mask_final = np.pad(mask_final, final_pad, mode='constant', constant_values=0)
            
            print(f"  Final: FLAIR={flair_final.shape}, Mask={mask_final.shape}, Lesion voxels={mask_final.sum()}")
            
            # Preprocess
            flair_normalized = preprocess_volume(flair_final)
            brain_mask = flair_final > np.percentile(flair_final, 1)
            
            # Verify preprocessing
            print(f"  Normalized: mean={flair_normalized.mean():.4f}, std={flair_normalized.std():.4f}")
            
            # Statistics
            lesion_volume_ml = float(mask_final.sum() * 0.001)
            
            # Save
            npz_path = output_dir / f'{case_id}.npz'
            np.savez_compressed(
                npz_path,
                image=flair_normalized.astype(np.float32),
                lesion_mask=mask_final.astype(np.uint8),
                brain_mask=brain_mask.astype(np.uint8)
            )
            
            metadata = {
                'case_id': case_id,
                'shape': tuple(int(x) for x in target_shape),
                'lesion_volume_ml': float(lesion_volume_ml),
                'lesion_voxels': int(mask_final.sum()),
                'npz_path': str(npz_path)
            }
            
            metadata_list.append(metadata)
            
        except Exception as e:
            print(f"\n{case_id}: ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print(f"COMPLETE! Processed {len(metadata_list)} cases")
    print("="*70)


if __name__ == '__main__':
    preprocess_neuralcup_flair()

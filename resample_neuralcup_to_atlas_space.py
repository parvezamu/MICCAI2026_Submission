"""
resample_neuralcup_to_atlas_space.py

Resample + preprocess NEURALCUP to match the SAME minimal preprocessing used for ATLAS/UOA:
- Brain mask: > 1st percentile
- Clip within brain to 99.5 percentile, then clamp to [0, p99.5]
- Z-score within brain mask
- IMPORTANT FIX: resample lesion mask onto the EXACT grid of the resampled T1
  (prevents affine/grid mismatch that can destroy DSC)

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
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


def preprocess_volume_like_training(volume: np.ndarray, clip_percentile: float = 99.5):
    """
    Match StrokeFoundationPreprocessor preprocessing (CPU path):
    1) brain_mask = volume > 1st percentile
    2) clip within brain to p99.5 then clamp to [0, p99.5]
    3) z-score within brain mask
    Returns: normalized_volume, brain_mask
    """
    # Brain mask (same logic as training)
    brain_mask = volume > np.percentile(volume, 1)

    # Clip within brain (same logic as training)
    brain_voxels = volume[brain_mask]
    if brain_voxels.size > 0:
        p = np.percentile(brain_voxels, clip_percentile)
        volume = np.clip(volume, 0, p)

    # Z-score within brain
    output = np.zeros_like(volume, dtype=np.float32)
    brain_voxels = volume[brain_mask]
    if brain_voxels.size > 0:
        mean = float(brain_voxels.mean())
        std = float(brain_voxels.std())
        if std > 0:
            output[brain_mask] = (volume[brain_mask] - mean) / std

    return output.astype(np.float32), brain_mask.astype(np.uint8)


def resample_neuralcup():
    """
    Resample NEURALCUP T1 to 1mm isotropic and resample mask to the EXACT grid of that T1.
    """
    neuralcup_dir = Path('/home/pahm409/NEURALCUP')
    t1_dir = neuralcup_dir / 'T1s'
    mask_dir = neuralcup_dir / 'lesion_masks_registered'

    output_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input pattern (as in your script)
    t1_files = sorted(t1_dir.glob('wmBBS*_3DT1.nii'))

    print("\n" + "=" * 70)
    print("RESAMPLING + PREPROCESSING NEURALCUP (1mm) - GRID-SAFE MASK RESAMPLING")
    print("=" * 70)
    print(f"Input T1:   {t1_dir}")
    print(f"Input Mask: {mask_dir}")
    print(f"Output:     {output_dir}")
    print("Target:     1mm isotropic; mask resampled to T1 grid")
    print("=" * 70 + "\n")

    metadata_list = []
    total = len(t1_files)

    for t1_file in tqdm(t1_files, desc="Processing"):
        bbs_id = t1_file.stem.replace('wm', '').replace('_3DT1', '')
        mask_file = mask_dir / f'{bbs_id}.nii.gz'

        if not mask_file.exists():
            print(f"  ⚠️  No mask for {bbs_id} -> skipping")
            continue

        try:
            # Load T1
            t1_nib = nib.load(str(t1_file))

            # Load mask
            mask_nib = nib.load(str(mask_file))

            # Resample T1 to 1mm isotropic
            t1_resampled = resample_to_output(t1_nib, voxel_sizes=(1.0, 1.0, 1.0), order=3)
            t1_resampled_data = t1_resampled.get_fdata().astype(np.float32)

            # IMPORTANT FIX: resample mask to the EXACT grid of t1_resampled
            mask_resampled = resample_from_to(mask_nib, t1_resampled, order=0)
            mask_resampled_data = mask_resampled.get_fdata().astype(np.float32)
            mask_resampled_data = (mask_resampled_data > 0.5).astype(np.uint8)

            # Sanity check
            if t1_resampled_data.shape != mask_resampled_data.shape:
                raise RuntimeError(
                    f"Shape mismatch after grid-safe resampling: "
                    f"T1 {t1_resampled_data.shape} vs mask {mask_resampled_data.shape}"
                )

            # Preprocess (match training)
            t1_normalized, brain_mask = preprocess_volume_like_training(t1_resampled_data, clip_percentile=99.5)

            # Lesion stats (1mm voxels)
            voxel_volume_ml = (1.0 * 1.0 * 1.0) / 1000.0
            lesion_vox = int(mask_resampled_data.sum())
            lesion_volume_ml = float(lesion_vox * voxel_volume_ml)

            # Save NPZ
            npz_path = output_dir / f'{bbs_id}.npz'
            np.savez_compressed(
                npz_path,
                image=t1_normalized.astype(np.float32),
                lesion_mask=mask_resampled_data.astype(np.uint8),
                brain_mask=brain_mask.astype(np.uint8)
            )

            metadata = {
                'case_id': bbs_id,
                't1_path': str(t1_file),
                'mask_path': str(mask_file),
                'shape': tuple(int(x) for x in t1_resampled_data.shape),
                'spacing': (1.0, 1.0, 1.0),
                'lesion_voxels': lesion_vox,
                'lesion_volume_ml': lesion_volume_ml,
                'npz_path': str(npz_path),
                'affine_t1_resampled': t1_resampled.affine.tolist(),
                'affine_mask_resampled': mask_resampled.affine.tolist(),
                'preprocessing': {
                    'brain_mask': 'volume > 1st percentile',
                    'clip': 'within brain to 99.5 percentile, then clamp [0, p99.5]',
                    'normalize': 'z-score within brain'
                },
                'resampling': {
                    't1': 'resample_to_output(voxel_sizes=(1,1,1), order=3)',
                    'mask': 'resample_from_to(mask, t1_resampled, order=0)'
                }
            }

            metadata_list.append(metadata)

        except Exception as e:
            print(f"\n  ✗ Error processing {bbs_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    lesion_volumes = [m['lesion_volume_ml'] for m in metadata_list]
    lesion_voxels = [m['lesion_voxels'] for m in metadata_list]

    summary = {
        'dataset_name': 'NEURALCUP_1mm',
        'total_cases': int(total),
        'successful_cases': int(len(metadata_list)),
        'failed_cases': int(total - len(metadata_list)),
        'target': '1mm_isotropic',
        'note': 'lesion mask resampled onto exact T1 grid using resample_from_to',
        'lesion_statistics': {
            'mean_volume_ml': float(np.mean(lesion_volumes)) if lesion_volumes else 0.0,
            'median_volume_ml': float(np.median(lesion_volumes)) if lesion_volumes else 0.0,
            'std_volume_ml': float(np.std(lesion_volumes)) if lesion_volumes else 0.0,
            'min_volume_ml': float(np.min(lesion_volumes)) if lesion_volumes else 0.0,
            'max_volume_ml': float(np.max(lesion_volumes)) if lesion_volumes else 0.0,
            'total_volume_ml': float(np.sum(lesion_volumes)) if lesion_volumes else 0.0,
            'mean_lesion_voxels': float(np.mean(lesion_voxels)) if lesion_voxels else 0.0
        },
        'case_metadata': metadata_list
    }

    summary_path = output_dir / 'NEURALCUP_1mm_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 70)
    print("NEURALCUP PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Processed: {len(metadata_list)}/{total}")
    if lesion_volumes:
        print(f"Mean lesion volume: {summary['lesion_statistics']['mean_volume_ml']:.2f} ml")
        print(f"Median lesion volume: {summary['lesion_statistics']['median_volume_ml']:.2f} ml")
        print(f"Min/Max lesion volume: {summary['lesion_statistics']['min_volume_ml']:.2f} / {summary['lesion_statistics']['max_volume_ml']:.2f} ml")
    print(f"Output:  {output_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    resample_neuralcup()


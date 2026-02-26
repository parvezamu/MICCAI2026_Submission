"""
preprocess_stroke_foundation.py

Preprocessing for stroke foundation model with GPU support
Fixed: JSON serialization for numpy types

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import numpy as np
import nibabel as nib
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm
import subprocess
import shutil
from glob import glob
import argparse
import torch


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class StrokeFoundationPreprocessor:
    """
    Lightweight preprocessor for datasets already in MNI space
    GPU-accelerated where possible
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalization_method: str = 'zscore',
        clip_percentile: float = 99.5,
        use_skull_strip: bool = False,
        skip_registration: bool = True,
        use_gpu: bool = True,
        **kwargs  # Ignore extra arguments like target_size
    ):
        """
        Args:
            target_spacing: Target voxel spacing
            normalization_method: 'zscore' or 'minmax'
            clip_percentile: Percentile for intensity clipping
            use_skull_strip: Whether to apply skull stripping
            skip_registration: Skip registration (data already in MNI)
            use_gpu: Use GPU for computations if available
        """
        self.target_spacing = target_spacing
        self.normalization_method = normalization_method
        self.clip_percentile = clip_percentile
        self.use_skull_strip = use_skull_strip
        self.skip_registration = skip_registration
        
        # Setup GPU
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        if self.use_gpu:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✓ Using CPU")
    
    def preprocess_case(
        self,
        image_path: str,
        mask_path: str,
        output_dir: str,
        case_id: str
    ) -> Optional[Dict]:
        """
        Minimal preprocessing for MNI-space data with GPU acceleration
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load image
            img_nib = nib.load(image_path)
            img_data = img_nib.get_fdata().astype(np.float32)
            
            # Load mask
            mask_nib = nib.load(mask_path)
            mask_data = mask_nib.get_fdata().astype(np.uint8)
            
            # Ensure binary mask
            mask_data = (mask_data > 0.5).astype(np.uint8)
            
            # Check alignment
            if img_data.shape != mask_data.shape:
                print(f"  ⚠️  Shape mismatch: img {img_data.shape} vs mask {mask_data.shape}")
                return None
            
            # Move to GPU if available
            if self.use_gpu:
                img_tensor = torch.from_numpy(img_data).to(self.device)
            else:
                img_tensor = img_data
            
            # Create brain mask (simple threshold)
            if self.use_gpu:
                brain_mask = (img_tensor > torch.quantile(img_tensor.flatten(), 0.01)).cpu().numpy()
            else:
                brain_mask = img_data > np.percentile(img_data, 1)
            
            # Intensity clipping
            brain_voxels = img_data[brain_mask]
            if len(brain_voxels) > 0:
                percentile_val = np.percentile(brain_voxels, self.clip_percentile)
                if self.use_gpu:
                    img_tensor = torch.clamp(img_tensor, 0, percentile_val)
                else:
                    img_data = np.clip(img_data, 0, percentile_val)
            
            # Intensity normalization (GPU accelerated)
            if self.use_gpu:
                img_normalized = self._normalize_intensity_gpu(img_tensor, brain_mask)
                img_normalized = img_normalized.cpu().numpy()
            else:
                img_normalized = self._normalize_intensity(img_data, brain_mask)
            
            # Calculate statistics
            voxel_volume_ml = np.prod(img_nib.header.get_zooms()[:3]) / 1000.0
            lesion_volume_ml = float(np.sum(mask_data) * voxel_volume_ml)
            
            # Prepare metadata - convert all numpy types to native Python types
            metadata = {
                'case_id': case_id,
                'original_image_path': str(image_path),
                'original_mask_path': str(mask_path),
                'shape': tuple(int(x) for x in img_data.shape),
                'spacing': tuple(float(x) for x in img_nib.header.get_zooms()[:3]),
                'lesion_volume_ml': float(lesion_volume_ml),
                'lesion_voxels': int(np.sum(mask_data)),
                'has_lesion': bool(lesion_volume_ml > 0),
                'preprocessing': 'minimal_mni_space_gpu' if self.use_gpu else 'minimal_mni_space',
                'normalization_method': self.normalization_method
            }
            
            # Save as npz
            npz_path = output_dir / f'{case_id}.npz'
            np.savez_compressed(
                npz_path,
                image=img_normalized.astype(np.float32),
                lesion_mask=mask_data.astype(np.uint8),
                brain_mask=brain_mask.astype(np.uint8)
            )
            
            # Save metadata
            pkl_path = output_dir / f'{case_id}_metadata.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            metadata['npz_path'] = str(npz_path)
            metadata['pkl_path'] = str(pkl_path)
            
            return metadata
            
        except Exception as e:
            print(f"  ✗ Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _normalize_intensity(
        self,
        data: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Z-score normalization within brain (CPU)"""
        output = np.zeros_like(data)
        brain_voxels = data[mask]
        
        if len(brain_voxels) > 0:
            mean = brain_voxels.mean()
            std = brain_voxels.std()
            if std > 0:
                output[mask] = (data[mask] - mean) / std
        
        return output
    
    def _normalize_intensity_gpu(
        self,
        data: torch.Tensor,
        mask: np.ndarray
    ) -> torch.Tensor:
        """Z-score normalization within brain (GPU)"""
        output = torch.zeros_like(data)
        mask_tensor = torch.from_numpy(mask).to(self.device)
        
        brain_voxels = data[mask_tensor]
        
        if brain_voxels.numel() > 0:
            mean = brain_voxels.mean()
            std = brain_voxels.std()
            if std > 0:
                output[mask_tensor] = (data[mask_tensor] - mean) / std
        
        return output



def find_image_mask_pairs(data_dir: str, image_pattern: str, mask_pattern: str) -> list:
    """
    Find matching image-mask pairs from patterns
    Special handling for ISLES2018 where numeric IDs differ between modalities
    """
    data_dir = Path(data_dir)
    
    # Find all images
    image_files = sorted(glob(str(data_dir / image_pattern), recursive=True))
    
    if not image_files:
        print(f"⚠️  No images found with pattern: {image_pattern}")
        return []
    
    pairs = []
    
    for img_path in image_files:
        img_path = Path(img_path)
        mask_path = None
        identifier = None
        
        # Extract identifier based on dataset structure
        if 'ATLAS' in str(data_dir) or 'sub-r' in img_path.name:
            # ATLAS: sub-r001s001_ses-1_...
            identifier = img_path.name.split('_space-')[0]
            
            # Find corresponding mask
            mask_candidates = glob(str(data_dir / mask_pattern), recursive=True)
            for mask_candidate in mask_candidates:
                if identifier in mask_candidate:
                    mask_path = mask_candidate
                    break
            
        elif 'UOA' in str(data_dir) or 'FNIRT' in str(data_dir):
            # UOA: X075_T1_FNIRT_MNI.nii.gz -> X075
            identifier = img_path.name.split('_T1_')[0]
            
            # Find corresponding mask
            mask_candidates = glob(str(data_dir / mask_pattern), recursive=True)
            for mask_candidate in mask_candidates:
                if identifier in mask_candidate:
                    mask_path = mask_candidate
                    break
            
        elif 'ISLES2018' in str(data_dir):
            # ISLES2018: Match by case folder, NOT by numeric ID
            # Structure: case_X/SMIR.*.CT_MTT.XXXXXX/SMIR.*.CT_MTT.XXXXXX.nii
            # Mask is:   case_X/SMIR.*.OT.YYYYYY/SMIR.*.OT.YYYYYY.nii
            # where XXXXXX != YYYYYY (different IDs!)
            
            # Get the case folder (e.g., case_1, case_10)
            case_folder = img_path.parent.parent  # Go up two levels to case_X/
            identifier = case_folder.name  # e.g., "case_1"
            
            # Find OT mask in the same case folder
            mask_candidates = list(case_folder.glob('SMIR.*.OT.*/SMIR.*.OT.*.nii'))
            
            if mask_candidates:
                mask_path = str(mask_candidates[0])  # Should be only one OT per case
            
        elif 'ISLES2022' in str(data_dir):
            # ISLES2022: sub-strokecase0001/sub-strokecase0001_ses-0001_dwi.nii.gz
            identifier = img_path.parent.name
            
            # Find corresponding mask
            mask_candidates = glob(str(data_dir / mask_pattern), recursive=True)
            for mask_candidate in mask_candidates:
                if identifier in mask_candidate:
                    mask_path = mask_candidate
                    break
        
        else:
            # Generic: use stem without extensions
            identifier = img_path.stem.split('.')[0]
            
            # Find corresponding mask
            mask_candidates = glob(str(data_dir / mask_pattern), recursive=True)
            for mask_candidate in mask_candidates:
                if identifier in mask_candidate:
                    mask_path = mask_candidate
                    break
        
        if mask_path:
            pairs.append({
                'case_id': identifier,
                'image': str(img_path),
                'mask': mask_path
            })
        else:
            if identifier:
                print(f"⚠️  No mask found for case: {identifier}")
            else:
                print(f"⚠️  Could not extract identifier from: {img_path}")
    
    return pairs


def find_image_mask_pairs1(data_dir: str, image_pattern: str, mask_pattern: str) -> list:
    """
    Find matching image-mask pairs from patterns
    """
    data_dir = Path(data_dir)
    
    # Find all images
    image_files = sorted(glob(str(data_dir / image_pattern), recursive=True))
    
    pairs = []
    
    for img_path in image_files:
        img_path = Path(img_path)
        
        # Extract identifier based on dataset structure
        if 'ATLAS' in str(data_dir) or 'sub-r' in img_path.name:
            # ATLAS: sub-r001s001_ses-1_...
            identifier = img_path.name.split('_space-')[0]
            
        elif 'UOA' in str(data_dir) or 'FNIRT' in str(data_dir):
            # UOA: X075_T1_FNIRT_MNI.nii.gz -> X075
            identifier = img_path.name.split('_T1_')[0]
            
        elif 'ISLES2018' in str(data_dir):
            # ISLES2018: case_1/SMIR...
            identifier = img_path.parent.name  # case_1, case_2, etc.
            
        elif 'ISLES2022' in str(data_dir):
            # ISLES2022: sub-strokecase0001
            identifier = img_path.parent.name
        
        else:
            # Generic: use stem without extensions
            identifier = img_path.stem.split('.')[0]
        
        # Find corresponding mask
        mask_candidates = glob(str(data_dir / mask_pattern), recursive=True)
        
        mask_path = None
        for mask_candidate in mask_candidates:
            if identifier in mask_candidate:
                mask_path = mask_candidate
                break
        
        if mask_path:
            pairs.append({
                'case_id': identifier,
                'image': str(img_path),
                'mask': mask_path
            })
        else:
            print(f"⚠️  No mask found for: {img_path.name}")
    
    return pairs


def process_dataset(config_path: str):
    """Process all datasets from configuration"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_base_dir = Path(config['output_base_dir'])
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract only valid preprocessing config parameters
    preprocessing_config = {
        'target_spacing': tuple(config['preprocessing'].get('target_spacing', [1.0, 1.0, 1.0])),
        'normalization_method': config['preprocessing'].get('normalization_method', 'zscore'),
        'clip_percentile': config['preprocessing'].get('clip_percentile', 99.5),
        'use_skull_strip': config['preprocessing'].get('use_skull_strip', False),
        'skip_registration': config['preprocessing'].get('skip_registration', True),
        'use_gpu': config['preprocessing'].get('use_gpu', True)
    }
    
    preprocessor = StrokeFoundationPreprocessor(**preprocessing_config)
    
    all_summaries = {}
    
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        data_dir = dataset_config['data_dir']
        image_pattern = dataset_config['image_pattern']
        mask_pattern = dataset_config['mask_pattern']
        
        print("\n" + "="*70)
        print(f"Processing: {dataset_name}")
        print(f"Directory: {data_dir}")
        print("="*70)
        
        # Find image-mask pairs
        print("\nFinding image-mask pairs...")
        pairs = find_image_mask_pairs(data_dir, image_pattern, mask_pattern)
        
        if not pairs:
            print(f"⚠️  No valid pairs found for {dataset_name}")
            continue
        
        print(f"✓ Found {len(pairs)} cases\n")
        
        # Create output directory for this dataset
        dataset_output_dir = output_base_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each case
        metadata_list = []
        
        for pair in tqdm(pairs, desc=f"Processing {dataset_name}"):
            metadata = preprocessor.preprocess_case(
                image_path=pair['image'],
                mask_path=pair['mask'],
                output_dir=dataset_output_dir,
                case_id=pair['case_id']
            )
            
            if metadata:
                metadata_list.append(metadata)
        
        # Create summary - convert all numpy types to native Python types
        lesion_volumes = [m['lesion_volume_ml'] for m in metadata_list]
        
        summary = {
            'dataset_name': dataset_name,
            'total_cases': len(pairs),
            'successful_cases': len(metadata_list),
            'failed_cases': len(pairs) - len(metadata_list),
            'lesion_statistics': {
                'mean_volume_ml': float(np.mean(lesion_volumes)),
                'median_volume_ml': float(np.median(lesion_volumes)),
                'std_volume_ml': float(np.std(lesion_volumes)),
                'min_volume_ml': float(np.min(lesion_volumes)),
                'max_volume_ml': float(np.max(lesion_volumes)),
                'total_volume_ml': float(np.sum(lesion_volumes))
            },
            'case_metadata': metadata_list
        }
        
        # Save dataset summary using custom encoder
        summary_path = dataset_output_dir / f'{dataset_name}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        all_summaries[dataset_name] = summary
        
        print(f"\n{'='*70}")
        print(f"✓ {dataset_name} complete!")
        print(f"  Processed: {len(metadata_list)}/{len(pairs)}")
        print(f"  Mean lesion volume: {summary['lesion_statistics']['mean_volume_ml']:.2f} ml")
        print(f"  Output: {dataset_output_dir}")
        print(f"{'='*70}\n")
    
    # Overall summary
    overall_summary = {
        'total_datasets': len(config['datasets']),
        'processed_datasets': len(all_summaries),
        'total_cases': sum(s['successful_cases'] for s in all_summaries.values()),
        'preprocessing_config': preprocessing_config,
        'dataset_summaries': all_summaries
    }
    
    overall_summary_path = output_base_dir / 'overall_summary.json'
    with open(overall_summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "="*70)
    print("ALL DATASETS PROCESSED!")
    print("="*70)
    print(f"Total cases: {overall_summary['total_cases']}")
    print(f"Summary: {overall_summary_path}\n")
    
    # Print per-dataset statistics
    print("Per-dataset statistics:")
    for name, summary in all_summaries.items():
        print(f"\n{name}:")
        print(f"  Cases: {summary['successful_cases']}")
        print(f"  Mean lesion: {summary['lesion_statistics']['mean_volume_ml']:.2f} ml")
        print(f"  Median lesion: {summary['lesion_statistics']['median_volume_ml']:.2f} ml")
        print(f"  Min lesion: {summary['lesion_statistics']['min_volume_ml']:.2f} ml")
        print(f"  Max lesion: {summary['lesion_statistics']['max_volume_ml']:.2f} ml")
        print(f"  Total lesion volume: {summary['lesion_statistics']['total_volume_ml']:.2f} ml")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess stroke datasets for foundation model (GPU accelerated)'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to dataset configuration JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    process_dataset(args.config)


if __name__ == '__main__':
    main()

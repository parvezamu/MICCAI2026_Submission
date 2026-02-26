"""
reprocess_isles_to_atlas_size.py

Reprocess ISLES2022 to match ATLAS dimensions (197, 233, 189)
Uses scipy for high-quality resampling

Author: Parvez
Date: January 2026
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import scipy.ndimage as ndi
import pickle
import json


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


def resample_to_target_size(data, target_size=(197, 233, 189), order=1):
    """
    Resample data to target size using scipy zoom
    
    Args:
        data: Input 3D array
        target_size: Target dimensions (D, H, W)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
               Use 1 for images, 0 for masks
    
    Returns:
        Resampled array
    """
    current_size = np.array(data.shape)
    target_size = np.array(target_size)
    
    # Calculate zoom factors
    zoom_factors = target_size / current_size
    
    # Resample
    resampled = ndi.zoom(data, zoom_factors, order=order)
    
    return resampled


def normalize_intensity(data, mask):
    """
    Z-score normalization within brain mask
    
    Args:
        data: 3D image array (D, H, W)
        mask: 3D binary mask (D, H, W) - must be same shape as data
    
    Returns:
        Normalized 3D array
    """
    # Ensure mask is boolean and same shape
    assert data.shape == mask.shape, f"Shape mismatch: data {data.shape} vs mask {mask.shape}"
    mask = mask.astype(bool)
    
    output = np.zeros_like(data)
    brain_voxels = data[mask]
    
    if len(brain_voxels) > 0:
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        if std > 0:
            output[mask] = (data[mask] - mean) / std
    
    return output


def reprocess_isles_case(
    image_path,
    mask_path,
    case_id,
    output_dir,
    target_size=(197, 233, 189),
    clip_percentile=99.5
):
    """
    Reprocess single ISLES case to ATLAS dimensions
    
    Args:
        image_path: Path to DWI image
        mask_path: Path to lesion mask
        case_id: Case identifier
        output_dir: Output directory
        target_size: Target dimensions to match ATLAS
        clip_percentile: Percentile for intensity clipping
    
    Returns:
        Metadata dict or None if failed
    """
    try:
        # Load image
        img_nib = nib.load(image_path)
        img_data = img_nib.get_fdata().astype(np.float32)
        
        # Load mask
        mask_nib = nib.load(mask_path)
        mask_data = mask_nib.get_fdata().astype(np.float32)
        mask_data = (mask_data > 0.5).astype(np.uint8)
        
        # Check alignment
        if img_data.shape != mask_data.shape:
            print(f"  ⚠️  Shape mismatch: img {img_data.shape} vs mask {mask_data.shape}")
            return None
        
        original_shape = img_data.shape
        
        # Create brain mask (simple threshold)
        brain_mask = (img_data > np.percentile(img_data, 1)).astype(np.uint8)
        
        # Intensity clipping BEFORE resampling
        brain_voxels = img_data[brain_mask > 0]
        if len(brain_voxels) > 0:
            percentile_val = np.percentile(brain_voxels, clip_percentile)
            img_data = np.clip(img_data, 0, percentile_val)
        
        # Resample to ATLAS size
        # print(f"    Resampling from {original_shape} to {target_size}...")
        
        # Resample image (linear interpolation)
        img_resampled = resample_to_target_size(img_data, target_size, order=1)
        
        # Resample mask (nearest neighbor to preserve binary values)
        mask_resampled = resample_to_target_size(mask_data.astype(np.float32), target_size, order=0)
        mask_resampled = (mask_resampled > 0.5).astype(np.uint8)
        
        # Resample brain mask (nearest neighbor)
        brain_mask_resampled = resample_to_target_size(brain_mask.astype(np.float32), target_size, order=0)
        brain_mask_resampled = (brain_mask_resampled > 0.5).astype(np.uint8)
        
        # Verify shapes before normalization
        assert img_resampled.shape == target_size, f"Image shape mismatch: {img_resampled.shape} vs {target_size}"
        assert mask_resampled.shape == target_size, f"Mask shape mismatch: {mask_resampled.shape} vs {target_size}"
        assert brain_mask_resampled.shape == target_size, f"Brain mask shape mismatch: {brain_mask_resampled.shape} vs {target_size}"
        
        # Normalize intensity AFTER resampling
        img_normalized = normalize_intensity(img_resampled, brain_mask_resampled)
        
        # Calculate statistics
        # Note: We need to adjust voxel volume for the resampled data
        # Original voxel volume
        original_spacing = img_nib.header.get_zooms()[:3]
        original_voxel_volume_ml = np.prod(original_spacing) / 1000.0
        
        # New effective spacing (approximate)
        zoom_factors = np.array(target_size) / np.array(original_shape)
        new_spacing = np.array(original_spacing) / zoom_factors
        new_voxel_volume_ml = np.prod(new_spacing) / 1000.0
        
        lesion_volume_ml = float(np.sum(mask_resampled) * new_voxel_volume_ml)
        
        # Prepare metadata
        metadata = {
            'case_id': case_id,
            'original_image_path': str(image_path),
            'original_mask_path': str(mask_path),
            'original_shape': tuple(int(x) for x in original_shape),
            'resampled_shape': tuple(int(x) for x in target_size),
            'original_spacing': tuple(float(x) for x in original_spacing),
            'resampled_spacing': tuple(float(x) for x in new_spacing),
            'lesion_volume_ml': float(lesion_volume_ml),
            'lesion_voxels': int(np.sum(mask_resampled)),
            'has_lesion': bool(lesion_volume_ml > 0),
            'preprocessing': 'resampled_to_atlas_size',
            'normalization_method': 'zscore'
        }
        
        # Save as npz
        npz_path = output_dir / f'{case_id}.npz'
        np.savez_compressed(
            npz_path,
            image=img_normalized.astype(np.float32),
            lesion_mask=mask_resampled.astype(np.uint8),
            brain_mask=brain_mask_resampled.astype(np.uint8)
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


def find_isles_pairs(isles_raw_dir):
    """
    Find ISLES2022 image-mask pairs
    
    Expected structure:
    ISLES2022/
      sub-strokecase0001/
        sub-strokecase0001_ses-0001_dwi.nii.gz
        sub-strokecase0001_ses-0001_msk.nii.gz
    """
    from glob import glob
    
    isles_dir = Path(isles_raw_dir)
    
    # Find all DWI images
    image_pattern = "sub-strokecase*/sub-strokecase*_ses-*_dwi.nii.gz"
    image_files = sorted(glob(str(isles_dir / image_pattern), recursive=True))
    
    pairs = []
    
    for img_path in image_files:
        img_path = Path(img_path)
        
        # Extract case ID (e.g., sub-strokecase0001)
        case_id = img_path.parent.name
        
        # Find corresponding mask
        mask_pattern = f"{case_id}_ses-*_msk.nii.gz"
        mask_candidates = list(img_path.parent.glob(mask_pattern))
        
        if mask_candidates:
            pairs.append({
                'case_id': case_id,
                'image': str(img_path),
                'mask': str(mask_candidates[0])
            })
        else:
            print(f"⚠️  No mask found for: {case_id}")
    
    return pairs


def main():
    """Main reprocessing function"""
    
    # Configuration
    ISLES_RAW_DIR = "/home/pahm409/ISLES2022_reg/ISLES2022/"  # Update this path
    OUTPUT_DIR = "/home/pahm409/preprocessed_stroke_foundation/ISLES2022_resampled"
    TARGET_SIZE = (197, 233, 189)  # ATLAS dimensions
    
    print("="*80)
    print("REPROCESSING ISLES2022 TO MATCH ATLAS DIMENSIONS")
    print("="*80)
    print(f"Input directory:  {ISLES_RAW_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size:      {TARGET_SIZE}")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image-mask pairs
    print("Finding ISLES2022 cases...")
    pairs = find_isles_pairs(ISLES_RAW_DIR)
    
    if not pairs:
        print("❌ No ISLES2022 cases found!")
        print(f"\nExpected structure in {ISLES_RAW_DIR}:")
        print("  sub-strokecase0001/")
        print("    sub-strokecase0001_ses-0001_dwi.nii.gz")
        print("    sub-strokecase0001_ses-0001_msk.nii.gz")
        return
    
    print(f"✓ Found {len(pairs)} cases\n")
    
    # Process each case
    metadata_list = []
    
    for pair in tqdm(pairs, desc="Reprocessing ISLES2022"):
        metadata = reprocess_isles_case(
            image_path=pair['image'],
            mask_path=pair['mask'],
            case_id=pair['case_id'],
            output_dir=output_dir,
            target_size=TARGET_SIZE
        )
        
        if metadata:
            metadata_list.append(metadata)
    
    # Create summary
    if metadata_list:
        lesion_volumes = [m['lesion_volume_ml'] for m in metadata_list]
        
        summary = {
            'dataset_name': 'ISLES2022_resampled',
            'total_cases': len(pairs),
            'successful_cases': len(metadata_list),
            'failed_cases': len(pairs) - len(metadata_list),
            'target_size': TARGET_SIZE,
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
        
        # Save summary
        summary_path = output_dir / 'ISLES2022_resampled_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print("\n" + "="*80)
        print("✓ REPROCESSING COMPLETE!")
        print("="*80)
        print(f"Processed: {len(metadata_list)}/{len(pairs)} cases")
        print(f"Output directory: {output_dir}")
        print(f"Summary saved: {summary_path}")
        print("\nLesion statistics:")
        print(f"  Mean volume: {summary['lesion_statistics']['mean_volume_ml']:.2f} ml")
        print(f"  Median volume: {summary['lesion_statistics']['median_volume_ml']:.2f} ml")
        print(f"  Min volume: {summary['lesion_statistics']['min_volume_ml']:.2f} ml")
        print(f"  Max volume: {summary['lesion_statistics']['max_volume_ml']:.2f} ml")
        print("="*80)
        
        # Verify a few volumes
        print("\n" + "="*80)
        print("VERIFICATION: Checking dimensions of processed volumes")
        print("="*80)
        
        for i in range(min(5, len(metadata_list))):
            case_id = metadata_list[i]['case_id']
            npz_path = output_dir / f'{case_id}.npz'
            
            data = np.load(npz_path)
            img_shape = data['image'].shape
            mask_shape = data['lesion_mask'].shape
            
            print(f"{case_id}:")
            print(f"  Image shape: {img_shape}")
            print(f"  Mask shape:  {mask_shape}")
            print(f"  Match target: {img_shape == TARGET_SIZE}")
    
    else:
        print("\n❌ No cases were successfully processed!")


if __name__ == '__main__':
    main()

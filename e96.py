"""
extract_patches_96x96x96.py

Extract 96x96x96 lesion-focused patches + centers from existing
preprocessed_stroke_foundation full-volume NPZs.

Input NPZ keys:  image, lesion_mask, brain_mask
Output NPZ keys: patches, masks, centers, original_shape, spacing,
                 lesion_volume, patient_id

Usage:
    python extract_patches_96x96x96.py \
        --input_dir  /hpc/pahm409/harvard/preprocessed_stroke_foundation \
        --output_dir /hpc/pahm409/harvard/preprocessed_stroke_foundation \
        --datasets   ATLAS UOA_Private \
        --patch_size 96 \
        --max_patches 20 \
        --min_lesion_overlap 0.01

Author: Parvez
Date:   February 2026
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import nibabel as nib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_patches_96.log"),
        logging.StreamHandler()
    ]
)


# ============================================================
# PATCH EXTRACTION
# ============================================================

def get_lesion_focused_centers(
    lesion_mask,        # (D, H, W) uint8
    original_shape,     # (D, H, W)
    patch_size,         # int  e.g. 96
    max_patches,        # int  max patches per volume
    min_lesion_overlap, # float  min fraction of patch that must be lesion
    rng,                # np.random.Generator
):
    """
    Sample patch centers from lesion voxels.

    Strategy:
      1. Collect all lesion voxel coordinates.
      2. Randomly sample up to max_patches of them as candidate centers.
      3. Clamp each center so the 96³ patch stays within volume bounds.
      4. Filter: keep only patches where lesion coverage >= min_lesion_overlap.
      5. De-duplicate (remove near-identical centers after clamping).

    Returns:
        centers  – np.ndarray (N, 3) int32, midpoint centers [d, h, w]
                   Each patch spans [c-half : c+half] on all axes.
    """
    half = patch_size // 2
    D, H, W = original_shape

    # All lesion voxel coords
    lesion_coords = np.argwhere(lesion_mask > 0)  # (L, 3)

    if len(lesion_coords) == 0:
        logging.warning("    No lesion voxels found — skipping lesion-focused sampling")
        return np.empty((0, 3), dtype=np.int32)

    # Sample candidate centers from lesion voxels
    n_candidates = min(max_patches * 5, len(lesion_coords))
    chosen_idx   = rng.choice(len(lesion_coords), size=n_candidates, replace=False)
    candidates   = lesion_coords[chosen_idx].astype(np.int32)  # (n_candidates, 3)

    # Clamp so patch stays within volume
    candidates[:, 0] = np.clip(candidates[:, 0], half, D - half)
    candidates[:, 1] = np.clip(candidates[:, 1], half, H - half)
    candidates[:, 2] = np.clip(candidates[:, 2], half, W - half)

    # De-duplicate after clamping
    candidates = np.unique(candidates, axis=0)

    # Filter by lesion overlap
    valid_centers = []
    for c in candidates:
        d0, d1 = c[0] - half, c[0] + half
        h0, h1 = c[1] - half, c[1] + half
        w0, w1 = c[2] - half, c[2] + half

        patch_lesion = lesion_mask[d0:d1, h0:h1, w0:w1]
        patch_voxels = patch_size ** 3
        overlap      = patch_lesion.sum() / patch_voxels

        if overlap >= min_lesion_overlap:
            valid_centers.append(c)

        if len(valid_centers) >= max_patches:
            break

    if len(valid_centers) == 0:
        # Fallback: use lesion centroid as single patch
        centroid = lesion_coords.mean(axis=0).astype(np.int32)
        centroid[0] = np.clip(centroid[0], half, D - half)
        centroid[1] = np.clip(centroid[1], half, H - half)
        centroid[2] = np.clip(centroid[2], half, W - half)
        valid_centers = [centroid]
        logging.warning("    No patches passed overlap filter — using lesion centroid")

    return np.array(valid_centers, dtype=np.int32)


def extract_patches_for_case(
    image,              # (D, H, W) float32  normalised image
    lesion_mask,        # (D, H, W) uint8
    centers,            # (N, 3) int32
    patch_size,         # int
):
    """
    Extract image patches and mask patches at the given centers.

    Returns:
        patches  – (N, ps, ps, ps) float32
        masks    – (N, ps, ps, ps) uint8
    """
    half    = patch_size // 2
    patches = []
    masks   = []

    for c in centers:
        d0, d1 = c[0] - half, c[0] + half
        h0, h1 = c[1] - half, c[1] + half
        w0, w1 = c[2] - half, c[2] + half

        patches.append(image[d0:d1, h0:h1, w0:w1].copy())
        masks.append(lesion_mask[d0:d1, h0:h1, w0:w1].copy())

    return np.stack(patches).astype(np.float32), np.stack(masks).astype(np.uint8)


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    input_dataset_dir,   # Path  e.g. .../preprocessed_stroke_foundation/ATLAS
    output_dataset_dir,  # Path  e.g. .../ATLAS_96x96x96
    patch_size,
    max_patches,
    min_lesion_overlap,
    seed=42,
    splits_file=None,    # NEW: path to splits JSON
    fold=None,           # NEW: fold number
    split='all',         # NEW: 'all', 'train', 'val', 'test'
):
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load splits if provided
    case_ids_to_process = None
    if splits_file and fold is not None and split != 'all':
        import json
        with open(splits_file) as f:
            splits = json.load(f)
        dataset_key = input_dataset_dir.name  # e.g. 'ATLAS' or 'UOA_Private'
        fold_key = f'fold_{fold}'
        if fold_key in splits and dataset_key in splits[fold_key]:
            case_ids_to_process = set(splits[fold_key][dataset_key][split])
            logging.info(f"  Processing {split} split: {len(case_ids_to_process)} cases")
        else:
            logging.warning(f"  Split {split} not found in splits file for {dataset_key} fold {fold}")

    # Find all case NPZs
    npz_files = sorted(input_dataset_dir.glob("*.npz"))
    
    # Filter by split if specified
    if case_ids_to_process is not None:
        npz_files = [p for p in npz_files if p.stem in case_ids_to_process]
    
    logging.info(f"  Found {len(npz_files)} NPZ files to process in {input_dataset_dir.name}")

    rng = np.random.default_rng(seed)

    summary_records = []
    patient_ids     = []
    skipped         = 0

    for npz_path in npz_files:
        case_id = npz_path.stem          # e.g. sub-r001s001_ses-1  or  A207
        t0      = time.time()

        try:
            data = np.load(npz_path)

            # Validate keys
            if 'image' not in data.files:
                logging.warning(f"  SKIP {case_id}: no 'image' key")
                skipped += 1
                continue
            if 'lesion_mask' not in data.files and 'mask' not in data.files:
                logging.warning(f"  SKIP {case_id}: no mask key")
                skipped += 1
                continue

            image       = data['image'].astype(np.float32)          # (D,H,W)
            lesion_mask = (
                data['lesion_mask'] if 'lesion_mask' in data.files
                else data['mask']
            ).astype(np.uint8)

            original_shape = np.array(image.shape, dtype=np.int32)  # (3,)
            D, H, W        = image.shape

            # Skip volumes too small for even one patch
            half = patch_size // 2
            if D < patch_size or H < patch_size or W < patch_size:
                logging.warning(
                    f"  SKIP {case_id}: volume {image.shape} smaller than patch {patch_size}³"
                )
                skipped += 1
                continue

            # Retrieve spacing if stored (else default 1mm iso)
            spacing = (
                tuple(float(x) for x in data['spacing'])
                if 'spacing' in data.files
                else (1.0, 1.0, 1.0)
            )

            # Lesion volume (ml)
            voxel_vol_ml  = float(np.prod(spacing)) / 1000.0
            lesion_volume = float(lesion_mask.sum()) * voxel_vol_ml

            # Sample centers
            centers = get_lesion_focused_centers(
                lesion_mask, (D, H, W),
                patch_size, max_patches, min_lesion_overlap, rng
            )

            if len(centers) == 0:
                logging.warning(f"  SKIP {case_id}: could not extract any patches")
                skipped += 1
                continue

            # Extract patches
            patches, masks = extract_patches_for_case(
                image, lesion_mask, centers, patch_size
            )

            # Save output NPZ
            out_npz = output_dataset_dir / f"{case_id}.npz"
            np.savez_compressed(
                out_npz,
                patches        = patches,           # (N, 96, 96, 96) float32
                masks          = masks,             # (N, 96, 96, 96) uint8
                centers        = centers,           # (N, 3)          int32   midpoints
                original_shape = original_shape,   # (3,)            int32
                spacing        = np.array(spacing, dtype=np.float32),
                lesion_volume  = np.float32(lesion_volume),
                patient_id     = np.array(case_id),
            )

            patient_ids.append(case_id)

            elapsed = time.time() - t0
            logging.info(
                f"  {case_id}: {len(patches)} patches  "
                f"shape={image.shape}  lesion={lesion_volume:.2f}ml  {elapsed:.1f}s"
            )

            summary_records.append({
                'case_id':        case_id,
                'n_patches':      int(len(patches)),
                'original_shape': original_shape.tolist(),
                'spacing':        list(spacing),
                'lesion_volume_ml': float(lesion_volume),
                'lesion_voxels':  int(lesion_mask.sum()),
            })

        except Exception as e:
            logging.error(f"  ERROR {case_id}: {e}")
            import traceback; traceback.print_exc()
            skipped += 1
            continue

    # Save patient_ids.pkl  (for ablation script compatibility)
    with open(output_dataset_dir / 'patient_ids.pkl', 'wb') as f:
        pickle.dump(patient_ids, f)

    # Save dataset summary JSON
    summary = {
        'dataset':          input_dataset_dir.name,
        'patch_size':       patch_size,
        'max_patches':      max_patches,
        'min_lesion_overlap': min_lesion_overlap,
        'total_cases':      len(npz_files),
        'processed_cases':  len(patient_ids),
        'skipped_cases':    skipped,
        'total_patches':    sum(r['n_patches'] for r in summary_records),
        'cases':            summary_records,
    }
    with open(output_dataset_dir / f'{input_dataset_dir.name}_96_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"\n  Done {input_dataset_dir.name}: "
        f"{len(patient_ids)} processed, {skipped} skipped, "
        f"{summary['total_patches']} total patches"
    )

    return patient_ids, summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract 96x96x96 lesion-focused patches from stroke_foundation NPZs'
    )
    parser.add_argument(
        '--input_dir', type=str,
        default='/hpc/pahm409/harvard/preprocessed_stroke_foundation',
        help='Root of preprocessed_stroke_foundation (contains ATLAS/, UOA_Private/)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/hpc/pahm409/harvard/preprocessed_stroke_foundation',
        help='Root where ATLAS_96x96x96/ and UOA_Private_96x96x96/ will be created'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        default=['ATLAS', 'UOA_Private'],
        help='Dataset subdirectory names to process'
    )
    parser.add_argument(
        '--patch_size', type=int, default=96,
        help='Patch size (isotropic, default: 96)'
    )
    parser.add_argument(
        '--max_patches', type=int, default=20,
        help='Max patches per volume (default: 20)'
    )
    parser.add_argument(
        '--min_lesion_overlap', type=float, default=0.01,
        help='Min fraction of patch that must contain lesion (default: 0.01)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--splits_file', type=str, default=None,
        help='Path to splits JSON (e.g. splits_5fold.json) - optional'
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help='Fold number (0-4) - required if --split is specified'
    )
    parser.add_argument(
        '--split', type=str, default='all', choices=['all', 'train', 'val', 'test'],
        help='Which split to process: all (default), train, val, or test'
    )
    args = parser.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    logging.info("=" * 70)
    logging.info("EXTRACTING 96x96x96 LESION-FOCUSED PATCHES")
    logging.info(f"  input_dir:           {input_root}")
    logging.info(f"  output_dir:          {output_root}")
    logging.info(f"  datasets:            {args.datasets}")
    logging.info(f"  patch_size:          {args.patch_size}³")
    logging.info(f"  max_patches:         {args.max_patches}")
    logging.info(f"  min_lesion_overlap:  {args.min_lesion_overlap}")
    logging.info(f"  seed:                {args.seed}")
    if args.splits_file:
        logging.info(f"  splits_file:         {args.splits_file}")
        logging.info(f"  fold:                {args.fold}")
        logging.info(f"  split:               {args.split}")
    logging.info("=" * 70)

    all_summaries = {}

    for dataset_name in args.datasets:
        input_dataset_dir  = input_root  / dataset_name
        
        # Output dir naming: if extracting specific split, append to dir name
        if args.split != 'all' and args.fold is not None:
            output_dataset_dir = output_root / f"{dataset_name}_96x96x96_fold{args.fold}_{args.split}"
        else:
            output_dataset_dir = output_root / f"{dataset_name}_96x96x96"

        if not input_dataset_dir.exists():
            logging.error(f"Input dir not found: {input_dataset_dir}")
            continue

        logging.info(f"\n{'='*70}")
        logging.info(f"Processing: {dataset_name}")
        logging.info(f"  Input:  {input_dataset_dir}")
        logging.info(f"  Output: {output_dataset_dir}")
        logging.info(f"{'='*70}")

        patient_ids, summary = process_dataset(
            input_dataset_dir  = input_dataset_dir,
            output_dataset_dir = output_dataset_dir,
            patch_size         = args.patch_size,
            max_patches        = args.max_patches,
            min_lesion_overlap = args.min_lesion_overlap,
            seed               = args.seed,
            splits_file        = args.splits_file,
            fold               = args.fold,
            split              = args.split,
        )

        all_summaries[dataset_name] = summary

    # Overall summary
    overall = {
        'patch_size':    args.patch_size,
        'max_patches':   args.max_patches,
        'split':         args.split,
        'fold':          args.fold,
        'datasets':      all_summaries,
        'output_dirs': {
            ds: str(output_root / (f"{ds}_96x96x96_fold{args.fold}_{args.split}" 
                                   if args.split != 'all' and args.fold is not None 
                                   else f"{ds}_96x96x96"))
            for ds in args.datasets
        }
    }
    
    summary_filename = (f'patches_96_fold{args.fold}_{args.split}_summary.json' 
                       if args.split != 'all' and args.fold is not None
                       else 'patches_96_overall_summary.json')
    
    with open(output_root / summary_filename, 'w') as f:
        json.dump(overall, f, indent=2)

    logging.info("\n" + "=" * 70)
    logging.info("ALL DONE")
    logging.info("=" * 70)
    for ds, summ in all_summaries.items():
        logging.info(
            f"  {ds}_96x96x96: "
            f"{summ['processed_cases']} cases, "
            f"{summ['total_patches']} patches"
        )
    logging.info(f"\nOutput structure:")
    for ds in args.datasets:
        if args.split != 'all' and args.fold is not None:
            out = output_root / f"{ds}_96x96x96_fold{args.fold}_{args.split}"
        else:
            out = output_root / f"{ds}_96x96x96"
        logging.info(f"  {out}/")
        logging.info(f"    <case_id>.npz  [patches, masks, centers, original_shape, ...]")
        logging.info(f"    patient_ids.pkl")
        logging.info(f"    {ds}_96_summary.json")


if __name__ == '__main__':
    main()

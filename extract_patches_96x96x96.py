"""
extract_patches_64x64x64.py

Extract 64x64x64 lesion-focused patches + centers from existing
preprocessed_stroke_foundation full-volume NPZs.

Input NPZ keys:  image, lesion_mask (or mask), brain_mask
Output NPZ keys: patches, masks, centers, original_shape, spacing,
                 lesion_volume, patient_id

Usage:
    python extract_patches_64x64x64.py \
        --input_dir  /hpc/pahm409/harvard/preprocessed_stroke_foundation \
        --output_dir /hpc/pahm409/harvard/preprocessed_stroke_foundation \
        --datasets   ATLAS UOA_Private \
        --patch_size 64 \
        --max_patches 20 \
        --min_lesion_overlap 0.0001
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_patches_64.log"),
        logging.StreamHandler()
    ]
)


# ============================================================
# PATCH CENTER SAMPLING
# ============================================================

def get_lesion_focused_centers(
    lesion_mask,
    original_shape,
    patch_size,
    max_patches,
    min_lesion_overlap,
    rng,
):
    """
    Sample patch centers from lesion voxels.

    Strategy:
      1. Collect all lesion voxel coordinates.
      2. Randomly sample up to max_patches*5 as candidates.
      3. Clamp each center so the patch stays within volume bounds.
      4. Filter: keep only patches where lesion coverage >= min_lesion_overlap.
      5. De-duplicate after clamping.

    Returns:
        centers – np.ndarray (N, 3) int32, midpoint centers [d, h, w]
                  Each patch spans [c-half : c+half] on all axes.
    """
    half = patch_size // 2
    D, H, W = original_shape

    lesion_coords = np.argwhere(lesion_mask > 0)  # (L, 3)

    if len(lesion_coords) == 0:
        logging.warning("    No lesion voxels found — skipping")
        return np.empty((0, 3), dtype=np.int32)

    n_candidates = min(max_patches * 5, len(lesion_coords))
    chosen_idx   = rng.choice(len(lesion_coords), size=n_candidates, replace=False)
    candidates   = lesion_coords[chosen_idx].astype(np.int32)

    # Clamp so patch stays within volume
    candidates[:, 0] = np.clip(candidates[:, 0], half, D - half)
    candidates[:, 1] = np.clip(candidates[:, 1], half, H - half)
    candidates[:, 2] = np.clip(candidates[:, 2], half, W - half)

    candidates = np.unique(candidates, axis=0)

    valid_centers = []
    for c in candidates:
        d0, d1 = c[0] - half, c[0] + half
        h0, h1 = c[1] - half, c[1] + half
        w0, w1 = c[2] - half, c[2] + half

        patch_lesion = lesion_mask[d0:d1, h0:h1, w0:w1]
        overlap      = patch_lesion.sum() / (patch_size ** 3)

        if overlap >= min_lesion_overlap:
            valid_centers.append(c)

        if len(valid_centers) >= max_patches:
            break

    if len(valid_centers) == 0:
        # Fallback: lesion centroid
        centroid    = lesion_coords.mean(axis=0).astype(np.int32)
        centroid[0] = np.clip(centroid[0], half, D - half)
        centroid[1] = np.clip(centroid[1], half, H - half)
        centroid[2] = np.clip(centroid[2], half, W - half)
        valid_centers = [centroid]
        logging.warning("    No patches passed overlap filter — using lesion centroid")

    return np.array(valid_centers, dtype=np.int32)


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patches_for_case(image, lesion_mask, centers, patch_size):
    """
    Extract image and mask patches at given centers.

    Returns:
        patches – (N, ps, ps, ps) float32
        masks   – (N, ps, ps, ps) uint8
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
    input_dataset_dir,
    output_dataset_dir,
    patch_size,
    max_patches,
    min_lesion_overlap,
    seed=42,
    splits_file=None,
    fold=None,
    split='all',
):
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Optionally filter to a specific split
    case_ids_to_process = None
    if splits_file is not None and fold is not None and split != 'all':
        with open(splits_file) as f:
            splits_data = json.load(f)
        dataset_key = input_dataset_dir.name
        fold_key    = f'fold_{fold}'
        if fold_key in splits_data and dataset_key in splits_data[fold_key]:
            case_ids_to_process = set(splits_data[fold_key][dataset_key][split])
            logging.info(f"  Processing {split} split: {len(case_ids_to_process)} cases")
        else:
            logging.warning(f"  Split '{split}' not found for {dataset_key} fold {fold}")

    npz_files = sorted(input_dataset_dir.glob("*.npz"))
    if case_ids_to_process is not None:
        npz_files = [p for p in npz_files if p.stem in case_ids_to_process]

    logging.info(f"  Found {len(npz_files)} NPZ files in {input_dataset_dir.name}")

    rng = np.random.default_rng(seed)
    summary_records = []
    patient_ids     = []
    skipped         = 0

    for npz_path in npz_files:
        case_id = npz_path.stem
        t0      = time.time()

        try:
            data = np.load(npz_path)

            if 'image' not in data.files:
                logging.warning(f"  SKIP {case_id}: no 'image' key")
                skipped += 1
                continue

            if 'lesion_mask' not in data.files and 'mask' not in data.files:
                logging.warning(f"  SKIP {case_id}: no mask key")
                skipped += 1
                continue

            image       = data['image'].astype(np.float32)
            lesion_mask = (
                data['lesion_mask'] if 'lesion_mask' in data.files else data['mask']
            ).astype(np.uint8)

            original_shape = np.array(image.shape, dtype=np.int32)
            D, H, W        = image.shape

            # Skip volumes too small for even one patch
            if D < patch_size or H < patch_size or W < patch_size:
                logging.warning(
                    f"  SKIP {case_id}: volume {image.shape} smaller than {patch_size}³"
                )
                skipped += 1
                continue

            spacing = (
                tuple(float(x) for x in data['spacing'])
                if 'spacing' in data.files
                else (1.0, 1.0, 1.0)
            )

            voxel_vol_ml  = float(np.prod(spacing)) / 1000.0
            lesion_volume = float(lesion_mask.sum()) * voxel_vol_ml

            centers = get_lesion_focused_centers(
                lesion_mask, (D, H, W),
                patch_size, max_patches, min_lesion_overlap, rng
            )

            if len(centers) == 0:
                logging.warning(f"  SKIP {case_id}: no valid patches")
                skipped += 1
                continue

            patches, masks = extract_patches_for_case(
                image, lesion_mask, centers, patch_size
            )

            out_npz = output_dataset_dir / f"{case_id}.npz"
            np.savez_compressed(
                out_npz,
                patches        = patches,
                masks          = masks,
                centers        = centers,
                original_shape = original_shape,
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
                'case_id':          case_id,
                'n_patches':        int(len(patches)),
                'original_shape':   original_shape.tolist(),
                'spacing':          list(spacing),
                'lesion_volume_ml': float(lesion_volume),
                'lesion_voxels':    int(lesion_mask.sum()),
            })

        except Exception as e:
            logging.error(f"  ERROR {case_id}: {e}")
            import traceback; traceback.print_exc()
            skipped += 1
            continue

    # Save patient_ids.pkl
    with open(output_dataset_dir / 'patient_ids.pkl', 'wb') as f:
        pickle.dump(patient_ids, f)

    # Save summary JSON — use the top-level json import, no local re-import
    ps_str   = f"{patch_size}x{patch_size}x{patch_size}"
    summary  = {
        'dataset':             input_dataset_dir.name,
        'patch_size':          patch_size,
        'max_patches':         max_patches,
        'min_lesion_overlap':  min_lesion_overlap,
        'total_cases':         len(npz_files),
        'processed_cases':     len(patient_ids),
        'skipped_cases':       skipped,
        'total_patches':       sum(r['n_patches'] for r in summary_records),
        'cases':               summary_records,
    }
    summary_path = output_dataset_dir / f"{input_dataset_dir.name}_{ps_str}_summary.json"
    with open(summary_path, 'w') as f:
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
        description='Extract lesion-focused patches from stroke_foundation NPZs'
    )
    parser.add_argument('--input_dir',  type=str,
                        default='/hpc/pahm409/harvard/preprocessed_stroke_foundation')
    parser.add_argument('--output_dir', type=str,
                        default='/hpc/pahm409/harvard/preprocessed_stroke_foundation')
    parser.add_argument('--datasets',   nargs='+', default=['ATLAS', 'UOA_Private'])
    parser.add_argument('--patch_size', type=int,   default=64,
                        help='Isotropic patch size (default: 64)')
    parser.add_argument('--max_patches',         type=int,   default=20)
    parser.add_argument('--min_lesion_overlap',  type=float, default=0.0001)
    parser.add_argument('--seed',                type=int,   default=42)
    parser.add_argument('--splits_file',         type=str,   default=None)
    parser.add_argument('--fold',                type=int,   default=None)
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'train', 'val', 'test'])
    args = parser.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)
    ps          = args.patch_size
    ps_str      = f"{ps}x{ps}x{ps}"

    logging.info("=" * 70)
    logging.info(f"EXTRACTING {ps_str} LESION-FOCUSED PATCHES")
    logging.info(f"  input_dir:           {input_root}")
    logging.info(f"  output_dir:          {output_root}")
    logging.info(f"  datasets:            {args.datasets}")
    logging.info(f"  patch_size:          {ps}³")
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
        input_dataset_dir = input_root / dataset_name

        if args.split != 'all' and args.fold is not None:
            out_name = f"{dataset_name}_{ps_str}_fold{args.fold}_{args.split}"
        else:
            out_name = f"{dataset_name}_{ps_str}"

        output_dataset_dir = output_root / out_name

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
            patch_size         = ps,
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
        'patch_size':  ps,
        'max_patches': args.max_patches,
        'split':       args.split,
        'fold':        args.fold,
        'datasets':    all_summaries,
        'output_dirs': {
            ds: str(output_root / (
                f"{ds}_{ps_str}_fold{args.fold}_{args.split}"
                if args.split != 'all' and args.fold is not None
                else f"{ds}_{ps_str}"
            ))
            for ds in args.datasets
        }
    }

    if args.split != 'all' and args.fold is not None:
        summary_fname = f'patches_{ps_str}_fold{args.fold}_{args.split}_summary.json'
    else:
        summary_fname = f'patches_{ps_str}_overall_summary.json'

    with open(output_root / summary_fname, 'w') as f:
        json.dump(overall, f, indent=2)

    logging.info("\n" + "=" * 70)
    logging.info("ALL DONE")
    logging.info("=" * 70)
    for ds, summ in all_summaries.items():
        logging.info(
            f"  {ds}_{ps_str}: "
            f"{summ['processed_cases']} cases, "
            f"{summ['total_patches']} total patches"
        )
    logging.info(f"\nOutput dirs:")
    for ds in args.datasets:
        out_name = (
            f"{ds}_{ps_str}_fold{args.fold}_{args.split}"
            if args.split != 'all' and args.fold is not None
            else f"{ds}_{ps_str}"
        )
        logging.info(f"  {output_root / out_name}/")
        logging.info(f"    <case_id>.npz  [patches, masks, centers, original_shape, ...]")
        logging.info(f"    patient_ids.pkl")
        logging.info(f"    {ds}_{ps_str}_summary.json")


if __name__ == '__main__':
    main()

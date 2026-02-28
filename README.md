# Towards Understanding Transfer Learning and Multimodal Training for Ischemic Stroke Lesion Segmentation: A Systematic Ablation Study

**MICCAI 2026 Submission**

> Parvez Ahmad, Benjamin Chong, Justin Fernandez, Vickie Shim, Nikola Kirilov Kasabov, Cathy M. Stinear, Winston D. Byblow, and Alan Wang
>
> Auckland Bioengineering Institute, The University of Auckland

---

## Abstract

This study investigates whether supervised BraTS 2024 pretraining and multimodal DWI+ADC joint training improve ischemic stroke lesion segmentation under lightweight architectures. We evaluate a ResNet-18–based U-Net and a transformer-enhanced UCTransNet3D across three datasets: ISLES 2022 (250 cases), ATLAS 2.0 (655 cases), and a private University of Auckland (UOA) cohort (177 cases), using five-fold cross-validation. Across all settings, BraTS 2024 pretraining did not consistently outperform training from scratch, suggesting transfer effectiveness is architecture- and configuration-dependent.

---

## Key Findings

- On **ISLES 2022**, all five training strategies achieved near-identical mean DSC (70.05%–70.52%), with scratch-trained shared encoder achieving the highest score
- On **ATLAS 2.0** and **UOA**, scratch training matched or outperformed BraTS transfer learning for both ResNet-18 and UCTransNet3D
- **DWI+ADC** multimodal inference provided only marginal gains over DWI-only segmentation
- Increasing architectural complexity (UCTransNet3D) did not improve transfer learning performance

---

## Architectures

### ResNet-18 U-Net
- 3D ResNet-18 encoder with symmetric U-Net decoder
- Four residual stages: channel dimensions 64, 128, 256, 512
- Trained with 96³ patches; sliding-window inference with 50% overlap
- Three configurations evaluated: DWI-only, separate DWI+ADC encoders, shared DWI+ADC encoder

### UCTransNet3D
- 3D CNN encoder + Multi-scale Channel Transformer + SE-enhanced decoder
- Cross-scale channel attention across L=4 transformer blocks with 4 heads
- Trained with 64³ patches (memory-constrained by multi-head attention)
- SE blocks for channel-wise recalibration in decoder

---

## Datasets

| Dataset | Modality | Cases (CV / Test) | Resolution |
|---|---|---|---|
| ISLES 2022 | DWI + ADC | 200 / 50 | Native space |
| ATLAS v2.0 | T1w MRI | 558 / 97 | 1×1×1 mm³, MNI |
| UOA Private | T1w MRI | 152 / 25 | 1×1×1 mm³, MNI |

---

## Training Strategies (ISLES 2022 Ablation)

| Exp | Strategy | Inference | Init |
|---|---|---|---|
| 1 | Separate encoders | DWI only | BraTS pretrained |
| 2 | Shared encoder | DWI+ADC (avg) | BraTS pretrained |
| 3 | Separate encoders | DWI only | Scratch |
| 4 | Shared encoder | DWI+ADC (avg) | Scratch |
| 5 | DWI-only baseline | DWI only | Scratch |

---

## Requirements

```bash
pip install torch nibabel numpy tqdm
```

For preprocessing:
```bash
pip install nibabel numpy torch tqdm
```

---

## Data Preprocessing

### ISLES 2022

ISLES 2022 uses DWI and ADC modalities in native (non-MNI) space and requires foreground cropping and resampling before training. Two scripts are provided depending on the use case.

---

**`preprocess_v2_isles.py` — Training preprocessing**

Crops each case to the brain foreground (detected from DWI), resamples to a fixed target shape, applies z-score normalisation on non-zero voxels for both DWI and ADC, and saves outputs as `.npz` files for training.

```bash
python preprocess_v2_isles.py \
  --isles-dir /path/to/ISLES2022/derivatives \
  --output-dir /path/to/preprocessed_isles_train \
  --target-shape 96 96 96
```

| Argument | Default | Description |
|---|---|---|
| `--isles-dir` | — | Path to ISLES 2022 derivatives folder |
| `--output-dir` | — | Output directory for `.npz` files |
| `--target-shape` | `96 96 96` | Resampling target shape (x y z) |

Output `.npz` keys: `dwi`, `adc`, `mask`

---

**`preprocess_isles_WITH_BBOX.py` — Evaluation preprocessing**

Same pipeline as above but additionally saves bounding box coordinates, original shape, cropped shape, and affine matrix into each `.npz`. This metadata is required at evaluation time to map predictions back to native space for proper DSC computation.

```bash
python preprocess_isles_WITH_BBOX.py \
  --isles-dir /path/to/ISLES2022/derivatives \
  --output-dir /path/to/preprocessed_isles_eval \
  --target-shape 96 96 96
```

| Argument | Default | Description |
|---|---|---|
| `--isles-dir` | — | Path to ISLES 2022 derivatives folder |
| `--output-dir` | — | Output directory for `.npz` files with metadata |
| `--target-shape` | `96 96 96` | Resampling target shape (x y z) |

Output `.npz` keys: `dwi`, `adc`, `mask`, `bbox`, `original_shape`, `cropped_shape`, `original_affine`

> ⚠️ Use `preprocess_isles_WITH_BBOX.py` only for evaluation. Training was conducted with `preprocess_v2_isles.py`.

---

### BraTS 2024

BraTS 2024 is used to pretrain the UCTransNet3D on brain tumour segmentation (T2-FLAIR), after which the model is fine-tuned on ATLAS and UOA stroke datasets. Preprocessing and 64³ patch-based training are handled by a single script.

**`preprocess_brats2024_t2flair.py` — BraTS 2024 preprocessing**

Loads T2-FLAIR and segmentation masks, resamples to `(197, 233, 189)`, binarises all tumour classes into a single mask, applies z-score normalisation on non-zero voxels, saves as `.npz`, and generates 5-fold train/val/test splits.

```bash
python preprocess_brats2024_t2flair.py \
  --brats-dir /path/to/BraTS2024/training_data \
  --output-dir /path/to/preprocessed_brats2024_t2flair
```

| Argument | Description |
|---|---|
| `--brats-dir` | Path to BraTS 2024 training data directory |
| `--output-dir` | Output directory for `.npz` files and splits JSON |

Output `.npz` keys: `image`, `mask`
Also generates: `brats2024_t2flair_splits_5fold.json`

---

**`train_brats_uctransnet3d.py` — UCTransNet3D pretraining on BraTS 2024**

Trains UCTransNet3D on BraTS 2024 T2-FLAIR tumour segmentation using 64³ patches. Patch extraction is handled internally — no separate patch extraction step is needed for BraTS. Run once per fold to produce pretrained checkpoints used for stroke fine-tuning.

```bash
python train_brats_uctransnet3d.py \
  --brats-dir /path/to/preprocessed_brats2024_t2flair \
  --splits-file brats2024_t2flair_splits_5fold.json \
  --output-dir /path/to/brats_uctransnet3d \
  --patch-size 64 \
  --batch-size 8 \
  --patches-per-volume 4 \
  --fold 0 --epochs 100
```

Change `--fold 0` through `--fold 4` for five-fold cross-validation. Best checkpoint saved as `fold_X/best_model.pth`.

| Argument | Default | Description |
|---|---|---|
| `--brats-dir` | — | Path to preprocessed BraTS `.npz` files |
| `--splits-file` | — | Path to `brats2024_t2flair_splits_5fold.json` |
| `--output-dir` | — | Output directory for checkpoints and history |
| `--fold` | `0` | Fold number (0–4) |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `4` | Batch size |
| `--patch-size` | `96` | Cubic patch size (use `64` for UCTransNet3D) |
| `--patches-per-volume` | `10` | Random patches sampled per volume per epoch |
| `--pretrain-ckpt` | None | Path to pretrained checkpoint (for fine-tuning arm) |

---

### ATLAS 2.0 and UOA

These datasets are in MNI space (1×1×1 mm³) and follow a two-step pipeline for both architectures: full-volume preprocessing first, then lesion-focused patch extraction. Patch size differs by architecture — **96³ for ResNet-18 U-Net, 64³ for UCTransNet3D**.

**Step 1 — `preprocess_stroke_foundation.py` — Full-volume preprocessing (run first, shared by both architectures)**

GPU-accelerated preprocessing for MNI-space data. Applies brain masking, intensity clipping, and z-score normalisation, then saves full-volume `.npz` files.

```bash
python preprocess_stroke_foundation.py --config config.json
```

Example `config.json`:
```json
{
  "output_base_dir": "/path/to/preprocessed_stroke_foundation",
  "preprocessing": {
    "target_spacing": [1.0, 1.0, 1.0],
    "normalization_method": "zscore",
    "clip_percentile": 99.5,
    "use_gpu": true
  },
  "datasets": [
    {
      "name": "ATLAS",
      "data_dir": "/path/to/ATLAS",
      "image_pattern": "**/*_T1w.nii.gz",
      "mask_pattern": "**/*_label-L_desc-T1lesion_mask.nii.gz"
    },
    {
      "name": "UOA_Private",
      "data_dir": "/path/to/UOA",
      "image_pattern": "**/*_T1_FNIRT_MNI.nii.gz",
      "mask_pattern": "**/*_lesion.nii.gz"
    }
  ]
}
```

Output per case: `<case_id>.npz` (keys: `image`, `lesion_mask`, `brain_mask`) + `<case_id>_metadata.pkl`

---

**Step 2 — `extract_patches_96x96x96.py` — Lesion-focused patch extraction (run after Step 1)**

Despite the filename, this script supports any patch size via `--patch_size`. Run it twice with different sizes — once for ResNet-18 (96³) and once for UCTransNet3D (64³).

**ResNet-18 U-Net (96³ patches):**

```bash
python extract_patches_96x96x96.py \
  --input_dir  /path/to/preprocessed_stroke_foundation \
  --output_dir /path/to/preprocessed_stroke_foundation \
  --datasets ATLAS UOA_Private \
  --patch_size 96 \
  --max_patches 20 \
  --min_lesion_overlap 0.01
```

**UCTransNet3D (64³ patches):**

```bash
python extract_patches_96x96x96.py \
  --input_dir  /path/to/preprocessed_stroke_foundation \
  --output_dir /path/to/preprocessed_stroke_foundation \
  --datasets ATLAS UOA_Private \
  --patch_size 64 \
  --max_patches 20 \
  --min_lesion_overlap 0.0001
```

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | — | Root folder containing `ATLAS/` and `UOA_Private/` subdirs from Step 1 |
| `--output_dir` | — | Root where patch directories will be created |
| `--datasets` | `ATLAS UOA_Private` | Dataset subdirectory names to process |
| `--patch_size` | `96` | Isotropic patch size (`96` for ResNet-18, `64` for UCTransNet3D) |
| `--max_patches` | `20` | Max patches extracted per volume |
| `--min_lesion_overlap` | `0.01` | Min lesion fraction required per patch |
| `--seed` | `42` | Random seed |
| `--splits_file` | None | Optional path to splits JSON to process a specific fold/split |
| `--fold` | None | Fold number (0–4), used with `--splits_file` |
| `--split` | `all` | `all`, `train`, `val`, or `test` |

Output per case: `<case_id>.npz` (keys: `patches`, `masks`, `centers`, `original_shape`, `spacing`, `lesion_volume`, `patient_id`) + `patient_ids.pkl` + `<dataset>_96_summary.json`

---

## Evaluation Metrics

- **ISLES 2022**: Dice Similarity Coefficient (DSC)
- **ATLAS / UOA**: DSC, lesion-wise F1, lesion count difference, absolute lesion volume difference (voxels)

---

## Results Summary

**ISLES 2022 (ResNet-18, 5-fold, 50-case held-out test set)**

| Strategy | Init | Mean DSC |
|---|---|---|
| Separate encoders (DWI only) | BraTS | 0.7041 |
| Shared encoder (DWI+ADC avg) | BraTS | 0.7030 |
| Separate encoders (DWI only) | Scratch | 0.7005 |
| **Shared encoder (DWI+ADC avg)** | **Scratch** | **0.7052** |
| DWI-only baseline | Scratch | 0.7025 |

**ATLAS & UOA (ResNet-18, 96³ patches)**

| Dataset | Strategy | DSC ± std | Lesion F1 | Lesion Count Diff | Vol Diff (voxels) |
|---|---|---|---|---|---|
| ATLAS | Transfer | 0.6372 ± 0.0081 | 0.6069 ± 0.0098 | 3.11 ± 0.24 | 7811 ± 806 |
| ATLAS | **Scratch** | **0.6290 ± 0.0065** | 0.5949 ± 0.0228 | 2.83 ± 0.17 | 7012 ± 530 |
| UOA | Transfer | 0.7367 ± 0.0137 | 0.5503 ± 0.0509 | 2.80 ± 0.29 | 4482 ± 1180 |
| UOA | **Scratch** | **0.7437 ± 0.0114** | 0.5601 ± 0.0273 | 2.73 ± 0.18 | 3644 ± 644 |

**ATLAS & UOA (UCTransNet3D, 64³ patches)**

Results reported across all 5 folds and, separately, across the 3 matched GroupNorm folds (folds 2–4) for a fair transfer vs scratch comparison. Folds 0–1 of the transfer condition used BatchNorm checkpoints due to a normalisation inconsistency during BraTS pretraining.

| Dataset | Strategy | Folds | DSC ± std | Lesion F1 | Lesion Count Diff | Vol Diff (voxels) |
|---|---|---|---|---|---|---|
| ATLAS | Transfer | all 5 | 0.6160 ± 0.0434 | 0.5836 ± 0.0640 | 3.36 ± 0.58 | 10097 ± 1547 |
| ATLAS | Transfer | GN (f2–4) | 0.6474 ± 0.0049 | 0.6341 ± 0.0213 | 2.96 ± 0.13 | 10835 ± 1096 |
| ATLAS | Scratch | all 5 | 0.6364 ± 0.0197 | 0.6105 ± 0.0339 | 2.93 ± 0.08 | 9210 ± 1743 |
| ATLAS | **Scratch** | **GN (f2–4)** | **0.6336 ± 0.0226** | 0.6029 ± 0.0254 | 2.94 ± 0.10 | 9720 ± 1824 |
| UOA | Transfer | all 5 | 0.6767 ± 0.0552 | 0.5263 ± 0.1040 | 3.05 ± 0.97 | 9053 ± 2144 |
| UOA | Transfer | GN (f2–4) | 0.6506 ± 0.0578 | 0.4565 ± 0.0749 | 3.67 ± 0.77 | 9929 ± 2408 |
| UOA | **Scratch** | **all 5** | **0.7279 ± 0.0192** | 0.6274 ± 0.0282 | 2.66 ± 0.40 | 9130 ± 2003 |
| UOA | Scratch | GN (f2–4) | 0.7191 ± 0.0207 | 0.6460 ± 0.0115 | 2.83 ± 0.45 | 9086 ± 1097 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ahmad2026transfer,
  title={Towards Understanding Transfer Learning and Multimodal Training for Ischemic Stroke Lesion Segmentation: A Systematic Ablation Study},
  author={Ahmad, Parvez and Chong, Benjamin and Fernandez, Justin and Shim, Vickie and Kasabov, Nikola Kirilov and Stinear, Cathy M. and Byblow, Winston D. and Wang, Alan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2026}
}
```

---

## Funding

This work was funded by HRC New Zealand (21/144), MBIE Catalyst NZ Singapore (UOAX2001), Marsden Fund (22-UOA-120), and Royal Society Catalyst (23-UOA-055-CSG).

---

## Contact

Parvez Ahmad — pahm409@aucklanduni.ac.nz  
Auckland Bioengineering Institute, The University of Auckland

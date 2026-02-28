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

## Repository Structure

---

## Repository Files

**Preprocessing**
- `preprocess_v2_isles.py`
- `preprocess_isles_WITH_BBOX.py`
- `preprocess_brats2024_t2flair.py`
- `preprocess_stroke_foundation.py`
- `extract_patches_96x96x96.py`

**Training — ResNet-18**
- `train_brats_t2flair_supervised_FIXED.py`
- `joint_training_MINIMAL_FIX.py`
- `joint_training_SEPARATE_LR.py`
- `joint_training_from_scratch.py`
- `channelwise_from_scratch.py`
- `dwi_only_baseline.py`
- `train_all_experiments_with_volume_validation.py`

**Training — UCTransNet3D**
- `train_brats_uctransnet3d.py`
- `finetune_uctransnet3d_stroke.py`

**Evaluation — ISLES 2022**
- `evaluate_joint_sliding_window_DEBUG.py`
- `evaluate_channelwise_sliding_window.py`
- `evaluate_joint_from_scratch.py`
- `evaluate_channelwise_from_scratch.py`
- `evaluate_dwi_only.py`

**Evaluation — ATLAS 2.0 and UOA**
- `evaluate_all_experiments.py`
- `evaluate_uctransnet3d_stroke.py`

---

## Data Preprocessing

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

## Model Training

### ResNet-18 U-Net (96³ patches)

#### Step 1 — BraTS 2024 Pretraining

**`train_brats_t2flair_supervised_FIXED.py` — ResNet-18 pretraining on BraTS 2024**

Trains a ResNet-18 U-Net on BraTS 2024 T2-FLAIR tumour segmentation using 96³ random patches. Produces pretrained encoder checkpoints used to initialise the transfer learning stroke fine-tuning experiments.

```bash
python train_brats_t2flair_supervised_FIXED.py \
  --brats-dir /path/to/preprocessed_brats2024_t2flair \
  --splits-file brats2024_t2flair_splits_5fold.json \
  --output-dir /path/to/brats_t2flair_supervised \
  --fold 0 --epochs 100 --batch-size 8
```

| Argument | Default | Description |
|---|---|---|
| `--brats-dir` | — | Path to preprocessed BraTS `.npz` files |
| `--splits-file` | — | Path to `brats2024_t2flair_splits_5fold.json` |
| `--output-dir` | — | Output directory for checkpoints |
| `--fold` | `0` | Fold number (0–4) |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |

---

#### Step 2 — ISLES 2022 Fine-tuning (4 strategies)

Four independent training scripts cover the ISLES 2022 ablation strategies. Two are shared here; the remaining two (scratch variants) follow the same interface.

---

**`joint_training_MINIMAL_FIX.py` — Separate encoders, transfer learning (Exp 1)**

Individual DWI and ADC U-Nets with separate ResNet-18 encoders. The DWI encoder is initialised from the BraTS pretrained checkpoint; the ADC encoder is randomly initialised. Both are trained jointly with `L = L_DWI + L_ADC`. Validation and inference use the DWI branch only. Includes spatially consistent augmentations (rotation, elastic deformation, scaling, flipping) plus independent intensity augmentations per modality.

```bash
python joint_training_MINIMAL_FIX.py \
  --brats-checkpoint /path/to/brats_t2flair_supervised/fold_0/best_model.pth \
  --isles-dir /path/to/preprocessed_isles_dual_v2 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/joint_training_minimal_fix \
  --fold 0
```

| Argument | Description |
|---|---|
| `--brats-checkpoint` | Path to BraTS pretrained checkpoint (required) |
| `--isles-dir` | Path to preprocessed ISLES 2022 `.npz` files |
| `--splits-file` | Path to ISLES 2022 5-fold splits JSON |
| `--output-dir` | Output directory for checkpoints and training log |
| `--fold` | Fold number (0–4) |

Fixed settings: 200 epochs, batch size 8, 96³ patches, DWI LR=1e-4 (pretrained), ADC LR=1e-3 (scratch), PolyLR scheduler (power=0.9), soft Dice loss.

---

**`joint_training_SEPARATE_LR.py` — Shared encoder (2-channel), transfer learning (Exp 2)**

DWI and ADC are concatenated as a 2-channel input and processed by a single shared ResNet-18 encoder initialised from BraTS pretrained weights (conv1 weights duplicated and scaled by 0.5 for 2-channel adaptation). Two separate decoders produce DWI and ADC segmentation outputs. Validation averages both decoder outputs. Encoder and decoder learning rates are set independently.

```bash
python joint_training_SEPARATE_LR.py \
  --brats-checkpoint /path/to/brats_t2flair_supervised/fold_0/best_model.pth \
  --isles-dir /path/to/preprocessed_isles_dual_v2 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/joint_training_dualchannel \
  --fold 0
```

| Argument | Description |
|---|---|
| `--brats-checkpoint` | Path to BraTS pretrained checkpoint (required) |
| `--isles-dir` | Path to preprocessed ISLES 2022 `.npz` files |
| `--splits-file` | Path to ISLES 2022 5-fold splits JSON |
| `--output-dir` | Output directory for checkpoints and training log |
| `--fold` | Fold number (0–4) |

Fixed settings: 200 epochs, batch size 8, 96³ patches, encoder LR=1e-4, decoder LR=1e-3, PolyLR scheduler (power=0.9), soft Dice loss.

---

**`joint_training_from_scratch.py` — Separate encoders, scratch (Exp 3)**

Identical architecture and hyperparameters to Exp 1 but both DWI and ADC encoders randomly initialised. No BraTS checkpoint required.

```bash
for i in 0 1 2 3 4; do
  python joint_training_from_scratch.py --fold $i
done
```

---

**`channelwise_from_scratch.py` — Shared encoder (2-channel), scratch (Exp 4)**

Identical architecture and hyperparameters to Exp 2 but shared encoder randomly initialised. No BraTS checkpoint required.

```bash
for i in 0 1 2 3 4; do
  python channelwise_from_scratch.py --fold $i
done
```

---

### ATLAS 2.0 and UOA Private (ResNet-18)

**`train_all_experiments_with_volume_validation.py` — All 20 ATLAS/UOA experiments**

Runs all 20 experiments (2 datasets × 2 conditions × 5 folds) sequentially. Both transfer and scratch conditions are handled by a single `use_transfer` flag per experiment entry — if a BraTS checkpoint is found for the fold it loads encoder weights; otherwise it falls back to random initialisation gracefully. Training uses lesion-focused patch sampling (70% lesion-centred, 30% random).

Validation runs at two levels: patch-level DSC every epoch, and volume-level sliding window DSC at epoch 1 and every 10 epochs thereafter. NIfTI predictions are saved at epoch 1 (up to 5 cases) for visual inspection.

```bash
CUDA_VISIBLE_DEVICES=0 python train_all_experiments_with_volume_validation.py
```

All paths are set at the top of the script:

| Variable | Description |
|---|---|
| `BRATS_BASE_DIR` | Root of BraTS pretrained checkpoints (`fold_X/best_model.pth`) |
| `PREPROCESSED_DIR` | Root of `preprocessed_stroke_foundation` outputs (Step 1) |
| `SPLITS_FILE` | Path to `splits_5fold.json` |
| `OUTPUT_DIR` | Output root for all checkpoints and logs |
| `EPOCHS` | Epochs per experiment (default: 100) |
| `BATCH_SIZE` | Batch size (default: 8) |

Fixed settings: 100 epochs, batch size 8, 96³ patches, LR=1e-4 (Adam), soft Dice loss. Uses `PatchDatasetWithCenters` for lesion-focused sampling.

Outputs per fold: `<dataset>/<transfer|scratch>/fold_X/best_model.pth`, `log.csv`, `volume_validation_epoch1/` (NIfTI), `summary_progress.json` (updated after each fold), `summary_final.json`.

---

### UCTransNet3D (64³ patches) — ATLAS and UOA Fine-tuning

**`finetune_uctransnet3d_stroke.py` — All 20 UCTransNet3D ATLAS/UOA experiments**

Fine-tunes UCTransNet3D on ATLAS and UOA across all combinations of dataset, condition (transfer/scratch), and fold. The `--run-all` flag iterates through all 20 experiments (2 datasets × 2 conditions × 5 folds) automatically. Transfer experiments initialise the encoder from the corresponding BraTS pretrained UCTransNet3D checkpoint (`fold_X/best_model.pth`); scratch experiments use random initialisation. Uses 64³ patches throughout for consistency with BraTS pretraining.

```bash
python finetune_uctransnet3d_stroke.py --run-all
```

All paths and hyperparameters are configured inside the script. Outputs the same structure as the ResNet-18 ATLAS/UOA experiments: `<dataset>/<transfer|scratch>/fold_X/best_model.pth`, `log.csv`, and per-fold progress JSON.

---

## Evaluation

### ISLES 2022 (ResNet-18)

All evaluation scripts use sliding window inference (50% overlap) over the 96³ preprocessed volume, then reverse the preprocessing (bbox unpadding + zoom) to restore predictions to native NIfTI space before computing DSC against the original ground truth masks.

> ⚠️ Evaluation requires `preprocess_isles_WITH_BBOX.py` outputs (not the training `.npz` files), as the bbox metadata is needed for spatial reconstruction.

---

**`evaluate_joint_sliding_window_DEBUG.py` — Separate encoders (Exp 1 & 3)**

Evaluates the separate-encoder model (`joint_training_MINIMAL_FIX.py`). Runs sliding window inference on DWI only, reconstructs predictions to original space using saved bbox metadata, and computes DSC in native NIfTI space. Also reports intermediate DSC in 96³ space for debugging.

```bash
python evaluate_joint_sliding_window_DEBUG.py \
  --checkpoint /path/to/joint_training_minimal_fix/fold_0/best_joint_model.pth \
  --isles-dir /path/to/preprocessed_isles_dual_WITH_BBOX \
  --isles-raw-dir /path/to/ISLES2022 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/test_results \
  --step-size 0.5
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to trained model checkpoint (required) |
| `--isles-dir` | — | Path to `preprocessed_isles_dual_WITH_BBOX` `.npz` files |
| `--isles-raw-dir` | — | Path to raw ISLES 2022 NIfTI files (for ground truth) |
| `--splits-file` | — | Path to ISLES 2022 5-fold splits JSON |
| `--output-dir` | — | Output directory for `test_results.json` |
| `--step-size` | `0.5` | Sliding window step size (0.5 = 50% overlap) |
| `--save-nifti` | flag | Save prediction and ground truth NIfTI files |

---

**`evaluate_channelwise_sliding_window.py` — Shared encoder, 2-channel (Exp 2 & 4)**

Evaluates the shared-encoder dual-channel model (`joint_training_SEPARATE_LR.py`). Takes concatenated DWI+ADC as input, runs sliding window inference, and reports DSC separately for the DWI decoder, ADC decoder, and averaged output. Final reported DSC uses the averaged prediction.

```bash
python evaluate_channelwise_sliding_window.py \
  --checkpoint /path/to/joint_training_dualchannel/fold_0/best_joint_model.pth \
  --isles-dir /path/to/preprocessed_isles_dual_WITH_BBOX \
  --isles-raw-dir /path/to/ISLES2022 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/channelwise_test_results \
  --step-size 0.5
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to trained model checkpoint (required) |
| `--isles-dir` | — | Path to `preprocessed_isles_dual_WITH_BBOX` `.npz` files |
| `--isles-raw-dir` | — | Path to raw ISLES 2022 NIfTI files |
| `--splits-file` | — | Path to ISLES 2022 5-fold splits JSON |
| `--output-dir` | — | Output directory for `test_results.json` |
| `--step-size` | `0.5` | Sliding window step size |
| `--save-nifti` | flag | Save per-case NIfTI predictions |

---

**`dwi_only_baseline.py` — DWI-only baseline (Exp 5), includes training + evaluation**

Single ResNet-18 U-Net trained on DWI only from scratch. Identical hyperparameters to Exp 3 (no pretraining). Training and test evaluation are run in one script — the best checkpoint is automatically evaluated after training completes.

```bash
python dwi_only_baseline.py \
  --isles-dir /path/to/preprocessed_isles_dual_v2 \
  --isles-raw-dir /path/to/ISLES2022 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/dwi_only_baseline \
  --fold 0 --gpu 0
```

| Argument | Default | Description |
|---|---|---|
| `--isles-dir` | — | Path to training `.npz` files (no bbox needed) |
| `--isles-raw-dir` | — | Path to raw ISLES 2022 NIfTI files |
| `--splits-file` | — | Path to ISLES 2022 5-fold splits JSON |
| `--output-dir` | — | Output directory |
| `--fold` | required | Fold number (0–4) |
| `--gpu` | `0` | GPU index |
| `--step-size` | `0.5` | Sliding window step size for evaluation |
| `--save-nifti` | flag | Save NIfTI predictions after evaluation |

---

**`evaluate_dwi_only.py` — Standalone evaluation for DWI-only baseline**

Standalone evaluation script for checkpoints from `dwi_only_baseline.py`. Useful for re-evaluating with a different step size or on a different split without retraining.

```bash
python evaluate_dwi_only.py \
  --checkpoint /path/to/dwi_only_baseline/fold_0/best_model.pth \
  --fold 0 --gpu 0 \
  --isles-dir /path/to/preprocessed_isles_dual_WITH_BBOX \
  --isles-raw-dir /path/to/ISLES2022 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/dwi_only_baseline_test_results \
  --step-size 0.75
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to `best_model.pth` (required) |
| `--fold` | required | Fold number (0–4) |
| `--gpu` | `0` | GPU index |
| `--isles-dir` | — | Path to `preprocessed_isles_dual_WITH_BBOX` `.npz` files |
| `--isles-raw-dir` | — | Path to raw ISLES 2022 NIfTI files |
| `--splits-file` | — | Path to ISLES 2022 5-fold splits JSON |
| `--output-dir` | — | Output directory for `test_results.json` |
| `--step-size` | `0.75` | Sliding window step size |
| `--save-nifti` | flag | Save NIfTI predictions |

---

**`evaluate_joint_from_scratch.py` — Separate encoders, scratch (Exp 3)**

Evaluation counterpart for `joint_training_from_scratch.py`. Same sliding window + reverse preprocessing pipeline as Exp 1.

```bash
python evaluate_joint_from_scratch.py \
  --checkpoint /path/to/fold_0/best_joint_model.pth \
  --isles-dir /path/to/preprocessed_isles_dual_WITH_BBOX \
  --isles-raw-dir /path/to/ISLES2022 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/results --step-size 0.5
```

---

**`evaluate_channelwise_from_scratch.py` — Shared encoder (2-channel), scratch (Exp 4)**

Evaluation counterpart for `channelwise_from_scratch.py`. Same pipeline as Exp 2 (reports DWI, ADC, and averaged DSC).

```bash
python evaluate_channelwise_from_scratch.py \
  --checkpoint /path/to/fold_0/best_joint_model.pth \
  --isles-dir /path/to/preprocessed_isles_dual_WITH_BBOX \
  --isles-raw-dir /path/to/ISLES2022 \
  --splits-file isles_dual_splits_5fold.json \
  --output-dir /path/to/results --step-size 0.5
```

---

### ATLAS 2.0 and UOA Private (ResNet-18)

**`evaluate_all_experiments.py` — All ATLAS/UOA experiments (transfer + scratch)**

Evaluates all ResNet-18 experiments on ATLAS and UOA. Runs sliding window inference on the held-out test split and computes four metrics: DSC, absolute volume difference (voxels), lesion-wise F1 (ATLAS standard: any single-voxel overlap counts as a match), and absolute lesion count difference. Results are saved per-fold and aggregated across folds.

```bash
# Evaluate all folds, both conditions for ATLAS
python evaluate_all_experiments.py --dataset ATLAS --gpu 0

# Evaluate all folds, both conditions for UOA
python evaluate_all_experiments.py --dataset UOA_Private --gpu 0

# Evaluate transfer condition only
python evaluate_all_experiments.py --dataset ATLAS --condition transfer --gpu 0

# Evaluate specific folds only
python evaluate_all_experiments.py --dataset ATLAS --folds 0 1 --gpu 0
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | `ATLAS` or `UOA_Private` |
| `--condition` | both | `transfer`, `scratch`, or omit for both |
| `--folds` | 0–4 | Specific fold numbers to evaluate |
| `--gpu` | `0` | GPU index |

Metrics reported: DSC, absolute volume difference (voxels), lesion-wise F1, lesion count difference.

---

### UCTransNet3D — ATLAS and UOA

**`evaluate_uctransnet3d_stroke.py` — All UCTransNet3D ATLAS/UOA experiments**

Evaluates all UCTransNet3D experiments (transfer + scratch, both datasets, all folds). Computes the same four metrics as the ResNet-18 evaluation: DSC, lesion-wise F1, lesion count difference, and absolute volume difference.

```bash
python evaluate_uctransnet3d_stroke.py \
  --exp-dir     /hpc/pahm409/uctransnet3d_stroke_experiments \
  --data-dir    /hpc/pahm409/harvard/preprocessed_stroke_foundation \
  --splits-file splits_5fold.json \
  --output-dir  /hpc/pahm409/uctransnet3d_stroke_test_results \
  --gpu 0
```

| Argument | Description |
|---|---|
| `--exp-dir` | Root of `finetune_uctransnet3d_stroke.py` outputs (`<dataset>/<transfer\|scratch>/fold_X/`) |
| `--data-dir` | Root of `preprocessed_stroke_foundation` outputs |
| `--splits-file` | Path to `splits_5fold.json` |
| `--output-dir` | Output directory for per-fold and aggregated JSON results |
| `--gpu` | GPU index |

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



---

## Funding

This work was funded by HRC New Zealand (21/144), MBIE Catalyst NZ Singapore (UOAX2001), Marsden Fund (22-UOA-120), and Royal Society Catalyst (23-UOA-055-CSG).

---

## Contact

Parvez Ahmad — pahm409@aucklanduni.ac.nz  
Auckland Bioengineering Institute, The University of Auckland

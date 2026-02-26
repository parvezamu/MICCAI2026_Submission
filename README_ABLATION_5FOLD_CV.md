# Ablation Study 5-Fold Cross-Validation - Clean Implementation

## Overview
This is a simplified version of your ablation study pipeline implementing **proper 5-fold cross-validation** without unnecessary repetitions. The ablation study compares different combinations of initialization methods (Random vs SimCLR) and architectural enhancements (Baseline vs MKDC+DS).

## Key Changes from Original

### 1. **Removed Multiple Runs Per Fold**
- **Old**: Each fold trained 3 times (runs 0, 1, 2) = 60 total experiments
- **New**: Each fold trained **once** = 20 total experiments
- **Rationale**: Standard 5-fold CV provides sufficient variance estimation

### 2. **Updated Output Directory**
- **Old**: `/home/pahm409/ablation_ds_main_only1`
- **New**: `/home/pahm409/ablation_atlas_uoa_5fold_cv`
- **Structure**:
  ```
  ablation_atlas_uoa_5fold_cv/
  ├── Random_Init/
  │   ├── none/              # Baseline
  │   │   ├── fold_0/
  │   │   ├── fold_1/
  │   │   └── ...
  │   └── mkdc_ds/           # MKDC + Deep Supervision
  │       ├── fold_0/
  │       └── ...
  ├── SimCLR_Pretrained/
  │   ├── none/              # Baseline
  │   └── mkdc_ds/           # MKDC + Deep Supervision
  └── logs/
  ```

### 3. **Simplified Scripts**

#### **train_ablation_5fold_clean.py**
- Removed `--run-id` parameter
- No `run_X` subdirectories
- Seeds: `42 + fold` (42, 43, 44, 45, 46)

#### **run_ablation_fold_5fold_clean.py**
- Trains 4 experiments per fold (no triple repetition)
- Simplified progress tracking

#### **run_all_ablation_folds_5fold_cv.py**
- Master script to run all 5 folds sequentially
- Automatic progress tracking

## Experiment Configurations

The ablation study tests **4 configurations**:

| # | Name | Initialization | Architecture | Description |
|---|------|---------------|--------------|-------------|
| 1 | **Exp1_Random_Baseline** | Random | Baseline | Standard U-Net decoder, random init |
| 2 | **Exp2_Random_MKDC_DS** | Random | MKDC + DS | Enhanced decoder, random init |
| 3 | **Exp3_SimCLR_Baseline** | SimCLR | Baseline | Standard decoder, pretrained encoder |
| 4 | **Exp4_SimCLR_MKDC_DS** | SimCLR | MKDC + DS | Enhanced decoder, pretrained encoder |

### Architecture Components

- **MKDC**: Multi-Kernel Depthwise Convolution on skip connections
- **DS**: Deep Supervision (auxiliary losses at decoder levels 4, 3, 2)
- **SimCLR**: Self-supervised pretrained encoder

## Dataset

- **ATLAS**: 655 cases (chronic stroke lesions)
- **UOA_Private**: ~200 cases (acute/subacute stroke)
- **Total**: ~855 cases with T1-weighted MRI + lesion masks
- **Split**: 5-fold cross-validation

## Usage

### Option 1: Train All Folds (Recommended)
```bash
# Run complete 5-fold cross-validation
python run_all_ablation_folds_5fold_cv.py --gpu 1

# Or start from specific fold (if interrupted)
python run_all_ablation_folds_5fold_cv.py --gpu 1 --start-fold 2
```

### Option 2: Train Individual Fold
```bash
# Train all 4 experiments for fold 0
python run_ablation_fold_5fold_clean.py --fold 0 --gpu 1

# Train all 4 experiments for fold 1
python run_ablation_fold_5fold_clean.py --fold 1 --gpu 1
```

### Option 3: Train Single Experiment
```bash
# Random Baseline for fold 0
python train_ablation_5fold_clean.py --fold 0 --attention none

# Random MKDC+DS for fold 0
python train_ablation_5fold_clean.py --fold 0 --attention mkdc --deep-supervision

# SimCLR Baseline for fold 1
python train_ablation_5fold_clean.py --fold 1 --attention none \
    --pretrained-checkpoint /path/to/simclr_checkpoint.pth

# SimCLR MKDC+DS for fold 2
python train_ablation_5fold_clean.py --fold 2 --attention mkdc --deep-supervision \
    --pretrained-checkpoint /path/to/simclr_checkpoint.pth
```

## Expected Timeline

- **Per experiment**: ~1.5-2 hours (100 epochs)
- **Per fold**: ~6-8 hours (4 experiments)
- **Complete 5-fold CV**: ~30-40 hours total

## Training Configuration

- **Batch Size**: 8
- **Learning Rate**: 0.0001 (AdamW with warmup + cosine annealing)
- **Patch Size**: 96×96×96
- **Patches per Volume**: 10 (training), 100 (validation)
- **Lesion Focus Ratio**: 0.7
- **Epochs**: 100
- **Warmup Epochs**: 5

## Validation Strategy

### Patch-level Validation (Every Epoch)
- Fast evaluation on random patches
- Used for monitoring training

### Full-volume Reconstruction (Every 5 Epochs)
- Complete volume reconstruction
- More accurate DSC computation
- Used for model selection

### Final Evaluation (Epoch 100)
- Full reconstruction with NIfTI outputs
- Per-case results in JSON

## Output Files

### Per Experiment Directory
```
exp_20250210_120000/
├── config.json                    # Training configuration
├── training_log.csv               # Epoch-wise metrics
├── curves_final.png               # Training curves
├── final_summary.json             # Best performance
├── checkpoints/
│   └── best_model.pth             # Best model
└── reconstructions_epoch_100/     # Final predictions
    ├── summary.json
    └── case_*/
        ├── prediction.nii.gz
        ├── ground_truth.nii.gz
        └── metadata.json
```

## SimCLR Checkpoint

The SimCLR pretrained checkpoint should be located at:
```
/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth
```

Update `SIMCLR_CHECKPOINT` in `run_ablation_fold_5fold_clean.py` if using a different checkpoint.

## Comparison: Old vs New

| Aspect | Old | New |
|--------|-----|-----|
| Runs per fold | 3 | 1 |
| Total experiments | 4 configs × 5 folds × 3 runs = **60** | 4 configs × 5 folds = **20** |
| Random seeds | 42 + fold*10 + run | 42 + fold |
| Directory depth | `init/config/fold_X/run_Y/exp_*/` | `init/config/fold_X/exp_*/` |
| Training time | ~90-120 hours | ~30-40 hours |

## Analysis Plan

After completion, you'll have:
- **5 validation DSC scores** per configuration (one per fold)
- **Mean ± std** across folds for each configuration
- Statistical comparison between:
  - Random vs SimCLR initialization
  - Baseline vs MKDC+DS architecture
  - Interaction effects

### Expected Comparisons

1. **Initialization Effect**:
   - Random Baseline vs SimCLR Baseline
   - Random MKDC+DS vs SimCLR MKDC+DS

2. **Architecture Effect**:
   - Random Baseline vs Random MKDC+DS
   - SimCLR Baseline vs SimCLR MKDC+DS

3. **Interaction Effect**:
   - Does SimCLR benefit more from MKDC+DS than random?
   - 2×2 factorial analysis

## Resume Training

If interrupted:
```bash
# Scripts automatically detect and skip completed experiments
python run_all_ablation_folds_5fold_cv.py --gpu 1

# Or manually resume specific experiment
python train_ablation_5fold_clean.py --fold 0 --attention mkdc \
    --deep-supervision \
    --resume-checkpoint /path/to/checkpoints/best_model.pth
```

## Monitoring Progress

```bash
# Check overall progress
cat /home/pahm409/ablation_atlas_uoa_5fold_cv/training_progress.json

# Check specific experiment log
tail -f /home/pahm409/ablation_atlas_uoa_5fold_cv/logs/Exp1_Random_Baseline/fold_0.log

# Check training curves
# Open: /home/pahm409/ablation_atlas_uoa_5fold_cv/Random_Init/none/fold_0/exp_*/curves_final.png
```

## GPU Configuration

The scripts are configured to use **GPU 1** by default (as specified in your original code). If you want to use a different GPU:

```bash
python run_all_ablation_folds_5fold_cv.py --gpu 0  # Use GPU 0
```

## Notes

1. **No Multiple Runs**: Standard 5-fold CV doesn't require multiple runs per fold. The variance across 5 folds is sufficient for robust evaluation.

2. **Fold-Specific Seeds**: Each fold uses a unique seed (42, 43, 44, 45, 46) ensuring reproducibility while maintaining independence.

3. **Automatic Skip**: Scripts automatically detect completed experiments and skip them, so you can safely re-run commands.

4. **Dataset**: Uses ATLAS + UOA_Private (preprocessed in `/home/pahm409/preprocessed_stroke_foundation/`)

5. **Same Quality**: This produces the same scientific quality results as the original implementation, just 3× faster.

## Expected Results Structure

```
ablation_atlas_uoa_5fold_cv/
├── Random_Init/
│   ├── none/
│   │   ├── fold_0/ → DSC_0
│   │   ├── fold_1/ → DSC_1
│   │   ├── fold_2/ → DSC_2
│   │   ├── fold_3/ → DSC_3
│   │   └── fold_4/ → DSC_4
│   │   → Mean ± Std across folds
│   └── mkdc_ds/
│       └── ... (same structure)
└── SimCLR_Pretrained/
    └── ... (same structure)
```

Final analysis will compare mean ± std across the 4 configurations to determine:
1. Does SimCLR pretraining help?
2. Does MKDC+DS architecture help?
3. Is there an interaction effect?

This is the standard approach for ablation studies in medical imaging!

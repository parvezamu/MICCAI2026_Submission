# DWI 5-Fold Cross-Validation - Clean Implementation

## Overview
This is a simplified version of your DWI training pipeline that implements **proper 5-fold cross-validation** without unnecessary repetitions.

## Key Changes

### 1. **Removed Multiple Runs Per Fold**
- **Old**: Each fold was trained 3 times (runs 0, 1, 2) with different random seeds
- **New**: Each fold is trained **once** with a fold-specific seed (42 + fold)
- **Rationale**: Standard 5-fold CV doesn't require multiple runs per fold. The 5 folds already provide sufficient variance estimation.

### 2. **Updated Output Directory**
- **Old**: `/home/pahm409/dwi_scratch_5fold`
- **New**: `/home/pahm409/dwi_isles_5fold_cv`
- **Structure**:
  ```
  dwi_isles_5fold_cv/
  ├── baseline/
  │   ├── fold_0/
  │   │   └── exp_20250210_120000/
  │   ├── fold_1/
  │   ├── fold_2/
  │   ├── fold_3/
  │   └── fold_4/
  ├── mkdc/
  │   ├── fold_0/
  │   └── ...
  ├── ds/
  │   └── ...
  ├── mkdc_ds/
  │   └── ...
  └── logs/
  ```

### 3. **Simplified Scripts**

#### **train_dwi_scratch_5fold_clean.py**
- Removed `--run-id` parameter
- Removed `run_X` subdirectories
- Simplified directory structure
- Seeds are now: `42 + fold` (i.e., 42, 43, 44, 45, 46 for folds 0-4)

#### **run_fold_5fold_clean.py**
- Trains all 4 experiment configs for a single fold
- No more triple repetition per fold
- Simplified progress tracking

#### **run_all_folds_5fold_cv.py**
- NEW master script to run all 5 folds sequentially
- Tracks overall progress
- Saves intermediate results

## Usage

### Option 1: Train All Folds (Recommended)
```bash
# Run complete 5-fold cross-validation
python run_all_folds_5fold_cv.py --gpu 0

# Or start from a specific fold (if interrupted)
python run_all_folds_5fold_cv.py --gpu 0 --start-fold 2
```

### Option 2: Train Individual Fold
```bash
# Train all experiments for fold 0
python run_fold_5fold_clean.py --fold 0 --gpu 0

# Train all experiments for fold 1
python run_fold_5fold_clean.py --fold 1 --gpu 0
```

### Option 3: Train Single Experiment
```bash
# Baseline for fold 0
python train_dwi_scratch_5fold_clean.py --fold 0

# MKDC + DS for fold 0
python train_dwi_scratch_5fold_clean.py --fold 0 --use-mkdc --deep-supervision

# MKDC only for fold 1
python train_dwi_scratch_5fold_clean.py --fold 1 --use-mkdc

# DS only for fold 2
python train_dwi_scratch_5fold_clean.py --fold 2 --deep-supervision
```

## Experiment Configurations

The pipeline trains 4 different architectures:

1. **Baseline**: Standard U-Net decoder
2. **MKDC**: Multi-Kernel Depthwise Convolution on skip connections
3. **DS**: Deep Supervision
4. **MKDC + DS**: Combined approach

## Expected Timeline

- **Per fold**: ~6-10 hours (4 experiments × 1.5-2.5 hours each)
- **Complete 5-fold CV**: ~30-50 hours total
- **Per experiment**: ~1.5-2.5 hours (100 epochs)

## Validation Strategy

### Patch-level Validation (Every Epoch)
- Fast evaluation on random patches
- Used for monitoring training progress

### Full-volume Reconstruction (Every 5 Epochs)
- Complete reconstruction of all validation volumes
- More accurate DSC computation
- Used for model selection

### Final Evaluation (Epoch 100)
- Full reconstruction with NIfTI outputs saved
- Detailed per-case results in JSON format

## Output Files

### Per Experiment Directory
```
exp_20250210_120000/
├── config.json                    # Training configuration
├── training_log.csv               # Epoch-wise metrics
├── curves_final.png               # Training curves
├── final_summary.json             # Best performance summary
├── checkpoints/
│   └── best_model.pth             # Best model checkpoint
└── reconstructions_epoch_100/     # Final predictions
    ├── summary.json
    └── case_*/
        ├── prediction.nii.gz
        ├── prediction_prob.nii.gz
        ├── ground_truth.nii.gz
        ├── volume.nii.gz
        └── metadata.json
```

### Progress Tracking
```
dwi_isles_5fold_cv/
├── training_progress.json         # Overall progress (from master script)
└── logs/
    ├── DWI_Exp1_Baseline/
    │   ├── fold_0.log
    │   ├── fold_1.log
    │   └── ...
    ├── DWI_Exp2_MKDC_DS/
    └── ...
```

## Key Features

### Automatic Resume
- Detects completed experiments and skips them
- Can resume interrupted training from best checkpoint
- Progress tracking across runs

### Smart Validation
- Patch validation every epoch (fast)
- Full reconstruction every 5 epochs (accurate)
- Final NIfTI outputs at epoch 100 only

### Memory Efficiency
- Mixed precision training (AMP)
- Gradient clipping
- Optimized DataLoader settings

## Random Seeds

Each fold uses a deterministic seed:
- Fold 0: seed = 42
- Fold 1: seed = 43
- Fold 2: seed = 44
- Fold 3: seed = 45
- Fold 4: seed = 46

This ensures reproducibility while maintaining independence across folds.

## Comparison: Old vs New

| Aspect | Old Implementation | New Implementation |
|--------|-------------------|-------------------|
| Runs per fold | 3 (run_0, run_1, run_2) | 1 |
| Total experiments | 4 configs × 5 folds × 3 runs = **60** | 4 configs × 5 folds = **20** |
| Random seeds | 42 + fold*10 + run | 42 + fold |
| Directory depth | `config/fold_X/run_Y/exp_*/` | `config/fold_X/exp_*/` |
| Total training time | ~90-150 hours | ~30-50 hours |
| Cross-validation | Valid but redundant | Standard 5-fold CV |

## Why This Is Better

1. **Standard Practice**: 5-fold CV is the standard - no need for multiple runs per fold
2. **Efficiency**: 3× faster (20 experiments vs 60)
3. **Cleaner Structure**: Simpler directory hierarchy
4. **Same Scientific Validity**: 5 folds provide sufficient variance estimation
5. **Easier Analysis**: One result per (config, fold) combination

## When to Use Multiple Runs

Multiple runs per fold would only be necessary if:
- You're comparing very similar methods and need ultra-precise variance estimates
- You're investigating the effect of initialization
- You have specific requirements for statistical testing

For standard architecture comparison (baseline vs MKDC vs DS vs MKDC+DS), 5-fold CV is sufficient.

## Resume Training

If training is interrupted:

```bash
# The scripts automatically detect and skip completed experiments
# Just re-run the same command:
python run_all_folds_5fold_cv.py --gpu 0

# Or resume from a specific checkpoint manually:
python train_dwi_scratch_5fold_clean.py --fold 0 --use-mkdc \
    --resume-checkpoint /path/to/checkpoints/best_model.pth
```

## Monitoring Progress

```bash
# Check overall progress
cat /home/pahm409/dwi_isles_5fold_cv/training_progress.json

# Check specific fold log
tail -f /home/pahm409/dwi_isles_5fold_cv/logs/DWI_Exp1_Baseline/fold_0.log

# Check training curves
# Open: /home/pahm409/dwi_isles_5fold_cv/baseline/fold_0/exp_*/curves_final.png
```

## Final Analysis

After all folds complete, you'll have:
- 5 validation DSC scores per configuration (one per fold)
- Mean ± std across folds for each configuration
- Best model checkpoint for each fold
- Detailed reconstruction results for final epoch

This is exactly what you need for proper cross-validation analysis!

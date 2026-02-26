# Quick Start Guide - Fold-Level Training

## The Problem You Had
You don't have time to run all 60 experiments sequentially. You need to demonstrate that best results aren't cherry-picked by showing results across all 5 folds with 3 runs each.

## The Solution
**Fold-level parallelization** - Pick a fold, run all 3 runs for that fold automatically.

---

## Usage

### For ATLAS T1 Experiments

```bash
# Train fold 0 (all 3 runs, 4 experiments = 12 total)
python train_single_fold_all_runs.py --fold 0

# Train fold 1
python train_single_fold_all_runs.py --fold 1

# Train fold 2
python train_single_fold_all_runs.py --fold 2

# Train fold 3
python train_single_fold_all_runs.py --fold 3

# Train fold 4
python train_single_fold_all_runs.py --fold 4

# Optional: specify GPU
python train_single_fold_all_runs.py --fold 0 --gpu 1
```

### For DWI Experiments

```bash
# Train fold 0 (all 3 runs, 4 experiments = 12 total)
python train_single_fold_dwi.py --fold 0

# Train fold 1
python train_single_fold_dwi.py --fold 1

# And so on...
python train_single_fold_dwi.py --fold 0 --gpu 0
```

---

## What Happens When You Run It

### Example: `python train_single_fold_all_runs.py --fold 0`

This runs **12 experiments** sequentially:

1. **Exp1_Random_Baseline** - Fold 0
   - Run 0
   - Run 1
   - Run 2

2. **Exp2_Random_MKDC_DS** - Fold 0
   - Run 0
   - Run 1
   - Run 2

3. **Exp3_SimCLR_Baseline** - Fold 0
   - Run 0
   - Run 1
   - Run 2

4. **Exp4_SimCLR_MKDC_DS** - Fold 0
   - Run 0
   - Run 1
   - Run 2

**Estimated time per fold: 18-30 hours** (with optimized DataLoaders)

---

## Smart Features

### 1. **Auto-Skip Completed Experiments**
- Checks if experiment already finished
- Skips completed runs automatically
- Shows progress summary

### 2. **Auto-Resume from Checkpoint**
- Detects interrupted experiments
- Resumes from last checkpoint
- No wasted training time

### 3. **Real-Time Progress**
- Shows training output live
- Saves to log files
- Progress tracking with ETA

### 4. **GPU Cleanup**
- Automatic cleanup between experiments
- Prevents memory issues
- Configurable wait time

---

## Parallel Execution Strategy

If you have **2 GPUs**, run folds in parallel:

```bash
# Terminal 1 (GPU 0)
python train_single_fold_dwi.py --fold 0 --gpu 0

# Terminal 2 (GPU 1)
python train_single_fold_all_runs.py --fold 0 --gpu 1
```

Or run different folds on same dataset:

```bash
# Terminal 1 (GPU 0)
python train_single_fold_all_runs.py --fold 0 --gpu 0

# Terminal 2 (GPU 1)
python train_single_fold_all_runs.py --fold 1 --gpu 1
```

**With 2 GPUs running simultaneously:**
- 5 folds × ~20 hours per fold = ~100 hours total
- Running 2 at a time = ~50 hours (~2 days)
- Do this for both ATLAS and DWI = ~4 days total

**Much better than 60 experiments × 2 days = 120 days sequentially!**

---

## Timeline Estimates

### Single Fold (4 experiments × 3 runs = 12 total)
- **With optimized DataLoaders**: 18-30 hours
- **With old num_workers=0**: 40-70 hours

### All 5 Folds (60 experiments total)

**Sequential (1 GPU):**
- ATLAS: 5 folds × 20 hours = 100 hours (~4 days)
- DWI: 5 folds × 20 hours = 100 hours (~4 days)
- **Total: ~8 days**

**Parallel (2 GPUs):**
- ATLAS on GPU 1: ~4 days
- DWI on GPU 0: ~4 days (running simultaneously)
- **Total: ~4 days**

**Super Parallel (2 GPUs, interleaved):**
- GPU 0: Fold 0 ATLAS, Fold 1 ATLAS, Fold 2 ATLAS
- GPU 1: Fold 0 DWI, Fold 1 DWI, Fold 2 DWI
- Keep switching to maximize both GPUs
- **Total: ~4-5 days**

---

## Recommended Workflow

### Day 1-2: Validate Setup
1. Test single fold to confirm optimization works:
   ```bash
   python train_single_fold_all_runs.py --fold 0 --gpu 1
   ```
2. Monitor GPU utilization (should be 90-100%)
3. Check epoch times are ~25-40 min (not 50-70 min)
4. Verify validation doesn't hang

### Day 2-6: Run All Folds
If test successful, launch remaining folds:

**Option A: Conservative (sequential)**
```bash
# Run folds one by one
python train_single_fold_all_runs.py --fold 0
python train_single_fold_all_runs.py --fold 1
python train_single_fold_all_runs.py --fold 2
python train_single_fold_all_runs.py --fold 3
python train_single_fold_all_runs.py --fold 4
```

**Option B: Aggressive (parallel with 2 GPUs)**
```bash
# Terminal 1 (GPU 0) - DWI
python train_single_fold_dwi.py --fold 0 --gpu 0
# When done:
python train_single_fold_dwi.py --fold 1 --gpu 0
# etc...

# Terminal 2 (GPU 1) - ATLAS
python train_single_fold_all_runs.py --fold 0 --gpu 1
# When done:
python train_single_fold_all_runs.py --fold 1 --gpu 1
# etc...
```

### Day 7: Analysis
- All experiments complete
- Collect results
- Generate tables/plots
- Write paper

---

## Monitoring Progress

### Check What's Running
```bash
# See active Python processes
ps aux | grep python

# See GPU usage
nvidia-smi

# Watch GPU usage live
watch -n 1 nvidia-smi
```

### Check Logs
```bash
# ATLAS logs
ls -lht /home/pahm409/ablation_ds_main_only/logs/

# DWI logs
ls -lht /home/pahm409/dwi_scratch_5fold/logs/

# View specific log
tail -f /home/pahm409/ablation_ds_main_only/logs/Exp1_Random_Baseline/fold_0_run_0.log
```

### Check Completed Experiments
```bash
# See what's finished for fold 0
ls -R /home/pahm409/ablation_ds_main_only/Random_Init/none/fold_0/
ls -R /home/pahm409/ablation_ds_main_only/Random_Init/mkdc_ds/fold_0/
ls -R /home/pahm409/ablation_ds_main_only/SimCLR_Pretrained/none/fold_0/
ls -R /home/pahm409/ablation_ds_main_only/SimCLR_Pretrained/mkdc_ds/fold_0/
```

---

## Troubleshooting

### If Validation Still Hangs
The optimized scripts use `num_workers=1` for validation. If hanging persists:

1. Open the optimized script
2. Change validation DataLoader to `num_workers=0`
3. Training will still be fast, validation slightly slower but safe

### If Running Out of Memory
Reduce batch size in the fold script:
```python
BATCH_SIZE = 6  # Instead of 8
```

### If GPU Not Fully Utilized
Increase training workers in optimized scripts:
```python
num_workers=6,  # Instead of 4
```

### If Process Killed
Script will auto-resume from checkpoint on next run of same fold

---

## Files Summary

1. **train_single_fold_all_runs.py** - ATLAS fold-level training
2. **train_single_fold_dwi.py** - DWI fold-level training
3. **corr_optimized.py** - Optimized ATLAS training script (uses num_workers=4)
4. **train_dwi_scratch_optimized.py** - Optimized DWI training script (uses num_workers=4)

---

## Key Points

✅ **Pick a fold, get all 3 runs done automatically**
✅ **Auto-skips completed experiments**
✅ **Auto-resumes from checkpoints**
✅ **2-3x faster with optimized DataLoaders**
✅ **Real-time progress tracking**
✅ **All 5 folds in ~4-8 days (instead of 4 months)**

**You can now demonstrate robust cross-validation without cherry-picking, in a reasonable timeframe!**

# DataLoader Optimization Guide

## Problem Summary

You were experiencing two related but opposite issues:

1. **Original Problem**: During validation, the training would hang/timeout when using multiple workers
2. **Your Solution**: Set `num_workers=0` for both training and validation
3. **New Problem**: Per-epoch time increased dramatically because all data loading happens on the main thread

## Root Cause Analysis

### Why Validation Was Hanging

The hanging during validation typically occurs due to:
- **Worker process management**: PyTorch spawns new worker processes for each DataLoader
- **Memory pressure**: Workers hold data in memory, causing fragmentation
- **Synchronization issues**: Validation switching from training mode can cause worker deadlocks
- **Large dataset sizes**: Your full-volume validation with complex preprocessing

### Why num_workers=0 Slows Training

With `num_workers=0`:
- **Synchronous loading**: All data preprocessing happens on the main GPU thread
- **GPU starvation**: GPU waits idle while CPU loads/preprocesses the next batch
- **No prefetching**: Can't prepare next batch while current batch is training
- **Pipeline breakdown**: The efficient data→GPU pipeline is broken

**Your preprocessing pipeline per batch:**
1. Load NIfTI files from disk (I/O bound)
2. Apply random cropping (CPU bound)
3. Apply data augmentation (CPU bound)
4. Convert to tensors (CPU bound)
5. Transfer to GPU (GPU bound)

With `num_workers=0`, steps 1-4 **block** the GPU during training!

## Optimal Solution: Asymmetric Configuration

### Strategy: Different Settings for Train vs Validation

**Training DataLoader** (speed-critical, ~1000+ batches per epoch):
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=4,              # Parallel data loading
    persistent_workers=True,     # Keep workers alive between epochs
    prefetch_factor=2,           # Prefetch 2 batches per worker
    pin_memory=True, 
    drop_last=True,
    timeout=0                    # Disable timeout
)
```

**Validation DataLoader** (happens once per epoch, ~100-200 batches):
```python
val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=1,               # Single worker to avoid hanging
    pin_memory=True,
    timeout=0                    # Disable timeout
)
```

### Key Parameters Explained

#### `num_workers`
- **Training: 4** - Creates 4 worker processes for parallel data loading
- **Validation: 1** - Single worker minimizes process management overhead and hanging risk
- Each worker runs independently, loading data in parallel
- Rule of thumb: 2-8 workers depending on CPU cores (you likely have 8-16 cores)

#### `persistent_workers=True`
- **Only for training** (requires num_workers > 0)
- Workers stay alive between epochs instead of respawning
- **Huge benefit**: Eliminates ~10-30 seconds of worker startup time per epoch
- **Memory trade-off**: Workers hold some data in memory, but worth it for speed

#### `prefetch_factor=2`
- **Only for training** (requires num_workers > 0)
- Each worker prefetches 2 batches ahead
- **Result**: GPU never starves waiting for data
- **Memory**: Uses 4 workers × 2 batches × batch_size worth of RAM

#### `timeout=0`
- **Critical**: Disables DataLoader timeout
- Prevents spurious "DataLoader worker timeout" errors
- PyTorch default timeout (often causes false positives with complex preprocessing)

## Expected Performance Impact

### Before (num_workers=0):
```
Epoch time breakdown:
├── Training: ~45-60 minutes (GPU idle during data loading)
├── Validation: ~8-12 minutes (safe, no hanging)
└── Total per epoch: ~53-72 minutes
```

### After (optimized configuration):
```
Epoch time breakdown:
├── Training: ~15-25 minutes (GPU fully utilized, parallel loading)
├── Validation: ~10-15 minutes (single worker, slightly slower but safe)
└── Total per epoch: ~25-40 minutes
```

**Expected speedup: 2-3x per epoch** (mainly from training acceleration)

For your 100-epoch experiments:
- Before: ~88-120 hours per experiment (3.7-5 days)
- After: ~40-67 hours per experiment (1.7-2.8 days)
- **Savings: ~2 days per experiment × 60 experiments = 120 days total saved!**

## Why This Works

### During Training (Most of the Time)
1. **4 workers continuously load data in parallel**
2. While GPU trains on batch N, workers prepare batches N+1, N+2, ..., N+8
3. **Zero GPU idle time** - data is always ready
4. `persistent_workers` means no restart overhead between epochs

### During Validation (Once per Epoch)
1. **Single worker** reduces process management complexity
2. Validation is shorter, so slight slowdown is acceptable
3. **Avoids hanging** - simpler process model, less memory pressure
4. Still much faster than your original validation-only bottleneck

## Alternative Configurations to Try

If you still experience hanging with `num_workers=1` during validation:

### Option A: Zero workers for validation only
```python
val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=0,      # Completely synchronous
    pin_memory=True
)
```
- Validation will be slower (~15-20 min instead of ~10-15 min)
- But training still fast (~15-25 min)
- Total epoch time: ~30-45 minutes (still 2x better than before!)

### Option B: Adjust number of training workers

If your system has limited RAM or CPU cores:
```python
# Try 2 workers instead of 4
train_loader = DataLoader(
    ...,
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    ...
)
```

### Option C: Disable prefetching if memory constrained
```python
train_loader = DataLoader(
    ...,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=1,  # Reduced from 2
    ...
)
```

## Monitoring and Debugging

### Check GPU Utilization During Training
```bash
watch -n 1 nvidia-smi
```

**Good signs:**
- GPU utilization: 90-100% during training
- GPU memory stable
- Power usage near maximum

**Bad signs:**
- GPU utilization: <80% (might need more workers)
- GPU utilization: Fluctuating wildly (data loading bottleneck)
- Memory steadily increasing (memory leak)

### Check if Workers are Active
Add this to your training loop temporarily:
```python
# At start of epoch
print(f"DataLoader workers: {train_loader.num_workers}")
print(f"Persistent workers: {train_loader.persistent_workers}")

# After first batch
print(f"First batch loaded successfully with {train_loader.num_workers} workers")
```

### Monitor for Hanging
If validation starts hanging again:
1. Check system memory: `htop` or `free -h`
2. Check for zombie processes: `ps aux | grep python`
3. Try reducing validation batch size
4. Try `num_workers=0` for validation as fallback

## Implementation Steps

### Step 1: Backup Your Current Scripts
```bash
cp corr.py corr_backup.py
cp train_dwi_scratch.py train_dwi_scratch_backup.py
```

### Step 2: Use the Optimized Scripts
The optimized versions are:
- `corr_optimized.py` - For ATLAS T1 experiments
- `train_dwi_scratch_optimized.py` - For ISLES DWI experiments

### Step 3: Test on a Single Fold
Before running all 60 experiments:
```bash
# Test 1 epoch with new settings
python corr_optimized.py --fold 0 --run-id 0 --epochs 1 --batch-size 8 \
    --output-dir ./test_optimized --attention none
```

Monitor for:
- Training speed increase
- Validation completes without hanging
- GPU utilization

### Step 4: Run Full Experiments
Once verified, update your wrapper scripts to use the optimized versions:
```python
# In train_all_experiments_robust.py, line 217
cmd = [
    'python', 'corr_optimized.py',  # Changed from 'corr.py'
    ...
]

# In train_all_dwi_experiments.py, similar change
cmd = [
    'python', 'train_dwi_scratch_optimized.py',  # Updated
    ...
]
```

## Files Provided

1. **corr_optimized.py** - ATLAS training with optimized DataLoaders
2. **train_dwi_scratch_optimized.py** - DWI training with optimized DataLoaders
3. **This guide** - Complete explanation and troubleshooting

## Summary

**What changed:**
- Training: `num_workers=0` → `num_workers=4` with `persistent_workers=True`
- Validation: `num_workers=0` → `num_workers=1` (conservative to avoid hanging)
- Added `timeout=0` to prevent spurious timeout errors
- Added `prefetch_factor=2` for better pipeline efficiency

**Expected result:**
- 2-3x faster per epoch
- Training phase: 2-4x speedup
- Validation phase: Similar or slightly slower (but safe)
- Overall: ~2 days saved per experiment

**Risk mitigation:**
- Validation uses minimal workers (1) to avoid hanging
- Can fall back to `num_workers=0` for validation if needed
- Training speedup alone makes this worthwhile
- Easy to revert if issues arise

## Next Steps

1. Test the optimized scripts on a single fold/run
2. Monitor GPU utilization and timing
3. If validation hangs, use `num_workers=0` for val_loader only
4. Once stable, update wrapper scripts and run all experiments

Good luck with your MICCAI submission!

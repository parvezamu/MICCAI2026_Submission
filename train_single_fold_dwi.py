"""
train_single_fold_dwi.py - Train ALL 3 runs for a specific fold (DWI experiments)

Usage:
    python train_single_fold_dwi.py --fold 0    # Runs fold 0: run_0, run_1, run_2
    python train_single_fold_dwi.py --fold 1    # Runs fold 1: run_0, run_1, run_2
    
This runs all 4 experiments × 3 runs = 12 experiments for the chosen fold
Estimated time per fold: ~2-3 days (with optimized DataLoaders)
"""

import subprocess
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "/home/pahm409/dwi_scratch_5fold"
GPU_ID = "0"

# Training parameters
RUNS = [0, 1, 2]
EPOCHS = 100
BATCH_SIZE = 8
PATCHES_PER_VOLUME = 10
LESION_FOCUS_RATIO = 0.7

# Experiment configurations
EXPERIMENTS = [
    {
        'name': 'DWI_Exp1_Baseline',
        'use_mkdc': False,
        'deep_supervision': False
    },
    {
        'name': 'DWI_Exp2_MKDC_DS',
        'use_mkdc': True,
        'deep_supervision': True
    },
    {
        'name': 'DWI_Exp3_MKDC',
        'use_mkdc': True,
        'deep_supervision': False
    },
    {
        'name': 'DWI_Exp4_DS',
        'use_mkdc': False,
        'deep_supervision': True
    }
]

CLEANUP_WAIT_SECONDS = 10


# ============================================================================
# EXPERIMENT STATUS DETECTION
# ============================================================================

def get_config_name(use_mkdc, deep_supervision):
    """Derive config name from flags (matches train_dwi_scratch.py logic)"""
    if use_mkdc and deep_supervision:
        return 'mkdc_ds'
    elif use_mkdc:
        return 'mkdc'
    elif deep_supervision:
        return 'ds'
    else:
        return 'baseline'


def get_experiment_status(base_dir, config_name, fold, run):
    """Check if experiment is completed"""
    fold_dir = Path(base_dir) / config_name / f'fold_{fold}' / f'run_{run}'
    
    if not fold_dir.exists():
        return 'not_started', None, None
    
    exp_dirs = sorted(fold_dir.glob('exp_*'))
    if not exp_dirs:
        return 'not_started', None, None
    
    latest_exp = exp_dirs[-1]
    
    # Check training log
    log_file = latest_exp / 'training_log.csv'
    if not log_file.exists():
        return 'in_progress', None, None
    
    # Check if completed 100 epochs
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if last_line:
                    epoch = int(last_line.split(',')[0])
                    if epoch >= EPOCHS - 1:
                        return 'completed', epoch, None
                    else:
                        # Find latest checkpoint
                        checkpoint_dir = latest_exp / 'checkpoints'
                        if checkpoint_dir.exists():
                            checkpoints = sorted(checkpoint_dir.glob('epoch_*.pth'))
                            if checkpoints:
                                return 'in_progress', epoch, str(checkpoints[-1])
    except Exception as e:
        print(f"Warning: Error reading log file: {e}")
    
    return 'in_progress', None, None


# ============================================================================
# TRAINING EXECUTOR
# ============================================================================

def run_single_experiment(exp_config, fold, run, output_dir, gpu_id, resume_checkpoint=None):
    """Run a single experiment with real-time output"""
    
    cmd = [
        'python', 'train_dwi_scratch_optimized.py',  # Use optimized version
        '--fold', str(fold),
        '--run-id', str(run),
        '--epochs', str(EPOCHS),
        '--batch-size', str(BATCH_SIZE),
        '--output-dir', output_dir,
        '--patches-per-volume', str(PATCHES_PER_VOLUME),
        '--lesion-focus-ratio', str(LESION_FOCUS_RATIO)
    ]
    
    # Add flags based on config (NOT --config-name)
    if exp_config['use_mkdc']:
        cmd.append('--use-mkdc')
    
    if exp_config['deep_supervision']:
        cmd.append('--deep-supervision')
    
    if resume_checkpoint:
        cmd.extend(['--resume-checkpoint', resume_checkpoint])
    
    # Environment with GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # Log file
    log_dir = Path(output_dir) / 'logs' / exp_config['name']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'fold_{fold}_run_{run}.log'
    
    print(f"\n{'='*80}")
    print(f"STARTING: {exp_config['name']}")
    print(f"  Fold {fold}, Run {run}")
    if resume_checkpoint:
        print(f"  📂 Resuming from: {resume_checkpoint}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log file: {log_file}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Real-time output
            for line in process.stdout:
                print(line, end='', flush=True)
                log_f.write(line)
                log_f.flush()
            
            process.wait()
            result_code = process.returncode
        
        duration = time.time() - start_time
        success = (result_code == 0)
        
        if success:
            print(f"\n{'='*80}")
            print(f"✅ COMPLETED: {exp_config['name']} - Fold {fold}, Run {run}")
            print(f"   Duration: {timedelta(seconds=int(duration))}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"❌ FAILED: {exp_config['name']} - Fold {fold}, Run {run}")
            print(f"   Return code: {result_code}")
            print(f"   Check log: {log_file}")
            print(f"{'='*80}\n")
        
        return success, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"❌ ERROR: {exp_config['name']} - Fold {fold}, Run {run}")
        print(f"   Exception: {str(e)}")
        print(f"{'='*80}\n")
        return False, duration


def cleanup_gpu(wait_seconds=10):
    """Wait for GPU cleanup"""
    print(f"\n{'='*80}")
    print(f"🧹 GPU cleanup: waiting {wait_seconds} seconds...")
    print(f"{'='*80}")
    for i in range(wait_seconds, 0, -1):
        print(f"   {i}...", end='\r', flush=True)
        time.sleep(1)
    print("   ✓ Ready for next experiment" + " "*20)
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train all runs for a specific fold (DWI)')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help='Fold to train (0-4)')
    parser.add_argument('--gpu', type=str, default=GPU_ID,
                        help='GPU ID to use')
    args = parser.parse_args()
    
    fold = args.fold
    gpu_id = args.gpu
    
    print("\n" + "="*80)
    print(f"DWI FOLD {fold} - ALL RUNS TRAINING")
    print("="*80)
    print(f"Configuration:")
    print(f"  Fold: {fold}")
    print(f"  Runs per experiment: {len(RUNS)}")
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Total runs to complete: {len(EXPERIMENTS) * len(RUNS)}")
    print(f"  GPU: {gpu_id}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Track progress
    total_experiments = len(EXPERIMENTS) * len(RUNS)
    completed = 0
    skipped = 0
    failed = 0
    
    overall_start = time.time()
    
    # Build experiment queue
    queue = []
    for exp in EXPERIMENTS:
        for run in RUNS:
            # Derive config_name from flags
            config_name = get_config_name(exp['use_mkdc'], exp['deep_supervision'])
            
            # Check status
            status, last_epoch, checkpoint = get_experiment_status(
                OUTPUT_DIR, config_name, fold, run
            )
            
            if status == 'completed':
                print(f"⏭️  SKIP: {exp['name']} - Fold {fold}, Run {run} (already completed)")
                skipped += 1
            else:
                queue.append({
                    'exp': exp,
                    'run': run,
                    'status': status,
                    'checkpoint': checkpoint,
                    'last_epoch': last_epoch
                })
    
    print(f"\n📊 Experiment Queue Summary:")
    print(f"   Total: {total_experiments}")
    print(f"   Already completed: {skipped}")
    print(f"   To run: {len(queue)}")
    print(f"   Estimated time: ~{len(queue) * 1.5:.1f} - {len(queue) * 2.5:.1f} hours\n")
    
    if not queue:
        print("✅ All experiments for this fold already completed!")
        return
    
    # Execute queue
    for idx, item in enumerate(queue, 1):
        exp = item['exp']
        run = item['run']
        resume_checkpoint = item['checkpoint']
        
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {idx}/{len(queue)}")
        print(f"# {exp['name']} - Fold {fold}, Run {run}")
        print(f"{'#'*80}\n")
        
        success, duration = run_single_experiment(
            exp, fold, run, OUTPUT_DIR, gpu_id, resume_checkpoint
        )
        
        if success:
            completed += 1
        else:
            failed += 1
        
        # Cleanup between experiments
        if idx < len(queue):
            cleanup_gpu(CLEANUP_WAIT_SECONDS)
        
        # Progress update
        total_done = completed + skipped
        remaining = total_experiments - total_done - failed
        elapsed = time.time() - overall_start
        
        if completed > 0:
            avg_time = elapsed / completed
            eta = avg_time * remaining
            
            print(f"\n📈 OVERALL PROGRESS:")
            print(f"   Fold {fold}: {total_done}/{total_experiments} complete")
            print(f"   ✅ Completed: {completed}")
            print(f"   ⏭️  Skipped: {skipped}")
            print(f"   ❌ Failed: {failed}")
            print(f"   ⏳ Remaining: {remaining}")
            print(f"   ⏱️  Elapsed: {timedelta(seconds=int(elapsed))}")
            print(f"   🎯 ETA: {timedelta(seconds=int(eta))}")
            print()
    
    # Final summary
    total_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print(f"DWI FOLD {fold} TRAINING COMPLETE!")
    print("="*80)
    print(f"Summary:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  ✅ Completed: {completed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏱️  Total time: {timedelta(seconds=int(total_time))}")
    print("="*80 + "\n")
    
    if failed > 0:
        print("⚠️  Some experiments failed. Check logs for details.")
        return 1
    else:
        print("🎉 All experiments completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())

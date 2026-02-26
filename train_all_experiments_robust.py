"""
train_all_experiments_robust.py - Complete robust training system with REAL-TIME OUTPUT

Handles:
- 4 Experiments: Random_Init vs SimCLR_Pretrained × none vs mkdc_ds
- 5 folds × 3 runs = 15 experiments each = 60 total experiments
- Auto-detection of completed experiments
- Resume from checkpoint for interrupted experiments
- Progress tracking and time estimation
- REAL-TIME training progress display
"""

import subprocess
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMCLR_CHECKPOINT = "/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR = "/home/pahm409/ablation_ds_main_only"
GPU_ID = "1"

# Training parameters
FOLDS = [0, 1, 2, 3, 4]
RUNS = [0, 1, 2]
EPOCHS = 100
BATCH_SIZE = 8
PATCHES_PER_VOLUME = 10
LESION_FOCUS_RATIO = 0.7

# Experiment configurations
EXPERIMENTS = [
    {
        'name': 'Exp1_Random_Baseline',
        'init': 'Random_Init',
        'attention': 'none',
        'deep_supervision': False,
        'pretrained': None
    },
    {
        'name': 'Exp2_Random_MKDC_DS',
        'init': 'Random_Init',
        'attention': 'mkdc',
        'deep_supervision': True,
        'pretrained': None
    },
    {
        'name': 'Exp3_SimCLR_Baseline',
        'init': 'SimCLR_Pretrained',
        'attention': 'none',
        'deep_supervision': False,
        'pretrained': SIMCLR_CHECKPOINT
    },
    {
        'name': 'Exp4_SimCLR_MKDC_DS',
        'init': 'SimCLR_Pretrained',
        'attention': 'mkdc',
        'deep_supervision': True,
        'pretrained': SIMCLR_CHECKPOINT
    }
]

# System settings
CLEANUP_WAIT_SECONDS = 10
STOP_ON_ERROR = False


# ============================================================================
# EXPERIMENT STATUS DETECTION
# ============================================================================

def get_experiment_status(base_dir, init_type, attention, ds, fold, run):
    """
    Check the status of a specific experiment
    
    Returns:
        status: 'not_started', 'in_progress', 'completed'
        last_epoch: Last completed epoch (or None)
        checkpoint_path: Path to checkpoint (or None)
    """
    # Build path
    ds_suffix = '_ds' if ds else ''
    fold_dir = Path(base_dir) / init_type / f'{attention}{ds_suffix}' / f'fold_{fold}' / f'run_{run}'
    
    if not fold_dir.exists():
        return 'not_started', None, None
    
    # Find the most recent experiment directory
    exp_dirs = sorted(fold_dir.glob('exp_*'))
    if not exp_dirs:
        return 'not_started', None, None
    
    latest_exp = exp_dirs[-1]
    
    # Check training log
    log_file = latest_exp / 'training_log.csv'
    if not log_file.exists():
        return 'not_started', None, None
    
    # Read last epoch from log
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) <= 1:  # Only header or empty
        return 'not_started', None, None
    
    last_line = lines[-1].strip()
    if last_line:
        last_epoch = int(last_line.split(',')[0])
    else:
        last_epoch = 0
    
    # Check if completed
    if last_epoch >= EPOCHS:
        return 'completed', last_epoch, None
    
    # Check for checkpoint
    checkpoint_path = latest_exp / 'checkpoints' / 'best_model.pth'
    if checkpoint_path.exists():
        return 'in_progress', last_epoch, str(checkpoint_path)
    
    return 'in_progress', last_epoch, None


def scan_all_experiments():
    """
    Scan all experiments and return status summary
    
    Returns:
        dict: {(exp_idx, fold, run): (status, last_epoch, checkpoint_path)}
    """
    status_dict = {}
    
    for exp_idx, exp in enumerate(EXPERIMENTS):
        for fold in FOLDS:
            for run in RUNS:
                status, last_epoch, checkpoint = get_experiment_status(
                    OUTPUT_DIR,
                    exp['init'],
                    exp['attention'],
                    exp['deep_supervision'],
                    fold,
                    run
                )
                status_dict[(exp_idx, fold, run)] = {
                    'status': status,
                    'last_epoch': last_epoch,
                    'checkpoint': checkpoint,
                    'exp_name': exp['name']
                }
    
    return status_dict


def print_status_summary(status_dict):
    """Print a nice summary of all experiment statuses"""
    
    print("\n" + "="*80)
    print("EXPERIMENT STATUS SUMMARY")
    print("="*80)
    
    for exp_idx, exp in enumerate(EXPERIMENTS):
        print(f"\n{exp['name']}:")
        print("-" * 70)
        
        for fold in FOLDS:
            fold_status = []
            for run in RUNS:
                key = (exp_idx, fold, run)
                info = status_dict[key]
                
                if info['status'] == 'completed':
                    fold_status.append(f"✅ R{run}")
                elif info['status'] == 'in_progress':
                    epoch = info['last_epoch'] or 0
                    fold_status.append(f"🔄 R{run}(E{epoch})")
                else:
                    fold_status.append(f"⬜ R{run}")
            
            print(f"  Fold {fold}: {' | '.join(fold_status)}")
    
    # Overall statistics
    total = len(EXPERIMENTS) * len(FOLDS) * len(RUNS)
    completed = sum(1 for v in status_dict.values() if v['status'] == 'completed')
    in_progress = sum(1 for v in status_dict.values() if v['status'] == 'in_progress')
    not_started = sum(1 for v in status_dict.values() if v['status'] == 'not_started')
    
    print("\n" + "="*80)
    print(f"Overall: {completed}/{total} completed ({completed/total*100:.1f}%)")
    print(f"  ✅ Completed: {completed}")
    print(f"  🔄 In Progress: {in_progress}")
    print(f"  ⬜ Not Started: {not_started}")
    print("="*80 + "\n")


# ============================================================================
# TRAINING EXECUTOR WITH RESUME SUPPORT AND REAL-TIME OUTPUT
# ============================================================================

def run_single_experiment(exp_config, fold, run, output_dir, gpu_id, resume_checkpoint=None):
    """
    Run a single experiment with optional resume from checkpoint
    Shows REAL-TIME training progress
    
    Returns:
        success (bool): Whether completed successfully
        duration (float): Time taken in seconds
    """
    
    cmd = [
        'python', 'corr.py',
        '--fold', str(fold),
        '--run-id', str(run),
        '--epochs', str(EPOCHS),
        '--batch-size', str(BATCH_SIZE),
        '--output-dir', output_dir,
        '--attention', exp_config['attention'],
        '--patches-per-volume', str(PATCHES_PER_VOLUME),
        '--lesion-focus-ratio', str(LESION_FOCUS_RATIO)
    ]
    
    # Add pretrained checkpoint if specified
    if exp_config['pretrained']:
        cmd.extend(['--pretrained-checkpoint', exp_config['pretrained']])
    
    # Add deep supervision if specified
    if exp_config['deep_supervision']:
        cmd.append('--deep-supervision')
    
    # Add resume checkpoint if provided
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
        # Use Popen for real-time output
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
            
            # Read and display output in real-time
            for line in process.stdout:
                # Print to screen (real-time)
                print(line, end='', flush=True)
                # Write to log file
                log_f.write(line)
                log_f.flush()
            
            # Wait for process to complete
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
# PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'overall_progress.json'
        self.start_time = time.time()
        self.times = []
    
    def log_completion(self, exp_name, fold, run, success, duration):
        """Log a completed experiment"""
        self.times.append(duration)
        
        log_entry = {
            'experiment': exp_name,
            'fold': fold,
            'run': run,
            'success': success,
            'duration_seconds': duration,
            'duration_formatted': str(timedelta(seconds=int(duration))),
            'timestamp': datetime.now().isoformat()
        }
        
        detail_log = self.log_dir / 'experiment_log.jsonl'
        with open(detail_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def print_stats(self, completed, total):
        """Print current statistics"""
        if self.times:
            avg_time = sum(self.times) / len(self.times)
            remaining = total - completed
            estimated_remaining = avg_time * remaining
        else:
            avg_time = 0
            estimated_remaining = 0
        
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("OVERALL PROGRESS")
        print("="*80)
        print(f"Completed: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"Remaining: {total - completed} experiments")
        
        if avg_time > 0:
            print(f"Average time: {timedelta(seconds=int(avg_time))}")
            print(f"Estimated remaining: {timedelta(seconds=int(estimated_remaining))}")
            
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            print(f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"Elapsed: {timedelta(seconds=int(elapsed))}")
        print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with smart resume"""
    
    print("\n" + "="*80)
    print("COMPLETE ABLATION STUDY - ALL 4 EXPERIMENTS")
    print("="*80)
    print(f"Total configurations: {len(EXPERIMENTS)}")
    print(f"Folds per config: {len(FOLDS)}")
    print(f"Runs per fold: {len(RUNS)}")
    print(f"Total experiments: {len(EXPERIMENTS) * len(FOLDS) * len(RUNS)}")
    print(f"Epochs per experiment: {EPOCHS}")
    print("="*80)
    
    # Scan existing experiments
    print("\n🔍 Scanning existing experiments...")
    status_dict = scan_all_experiments()
    print_status_summary(status_dict)
    
    # Ask user to confirm
    response = input("\nProceed with training? [Y/n]: ").strip().lower()
    if response == 'n':
        print("Aborted.")
        return
    
    # Initialize tracker
    tracker = ProgressTracker(OUTPUT_DIR)
    
    total_experiments = len(EXPERIMENTS) * len(FOLDS) * len(RUNS)
    completed_count = sum(1 for v in status_dict.values() if v['status'] == 'completed')
    failed_count = 0
    failed_list = []
    
    overall_start = time.time()
    
    # Train in order: Exp1 → Exp2 → Exp3 → Exp4
    for exp_idx, exp_config in enumerate(EXPERIMENTS):
        print("\n" + "="*80)
        print(f"STARTING {exp_config['name']}")
        print("="*80 + "\n")
        
        for fold in FOLDS:
            for run in RUNS:
                key = (exp_idx, fold, run)
                info = status_dict[key]
                
                # Skip if completed
                if info['status'] == 'completed':
                    print(f"⏭️  Skipping {exp_config['name']} - Fold {fold}, Run {run} (already completed)")
                    continue
                
                # Run experiment (with resume if in_progress)
                resume_checkpoint = info['checkpoint'] if info['status'] == 'in_progress' else None
                
                success, duration = run_single_experiment(
                    exp_config=exp_config,
                    fold=fold,
                    run=run,
                    output_dir=OUTPUT_DIR,
                    gpu_id=GPU_ID,
                    resume_checkpoint=resume_checkpoint
                )
                
                # Log result
                tracker.log_completion(exp_config['name'], fold, run, success, duration)
                
                if success:
                    completed_count += 1
                    # Update status
                    status_dict[key]['status'] = 'completed'
                else:
                    failed_count += 1
                    failed_list.append((exp_config['name'], fold, run))
                    
                    if STOP_ON_ERROR:
                        print("\n⛔ Stopping due to error")
                        break
                
                # Print progress
                tracker.print_stats(completed_count, total_experiments)
                
                # GPU cleanup (unless it's the last experiment)
                if not (exp_idx == len(EXPERIMENTS)-1 and fold == FOLDS[-1] and run == RUNS[-1]):
                    cleanup_gpu(CLEANUP_WAIT_SECONDS)
            
            if STOP_ON_ERROR and failed_count > 0:
                break
        
        if STOP_ON_ERROR and failed_count > 0:
            break
    
    # Final summary
    overall_duration = time.time() - overall_start
    
    print("\n" + "="*80)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"Total time: {timedelta(seconds=int(overall_duration))}")
    print(f"Successful: {completed_count}/{total_experiments}")
    print(f"Failed: {failed_count}/{total_experiments}")
    
    if failed_list:
        print(f"\n❌ Failed experiments:")
        for exp_name, fold, run in failed_list:
            print(f"   - {exp_name}: Fold {fold}, Run {run}")
    
    print(f"\nAll results in: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Save final summary
    summary = {
        'total_experiments': total_experiments,
        'completed': completed_count,
        'failed': failed_count,
        'failed_experiments': [
            {'experiment': e, 'fold': f, 'run': r} 
            for e, f, r in failed_list
        ],
        'total_duration_seconds': overall_duration,
        'total_duration_formatted': str(timedelta(seconds=int(overall_duration))),
        'completion_time': datetime.now().isoformat()
    }
    
    summary_file = Path(OUTPUT_DIR) / 'ablation_study_complete_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"📊 Final summary saved to: {summary_file}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("You can resume by running this script again.")
        print("Completed experiments will be automatically detected and skipped.\n")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


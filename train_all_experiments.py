"""
train_all_experiments_structured.py

Train ALL experimental conditions with CLEAR structure:

EXPERIMENT 1: Random Init + Baseline (no DS)
  Fold 0: Run 0, Run 1, Run 2
  Fold 1: Run 0, Run 1, Run 2
  Fold 2: Run 0, Run 1, Run 2
  Fold 3: Run 0, Run 1, Run 2
  Fold 4: Run 0, Run 1, Run 2
  = 15 models

EXPERIMENT 2: Random Init + MKDC + DS (main-only)
  Fold 0: Run 0, Run 1, Run 2
  Fold 1: Run 0, Run 1, Run 2
  Fold 2: Run 0, Run 1, Run 2
  Fold 3: Run 0, Run 1, Run 2
  Fold 4: Run 0, Run 1, Run 2
  = 15 models

EXPERIMENT 3: SimCLR + Baseline (no DS)
  Fold 0: Run 0, Run 1, Run 2
  Fold 1: Run 0, Run 1, Run 2
  Fold 2: Run 0, Run 1, Run 2
  Fold 3: Run 0, Run 1, Run 2
  Fold 4: Run 0, Run 1, Run 2
  = 15 models

EXPERIMENT 4: SimCLR + MKDC + DS (main-only)
  Fold 0: Run 0, Run 1, Run 2
  Fold 1: Run 0, Run 1, Run 2
  Fold 2: Run 0, Run 1, Run 2
  Fold 3: Run 0, Run 1, Run 2
  Fold 4: Run 0, Run 1, Run 2
  = 15 models

TOTAL: 4 experiments × 5 folds × 3 runs = 60 models

Author: Parvez
Date: January 2026
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime
import time


class ExperimentConfig:
    """Configuration for one experiment"""
    def __init__(self, exp_id, name, pretrained, attention, deep_supervision):
        self.exp_id = exp_id
        self.name = name
        self.pretrained = pretrained
        self.attention = attention
        self.deep_supervision = deep_supervision
    
    def __repr__(self):
        return f"Exp{self.exp_id}: {self.name}"


def run_single_training(
    experiment: ExperimentConfig,
    fold: int,
    run_id: int,
    output_dir: str,
    epochs: int,
    batch_size: int,
    gpu_id: str
):
    """
    Run a SINGLE training instance
    
    Returns:
        success (bool), duration (seconds), error (str or None)
    """
    
    cmd = [
        'python', 'train_segmentation_corrected_DS_MAIN_ONLY.py',
        '--fold', str(fold),
        '--run-id', str(run_id),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--output-dir', output_dir,
        '--attention', experiment.attention,
        '--patches-per-volume', '10',
        '--lesion-focus-ratio', '0.7',
        '--validate-recon-every', '5',
        '--save-nifti-every', '10'
    ]
    
    if experiment.pretrained:
        cmd.extend(['--pretrained-checkpoint', experiment.pretrained])
    
    if experiment.deep_supervision:
        cmd.append('--deep-supervision')
        # Main-only loss: weights = [1.0, 0.0, 0.0, 0.0]
        cmd.extend(['--ds-weights', '1.0', '0.0', '0.0', '0.0'])
    
    # Set GPU
    env = {**subprocess.os.environ, 'CUDA_VISIBLE_DEVICES': gpu_id}
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {experiment.name}")
    print(f"  Fold: {fold} | Run: {run_id}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, env=env)
        duration = time.time() - start_time
        return True, duration, None
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def print_experiment_structure():
    """Print the complete training structure"""
    print(f"\n{'='*80}")
    print("COMPLETE TRAINING STRUCTURE")
    print(f"{'='*80}\n")
    
    experiments = [
        "Experiment 1: Random Init + Baseline (no DS)",
        "Experiment 2: Random Init + MKDC + DS (main-only)",
        "Experiment 3: SimCLR + Baseline (no DS)",
        "Experiment 4: SimCLR + MKDC + DS (main-only)"
    ]
    
    for exp_name in experiments:
        print(f"{exp_name}")
        for fold in range(5):
            print(f"  Fold {fold}: Run 0, Run 1, Run 2")
        print(f"  = 15 models\n")
    
    print(f"TOTAL: 4 × 5 × 3 = 60 models")
    print(f"{'='*80}\n")


def train_all_experiments(
    simclr_checkpoint: str,
    output_dir: str,
    gpu_id: str,
    epochs: int,
    batch_size: int,
    resume_from: dict = None
):
    """
    Train all 60 models with EXPLICIT structure
    
    Args:
        simclr_checkpoint: Path to SimCLR checkpoint
        output_dir: Base output directory
        gpu_id: GPU to use
        epochs: Epochs per training
        batch_size: Batch size (should be 8)
        resume_from: Dict with 'experiment', 'fold', 'run' to resume from
    """
    
    # Define all 4 experiments
    experiments = [
        ExperimentConfig(
            exp_id=1,
            name='Random Init + Baseline',
            pretrained=None,
            attention='none',
            deep_supervision=False
        ),
        ExperimentConfig(
            exp_id=2,
            name='Random Init + MKDC + DS (main-only)',
            pretrained=None,
            attention='mkdc',
            deep_supervision=True
        ),
        ExperimentConfig(
            exp_id=3,
            name='SimCLR + Baseline',
            pretrained=simclr_checkpoint,
            attention='none',
            deep_supervision=False
        ),
        ExperimentConfig(
            exp_id=4,
            name='SimCLR + MKDC + DS (main-only)',
            pretrained=simclr_checkpoint,
            attention='mkdc',
            deep_supervision=True
        )
    ]
    
    # Setup logging
    log_dir = Path(output_dir) / 'training_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    # Track progress
    total_models = 60
    completed_models = 0
    failed_models = []
    
    start_time = datetime.now()
    
    # Print structure
    print_experiment_structure()
    
    # Open log file
    with open(log_file, 'w') as log:
        log.write(f"{'='*80}\n")
        log.write(f"TRAINING ALL EXPERIMENTS\n")
        log.write(f"Started: {start_time}\n")
        log.write(f"{'='*80}\n\n")
        
        # Loop through all experiments
        for experiment in experiments:
            
            # Skip if resuming
            if resume_from and experiment.exp_id < resume_from['experiment']:
                completed_models += 15  # Skip entire experiment
                continue
            
            print(f"\n{'#'*80}")
            print(f"# {experiment}")
            print(f"{'#'*80}\n")
            
            log.write(f"\n{'='*80}\n")
            log.write(f"{experiment}\n")
            log.write(f"{'='*80}\n\n")
            
            # Loop through all 5 folds
            for fold in range(5):
                
                # Skip if resuming
                if resume_from and experiment.exp_id == resume_from['experiment'] and fold < resume_from['fold']:
                    completed_models += 3  # Skip entire fold
                    continue
                
                print(f"\n{'-'*80}")
                print(f"FOLD {fold}")
                print(f"{'-'*80}\n")
                
                log.write(f"\nFold {fold}:\n")
                
                # Loop through 3 runs per fold
                for run_id in range(3):
                    
                    # Skip if resuming
                    if (resume_from and 
                        experiment.exp_id == resume_from['experiment'] and 
                        fold == resume_from['fold'] and 
                        run_id < resume_from['run']):
                        completed_models += 1
                        continue
                    
                    model_number = completed_models + 1
                    
                    print(f"\n{'='*80}")
                    print(f"MODEL {model_number}/{total_models}")
                    print(f"{experiment.name}")
                    print(f"Fold {fold}, Run {run_id}")
                    print(f"{'='*80}\n")
                    
                    # Train
                    success, duration, error = run_single_training(
                        experiment=experiment,
                        fold=fold,
                        run_id=run_id,
                        output_dir=output_dir,
                        epochs=epochs,
                        batch_size=batch_size,
                        gpu_id=gpu_id
                    )
                    
                    completed_models += 1
                    
                    # Log result
                    duration_str = f"{duration/3600:.2f}h"
                    
                    if success:
                        status = f"✓ SUCCESS ({duration_str})"
                        print(f"\n{status}\n")
                        log.write(f"  Run {run_id}: {status}\n")
                    else:
                        status = f"✗ FAILED ({duration_str}): {error}"
                        print(f"\n{status}\n")
                        log.write(f"  Run {run_id}: {status}\n")
                        
                        failed_models.append({
                            'experiment': experiment.exp_id,
                            'experiment_name': experiment.name,
                            'fold': fold,
                            'run': run_id,
                            'error': error
                        })
                    
                    # Print overall progress
                    print(f"{'='*80}")
                    print(f"OVERALL PROGRESS: {completed_models}/{total_models} models")
                    print(f"Failed: {len(failed_models)}")
                    elapsed = datetime.now() - start_time
                    print(f"Elapsed: {elapsed}")
                    if completed_models > 0:
                        avg_time = elapsed.total_seconds() / completed_models
                        remaining = (total_models - completed_models) * avg_time
                        print(f"Estimated remaining: {remaining/3600:.1f}h")
                    print(f"{'='*80}\n")
                    
                    log.flush()
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# ALL TRAINING COMPLETE")
    print(f"{'#'*80}\n")
    print(f"Total models: {total_models}")
    print(f"Successful: {completed_models - len(failed_models)}")
    print(f"Failed: {len(failed_models)}")
    print(f"Total duration: {total_duration}")
    print(f"\nLog file: {log_file}\n")
    
    if failed_models:
        print(f"Failed models:")
        for fail in failed_models:
            print(f"  {fail['experiment_name']}")
            print(f"    Fold {fail['fold']}, Run {fail['run']}")
            print(f"    Error: {fail['error']}\n")
    
    # Save summary JSON
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_duration_hours': total_duration.total_seconds() / 3600,
        'total_models': total_models,
        'successful_models': completed_models - len(failed_models),
        'failed_models': failed_models
    }
    
    summary_file = log_dir / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved: {summary_file}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train all 60 models (4 experiments × 5 folds × 3 runs)'
    )
    
    parser.add_argument('--simclr-checkpoint', type=str, required=True,
                       help='Path to SimCLR pretrained checkpoint')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/pahm409/ablation_ds_main_only',
                       help='Base output directory')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (recommended: 8)')
    
    # Resume options
    parser.add_argument('--resume-experiment', type=int, default=None,
                       choices=[1, 2, 3, 4],
                       help='Resume from this experiment')
    parser.add_argument('--resume-fold', type=int, default=None,
                       choices=[0, 1, 2, 3, 4],
                       help='Resume from this fold')
    parser.add_argument('--resume-run', type=int, default=None,
                       choices=[0, 1, 2],
                       help='Resume from this run')
    
    args = parser.parse_args()
    
    # Verify checkpoint
    if not Path(args.simclr_checkpoint).exists():
        print(f"Error: SimCLR checkpoint not found: {args.simclr_checkpoint}")
        sys.exit(1)
    
    # Prepare resume info
    resume_from = None
    if args.resume_experiment is not None:
        resume_from = {
            'experiment': args.resume_experiment,
            'fold': args.resume_fold or 0,
            'run': args.resume_run or 0
        }
    
    # Print configuration
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"SimCLR checkpoint: {args.simclr_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nTotal models: 60")
    print(f"  - 4 experiments")
    print(f"  - 5 folds per experiment")
    print(f"  - 3 runs per fold")
    
    if resume_from:
        print(f"\nResuming from:")
        print(f"  Experiment: {resume_from['experiment']}")
        print(f"  Fold: {resume_from['fold']}")
        print(f"  Run: {resume_from['run']}")
    
    print(f"\nEstimated time: ~240 hours (~10 days)")
    print(f"{'='*80}\n")
    
    response = input("Start training? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        sys.exit(0)
    
    # Start training
    train_all_experiments(
        simclr_checkpoint=args.simclr_checkpoint,
        output_dir=args.output_dir,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from=resume_from
    )


if __name__ == '__main__':
    main()

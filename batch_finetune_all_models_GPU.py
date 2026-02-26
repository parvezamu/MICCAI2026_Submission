#!/usr/bin/env python3
"""
batch_finetune_all_models_GPU.py

GPU-AWARE batch fine-tuning wrapper
Properly sets GPU BEFORE importing torch
"""

import subprocess
import json
from pathlib import Path
import argparse
import time
from datetime import datetime
import os


EXPERIMENTS = [
    {
        'name': 'Exp1_Random_Baseline',
        'init': 'Random_Init',
        'attention': 'none',
        'deep_supervision': False
    },
    {
        'name': 'Exp2_Random_MKDC_DS',
        'init': 'Random_Init',
        'attention': 'mkdc',
        'deep_supervision': True
    },
    {
        'name': 'Exp3_SimCLR_Baseline',
        'init': 'SimCLR_Pretrained',
        'attention': 'none',
        'deep_supervision': False
    },
    {
        'name': 'Exp4_SimCLR_MKDC_DS',
        'init': 'SimCLR_Pretrained',
        'attention': 'mkdc',
        'deep_supervision': True
    }
]


def find_best_checkpoint(base_dir, init_type, attention, ds, fold, run):
    """Find best_model.pth for a specific experiment/fold/run"""
    ds_suffix = '_ds' if ds else ''
    
    fold_dir = Path(base_dir) / init_type / f'{attention}{ds_suffix}' / f'fold_{fold}' / f'run_{run}'
    
    if not fold_dir.exists():
        return None
    
    exp_dirs = sorted(fold_dir.glob('exp_*'))
    if not exp_dirs:
        return None
    
    latest_exp = exp_dirs[-1]
    checkpoint = latest_exp / 'checkpoints' / 'best_model.pth'
    
    if checkpoint.exists():
        return str(checkpoint)
    
    return None


def run_finetuning_with_gpu(checkpoint_path, fold, output_dir, epochs=50, batch_size=8, 
                            decoder_lr=0.0001, encoder_lr_ratio=0.1, freeze_epochs=3,
                            isles_only=False, gpu_id="3"):
    """
    Run fine-tuning with PROPER GPU setting
    
    Key: Set CUDA_VISIBLE_DEVICES in environment BEFORE calling Python
    """
    
    # Create a temporary Python script with GPU set at the top
    temp_script = f"""#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import sys
sys.argv = [
    'finetune_on_isles_FIXED.py',
    '--pretrained-checkpoint', '{checkpoint_path}',
    '--fold', '{fold}',
    '--output-dir', '{output_dir}',
    '--epochs', '{epochs}',
    '--batch-size', '{batch_size}',
    '--decoder-lr', '{decoder_lr}',
    '--encoder-lr-ratio', '{encoder_lr_ratio}',
    '--freeze-encoder-epochs', '{freeze_epochs}'
]

if {isles_only}:
    sys.argv.append('--isles-only')

# Import and run
sys.path.insert(0, '.')
exec(open('finetune_on_isles_FIXED.py').read())
"""
    
    # Save temp script
    temp_file = Path(f'/tmp/finetune_gpu_{os.getpid()}.py')
    with open(temp_file, 'w') as f:
        f.write(temp_script)
    
    print(f"\n{'='*80}")
    print(f"Running fine-tuning on GPU {gpu_id}...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Fold: {fold}")
    print(f"{'='*80}\n")
    
    try:
        start_time = time.time()
        
        # Run with clean environment
        result = subprocess.run(
            ['python3', str(temp_file)],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours
        )
        
        duration = time.time() - start_time
        
        # Clean up temp file
        temp_file.unlink()
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS (took {duration/60:.1f} minutes)")
            
            # Parse results
            try:
                for line in result.stdout.split('\n'):
                    if 'Best Val DSC:' in line:
                        dsc_str = line.split('Best Val DSC:')[1].split()[0]
                        return True, float(dsc_str), duration
            except:
                pass
            
            return True, None, duration
        else:
            print(f"\n❌ FAILED")
            print(f"Error (last 500 chars): {result.stderr[-500:]}")
            return False, None, duration
    
    except subprocess.TimeoutExpired:
        print(f"\n❌ TIMEOUT (>2 hours)")
        if temp_file.exists():
            temp_file.unlink()
        return False, None, 7200
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False, None, 0


def main():
    parser = argparse.ArgumentParser(description='GPU-aware batch fine-tuning')
    parser.add_argument('--base-dir', type=str,
                       default='/home/pahm409/ablation_ds_main_only1',
                       help='Base directory with trained T1 models')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/finetuned_isles_all_models',
                       help='Output directory for fine-tuned models')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Folds to process')
    parser.add_argument('--runs', type=int, nargs='+', default=[0, 1, 2],
                       help='Runs to process')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--decoder-lr', type=float, default=0.0001,
                       help='Decoder learning rate')
    parser.add_argument('--encoder-lr-ratio', type=float, default=0.1,
                       help='Encoder LR ratio')
    parser.add_argument('--freeze-epochs', type=int, default=3,
                       help='Epochs to freeze encoder')
    parser.add_argument('--isles-only', action='store_true',
                       help='Train on ISLES only (no ATLAS/UOA)')
    parser.add_argument('--gpu', type=str, default='3',
                       help='GPU ID to use')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['Exp1_Random_Baseline', 'Exp2_Random_MKDC_DS',
                               'Exp3_SimCLR_Baseline', 'Exp4_SimCLR_MKDC_DS'],
                       default=None,
                       help='Specific experiments to run (default: all)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"⚠️  WARNING: CUDA not available!")
        else:
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs")
    except ImportError:
        print("⚠️  WARNING: PyTorch not imported in main process (OK for subprocess)")
    
    # Filter experiments
    if args.experiments:
        experiments_to_run = [e for e in EXPERIMENTS if e['name'] in args.experiments]
    else:
        experiments_to_run = EXPERIMENTS
    
    print(f"\n{'='*80}")
    print(f"BATCH FINE-TUNING ON ISLES DWI (GPU-AWARE)")
    print(f"{'='*80}")
    print(f"GPU: {args.gpu}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experiments: {len(experiments_to_run)}")
    print(f"Folds: {args.folds}")
    print(f"Runs: {args.runs}")
    print(f"Total models: {len(experiments_to_run) * len(args.folds) * len(args.runs)}")
    print(f"{'='*80}\n")
    
    # Find checkpoints
    print("Finding checkpoints...")
    checkpoints = []
    
    for exp in experiments_to_run:
        for fold in args.folds:
            for run in args.runs:
                checkpoint = find_best_checkpoint(
                    args.base_dir, exp['init'], exp['attention'],
                    exp['deep_supervision'], fold, run
                )
                
                if checkpoint:
                    checkpoints.append({
                        'experiment': exp['name'],
                        'fold': fold,
                        'run': run,
                        'checkpoint': checkpoint
                    })
                    print(f"  ✓ {exp['name']} - Fold {fold} - Run {run}")
                else:
                    print(f"  ✗ {exp['name']} - Fold {fold} - Run {run}")
    
    print(f"\nFound {len(checkpoints)} checkpoints\n")
    
    if len(checkpoints) == 0:
        print("❌ No checkpoints found!")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each checkpoint
    results = []
    failed = []
    
    overall_start = time.time()
    
    for idx, item in enumerate(checkpoints, 1):
        exp_name = item['experiment']
        fold = item['fold']
        run = item['run']
        checkpoint = item['checkpoint']
        
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(checkpoints)}] {exp_name} - Fold {fold} - Run {run}")
        print(f"{'#'*80}")
        
        # Create output directory
        exp_output = Path(args.output_dir) / exp_name / f'fold_{fold}' / f'run_{run}'
        
        success, dsc, duration = run_finetuning_with_gpu(
            checkpoint_path=checkpoint,
            fold=fold,
            output_dir=str(exp_output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            decoder_lr=args.decoder_lr,
            encoder_lr_ratio=args.encoder_lr_ratio,
            freeze_epochs=args.freeze_epochs,
            isles_only=args.isles_only,
            gpu_id=args.gpu
        )
        
        if success:
            results.append({
                'experiment': exp_name,
                'fold': fold,
                'run': run,
                'pretrained_checkpoint': checkpoint,
                'finetuned_dsc': dsc,
                'duration_minutes': duration / 60
            })
        else:
            failed.append({
                'experiment': exp_name,
                'fold': fold,
                'run': run,
                'checkpoint': checkpoint
            })
        
        # Progress
        elapsed = time.time() - overall_start
        avg_time = elapsed / idx
        remaining = (len(checkpoints) - idx) * avg_time
        
        print(f"\n📊 PROGRESS:")
        print(f"   Completed: {idx}/{len(checkpoints)}")
        print(f"   Success: {len(results)}, Failed: {len(failed)}")
        print(f"   Elapsed: {elapsed/3600:.1f}h")
        print(f"   ETA: {remaining/3600:.1f}h\n")
    
    # Final summary
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"BATCH FINE-TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful: {len(results)}/{len(checkpoints)}")
    print(f"Failed: {len(failed)}/{len(checkpoints)}")
    if results:
        print(f"Avg time per model: {(total_time/len(results))/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'gpu': args.gpu,
        'config': vars(args),
        'successful': results,
        'failed': failed,
        'total_models': len(checkpoints),
        'total_time_hours': total_time / 3600
    }
    
    summary_file = Path(args.output_dir) / 'batch_finetune_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_file}\n")
    
    # Results by experiment
    if results:
        print(f"\n{'='*80}")
        print("RESULTS BY EXPERIMENT")
        print(f"{'='*80}\n")
        
        for exp in experiments_to_run:
            exp_results = [r for r in results if r['experiment'] == exp['name']]
            if exp_results:
                print(f"{exp['name']}:")
                for r in sorted(exp_results, key=lambda x: (x['fold'], x['run'])):
                    dsc_str = f"{r['finetuned_dsc']:.4f}" if r['finetuned_dsc'] else "N/A"
                    time_str = f"{r['duration_minutes']:.1f}min"
                    print(f"  Fold {r['fold']}, Run {r['run']}: DSC={dsc_str} ({time_str})")
                
                # Average per fold
                by_fold = {}
                for r in exp_results:
                    if r['finetuned_dsc'] is not None:
                        if r['fold'] not in by_fold:
                            by_fold[r['fold']] = []
                        by_fold[r['fold']].append(r['finetuned_dsc'])
                
                if by_fold:
                    print(f"\n  Average per fold:")
                    for fold in sorted(by_fold.keys()):
                        avg = sum(by_fold[fold]) / len(by_fold[fold])
                        print(f"    Fold {fold}: {avg:.4f} (n={len(by_fold[fold])})")
                print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
check_training_completion.py

Check training completion status from log files and identify best models
"""

import re
from pathlib import Path
import json
from collections import defaultdict


def parse_log_file(log_file):
    """
    Parse training log to check completion and extract best DSC
    
    Returns:
        completed (bool): True if 100 epochs completed
        best_dsc (float): Best validation DSC achieved
        best_epoch (int): Epoch with best DSC
        last_epoch (int): Last completed epoch
    """
    if not log_file.exists():
        return False, None, None, None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Check for completion message
    completed = 'FOLD' in content and 'TRAINING COMPLETE' in content
    
    # Extract epoch information
    # Look for: "Epoch XX/100" or "Fold X - Epoch XX/100"
    epoch_pattern = r'Epoch (\d+)/(\d+)'
    epochs = re.findall(epoch_pattern, content)
    
    if epochs:
        last_epoch_num, total_epochs = epochs[-1]
        last_epoch = int(last_epoch_num)
        total = int(total_epochs)
        completed = (last_epoch >= total) or completed
    else:
        last_epoch = None
        completed = False
    
    # Extract best DSC information
    # Look for: "NEW BEST! DSC: X.XXXX" or "Best DSC: X.XXXX (Epoch XX)"
    best_dsc = None
    best_epoch = None
    
    # Pattern 1: "NEW BEST! DSC: X.XXXX"
    new_best_pattern = r'NEW BEST! DSC: ([\d\.]+)'
    new_bests = re.findall(new_best_pattern, content)
    if new_bests:
        best_dsc = float(new_bests[-1])  # Last "NEW BEST" is the best
    
    # Pattern 2: "Best DSC: X.XXXX (Epoch XX)"
    best_pattern = r'Best DSC: ([\d\.]+) \(Epoch (\d+)\)'
    best_matches = re.findall(best_pattern, content)
    if best_matches:
        best_dsc_str, best_epoch_str = best_matches[-1]
        best_dsc = float(best_dsc_str)
        best_epoch = int(best_epoch_str)
    
    return completed, best_dsc, best_epoch, last_epoch


def find_best_checkpoint(base_dir, init_type, attention, ds, fold, run):
    """Find the checkpoint path for a completed model"""
    ds_suffix = '_ds' if ds else ''
    
    fold_dir = Path(base_dir) / init_type / f'{attention}{ds_suffix}' / f'fold_{fold}' / f'run_{run}'
    
    if not fold_dir.exists():
        return None
    
    # Find latest experiment directory
    exp_dirs = sorted(fold_dir.glob('exp_*'))
    if not exp_dirs:
        return None
    
    latest_exp = exp_dirs[-1]
    checkpoint = latest_exp / 'checkpoints' / 'best_model.pth'
    
    if checkpoint.exists():
        return str(checkpoint)
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check training completion status')
    parser.add_argument('--base-dir', type=str,
                       default='/home/pahm409/ablation_ds_main_only1',
                       help='Base directory with experiments')
    parser.add_argument('--logs-dir', type=str,
                       default='/home/pahm409/ablation_ds_main_only1/logs',
                       help='Directory with log files')
    parser.add_argument('--save-completed', type=str,
                       default='completed_models.json',
                       help='Save list of completed models to JSON')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    logs_dir = Path(args.logs_dir)
    
    # Define experiments
    experiments = [
        {
            'name': 'Exp1_Random_Baseline',
            'log_name': 'Exp1_Random_Baseline',
            'init': 'Random_Init',
            'attention': 'none',
            'deep_supervision': False
        },
        {
            'name': 'Exp2_Random_MKDC_DS',
            'log_name': 'Exp2_Random_MKDC_DS',
            'init': 'Random_Init',
            'attention': 'mkdc',
            'deep_supervision': True
        },
        {
            'name': 'Exp3_SimCLR_Baseline',
            'log_name': 'Exp3_SimCLR_Baseline',
            'init': 'SimCLR_Pretrained',
            'attention': 'none',
            'deep_supervision': False
        },
        {
            'name': 'Exp4_SimCLR_MKDC_DS',
            'log_name': 'Exp4_SimCLR_MKDC_DS',
            'init': 'SimCLR_Pretrained',
            'attention': 'mkdc',
            'deep_supervision': True
        }
    ]
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETION STATUS")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"{'='*80}\n")
    
    all_results = []
    completed_models = []
    
    for exp in experiments:
        exp_log_dir = logs_dir / exp['log_name']
        
        if not exp_log_dir.exists():
            print(f"⚠️  {exp['name']}: Log directory not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"{exp['name']}")
        print(f"{'='*80}")
        print(f"{'Fold':<6} {'Run':<5} {'Status':<12} {'Epoch':<8} {'Best DSC':<10} {'Best@':<8} {'Checkpoint'}")
        print("-" * 80)
        
        exp_completed = []
        exp_incomplete = []
        
        for fold in range(5):
            for run in range(3):
                log_file = exp_log_dir / f'fold_{fold}_run_{run}.log'
                
                if not log_file.exists():
                    print(f"{fold:<6} {run:<5} {'MISSING':<12} {'-':<8} {'-':<10} {'-':<8} -")
                    continue
                
                completed, best_dsc, best_epoch, last_epoch = parse_log_file(log_file)
                
                # Check for checkpoint
                checkpoint = find_best_checkpoint(
                    base_dir, exp['init'], exp['attention'],
                    exp['deep_supervision'], fold, run
                )
                
                status = "✓ DONE" if completed else f"⏳ @Epoch{last_epoch}"
                dsc_str = f"{best_dsc:.4f}" if best_dsc else "-"
                epoch_str = str(best_epoch) if best_epoch else "-"
                last_str = str(last_epoch) if last_epoch else "-"
                ckpt_str = "✓" if checkpoint else "✗"
                
                print(f"{fold:<6} {run:<5} {status:<12} {last_str:<8} {dsc_str:<10} {epoch_str:<8} {ckpt_str}")
                
                result = {
                    'experiment': exp['name'],
                    'fold': fold,
                    'run': run,
                    'completed': completed,
                    'best_dsc': best_dsc,
                    'best_epoch': best_epoch,
                    'last_epoch': last_epoch,
                    'checkpoint': checkpoint,
                    'log_file': str(log_file)
                }
                
                all_results.append(result)
                
                if completed and checkpoint:
                    exp_completed.append(result)
                    completed_models.append(result)
                elif not completed:
                    exp_incomplete.append(result)
        
        # Summary for this experiment
        total = 5 * 3  # 5 folds × 3 runs
        n_completed = len(exp_completed)
        n_incomplete = len(exp_incomplete)
        n_missing = total - n_completed - n_incomplete
        
        print("-" * 80)
        print(f"Summary: {n_completed}/{total} completed, {n_incomplete} in progress, {n_missing} missing")
        
        if exp_completed:
            dscs = [r['best_dsc'] for r in exp_completed if r['best_dsc'] is not None]
            if dscs:
                mean_dsc = sum(dscs) / len(dscs)
                print(f"Completed models - Mean DSC: {mean_dsc:.4f} (n={len(dscs)})")
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_models = len(all_results)
    n_completed = len(completed_models)
    n_incomplete = sum(1 for r in all_results if not r['completed'] and r['last_epoch'] is not None)
    n_missing = total_models - n_completed - n_incomplete
    
    print(f"Total models: {total_models}")
    print(f"  ✓ Completed with checkpoint: {n_completed}")
    print(f"  ⏳ In progress: {n_incomplete}")
    print(f"  ✗ Missing/Not started: {n_missing}")
    
    if completed_models:
        # Group by experiment
        by_exp = defaultdict(list)
        for r in completed_models:
            by_exp[r['experiment']].append(r)
        
        print(f"\n{'='*80}")
        print(f"COMPLETED MODELS BY EXPERIMENT")
        print(f"{'='*80}")
        
        for exp_name in sorted(by_exp.keys()):
            models = by_exp[exp_name]
            dscs = [r['best_dsc'] for r in models if r['best_dsc'] is not None]
            
            print(f"\n{exp_name}: {len(models)} models")
            
            if dscs:
                mean_dsc = sum(dscs) / len(dscs)
                print(f"  Mean DSC: {mean_dsc:.4f}")
                print(f"  Range: [{min(dscs):.4f}, {max(dscs):.4f}]")
                
                # By fold
                by_fold = defaultdict(list)
                for r in models:
                    if r['best_dsc'] is not None:
                        by_fold[r['fold']].append(r['best_dsc'])
                
                print(f"  Per fold:")
                for fold in sorted(by_fold.keys()):
                    fold_dscs = by_fold[fold]
                    avg = sum(fold_dscs) / len(fold_dscs)
                    print(f"    Fold {fold}: {avg:.4f} (n={len(fold_dscs)})")
    
    # Save completed models list
    if completed_models and args.save_completed:
        output_file = Path(args.save_completed)
        
        # Organize by experiment/fold/run
        organized = {
            'summary': {
                'total_completed': len(completed_models),
                'by_experiment': {}
            },
            'models': completed_models
        }
        
        # Count by experiment
        for exp_name in set(r['experiment'] for r in completed_models):
            exp_models = [r for r in completed_models if r['experiment'] == exp_name]
            organized['summary']['by_experiment'][exp_name] = {
                'count': len(exp_models),
                'folds': {}
            }
            
            # Group by fold
            for fold in range(5):
                fold_models = [r for r in exp_models if r['fold'] == fold]
                if fold_models:
                    dscs = [r['best_dsc'] for r in fold_models if r['best_dsc']]
                    organized['summary']['by_experiment'][exp_name]['folds'][fold] = {
                        'count': len(fold_models),
                        'runs': [r['run'] for r in fold_models],
                        'mean_dsc': sum(dscs) / len(dscs) if dscs else None,
                        'checkpoints': [r['checkpoint'] for r in fold_models]
                    }
        
        with open(output_file, 'w') as f:
            json.dump(organized, f, indent=2)
        
        print(f"\n✓ Completed models saved to: {output_file}")
        print(f"  Use this file for batch fine-tuning!\n")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

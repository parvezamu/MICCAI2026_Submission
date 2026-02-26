#!/usr/bin/env python3
"""
batch_finetune_COMPLETED_models.py

Fine-tune ONLY completed models (based on completion check)
"""

import subprocess
import json
from pathlib import Path
import argparse
import time
from datetime import datetime
import os


def run_finetuning_with_gpu(checkpoint_path, fold, output_dir, epochs=50, batch_size=8, 
                            decoder_lr=0.0001, encoder_lr_ratio=0.1, freeze_epochs=3,
                            isles_only=False, gpu_id="3"):
    """Run fine-tuning with proper GPU setting"""
    
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

sys.path.insert(0, '.')
exec(open('finetune_on_isles_FIXED.py').read())
"""
    
    temp_file = Path(f'/tmp/finetune_gpu_{os.getpid()}_{fold}.py')
    with open(temp_file, 'w') as f:
        f.write(temp_script)
    
    print(f"\n{'='*80}")
    print(f"Running fine-tuning on GPU {gpu_id}...")
    print(f"  Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Fold: {fold}")
    print(f"{'='*80}\n")
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            ['python3', str(temp_file)],
            capture_output=True,
            text=True,
            timeout=7200
        )
        
        duration = time.time() - start_time
        
        temp_file.unlink()
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS (took {duration/60:.1f} minutes)")
            
            # Parse DSC from output
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
            print(f"Error: {result.stderr[-500:]}")
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
    parser = argparse.ArgumentParser(description='Fine-tune completed models only')
    parser.add_argument('--completed-json', type=str,
                       default='completed_models.json',
                       help='JSON file with completed models (from check_training_completion.py)')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/finetuned_isles_completed',
                       help='Output directory for fine-tuned models')
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
                       help='Train on ISLES only')
    parser.add_argument('--gpu', type=str, default='3',
                       help='GPU ID')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['Exp1_Random_Baseline', 'Exp2_Random_MKDC_DS',
                               'Exp3_SimCLR_Baseline', 'Exp4_SimCLR_MKDC_DS'],
                       default=None,
                       help='Specific experiments to run')
    parser.add_argument('--folds', type=int, nargs='+',
                       default=None,
                       help='Specific folds to run (default: all)')
    parser.add_argument('--max-models', type=int, default=None,
                       help='Maximum number of models to fine-tune (for testing)')
    
    args = parser.parse_args()
    
    # Load completed models
    completed_file = Path(args.completed_json)
    if not completed_file.exists():
        print(f"❌ Completed models file not found: {completed_file}")
        print(f"   Run: python check_training_completion.py --save-completed {completed_file}")
        return
    
    with open(completed_file, 'r') as f:
        data = json.load(f)
    
    completed_models = data['models']
    
    # Filter models
    models_to_finetune = []
    
    for model in completed_models:
        # Filter by experiment
        if args.experiments and model['experiment'] not in args.experiments:
            continue
        
        # Filter by fold
        if args.folds and model['fold'] not in args.folds:
            continue
        
        # Only process models with checkpoints
        if model['checkpoint']:
            models_to_finetune.append(model)
    
    # Apply max models limit
    if args.max_models:
        models_to_finetune = models_to_finetune[:args.max_models]
    
    print(f"\n{'='*80}")
    print(f"BATCH FINE-TUNING (COMPLETED MODELS ONLY)")
    print(f"{'='*80}")
    print(f"GPU: {args.gpu}")
    print(f"Completed models available: {len(completed_models)}")
    print(f"Models to fine-tune: {len(models_to_finetune)}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    if len(models_to_finetune) == 0:
        print("❌ No models to fine-tune after filtering!")
        return
    
    # Show what will be processed
    by_exp = {}
    for model in models_to_finetune:
        exp = model['experiment']
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(f"Fold {model['fold']}, Run {model['run']}")
    
    print("Models to process:")
    for exp, models in sorted(by_exp.items()):
        print(f"  {exp}: {len(models)} models")
        for m in models[:3]:
            print(f"    - {m}")
        if len(models) > 3:
            print(f"    ... and {len(models)-3} more")
    print()
    
    # Confirm
    if len(models_to_finetune) > 10:
        print(f"⚠️  This will fine-tune {len(models_to_finetune)} models (~{len(models_to_finetune)*1}h)")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each model
    results = []
    failed = []
    
    overall_start = time.time()
    
    for idx, model in enumerate(models_to_finetune, 1):
        exp_name = model['experiment']
        fold = model['fold']
        run = model['run']
        checkpoint = model['checkpoint']
        pretrain_dsc = model['best_dsc']
        
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(models_to_finetune)}] {exp_name} - Fold {fold} - Run {run}")
        print(f"# Pre-training DSC: {pretrain_dsc:.4f}" if pretrain_dsc else "")
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
                'pretrained_dsc': pretrain_dsc,
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
        if idx > 0:
            avg_time = elapsed / idx
            remaining = (len(models_to_finetune) - idx) * avg_time
            
            print(f"\n📊 PROGRESS:")
            print(f"   Completed: {idx}/{len(models_to_finetune)}")
            print(f"   Success: {len(results)}, Failed: {len(failed)}")
            print(f"   Elapsed: {elapsed/3600:.1f}h")
            print(f"   ETA: {remaining/3600:.1f}h")
            print(f"   Avg time/model: {avg_time/60:.1f}min\n")
    
    # Final summary
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"BATCH FINE-TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful: {len(results)}/{len(models_to_finetune)}")
    print(f"Failed: {len(failed)}/{len(models_to_finetune)}")
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
        'total_models': len(models_to_finetune),
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
        
        for exp_name in sorted(set(r['experiment'] for r in results)):
            exp_results = [r for r in results if r['experiment'] == exp_name]
            
            print(f"{exp_name}: {len(exp_results)} models")
            print(f"{'Fold':<6} {'Run':<5} {'Pre-DSC':<10} {'Fine-DSC':<10} {'Gain':<8}")
            print("-" * 45)
            
            for r in sorted(exp_results, key=lambda x: (x['fold'], x['run'])):
                pre_dsc = r['pretrained_dsc'] if r['pretrained_dsc'] else 0.0
                fine_dsc = r['finetuned_dsc'] if r['finetuned_dsc'] else 0.0
                gain = fine_dsc - pre_dsc if (r['pretrained_dsc'] and r['finetuned_dsc']) else 0.0
                
                print(f"{r['fold']:<6} {r['run']:<5} {pre_dsc:<10.4f} {fine_dsc:<10.4f} {gain:+.4f}")
            
            # Averages
            pre_dscs = [r['pretrained_dsc'] for r in exp_results if r['pretrained_dsc']]
            fine_dscs = [r['finetuned_dsc'] for r in exp_results if r['finetuned_dsc']]
            
            if pre_dscs and fine_dscs:
                avg_pre = sum(pre_dscs) / len(pre_dscs)
                avg_fine = sum(fine_dscs) / len(fine_dscs)
                avg_gain = avg_fine - avg_pre
                
                print("-" * 45)
                print(f"{'Mean':<6} {'':<5} {avg_pre:<10.4f} {avg_fine:<10.4f} {avg_gain:+.4f}")
            print()


if __name__ == '__main__':
    main()

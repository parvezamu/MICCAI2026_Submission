#!/usr/bin/env python3
"""
finetune_interactive.py

Interactive fine-tuning - you decide whether to continue after each model
Perfect for medical domain where quality matters more than speed
"""

import subprocess
import json
from pathlib import Path
import argparse
import time
from datetime import datetime
import os


def run_single_finetuning(checkpoint_path, fold, output_dir, epochs=50, batch_size=8,
                         decoder_lr=0.0001, encoder_lr_ratio=0.1, freeze_epochs=3,
                         isles_only=False, gpu_id="3", timeout_hours=3):
    """Run fine-tuning on a single model"""
    
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
    
    temp_file = Path(f'/tmp/finetune_interactive_{os.getpid()}.py')
    with open(temp_file, 'w') as f:
        f.write(temp_script)
    
    print(f"\n{'='*80}")
    print(f"RUNNING FINE-TUNING")
    print(f"{'='*80}")
    print(f"GPU: {gpu_id}")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Fold: {fold}")
    print(f"Epochs: {epochs}")
    print(f"Timeout: {timeout_hours} hours")
    print(f"{'='*80}\n")
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            ['python3', str(temp_file)],
            capture_output=True,
            text=True,
            timeout=timeout_hours * 3600
        )
        
        duration = time.time() - start_time
        
        temp_file.unlink()
        
        if result.returncode == 0:
            print(f"\n{'='*80}")
            print(f"✅ SUCCESS")
            print(f"{'='*80}")
            print(f"Time taken: {duration/60:.1f} minutes")
            
            # Try to extract DSC
            dsc = None
            try:
                for line in result.stdout.split('\n'):
                    if 'Best Val DSC:' in line:
                        dsc_str = line.split('Best Val DSC:')[1].split()[0]
                        dsc = float(dsc_str)
                        print(f"Best validation DSC: {dsc:.4f}")
                        break
            except:
                pass
            
            print(f"{'='*80}\n")
            
            return True, dsc, duration
        else:
            print(f"\n{'='*80}")
            print(f"❌ FAILED")
            print(f"{'='*80}")
            print(f"Error (last 1000 chars):")
            print(result.stderr[-1000:])
            print(f"{'='*80}\n")
            return False, None, duration
    
    except subprocess.TimeoutExpired:
        print(f"\n{'='*80}")
        print(f"❌ TIMEOUT")
        print(f"{'='*80}")
        print(f"Exceeded {timeout_hours} hours")
        print(f"{'='*80}\n")
        if temp_file.exists():
            temp_file.unlink()
        return False, None, timeout_hours * 3600
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR: {e}")
        print(f"{'='*80}\n")
        if temp_file.exists():
            temp_file.unlink()
        return False, None, 0


def show_model_info(model):
    """Display model information"""
    print(f"\n{'='*80}")
    print(f"MODEL INFORMATION")
    print(f"{'='*80}")
    print(f"Experiment: {model['experiment']}")
    print(f"Fold: {model['fold']}")
    print(f"Run: {model['run']}")
    print(f"Pre-training DSC: {model['best_dsc']:.4f}" if model['best_dsc'] else "Pre-training DSC: N/A")
    print(f"Pre-training Epoch: {model['best_epoch']}" if model['best_epoch'] else "")
    print(f"Checkpoint: {model['checkpoint']}")
    print(f"{'='*80}\n")


def ask_user_continue(model_num, total_models, results_so_far):
    """Ask user if they want to continue"""
    print(f"\n{'='*80}")
    print(f"PROGRESS: {model_num}/{total_models} models completed")
    print(f"{'='*80}")
    
    if results_so_far:
        successful = [r for r in results_so_far if r['success']]
        failed = [r for r in results_so_far if not r['success']]
        
        print(f"Results so far:")
        print(f"  ✅ Successful: {len(successful)}")
        print(f"  ❌ Failed: {len(failed)}")
        
        if successful:
            dscs = [r['finetuned_dsc'] for r in successful if r['finetuned_dsc']]
            if dscs:
                avg_dsc = sum(dscs) / len(dscs)
                print(f"  📊 Average fine-tuned DSC: {avg_dsc:.4f}")
            
            avg_time = sum(r['duration_minutes'] for r in successful) / len(successful)
            print(f"  ⏱️  Average time: {avg_time:.1f} minutes/model")
    
    print(f"{'='*80}\n")
    
    response = input("Continue to next model? (yes/no/quit): ").strip().lower()
    
    if response in ['quit', 'q', 'exit']:
        return 'quit'
    elif response in ['yes', 'y']:
        return 'continue'
    elif response in ['no', 'n']:
        return 'skip'
    else:
        print("Invalid input. Type 'yes', 'no', or 'quit'")
        return ask_user_continue(model_num, total_models, results_so_far)


def main():
    parser = argparse.ArgumentParser(description='Interactive one-by-one fine-tuning')
    parser.add_argument('--completed-json', type=str,
                       default='completed_models.json',
                       help='JSON with completed models')
    parser.add_argument('--output-dir', type=str,
                       default='/home/pahm409/finetuned_isles_interactive',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Fine-tuning epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--decoder-lr', type=float, default=0.0001,
                       help='Decoder learning rate')
    parser.add_argument('--encoder-lr-ratio', type=float, default=0.1,
                       help='Encoder LR ratio')
    parser.add_argument('--freeze-epochs', type=int, default=3,
                       help='Epochs to freeze encoder')
    parser.add_argument('--timeout-hours', type=float, default=3,
                       help='Timeout in hours per model')
    parser.add_argument('--isles-only', action='store_true',
                       help='Train on ISLES only')
    parser.add_argument('--gpu', type=str, default='3',
                       help='GPU ID')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['Exp1_Random_Baseline', 'Exp2_Random_MKDC_DS',
                               'Exp3_SimCLR_Baseline', 'Exp4_SimCLR_MKDC_DS'],
                       default=None,
                       help='Specific experiments')
    parser.add_argument('--folds', type=int, nargs='+',
                       default=None,
                       help='Specific folds')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from model N (0-indexed)')
    parser.add_argument('--auto-yes', action='store_true',
                       help='Automatically continue without asking (for batch mode)')
    
    args = parser.parse_args()
    
    # Load completed models
    completed_file = Path(args.completed_json)
    if not completed_file.exists():
        print(f"❌ File not found: {completed_file}")
        print(f"   Run: python check_training_completion.py")
        return
    
    with open(completed_file, 'r') as f:
        data = json.load(f)
    
    completed_models = data['models']
    
    # Filter
    models_to_finetune = []
    for model in completed_models:
        if args.experiments and model['experiment'] not in args.experiments:
            continue
        if args.folds and model['fold'] not in args.folds:
            continue
        if model['checkpoint']:
            models_to_finetune.append(model)
    
    # Apply start-from
    if args.start_from > 0:
        models_to_finetune = models_to_finetune[args.start_from:]
        print(f"⏭️  Starting from model {args.start_from}")
    
    print(f"\n{'='*80}")
    print(f"INTERACTIVE FINE-TUNING")
    print(f"{'='*80}")
    print(f"Mode: {'AUTO (no prompts)' if args.auto_yes else 'INTERACTIVE (ask after each)'}")
    print(f"GPU: {args.gpu}")
    print(f"Models to fine-tune: {len(models_to_finetune)}")
    print(f"Epochs: {args.epochs}")
    print(f"Timeout: {args.timeout_hours} hours per model")
    print(f"Expected total time: ~{len(models_to_finetune) * args.timeout_hours * 0.6:.1f} hours")
    print(f"{'='*80}\n")
    
    if len(models_to_finetune) == 0:
        print("❌ No models to fine-tune!")
        return
    
    # Show all models
    print("Models in queue:")
    for i, model in enumerate(models_to_finetune):
        dsc_str = f"DSC={model['best_dsc']:.4f}" if model['best_dsc'] else "DSC=N/A"
        print(f"  [{i+1}] {model['experiment']} - Fold {model['fold']}, Run {model['run']} - {dsc_str}")
    print()
    
    if not args.auto_yes:
        response = input("Start fine-tuning? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled")
            return
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each model
    results = []
    overall_start = time.time()
    
    for idx, model in enumerate(models_to_finetune):
        exp_name = model['experiment']
        fold = model['fold']
        run = model['run']
        checkpoint = model['checkpoint']
        pretrain_dsc = model['best_dsc']
        
        print(f"\n{'#'*80}")
        print(f"# MODEL {idx+1}/{len(models_to_finetune)}")
        print(f"{'#'*80}")
        
        show_model_info(model)
        
        exp_output = Path(args.output_dir) / exp_name / f'fold_{fold}' / f'run_{run}'
        
        success, dsc, duration = run_single_finetuning(
            checkpoint_path=checkpoint,
            fold=fold,
            output_dir=str(exp_output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            decoder_lr=args.decoder_lr,
            encoder_lr_ratio=args.encoder_lr_ratio,
            freeze_epochs=args.freeze_epochs,
            isles_only=args.isles_only,
            gpu_id=args.gpu,
            timeout_hours=args.timeout_hours
        )
        
        # Save result
        result = {
            'experiment': exp_name,
            'fold': fold,
            'run': run,
            'pretrained_checkpoint': checkpoint,
            'pretrained_dsc': pretrain_dsc,
            'finetuned_dsc': dsc,
            'duration_minutes': duration / 60,
            'success': success
        }
        results.append(result)
        
        # Show result
        if success and dsc:
            gain = dsc - pretrain_dsc if pretrain_dsc else 0.0
            print(f"📊 RESULT:")
            print(f"   Pre-training DSC:  {pretrain_dsc:.4f}")
            print(f"   Fine-tuned DSC:    {dsc:.4f}")
            print(f"   Gain:              {gain:+.4f}")
            print(f"   Time:              {duration/60:.1f} minutes")
        
        # Ask user whether to continue (unless auto-yes or last model)
        if idx < len(models_to_finetune) - 1:  # Not last model
            if args.auto_yes:
                print("\n⏭️  Auto-continuing to next model...\n")
            else:
                decision = ask_user_continue(idx + 1, len(models_to_finetune), results)
                
                if decision == 'quit':
                    print("\n⛔ User requested quit. Stopping.")
                    break
                elif decision == 'skip':
                    print("\n⏭️  Skipping remaining models.")
                    break
                # else: continue
    
    # Final summary
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING SESSION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Models processed: {len(results)}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['duration_minutes'] for r in successful) / len(successful)
        print(f"  ⏱️  Avg time: {avg_time:.1f} min/model")
    
    print(f"{'='*80}\n")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'results': results,
        'total_time_hours': total_time / 3600,
        'successful_count': len(successful),
        'failed_count': len(failed)
    }
    
    summary_file = Path(args.output_dir) / 'interactive_finetune_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_file}\n")
    
    # Results table
    if results:
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}\n")
        
        print(f"{'Exp':<25} {'Fold':<5} {'Run':<4} {'Pre-DSC':<8} {'Fine-DSC':<9} {'Gain':<8} {'Time':<8} {'Status'}")
        print("-" * 90)
        
        for r in results:
            exp_short = r['experiment'][:24]
            pre = r['pretrained_dsc'] if r['pretrained_dsc'] else 0.0
            fine = r['finetuned_dsc'] if r['finetuned_dsc'] else 0.0
            gain = fine - pre if (r['pretrained_dsc'] and r['finetuned_dsc']) else 0.0
            status = "✓" if r['success'] else "✗"
            
            print(f"{exp_short:<25} {r['fold']:<5} {r['run']:<4} {pre:<8.4f} {fine:<9.4f} {gain:+.4f}   {r['duration_minutes']:<7.1f}  {status}")
        
        print()


if __name__ == '__main__':
    main()

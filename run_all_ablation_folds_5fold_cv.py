#!/usr/bin/env python3
"""
Master script to run all 5 folds for Ablation Study 5-Fold Cross-Validation
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

OUTPUT_DIR = "/home/pahm409/ablation_atlas_uoa_5fold_cv"
GPU_ID = "1"
FOLDS = [0, 1, 2, 3, 4]

def run_fold(fold, gpu_id):
    """Run all experiments for a single fold"""
    print(f"\n{'='*80}")
    print(f"STARTING FOLD {fold}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'run_ablation_fold_5fold_clean.py',
        '--fold', str(fold),
        '--gpu', str(gpu_id)
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=False
        )
        
        duration = time.time() - start_time
        success = (result.returncode == 0)
        
        if success:
            print(f"\n{'='*80}")
            print(f"✅ FOLD {fold} COMPLETE")
            print(f"   Duration: {timedelta(seconds=int(duration))}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"❌ FOLD {fold} FAILED")
            print(f"   Return code: {result.returncode}")
            print(f"{'='*80}\n")
        
        return success, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"❌ FOLD {fold} ERROR: {str(e)}")
        print(f"{'='*80}\n")
        return False, duration


def main():
    parser = argparse.ArgumentParser(description='Run complete 5-fold cross-validation for Ablation Study')
    parser.add_argument('--gpu', type=str, default=GPU_ID,
                        help='GPU ID to use')
    parser.add_argument('--start-fold', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Starting fold (default: 0)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ABLATION 5-FOLD CROSS-VALIDATION - MASTER RUNNER")
    print("="*80)
    print(f"Configuration:")
    print(f"  Total folds: {len(FOLDS)}")
    print(f"  Starting from fold: {args.start_fold}")
    print(f"  Experiments per fold: 4")
    print(f"  GPU: {args.gpu}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Estimated total time: ~30-40 hours")
    print("="*80 + "\n")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Track progress
    results = {}
    overall_start = time.time()
    
    folds_to_run = [f for f in FOLDS if f >= args.start_fold]
    
    for fold in folds_to_run:
        success, duration = run_fold(fold, args.gpu)
        results[fold] = {
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save progress
        progress_file = Path(OUTPUT_DIR) / 'training_progress.json'
        with open(progress_file, 'w') as f:
            json.dump({
                'completed_folds': results,
                'total_time_so_far': time.time() - overall_start,
                'last_update': datetime.now().isoformat()
            }, f, indent=4)
    
    # Final summary
    total_time = time.time() - overall_start
    successful_folds = sum(1 for r in results.values() if r['success'])
    failed_folds = len(results) - successful_folds
    
    print("\n" + "="*80)
    print("ABLATION 5-FOLD CROSS-VALIDATION COMPLETE!")
    print("="*80)
    print(f"Summary:")
    print(f"  Total folds: {len(results)}")
    print(f"  ✅ Successful: {successful_folds}")
    print(f"  ❌ Failed: {failed_folds}")
    print(f"  ⏱️  Total time: {timedelta(seconds=int(total_time))}")
    print(f"\nFold Details:")
    for fold, result in sorted(results.items()):
        status = "✅" if result['success'] else "❌"
        duration = timedelta(seconds=int(result['duration']))
        print(f"  {status} Fold {fold}: {duration}")
    print("="*80 + "\n")
    
    if failed_folds > 0:
        print("⚠️  Some folds failed. Check individual fold logs for details.")
        return 1
    else:
        print("🎉 All folds completed successfully!")
        print(f"\nResults saved to: {OUTPUT_DIR}")
        return 0


if __name__ == "__main__":
    exit(main())

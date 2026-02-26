"""
train_experiment_4.py - SimCLR + MKDC + DS (main-only)
"""
import subprocess
import os

SIMCLR_CHECKPOINT = "/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"  # ← CHANGE THIS!
OUTPUT_DIR = "/home/pahm409/ablation_ds_main_only"
GPU = "1"

print(f"\n{'='*80}")
print("EXPERIMENT 4: SimCLR + MKDC + DS (main-only)")
print(f"{'='*80}\n")

for fold in range(5):
    for run in range(3):
        cmd = [
            'python', 'corr5.py',
            '--fold', str(fold),
            '--run-id', str(run),
            '--epochs', '100',
            '--batch-size', '8',
            '--output-dir', OUTPUT_DIR,
            '--pretrained-checkpoint', SIMCLR_CHECKPOINT,
            '--attention', 'mkdc',
            '--deep-supervision',
            '--ds-weights', '1.0', '0.0', '0.0', '0.0',
            '--patches-per-volume', '10',
            '--lesion-focus-ratio', '0.7',
            '--validate-recon-every', '5',
            '--save-nifti-every', '10'
        ]
        
        print(f"\n{'='*80}")
        print(f"Experiment 4: Fold {fold}, Run {run}")
        print(f"{'='*80}\n")
        
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPU}
        subprocess.run(cmd, env=env, check=True)

print(f"\n{'='*80}")
print("EXPERIMENT 4 COMPLETE!")
print(f"{'='*80}\n")

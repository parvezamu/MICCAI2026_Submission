"""
train_experiment_2.py - Train ONLY Experiment 2
"""
import subprocess
simclr = "/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"

#simclr = "/path/to/simclr.pth"
output = "/home/pahm409/ablation_ds_main_only"
gpu = "3"

for fold in range(5):
    for run in range(3):
        cmd = [
            'python', 'corr1.py',
            '--fold', str(fold),
            '--run-id', str(run),
            '--epochs', '100',
            '--batch-size', '8',
            '--output-dir', output,
            '--attention', 'mkdc',
            '--deep-supervision',
            '--ds-weights', '1.0', '0.0', '0.0', '0.0',
            '--patches-per-volume', '10',
            '--lesion-focus-ratio', '0.7'
        ]
        
        print(f"\n{'='*80}")
        print(f"Experiment 2: Fold {fold}, Run {run}")
        print(f"{'='*80}\n")
        
        env = {'CUDA_VISIBLE_DEVICES': gpu}
        subprocess.run(cmd, env={**subprocess.os.environ, **env}, check=True)

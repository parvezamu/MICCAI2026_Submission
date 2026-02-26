"""
run_ablation_study.py

COMPLETE ablation study using train_segmentation_corrected.py

Author: Parvez
Date: January 2026
"""

import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import subprocess
import sys

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR = "/home/pahm409/ablation_study_fold0"
FOLD = 0
BATCH_SIZE =  8
BASE_LR = 0.0001
BASE_BATCH = 8
LEARNING_RATE = BASE_LR * (BATCH_SIZE / BASE_BATCH)  # 0.0004

print("="*80)
print("ABLATION STUDY - Does SimCLR help?")
print("="*80)
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE:.6f}")
print("="*80)

experiments = [
    {
        "name": "1. Random Init + Baseline",
        "args": [
            "--fold", str(FOLD),
            "--attention", "none",
            "--epochs", "100",
            "--batch-size", str(BATCH_SIZE),
            "--lr", str(LEARNING_RATE),
            "--optimizer", "adamw",
            "--output-dir", OUTPUT_DIR
        ]
    }
   
]

for exp in experiments:
    print(f"\n{'='*80}")
    print(f"{exp['name']}")
    print(f"{'='*80}\n")
    
    cmd = ["python", "train_segmentation_corrected.py"] + exp["args"]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {exp['name']}")
        sys.exit(1)
    
    print(f"\n✓ DONE: {exp['name']}")

print("\n" + "="*80)
print("ABLATION COMPLETE")
print("="*80)

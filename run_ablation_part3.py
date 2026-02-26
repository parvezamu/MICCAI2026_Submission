"""
run_ablation_part3.py - GPU 2
SimCLR + MKDC + DS (your current best)
"""
import subprocess

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR = "/home/pahm409/ablation_study_fold0"
FOLD = 0
BATCH_SIZE =  8
LEARNING_RATE = 0.0004
GPU_ID = "0"

print("="*80)
print("ABLATION PART 3 - SimCLR + MKDC + DS")
print(f"GPU: {GPU_ID}")
print("="*80)

cmd = [
    "python", "corr.py",
    "--fold", str(FOLD),
    "--attention", "mkdc",
    "--deep-supervision",
    "--pretrained-checkpoint", CHECKPOINT,
    "--epochs", "100",
    "--batch-size", str(BATCH_SIZE),
    "--lr", str(LEARNING_RATE),
    "--optimizer", "adamw",
    "--output-dir", OUTPUT_DIR
]

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

print(f"Command: {' '.join(cmd)}\n")
subprocess.run(cmd)
print("\n✓ PART 3 COMPLETE")

"""
run_ablation_part2.py - GPU 1
Random Init + MKDC + DS
"""
import subprocess

OUTPUT_DIR = "/home/pahm409/ablation_study_fold0"
FOLD = 0
BATCH_SIZE =  8
LEARNING_RATE = 0.0004
GPU_ID = "1"

print("="*80)
print("ABLATION PART 2 - Random Init + MKDC + DS")
print(f"GPU: {GPU_ID}")
print("="*80)

cmd = [
    "python", "corr1.py",
    "--fold", str(FOLD),
    "--attention", "mkdc",
    "--deep-supervision",
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
print("\n✓ PART 2 COMPLETE")

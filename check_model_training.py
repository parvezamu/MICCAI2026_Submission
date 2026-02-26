"""
check_model_training.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

checkpoint_path = '/home/pahm409/ablation_ds_main_only/SimCLR_Pretrained/mkdc_ds/fold_0/run_0/exp_20260129_210732/checkpoints/best_model.pth'

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*70)
print("CHECKPOINT METADATA")
print("="*70)

for key in checkpoint.keys():
    if key not in ['model_state_dict', 'optimizer_state_dict']:
        print(f"{key}: {checkpoint[key]}")

print("\n" + "="*70)
print("LOOKING FOR TRAINING INFO")
print("="*70)

# Check if there's info about training dataset
possible_keys = ['dataset', 'datasets', 'train_dataset', 'config', 'args']
for key in possible_keys:
    if key in checkpoint:
        print(f"\n{key}:")
        print(checkpoint[key])


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
import torch

ckpt_path = "/home/pahm409/dwi_scratch_5fold/baseline/fold_0/run_1/exp_20260202_011625/checkpoints/best_model.pth"

checkpoint = torch.load(ckpt_path, map_location="cpu")

print("Training config:")
print(f"  Patch size: {checkpoint.get('patch_size', 'NOT SAVED')}")

'''

import torch
from models.resnet3d import resnet3d_18
from finetune_on_isles_DEBUG import SegmentationModel

# Load Random Init checkpoint
checkpoint = torch.load('/home/pahm409/finetuned_on_isles_5fold/fold_0/finetune_20260201_054345/checkpoints/best_finetuned_model.pth', map_location='cpu')

encoder = resnet3d_18(in_channels=1)
model = SegmentationModel(encoder, num_classes=2, attention_type='none', deep_supervision=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_input = torch.randn(1, 1, 96, 96, 96)
with torch.no_grad():
    output = model(test_input)
    print(f"Random Init output shape: {output.shape}")
    
    # Also check encoder
    enc_features = model.encoder(test_input)
    for i, feat in enumerate(enc_features):
        print(f"  x{i+1}: {feat.shape}")

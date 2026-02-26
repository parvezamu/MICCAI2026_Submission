import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from models.resnet3d import resnet3d_18

CKPT = "/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth"

# -------- Random encoder --------
enc_rand = resnet3d_18(in_channels=1)
rand_norm = enc_rand.conv1.weight.data.float().norm().item()

# -------- Pretrained encoder --------
enc_pre = resnet3d_18(in_channels=1)

ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt["model_state_dict"]

# remove DataParallel prefix if present
state = {k.replace("module.", ""): v for k, v in state.items()}

# extract encoder-only weights
enc_state = {k.replace("encoder.", ""): v
             for k, v in state.items()
             if k.startswith("encoder.")}

missing, unexpected = enc_pre.load_state_dict(enc_state, strict=False)

pre_norm = enc_pre.conv1.weight.data.float().norm().item()

print("Random conv1 norm     :", rand_norm)
print("Pretrained conv1 norm :", pre_norm)
print("Absolute difference   :", abs(rand_norm - pre_norm))
print("Missing keys          :", len(missing))
print("Unexpected keys       :", len(unexpected))


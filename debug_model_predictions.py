"""
debug_model_predictions.py

Check raw model outputs for working vs failing cases

Author: Parvez
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel


class ResNet3DEncoder(nn.Module):
    def __init__(self, base_encoder):
        super(ResNet3DEncoder, self).__init__()
        self.conv1 = base_encoder.conv1
        self.bn1 = base_encoder.bn1
        self.maxpool = base_encoder.maxpool
        self.layer1 = base_encoder.layer1
        self.layer2 = base_encoder.layer2
        self.layer3 = base_encoder.layer3
        self.layer4 = base_encoder.layer4
    
    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1, x2, x3, x4, x5]


class UNetDecoder3D(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetDecoder3D, self).__init__()
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x_up, x_skip):
        import torch.nn.functional as F
        d_up, h_up, w_up = x_up.shape[2:]
        d_skip, h_skip, w_skip = x_skip.shape[2:]
        if d_up != d_skip or h_up != h_skip or w_up != w_skip:
            diff_d, diff_h, diff_w = d_skip - d_up, h_skip - h_up, w_skip - w_up
            if diff_d > 0 or diff_h > 0 or diff_w > 0:
                padding = [max(0, diff_w // 2), max(0, diff_w - diff_w // 2),
                          max(0, diff_h // 2), max(0, diff_h - diff_h // 2),
                          max(0, diff_d // 2), max(0, diff_d - diff_d // 2)]
                x_up = F.pad(x_up, padding)
            elif diff_d < 0 or diff_h < 0 or diff_w < 0:
                d_start, h_start, w_start = max(0, -diff_d // 2), max(0, -diff_h // 2), max(0, -diff_w // 2)
                x_up = x_up[:, :, d_start:d_start + d_skip, h_start:h_start + h_skip, w_start:w_start + w_skip]
        return x_up
    
    def forward(self, encoder_features):
        x1, x2, x3, x4, x5 = encoder_features
        x = self.up4(x5)
        x = self._match_size(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        x = self.up3(x)
        x = self._match_size(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = self._match_size(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = self._match_size(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        return self.final_conv(x)


class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        import torch.nn.functional as F
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='trilinear', align_corners=False)
        return seg_logits


def debug_predictions():
    device = torch.device('cuda:0')
    
    # Load model
    checkpoint_path = '/home/pahm409/patch_reconstruction_experiments_5fold/fold_0/patch_recon_20260110_170514/checkpoints/best_model.pth'
    simclr_path = '/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = resnet3d_18(in_channels=1)
    simclr_checkpoint = torch.load(simclr_path, map_location=device)
    simclr_model = SimCLRModel(encoder, projection_dim=128, hidden_dim=512)
    simclr_model.load_state_dict(simclr_checkpoint['model_state_dict'])
    
    model = SegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    neuralcup_dir = Path('/home/pahm409/preprocessed_NEURALCUP_1mm/NEURALCUP')
    
    working = ['BBS315', 'BBS318']
    failing = ['BBS302', 'BBS328']
    
    print("="*70)
    print("RAW MODEL OUTPUTS")
    print("="*70)
    
    for case_list, label in [(working, "WORKING"), (failing, "FAILING")]:
        print(f"\n{label} CASES:")
        print("-"*70)
        
        for case_id in case_list:
            npz_file = neuralcup_dir / f'{case_id}.npz'
            data = np.load(npz_file)
            volume = data['image']
            mask = data['lesion_mask']
            
            # Extract center patch
            center = np.array(volume.shape) // 2
            half_size = np.array([48, 48, 48])
            lower = center - half_size
            upper = center + half_size
            
            patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
            patch_mask = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
            
            # Predict
            with torch.no_grad():
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                logits = model(patch_tensor)
                print(f"  Logits shape: {logits.shape}")
                probs = torch.softmax(logits, dim=1)
                print(f"  Probs shape: {probs.shape}")
                print(f"  Prob[bg] with dim=1: mean={probs[0,0].mean():.4f}, max={probs[0,0].max():.4f}")
                print(f"  Prob[lesion] with dim=1: mean={probs[0,1].mean():.4f}, max={probs[0,1].max():.4f}")
                
                prob_bg = probs[0, 0].cpu().numpy()
                prob_lesion = probs[0, 1].cpu().numpy()
                
                print(f"\n{case_id}:")
                print(f"  Patch has lesion: {patch_mask.sum()} voxels")
                print(f"  Prob[background]: mean={prob_bg.mean():.4f}, max={prob_bg.max():.4f}")
                print(f"  Prob[lesion]:     mean={prob_lesion.mean():.4f}, max={prob_lesion.max():.4f}")
                print(f"  Predicted voxels (>0.5): {(prob_lesion > 0.5).sum()}")
                print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")


if __name__ == '__main__':
    debug_predictions()

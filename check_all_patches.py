"""
check_all_patches.py

Check predictions for ALL 10 random patches

Author: Parvez
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import torch.nn.functional as F

sys.path.append('.')
from models.resnet3d import resnet3d_18, SimCLRModel

# [Copy the ResNet3DEncoder, UNetDecoder3D, and SegmentationModel classes from before]

class GDiceLoss(nn.Module):
    """Generalized Dice Loss"""
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(GDiceLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
    
    def forward(self, net_output, gt):
        shp_x = net_output.shape
        shp_y = gt.shape
        
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))
            
            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x, device=net_output.device, dtype=net_output.dtype)
                y_onehot.scatter_(1, gt, 1)
        
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        
        net_output = net_output.double()
        y_onehot = y_onehot.double()
        
        w = 1 / (einsum("bcdhw->bc", y_onehot).type(torch.float64) + 1e-10) ** 2
        intersection = w * einsum("bcdhw,bcdhw->bc", net_output, y_onehot)
        union = w * (einsum("bcdhw->bc", net_output) + einsum("bcdhw->bc", y_onehot))
        
        divided = -2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()
        
        return gdc.float()


def compute_dsc(pred, target, smooth=1e-6):
    """Compute DSC"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dsc = (2. * intersection + smooth) / (union + smooth)
    return dsc.item()


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
        d_up, h_up, w_up = x_up.shape[2:]
        d_skip, h_skip, w_skip = x_skip.shape[2:]
        
        if d_up != d_skip or h_up != h_skip or w_up != w_skip:
            diff_d = d_skip - d_up
            diff_h = h_skip - h_up
            diff_w = w_skip - w_up
            
            if diff_d > 0 or diff_h > 0 or diff_w > 0:
                padding = [
                    max(0, diff_w // 2), max(0, diff_w - diff_w // 2),
                    max(0, diff_h // 2), max(0, diff_h - diff_h // 2),
                    max(0, diff_d // 2), max(0, diff_d - diff_d // 2)
                ]
                x_up = F.pad(x_up, padding)
            elif diff_d < 0 or diff_h < 0 or diff_w < 0:
                d_start = max(0, -diff_d // 2)
                h_start = max(0, -diff_h // 2)
                w_start = max(0, -diff_w // 2)
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
        
        x = self.final_conv(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.encoder = ResNet3DEncoder(encoder)
        self.decoder = UNetDecoder3D(num_classes=num_classes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        enc_features = self.encoder(x)
        seg_logits = self.decoder(enc_features)
        
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='trilinear', align_corners=False)
        
        return seg_logits

def check_all_patches():
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
    
    # One working, one failing
    cases = ['BBS318', 'BBS328']
    
    patch_size = np.array([96, 96, 96])
    half_size = patch_size // 2
    num_patches = 10
    
    np.random.seed(42)  # Same as test!
    
    for case_id in cases:
        print("\n" + "="*70)
        print(f"{case_id}")
        print("="*70)
        
        npz_file = neuralcup_dir / f'{case_id}.npz'
        data = np.load(npz_file)
        volume = data['image']
        mask = data['lesion_mask']
        
        vol_shape = np.array(volume.shape)
        min_center = half_size
        max_center = vol_shape - half_size
        
        for dim in range(3):
            if min_center[dim] >= max_center[dim]:
                min_center[dim] = vol_shape[dim] // 2
                max_center[dim] = vol_shape[dim] // 2
        
        # Check all 10 patches
        for patch_idx in range(num_patches):
            center = np.array([
                np.random.randint(min_center[0], max_center[0] + 1),
                np.random.randint(min_center[1], max_center[1] + 1),
                np.random.randint(min_center[2], max_center[2] + 1)
            ])
            
            lower = center - half_size
            upper = center + half_size
            
            patch = volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
            patch_mask = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
            
            # Predict
            with torch.no_grad():
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                logits = model(patch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                prob_lesion = probs[1]
                pred_voxels = (prob_lesion > 0.5).sum()
                gt_voxels = patch_mask.sum()
                max_prob = prob_lesion.max()
                mean_prob = prob_lesion.mean()
                
                # Probability AT lesion locations
                if gt_voxels > 0:
                    lesion_coords = np.where(patch_mask > 0)
                    prob_at_lesion = prob_lesion[lesion_coords].mean()
                else:
                    prob_at_lesion = 0
                
                print(f"Patch {patch_idx+1}: GT={int(gt_voxels):5d}, Pred={int(pred_voxels):5d}, "
                      f"MaxProb={max_prob:.3f}, MeanProb={mean_prob:.4f}, ProbAtLesion={prob_at_lesion:.4f}")


if __name__ == '__main__':
    check_all_patches()

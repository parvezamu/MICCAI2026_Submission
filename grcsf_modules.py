"""
grcsf_modules.py

Implementation of GRCSF components:
- Global Compensation Unit (GCU)
- Regional Compensation Unit (RCU)
- MAE-based residual map generation

Author: Parvez
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class SqueezeExcitation3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D features
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class GlobalCompensationUnit(nn.Module):
    """
    Global Compensation Unit (GCU) from GRCSF paper
    
    Recovers information lost during downsampling by:
    1. Re-upsampling downsampled features
    2. Computing cosine similarity with skip features
    3. Creating residual to enhance skip connections
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se_skip = SqueezeExcitation3D(channels, reduction)
        self.se_up = SqueezeExcitation3D(channels, reduction)
    
    def forward(self, skip_feature, upsampled_feature):
        """
        Args:
            skip_feature: Feature from encoder (F in paper) [B, C, D, H, W]
            upsampled_feature: Feature from decoder (U in paper) [B, C, D, H, W]
        
        Returns:
            compensated_skip: Enhanced skip connection [B, C, D, H, W]
            residual_map: Residual showing information recovery [B, C, D, H, W]
        """
        # Get dimensions
        target_size = skip_feature.shape[2:]
        
        # Apply SE to decoder feature
        u_weighted = self.se_up(upsampled_feature)
        
        # Upsample to match skip feature size (this simulates re-upsampling after downsampling)
        # In practice, we already have upsampled_feature at the right size
        ru = u_weighted
        
        # Apply SE to skip feature
        f_weighted = self.se_skip(skip_feature)
        
        # Compute cosine similarity pixel-wise
        # Normalize along channel dimension
        ru_norm = F.normalize(ru, p=2, dim=1)
        f_norm = F.normalize(f_weighted, p=2, dim=1)
        
        # Cosine similarity (element-wise product then sum over channels)
        similarity = (ru_norm * f_norm).sum(dim=1, keepdim=True)  # [B, 1, D, H, W]
        
        # Create residual map
        residual_map = similarity * skip_feature  # Element-wise scaling
        
        # Enhanced skip connection
        compensated_skip = skip_feature + residual_map
        
        return compensated_skip, residual_map


class CrossAttention3D(nn.Module):
    """
    Cross-attention between decoder features and MAE residual maps
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query_conv = nn.Conv3d(channels, channels, 1)
        self.key_conv = nn.Conv3d(channels, channels, 1)
        self.value_conv = nn.Conv3d(channels, channels, 1)
        self.out_conv = nn.Conv3d(channels, channels, 1)
        
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, decoder_features, mae_features):
        """
        Args:
            decoder_features: [B, C, D, H, W]
            mae_features: [B, C, D, H, W]
        
        Returns:
            attended_features: [B, C, D, H, W]
        """
        B, C, D, H, W = decoder_features.shape
        
        # Generate Q, K, V
        Q = self.query_conv(decoder_features)  # [B, C, D, H, W]
        K = self.key_conv(mae_features)        # [B, C, D, H, W]
        V = self.value_conv(mae_features)      # [B, C, D, H, W]
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b (h c) d h w -> b h (d h w) c', h=self.num_heads)
        K = rearrange(K, 'b (h c) d h w -> b h (d h w) c', h=self.num_heads)
        V = rearrange(V, 'b (h c) d h w -> b h (d h w) c', h=self.num_heads)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, DHW, DHW]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, DHW, C//H]
        
        # Reshape back
        out = rearrange(out, 'b h (d h w) c -> b (h c) d h w', 
                       h=self.num_heads, d=D, h=H, w=W)
        
        # Output projection
        out = self.out_conv(out)
        
        return out


class PatchImportanceScorer(nn.Module):
    """
    Scores importance of each patch based on likelihood of containing lesions
    """
    def __init__(self, channels, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        
        # Global context encoding
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Importance scoring network
        self.scorer = nn.Sequential(
            nn.Conv3d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, C, D, H, W]
        
        Returns:
            importance_scores: [B, 1, D, H, W]
        """
        # Compute patch-wise importance
        scores = self.scorer(features)
        return scores


class RegionalCompensationUnit(nn.Module):
    """
    Regional Compensation Unit (RCU) from GRCSF paper
    
    Fuses MAE residual maps with decoder features using:
    1. Cross-attention between residual maps and decoder features
    2. Patch-based importance scoring
    3. Weighted fusion with learnable weights
    """
    def __init__(self, channels, patch_size=8, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        
        # Cross-attention for MAE residual maps (50% and 75% masking)
        self.cross_attn_50 = CrossAttention3D(channels, num_heads)
        self.cross_attn_75 = CrossAttention3D(channels, num_heads)
        
        # Importance scorer
        self.importance_scorer = PatchImportanceScorer(channels, patch_size)
        
        # Learnable fusion weights
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, decoder_features, mae_residual_50, mae_residual_75):
        """
        Args:
            decoder_features: [B, C, D, H, W]
            mae_residual_50: [B, 1, D, H, W] - MAE residual map (50% masking)
            mae_residual_75: [B, 1, D, H, W] - MAE residual map (75% masking)
        
        Returns:
            fused_features: [B, C, D, H, W]
        """
        # Expand residual maps to match channel dimension
        mae_50_expanded = mae_residual_50.expand(-1, decoder_features.size(1), -1, -1, -1)
        mae_75_expanded = mae_residual_75.expand(-1, decoder_features.size(1), -1, -1, -1)
        
        # Cross-attention
        attn_50 = self.cross_attn_50(decoder_features, mae_50_expanded)
        attn_75 = self.cross_attn_75(decoder_features, mae_75_expanded)
        
        # Importance scoring
        importance = self.importance_scorer(decoder_features)
        
        # Weight attention outputs by importance
        weighted_50 = attn_50 * importance * self.w1
        weighted_75 = attn_75 * importance * self.w2
        
        # Fuse with original decoder features
        fused = decoder_features + weighted_50 + weighted_75
        
        # Final projection
        out = self.out_conv(fused)
        
        return out


class MAEResidualGenerator(nn.Module):
    """
    Generates residual maps from MAE reconstruction
    
    Uses pre-trained MAE models to:
    1. Reconstruct masked images
    2. Compute pixel-wise difference (residual)
    3. Average over multiple iterations for stability
    """
    def __init__(self, mae_model_50, mae_model_75, num_iterations=5):
        super().__init__()
        self.mae_50 = mae_model_50
        self.mae_75 = mae_model_75
        self.num_iterations = num_iterations
        
        # Freeze MAE models
        for param in self.mae_50.parameters():
            param.requires_grad = False
        for param in self.mae_75.parameters():
            param.requires_grad = False
        
        self.mae_50.eval()
        self.mae_75.eval()
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Input image [B, 1, D, H, W]
        
        Returns:
            residual_50: [B, 1, D, H, W]
            residual_75: [B, 1, D, H, W]
        """
        B = x.size(0)
        
        # Initialize accumulators
        residuals_50 = []
        residuals_75 = []
        
        # Generate multiple reconstructions and average
        for _ in range(self.num_iterations):
            # MAE 50% masking
            recon_50 = self.mae_50(x)
            residual_50 = torch.abs(x - recon_50)
            residuals_50.append(residual_50)
            
            # MAE 75% masking
            recon_75 = self.mae_75(x)
            residual_75 = torch.abs(x - recon_75)
            residuals_75.append(residual_75)
        
        # Average residuals
        avg_residual_50 = torch.stack(residuals_50).mean(dim=0)
        avg_residual_75 = torch.stack(residuals_75).mean(dim=0)
        
        return avg_residual_50, avg_residual_75


class GRCSFDecoder(nn.Module):
    """
    GRCSF Decoder with GCU and RCU modules
    """
    def __init__(self, encoder_channels=[64, 64, 128, 256, 512], 
                 decoder_channels=[256, 128, 64, 64],
                 num_classes=2,
                 use_gcu=True,
                 use_rcu=True,
                 rcu_layers=[1, 2, 3]):  # Which decoder layers to apply RCU
        super().__init__()
        
        self.use_gcu = use_gcu
        self.use_rcu = use_rcu
        self.rcu_layers = rcu_layers
        
        # Decoder blocks
        self.up4 = nn.ConvTranspose3d(encoder_channels[4], decoder_channels[0], 2, stride=2)
        self.dec4 = self._decoder_block(encoder_channels[3] + decoder_channels[0], decoder_channels[0])
        
        self.up3 = nn.ConvTranspose3d(decoder_channels[0], decoder_channels[1], 2, stride=2)
        self.dec3 = self._decoder_block(encoder_channels[2] + decoder_channels[1], decoder_channels[1])
        
        self.up2 = nn.ConvTranspose3d(decoder_channels[1], decoder_channels[2], 2, stride=2)
        self.dec2 = self._decoder_block(encoder_channels[1] + decoder_channels[2], decoder_channels[2])
        
        self.up1 = nn.ConvTranspose3d(decoder_channels[2], decoder_channels[3], 2, stride=2)
        self.dec1 = self._decoder_block(encoder_channels[0] + decoder_channels[3], decoder_channels[3])
        
        # Global Compensation Units
        if self.use_gcu:
            self.gcu4 = GlobalCompensationUnit(encoder_channels[3])
            self.gcu3 = GlobalCompensationUnit(encoder_channels[2])
            self.gcu2 = GlobalCompensationUnit(encoder_channels[1])
            self.gcu1 = GlobalCompensationUnit(encoder_channels[0])
        
        # Regional Compensation Units
        if self.use_rcu:
            if 0 in self.rcu_layers:
                self.rcu4 = RegionalCompensationUnit(decoder_channels[0])
            if 1 in self.rcu_layers:
                self.rcu3 = RegionalCompensationUnit(decoder_channels[1])
            if 2 in self.rcu_layers:
                self.rcu2 = RegionalCompensationUnit(decoder_channels[2])
            if 3 in self.rcu_layers:
                self.rcu1 = RegionalCompensationUnit(decoder_channels[3])
        
        # Final convolution
        self.final_conv = nn.Conv3d(decoder_channels[3], num_classes, 1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x_up, x_skip):
        """Ensure upsampled feature matches skip feature size"""
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], 
                                mode='trilinear', align_corners=False)
        return x_up
    
    def forward(self, encoder_features, mae_residuals=None):
        """
        Args:
            encoder_features: List of [x1, x2, x3, x4, x5] from encoder
            mae_residuals: Tuple of (residual_50, residual_75) or None
        
        Returns:
            segmentation logits: [B, num_classes, D, H, W]
        """
        x1, x2, x3, x4, x5 = encoder_features
        
        if mae_residuals is not None:
            mae_50, mae_75 = mae_residuals
        
        # Decoder level 4
        x = self.up4(x5)
        x = self._match_size(x, x4)
        
        if self.use_gcu:
            x4_comp, _ = self.gcu4(x4, x)
            x = torch.cat([x, x4_comp], dim=1)
        else:
            x = torch.cat([x, x4], dim=1)
        
        x = self.dec4(x)
        
        if self.use_rcu and 0 in self.rcu_layers and mae_residuals is not None:
            mae_50_sized = F.interpolate(mae_50, size=x.shape[2:], mode='trilinear', align_corners=False)
            mae_75_sized = F.interpolate(mae_75, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = self.rcu4(x, mae_50_sized, mae_75_sized)
        
        # Decoder level 3
        x = self.up3(x)
        x = self._match_size(x, x3)
        
        if self.use_gcu:
            x3_comp, _ = self.gcu3(x3, x)
            x = torch.cat([x, x3_comp], dim=1)
        else:
            x = torch.cat([x, x3], dim=1)
        
        x = self.dec3(x)
        
        if self.use_rcu and 1 in self.rcu_layers and mae_residuals is not None:
            mae_50_sized = F.interpolate(mae_50, size=x.shape[2:], mode='trilinear', align_corners=False)
            mae_75_sized = F.interpolate(mae_75, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = self.rcu3(x, mae_50_sized, mae_75_sized)
        
        # Decoder level 2
        x = self.up2(x)
        x = self._match_size(x, x2)
        
        if self.use_gcu:
            x2_comp, _ = self.gcu2(x2, x)
            x = torch.cat([x, x2_comp], dim=1)
        else:
            x = torch.cat([x, x2], dim=1)
        
        x = self.dec2(x)
        
        if self.use_rcu and 2 in self.rcu_layers and mae_residuals is not None:
            mae_50_sized = F.interpolate(mae_50, size=x.shape[2:], mode='trilinear', align_corners=False)
            mae_75_sized = F.interpolate(mae_75, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = self.rcu2(x, mae_50_sized, mae_75_sized)
        
        # Decoder level 1
        x = self.up1(x)
        x = self._match_size(x, x1)
        
        if self.use_gcu:
            x1_comp, _ = self.gcu1(x1, x)
            x = torch.cat([x, x1_comp], dim=1)
        else:
            x = torch.cat([x, x1], dim=1)
        
        x = self.dec1(x)
        
        if self.use_rcu and 3 in self.rcu_layers and mae_residuals is not None:
            mae_50_sized = F.interpolate(mae_50, size=x.shape[2:], mode='trilinear', align_corners=False)
            mae_75_sized = F.interpolate(mae_75, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = self.rcu1(x, mae_50_sized, mae_75_sized)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

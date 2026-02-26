"""
models/resnet3d.py

3D ResNet for SimCLR foundation model

Author: Parvez
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    """3D Basic Block for ResNet"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D ResNet encoder for SimCLR"""
    
    def __init__(self, block, num_blocks, in_channels=1, base_width=64):
        super(ResNet3D, self).__init__()
        self.in_planes = base_width
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels, base_width, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_width)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Output dimension
        self.out_dim = base_width * 8 * block.expansion
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, 1, D, H, W)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return out


def resnet3d_18(**kwargs):
    """ResNet3D-18"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)


def resnet3d_34(**kwargs):
    """ResNet3D-34"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], **kwargs)


class SimCLRModel(nn.Module):
    """SimCLR model with projection head"""
    
    def __init__(
        self,
        encoder,
        projection_dim=128,
        hidden_dim=512
    ):
        super(SimCLRModel, self).__init__()
        
        self.encoder = encoder
        encoder_dim = encoder.out_dim
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x):
        # Get encoder features
        h = self.encoder(x)
        
        # Project to lower dimension
        z = self.projection_head(h)
        
        return h, z


if __name__ == '__main__':
    # Test the model
    model = SimCLRModel(resnet3d_18())
    
    # Test input
    x = torch.randn(2, 1, 96, 128, 128)
    h, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {h.shape}")
    print(f"Projection output shape: {z.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


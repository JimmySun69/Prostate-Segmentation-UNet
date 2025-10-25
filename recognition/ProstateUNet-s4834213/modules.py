# modules.py
"""
Contains the model architecture for the deep 3D U-Net.
This file is self-contained and uses only PyTorch components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
# Helper Modules
# ===================================================================

class AdaIN3D(nn.Module):
    """Adaptive Instance Normalization for 3D inputs."""
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, ch, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1, 1))
        self.inorm = nn.InstanceNorm3d(ch, affine=False)

    def forward(self, x):
        return self.a * x + self.b * self.inorm(x)

class DilatedConvBlock3D(nn.Module):
    """A 3D convolutional block with dilation and AdaIN, used in the CAM."""
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = dilation
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False)
        self.norm = AdaIN3D(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ContextAggregationModule(nn.Module):
    """Aggregates multi-scale context using dilated convolutions at the bottleneck."""
    def __init__(self, ch):
        super().__init__()
        self.blocks = nn.Sequential(
            DilatedConvBlock3D(ch, ch, dilation=2),
            DilatedConvBlock3D(ch, ch, dilation=4),
            DilatedConvBlock3D(ch, ch, dilation=8),
        )

    def forward(self, x):
        # Add a residual connection for training stability
        return x + self.blocks(x)

class ConvBlock3D(nn.Module):
    """A standard double-convolution block used throughout the U-Net."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# ===================================================================
# Main Model: Deep 3D U-Net
# ===================================================================

class ImprovedUNet3D_CAN(nn.Module):
    """
    A deep 3D U-Net with multiple downsampling stages to capture multi-scale context.
    It uses skip connections to combine deep, contextual features with shallow,
    fine-grained features.
    """
    def __init__(self, in_ch=1, out_ch=6, base=32):
        super().__init__()
        
        # --- Encoder Path ---
        self.d1 = ConvBlock3D(in_ch, base)
        self.p1 = nn.MaxPool3d(2)
        
        self.d2 = ConvBlock3D(base, base * 2)
        self.p2 = nn.MaxPool3d(2)
        
        self.d3 = ConvBlock3D(base * 2, base * 4)
        self.p3 = nn.MaxPool3d(2)
        
        # --- Bottleneck with Context Aggregation ---
        self.bottleneck = ConvBlock3D(base * 4, base * 8)
        self.cam = ContextAggregationModule(base * 8)
        
        # --- Decoder Path ---
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.up3 = ConvBlock3D(base * 8, base * 4) # Input is (base*4 + base*4) from skip
        
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.up2 = ConvBlock3D(base * 4, base * 2) # Input is (base*2 + base*2) from skip
        
        self.u1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.up1 = ConvBlock3D(base * 2, base)   # Input is (base + base) from skip
        
        # --- Final Output Layer ---
        self.outc = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip1 = self.d1(x)
        p1 = self.p1(skip1)
        
        skip2 = self.d2(p1)
        p2 = self.p2(skip2)
        
        skip3 = self.d3(p2)
        p3 = self.p3(skip3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        b = self.cam(b)
        
        # Decoder with skip connections
        up3 = self.u3(b)
        merge3 = torch.cat([up3, skip3], dim=1)
        up3 = self.up3(merge3)
        
        up2 = self.u2(up3)
        merge2 = torch.cat([up2, skip2], dim=1)
        up2 = self.up2(merge2)
        
        up1 = self.u1(up2)
        merge1 = torch.cat([up1, skip1], dim=1)
        up1 = self.up1(merge1)
        
        return self.outc(up1)

"""
modules.py
Contains the model architecture and custom loss function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# = aCAN3D-style Improved U-Net
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
    """A 3D convolutional block with dilation and AdaIN."""
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = dilation
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False)
        self.norm = AdaIN3D(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ContextAggregationModule(nn.Module):
    """Aggregates multi-scale context using dilated convolutions."""
    def __init__(self, ch):
        super().__init__()
        self.blocks = nn.Sequential(
            DilatedConvBlock3D(ch, ch, dilation=2),
            DilatedConvBlock3D(ch, ch, dilation=4),
            DilatedConvBlock3D(ch, ch, dilation=8),
            DilatedConvBlock3D(ch, ch, dilation=1),
        )

    def forward(self, x):
        return self.blocks(x)

class ImprovedUNet3D_CAN(nn.Module):
    """The main Improved 3D U-Net model architecture."""
    def __init__(self, in_ch=1, out_ch=6, base=32):
        super().__init__()
        self.down1 = DilatedConvBlock3D(in_ch, base)
        self.down2 = DilatedConvBlock3D(base, base * 2, dilation=1)
        self.pool = nn.MaxPool3d(2)
        self.cam = ContextAggregationModule(base * 2)
        self.up = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.outc = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(self.down2(x1))
        x_cam = self.cam(x2)
        x_up = self.up(x_cam)
        # Skip connection
        x_sum = x_up + x1
        return self.outc(x_sum)

# Loss: Dice-Squared Focal (DSF)
class DiceSquaredFocalLoss(nn.Module):
    """A custom loss combining Dice-Squared and Focal Loss."""
    def __init__(self, gamma=2.0, eps=1e-6):
        super().__init__()
        self.gamma, self.eps = gamma, eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        t = targets.squeeze(1).long()
        onehot = F.one_hot(t, num_classes).permute(0, 4, 1, 2, 3).float()

        # Dice-Squared Loss component
        diff2 = (probs - onehot).pow(2)
        numer = diff2.sum(dim=(0, 2, 3, 4))
        denom = probs.pow(2).sum(dim=(0, 2, 3, 4)) + onehot.pow(2).sum(dim=(0, 2, 3, 4)) + self.eps
        dsl = (numer / denom + diff2.mean(dim=(0, 2, 3, 4))).mean()

        # Focal Loss component
        pt = (probs * onehot).sum(dim=1).clamp_min(self.eps)
        focal = -((1 - pt).pow(self.gamma) * pt.log()).mean()
        
        return dsl + focal


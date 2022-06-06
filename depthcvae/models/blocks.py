import torch
import torch.nn as nn

from typing import Tuple, Optional, Callable
from torch import Tensor

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(self.up(x))


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor: int = 2, out_size: Tuple = None):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        if out_size is not None:
            self.up = nn.Upsample(out_size=out_size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        return self.up(self.conv(x))

class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # TODO: Fix so bilinear doesn't need an extra Conv layer
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # Only for reducing channels
            )
            self.conv = BasicBlock(out_channels*2, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = BasicBlock(out_channels*2, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x1, x2], dim=1)
        
        return self.conv(x)

class ResUpSkip(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # TODO: Fix so bilinear doesn't need an extra Conv layer
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # Only for reducing channels
            )
            self.conv = BasicBlock(out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = BasicBlock(out_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = x1 + x2
        
        return self.conv(x)

class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDown, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.basicblock = BasicBlock(in_channels, out_channels)
    
    def forward(self, x):
        return self.basicblock(self.down(x))

class ResDownCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDownCat, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.basicblock = BasicBlock(out_channels*2, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.down(x1)
        x1 = self.norm(x1)

        x = torch.cat([x1, x2], dim=1)

        return self.basicblock(x)

class ResCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.basicblock = BasicBlock(in_channels*2, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        return self.basicblock(x)

class ResInCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.basicblock = BasicBlock(out_channels*2, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.down(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)

        x = torch.cat([x1, x2], dim=1)

        return self.basicblock(x)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                do_downsample: bool = True, groups: int = 1, base_width: int = 64,
                dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.downsaple = None
        if do_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1),
                nn.BatchNorm2d(planes)
            )
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
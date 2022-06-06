import torch.nn as nn
from torchvision.models import resnet18

from .blocks import ResUp, BasicBlock, OutConv

class UNet(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(UNet, self).__init__()
        model = resnet18(pretrained=cfg.MODEL.UNET.ENCODER.PRETRAINED)

        conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)                   # Replace in down1 if using other input dimensions
        self.down1 = nn.Sequential(model.conv1, model.bn1, model.relu)  # (3, 64)
        self.down2 = nn.Sequential(model.maxpool, model.layer1)         # (64, 64)
        self.down3 = model.layer2                                       # (64, 128)
        self.down4 = model.layer3                                       # (128, 256)
        self.down5 = model.layer4                                       # (256, 512)

        self.d_inc = BasicBlock(512, 512)
        self.up1 = ResUp(512, 256, bilinear)
        self.up2 = ResUp(256, 128, bilinear)
        self.up3 = ResUp(128, 64, bilinear)
        self.up4 = ResUp(64, 64, bilinear)
        self.out = nn.Sequential(
            OutConv(64, 1),
            nn.Sigmoid()
        )

        self.encoder = [self.down1, self.down2, self.down3, self.down4, self.down5]
        self.decoder = [self.d_inc, self.up1, self.up2, self.up3, self.up4, self.out]

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x1_out = self.d_inc(x5)
        x2_out = self.up1(x1_out, x4)
        x3_out = self.up2(x2_out, x3)
        x4_out = self.up3(x3_out, x2)
        x5_out = self.up4(x4_out, x1)

        feature_maps = (x1_out, x2_out, x3_out, x4_out, x5_out)

        out = self.out(x5_out)

        return feature_maps, out
""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .aspp import build_aspp


class UNet_res(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 3, bilinear=True):
        super(UNet_res, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.bottom = DoubleConv(1024 // factor, 1024 // factor)
        self.aspp = build_aspp(1024 // factor, 16, torch.nn.InstanceNorm2d)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up_without_cat(128//factor, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_5_bottom = self.bottom(x5)
        x_5_aspp = self.aspp(x5)
        x_5 = x_5_bottom + x_5_aspp
        x6 = self.up1(x_5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8)
        logits = self.outc(x9)
        return torch.tanh(x + logits)

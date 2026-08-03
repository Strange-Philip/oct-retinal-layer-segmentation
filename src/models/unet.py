import torch.nn as nn

from .double_conv import DoubleConv
from .down_block import DownBlock
from .up_block import UpBlock

features = [32,64,128,256]

class UNet(nn.Module):

    def __init__(self, n_channels=1, n_classes=8):
        super().__init__()

        self.inc = DoubleConv(n_channels, features[0])


        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        self.down4 = DownBlock(features[3], features[3] * 2)

        self.up1 = UpBlock(features[3] * 2, features[3])
        self.up2 = UpBlock(features[3], features[2])
        self.up3 = UpBlock(features[2], features[1])
        self.up4 = UpBlock(features[1], features[0])

        self.outc = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits
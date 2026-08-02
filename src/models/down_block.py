import torch.nn as nn

from .double_conv import DoubleConv


class DownBlock(nn.Module):
    """
    Downsampling block:
    MaxPool -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)
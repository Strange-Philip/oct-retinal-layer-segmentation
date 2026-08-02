import torch
import torch.nn as nn

from .double_conv import DoubleConv


class UpBlock(nn.Module):
    """
    Upsampling block:
    Transposed Conv -> Concatenate Skip -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Handle possible size mismatches
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]

        x = nn.functional.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)
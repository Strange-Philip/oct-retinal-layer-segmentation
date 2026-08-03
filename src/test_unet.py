import torch
from src.models.unet import UNet

model = UNet()

x = torch.randn(1, 1, 128, 128)

y = model(x)

print(y.shape)
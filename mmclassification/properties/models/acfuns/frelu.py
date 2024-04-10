import torch
import torch.nn as nn


class FReLU(nn.Module):
    def __init__(self, in_channel, k_size=3):
        super(FReLU,self).__init__()
        self.conv_hw = nn.Conv2d(in_channel, in_channel, k_size, stride=1, padding=1, groups=in_channel, bias=False)
        self.normal = nn.BatchNorm2d(in_channel)

    def forward(self,x):
        x_hw = self.conv_hw(x)
        x_hw = self.normal(x_hw)
        return torch.maximum(x_hw, x)

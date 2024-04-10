import torch
import math
import torch.nn as nn


class CReLU(nn.Module):
    def __init__(self, in_channel, k_size=3):
        super(CReLU,self).__init__()
        self.conv_hw = nn.Conv2d(in_channel, in_channel, k_size, stride=1, padding=1, groups=in_channel, bias=False)
        self.conv_c = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=False)
        def backward_hook_hw(grad):
            out =grad.clone()
            out[self.mask_hw]=0
            return out
        self.mask_hw = torch.zeros(self.conv_hw.weight.shape).bool()
        self.register_buffer('hw',torch.Tensor([[1,0,1],[0,0,0],[1,0,1]]).bool())
        self.register_parameter('residual', nn.Parameter(torch.rand(1, in_channel, 1, 1)))
        self.register_buffer('presidual',torch.ones(1, in_channel, 1, 1))
        self.mask_hw += self.hw
        self.conv_hw.weight.data[self.mask_hw.bool()]=0
        self.conv_hw.weight.register_hook(backward_hook_hw)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x_chw = self.conv_c(self.conv_hw(x))
        x = (self.presidual - self.residual) * self.relu(x_chw) + self.residual * x

        return x
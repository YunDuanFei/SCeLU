import torch
import torch.nn as nn
import math


class GeLU(nn.Module):
    def __init__(self, in_channel):
        super(GeLU,self).__init__()
        self.register_buffer('alpha',torch.sqrt(torch.Tensor([2 / math.pi]).float()))

    def forward(self,x):
        x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return x

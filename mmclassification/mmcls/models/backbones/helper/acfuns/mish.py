import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(x):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return x * torch.tanh(F.softplus(x))

class Mish(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.mish = mish

    def forward(self, x):
        return self.mish(x)

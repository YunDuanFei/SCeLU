import torch
import torch.nn as nn


class Acon(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    def __init__(self, in_channel):
        super(Acon,self).__init__()
        self.p1 = nn.Parameter(torch.randn(1, in_channel, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, in_channel, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, in_channel, 1, 1))

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

import torch
import torch.nn as nn


class PReLU(nn.Module):
    def __init__(self, in_channel):
        super(PReLU,self).__init__()
        self.prelu = nn.PReLU()

    def forward(self,x):
        x = self.prelu(x)
        return x

import torch
import torch.nn as nn


class ELU(nn.Module):
    def __init__(self, in_channel):
        super(ELU,self).__init__()
        self.elu = nn.ELU(inplace=True)

    def forward(self,x):
        x = self.elu(x)
        return x

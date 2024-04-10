import torch
import torch.nn as nn


class ReLU(nn.Module):
    def __init__(self, in_channel):
        super(ReLU,self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu(x)
        return x

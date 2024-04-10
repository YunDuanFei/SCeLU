import torch
import torch.nn as nn


class ReLU6(nn.Module):
    def __init__(self, in_channel):
        super(ReLU6,self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self,x):
        x = self.relu6(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.swish(x)

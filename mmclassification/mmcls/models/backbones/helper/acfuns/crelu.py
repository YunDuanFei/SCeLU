import torch
import math
import torch.nn as nn


class CReLU(nn.Module):
    def __init__(self, in_channel, k_size=3, gamma=1.2, b=0.1):
        super(CReLU,self).__init__()
        t = int(abs(math.log(in_channel, 2) + b) / gamma)
        ck = t if t % 2 else t + 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cconv = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.lsconv = nn.Conv1d(1, 1, kernel_size=ck, padding=int(ck / 2), bias=True)
        self.sigmoid = nn.Sigmoid()
        self.hw_conv = nn.Conv2d(in_channel, in_channel, k_size, stride=1, padding=1, groups=in_channel, bias=True)
        self.hw_bn = nn.BatchNorm2d(in_channel)
        self.c_bn = nn.BatchNorm2d(in_channel)

    def forward(self,x):
        N, C, H, W = x.shape
        x_ls = self.softmax(self.cconv(x).view(N, 1, H*W).transpose(-1, -2))
        x_ls = torch.matmul(x.view(N, C, H*W), x_ls)
        x_ls = ((x_ls + self.avgpool(x).squeeze(-1)) / 2.).transpose(-1, -2)
        x_ls = self.sigmoid(self.c_bn(self.lsconv(x_ls).transpose(-1, -2).unsqueeze(-1)))
        x_ls = self.hw_bn(x_ls * self.hw_conv(x))
        x = torch.maximum(x_ls, x)

        return x

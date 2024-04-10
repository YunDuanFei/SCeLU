import torch
import torch.nn as nn
from .acfuns import Acon, CReLU, DReLU, ELU, FReLU, GeLU, MetaAcon, Mish, PReLU, ReLU6, ReLU, Swish
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, acfun, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.acfun1 = acfun(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.acfun2 = acfun(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.acfun1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.acfun2(out)
        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, acfun, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.acfun1 = acfun(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], acfun, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], acfun, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], acfun, stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, acfun, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, acfun, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.acfun1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# CIFAR-10 models
def ResNet20_Acon():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], Acon)

def ResNet20_CReLU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], CReLU)

def ResNet20_DReLU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], DReLU)

def ResNet20_ELU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], ELU)

def ResNet20_FReLU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], FReLU)

def ResNet20_GeLU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], GeLU)

def ResNet20_MetaAcon():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], MetaAcon)

def ResNet20_Mish():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], Mish)

def ResNet20_PReLU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], PReLU)

def ResNet20_ReLU6():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], ReLU6)

def ResNet20_ReLU():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], ReLU)

def ResNet20_Swish():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], Swish)


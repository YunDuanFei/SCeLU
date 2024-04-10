############################################################################################################################################
# code reference https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
# https://github.com/rwightman/pytorch-image-models/blob/f7325c7b712100f79a9ab4ae54118d259c11bacf/timm/models/layers/global_context.py#L19
# Paper: `GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond` - https://arxiv.org/abs/1904.11492
############################################################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class GcAtt(nn.Module):
    def __init__(self, in_channels):
        super(GcAtt, self).__init__()
        self.channel = create_attn(attn_type='gc', channels=in_channels)

    def forward(self, x):
        y = self.channel(x)

        return y
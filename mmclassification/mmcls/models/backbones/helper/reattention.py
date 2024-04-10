from typing import Dict

import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.utils import Registry

from .attens import CoordAtt, EcaAtt, GcAtt, SeAtt

ATTENTION_LAYERS = Registry('attention layer')
ATTENTION_LAYERS.register_module('CoordAtt', module=CoordAtt)
ATTENTION_LAYERS.register_module('EcaAtt', module=EcaAtt)
ATTENTION_LAYERS.register_module('GcAtt', module=GcAtt)
ATTENTION_LAYERS.register_module('SeAtt', module=SeAtt)

def build_attention_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build attention layer.
    Args:
        cfg (dict): The attention layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an attention layer.
    Returns:
        nn.Module: Created attention layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    att_type = cfg_.pop('type')
    if att_type not in ATTENTION_LAYERS:
        raise KeyError(f'Unrecognized layer type {att_type}')
    else:
        acf_layer = ATTENTION_LAYERS.get(att_type)
    layer = acf_layer(*args, **kwargs, **cfg_)

    return layer
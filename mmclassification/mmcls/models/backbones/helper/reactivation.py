from typing import Dict

import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.utils import Registry

from .acfuns import Acon, CReLU, DReLU, ELU, FReLU, GeLU, MetaAcon, Mish, PReLU, ReLU6, ReLU, Swish

ACTIVATION_LAYERS = Registry('activation layer')
ACTIVATION_LAYERS.register_module('Acon', module=Acon)
ACTIVATION_LAYERS.register_module('CReLU', module=CReLU)
ACTIVATION_LAYERS.register_module('DReLU', module=DReLU)
ACTIVATION_LAYERS.register_module('ELU', module=ELU)
ACTIVATION_LAYERS.register_module('FReLU', module=FReLU)
ACTIVATION_LAYERS.register_module('GeLU', module=GeLU)
ACTIVATION_LAYERS.register_module('MetaAcon', module=MetaAcon)
ACTIVATION_LAYERS.register_module('Mish', module=Mish)
ACTIVATION_LAYERS.register_module('PReLU', module=PReLU)
ACTIVATION_LAYERS.register_module('ReLU6', module=ReLU6)
ACTIVATION_LAYERS.register_module('ReLU', module=ReLU)
ACTIVATION_LAYERS.register_module('Swish', module=Swish)

def build_activation_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    acf_type = cfg_.pop('type')
    if acf_type not in ACTIVATION_LAYERS:
        raise KeyError(f'Unrecognized layer type {acf_type}')
    else:
        acf_layer = ACTIVATION_LAYERS.get(acf_type)
    layer = acf_layer(*args, **kwargs, **cfg_)

    return layer
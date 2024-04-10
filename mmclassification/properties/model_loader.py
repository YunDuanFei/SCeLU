import os
import torch, torchvision
from models import resnet

# map between model name and function
models = {
    'resnet20_acon'        : resnet.ResNet20_Acon,
    'resnet20_crelu'       : resnet.ResNet20_CReLU,
    'resnet20_drelu'       : resnet.ResNet20_DReLU,
    'resnet20_elu'         : resnet.ResNet20_ELU,
    'resnet20_frelu'       : resnet.ResNet20_FReLU,
    'resnet20_gelu'        : resnet.ResNet20_GeLU,
    'resnet20_metaacon'    : resnet.ResNet20_MetaAcon,
    'resnet20_mish'        : resnet.ResNet20_Mish,
    'resnet20_prelu'       : resnet.ResNet20_PReLU,
    'resnet20_relu6'       : resnet.ResNet20_ReLU6,
    'resnet20_relu'        : resnet.ResNet20_ReLU,
    'resnet20_swish'       : resnet.ResNet20_Swish,
}


def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)

        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net

def land_load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = load(model_name, model_file, data_parallel)
    return net
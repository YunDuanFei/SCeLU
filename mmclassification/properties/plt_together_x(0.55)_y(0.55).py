"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import plot_2D
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    acon = './trained_nets/resnet20_acon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    crelu = './trained_nets/resnet20_crelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    drelu = './trained_nets/resnet20_drelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    elu = './trained_nets/resnet20_elu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    frelu = './trained_nets/resnet20_frelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    gelu = './trained_nets/resnet20_gelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    metaacon = './trained_nets/resnet20_metaacon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    mish = './trained_nets/resnet20_mish_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    prelu = './trained_nets/resnet20_prelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    relu6 = './trained_nets/resnet20_relu6_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    relu = './trained_nets/resnet20_relu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'
    swish = './trained_nets/resnet20_swish_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.55,0.55,51]x[-0.55,0.55,51].h5'

    acfuns = [acon, crelu, drelu, elu, frelu, gelu, mish, prelu, relu6, relu, swish, metaacon]

    for fun in acfuns:
        plot_2D.plot_2d_contour(fun, 'train_loss', 0.1, 10, 0.5, False)

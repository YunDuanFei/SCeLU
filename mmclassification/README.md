# SCeLU
## Learning Local Spatial and Global Context Activation for Visual Recognition

Supported Models 
----------
mobilenet_v2_0.17, mobilenet_v2_0.5, mobilenet_v2_0.75, mobilenet_v2_1.0

shufflenet_v2_0.5, shufflenet_v2_1.0, shufflenet_v2_1.5

resnet18, resnet34, resnet50, resnet101
----------

Supported activation functions
--------------------------------------------------------------------------------
Acon, CReLU, DReLU, ELU, FReLU, GeLU, MetaAcon, Mish, PReLU, ReLU6, ReLU, Swish
--------------------------------------------------------------------------------

Usage
----------
```
bash ./tools/dist_train.sh ./configs/XXXX/XXXX/XXXX.py XX

for example
bash ./tools/dist_train.sh ./configs/mobilenet_v2/mobilenet_v2_1.0x/mobilenet_v2_1.0x_CReLU.py 2
```
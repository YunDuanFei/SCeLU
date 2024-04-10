################################################################after################################################################
# Acon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Acon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Acon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.acfun3

# CReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_CReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/CReLU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.layer3.22.acfun3

# DReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_DReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/DReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.layer3.22.acfun3

# ELU 
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ELU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ELU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.layer3.22.acfun3

# FReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_FReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/FReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.layer3.22.acfun3

# GeLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_GeLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/GeLU/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.acfun3

# MetaAcon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_MetaAcon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/MetaAcon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.acfun3

# Mish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Mish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Mish/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.acfun3

# PReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_PReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/PReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.layer3.22.acfun3

# ReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.layer3.22.acfun3

# ReLU6
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU6.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU6/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.acfun3

# Swish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Swish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Swish/best_accuracy_top-1_epoch_99.pth --target-layers backbone.layer3.22.acfun3



################################################################before################################################################
# Acon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Acon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Acon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.bn3

# CReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_CReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/CReLU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.layer3.22.bn3

# DReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_DReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/DReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.layer3.22.bn3

# ELU 
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ELU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ELU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.layer3.22.bn3

# FReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_FReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/FReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.layer3.22.bn3

# GeLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_GeLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/GeLU/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.bn3

# MetaAcon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_MetaAcon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/MetaAcon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.bn3

# Mish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Mish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Mish/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.bn3

# PReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_PReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/PReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.layer3.22.bn3

# ReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.layer3.22.bn3

# ReLU6
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU6.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU6/best_accuracy_top-1_epoch_100.pth --target-layers backbone.layer3.22.bn3

# Swish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Swish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Swish/best_accuracy_top-1_epoch_99.pth --target-layers backbone.layer3.22.bn3


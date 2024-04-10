################################################################after################################################################
# Acon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Acon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Acon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.acfun1

# CReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_CReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/CReLU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.acfun1

# DReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_DReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/DReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.acfun1

# ELU 
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ELU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ELU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.acfun1

# FReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_FReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/FReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.acfun1

# GeLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_GeLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/GeLU/best_accuracy_top-1_epoch_100.pth --target-layers backbone.acfun1

# MetaAcon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_MetaAcon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/MetaAcon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.acfun1

# Mish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Mish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Mish/best_accuracy_top-1_epoch_100.pth --target-layers backbone.acfun1

# PReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_PReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/PReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.acfun1

# ReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.acfun1

# ReLU6
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU6.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU6/best_accuracy_top-1_epoch_100.pth --target-layers backbone.acfun1

# Swish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Swish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Swish/best_accuracy_top-1_epoch_99.pth --target-layers backbone.acfun1



################################################################before################################################################
# Acon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Acon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Acon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.bn1

# CReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_CReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/CReLU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.bn1

# DReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_DReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/DReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.bn1

# ELU 
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ELU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ELU/best_accuracy_top-1_epoch_99.pth --target-layers backbone.bn1

# FReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_FReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/FReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.bn1

# GeLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_GeLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/GeLU/best_accuracy_top-1_epoch_100.pth --target-layers backbone.bn1

# MetaAcon
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_MetaAcon.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/MetaAcon/best_accuracy_top-1_epoch_100.pth --target-layers backbone.bn1

# Mish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Mish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Mish/best_accuracy_top-1_epoch_100.pth --target-layers backbone.bn1

# PReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_PReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/PReLU/best_accuracy_top-1_epoch_96.pth --target-layers backbone.bn1

# ReLU
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU/best_accuracy_top-1_epoch_97.pth --target-layers backbone.bn1

# ReLU6
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_ReLU6.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/ReLU6/best_accuracy_top-1_epoch_100.pth --target-layers backbone.bn1

# Swish
python vis_cam.py --img ./Images/rows/ --config /home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet101/resnet101_2xb32_Swish.py --checkpoint /home/jb101a-0/projects/crelu/mmclassification/tools/visualizations/Weights/ResNets/resnet101/Swish/best_accuracy_top-1_epoch_99.pth --target-layers backbone.bn1


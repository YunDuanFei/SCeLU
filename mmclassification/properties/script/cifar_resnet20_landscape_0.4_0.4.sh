mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_acon --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_acon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_acon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_crelu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_crelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_crelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_drelu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_drelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_drelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_elu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_elu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_elu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_frelu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_frelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_frelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_gelu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_gelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_gelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_metaacon --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_metaacon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_metaacon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_mish --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_mish_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_mish_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_prelu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_prelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_prelu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_relu6 --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_relu6_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_relu6_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_swish --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_swish_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_swish_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet20_relu --x=-0.4:0.4:51 --y=-0.4:0.4:51 --model_file trained_nets/resnet20_relu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
python h52vtp.py --surf_file trained_nets/resnet20_relu_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.4,0.4,51]x[-0.4,0.4,51].h5 --surf_name train_loss --zmax  10 --log


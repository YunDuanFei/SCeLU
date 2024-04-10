import h5py





hfile = h5py.File('/home/jb101a-0/projects/crelu/mmclassification/properties/trained_nets/resnet20_acon_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=100/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-0.1,0.1,51]x[-0.1,0.1,51].h5')
if hfile.__bool__():
	hfile.close()
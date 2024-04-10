_base_ = [
    '../../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../../_base_/schedules/imagenet_bs1024_linearlr_bn_nowd.py',
    '../../_base_/default_runtime.py'
]


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ShuffleNetV2', widen_factor=1.5, act_cfg=dict(type='ELU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

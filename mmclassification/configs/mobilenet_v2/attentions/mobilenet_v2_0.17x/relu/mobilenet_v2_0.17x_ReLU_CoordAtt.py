_base_ = [
    '../../../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../../../_base_/schedules/imagenet_bs32_step_100.py',
    '../../../../_base_/default_runtime.py'
]



# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=0.17, act_cfg=dict(type='ReLU'), att_cfg=dict(type='CoordAtt')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

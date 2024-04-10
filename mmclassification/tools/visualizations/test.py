from pathlib import Path

path = Path('/home/jb101a-0/projects/crelu/mmclassification/configs/resnet/resnet18/resnet18_2xb32_ReLU.py')


print(path.parent)

print(Path(path).stem.split('_'))
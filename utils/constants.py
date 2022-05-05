import os
from pathlib import Path

curr_file_path = Path(os.path.abspath(__file__))
ROOT_DIR = curr_file_path.parent.parent.parent.absolute()

num_classes = {
	'MNIST': 10,
	'cifar10': 10,
	'SVHN': 10,
    'FASHION': 10,
    'cifar100': 100,
    'tiny-imagenet': 200
}

num_channels = {
    'MNIST': 1,
	'cifar10': 3,
	'SVHN': 3,
    'FASHION': 1,
    'cifar100': 3,
    'tiny-imagenet': 3
}
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from diy_dataset import VADdataset
from split_dataset import read_split_data

# _DATASETS_MAIN_PATH = '/home/Datasets'
_DATASETS_MAIN_PATH = '/home/jclou/kwsprj/data'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'VAD':os.path.join(_DATASETS_MAIN_PATH,"vaddata"),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    }
}


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)
    elif name == 'mnist':
        return datasets.MNIST(  root=_dataset_path['mnist'], 
                                train=train, 
                                transform=transform,
                                target_transform=target_transform,
                                download=download )
                    #             transform=transforms.Compose([
                    #    transforms.ToTensor(),
                    #    transforms.Normalize((0.1307,), (0.3081,))
                    #     ])
    elif name == 'VAD':
        train_voice_path, train_voice_label, val_voice_path, val_voice_label = read_split_data(
            root = _dataset_path['VAD'],
            val_rate = 0.3
        )
        if train:
            return VADdataset( 
                voice_path  = train_voice_path,
                voice_label = train_voice_label,
                transform   = None
            )
        else:
            return VADdataset(
                voice_path  = val_voice_path,
                voice_label = val_voice_label,
                transform   = None
            )
                        
                        

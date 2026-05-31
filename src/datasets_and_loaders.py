import torch
import os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils import data
import numpy as np
from glob import glob
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image
import random
from pathlib import Path
import pandas as pd
from torchvision.transforms import ToTensor
import requests
import tarfile
from typing import List, Callable, Tuple, Generator, Union
from tqdm import tqdm
import gdown

# ========================================================================================================
# ======================================== For BiasedMNIST: data =========================================
# Adapted from: https://github.com/clovaai/rebias/blob/master/datasets/colour_mnist.py

class BiasedMNIST(MNIST):

    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(self.biased_targets[index])


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9):
        super(ColourBiasedMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels)

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]


def get_biased_mnist_dataloader(root, batch_size, data_label_correlation,
                                n_confusing_labels=9, train=True, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                download=True, data_label_correlation=data_label_correlation,
                                n_confusing_labels=n_confusing_labels)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader



# ========================================================================================================
# =========================================== For CelebA: data ===========================================
# Adapted from:
#   https://github.com/EIDOSLAB/bridging-debiasing-privacy-deep-learning/blob/master/src/utils.py
#   https://github.com/EIDOSLAB/bridging-debiasing-privacy-deep-learning/blob/master/src/datasets/celeba.py
#   https://github.com/EIDOSLAB/bridging-debiasing-privacy-deep-learning/blob/master/src/train_celeba.py

# NOTE: automatic download disabled. Please manually download the data (see instructions in the README in root).

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, split='train', target='Blond_Hair', bias_attr='Male', unbiased=True, seed=42):
        path = root
        ensure_dir(path)

        if not os.path.isdir(os.path.join(path, 'CelebA')):
            # self.download_dataset(path)
            pass
        path = os.path.join(path, 'CelebA')
        self.split = split

        split_df = pd.read_csv(os.path.join(path, 'list_eval_partition.csv'))
        splits = {
            'train': 0,
            'valid': 1,
            'test': 2
        }
        partition_idx = split_df['split'] == splits[split]

        self.attr_df = pd.read_csv(os.path.join(path, 'list_attr_celeba.csv'), sep=' ').replace(-1, 0)
        
        # keep only relevant split train/val/test
        self.attr_df = self.attr_df[partition_idx]

        #swap male/female
        self.attr_df['Male'] = ~self.attr_df['Male']+2

        if split == 'valid':
            min_size = self.attr_df.groupby([target, bias_attr]).count().min()['image']

            # construct unbiased dataset (equal size for (at, ab))
            unbiased_df = self.attr_df.groupby([target, bias_attr]).apply(lambda group: group.sample(min_size, random_state=seed)).reset_index(drop=True)

            # remove bias-confictling pairs
            bias_conflicting_df = unbiased_df[unbiased_df[target] != unbiased_df[bias_attr]]

            if unbiased:
                self.attr_df = unbiased_df
            else:
                self.attr_df = bias_conflicting_df

        print(self.attr_df.groupby([target, bias_attr]).count())


        self.target = target
        self.bias_attr = bias_attr
        self.path = path

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        T_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        T_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


        self.T = T_train if split == 'train' else T_test

    # def download_dataset(self, path):     # COMMENTED TO AVOID DOWNLOADING...
    #     url = "https://drive.google.com/uc?id=1ebDzE4vsjPB4klNyTywjrZqGhUsFxZqb"
    #     output = os.path.join(path, 'celeba.tar.gz')
    #     print(f'=> Downloading CelebA dataset from {url}')
    #     gdown.download(url, output, quiet=False)

    #     print('=> Extracting dataset..')
    #     tar = tarfile.open(os.path.join(path, 'celeba.tar.gz'), 'r:gz')
    #     tar.extractall(path=path)
    #     tar.close()
    #     os.remove(output)


    def __getitem__(self, index):
        data = self.attr_df.iloc[index]
        img_name = data['image']
        bias = data[self.bias_attr]
        target_attr = data[self.target]

        image = PIL.Image.open(os.path.join(self.path, 'img_align_celeba', img_name))
        return self.T(image), target_attr, bias

    def __len__(self):
        return len(self.attr_df)
    

def load_celeba(args, shuffle=True):

    target_attr = 'Blond_Hair'
    bias_attr = 'Male'
    seed = args.seed
    batch_size = 256

    train_dataset = CelebA(root='data/', split='train', target=target_attr, bias_attr=bias_attr, seed=seed)
    valid_dataset = CelebA(root='data/', split='train', target=target_attr, bias_attr=bias_attr, seed=seed)
    
    unbiased_dataset = CelebA(root='data/', split='valid', unbiased=True, target=target_attr, bias_attr=bias_attr, seed=seed)
    conflicting_dataset = CelebA(root='data/', split='valid', unbiased=False, target=target_attr, bias_attr=bias_attr, seed=seed)

    target = target_attr
    min_size = int(0.2*train_dataset.attr_df.groupby([target, bias_attr]).count().min()['image'])
    valid_dataset.attr_df = valid_dataset.attr_df.groupby([target, bias_attr]).apply(lambda group: group.sample(min_size, random_state=seed)).reset_index(drop=True).copy()
    train_dataset.attr_df = train_dataset.attr_df[~train_dataset.attr_df.image.isin(valid_dataset.attr_df.image)].copy()
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True
    )

    unbiased_loader = torch.utils.data.DataLoader(
        unbiased_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True
    )

    conflicting_loader = torch.utils.data.DataLoader(
        conflicting_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True
    )

    return {'train': train_loader,'valid': valid_loader, 'unbiased_test': unbiased_loader, 'bias_conflicting_test': conflicting_loader}



# ========================================================================================================
# ===================================== For Cifar10-Corrupted: data ======================================
# Adapted from: https://github.com/EIDOSLAB/unbiased-contrastive-learning/blob/master/debiasing/data/corrupted_cifar.py

# NOTE: automatic download disabled. Please manually download the data (see instructions in the README in root).

class CorruptedCIFAR10(Dataset):
    def __init__(self, root, split, percent, transform=None, image_path_list=None):
        super().__init__()
        
        if not os.path.isdir(os.path.join(root, 'cifar10c')):
            self.download_dataset(root)
        root = os.path.join(root, 'cifar10c', percent)

        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split == 'train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def download_dataset(self, path):           # FIXME: COMMENTED TO AVOID DOWNLOADING...
        # url = "https://drive.google.com/uc?id=1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-"
        # output = os.path.join(path, 'cifar10c.tar.gz')
        # print(f'=> Downloading corrupted CIFAR10 dataset from {url}')
        # gdown.download(url, output, quiet=False)

        # print('=> Extracting dataset..')
        # tar = tarfile.open(os.path.join(path, 'cifar10c.tar.gz'), 'r:gz')
        # tar.extractall(path=path)
        # tar.close()
        # os.remove(output)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, bias = int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label, bias

# ========================================================================================================
# ====================================== For Multicolor-MNIST: data ======================================

# FROM https://github.com/zhihengli-UR/DebiAN/blob/main/datasets/multi_color_mnist.py


class MultiColorMNIST(Dataset):
    attribute_names = ["digit", "LColor", 'RColor']
    basename = 'multi_color_mnist'
    target_attr_index = 0
    left_color_bias_attr_index = 1
    right_color_bias_attr_index = 2

    def __init__(self, root, split, left_color_skew, right_color_skew, severity, transform=ToTensor()):
        super().__init__()

        assert split in ['train', 'valid']
        assert left_color_skew in [0.005, 0.01, 0.02, 0.05]
        assert right_color_skew in [0.005, 0.01, 0.02, 0.05]
        assert severity in [1, 2, 3, 4]

        root = os.path.join(root, self.basename, f'ColoredMNIST-SkewedA{left_color_skew}-SkewedB{right_color_skew}-Severity{severity}')
        assert os.path.exists(root), f'{root} does not exist'

        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)

        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        self.transform = transform

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, idx):
        image, attr = self.data[idx], self.attr[idx]
        if self.transform is not None:
            image = self.transform(image)

        # return image, attr  
        return image, attr[0], torch.vstack((attr[1], attr[2]))   # FORMAT: image, target, vstack(bias_left, bias_right)



# ========================================================================================================
# ========================================================================================================

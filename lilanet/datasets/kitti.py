import os
import random
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from lilanet.datasets.transforms import Compose, RandomHorizontalFlip, Normalize


class KITTI(data.Dataset):
    """`KITTI LiDAR`_ Dataset.

    Args:
        root (string): Root directory of the ``lidar_2d`` and ``ImageSet`` folder.
        split (string, optional): Select the split to use, ``train``, ``val`` or ``all``
        transform (callable, optional): A function/transform that  takes in distance, reflectivity
            and target tensors and returns a transformed version.
    """

    Class = namedtuple('Class', ['name', 'id', 'color'])

    classes = [
        Class('unknown', 0, (0, 0, 0)),
        Class('car', 1, (31, 143, 94)),
        Class('pedestrian', 2, (168, 140, 181)),
        Class('cyclist', 3, (148, 184, 224)),
    ]

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.lidar_path = os.path.join(self.root, 'lidar_2d')
        self.split = os.path.join(self.root, 'ImageSet', '{}.txt'.format(split))
        self.transform = transform
        self.lidar = []

        if split not in ['train', 'val', 'all']:
            raise ValueError('Invalid split! Use split="train", split="val" or split="all"')

        with open(self.split) as file:
            images = ['{}.npy'.format(x.strip()) for x in file.readlines()]
            for img in images:
                lidar_2d = os.path.join(self.lidar_path, img)
                self.lidar.append(lidar_2d)

    def __getitem__(self, index):
        record = np.load(self.lidar[index]).astype(np.float32, copy=False)
        record = torch.as_tensor(record).permute(2, 0, 1).contiguous()

        distance = record[3, :, :]
        reflectivity = record[4, :, :]
        label = record[5, :, :].long()

        if self.transform:
            distance, reflectivity, label = self.transform(distance, reflectivity, label)

        return distance, reflectivity, label

    def __len__(self):
        return len(self.lidar)

    @staticmethod
    def num_classes():
        return len(KITTI.classes)

    @staticmethod
    def mean():
        return [0.21, 12.12]

    @staticmethod
    def std():
        return [0.16, 12.32]

    @staticmethod
    def class_weights():
        return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        RandomHorizontalFlip(),
        Normalize(mean=KITTI.mean(), std=KITTI.std())
    ])


    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())


    dataset = KITTI('../../data/kitti', transform=joint_transforms)
    distance, reflectivity, label = random.choice(dataset)

    print('Distance size: ', distance.size())
    print('Reflectivity size: ', reflectivity.size())
    print('Label size: ', label.size())

    distance_map = Image.fromarray((255 * _normalize(distance.numpy())).astype(np.uint8))
    reflectivity_map = Image.fromarray((255 * _normalize(reflectivity.numpy())).astype(np.uint8))
    label_map = Image.fromarray((255 * _normalize(label.numpy())).astype(np.uint8))

    f = plt.figure()
    a = f.add_subplot(1, 2, 1)
    a.set_title('Distance')
    plt.imshow(distance_map)
    a = f.add_subplot(1, 2, 2)
    a.set_title('Reflectivity')
    plt.imshow(reflectivity_map)

    plt.show()

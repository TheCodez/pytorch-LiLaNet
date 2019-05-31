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
        Class('car', 1, (0, 0, 142)),
        Class('pedestrian', 2, (220, 20, 60)),
        Class('cyclist', 3, (119, 11, 32)),
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
        label = record[5, :, :]

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

    @staticmethod
    def get_colormap():
        cmap = torch.zeros([256, 3], dtype=torch.uint8)

        for cls in KITTI.classes:
            cmap[cls.id, :] = torch.tensor(cls.color, dtype=torch.uint8)

        return cmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        RandomHorizontalFlip(),
        Normalize(mean=KITTI.mean(), std=KITTI.std())
    ])


    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())


    def visualize_seg(label_map, one_hot=False):
        if one_hot:
            label_map = np.argmax(label_map, axis=-1)

        out = np.zeros((label_map.shape[0], label_map.shape[1], 3))

        for l in range(1, KITTI.num_classes()):
            mask = label_map == l
            out[mask, 0] = np.array(KITTI.classes[l].color[1])
            out[mask, 1] = np.array(KITTI.classes[l].color[0])
            out[mask, 2] = np.array(KITTI.classes[l].color[2])

        return out


    dataset = KITTI('../../data/kitti', transform=joint_transforms)
    distance, reflectivity, label = random.choice(dataset)

    print('Distance size: ', distance.size())
    print('Reflectivity size: ', reflectivity.size())
    print('Label size: ', label.size())

    distance_map = Image.fromarray((255 * _normalize(distance.numpy())).astype(np.uint8))
    reflectivity_map = Image.fromarray((255 * _normalize(reflectivity.numpy())).astype(np.uint8))
    label_map = Image.fromarray((255 * visualize_seg(label.numpy())).astype(np.uint8))

    blend_map = Image.blend(distance_map.convert('RGBA'), label_map.convert('RGBA'), alpha=0.4)

    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.title("Distance")
    plt.imshow(distance_map)
    plt.subplot(222)
    plt.title("Reflectivity")
    plt.imshow(reflectivity_map)
    plt.subplot(223)
    plt.title("Label")
    plt.imshow(label_map)
    plt.subplot(224)
    plt.title("Result")
    plt.imshow(blend_map)

    plt.show()

from collections import namedtuple

import torch

LidarClass = namedtuple('LidarClass', ['name', 'train_id', 'lidar_name', 'lidar_id',
                                       'color'])

Class = namedtuple('Class', ['lidar_name', 'lidar_id', 'color'])

classes = [
    Class('road', 19, (128, 64, 128)),
    Class('sidewalk', 20, (244, 35, 232)),
    Class('construction', 21, (70, 70, 70)),
    Class('construction', 21, (70, 70, 70)),
    Class('unlabeled', 0, (0, 0, 0)),
    Class('pole', 22, (153, 153, 153)),
    Class('construction', 21, (70, 70, 70)),
    Class('traffic sign', 23, (220, 220, 0)),
    Class('vegetation', 24, (104, 131, 15)),
    Class('terrain', 25, (148, 255, 144)),
    Class('sky', 26, (0, 0, 0)),
    Class('person', 27, (220, 20, 60)),
    Class('rider', 28, (255, 0, 0)),
    Class('small vehicle', 29, (0, 0, 142)),
    Class('large vehicle', 30, (0, 0, 70)),
    Class('large vehicle', 30, (0, 0, 70)),
    Class('large vehicle', 30, (0, 0, 70)),
    Class('two wheeler', 31, (119, 11, 32)),
    Class('two wheeler', 31, (119, 11, 32)),
]

lidar_classes = [
    LidarClass('road', 0, 'road', 19, (128, 64, 128)),
    LidarClass('sidewalk', 1, 'sidewalk', 20, (244, 35, 232)),
    LidarClass('building', 2, 'construction', 21, (70, 70, 70)),
    LidarClass('wall', 3, 'construction', 21, (70, 70, 70)),
    LidarClass('fence', 4, 'unlabeled', 0, (0, 0, 0)),
    LidarClass('pole', 5, 'pole', 22, (153, 153, 153)),
    LidarClass('traffic light', 6, 'construction', 21, (70, 70, 70)),
    LidarClass('traffic sign', 7, 'traffic sign', 23, (220, 220, 0)),
    LidarClass('vegetation', 8, 'vegetation', 24, (104, 131, 15)),
    LidarClass('terrain', 9, 'terrain', 25, (148, 255, 144)),
    LidarClass('sky', 10, 'sky', 26, (0, 0, 0)),
    LidarClass('person', 11, 'person', 27, (220, 20, 60)),
    LidarClass('rider', 12, 'rider', 28, (255, 0, 0)),
    LidarClass('car', 13, 'small vehicle', 29, (0, 0, 142)),
    LidarClass('truck', 14, 'large vehicle', 30, (0, 0, 70)),
    LidarClass('bus', 15, 'large vehicle', 30, (0, 0, 70)),
    LidarClass('train', 16, 'large vehicle', 30, (0, 0, 70)),
    LidarClass('motorcycle', 17, 'two wheeler', 31, (119, 11, 32)),
    LidarClass('bicycle', 18, 'two wheeler', 31, (119, 11, 32)),
]

train_id_to_lidar_id = {0: 19, 1: 20, 2: 21, 3: 21, 4: 0, 5: 22, 6: 21, 7: 23, 8: 24, 9: 25, 10: 26, 11: 27, 12: 28,
                        13: 29, 14: 30, 15: 30, 16: 30, 17: 31, 18: 31, 255: 0}


def convert_classes_to_lidar_classes(target):
    target_copy = target.clone()
    for cls in lidar_classes:
        target_copy[target == cls.train_id] = cls.lidar_id

    return target_copy


def get_lidar_colormap():
    cmap = torch.zeros([256, 3], dtype=torch.uint8)

    for cls in lidar_classes:
        cmap[cls.lidar_id, :] = torch.tensor(cls.color)

    return cmap

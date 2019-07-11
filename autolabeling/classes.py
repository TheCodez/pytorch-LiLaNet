from collections import namedtuple

import torch

LidarClass = namedtuple('LidarClass', ['name', 'train_id', 'lidar_name', 'lidar_id',
                                       'color'])
lidar_classes = [
    LidarClass('road', 0, 'road', 0, (128, 64, 128)),
    LidarClass('sidewalk', 1, 'sidewalk', 1, (244, 35, 232)),
    LidarClass('building', 2, 'construction', 2, (70, 70, 70)),
    LidarClass('wall', 3, 'construction', 2, (70, 70, 70)),
    LidarClass('fence', 4, 'unlabeled', 255, (70, 70, 70)),
    LidarClass('pole', 5, 'pole', 3, (153, 153, 153)),
    LidarClass('traffic light', 6, 'construction', 2, (70, 70, 70)),
    LidarClass('traffic sign', 7, 'traffic sign', 4, (220, 220, 0)),
    LidarClass('vegetation', 8, 'vegetation', 5, (104, 131, 15)),
    LidarClass('terrain', 9, 'terrain', 6, (148, 255, 144)),
    LidarClass('sky', 10, 'sky', 7, (0, 0, 0)),
    LidarClass('person', 11, 'person', 8, (220, 20, 60)),
    LidarClass('rider', 12, 'rider', 9, (255, 0, 0)),
    LidarClass('car', 13, 'small vehicle', 10, (0, 0, 142)),
    LidarClass('truck', 14, 'large vehicle', 11, (0, 0, 70)),
    LidarClass('bus', 15, 'large vehicle', 11, (0, 0, 70)),
    LidarClass('train', 16, 'large vehicle', 11, (0, 0, 70)),
    LidarClass('motorcycle', 17, 'two wheeler', 12, (119, 11, 32)),
    LidarClass('bicycle', 18, 'two wheeler', 12, (119, 11, 32)),
]


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

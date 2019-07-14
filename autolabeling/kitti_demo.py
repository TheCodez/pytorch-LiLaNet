import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pykitti
import torch
import torchvision.transforms.functional as F
from torch import hub
from torchvision.datasets import Cityscapes

from autolabeling import autolabel
from autolabeling.classes import get_lidar_colormap
from lilanet.utils import colorize_seg


def get_cityscapes_colormap():
    cmap = torch.zeros([256, 3], dtype=torch.uint8)

    for cls in Cityscapes.classes:
        cmap[cls.id, :] = torch.tensor(cls.color)

    return cmap


def convert_train_id_to_id(target):
    target_copy = target.clone()

    for cls in Cityscapes.classes:
        target_copy[target == cls.train_id] = cls.id

    return target_copy


def show_lidar_on_image(points, image, segmentation, T_cam0, K_cam0):
    points_2d = autolabel.pinhole_projection(points, T_cam0, K_cam0)

    cmap = get_cityscapes_colormap()
    segmentation = convert_train_id_to_id(segmentation)
    vis = colorize_seg(segmentation.cpu(), cmap)
    height, width = segmentation.shape

    for i in range(points.shape[0]):
        img_x = points_2d[i, 0]
        img_y = points_2d[i, 1]
        img_x = np.clip(img_x, 0, width - 1)
        img_y = np.clip(img_y, 0, height - 1)

        color = vis[:, img_y, img_x].tolist()
        cv2.circle(image, (img_x, img_y), 2, color=tuple(color), thickness=-1)

    return image


def show_lidar_depth_on_image(pc_velo, img, T_cam0, K_cam0):
    points_2d = autolabel.pinhole_projection(pc_velo, T_cam0, K_cam0)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(pc_velo.shape[0]):
        depth = np.sqrt(pc_velo[i, 0] ** 2 + pc_velo[i, 1] ** 2 + pc_velo[i, 2] ** 2)
        idx = np.clip(int(640.0 / depth), 0, 255)
        color = cmap[idx, :]
        img_x = points_2d[i, 0]
        img_y = points_2d[i, 1]
        cv2.circle(img, (img_x, img_y), 2, color=tuple(color), thickness=-1)

    return img


def plot_images(file_name, distance, reflectivity, label, segmentation, img, proj_img):
    cmap = get_lidar_colormap()
    cs_cmap = get_cityscapes_colormap()

    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    distance_map = F.to_pil_image(_normalize(distance.squeeze()))
    reflectivity_map = F.to_pil_image(_normalize(reflectivity.squeeze()))
    label_map = F.to_pil_image(colorize_seg(label.squeeze(), cmap).cpu())

    segmentation = convert_train_id_to_id(segmentation)
    segmentation_map = F.to_pil_image(colorize_seg(segmentation.squeeze(), cs_cmap).cpu())

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(231)
    plt.title("Camera Image")
    plt.imshow(img)

    plt.subplot(232)
    plt.title("Semantic Image")
    plt.imshow(segmentation_map)

    plt.subplot(233)
    plt.title("Semantic Transfer")
    plt.imshow(proj_img)

    plt.subplot(234)
    plt.title("Distance")
    plt.imshow(distance_map)

    plt.subplot(235)
    plt.title("Reflectivity")
    plt.imshow(reflectivity_map)

    plt.subplot(236)
    plt.title("Label")
    plt.imshow(label_map)

    plt.tight_layout()
    plt.show()
    fig.savefig(file_name, dpi=200)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    basedir = '../data/kitti_raw'
    date = '2011_09_26'
    drive = '0005'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = pykitti.raw(basedir, date, drive)
    idx = 16
    file_name = "{}_{}_{}.png".format(date, drive, os.path.basename(dataset.cam2_files[idx])[:-4])

    model = hub.load('TheCodez/pytorch-GoogLeNet-FCN', 'googlenet_fcn', pretrained='cityscapes')
    model = model.to(device)
    model.eval()

    img = dataset.get_cam2(idx)
    pc_velo = dataset.get_velo(idx)

    print("Inference")
    pred = autolabel.semantic_segmentation(model, img, device)

    pc_velo = autolabel.get_points_in_fov_90(pc_velo)

    print("Transferring labels")
    pc_labels = autolabel.transfer_labels(pc_velo, pred, dataset.calib.T_cam0_velo, dataset.calib.K_cam0)

    print("Spherical projection")
    lidar = autolabel.spherical_projection(pc_labels)

    proj_img = show_lidar_on_image(pc_velo, np.array(img), pred, dataset.calib.T_cam0_velo, dataset.calib.K_cam0)

    record = torch.as_tensor(lidar, dtype=torch.float32).permute(2, 0, 1).contiguous()
    reflectivity = record[3, :, :]
    distance = record[4, :, :]
    label = record[5, :, :]

    plot_images(file_name, distance, reflectivity, label, pred, img, proj_img)

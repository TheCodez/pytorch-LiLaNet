import numpy as np
import torch
from torchvision import transforms

from autolabeling.classes import convert_classes_to_lidar_classes


def get_points_in_fov_90(pc, clip_distance=0.0, max_distance=500):
    fov_inds = (pc[:, 0] > pc[:, 1]) & (pc[:, 0] > -pc[:, 1])
    fov_inds = fov_inds & (pc[:, 0] > clip_distance) & (pc[:, 0] < max_distance)
    return pc[fov_inds, :]


def pinhole_projection(points, img, T_cam0, K_cam0):
    points_2d = np.zeros((points.shape[0], 2), dtype=np.int32)
    height, width = img.shape
    points = np.dot(points, np.transpose(T_cam0))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # pinhole camera
    x_img = (np.dot(K_cam0[0, 0], x / z) + K_cam0[0, 2]).astype(np.int32)
    y_img = (np.dot(K_cam0[1, 1], y / z) + K_cam0[1, 2]).astype(np.int32)

    # clamp values
    x_img = np.clip(x_img, 0, width - 1)
    y_img = np.clip(y_img, 0, height - 1)

    points_2d[:, 0] = x_img
    points_2d[:, 1] = y_img

    return points_2d


def transfer_labels(points, semantic, T_cam0, K_cam0):
    labels = np.ones((points.shape[0], 1), dtype=points.dtype) * 255

    points_2d = pinhole_projection(points, semantic, T_cam0, K_cam0)
    semantic = convert_classes_to_lidar_classes(semantic)

    x_img = points_2d[:, 0]
    y_img = points_2d[:, 1]

    label = semantic[y_img, x_img]
    labels[:, 0] = label

    return np.concatenate((points, labels), axis=1)


def spherical_projection(points, height=64, width=512):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    reflectivity = points[:, 3]
    label = points[:, 4]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    theta = np.arcsin(z / d)
    phi = np.arctan2(y, x)

    idx_h = height - 1 - ((height - 1) * (theta - theta.min()) / (theta.max() - theta.min()))
    idx_w = width - 1 - ((width - 1) * (phi - phi.min()) / (phi.max() - phi.min()))

    idx_h = np.floor(idx_h).astype(np.int32)
    idx_w = np.floor(idx_w).astype(np.int32)

    projected_img = np.zeros((height, width, 6), dtype=np.float32)
    projected_img[idx_h, idx_w, 0] = x
    projected_img[idx_h, idx_w, 1] = y
    projected_img[idx_h, idx_w, 2] = z
    projected_img[idx_h, idx_w, 3] = d
    projected_img[idx_h, idx_w, 4] = reflectivity
    projected_img[idx_h, idx_w, 5] = label

    return projected_img


def semantic_segmentation(model, img, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        image = transform(img)
        image = image.unsqueeze(0)
        image = image.to(device)

        pred = model(image)
        pred = pred.argmax(dim=1)

    return pred.squeeze().cpu()

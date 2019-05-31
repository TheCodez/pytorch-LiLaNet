import torch


def visualize_seg(image, cmap):
    out = torch.zeros([3, image.size(0), image.size(1)], dtype=torch.uint8)

    for label in range(1, len(cmap)):
        mask = image == label

        out[0, mask] = cmap[label, 0]
        out[1, mask] = cmap[label, 1]
        out[2, mask] = cmap[label, 2]

    return out

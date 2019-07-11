import os
import tempfile

import torch


def colorize_seg(image, cmap):
    out = torch.zeros([3, image.size(0), image.size(1)], dtype=torch.uint8)

    for label in range(len(cmap)):
        mask = image == label

        out[0, mask] = cmap[label, 0]
        out[1, mask] = cmap[label, 1]
        out[2, mask] = cmap[label, 2]

    return out


def save(obj, dir, file_name):
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.expanduser(dir))

    try:
        torch.save(obj, tmp.file)
    except BaseException:
        tmp.close()
        os.remove(tmp.name)
        raise
    else:
        tmp.close()
        os.rename(tmp.name, os.path.join(dir, file_name))

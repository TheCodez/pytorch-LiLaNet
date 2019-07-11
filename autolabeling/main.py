import os
from argparse import ArgumentParser

import numpy as np
import pykitti
import torch
import torch.hub as hub

from autolabeling import autolabel


def run(args):
    torch.cuda.empty_cache()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = hub.load('TheCodez/pytorch-GoogLeNet-FCN', 'googlenet_fcn', pretrained='cityscapes')
    model = model.to(device)
    model.eval()

    for date in args.dates:
        for drive in args.drives:
            data = pykitti.raw(args.dir, date, drive)

            for i, image in enumerate(data.cam2):
                name = os.path.basename(data.cam2_files[i])[:-4]
                file_name = '{}_{}_{}.npy'.format(date, drive, name)
                pc_velo = data.get_velo(i)

                print("Processing: ", file_name)
                pred = autolabel.semantic_segmentation(model, image, device)
                pc_velo = autolabel.get_points_in_fov_90(pc_velo)
                pc_labels = autolabel.transfer_labels(pc_velo, pred, data.calib.T_cam0_velo, data.calib.K_cam0)
                record = autolabel.spherical_projection(pc_labels)

                os.makedirs(args.output_dir, exist_ok=True)
                np.save(os.path.join(args.output_dir, file_name), record)
                print("Saved: ", file_name)

    print("Processing finished")


if __name__ == '__main__':
    parser = ArgumentParser('Autolabeling')
    parser.add_argument('--dir', default='../data/kitti_raw',
                        help='directory of the raw dataset')
    parser.add_argument('--output-dir', default='../data/kitti_processed',
                        help='directory to save model checkpoints')
    parser.add_argument("--dates", nargs='+', default=['2011_09_26'],
                        help="the dates to use")
    parser.add_argument("--drives", nargs='+', default=['0048'],
                        help="the drives to use")

    run(parser.parse_args())

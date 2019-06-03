# pytorch-LiLa ![alt text](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

Inofficial PyTorch implementation of [Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation](https://arxiv.org/abs/1804.09915) (Piewak et al., 2018).

## Differences:

The Autolabeling process is currently not used, instead the converted KITTI data from [SqueezeSeg](https://github.com/BichenWuUCB/SqueezeSeg) is used.
For better convergence we add batch normalization after each convolutional layer.

## Results:

|            | Car      | Pedestrian | Cyclist  | mIoU     |
|:----------:|----------|------------|----------|----------|
| SqueezeSeg | **64.6** | 21.8       | 25.1     | 37.2     |
| LiLaNet    | 56.7     | **38.6**   | **26.1** | **40.5** |

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the [KITTI Lidar](https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz) dataset

## Usage

Train model:

**Important**: The ```dataset-dir``` must contain the ```lidar_2d``` and the ```ImageSet``` folder.

```bash
python train_kitti.py --dataset-dir 'data/kitti'
```

from pathlib import Path
import numpy as np
from osgeo import gdal
import rasterio
import math

import torch
from torch.utils.data import Dataset

from utils import make_tuple


def get_pair_path(directory):   #获取影像文件路径 ：从文件夹 directory 中匹配 4 张影像，并按顺序返回文件路径

    # 一组输入影像
    ref_label, pred_label = directory.name.split('_')   #将directory.name中以'_'为基准，划分为前后两部分。分别赋值给ref_label, pred_label

    FINE_PREFIX = 'G_'   #定义前缀 FINE_PREFIX（高分影像）和 COARSE_PREFIX（低分影像）。
    COARSE_PREFIX = 'L_'

    paths: list = [None] * 4

    def match(path: Path):
        return {
            COARSE_PREFIX + ref_label in path.stem: 0,
            FINE_PREFIX + ref_label in path.stem: 1,
            COARSE_PREFIX + pred_label in path.stem: 2,
            FINE_PREFIX + pred_label in path.stem: 3
        }

    for f in Path(directory).glob('*.tif'):
        paths[match(f)[True]] = f.absolute().resolve()
    return paths

def load_image_pair(directory):
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(directory)
    # 将组织好的数据转为Image对象
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            images.append(im)

    return images



class PatchSet(Dataset):
    def __init__(self, image_dir, image_size, patch_size, patch_stride=None):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)

        # 计算有效的 patch 数量
        self.n_patch_x = (self.image_size[0] - self.patch_size[0]) // self.patch_stride[0] + 1
        self.n_patch_y = (self.image_size[1] - self.patch_size[1]) // self.patch_stride[1] + 1
        self.num_patch = self.num_im_pairs * self.n_patch_x * self.n_patch_y


    @staticmethod
    # def transform(data):
    #     data[data < 0] = 0
    #     data = data.astype(np.float32)
    #     data = torch.from_numpy(data)
    #     out = data.mul_(0.0001)
    #     return out

    def transform(data):
        # 转 float32
        data = data.astype(np.float32)
        data[data < 0] = 0
        data = data / 255.0
        return torch.from_numpy(data)

    def map_index(self, index):
        id_n = index // (self.n_patch_x * self.n_patch_y)
        residual = index % (self.n_patch_x * self.n_patch_y)
        id_x = self.patch_stride[0] * (residual % self.n_patch_x)
        id_y = self.patch_stride[1] * (residual // self.n_patch_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n])
        patches = [None] * len(images)

        for i in range(len(patches)):
            im = images[i]
            # 确保切片不会超出图像边界
            if id_x + self.patch_size[0] <= im.shape[1] and id_y + self.patch_size[1] <= im.shape[2]:
                patch = im[:, id_x:id_x + self.patch_size[0], id_y:id_y + self.patch_size[1]]
                patches[i] = self.transform(patch)
            else:
                # 如果超出边界，返回 None，后续过滤
                return None

        del images[:]
        del images
        return patches

    def __len__(self):
        return self.num_patch

# 可选：添加过滤函数以移除无效 patch
def filter_dataset(dataset):
    valid_indices = []
    for i in range(len(dataset)):
        patches = dataset[i]
        if patches is not None and all(p.shape == (6, dataset.patch_size[0], dataset.patch_size[1]) for p in patches):
            valid_indices.append(i)
    return torch.utils.data.Subset(dataset, valid_indices)



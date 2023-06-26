# -*- coding: utf-8 -*-
"""
@Author  : Morvan Li
@FileName: images_mean_std.py
@Software: PyCharm
@Time    : 6/26/23 9:47 AM
"""

import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


# method 1
def images_mean_std_1(train_data: str) -> Tuple[np.array, np.array]:
    """
    @param train_data_path:
    """
    print("The first method calculate mean and std of images")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # if the image is color, else mean = torch.zeros(1)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for data, _ in train_loader:
        for c in range(3):
            mean[c] += data[:, c, :, :].mean()
            std[c] += data[:, c, :, :].std()

    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


# method 2

def images_mean_std_2(img_dir: str) -> Tuple[np.array, np.array]:
    img_channels = 3  # 图像通道数
    assert os.path.exists(img_dir), f"{img_dir} does not exist."  # 检查图像文件夹是否存在
    imgs_list = [os.path.join(root, name) for root, dirs, files in os.walk(img_dir) for name in files]  # 获取所有图像文件的路径列表
    cumulative_mean = np.zeros(img_channels)  # 初始化累积均值
    cumulative_std = np.zeros(img_channels)  # 初始化累积标准差
    for img_path in imgs_list:
        # 　img -> [H, W, C]
        img = np.array(Image.open(img_path)) / 255.  # 读取图像并将像素值缩放到[0, 1]范围
        cumulative_mean += img.mean(axis=(0, 1))  # 累积均值
        cumulative_std += img.std(axis=(0, 1))  # 累积标准差
    mean = cumulative_mean / len(imgs_list)  # 计算均值
    std = cumulative_std / len(imgs_list)  # 计算标准差

    return list(mean), list(std)


if __name__ == '__main__':
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = ImageFolder(root=os.path.join(image_path, "train"),
                                transform=data_transform["train"])

    print(images_mean_std_1(train_dataset))
    print(images_mean_std_2(os.path.join(image_path, "train")))

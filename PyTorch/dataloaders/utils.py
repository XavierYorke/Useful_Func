#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/3/29
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com

import os
import os.path as osp
import random
from monai import transforms


train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.ScaleIntensityRanged(
        keys=["image"], a_min=100, a_max=300,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    transforms.AddChanneld(keys=["image", "label"]),
    transforms.SpatialPadd(keys=["image", "label"],
                           spatial_size=[512, 512, 96],
                           method='symmetric'),
    transforms.RandCropByPosNegLabeld(keys=["image", "label"],
                                      label_key="label",
                                      spatial_size=[64, 64, 64],
                                      num_samples=2, pos=1, neg=0),
    transforms.EnsureTyped(keys=["image", "label"])
])

val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.ScaleIntensityRanged(
        keys=["image"], a_min=100, a_max=300,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    transforms.AddChanneld(keys=["image", "label"]),
    transforms.SpatialPadd(keys=["image", "label"],
                           spatial_size=[512, 512, 96],
                           method='symmetric'),
    transforms.RandCropByPosNegLabeld(keys=["image", "label"],
                                      label_key="label",
                                      spatial_size=[64, 64, 64],
                                      num_samples=2, pos=1, neg=0),
    transforms.EnsureTyped(keys=["image", "label"])
])


def split_ds(data_path, split):
    images_path = []
    labels_path = []

    for curr_path, sec_paths, _ in os.walk(data_path):
        # print(curr_path)
        for sec_path in sec_paths:
            sec_path = osp.join(curr_path, sec_path)
            # print(sec_path)
            for _, _, files_name in os.walk(sec_path):
                for file_name in files_name:
                    file_path = osp.join(sec_path, file_name)
                    if 'origin' in file_path:
                        images_path.append(file_path)
                    elif 'ias' in file_path:
                        labels_path.append(file_path)
    total_size = len(images_path)
    total_index = [i for i in range(total_size)]
    random.shuffle(total_index)

    train_index = total_index[:int(total_size * split)]
    val_index = total_index[int(total_size * split):]
    train_list = []
    val_list = []
    for i, (image, label) in enumerate(zip(images_path, labels_path)):
        if i in train_index:
            train_list.append((image, label))
        if i in val_index:
            val_list.append((image, label))
    train_dic = [{'image': image, 'label': label} for image, label in train_list]
    val_dic = [{'image': image, 'label': label} for image, label in val_list]

    return train_dic, val_dic


if __name__ == '__main__':
    path = r'your data path'
    train_dict, val_dict = split_ds(path, 0.8)
    print(len(train_dict))
    print(len(val_dict))

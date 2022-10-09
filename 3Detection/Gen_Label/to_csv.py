from pprint import pp


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 15:13:43
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
import matplotlib.patches as patch
import csv
from tqdm import tqdm

# 根据标签获取bbox
rgbmask = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint8)


# 从label图得到 boundingbox 和图上连通域数量 object_num
def getboundingbox(image):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((image.shape[0], image.shape[1],
                    image.shape[2]), dtype=np.uint8)
    mask = np.where(image > 0, 1, 0)

    # mask[np.where(np.all(image == 1, axis=-1))[:2]] = 1
    # 删掉小于10像素的目标
    mask_without_small = morphology.remove_small_objects(
        mask, min_size=10, connectivity=2)
    # 连通域标记
    label_image = measure.label(mask_without_small)
    # 统计object个数
    object_num = len(measure.regionprops(label_image))
    boundingbox = list()
    centroid = list()
    for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
        boundingbox.append(region.bbox)
        centroid.append(region.centroid)
    return object_num, boundingbox, centroid


def json_bbox(nodule, img_or, img_sp):
    x, y, z, d = nodule[0] * img_sp[0] + img_or[0], nodule[1] * img_sp[1] + \
        img_or[1], nodule[2] * img_sp[2] + img_or[2], nodule[3] * img_sp[0] * 2
    return (x, y, z, d)


if __name__ == '__main__':

    raw_path = r'D:/Dataset/lesion/'
    save_path = r'D:/Dataset/Det_lesion/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(raw_path)
    for file in tqdm(files):
        # print(file)
        mask_file = file + '_lesion.nii.gz'

        mask_path = os.path.join(raw_path, file, mask_file)
        mask_img = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_img)
        mask_array = np.transpose(mask_array, (1, 2, 0))

        # 获取bbox
        object_num, boundingbox, centroid = getboundingbox(mask_array)

        for box, center in zip(boundingbox, centroid):
            diameter = np.sqrt(
                (box[0] - center[0])**2 + (box[1] - center[1])**2 + (box[2] - center[2])**2)
            nodule = center + (diameter / 2,)
            OR = mask_img.GetOrigin()   # 从图像中读取原点,即体素的中心
            SP = mask_img.GetSpacing()  # 像素间距,即体素的大小
            nodule = json_bbox(nodule, OR, SP)
            # 将nodule写入csv文件
            with open('./lesion.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                row = (file, ) + nodule
                writer.writerow(row)
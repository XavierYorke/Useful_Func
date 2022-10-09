#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/26 17:17:33
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
import matplotlib.patches as patch


def json_bbox(nodule, img_or, img_sp):
    x, y, z, d = nodule[0] * img_sp[0] + img_or[0], nodule[1] * img_sp[1] + \
        img_or[1], nodule[2] * img_sp[2] + img_or[2], nodule[3] * img_sp[0] * 2

    return (x, y, z, d)


def json_test(nodule, img_or, img_sp):
    print('test nodule: ', nodule)
    x, y, z, d = nodule[0] * img_sp[0] + img_or[0], nodule[1] * img_sp[1] + \
        img_or[1], nodule[2] * img_sp[2] + img_or[2], nodule[3] * img_sp[0] * 2
    print('test :', x, y, z, d)
    x, y, z = int((x-img_or[0])/img_sp[0]), int(
        (y-img_or[1])/img_sp[1]), int((z-img_or[2])/img_sp[2])    # 将坐标转换为像素坐标
    d = int(d/img_sp[0]/2)  # 将直径转换为像素坐标

    print('test: ', x, y, z, d)


rgbmask = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint8)


# 从label图得到 boundingbox 和图上连通域数量 object_num
def getboundingbox(image):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((image.shape[0], image.shape[1],
                    image.shape[2]), dtype=np.uint8)
    mask[np.where(image == 1)] = 1

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


# radius是正方形边长一半，pad是边的宽度,max_show_num最大展示数
def show_nodules(ct_scan, nodules, Origin, Spacing, pad=2, max_show_num=4):

    show_index = []
    # lable是一个nx4维的数组，n是肺结节数目，4代表x,y,z,以及直径
    # for idx in range(nodules.shape[0]):
    #     if idx < max_show_num:
    # if abs(nodules[0]) + abs(nodules[1]) + abs(nodules[2]) + abs(nodules[3]) == 0:  # 如果没有坐标，则不显示
    #     continue
    # print('nodules: ', nodules)

    x, y, z = int((nodules[0]-Origin[0])/Spacing[0]), int(
        (nodules[1]-Origin[1])/Spacing[1]), int((nodules[2]-Origin[2])/Spacing[2])    # 将坐标转换为像素坐标
    #     x, y, z = int(nodules[0]), int(nodules[1]), int(nodules[2])
    # print(x, y, z)
    ct_scan = np.transpose(ct_scan, (2, 0, 1))
    data = ct_scan[z]
    print(data.shape)
    radius = int(nodules[3]/Spacing[0]/2)  # 将直径转换为像素坐标
    #     radius = int(nodules[3])
    # print(radius)
    #pad = 2*radius
    # 注意 y代表纵轴，x代表横轴
    data[x, y] = 0
    data[max(0, x - radius):min(data.shape[0], x + radius),
         max(0, y - radius - pad):max(0, y - radius)] = 3000  # 竖线

    data[max(0, x - radius):min(data.shape[0], x + radius),
         min(data.shape[1], y + radius):min(data.shape[1], y + radius + pad)] = 3000  # 竖线

    data[max(0, x - radius - pad):max(0, x - radius),
         max(0, y - radius):min(data.shape[1], y + radius)] = 3000  # 横线

    data[min(data.shape[0], x + radius):min(data.shape[0], x + radius + pad),
         max(0, y - radius):min(data.shape[1], y + radius)] = 3000  # 横线

    # if z in show_index:  # 检查是否有结节在同一张切片，如果有，只显示一张
    #     continue
    show_index.append(z)
    # plt.figure(idx)
    # plt.imshow(data, cmap='gray')
    return data


raw_path = 'D:/Dataset/lesion'
# raw_path = 'F:/Projects/Detection_Demo/datasets'
files = os.listdir(raw_path)
for file in files:
    orig_file = file + '_origin.nii.gz'
    orig_path = os.path.join(raw_path, file, orig_file)
    orig_img = sitk.ReadImage(orig_path)
    orig_array = sitk.GetArrayFromImage(orig_img)
    orig_array = np.transpose(orig_array, (1, 2, 0))

    mask_file = file + '_lesion.nii.gz'
    mask_path = os.path.join(raw_path, file, mask_file)
    mask_img = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_img)
    mask_array = np.transpose(mask_array, (1, 2, 0))
    object_num, boundingbox, centroid = getboundingbox(mask_array)
    # print(object_num)
    # print('boundingbox:', boundingbox)
    # print('centroid: ', centroid)

    for box, center in zip(boundingbox, centroid):
        diameter = np.sqrt((box[0] - center[0])**2 +
                           (box[1] - center[1])**2 + (box[2] - center[2])**2)
        fig = plt.figure(figsize=(15, 15))
        # 标签
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(mask_array[:, :, int((box[2] + box[5]) / 2)], cmap='gray')
        rect1 = patch.Rectangle((box[1], box[0]), box[4] - box[1],
                                box[3] - box[0], linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect1)
        ax1.set_title('mask')

        # 原图
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(orig_array[:, :, int((box[2] + box[5]) / 2)], cmap='gray')
        rect2 = patch.Rectangle((box[1], box[0]), box[4] - box[1],
                                box[3] - box[0], linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect2)
        ax2.set_title('orig')

        # 逆转换
        ax3 = fig.add_subplot(1, 3, 3)
        nodule = center + (diameter / 2,)
        # print(nodule)

        OR = mask_img.GetOrigin()   # 从图像中读取原点,即体素的中心
        print('or: ', OR)
        SP = mask_img.GetSpacing()  # 像素间距,即体素的大小
        print('sp: ', SP)
        json_test(nodule, OR, SP)
        nodule = json_bbox(nodule, OR, SP)
        data = show_nodules(orig_array, nodule, OR, SP)
        # data = show_nodules(mask_array, nodule, OR, SP)
        data[data == 1] = 2000
        ax3.imshow(data, cmap='gray')

    plt.show()
    break

# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 09:55:55
@Author  :   XavierYorke
@Contact :   mzlxavier1230@gmail.com
'''

import csv
import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from skimage import morphology, measure


# def nii2csv_single(mhd_path, seriesuid, csv_path=None, expand_number=2):
#     f = open(csv_path, 'wb', encoding='utf-8', newline='')
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])
#     img = sitk.ReadImage(mhd_path)
#     data = sitk.GetArrayFromImage(img)  # (z, y, x)
#     data = np.transpose(data, axes=(2, 1, 0))
#     data[data > 0] = 1
#     offArr, eleArr = img.GetOrigin(), img.GetSpacing()
#     # 去除其中小于10个像素的目标，防止噪声的干扰
#     label = morphology.remove_small_objects(data, min_size=10, connectivity=2)
#     label = measure.label(label > 0)
#     for region in measure.regionprops(label):
#         bbox, centroid = region.bbox, region.centroid
#         r_x = int((bbox[3] - bbox[0] + expand_number) / 2) * eleArr[0]
#         r_y = int((bbox[4] - bbox[1] + expand_number) / 2) * eleArr[1]
#         r_z = int((bbox[5] - bbox[2] + expand_number) / 2) * eleArr[2]
#         r = np.max([r_x, r_y, r_z])
#
#         # 计算中心点坐标信息, 并进行处理以恢复到世界坐标系
#         x = int((bbox[3] + bbox[0]) / 2) * eleArr[0] + offArr[0]
#         y = int((bbox[4] + bbox[1]) / 2) ** eleArr[1] + offArr[1]
#         z = int((bbox[5] + bbox[2]) / 2) * eleArr[2] + offArr[2]
#         # 将获得的信息写如到csv文件中
#         csv_writer.writerow([str(seriesuid), str(x), str(y), str(z), str(r)])
#     f.close()


def nii2csv_info(nii_path, seriesuid, expand_number=2):
    out_list = []
    img = sitk.ReadImage(nii_path)
    data = sitk.GetArrayFromImage(img)  # (z, y, x)
    data = np.transpose(data, axes=(2, 1, 0))   # (x, y, z)
    data[data > 0] = 1
    offArr, eleArr = img.GetOrigin(), img.GetSpacing()
    # print(offArr, eleArr)
    # 去除其中小于10个像素的目标，防止噪声的干扰
    label = morphology.remove_small_objects(data, min_size=4, connectivity=2)
    label = measure.label(label > 0)
    for region in measure.regionprops(label):
        bbox, centroid = region.bbox, region.centroid
        r_x = int((bbox[3] - bbox[0] + expand_number) / 2) * eleArr[0]
        r_y = int((bbox[4] - bbox[1] + expand_number) / 2) * eleArr[1]
        r_z = int((bbox[5] - bbox[2] + expand_number) / 2) * eleArr[2]
        r = np.max([r_x, r_y, r_z])

        # 计算中心点坐标信息, 并进行处理以恢复到世界坐标系
        x = int((bbox[3] + bbox[0]) / 2) * eleArr[0] + offArr[0]
        y = int((bbox[4] + bbox[1]) / 2) * eleArr[1] + offArr[1]
        z = int((bbox[5] + bbox[2]) / 2) * eleArr[2] + offArr[2]
        # 将获得的信息写如到csv文件中
        # print([str(seriesuid), str(x), str(y), str(z), str(r)])
        out_list.append([str(seriesuid), str(x), str(y), str(z), str(r)])
    return out_list


if __name__ == '__main__':
    raw_path = r'D:/Datasets/lungdata'
    csv_path = 'lesion_full.csv'

    fid = open(csv_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(fid)
    # csv_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])

    files = sorted(os.listdir(raw_path))

    for step, f in tqdm(enumerate(files, 1)):
        origin_path = os.path.join(raw_path, f, str(f) + '_origin.nii.gz')
        label_path = os.path.join(raw_path, f, str(f) + '_lesion.nii.gz')
        if os.path.exists(origin_path) and os.path.exists(label_path):
            csv_info = nii2csv_info(label_path, f, expand_number=2)
            for index in range(len(csv_info)):
                info = csv_info[index]
                csv_writer.writerow(info)
    fid.close()


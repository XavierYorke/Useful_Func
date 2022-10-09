#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 17:23:04
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy as np
from collections import Counter
flag = False

for root, dirs, files, in tqdm(os.walk("D:/Dataset/lesion/")):  # 遍历文件夹
    for file in files:
        basename = os.path.basename(root)
        ct_array = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(root, basename + '_origin.nii.gz')))  # 原始图像
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(root, basename + '_lesion.nii.gz')))  # 标签图像
        # seg_bg_mask = seg_array == 0 #背景 背景占的内存大 很难统计 就直接注释了
        seg_pulmonary_artery_mask = seg_array == 1  # 目标前景 我之前的血管标签为1

        #ct_bg = ct_array[np.where(seg_bg_mask > 0)]
        ct_pulmonary_artery = ct_array[np.where(
            seg_pulmonary_artery_mask > 0)]  # 根据标签位置直接在原图上截

        #ct_bg = np.float32(ct_bg)
        ct_pulmonary_artery = np.float32(ct_pulmonary_artery)
        if flag == False:
            #bg = ct_bg
            artery = ct_pulmonary_artery
            flag = True

        else:
            #bg = np.concatenate((bg,ct_bg),axis=0)

            artery = np.concatenate(
                (artery, ct_pulmonary_artery), axis=0)  # 一例例 数据保存就行


# print("背景最大值",bg.max())
# print("背景最小值",bg.min())
# print("背景方差",bg.var())

#bg  = bg.astype(np.int16)
print("前景最大值", artery.max())
print("前景最小值", artery.min())
print("前景方差", artery.var())


sns.distplot(artery, color='red', label="artery")
# sns.distplot(artery,color="r",label="pulmonary_artery",norm_hist=False,hist=True,kde=True)
plt.legend()
plt.savefig("lesion.png")
plt.show()


# artery = Counter(artery)
# print(artery)


# 施工中 保存为excel的方式 进行查看
# a_pd = pd.DataFrame(artery)
# # create writer to write an excel file
# writer = pd.ExcelWriter('artery.xlsx')
# # write in ro file, 'sheet1' is the page title, float_format is the accuracy of data
# a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
# # save file
# writer.save()
# # close writer
# writer.close()

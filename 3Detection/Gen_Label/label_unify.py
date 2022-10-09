#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 14:51:39
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import os
import os.path as osp
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# 将label中大于0的标记统一为1并保存
def label_unify(label_path):
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)
    label_array = np.where(label_array > 0, 1, 0)
    label = sitk.GetImageFromArray(label_array)
    sitk.WriteImage(label, label_path)


if __name__ == '__main__':
    root = r'D:/Dataset/lungdata'
    folders = os.listdir(root)
    for folder in tqdm(folders):
        label_unify(osp.join(root, folder, folder + '_airway.nii.gz'))
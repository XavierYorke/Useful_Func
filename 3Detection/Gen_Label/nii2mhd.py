#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/09/24 09:55:55
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import os
import SimpleITK as sitk
from tqdm import tqdm


# nii文件转mhd文件
def nii2mhd(file_name, raw_root, save_root):
    file_root = os.path.join(save_root, file_name.split('_orig')[0])
    if not os.path.exists(file_root):
        os.makedirs(file_root)

    file_path = os.path.join(raw_root, file_name.split('_orig')[0], file_name)
    reader = sitk.ReadImage(file_path)
    PixelSpacing = reader.GetSpacing()
    Origin = reader.GetOrigin()

    #Origin = referencect.ImagePositionPatient
    img = sitk.GetArrayFromImage(reader)  # z y x
    img2 = sitk.GetImageFromArray(img)  # z y x
    img2.SetSpacing(PixelSpacing)
    img2.SetOrigin(Origin)
    file_path = os.path.join(file_root, file_name.split('_orig')[0] + '.mhd')
    sitk.WriteImage(img2, file_path)


if __name__ == '__main__':

    raw_path = r'D:/Dataset/lesion/'
    save_path = r'D:/Dataset/Det_lesion/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(raw_path)
    for file in tqdm(files):
        orig_file = file + '_origin.nii.gz'
        orig_path = os.path.join(raw_path, file, orig_file)
        orig_img = sitk.ReadImage(orig_path)
        orig_array = sitk.GetArrayFromImage(orig_img)

        # nii2mhd
        nii2mhd(orig_file, raw_path, save_path)

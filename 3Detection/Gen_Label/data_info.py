import os
import SimpleITK as sitk
import csv
from tqdm import tqdm


# 查看nii文件
def get_img(file_path):
    ct_scan = sitk.ReadImage(file_path)
    return ct_scan


root = r'D:/Datasets/lungdata'
file_list = os.listdir(root)
csv_path = 'saved/lesion_info.csv'
fid = open(csv_path, 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(fid)
for file in tqdm(file_list):
    image_path = os.path.join(root, file, file + '_origin.nii.gz')
    label_path = os.path.join(root, file, file + '_lesion.nii.gz')
    if os.path.exists(image_path) and os.path.exists(label_path):
        image = get_img(image_path)
        label = get_img(label_path)
        label_array = sitk.GetArrayFromImage(label)
        # print(label.GetSize())
        info = [i for i in label.GetSize()]
        info = [file] + info
        csv_writer.writerow(info)
fid.close()

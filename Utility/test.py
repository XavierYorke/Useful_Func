import os
from glob import glob

root = r'D:/Dataset/lungdata'

folders = os.listdir(root)


train_files = [glob(os.path.join(root, folder, folder + '_origin.nii.gz')) for folder in folders]


print(train_files)
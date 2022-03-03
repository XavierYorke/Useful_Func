#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/03/04 00:54:13
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''


from torch.utils.data import Dataset
from PIL import Image


# Defines the format in which the file is read
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):  
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()  
        fh = open(txt, 'r')  
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()  
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = self.loader(img)  
        label = self.loader(label)
        if self.transform is not None:
            img = self.transform(img)  
            label = self.target_transform(label)
        return img, label  

    def __len__(self):
        return len(self.imgs)

    def get_path(self, index):
        img, label = self.imgs[index]
        return img, label


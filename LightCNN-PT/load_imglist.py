# coding:utf-8
# !/usr/bin/python3

import os
import os.path
from PIL import Image
import torch.utils.data as data


################################################################################
def default_loader(path):
    return Image.open(path).convert('L')


def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


################################################################################
class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None,
                 list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        if self.transform is not None: img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)


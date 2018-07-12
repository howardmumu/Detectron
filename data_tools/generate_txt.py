import numpy as np
from matplotlib import pyplot as plt
import cv2
import shutil


def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if len(FileNames) > 0:
        for fn in FileNames:
            if len(FlagStr) > 0:
                if IsSubString(FlagStr, fn):
                    fullfilename = os.path.join(FindPath, fn)
                    FileList.append(fullfilename)
            else:
                fullfilename = os.path.join(FindPath, fn)
                FileList.append(fullfilename)

    if len(FileList) > 0:
        FileList.sort()

    return FileList


def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


with open('/media/shuhao/harddisk1/data/images/youshang/train.txt', 'w') as txt:
    imgfile = GetFileList('/media/shuhao/harddisk1/data/images/youshang/train/1')
    for img in imgfile:
        img = img.split('/')[-3:]
        img = '/'.join(img)
        str = img + ' ' + '1' + '\n'
        txt.writelines(str)

    imgfile = GetFileList('/media/shuhao/harddisk1/data/images/youshang/train/0')
    for img in imgfile:
        img = img.split('/')[-3:]
        img = '/'.join(img)
        str = img + ' ' + '0' + '\n'
        txt.writelines(str)

with open('/media/shuhao/harddisk1/data/images/youshang/test.txt', 'w') as txt:
    imgfile = GetFileList('/media/shuhao/harddisk1/data/images/youshang/test/1')
    for img in imgfile:
        img = img.split('/')[-3:]
        img = '/'.join(img)
        str = img + ' ' + '1' + '\n'
        txt.writelines(str)

    imgfile = GetFileList('/media/shuhao/harddisk1/data/images/youshang/test/0')
    for img in imgfile:
        img = img.split('/')[-3:]
        img = '/'.join(img)
        str = img + ' ' + '0' + '\n'
        txt.writelines(str)

import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
path = "E:\\dataBase\\end_data\\img\\"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
all=0
for filename in files:
    filePath.append(path + filename)
for fileEverPath in filePath:
    tree = ET.parse(fileEverPath)
    root = tree.getroot()
    path_text = root.find('filename').text
    if fileEverPath.find(' ')!=-1:
        all+=1
        new_name=fileEverPath.replace(' ','')
        os.rename(fileEverPath,new_name)
        # root.find('filename').text=path_text.replace(' ','')
        # tree.write(fileEverPath)
        # end_name=fileEverPath.replace(' ','')
        #
        # print(end_name)
print(all)
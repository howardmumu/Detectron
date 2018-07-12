import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
path = "E:\\dataBase\\end_data\\xml\\"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
all=0
for filename in files:
    filePath.append(path + filename)
for fileEverPath in filePath:
    try:
        tree = ET.parse(fileEverPath)
        root = tree.getroot()
        path_text = root.find('path').text
        if path_text.find(' ') != -1:
            all += 1
            new_name = path_text.replace(' ', '')
            root.find('path').text = new_name
            tree.write(fileEverPath)
            # root.find('filename').text=path_text.replace(' ','')
            # tree.write(fileEverPath)
            # end_name=fileEverPath.replace(' ','')
            #
            # print(end_name)
    except:
        print(fileEverPath)
print(all)
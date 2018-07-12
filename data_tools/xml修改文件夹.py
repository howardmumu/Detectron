import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
path = "E:\\dataBase\\end_data\\xml\\"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
for filename in files:
    filePath.append(path + filename)
for fileEverPath in filePath:
    # try:
        tree = ET.parse(fileEverPath)
        root = tree.getroot()
        root.find('folder').text='imge'
        tree.write(fileEverPath)
    # except:
    #     print(fileEverPath)
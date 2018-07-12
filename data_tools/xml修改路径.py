import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
path = "/media/shuhao/harddisk1/data/0611/xml/Changtao/"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
for filename in files:
    filePath.append(path + filename)
for fileEverPath in filePath:
    tree = ET.parse(fileEverPath)
    root = tree.getroot()
    path_text = root.find('path').text
    all_part=path_text.split('\\')
    root.find('path').text="/media/shuhao/harddisk1/data/0611/images/changtao/"+all_part[-1]
    tree.write(fileEverPath)
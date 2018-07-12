import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
path = "E:\\dataBase\\cd\\xml"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
label_map_path = b'E:\\dataBase\\cd\\cd_label_map.pbtxt'
label_map_dict = label_map_util.get_label_map_dict(label_map_path)
for filename in files:
    filePath.append(path +'\\'+ filename)
for fileEverPath in filePath:
    tree = ET.parse(fileEverPath)
    root = tree.getroot()
    img_name = root.find('filename').text
    objects = root.findall('object')
    for object in objects:
        obj_name = object.find('name').text
        if obj_name in label_map_dict:
            i=0
        else:
            if len(object)<=1:
                os.remove(fileEverPath)
            else:
                root.remove(object)
                print(fileEverPath)
                tree.write(fileEverPath)
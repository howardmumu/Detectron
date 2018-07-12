import os
import cv2

import xml.etree.ElementTree as ET
path = r"D:\voc\VOCdevkit\VOC2018\Annotations\\"    # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
all_num=0
for filename in files:
    filePath.append(path +'\\'+ filename)
for fileEverPath in filePath:
    try:
        tree = ET.parse(fileEverPath)
        root = tree.getroot()
        img_name = root.find('filename').text
        objects = root.findall('object')
        if objects == []:
            os.remove(fileEverPath)
            print('delete')
            all_num += 1
    except :
        os.remove(fileEverPath)
        print(fileEverPath)
print(all_num)



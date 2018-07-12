import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random
path = r"E:\dataBase\end_data\lr\xml" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
filePath = []
xmins = []
xmaxs = []
ymins = []
ymaxs = []
how_many=1
for filename in files:
    filePath.append(path+'\\'+filename)
for fileEverPath in filePath:
    tree = ET.parse(fileEverPath)
    root = tree.getroot()
    size = root.find('size')
    how_many = 1
    height = int(size.find('height').text)  # Image height
    width = int(size.find('width').text)  # Image width
    img_path = root.find('path').text
    img_name = root.find('filename').text
    if ((img_name).find('.png') >= 0):
        img_name = img_name[0:-4]
    img_save_path = 'E:\dataBase\end_data\lr\out\\'
    objects = root.findall('object')
    for object in objects:
        box = object.find('bndbox')
        class_txt = object.find('name').text
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        re=[xmin,ymin,xmax,ymax]
        cv2.rectangle(img, (re[0], re[1]), (re[2], re[3]), (0, 0, 255), 2)
        cv2.imwrite(img_save_path + img_name +'_'+class_txt+'_'+str(how_many)+ '.png', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        how_many+=1
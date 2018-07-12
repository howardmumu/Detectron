import os
import cv2

import xml.etree.ElementTree as ET
path = "/media/shuhao/harddisk1/data/annotations/advertise/xml/test"
files = os.listdir(path)
filePath = []
all_num=0
for filename in files:
    filePath.append(path +'/'+ filename)
for fileEverPath in filePath:
    tree = ET.parse(fileEverPath)
    root = tree.getroot()
    img_name = root.find('filename').text
    objects = root.findall('object')
    for object in objects:
        obj_name = object.find('name').text
        if obj_name=="\\":
            all_num+=1
            print(fileEverPath)
            print(len(objects))
            # os.remove(fileEverPath)
            if len(object)<=1:
                            os.remove(fileEverPath)
                            print('delete file '+fileEverPath)
            else:
                            root.remove(object)
                            print('delete object '+fileEverPath)
                            tree.write(fileEverPath)

print(all_num)



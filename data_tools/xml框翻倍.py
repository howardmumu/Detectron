import os
import numpy as np
import xml.etree.ElementTree as ET
path = r"D:\voc\VOCdevkit\VOC2018\529\0529slc\2xml\\"
files = os.listdir(path)
num=0
widthset=set()
for xml in files:
    xml_path=path+xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('size')
    for object in objects:
        width = int(object.find('width').text)*2
        height = int(object.find('height').text)*2

        object.find('width').text=str(width)
        object.find('height').text = str(height)
        tree.write(xml_path)
    objects2 = root.findall('object')
    for object in objects2:
        box = object.find('bndbox')
        xmin = int(box.find('xmin').text)*2
        ymin = int(box.find('ymin').text)*2
        xmax = int(box.find('xmax').text) * 2
        ymax = int(box.find('ymax').text) * 2
        box.find('xmin').text=str(xmin)
        box.find('ymin').text = str(ymin)
        box.find('xmax').text = str(xmax)
        box.find('ymax').text = str(ymax)
        tree.write(xml_path)
#     for object in objects:
#         object_name = object.find('name').text
#
#             # print(xml_path)
# #             tree.write(xml_path)
# # print(num)
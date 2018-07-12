import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
path = "/media/shuhao/harddisk1/data/end_data/xml"
files = [x for x in os.listdir(path) if x.endswith('.xml')]
filePath = []
objects_set=set()
for filename in files:
    filePath.append(path +'/'+ filename)
for fileEverPath in filePath:
    try:
        tree = ET.parse(fileEverPath)
        root = tree.getroot()
        img_name = root.find('filename').text
        objects = root.findall('object')
        for object in objects:
            obj_name = object.find('name').text
            objects_set.add(obj_name)
    except Exception as err:
        os.remove(fileEverPath)
        print(fileEverPath+'deleted')

print(objects_set)
print(len(objects_set))
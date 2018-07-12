import os
import numpy as np
import xml.etree.ElementTree as ET
path = r"D:\voc\VOCdevkit\VOC2018\Annotations\\"
files = os.listdir(path)
num=0
for xml in files:
    xml_path=path+xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('size')
    for object in objects:
        width = object.find('width').text
        height = object.find('height').text
        if width=='0' or height=='0':
            os.remove(xml_path)
            num+=1
print(num)
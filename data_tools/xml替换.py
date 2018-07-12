import os
import numpy as np
import xml.etree.ElementTree as ET
path = r"/media/shuhao/harddisk1/data/Annotations/train/"

files = os.listdir(path)
num=0
for xml in files:
    xml_path=path+xml
    print(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    for object in objects:
        object_name = object.find('name').text
        if object_name=='\\':
            object.find('name').text='ggp'
            num += 1
            print(xml_path)
            tree.write(xml_path)
print(num)
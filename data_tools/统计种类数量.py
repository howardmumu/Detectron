import os
import cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import json

json_path = "/media/shuhao/harddisk1/data/annotations/trash/trash_train_172589.json"
with open(json_path) as fp:
    coco = json.load(fp)
print('read {} annotations'.format(len(coco['annotations'])))

anns = coco['annotations']
count = dict()
cat_map = dict()
for ind, cat in enumerate(coco['categories']):
    cat_map[ind+1] = cat['name']
    count[ind+1] = 0
for ann in anns:
    t = ann['category_id']
    count[t] += 1
print('--------------------------------------')
for key in count:
    print('category {} has {} boxes'.format(cat_map[key], count[key]))




# for fileEverPath in filePath:
#     tree = ET.parse(fileEverPath)
#     root = tree.getroot()
#     img_name = root.find('filename').text
#     objects = root.findall('object')
#     for object in objects:
#         obj_name = object.find('name').text
#         objects_set.add(obj_name)

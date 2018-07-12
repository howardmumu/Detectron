import os
import json

json_file = '/media/shuhao/harddisk1/data/annotations/general/general_train_1489314.json'
with open(json_file) as f:
    coco = json.load(f)


anns = coco['annotations']
print 'loaded {} annotations'.format(len(anns))
cats = dict()

ALL_CATS = dict()
for item in coco['categories']:
    ALL_CATS[item['id']] = item['name']

for ann in anns:
    ann_cat = ann['category_id']
    if ann_cat in cats:
        cats[ann_cat] += 1
    else:
        cats[ann_cat] = 1

for id in cats:
    print 'category {} has {} boxes'.format(ALL_CATS[id], cats[id])

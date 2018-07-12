import json
import os

json_path = '/media/shuhao/harddisk1/data/annotations/general/78-19clasess.json'
txt_path = '/media/shuhao/harddisk1/data/annotations/general/general_train_1489314.txt'

with open(json_path) as f:
    coco = json.load(f)

image_list = coco['images']
anns = coco['annotations']

img_dict = {}

for img in image_list:
    id = img['id']
    img_dict[id] = {'file_name': img['file_name'],
                    'height': img['height'],
                    'width': img['width'],
                    'bboxes': [],
                    'labels': []}

for ann in anns:
    id = ann['image_id']
    if ann['bbox'][2] <= 1 or ann['bbox'][3] <= 1:
        continue
    img_dict[id]['bboxes'].append(ann['bbox'])
    img_dict[id]['labels'].append(ann['category_id'])

file_index = 0
with open(txt_path, 'w') as f:
    for id in img_dict:
        item = img_dict[id]
        assert len(item['bboxes']) == len(item['labels'])
        if len(item['bboxes']) == 0:
            continue

        import numpy as np
        bboxes = np.array(item['bboxes'])
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] - 1
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] - 1
        labels = item['labels']
        num_b = len(bboxes)
        f.write('# {}\n{}\n3\n{}\n{}\n{}\n'.format(file_index, item['file_name'], item['height'], item['width'], num_b))
        for i in range(num_b):
            str_bbox = [str(e) for e in bboxes[i]]
            f.write('{} 0 0 '.format(labels[i]) + ' '.join(str_bbox) + '\n')
        f.write('0\n')

        file_index += 1



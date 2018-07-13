# coding=utf-8
#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import os
import time
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# model files
# cfg_file = '/home/shuhao/Documents/train_configs/e2e_faster_rcnn_R-101-FPN_2x.yaml'
cfg_file = '/home/shuhao/Documents/train_configs/trash_e2e_faster_rcnn_R-101-FPN.yaml'
# model_file = '/media/shuhao/harddisk1/model/general_frcnn_res101_001/train/general_train/generalized_rcnn/model_final.pkl'
model_file = '/media/shuhao/harddisk1/model/trash_frcnn_res101_001/train/trash_train/generalized_rcnn/model_iter174999.pkl'

# image files
image_file_or_folder = r'/media/shuhao/harddisk1/data/images/evalimg_mini'
output_dir = '/media/shuhao/harddisk1/data/images/eval_output_trash_101_mini'

workspace.GlobalInit(['caffe2', '--caffe2_log_level=3'])

merge_cfg_from_file(cfg_file)
cfg.NUM_GPUS = 1
weights = cache_url(model_file, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)
t = time.time()
model = infer_engine.initialize_model_from_cfg(weights)
print('load model took {} seconds'.format(time.time() - t))
dummy_coco_dataset = dummy_datasets.get_trash_dataset()


def generate_crops(im, crop_num):
    assert len(crop_num)== 2
    h_crop = crop_num[0]
    w_crop = crop_num[1]
    height, width = im.shape[0:2]
    patch_height = int(round(height / h_crop))
    patch_width = int(round(width / w_crop))
    im = np.array(im)

    im_list = []
    for h in range(h_crop):
        for w in range(w_crop):
            if h == h_crop-1 and w == w_crop-1:
                patch = im[h * patch_height:, w * patch_width:, :]
            elif h == h_crop-1:
                patch = im[h * patch_height:,w * patch_width : (w+1) * patch_width, :]
            elif w == w_crop-1:
                patch = im[h * patch_height : (h+1) * patch_height, w * patch_width:, :]
            else:
                patch = im[h * patch_height : (h+1) * patch_height, w * patch_width : (w+1) * patch_width, :]
            origin = (h * patch_height, w * patch_width)
            im_list.append((origin, patch))

    return im_list


def vis_one_image_opencv(
        im, boxes, classes, thresh=0.9,
        show_box=False, dataset=None, show_class=False):
    """Constructs a numpy array with the detections visualized."""
    boxes = np.array(boxes)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return None

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        if show_box:
            im = vis_utils.vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show class (off by default)
        if show_class:
            class_str = vis_utils.get_class_string(classes[i], score, dataset)
            im = vis_utils.vis_class(im, (bbox[0], bbox[1] - 2), class_str)

    return im


def detect(im_name):
    im = cv2.imread(im_name)
    im_list = generate_crops(im, (2, 3))
    timers = defaultdict(Timer)
    final_boxes = []
    final_classes = []

    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )

    with c2_utils.NamedCudaScope(0):
        t = time.time()
        for origin, patch in im_list:

            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, patch, None, timers=timers
            )
            if isinstance(cls_boxes, list):
                boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
                    cls_boxes, cls_segms, cls_keyps)
            if boxes is None:
                continue
            boxes = np.array(boxes)
            boxes[:, 0] += origin[1]
            boxes[:, 1] += origin[0]
            boxes[:, 2] += origin[1]
            boxes[:, 3] += origin[0]
            final_boxes.extend(boxes)
            final_classes.extend(classes)
        print('took {} seconds.'.format(time.time() - t))

    result_im = vis_one_image_opencv(
        im,  # BGR -> RGB for visualization
        final_boxes, final_classes,
        dataset=dummy_coco_dataset,
        show_box=True,
        show_class=True,
        thresh=0.7,
    )
    return result_im


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.isdir(image_file_or_folder):
        im_list = glob.iglob(image_file_or_folder + '/*.jpg')
    else:
        im_list = [image_file_or_folder]
    for i, im_name in enumerate(im_list):
        im = detect(im_name)
        if im is None:
            continue
        file_name = os.path.basename(im_name)
        cv2.imwrite(os.path.join(output_dir, file_name), im)


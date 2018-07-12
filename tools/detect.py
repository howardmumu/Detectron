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

cfg_file = '/home/shuhao/Documents/train_configs/trash_4gpu_e2e_faster_rcnn_R-50-FPN.yaml'
model_file = '/media/shuhao/harddisk1/model/trash_frcnn_res50_001/model_final.pkl'
image_file_or_folder = '/home/shuhao/Pictures/test/6289.jpg'

workspace.GlobalInit(['caffe2', '--caffe2_log_level=3'])

merge_cfg_from_file(cfg_file)
cfg.NUM_GPUS = 1
weights = cache_url(model_file, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)

model = infer_engine.initialize_model_from_cfg(weights)
dummy_coco_dataset = dummy_datasets.get_trash_dataset()


def vis_one_image_opencv(
        im, boxes, classes, thresh=0.9,
        show_box=False, dataset=None, show_class=False):
    """Constructs a numpy array with the detections visualized."""
    boxes = np.array(boxes)
    print(boxes)
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


def detect():
    if os.path.isdir(image_file_or_folder):
        im_list = glob.iglob(image_file_or_folder + '/*.jpg')
    else:
        im_list = [image_file_or_folder]

    for i, im_name in enumerate(im_list):
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )

        return vis_utils.vis_one_image_opencv(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            show_box=True,
            show_class=True,
            thresh=0.5,
            kp_thresh=2
        )


def detect_image(img):
    im = img.copy()
    timers = defaultdict(Timer)
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )

    if isinstance(cls_boxes, list):
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)
    else:
        boxes = None
        classes = None

    if boxes is None:
        return img
    ret = vis_one_image_opencv(
        im,  # BGR -> RGB for visualization
        boxes,
        classes,
        dataset=dummy_coco_dataset,
        show_box=True,
        show_class=True,
        thresh=0.7,
    )
    if ret is None:
        return img
    else:
        return ret


if __name__ == '__main__':
    vidcap = cv2.VideoCapture(0)
    while 1:
        success, image = vidcap.read()
        # Display the resulting frame
        cv2.imshow('frame', detect_image(image))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



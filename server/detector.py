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

cfg_file = '/home/shuhao/Documents/train_configs/general_e2e_faster_rcnn_R-101-FPN_002.yaml'
model_file = '/media/shuhao/harddisk1/model/general_frcnn_res101_002/train/general_train/generalized_rcnn/model_final.pkl'

workspace.GlobalInit(['caffe2', '--caffe2_log_level=3'])

merge_cfg_from_file(cfg_file)
cfg.NUM_GPUS = 1
weights = cache_url(model_file, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)

model = infer_engine.initialize_model_from_cfg(weights)
dummy_coco_dataset = dummy_datasets.get_general_dataset()

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
        return None
    else:
        result = np.zeros((len(boxes), 6))
        for i, box in enumerate(boxes):
            result[i, :5] = box
            result[i, 5] = classes[i]
        return result.tostring()



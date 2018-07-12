import numpy as np
from PIL import Image
from PIL import ImageOps
import time
import sys
import cv2

import random


# DATA AUMENTATION

def random_crop(im, crop_margin, crop_w, crop_h):
    # Crops a random region of the image that will be used for training. Margin won't be included in crop.
    width, height = im.size
    margin = crop_margin

    # Handle smaller images
    if width < crop_w + 3 + margin:
        im = im.resize((crop_w + 3 + margin, height), Image.ANTIALIAS)
        width, height = im.size
    if height < crop_h + 3 + margin:
        im = im.resize((width, crop_h + 3 + margin), Image.ANTIALIAS)
        width, height = im.size

    left = random.randint(margin, width - crop_w - 1 - margin)
    top = random.randint(margin, height - crop_h - 1 - margin)
    im = im.crop((left, top, left + crop_w, top + crop_h))
    return im

# def rotate_image(im):
#     if (random.random() > rotate_prob):
#         return im
#     return im.rotate(random.randint(-self.rotate_angle, self.rotate_angle))

def saturation_value_jitter_image(self, im):
    if (random.random() > self.HSV_prob):
        return im
    # im = im.convert('HSV')
    data = np.array(im)  # "data" is a height x width x 3 numpy array
    if len(data.shape) < 3: return im
    hsv_data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    hsv_data[:, :, 1] = hsv_data[:, :, 1] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
    hsv_data[:, :, 2] = hsv_data[:, :, 2] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
    data = cv2.cvtColor(hsv_data, cv2.COLOR_HSV2RGB)
    im = Image.fromarray(data, 'RGB')
    # im = im.convert('RGB')
    return im

def rescale_image(self, im):
    if (random.random() > self.scaling_prob):
        return im
    width, height = im.size
    im = im.resize((int(width * self.scaling_factor), int(height * self.scaling_factor)), Image.ANTIALIAS)
    return im

def color_casting(self, im):
    if (random.random() > self.color_casting_prob):
        return im
    data = np.array(im)  # "data" is a height x width x 3 numpy array
    if len(data.shape) < 3: return im
    ch = random.randint(0, 2)
    jitter = random.randint(0, self.color_casting_jitter)
    data[:, :, ch] = data[:, :, ch] + jitter
    im = Image.fromarray(data, 'RGB')
    return im
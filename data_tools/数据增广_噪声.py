# -*- coding:utf-8 -*-
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from PIL import ImageFilter
import numpy as np
import random
import threading, os, time
import logging


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
def BLUR(img):
    return img.filter(ImageFilter.BLUR)
def Caton(img):
    return img.filter(ImageFilter.CONTOUR)
def Edeg(img):
    return img.filter(ImageFilter.EDGE_ENHANCE)
def Rank(img):
    return img.filter(ImageFilter.RankFilter(5,12))
img=Image.open("img/big_q182.jpg", mode="r")
defarr=["randomColor","Edeg","Caton","BLUR","Rank"]

img=Rank(img)
img.save("img/big1.jpg")
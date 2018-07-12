#coding:utf-8
import numpy as np
import cv2
import os

path = 'citybackground/'  #读取路径 
size_pic = {}
for i in os.listdir(path):
    if i.endswith('.jpg'):
        src = cv2.imread('citybackground/' + i, -1)
        hight,width = src.shape[0:2]
        file_name = "%a X %s" %(hight,width)
        if file_name in size_pic:
            cv2.imwrite("/pic1/"+file_name + "/" + i + ".jpg",src)
            size_pic[file_name].append(pic_name)
        else:
            pic_name = [i]
            size_pic[file_name] = pic_name
            isExist = os.path.exists(file_name)
            if not isExist:
                os.makedirs("pic1/"+file_name)
            cv2.imwrite("pic1/" + file_name + "/" + i + ".jpg", src)




# img = cv2.imread('citybackground/03050001.jpg',-1)
# hight,width= img.shape[0:2]
# print(hight,width)


# path_file = "城市背景/03050004.jpg"
# img = cv2.imdecode(np.fromfile(path_file,dtype=np.uint8),-1)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import os
import cv2
import shutil
path=r'E:\dataBase\cdjj\杭州出店经营1'
save_path=r'E:\\cdjy'
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
filePath2 = []
for filename in files:
    filePath.append(path +'\\' +filename)
for dir in filePath:
    files2 = os.listdir(dir)
    for files3 in files2:
        filePath2.append(dir+'\\'+files3)
for pic in filePath2:
    files_end = os.listdir(pic)
    for pic_name in files_end:
        pic_end=pic+'\\'+pic_name
        shutil.copy(pic_end,save_path)
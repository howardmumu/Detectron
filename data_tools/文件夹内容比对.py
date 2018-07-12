#-*- coding: UTF-8 -*-
import os

Pic_path = r"D:\voc\VOCdevkit\VOC2018\529\0529slc\img\\"  # 文件夹目录
path = r"D:\voc\VOCdevkit\VOC2018\529\0529slc\xml\\"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
files2 = os.listdir(Pic_path)
XML_name = []
Pic_name = []
qiuhe=set()
all_num=0
for filename in files:
    XML_name.append(filename.split('.')[0])
    qiuhe.add(filename.split('.')[0])
for imgname in files2:
    Pic_name.append(imgname.split('.')[0])
    qiuhe.add(imgname.split('.')[0])

# for everyname in Pic_name:
#     if (everyname not in XML_name):
#         no_pic=Pic_path+everyname+".jpg"
#         os.remove(no_pic)

print(len(XML_name))
print(len(Pic_name))
for everyname in Pic_name:
    if (everyname in XML_name):
        no_xml=Pic_path+everyname+".jpg"
        all_num+=1
        print(no_xml)
    else:
        no_xml = Pic_path + everyname + ".jpg"
        os.remove(no_xml)
print(all_num)
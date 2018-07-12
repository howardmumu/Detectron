# coding=utf-8
import os
path = r"D:\voc\VOCdevkit\VOC2018\Annotations"
files = os.listdir(path)
filePath = []
kind=set()
objects_set=set()
for filename in files:
    kin=filename.split('.')[0]
    old=path +'\\'+ filename
    new=path +'\\'+kin+".jpg"
    os.renames(old,new)



# kind=set()
# import os
# import xml.etree.ElementTree as ET
# path = "E:\\dataBase\\end_data\\xml\\"  # 文件夹目录
# files = os.listdir(path)  # 得到文件夹下的所有文件名称
# filePath = []
# for filename in files:
#     filePath.append(path + filename)
# for fileEverPath in filePath:
#     tree = ET.parse(fileEverPath)
#     root = tree.getroot()
#     path_text = root.find('path').text
#     all_kind=path_text.split('.')[-1]
#     kind.add(all_kind)
#     print(all_kind)
#     # root.find('path').text="E:\\dataBase\\end_data\\img\\"+all_part[-1]
#     # tree.write(fileEverPath)
# print(kind)
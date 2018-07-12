import os
path = r"D:\voc\VOCdevkit\VOC2018\529\b3img"
files = os.listdir(path)
filePath = []
objects_set=set()
for filename in files:
    old=path +'\\'+ filename
    new=path +'\\'+ "b3_"+filename
    os.renames(old,new)
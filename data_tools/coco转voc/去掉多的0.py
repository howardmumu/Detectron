import os
imgpath=r"G:\人_机动车_非机动车\train2017"
files=os.listdir(imgpath)
for every in files:
    old=imgpath+"\\"+every
    new=imgpath+"\\"+every[6:]
    os.rename(old,new)
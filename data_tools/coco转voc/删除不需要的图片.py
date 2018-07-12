import json
import os

nameStr = []

with open("G:\人_机动车_非机动车\coco2voc\COCO_train.json", "r+") as f:
    data = json.load(f)
    print("read ready")

for i in data:
    imgName = str(i["filename"]).zfill(12) + ".jpg"
    nameStr.append(imgName)

nameStr = set(nameStr)
print(nameStr)
print(len(nameStr))

path = "G:/人_机动车_非机动车/train2017/"

for file in os.listdir(path):
    if (file not in nameStr):
        os.remove(path + file)
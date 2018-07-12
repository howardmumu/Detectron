import json

className = {
    1: 'person',
    2: 'bicycle',
    6: 'bus',
    3: 'car',
    4: 'motorbike'
}

classNum = [1, 2, 3, 4,  6]


def writeNum(Num):
    with open("G:\人_机动车_非机动车\coco2voc\COCO_train.json", "a+") as f:
        f.write(str(Num))

inputfile = []
inner = {}
##向test.json文件写入内容
with open("G:\人_机动车_非机动车\coco2voc\instances_train2017.json", "r+") as f:
    allData = json.load(f)
    data = allData["annotations"]
    print(data[1])
    print("read ready")

for i in data:
    if (i['category_id'] in classNum):
        inner = {
            "filename": str(i["image_id"]).zfill(6),
            "name": className[i["category_id"]],
            "bndbox": i["bbox"]
        }
        inputfile.append(inner)
inputfile = json.dumps(inputfile)
writeNum(inputfile)
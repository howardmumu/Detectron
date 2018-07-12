import xml.dom
import xml.dom.minidom
import os
# from PIL import Image
import cv2
import json

# xml文件规范定义

# _TXT_PATH= '../../SynthText-master/myresults/groundtruth'
_IMAGE_PATH =r'G:\person\train2017'

_INDENT = '' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = 'COCO2014'
_ROOT_NODE = 'annotation'
_DATABASE_NAME = 'LOGODection'
_ANNOTATION = 'COCO2014'
_AUTHOR = 'HHJ'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

# _IMAGE_COPY_PATH= 'JPEGImages'
_ANNOTATION_SAVE_PATH = "G:\person\cocoxml"


# _IMAGE_CHANNEL= 3

# 封装创建节点的过�?
def createElementNode(doc, tag, attr):  # 创建一个元素节�?
    element_node = doc.createElement(tag)

    # 创建一个文本节�?
    text_node = doc.createTextNode(attr)

    # 将文本节点作为元素节点的子节�?
    element_node.appendChild(text_node)

    return element_node


# 封装添加一个子节点的过�?
def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)

    parent_node.appendChild(child_node)


# object节点比较特殊
def createObjectNode(doc, attrs):
    object_node = doc.createElement('object')

    createChildNode(doc, 'name', attrs['name'],
                    object_node)

    createChildNode(doc, 'pose',
                    _POSE, object_node)

    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)

    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)

    bndbox_node = doc.createElement('bndbox')

    createChildNode(doc, 'xmin', str(int(attrs['bndbox'][0])),
                    bndbox_node)

    createChildNode(doc, 'ymin', str(int(attrs['bndbox'][1])),
                    bndbox_node)

    createChildNode(doc, 'xmax', str(int(attrs['bndbox'][0] + attrs['bndbox'][2])),
                    bndbox_node)

    createChildNode(doc, 'ymax', str(int(attrs['bndbox'][1] + attrs['bndbox'][3])),
                    bndbox_node)

    object_node.appendChild(bndbox_node)

    return object_node


# 将documentElement写入XML文件�?
def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')

    doc.writexml(tmpfile, addindent='' * 4, newl='\n', encoding='utf-8')

    tmpfile.close()

    # 删除第一行默认添加的标记

    fin = open('tmp.xml')
    # print(filename)
    fout = open(filename, 'w')
    # print(os.path.dirname(fout))

    lines = fin.readlines()

    for line in lines[1:]:

        if line.split():
            fout.writelines(line)

        # new_lines = ''.join(lines[1:])

        # fout.write(new_lines)

    fin.close()

    fout.close()


if __name__ == "__main__":
    ##读取图片列表
    img_path = r"G:/person/train2017"
    fileList = os.listdir(img_path)
    if fileList == 0:
        os._exit(-1)

    with open("G:\person\coco2voc\COCO_train.json", "r") as f:
        ann_data = json.load(f)

    current_dirpath = os.path.dirname(os.path.abspath('__file__'))

    if not os.path.exists(_ANNOTATION_SAVE_PATH):
        os.mkdir(_ANNOTATION_SAVE_PATH)

    # if not os.path.exists(_IMAGE_COPY_PATH):
    #     os.mkdir(_IMAGE_COPY_PATH)

    for imageName in fileList:

        saveName = imageName.strip(".jpg")
        print(saveName)
        # pos = fileList[xText].rfind(".")
        # textName = fileList[xText][:pos]

        # ouput_file = open(_TXT_PATH + '/' + fileList[xText])
        # ouput_file =open(_TXT_PATH)

        # lines = ouput_file.readlines()

        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (saveName + '.xml'))
        # with open(xml_file_name,"w") as f:
        #     pass

        img = cv2.imread(img_path+"/"+imageName)
        print(img_path+"/"+imageName)
        height, width, channel = img.shape
        print(height, width, channel)

        my_dom = xml.dom.getDOMImplementation()

        doc = my_dom.createDocument(None, _ROOT_NODE, None)

        # 获得根节�?
        root_node = doc.documentElement

        # folder节点

        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)

        # filename节点

        createChildNode(doc, 'filename', saveName + '.jpg', root_node)

        # source节点

        source_node = doc.createElement('source')

        # source的子节点

        createChildNode(doc, 'database', _DATABASE_NAME, source_node)

        createChildNode(doc, 'annotation', _ANNOTATION, source_node)

        createChildNode(doc, 'image', 'flickr', source_node)

        createChildNode(doc, 'flickrid', 'NULL', source_node)

        root_node.appendChild(source_node)

        # owner节点

        owner_node = doc.createElement('owner')

        # owner的子节点

        createChildNode(doc, 'flickrid', 'NULL', owner_node)

        createChildNode(doc, 'name', _AUTHOR, owner_node)

        root_node.appendChild(owner_node)

        # size节点

        size_node = doc.createElement('size')

        createChildNode(doc, 'width', str(width), size_node)

        createChildNode(doc, 'height', str(height), size_node)

        createChildNode(doc, 'depth', str(channel), size_node)

        root_node.appendChild(size_node)

        # segmented节点

        createChildNode(doc, 'segmented', _SEGMENTED, root_node)

        for ann in ann_data:
            if (saveName == ann["filename"]):
                # object节点
                object_node = createObjectNode(doc, ann)
                root_node.appendChild(object_node)

            else:
                continue

        # 构建XML文件名称

        print(xml_file_name)

        # 创建XML文件

        # createXMLFile(attrs, width, height, xml_file_name)

        # # 写入文件
        #
        writeXMLFile(doc, xml_file_name)
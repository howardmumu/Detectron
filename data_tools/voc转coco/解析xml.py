import os

file_name=r'xml\1803141780.xml'
def analyze_xml(file_name):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''
    fp=open(file_name)

    class_name=[]

    rectangle_position=[]

    for p in fp:
        if '<object>' in p:
            class_name.append(next(fp).split('>')[1].split('<')[0])

        if '<bndbox>' in p:
            rectangle = []
            [rectangle.append(int(next(fp).split('>')[1].split('<')[0])) for _ in range(4)]

            rectangle_position.append(rectangle)

    # print(class_name)
    # print(rectangle_position)

    fp.close()

    return class_name,rectangle_position
print(analyze_xml(file_name))

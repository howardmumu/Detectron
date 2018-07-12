import os
print (os.getcwd())
path = "E:\\cd_end\\标定数据\\杭州出店经营1\\0310xml"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
filePath = []
ImagePath = []
for filename in files:
    filePath.append(path+'\\'+filename+'\\')

print(filePath)
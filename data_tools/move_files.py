import os
import shutil

src_folder = '/media/shuhao/harddisk1/data/images/youshang/train/neg'
dst_folder = '/media/shuhao/harddisk1/data/images/youshang/temp/0'

for file in os.listdir(src_folder):
    src = os.path.join(src_folder, file)
    dst = os.path.join(dst_folder, file)
    shutil.copyfile(src, dst)


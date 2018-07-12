import os
import cv2
import shutil

source_folder = ['/media/shuhao/harddisk1/data/images/youshang/original/pic']
dest_folder = ['/media/shuhao/harddisk1/data/images/youshang/original/positive']
image_size = (224, 224)

image_index = 1

for i, s_folder in enumerate(source_folder):
    s_folder = os.path.join(source_folder, s_folder)
    d_folder = dest_folder[i]
    s_subfolders = [os.path.join(s_folder, fo) for fo in os.listdir(s_folder) if os.path.isdir(os.path.join(s_folder, fo))]
    print 'got {} folders from {}'.format(len(s_subfolders), s_folder)
    for sub in s_subfolders:
        print 'reading images from {}'.format(sub)
        images = [os.path.join(sub, f) for f in os.listdir(sub)]
        for image_name in images:
            # im = cv2.imread(image_name)
            # # assert im is not None, \
            # #     'Failed to read image \'{}\''.format(image_name)
            # if im is None:
            #     continue
            # im = cv2.resize(im, image_size)
            save_name = os.path.join(d_folder, '{0:08d}.jpg'.format(image_index))
            while os.path.exists(save_name):
                image_index += 1
                save_name = os.path.join(d_folder, '{0:08d}.jpg'.format(image_index))
            shutil.copyfile(image_name, save_name)
            # cv2.imwrite(save_name, im)
            image_index += 1


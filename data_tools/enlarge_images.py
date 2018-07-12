import cv2
import os

input_file = '/media/shuhao/harddisk1/data/0611/images/changtao'
output_file = '/media/shuhao/harddisk1/data/0611/images/large_images'

images = os.listdir(input_file)

for image in images:
    img = cv2.imread(os.path.join(input_file, image))
    img = cv2.resize(img, (600, 400))
    cv2.imwrite(os.path.join(output_file, image), img)
print('successfully transformed {} images'.format(len(images)))

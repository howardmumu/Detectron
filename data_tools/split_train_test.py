import os
import random
import shutil

# xml_path = '/media/shuhao/harddisk1/data/annotations/advertise/xml'
# train_path = os.path.join(xml_path, 'train')
# test_path = os.path.join(xml_path, 'test')
#
# if not os.path.exists(train_path):
#     os.makedirs(train_path)
#     os.makedirs(test_path)
#
# all_xml = [x for x in os.listdir(xml_path) if x.endswith('.xml')]
# print('total {} files.'.format(len(all_xml)))
# test_xml = random.sample(all_xml, int(0.1 * len(all_xml)))
# train_xml = [x for x in all_xml if x not in test_xml]
#
# print('got {} train files'.format(len(train_xml)))
# print('got {} test files'.format(len(test_xml)))
# print('saving train and test files ...')
#
# for file in train_xml:
#     shutil.copyfile(os.path.join(xml_path, file), os.path.join(train_path, file))
#
# for file in test_xml:
#     shutil.copyfile(os.path.join(xml_path, file), os.path.join(test_path, file))
#
# print('done')

pos_path = '/media/shuhao/harddisk1/data/images/youshang/original/positive'
neg_path = '/media/shuhao/harddisk1/data/images/youshang/original/negative'

train_pos_path = '/media/shuhao/harddisk1/data/images/youshang/original/train/1'
test_pos_path = '/media/shuhao/harddisk1/data/images/youshang/original/test/1'
train_neg_path = '/media/shuhao/harddisk1/data/images/youshang/original/train/0'
test_neg_path = '/media/shuhao/harddisk1/data/images/youshang/original/test/0'

pos_images = os.listdir(pos_path)
neg_images = os.listdir(neg_path)

print('loaded {} images from {}'.format(len(pos_images), pos_path))
print('loaded {} images from {}'.format(len(neg_images), neg_path))

test_pos_images = random.sample(pos_images, 1716)
train_pos_images = [x for x in pos_images if x not in test_pos_images]

test_neg_images = random.sample(neg_images, 1776)
train_neg_images = [x for x in neg_images if x not in test_neg_images]

for im in train_pos_images:
    shutil.copyfile(os.path.join(pos_path, im), os.path.join(train_pos_path, im))
print('copied {} images to {}'.format(len(train_pos_images), train_pos_path))

for im in train_neg_images:
    shutil.copyfile(os.path.join(neg_path, im), os.path.join(train_neg_path, im))
print('copied {} images to {}'.format(len(train_neg_images), train_neg_path))

for im in test_pos_images:
    shutil.copyfile(os.path.join(pos_path, im), os.path.join(test_pos_path, im))
print('copied {} images to {}'.format(len(test_pos_images), test_pos_path))

for im in test_neg_images:
    shutil.copyfile(os.path.join(neg_path, im), os.path.join(test_neg_path, im))
print('copied {} images to {}'.format(len(test_neg_images), test_neg_path))




import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(xml_file):
  # TODO(user): Populate the following variables from your example.
  tree = ET.ElementTree(file=xml_file)
  root = tree.getroot()
  size = root.find('size')

  height = int(size.find('height').text) # Image height
  width = int(size.find('width').text) # Image width
  if float(height)==0:
      height=1
  if float(width)==0:
      width=1
  filename = root.find('filename').text.encode() # Filename of the image. Empty if image is not from file
  path = root.find('path').text
  # img = load_image(path)
  # encoded_image_data = _bytes_feature(tf.compat.as_bytes(img.tostring()))
  # TODO: remove below two lines to not modify path
  # temp_path = '/media/shuhao/harddisk1/data/0611/images/large_images'
  # path = os.path.join(temp_path, path.split('/')[-1])
  encoded_image_data = tf.gfile.FastGFile(path, 'rb').read()
  # image_handle = open(path, 'rb')
  # encoded_image_data = image_handle.read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
 # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
 # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  label_map_path = b'/home/shuhao/work/models/research/object_detection/data/advertise_label_map.pbtxt'
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)

  objects = root.findall('object')
  for object in objects:

      box = object.find('bndbox')
      xmin = float(box.find('xmin').text) / float(width)
      ymin = float(box.find('ymin').text) / float(height)
      xmax = float(box.find('xmax').text) / float(width)
      ymax = float(box.find('ymax').text) / float(height)
      class_txt = object.find('name').text.encode()
      xmins.append(xmin)
      ymins.append(ymin)
      xmaxs.append(xmax)
      ymaxs.append(ymax)
      classes_text.append(class_txt)
      #print(label_map_dict[object.find('name').text])
      classes.append(label_map_dict[object.find('name').text])
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example
def main(_):
  out_path=b'/media/shuhao/harddisk1/data/tfrecords/large_single.record'
  writer = tf.python_io.TFRecordWriter(out_path)
  input_folder = b'/media/shuhao/harddisk1/data/Annotations/'
  xml_list = [file for file in os.listdir(input_folder) if file.endswith('.xml'.encode(encoding="utf-8"))]
  for xml in xml_list:
    tf_example = create_tf_example(os.path.join(input_folder, xml))
    writer.write(tf_example.SerializeToString())
  writer.close()
if __name__ == '__main__':
  tf.app.run()
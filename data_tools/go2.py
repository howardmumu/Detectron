import numpy as np
import sys
import tensorflow as tf
import cv2
import os
import time
from matplotlib import pyplot as plt
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
PATH_TO_CKPT = '/media/shuhao/harddisk1/model/advertise_resnet50_002/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/shuhao/work/models/research/object_detection/data/advertise_label_map.pbtxt'
NUM_CLASSES = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#加载模型.pb到网络里
config = tf.ConfigProto()
all_time=0
start1 = time.time()
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
end = time.time()
seconds = int(1000*(end - start1))
print("模型加载耗时:{0} ms".format(seconds))
#加载标签
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# image_path=r'/home/shuhao/work/data_tools/img'
image_path = '/media/shuhao/Seagate Expansion Drive/test'
test_xml_path = '/media/shuhao/harddisk1/data/Annotations/test'
img_list = os.listdir(image_path)
TEST_IMAGE_PATHS = [ os.path.join(image_path, i) for i in img_list ]
with tf.Session(graph=detection_graph) as sess:
        for image_where in TEST_IMAGE_PATHS:
            image_np = cv2.imread(image_where)
            # plt.imshow(image_np)
            # plt.show()
            # image_np = cv2.resize(image_np, (1920,1080), interpolation=cv2.INTER_CUBIC)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            start = time.time()
            (boxes, scores, classes, num) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            end = time.time()
            one_step=end-start
            all_time=all_time+one_step
            seconds = int(1000*(end - start))
            print("图片耗时:{0} ms".format(seconds))
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=4)
            name3=image_where[21:]
            print(name3,classes)
            pic_score = np.squeeze(scores).tolist()
            num_obeject = 0
            for evert_score in pic_score:
                if (evert_score >= 0.5):
                    num_obeject = num_obeject + 1
            if num_obeject >= 1:
                savePath="/media/shuhao/Seagate Expansion Drive/test/out/"+"mask_"+str(random.randint(1,999999))+".jpg"
                 #cv2.imwrite("out\\"+name3, image_np,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                cv2.imwrite(savePath, image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                print('没看出来')


            #print('类别',np.squeeze(classes).astype(np.int32))
        print("所有图片总耗时:{0} 秒".format(all_time))

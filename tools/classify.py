import numpy as np
import skimage.io
import skimage.transform
import random
from matplotlib import pyplot
import os
from caffe2.python import workspace
import operator

CAFFE_MODELS = '/media/shuhao/harddisk1/model/youshang_resnet50'
MODEL = ['init_net.pb', 'predict_net.pb']
IMAGE_LOCATION = '/media/shuhao/harddisk1/model/youshang_resnet50/test/00004_aug.jpg'
INPUT_IMAGE_SIZE = 224

# make sure all of the files are around...
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0])
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[1])
mean = 128

# Function to crop the center cropX x cropY pixels from the input image
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

# Function to rescale the input image to the desired height and/or width. This function will preserve
#   the aspect ratio of the original image while making the image the correct scale so we can retrieve
#   a good center crop. This function is best used with center crop to resize any size input images into
#   specific sized images that our model can use.
def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled

# Load the image as a 32-bit float
#    Note: skimage.io.imread returns a HWC ordered RGB image of some size
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print("Original Image Shape: " , img.shape)

# Rescale the image to comply with our desired input size. This will not make the image 227x227
#    but it will make either the height or width 227 so we can get the ideal center crop.
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image Shape after rescaling: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Rescaled image')

# Crop the center 227x227 pixels of the image so we can feed it to our model
img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image Shape after cropping: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Center Cropped')

# switch to CHW (HWC --> CHW)
img = img.swapaxes(1, 2).swapaxes(0, 1)
print("CHW Image Shape: " , img.shape)

pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

# switch to BGR (RGB --> BGR)
img = img[(2, 1, 0), :, :]

# remove mean for better results
img = img * 255 - mean

# add batch size axis which completes the formation of the NCHW shaped input that we want
img = img[np.newaxis, :, :, :].astype(np.float32)
img.reshape(1,3,224,224)
print("NCHW image (ready to be used as input): ", img.shape)


# Read the contents of the input protobufs into local variables# Read
with open(INIT_NET, "rb") as f:
    init_net = f.read()
with open(PREDICT_NET, "rb") as f:
    predict_net = f.read()

# Initialize the predictor from the input protobufs
p = workspace.Predictor(init_net, predict_net)

# Run the net and return prediction
# results = p.run({'data': img})
#
# # Turn it into something we can play with and examine which is in a multi-dimensional array
# results = np.asarray(results)
# print("results shape: ", results.shape)
#
# # Quick way to get the top-1 prediction result
# # Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
# preds = np.squeeze(results)
# # Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
# curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
# print("Prediction: ", curr_pred)
# print("Confidence: ", curr_conf)

# List of input images to be fed
IMAGE_ROOT = '/media/shuhao/harddisk1/model/youshang_resnet50/test'
images = [os.path.join(IMAGE_ROOT, f) for f in os.listdir(IMAGE_ROOT)]
images = random.sample(images, 8)

# Allocate space for the batch of formatted images
NCHW_batch = np.zeros((len(images),3,INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE))
print ("Batch Shape: ",NCHW_batch.shape)

# For each of the images in the list, format it and place it in the batch
for i,curr_img in enumerate(images):
    img = skimage.img_as_float(skimage.io.imread(curr_img)).astype(np.float32)
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = img[(2, 1, 0), :, :]
    img = img * 255 - mean
    NCHW_batch[i] = img

print("NCHW image (ready to be used as input): ", NCHW_batch.shape)

# Run the net on the batch
results = p.run([NCHW_batch.astype(np.float32)])

# Turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)

# Squeeze out the unnecessary axis
preds = np.squeeze(results)
print("Squeezed Predictions Shape, with batch size {}: {}".format(len(images),preds.shape))

# Describe the results
for i,pred in enumerate(preds):
    print("Results for: '{}'".format(images[i]))
    # Get the prediction and the confidence by finding the maximum value
    #   and index of maximum value in preds array
    curr_pred, curr_conf = max(enumerate(pred), key=operator.itemgetter(1))
    print("\tPrediction: ", curr_pred)
    print("\tConfidence: ", curr_conf)




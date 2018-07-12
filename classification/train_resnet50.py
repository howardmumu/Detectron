
import numpy as np
import time
import os

from caffe2.python import core, workspace, model_helper, net_drawer, memonger, brew
from caffe2.python import data_parallel_model as dpm
from caffe2.python.models import resnet
from caffe2.proto import caffe2_pb2
workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])


# load datasets

data_folder = '/path/to/resnet_trainer'
train_data_db = os.path.join(data_folder, "imagenet_cars_boats_train")
train_data_db_type = "lmdb"

# 640 cars and 640 boats = 1280

train_data_count = 1280
test_data_db = os.path.join(data_folder, "imagenet_cars_boats_val")
test_data_db_type = "lmdb"

# 48 cars and 48 boats = 96

test_data_count = 96

assert os.path.exists(train_data_db)
assert os.path.exists(test_data_db)

# Number of gpu
# For example, gpus = [0, 1, 2, n]
gpus = [0]

# Batch size of 32 sums up to roughly 5GB of memory per device
batch_per_device = 32
total_batch_size = batch_per_device * len(gpus)

# two labels: car and boat  TODO: change
num_labels = 2

# base learning rate
base_learning_rate = 0.0004 * total_batch_size

# change learning rate every 10 epochs
stepsize = int(10 * train_data_count / total_batch_size)

# Weight decay (L2 regularization)
weight_decay = 1e-4

################
# create CNN net
################
# model helpe object
# Only need one parameter (net name), for workspace usage
# e.g.
catos_model = model_helper.ModelHelper(name="catos")

# clear workspace before creating the net
workspace.ResetWorkspace()

# read from database
reader = catos_model.CreateDB(name, db, db_type)


#TODO: get image size
def add_image_input_ops(model):
    # Use ImageInput operator to preprocess images
    data, label = model.ImageInput(reader,
                                   ["data", "label"],
                                   batch_size=batch_per_device,
                                   mean=128.,
                                   std=128.,
                                   scale=256,
                                   crop=224,
                                   is_test=False,
                                   mirror=1
                                  )
    # no BP
    data = model.StopGradient(data, data)


def create_resnet50_model_ops(model, loss_scale):
    # Create Resnet
    [softmax, loss] = resnet.create_resnet50(model,
                                             "data",
                                             num_input_channels=3,
                                             num_labels=num_labels,
                                             label="label", )
    prefix = model.net.Proto().name
    loss = model.Scale(loss, prefix + "_loss", scale=loss_scale)
    model.Accuracy([softmax, "label"], prefix + "_accuracy")
    return [loss]


def add_parameter_update_ops(model):
    model.AddWeightDecay(weight_decay)
    iter = model.Iter("iter")
    lr = model.net.LearningRate([iter],
                                "lr",
                                base_lr=base_learning_rate,
                                policy="step",
                                stepsize=stepsize,
                                gamma=0.1, )
    # Momentum SGD update
    for param in model.GetParams():
        param_grad = model.param_to_grad[param]
        param_momentum = model.param_init_net.ConstantFill([param],
                                                           param + '_momentum', value=0.0)

        # Update param_grad and param_momentum in place
        model.net.MomentumSGDUpdate([param_grad, param_momentum, lr, param],
                                    [param_grad, param_momentum, param],
                                    momentum=0.9,
                                    # Nesterov Momentum works slightly better than standard
                                    nesterov=1, )


def optimize_gradient_memory(model, loss):
    model.net._net = memonger.share_grad_blobs(model.net,
                                               loss,
                                               set(model.param_to_grad.values()),
                                               # memonger needs namescope param,created here and will be used later
                                               namescope="imonaboat",
                                               share_activations=False)


train_model = model_helper.ModelHelper(name="train",)
reader = train_model.CreateDB("train_reader",
                              db=train_data_db,
                              db_type=train_data_db_type, )
dpm.Parallelize_GPU(train_model,
                    input_builder_fun=add_image_input_ops,
                    forward_pass_builder_fun=create_resnet50_model_ops,
                    param_update_builder_fun=add_parameter_update_ops,
                    devices=gpus,
                    optimize_gradient_memory=True, )


workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)


# Create test model
test_model = model_helper.ModelHelper(name="test",)

reader = test_model.CreateDB("test_reader",
                             db=test_data_db,
                             db_type=test_data_db_type,)

# Validation is parallelized across devices as well
dpm.Parallelize_GPU(test_model,
                    input_builder_fun=add_image_input_ops,
                    forward_pass_builder_fun=create_resnet50_model_ops,
                    param_update_builder_fun=None,
                    devices=gpus,)

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)


# Visualization
from caffe2.python import visualize
from matplotlib import pyplot as plt

def display_images_and_confidence():
    images = []
    confidences = []
    n = 16
    data = workspace.FetchBlob("gpu_0/data")
    label = workspace.FetchBlob("gpu_0/label")
    softmax = workspace.FetchBlob("gpu_0/softmax")
    for arr in zip(data[0:n], label[0:n], softmax[0:n]):
        # CHW to HWC, normalize to [0.0, 1.0], and BGR to RGB
        bgr = (arr[0].swapaxes(0, 1).swapaxes(1, 2) + 1.0) / 2.0
        rgb = bgr[...,::-1]
        images.append(rgb)
        confidences.append(arr[2][arr[1]])

    # Create grid for images
    fig, rows = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    plt.tight_layout(h_pad=2)

    # Display images and the models confidence in their label
    items = zip([ax for cols in rows for ax in cols], images, confidences)
    for (ax, image, confidence) in items:
        ax.imshow(image)
        if confidence >= 0.5:
            ax.set_title("RIGHT ({:.1f}%)".format(confidence * 100.0), color='green')
        else:
            ax.set_title("WRONG ({:.1f}%)".format(confidence * 100.0), color='red')

    plt.show()


def accuracy(model):
    accuracy = []
    prefix = model.net.Proto().name
    for device in model._devices:
        accuracy.append(
            np.asscalar(workspace.FetchBlob("gpu_{}/{}_accuracy".format(device, prefix))))
    return np.average(accuracy)


#################################
# Multi-gpu training and testing
#################################
# training epoch
num_epochs = 2
for epoch in range(num_epochs):
    # number of iters per epoch
    num_iters = int(train_data_count / total_batch_size)
    for iter in range(num_iters):
        t1 = time.time()
        # one iter
        workspace.RunNet(train_model.net.Proto().name)
        t2 = time.time()
        dt = t2 - t1

        print((
            "Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
            " (epoch {:>" + str(len(str(num_epochs))) + "}/{})" +
            " ({:.2f} images/sec)").
            format(iter+1, num_iters, epoch+1, num_epochs, total_batch_size/dt))

        # train accuracy
        train_accuracy = accuracy(train_model)

    # test net
    test_accuracies = []
    for _ in range(test_data_count / total_batch_size):
        workspace.RunNet(test_model.net.Proto().name)
        test_accuracies.append(accuracy(test_model))
    test_accuracy = np.average(test_accuracies)

    print(
        "Train accuracy: {:.3f}, test accuracy: {:.3f}".
        format(train_accuracy, test_accuracy))

    # Output images with confidence scores as the caption
    display_images_and_confidence()



